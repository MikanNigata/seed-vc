from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

import inference as seed_inference
from modules.commons import str2bool
from mosaic_svc.p0.audio_features import extract_campplus_style, load_audio_tensor
from mosaic_svc.p0.f0_embedding_adapter import install_f0_embedding_adapter
from mosaic_svc.p0.f0_condition_adapter import load_f0_condition_adapter
from mosaic_svc.p0.f0_block_adapter import install_f0_block_adapter
from mosaic_svc.p0.f0_dit_adapter import install_f0_dit_adapter
from mosaic_svc.p0.prompt_adapter import PromptAdapterConfig, install_prompt_adapter
from mosaic_svc.p0.prototype_bank import PrototypeBank
from mosaic_svc.p0.style_adapter import StyleAdapterConfig, install_style_slice_adapter
from mosaic_svc.p4.prompt_mel_lora import PromptMelLoRAConfig, install_prompt_mel_lora
from mosaic_svc.p5.kv_lora import install_kv_lora
from mosaic_svc.temporal.style_schedule import (
    TemporalStyleConfig,
    build_temporal_style_schedule,
    install_temporal_style_merge,
    load_temporal_memory_records,
    save_temporal_style_summary,
)
from mosaic_svc.f0_guidance import amplify_f0_condition


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _semantic_with_overlap(semantic_fn, waves_16k: torch.Tensor, max_seconds: int = 30, overlap_seconds: int = 5):
    if waves_16k.size(-1) <= 16000 * max_seconds:
        return semantic_fn(waves_16k)

    chunks = []
    buffer = None
    traversed = 0
    while traversed < waves_16k.size(-1):
        if buffer is None:
            chunk = waves_16k[:, traversed : traversed + 16000 * max_seconds]
        else:
            chunk = torch.cat(
                [buffer, waves_16k[:, traversed : traversed + 16000 * (max_seconds - overlap_seconds)]],
                dim=-1,
            )
        features = semantic_fn(chunk)
        chunks.append(features if traversed == 0 else features[:, 50 * overlap_seconds :])
        buffer = chunk[:, -16000 * overlap_seconds :]
        traversed += max_seconds * 16000 if traversed == 0 else chunk.size(-1) - 16000 * overlap_seconds
    return torch.cat(chunks, dim=1)


def _style_from_audio(campplus_model, path: str, sr: int, device: torch.device, max_seconds: float = 25.0) -> torch.Tensor:
    audio = load_audio_tensor(path, sr=sr, device=device, max_seconds=max_seconds)
    return extract_campplus_style(campplus_model, audio, sr, device)


def _temporal_embedding_loader(
    memory: str,
    campplus_model,
    sr: int,
    device: torch.device,
    context_seconds: float,
):
    if context_seconds <= 0:
        raise ValueError("--temporal-context-seconds must be positive")
    root, metadata, _ = load_temporal_memory_records(memory)
    source_path = Path(str(metadata.get("source_path", "")))
    source_audio = None
    if source_path.is_file():
        source_audio = librosa.load(str(source_path), sr=sr, mono=True)[0]

    def load(record: dict) -> torch.Tensor:
        clip = None
        if source_audio is not None:
            center = (float(record["start_seconds"]) + float(record["end_seconds"])) / 2.0
            clip_samples = max(1, int(round(context_seconds * sr)))
            start = max(0, int(round(center * sr)) - clip_samples // 2)
            start = min(start, max(0, len(source_audio) - clip_samples))
            clip = source_audio[start : start + clip_samples]
        if clip is None or len(clip) < int(0.25 * sr):
            patch_path = (root / str(record["audio_path"])).resolve()
            if not patch_path.is_file():
                raise FileNotFoundError(f"Temporal patch audio does not exist: {patch_path}")
            clip = librosa.load(str(patch_path), sr=sr, mono=True)[0]
        audio = torch.tensor(clip).unsqueeze(0).float().to(device)
        return extract_campplus_style(campplus_model, audio, sr, device)

    return load


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    _seed_all(args.seed)
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = seed_inference.load_models(args)
    sr = mel_fn_args["sampling_rate"]
    device = seed_inference.device

    f0_condition_adapter = None
    if args.f0_condition_adapter:
        f0_condition_adapter = load_f0_condition_adapter(
            args.f0_condition_adapter,
            device,
            trainable=False,
            strength=args.f0_condition_adapter_strength,
        )

    f0_dit_adapter = None
    f0_dit_merge = None
    if args.f0_dit_adapter:
        f0_dit_adapter, f0_dit_merge = install_f0_dit_adapter(
            model,
            state_path=args.f0_dit_adapter,
            trainable=False,
            strength=args.f0_dit_adapter_strength,
        )

    f0_block_adapter = None
    f0_block_wrappers = None
    if args.f0_block_adapter:
        f0_block_adapter, f0_block_wrappers = install_f0_block_adapter(
            model,
            state_path=args.f0_block_adapter,
            trainable=False,
            strength=args.f0_block_adapter_strength,
        )

    if args.f0_embedding_adapter:
        install_f0_embedding_adapter(
            model,
            state_path=args.f0_embedding_adapter,
            trainable=False,
            strength=args.f0_embedding_adapter_strength,
        )

    if args.style_adapter:
        config = StyleAdapterConfig(
            rank=args.style_adapter_rank,
            dropout=args.style_adapter_dropout,
            initial_scale=args.style_adapter_initial_scale,
        )
        install_style_slice_adapter(model, config=config, state_path=args.style_adapter, trainable=False)
    elif args.install_zero_style_adapter:
        install_style_slice_adapter(
            model,
            config=StyleAdapterConfig(
                rank=args.style_adapter_rank,
                dropout=args.style_adapter_dropout,
                initial_scale=args.style_adapter_initial_scale,
            ),
            trainable=False,
        )

    if args.prompt_mel_lora:
        install_prompt_mel_lora(
            model,
            config=PromptMelLoRAConfig(
                rank=args.prompt_mel_lora_rank,
                dropout=args.prompt_mel_lora_dropout,
                initial_scale=args.prompt_mel_lora_initial_scale,
                max_scale=args.prompt_mel_lora_max_scale,
            ),
            state_path=args.prompt_mel_lora,
            trainable=False,
        )

    if args.kv_lora:
        install_kv_lora(model, state_path=args.kv_lora, trainable=False)

    if args.prompt_adapter:
        install_prompt_adapter(
            model,
            config=PromptAdapterConfig(
                rank=args.prompt_adapter_rank,
                dropout=args.prompt_adapter_dropout,
                initial_scale=args.prompt_adapter_initial_scale,
                max_scale=args.prompt_adapter_max_scale,
                source_only=args.prompt_adapter_source_only,
            ),
            state_path=args.prompt_adapter,
            trainable=False,
            strength=args.prompt_adapter_strength,
        )
    elif args.install_zero_prompt_adapter:
        install_prompt_adapter(
            model,
            config=PromptAdapterConfig(
                rank=args.prompt_adapter_rank,
                dropout=args.prompt_adapter_dropout,
                initial_scale=args.prompt_adapter_initial_scale,
                max_scale=args.prompt_adapter_max_scale,
                source_only=args.prompt_adapter_source_only,
            ),
            trainable=False,
            strength=args.prompt_adapter_strength,
        )

    source_audio_np = librosa.load(args.source, sr=sr)[0]
    prompt_audio_np = librosa.load(args.prompt, sr=sr)[0]

    f0_condition = args.f0_condition
    sr = 22050 if not f0_condition else 44100
    hop_length = 256 if not f0_condition else 512
    max_context_window = sr // hop_length * args.max_context_seconds
    overlap_frame_len = args.overlap_frames
    overlap_wave_len = overlap_frame_len * hop_length

    source_audio = torch.tensor(source_audio_np).unsqueeze(0).float().to(device)
    prompt_audio = torch.tensor(prompt_audio_np[: int(sr * args.prompt_seconds)]).unsqueeze(0).float().to(device)
    source_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    prompt_16k = torchaudio.functional.resample(prompt_audio, sr, 16000)

    t0 = time.time()
    S_alt = _semantic_with_overlap(semantic_fn, source_16k)
    S_ori = semantic_fn(prompt_16k)

    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(prompt_audio.to(device).float())
    target_lengths = torch.LongTensor([int(mel.size(2) * args.length_adjust)]).to(device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(device)

    style_audio = args.style_audio or args.prompt
    style2 = _style_from_audio(campplus_model, style_audio, sr, device, max_seconds=args.prompt_seconds)
    canonical_style = _style_from_audio(campplus_model, args.prompt, sr, device, max_seconds=args.prompt_seconds)

    if args.prototype_bank:
        bank = PrototypeBank.load(args.prototype_bank)
        style2 = bank.corrected_style(
            canonical=canonical_style.detach().cpu(),
            max_norm_ratio=args.prototype_max_norm_ratio,
            max_gate=args.prototype_max_gate,
            strength=args.prototype_strength,
        ).to(device)

    auto_f0_shift_semitones = 0.0
    if f0_condition:
        F0_ori = f0_fn(prompt_16k[0], thred=0.03)
        F0_alt = f0_fn(source_16k[0], thred=0.03)
        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]
        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]
        log_f0_alt = torch.log(F0_alt + 1e-5)
        if args.auto_f0_adjust and voiced_F0_ori.numel() > 0 and voiced_F0_alt.numel() > 0:
            auto_f0_shift_semitones = float(
                12.0
                * (
                    torch.median(torch.log(voiced_F0_ori + 1e-5))
                    - torch.median(torch.log(voiced_F0_alt + 1e-5))
                )
                / np.log(2.0)
            )
            log_f0_alt[F0_alt > 1] = (
                log_f0_alt[F0_alt > 1]
                - torch.median(torch.log(voiced_F0_alt + 1e-5))
                + torch.median(torch.log(voiced_F0_ori + 1e-5))
            )
        shifted_f0_alt = torch.exp(log_f0_alt)
        if args.semi_tone_shift != 0:
            shifted_f0_alt[F0_alt > 1] = seed_inference.adjust_f0_semitones(
                shifted_f0_alt[F0_alt > 1], args.semi_tone_shift
            )
    else:
        F0_ori = None
        shifted_f0_alt = None

    cond, *_ = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
    if args.f0_guidance_scale != 1.0:
        if not f0_condition:
            raise ValueError("--f0-guidance-scale requires --f0-condition True")
        cond_without_f0, *_ = model.length_regulator(
            S_alt,
            ylens=target_lengths,
            n_quantizers=3,
            f0=None,
        )
        cond = amplify_f0_condition(cond, cond_without_f0, args.f0_guidance_scale)
    if f0_condition_adapter is not None:
        cond = f0_condition_adapter(cond, shifted_f0_alt)
    prompt_condition, *_ = model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

    f0_dit_schedule = None
    if f0_dit_adapter is not None:
        if not f0_condition:
            raise ValueError("--f0-dit-adapter requires --f0-condition True")
        f0_dit_schedule = f0_dit_adapter.schedule(shifted_f0_alt, cond.size(1))
    f0_block_features = None
    if f0_block_adapter is not None:
        if not f0_condition:
            raise ValueError("--f0-block-adapter requires --f0-condition True")
        f0_block_features = f0_block_adapter.features(shifted_f0_alt, cond.size(1))

    temporal_merge = None
    temporal_schedule = None
    temporal_summary = None
    if bool(args.temporal_query) != bool(args.temporal_memory):
        raise ValueError("--temporal-query and --temporal-memory must be provided together")
    if args.temporal_query:
        temporal_config = TemporalStyleConfig(
            style_dim=style2.size(-1),
            max_gate=args.temporal_max_gate,
            max_norm_ratio=args.temporal_max_norm_ratio,
            strength=args.temporal_strength,
            min_confidence=args.temporal_min_confidence,
            min_source_f0_confidence=args.temporal_min_source_f0_confidence,
            min_patch_f0_confidence=args.temporal_min_patch_f0_confidence,
            min_patch_quality=args.temporal_min_patch_quality,
            min_weight_margin=args.temporal_min_weight_margin,
            max_register_distance=args.temporal_max_register_distance,
            max_voiced_ratio_distance=args.temporal_max_voiced_ratio_distance,
            smoothing_seconds=args.temporal_smoothing_seconds,
        )
        embedding_loader = _temporal_embedding_loader(
            args.temporal_memory,
            campplus_model,
            sr,
            device,
            args.temporal_context_seconds,
        )
        temporal_schedule, temporal_summary = build_temporal_style_schedule(
            args.temporal_query,
            args.temporal_memory,
            style2,
            embedding_loader,
            source_frames=cond.size(1),
            config=temporal_config,
        )
        temporal_schedule = temporal_schedule.to(device)
        temporal_merge = install_temporal_style_merge(model, style_dim=style2.size(-1))

    max_source_window = max_context_window - mel2.size(2)
    if max_source_window <= 0:
        raise ValueError("prompt is too long for max_context_window; reduce --prompt-seconds")

    # Model/adapter construction consumes RNG. Reset here so ablations use the
    # exact same diffusion noise regardless of which optional modules are loaded.
    _seed_all(args.seed)
    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        if f0_dit_merge is not None:
            chunk_f0 = f0_dit_schedule[:, processed_frames : processed_frames + chunk_cond.size(1)]
            prompt_f0 = torch.zeros(
                chunk_f0.size(0),
                prompt_condition.size(1),
                chunk_f0.size(2),
                device=chunk_f0.device,
                dtype=chunk_f0.dtype,
            )
            f0_dit_merge.set_schedule(torch.cat([prompt_f0, chunk_f0], dim=1))
        if f0_block_wrappers is not None:
            chunk_f0_features = f0_block_features[
                :, processed_frames : processed_frames + chunk_cond.size(1)
            ]
            prompt_f0_features = torch.zeros(
                chunk_f0_features.size(0),
                prompt_condition.size(1),
                chunk_f0_features.size(2),
                device=chunk_f0_features.device,
                dtype=chunk_f0_features.dtype,
            )
            block_features = torch.cat([prompt_f0_features, chunk_f0_features], dim=1)
            for wrapper in f0_block_wrappers:
                wrapper.set_features(block_features)
        if temporal_merge is not None:
            chunk_style = temporal_schedule[:, processed_frames : processed_frames + chunk_cond.size(1)]
            prompt_style = style2[:, None, :].expand(-1, prompt_condition.size(1), -1)
            temporal_merge.set_schedule(torch.cat([prompt_style, chunk_style], dim=1))

        with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(device),
                mel2,
                style2,
                None,
                args.diffusion_steps,
                inference_cfg_rate=args.inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1) :]

        vc_wave = vocoder_fn(vc_target.float()).squeeze()[None, :]
        if processed_frames == 0:
            if is_last_chunk:
                generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                break
            generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
            previous_chunk = vc_wave[0, -overlap_wave_len:]
        elif is_last_chunk:
            generated_wave_chunks.append(seed_inference.crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len))
            break
        else:
            generated_wave_chunks.append(
                seed_inference.crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                    overlap_wave_len,
                )
            )
            previous_chunk = vc_wave[0, -overlap_wave_len:]
        processed_frames += vc_target.size(2) - overlap_frame_len

    if temporal_merge is not None:
        temporal_merge.set_schedule(None)
    if f0_dit_merge is not None:
        f0_dit_merge.set_schedule(None)
    if f0_block_wrappers is not None:
        for wrapper in f0_block_wrappers:
            wrapper.set_features(None)

    output_audio = torch.from_numpy(np.concatenate(generated_wave_chunks)).float().unsqueeze(0)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(args.source).stem
    prompt_name = Path(args.prompt).stem
    if args.kv_lora and args.style_adapter:
        mode = "p8"
    elif args.kv_lora:
        mode = "p5"
    elif args.prompt_mel_lora:
        mode = "p4"
    elif args.prompt_adapter:
        mode = "m2"
    elif args.prototype_bank:
        mode = "d0"
    elif args.style_adapter or args.install_zero_style_adapter:
        mode = "c"
    else:
        mode = "b"
    if temporal_schedule is not None:
        mode = f"{mode}_ttm1"
    if args.f0_embedding_adapter:
        strength_label = f"{args.f0_embedding_adapter_strength:g}".replace(".", "p")
        mode = f"{mode}_f0a{strength_label}"
    if f0_condition_adapter is not None:
        strength_label = f"{args.f0_condition_adapter_strength:g}".replace(".", "p")
        mode = f"{mode}_f0c{strength_label}"
    if f0_dit_adapter is not None:
        strength_label = f"{args.f0_dit_adapter_strength:g}".replace(".", "p")
        mode = f"{mode}_f0d{strength_label}"
    if f0_block_adapter is not None:
        strength_label = f"{args.f0_block_adapter_strength:g}".replace(".", "p")
        mode = f"{mode}_f0b{strength_label}"
    if args.f0_guidance_scale != 1.0:
        scale_label = f"{args.f0_guidance_scale:.2f}".replace(".", "p")
        mode = f"{mode}_f0g{scale_label}"
    out_path = out_dir / f"mosaic_p0_{mode}_{source_name}_{prompt_name}_{args.diffusion_steps}.wav"
    torchaudio.save(str(out_path), output_audio.cpu(), sr)
    if temporal_summary is not None:
        temporal_summary.update(
            {
                "query_path": str(Path(args.temporal_query).resolve()),
                "memory_path": str(Path(args.temporal_memory).resolve()),
                "context_seconds": args.temporal_context_seconds,
                "output_path": str(out_path.resolve()),
            }
        )
        save_temporal_style_summary(temporal_summary, out_path.with_suffix(".temporal.json"))
    print(f"Saved: {out_path}")
    print(f"Elapsed seconds: {time.time() - t0:.2f}")
    print(f"Style source: {style_audio}")
    print(f"Prototype bank: {args.prototype_bank or 'none'}")
    print(f"Prompt adapter: {args.prompt_adapter or 'none'}")
    print(f"Prompt-mel LoRA: {args.prompt_mel_lora or 'none'}")
    print(f"K/V LoRA: {args.kv_lora or 'none'}")
    print(f"Temporal query: {args.temporal_query or 'none'}")
    print(f"Temporal memory: {args.temporal_memory or 'none'}")
    print(f"F0 guidance scale: {args.f0_guidance_scale:.3f}")
    print(f"Auto F0 shift semitones: {auto_f0_shift_semitones:.4f}")
    print(f"F0 embedding adapter: {args.f0_embedding_adapter or 'none'}")
    print(f"F0 embedding adapter strength: {args.f0_embedding_adapter_strength:g}")
    print(f"F0 condition adapter: {args.f0_condition_adapter or 'none'}")
    print(f"F0 condition adapter strength: {args.f0_condition_adapter_strength:g}")
    print(f"F0 DiT adapter: {args.f0_dit_adapter or 'none'}")
    print(f"F0 DiT adapter strength: {args.f0_dit_adapter_strength:g}")
    print(f"F0 block adapter: {args.f0_block_adapter or 'none'}")
    print(f"F0 block adapter strength: {args.f0_block_adapter_strength:g}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mosaic-SVC P0 inference over frozen Seed-VC.")
    parser.add_argument("--source", required=True, help="Source singing vocal wav/flac/mp3.")
    parser.add_argument("--prompt", required=True, help="Fixed canonical prompt audio.")
    parser.add_argument("--style-audio", default=None, help="Optional different audio for CAMPPlus style ablation.")
    parser.add_argument("--output", default="./reconstructed")
    parser.add_argument("--diffusion-steps", type=int, default=40)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--f0-condition", type=str2bool, default=True)
    parser.add_argument("--auto-f0-adjust", type=str2bool, default=False)
    parser.add_argument("--semi-tone-shift", type=int, default=0)
    parser.add_argument(
        "--f0-guidance-scale",
        type=float,
        default=1.0,
        help="Amplify the learned F0 component after length regulation; 1.0 preserves baseline behavior.",
    )
    parser.add_argument("--f0-embedding-adapter", default=None)
    parser.add_argument("--f0-embedding-adapter-strength", type=float, default=1.0)
    parser.add_argument("--f0-condition-adapter", default=None)
    parser.add_argument("--f0-condition-adapter-strength", type=float, default=1.0)
    parser.add_argument("--f0-dit-adapter", default=None)
    parser.add_argument("--f0-dit-adapter-strength", type=float, default=1.0)
    parser.add_argument("--f0-block-adapter", default=None)
    parser.add_argument("--f0-block-adapter-strength", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--prompt-seconds", type=float, default=25.0)
    parser.add_argument("--max-context-seconds", type=int, default=30)
    parser.add_argument("--overlap-frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--style-adapter", default=None)
    parser.add_argument("--install-zero-style-adapter", type=str2bool, default=False)
    parser.add_argument("--style-adapter-rank", type=int, default=4)
    parser.add_argument("--style-adapter-dropout", type=float, default=0.10)
    parser.add_argument("--style-adapter-initial-scale", type=float, default=0.05)
    parser.add_argument("--prompt-adapter", default=None)
    parser.add_argument("--install-zero-prompt-adapter", type=str2bool, default=False)
    parser.add_argument("--prompt-adapter-rank", type=int, default=8)
    parser.add_argument("--prompt-adapter-dropout", type=float, default=0.05)
    parser.add_argument("--prompt-adapter-initial-scale", type=float, default=0.03)
    parser.add_argument("--prompt-adapter-max-scale", type=float, default=0.20)
    parser.add_argument("--prompt-adapter-strength", type=float, default=1.0)
    parser.add_argument("--prompt-adapter-source-only", type=str2bool, default=True)
    parser.add_argument("--prototype-bank", default=None)
    parser.add_argument("--prototype-strength", type=float, default=1.0)
    parser.add_argument("--prototype-max-norm-ratio", type=float, default=0.10)
    parser.add_argument("--prototype-max-gate", type=float, default=0.25)
    parser.add_argument("--prompt-mel-lora", default=None)
    parser.add_argument("--prompt-mel-lora-rank", type=int, default=8)
    parser.add_argument("--prompt-mel-lora-dropout", type=float, default=0.05)
    parser.add_argument("--prompt-mel-lora-initial-scale", type=float, default=0.02)
    parser.add_argument("--prompt-mel-lora-max-scale", type=float, default=0.10)
    parser.add_argument("--kv-lora", default=None)
    parser.add_argument("--temporal-query", default=None)
    parser.add_argument("--temporal-memory", default=None)
    parser.add_argument("--temporal-strength", type=float, default=1.0)
    parser.add_argument("--temporal-max-gate", type=float, default=0.25)
    parser.add_argument("--temporal-max-norm-ratio", type=float, default=0.10)
    parser.add_argument("--temporal-min-confidence", type=float, default=0.45)
    parser.add_argument("--temporal-min-source-f0-confidence", type=float, default=0.35)
    parser.add_argument("--temporal-min-patch-f0-confidence", type=float, default=0.50)
    parser.add_argument("--temporal-min-patch-quality", type=float, default=0.90)
    parser.add_argument("--temporal-min-weight-margin", type=float, default=0.015)
    parser.add_argument("--temporal-max-register-distance", type=float, default=0.20)
    parser.add_argument("--temporal-max-voiced-ratio-distance", type=float, default=0.35)
    parser.add_argument("--temporal-smoothing-seconds", type=float, default=0.50)
    parser.add_argument("--temporal-context-seconds", type=float, default=2.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
