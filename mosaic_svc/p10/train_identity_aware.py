from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

import inference as seed_inference
from modules.commons import str2bool
from mosaic_svc.p0.audio_features import load_audio_tensor
from mosaic_svc.p0.style_adapter import install_style_slice_adapter, save_style_adapter
from mosaic_svc.p4.train_prompt_mel_lora import _crop, _features, _read_paths
from mosaic_svc.p5.kv_lora import install_kv_lora


def _freeze(module) -> None:
    if isinstance(module, torch.nn.Module):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False


def _speaker_embedding(campplus_model, waveform: torch.Tensor, sr: int) -> torch.Tensor:
    waveform_16k = torchaudio.functional.resample(waveform, sr, 16000)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform_16k,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    feature = feature - feature.mean(dim=0, keepdim=True)
    embedding = campplus_model(feature.unsqueeze(0))
    return torch.nn.functional.normalize(embedding.float(), dim=-1)


def _identity_loss(vocoder, campplus_model, predicted_mel, prompt_len: int, target, sr: int) -> torch.Tensor:
    source_mel = predicted_mel[:, :, prompt_len:]
    waveform = vocoder(source_mel.float()).reshape(1, -1)
    embedding = _speaker_embedding(campplus_model, waveform, sr)
    return 1.0 - torch.nn.functional.cosine_similarity(embedding, target, dim=-1).mean()


@torch.no_grad()
def _validate(args, model, semantic_fn, f0_fn, vocoder, campplus_model, mel_fn, sr, device, paths, style, target):
    model.cfm.eval()
    reconstruction_losses = []
    identity_losses = []
    for index, path in enumerate(paths):
        audio = load_audio_tensor(path, sr=sr, device=device, max_seconds=None)
        audio = _crop(audio, sr, args.segment_seconds, random_crop=False, min_rms=args.min_rms)
        mel, condition, lengths = _features(model, semantic_fn, f0_fn, mel_fn, audio, sr)
        prompt_len = torch.LongTensor(
            [max(1, min(int(lengths.item() * args.prompt_ratio), lengths.item() - 1))]
        ).to(device)
        torch.manual_seed(args.seed + index)
        reconstruction, predicted_mel = model.cfm(mel, lengths, prompt_len, condition, style)
        identity = _identity_loss(
            vocoder, campplus_model, predicted_mel, int(prompt_len.item()), target, sr
        )
        reconstruction_losses.append(float(reconstruction.detach().cpu()))
        identity_losses.append(float(identity.detach().cpu()))
    model.cfm.train()
    model.cfm.estimator.class_dropout_prob = 0.0
    return (
        sum(reconstruction_losses) / len(reconstruction_losses),
        sum(identity_losses) / len(identity_losses),
    )


def run(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    seed_args = argparse.Namespace(f0_condition=True, checkpoint=args.checkpoint, config=args.config, fp16=args.fp16)
    model, semantic_fn, f0_fn, vocoder, campplus_model, mel_fn, mel_fn_args = seed_inference.load_models(seed_args)
    device = seed_inference.device
    sr = mel_fn_args["sampling_rate"]
    if args.kv_lora:
        install_kv_lora(model, state_path=args.kv_lora, trainable=False)
    adapter = install_style_slice_adapter(model, state_path=args.style_adapter, trainable=True)
    _freeze(vocoder)
    _freeze(campplus_model)
    model.cfm.train()
    model.cfm.estimator.class_dropout_prob = 0.0
    adapter.train()

    profile = torch.load(args.identity_profile, map_location="cpu")
    target = torch.nn.functional.normalize(profile["centroid"].float().to(device).view(1, -1), dim=-1)
    canonical_audio = load_audio_tensor(args.canonical, sr=sr, device=device, max_seconds=args.canonical_seconds)
    with torch.no_grad():
        prompt_16k = torchaudio.functional.resample(canonical_audio, sr, 16000)
        feature = torchaudio.compliance.kaldi.fbank(
            prompt_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feature = feature - feature.mean(dim=0, keepdim=True)
        style = campplus_model(feature.unsqueeze(0)).float().detach()

    train_paths = _read_paths(args.train_manifest)
    validation_paths = _read_paths(args.validation_manifest)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    history, recent = [], []
    best_objective = float("inf")
    best_path = output_dir / "identity_style_adapter_best.pt"
    stale = 0

    for step in tqdm(range(1, args.steps + 1)):
        audio = load_audio_tensor(random.choice(train_paths), sr=sr, device=device, max_seconds=None)
        audio = _crop(audio, sr, args.segment_seconds, random_crop=True, min_rms=args.min_rms)
        mel, condition, lengths = _features(model, semantic_fn, f0_fn, mel_fn, audio, sr)
        prompt_frames = max(1, min(int(lengths.item() * args.prompt_ratio), lengths.item() - 1))
        prompt_len = torch.LongTensor([prompt_frames]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
            reconstruction, predicted_mel = model.cfm(mel, lengths, prompt_len, condition, style)
        identity = _identity_loss(vocoder, campplus_model, predicted_mel, prompt_frames, target, sr)
        progress = min(1.0, step / max(1, args.identity_warmup_steps))
        identity_weight = args.identity_weight * progress
        regularization = adapter.up.weight.float().square().mean()
        loss = reconstruction.float() + identity_weight * identity + args.adapter_l2 * regularization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        recent.append((float(reconstruction.detach().cpu()), float(identity.detach().cpu())))

        if step % args.validate_every == 0 or step == args.steps:
            validation_reconstruction, validation_identity = _validate(
                args,
                model,
                semantic_fn,
                f0_fn,
                vocoder,
                campplus_model,
                mel_fn,
                sr,
                device,
                validation_paths,
                style,
                target,
            )
            reconstruction_mean = sum(item[0] for item in recent) / len(recent)
            identity_mean = sum(item[1] for item in recent) / len(recent)
            objective = validation_reconstruction + args.identity_weight * validation_identity
            checkpoint = output_dir / f"identity_style_adapter_step_{step:06d}.pt"
            save_style_adapter(adapter, checkpoint)
            improved = objective < best_objective - args.min_delta
            if improved:
                best_objective = objective
                shutil.copy2(checkpoint, best_path)
                stale = 0
            else:
                stale += 1
            row = {
                "step": step,
                "train_reconstruction": reconstruction_mean,
                "train_identity_loss": identity_mean,
                "validation_reconstruction": validation_reconstruction,
                "validation_identity_loss": validation_identity,
                "selection_objective": objective,
                "gate": float(adapter.scale.detach().cpu()),
                "best": improved,
                "checkpoint": str(checkpoint),
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))
            recent.clear()
            if args.patience > 0 and stale >= args.patience:
                print(f"Early stopping at step {step}")
                break

    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved best identity-aware adapter: {best_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Identity-aware Style-Slice training over frozen Seed-VC.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--style-adapter", required=True)
    parser.add_argument("--identity-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kv-lora", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--segment-seconds", type=float, default=1.5)
    parser.add_argument("--canonical-seconds", type=float, default=12.0)
    parser.add_argument("--prompt-ratio", type=float, default=0.25)
    parser.add_argument("--identity-weight", type=float, default=0.05)
    parser.add_argument("--identity-warmup-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adapter-l2", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--min-rms", type=float, default=0.005)
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
