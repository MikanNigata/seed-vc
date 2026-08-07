from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

import inference as seed_inference
from modules.commons import str2bool
from modules.length_regulator import f0_to_coarse
from mosaic_svc.p0.audio_features import extract_campplus_style, load_audio_tensor
from mosaic_svc.p0.f0_embedding_adapter import (
    F0EmbeddingAdapterConfig,
    install_f0_embedding_adapter,
    save_f0_embedding_adapter,
)
from mosaic_svc.p0.pitch_probe import load_pitch_probe
from mosaic_svc.p4.train_prompt_mel_lora import _crop, _read_paths


@torch.no_grad()
def _acoustic_inputs(model, semantic_fn, f0_fn, mel_fn, audio: torch.Tensor, sr: int):
    mel = mel_fn(audio.float())
    audio_16k = torchaudio.functional.resample(audio, sr, 16000)
    semantic = semantic_fn(audio_16k)
    f0 = torch.from_numpy(f0_fn(audio_16k[0], thred=0.03)).to(audio.device)[None]
    lengths = torch.LongTensor([mel.size(2)]).to(audio.device)
    return mel, semantic, f0, lengths


def _condition(model, semantic, f0, lengths):
    condition, *_ = model.length_regulator(semantic, ylens=lengths, n_quantizers=3, f0=f0)
    common = min(condition.size(1), int(lengths.item()))
    return condition[:, :common], torch.LongTensor([common]).to(condition.device)


def _pitch_labels(f0: torch.Tensor, frames: int) -> torch.Tensor:
    coarse = f0_to_coarse(f0, 256).float().unsqueeze(1)
    return F.interpolate(coarse, size=frames, mode="nearest").squeeze(1).long()


def _pitch_loss(
    probe,
    predicted_mel: torch.Tensor,
    labels: torch.Tensor,
    prompt_frames: int,
    unvoiced_weight: float,
) -> torch.Tensor:
    logits = probe(predicted_mel.float())[:, :, prompt_frames:]
    targets = labels[:, prompt_frames : prompt_frames + logits.size(-1)]
    weights = torch.ones(logits.size(1), device=logits.device)
    weights[1] = unvoiced_weight
    return F.cross_entropy(logits, targets, weight=weights)


@torch.no_grad()
def _validate(args, model, probe, semantic_fn, f0_fn, mel_fn, sr, device, paths, styles):
    reconstruction_losses = []
    pitch_losses = []
    model.cfm.eval()
    for path_index, path in enumerate(paths):
        audio = load_audio_tensor(path, sr=sr, device=device, max_seconds=None)
        audio = _crop(audio, sr, args.segment_seconds, random_crop=False, min_rms=args.min_rms)
        mel, semantic, f0, lengths = _acoustic_inputs(model, semantic_fn, f0_fn, mel_fn, audio, sr)
        condition, lengths = _condition(model, semantic, f0, lengths)
        mel = mel[:, :, : lengths.item()]
        labels = _pitch_labels(f0, int(lengths.item()))
        prompt_len = torch.LongTensor(
            [max(1, min(int(lengths.item() * args.prompt_ratio), lengths.item() - 1))]
        ).to(device)
        for style_index, style in enumerate(styles):
            torch.manual_seed(args.seed + path_index * 97 + style_index)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
                reconstruction, predicted_mel = model.cfm(
                    mel, lengths, prompt_len, condition, style
                )
                pitch = _pitch_loss(
                    probe,
                    predicted_mel,
                    labels,
                    int(prompt_len.item()),
                    args.unvoiced_weight,
                )
            reconstruction_losses.append(float(reconstruction.detach().cpu()))
            pitch_losses.append(float(pitch.detach().cpu()))
    return (
        sum(reconstruction_losses) / len(reconstruction_losses),
        sum(pitch_losses) / len(pitch_losses),
    )


def run(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    seed_args = argparse.Namespace(
        f0_condition=True,
        checkpoint=args.checkpoint,
        config=args.config,
        fp16=args.fp16,
    )
    model, semantic_fn, f0_fn, _, campplus_model, mel_fn, mel_fn_args = seed_inference.load_models(seed_args)
    device = seed_inference.device
    sr = mel_fn_args["sampling_rate"]
    adapter = install_f0_embedding_adapter(
        model,
        config=F0EmbeddingAdapterConfig(
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            initial_scale=args.initial_scale,
            max_scale=args.max_scale,
        ),
        trainable=True,
    )
    probe = load_pitch_probe(args.pitch_probe, device, trainable=False)
    model.cfm.train()
    model.cfm.estimator.class_dropout_prob = 0.0

    styles = []
    for canonical in args.canonical:
        audio = load_audio_tensor(canonical, sr=sr, device=device, max_seconds=args.canonical_seconds)
        styles.append(extract_campplus_style(campplus_model, audio, sr, device).detach())
    train_paths = _read_paths(args.train_manifest)
    validation_paths = _read_paths(args.validation_manifest)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    history: list[dict] = []
    recent_losses: list[tuple[float, float]] = []
    best_objective = float("inf")
    best_path = output_dir / "f0_embedding_adapter_best.pt"
    stale_validations = 0

    for step in tqdm(range(1, args.steps + 1)):
        audio = load_audio_tensor(random.choice(train_paths), sr=sr, device=device, max_seconds=None)
        audio = _crop(audio, sr, args.segment_seconds, random_crop=True, min_rms=args.min_rms)
        mel, semantic, f0, lengths = _acoustic_inputs(model, semantic_fn, f0_fn, mel_fn, audio, sr)
        condition, lengths = _condition(model, semantic, f0, lengths)
        mel = mel[:, :, : lengths.item()]
        labels = _pitch_labels(f0, int(lengths.item()))
        prompt_len = torch.LongTensor(
            [max(1, min(int(lengths.item() * args.prompt_ratio), lengths.item() - 1))]
        ).to(device)
        style = random.choice(styles)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
            reconstruction, predicted_mel = model.cfm(
                mel, lengths, prompt_len, condition, style
            )
            pitch = _pitch_loss(
                probe,
                predicted_mel,
                labels,
                int(prompt_len.item()),
                args.unvoiced_weight,
            )
            regularization = adapter.up.weight.float().square().mean()
            pitch_weight = args.pitch_loss_weight * min(
                1.0, step / max(1, args.pitch_warmup_steps)
            )
            loss = reconstruction + pitch_weight * pitch + args.adapter_l2 * regularization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        recent_losses.append(
            (float(reconstruction.detach().cpu()), float(pitch.detach().cpu()))
        )

        if step % args.validate_every == 0 or step == args.steps:
            validation_reconstruction, validation_pitch = _validate(
                args,
                model,
                probe,
                semantic_fn,
                f0_fn,
                mel_fn,
                sr,
                device,
                validation_paths,
                styles,
            )
            train_reconstruction = sum(item[0] for item in recent_losses) / len(recent_losses)
            train_pitch = sum(item[1] for item in recent_losses) / len(recent_losses)
            objective = validation_reconstruction + args.pitch_loss_weight * validation_pitch
            checkpoint = output_dir / f"f0_embedding_adapter_step_{step:06d}.pt"
            save_f0_embedding_adapter(adapter, checkpoint)
            improved = objective < best_objective - args.min_delta
            if improved:
                best_objective = objective
                shutil.copy2(checkpoint, best_path)
                stale_validations = 0
            else:
                stale_validations += 1
            row = {
                "step": step,
                "train_reconstruction": train_reconstruction,
                "train_pitch_loss": train_pitch,
                "validation_reconstruction": validation_reconstruction,
                "validation_pitch_loss": validation_pitch,
                "selection_objective": objective,
                "gate": float(adapter.scale.detach().cpu()),
                "adapter_rms": float(adapter.up.weight.float().square().mean().sqrt().detach().cpu()),
                "best": improved,
                "checkpoint": str(checkpoint),
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))
            recent_losses.clear()
            if args.patience > 0 and stale_validations >= args.patience:
                print(f"Early stopping at step {step}")
                break

    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    metadata = {
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "best_objective": best_objective,
        "best_checkpoint": str(best_path),
        "train_manifest": args.train_manifest,
        "validation_manifest": args.validation_manifest,
        "canonical": args.canonical,
        "pitch_probe": args.pitch_probe,
        "pitch_loss_weight": args.pitch_loss_weight,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved best F0 embedding adapter: {best_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a target-speaker F0 embedding adapter.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--canonical", action="append", required=True)
    parser.add_argument("--pitch-probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--canonical-seconds", type=float, default=12.0)
    parser.add_argument("--prompt-ratio", type=float, default=0.30)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--initial-scale", type=float, default=0.05)
    parser.add_argument("--max-scale", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adapter-l2", type=float, default=0.001)
    parser.add_argument("--pitch-loss-weight", type=float, default=0.10)
    parser.add_argument("--pitch-warmup-steps", type=int, default=100)
    parser.add_argument("--unvoiced-weight", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-rms", type=float, default=0.005)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
