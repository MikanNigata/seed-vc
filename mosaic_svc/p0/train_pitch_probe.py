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
from modules.length_regulator import f0_to_coarse
from mosaic_svc.p0.audio_features import load_audio_tensor
from mosaic_svc.p0.pitch_probe import PitchProbe, PitchProbeConfig, save_pitch_probe
from mosaic_svc.p4.train_prompt_mel_lora import _crop, _read_paths


@torch.no_grad()
def _features(f0_fn, mel_fn, audio: torch.Tensor, sr: int):
    mel = mel_fn(audio.float())
    audio_16k = torchaudio.functional.resample(audio, sr, 16000)
    f0 = torch.from_numpy(f0_fn(audio_16k[0], thred=0.03)).to(audio.device)[None]
    coarse = f0_to_coarse(f0, 256).float().unsqueeze(1)
    labels = F.interpolate(coarse, size=mel.size(2), mode="nearest").squeeze(1).long()
    return mel, labels


def _loss(logits: torch.Tensor, labels: torch.Tensor, unvoiced_weight: float) -> torch.Tensor:
    weights = torch.ones(logits.size(1), device=logits.device)
    weights[1] = unvoiced_weight
    return F.cross_entropy(logits, labels, weight=weights)


@torch.no_grad()
def _validate(args, probe, f0_fn, mel_fn, sr, device, paths):
    probe.eval()
    losses = []
    accuracies = []
    voiced_mae = []
    for path in paths:
        audio = load_audio_tensor(path, sr=sr, device=device, max_seconds=None)
        audio = _crop(audio, sr, args.segment_seconds, random_crop=False, min_rms=args.min_rms)
        mel, labels = _features(f0_fn, mel_fn, audio, sr)
        logits = probe(mel.float())
        losses.append(float(_loss(logits, labels, args.unvoiced_weight).cpu()))
        prediction = logits.argmax(dim=1)
        accuracies.append(float((prediction == labels).float().mean().cpu()))
        voiced = labels > 1
        if voiced.any():
            voiced_mae.append(float((prediction[voiced] - labels[voiced]).abs().float().mean().cpu()))
    probe.train()
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies), sum(voiced_mae) / len(voiced_mae)


def run(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    seed_args = argparse.Namespace(f0_condition=True, checkpoint=args.checkpoint, config=args.config, fp16=args.fp16)
    _, _, f0_fn, _, _, mel_fn, mel_fn_args = seed_inference.load_models(seed_args)
    device = seed_inference.device
    sr = mel_fn_args["sampling_rate"]
    probe = PitchProbe(
        PitchProbeConfig(
            mel_bins=mel_fn_args["num_mels"],
            hidden_dim=args.hidden_dim,
            f0_bins=256,
            kernel_size=args.kernel_size,
            layers=args.layers,
            dropout=args.dropout,
        )
    ).to(device)
    train_paths = _read_paths(args.train_manifest)
    validation_paths = _read_paths(args.validation_manifest)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    history: list[dict] = []
    recent: list[float] = []
    best_loss = float("inf")
    best_path = output_dir / "pitch_probe_best.pt"

    for step in tqdm(range(1, args.steps + 1)):
        audio = load_audio_tensor(random.choice(train_paths), sr=sr, device=device, max_seconds=None)
        audio = _crop(audio, sr, args.segment_seconds, random_crop=True, min_rms=args.min_rms)
        mel, labels = _features(f0_fn, mel_fn, audio, sr)
        if args.noise_std > 0.0:
            mel = mel + torch.randn_like(mel) * random.uniform(0.0, args.noise_std)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
            logits = probe(mel.float())
            loss = _loss(logits, labels, args.unvoiced_weight)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        recent.append(float(loss.detach().cpu()))

        if step % args.validate_every == 0 or step == args.steps:
            validation_loss, accuracy, bin_mae = _validate(
                args, probe, f0_fn, mel_fn, sr, device, validation_paths
            )
            checkpoint = output_dir / f"pitch_probe_step_{step:06d}.pt"
            metadata = {"step": step, "validation_loss": validation_loss}
            save_pitch_probe(probe, checkpoint, metadata)
            improved = validation_loss < best_loss - args.min_delta
            if improved:
                best_loss = validation_loss
                shutil.copy2(checkpoint, best_path)
            row = {
                "step": step,
                "train_loss": sum(recent) / len(recent),
                "validation_loss": validation_loss,
                "validation_accuracy": accuracy,
                "validation_voiced_bin_mae": bin_mae,
                "best": improved,
                "checkpoint": str(checkpoint),
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))
            recent.clear()

    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved best pitch probe: {best_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a frozen mel-to-F0 probe for pitch supervision.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--unvoiced-weight", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--min-rms", type=float, default=0.005)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
