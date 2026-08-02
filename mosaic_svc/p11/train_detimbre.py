from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import librosa
import torch
from tqdm import tqdm

from modules.commons import str2bool
from mosaic_svc.p11.content_teacher import ContentTeacher, ContentTeacherConfig, save_content_teacher
from mosaic_svc.p11.encoders import FrozenTeacherEncoders
from mosaic_svc.p11.losses import detimbre_loss
from mosaic_svc.p11.perturb import timbre_perturb


def _read_paths(manifest: str | Path) -> list[str]:
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paths = [row["path"] for row in rows if row.get("approved", "true").lower() in {"1", "true", "yes", "y"}]
    if not paths:
        raise ValueError("manifest has no approved audio")
    return paths


def _load_crop(path: str, seconds: float, random_crop: bool) -> torch.Tensor:
    audio, _ = librosa.load(path, sr=16000, mono=True)
    length = int(seconds * 16000)
    if audio.size > length:
        start = random.randint(0, audio.size - length) if random_crop else 0
        audio = audio[start : start + length]
    elif audio.size < length:
        audio = torch.nn.functional.pad(torch.from_numpy(audio), (0, length - audio.size)).numpy()
    return torch.from_numpy(audio).float().unsqueeze(0)


@torch.no_grad()
def _validate(args, encoders, teacher, paths, device) -> dict[str, float]:
    teacher.eval()
    totals = {"loss": 0.0, "consistency": 0.0, "retention": 0.0, "delta": 0.0}
    for index, path in enumerate(paths):
        random.seed(args.seed + index)
        audio = _load_crop(path, args.segment_seconds, random_crop=False).to(device)
        perturbed = timbre_perturb(audio, 16000)
        cv, whisper = encoders(audio)
        cv_alt, whisper_alt = encoders(perturbed)
        fused = teacher.fusion(cv, whisper)
        output = teacher.detimbre(fused)
        output_alt = teacher(cv_alt, whisper_alt)
        loss, parts = detimbre_loss(
            output,
            output_alt,
            fused,
            consistency_weight=args.consistency_weight,
            retention_weight=args.retention_weight,
            delta_weight=args.delta_weight,
        )
        totals["loss"] += float(loss.cpu())
        for key, value in parts.items():
            totals[key] += float(value.cpu())
    teacher.train()
    return {key: value / len(paths) for key, value in totals.items()}


def run(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = FrozenTeacherEncoders(args.contentvec, args.whisper, device=device, fp16=args.fp16)
    teacher = ContentTeacher(
        ContentTeacherConfig(
            hidden_dim=args.hidden_dim,
            bottleneck_dim=args.bottleneck_dim,
            layers=args.layers,
            max_whisper_gate=args.max_whisper_gate,
            initial_whisper_gate=args.initial_whisper_gate,
            max_adapter_scale=args.max_adapter_scale,
            initial_adapter_scale=args.initial_adapter_scale,
            dropout=args.dropout,
        )
    ).to(device)
    train_paths = _read_paths(args.train_manifest)
    validation_paths = _read_paths(args.validation_manifest)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    best = float("inf")
    stale = 0
    history = []
    best_path = output_dir / "content_teacher_best.pt"

    for step in tqdm(range(1, args.steps + 1)):
        audio = _load_crop(random.choice(train_paths), args.segment_seconds, random_crop=True).to(device)
        perturbed = timbre_perturb(audio, 16000)
        with torch.no_grad():
            cv, whisper = encoders(audio)
            cv_alt, whisper_alt = encoders(perturbed)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
            fused = teacher.fusion(cv, whisper)
            output = teacher.detimbre(fused)
            output_alt = teacher(cv_alt, whisper_alt)
            loss, parts = detimbre_loss(
                output,
                output_alt,
                fused,
                consistency_weight=args.consistency_weight,
                retention_weight=args.retention_weight,
                delta_weight=args.delta_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % args.validate_every == 0 or step == args.steps:
            metrics = _validate(args, encoders, teacher, validation_paths, device)
            checkpoint = output_dir / f"content_teacher_step_{step:06d}.pt"
            save_content_teacher(teacher, checkpoint)
            improved = metrics["loss"] < best - args.min_delta
            if improved:
                best = metrics["loss"]
                shutil.copy2(checkpoint, best_path)
                stale = 0
            else:
                stale += 1
            row = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                **{f"validation_{key}": value for key, value in metrics.items()},
                "whisper_gate": float(teacher.fusion.whisper_gate.detach().cpu()),
                "adapter_scale": float(teacher.detimbre.scale.detach().cpu()),
                "best": improved,
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))
            if args.patience > 0 and stale >= args.patience:
                break

    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved content teacher: {best_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Mosaic-SVC P11 Content Fusion and De-Timbre Adapter.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--contentvec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--whisper", default="openai/whisper-small")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-whisper-gate", type=float, default=0.30)
    parser.add_argument("--initial-whisper-gate", type=float, default=0.10)
    parser.add_argument("--max-adapter-scale", type=float, default=0.25)
    parser.add_argument("--initial-adapter-scale", type=float, default=0.05)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--retention-weight", type=float, default=0.5)
    parser.add_argument("--delta-weight", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
