from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from modules.commons import str2bool
from mosaic_svc.p11.grl import SpeakerAdversarialProbe
from mosaic_svc.p13.distillation import dynamic_chunk, student_distillation_loss
from mosaic_svc.r16.streaming_modules import CausalContentStudent, StreamingConfig


def _read_manifest(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paths = [row.get("feature_path") or row.get("path") for row in rows]
    paths = [path for path in paths if path and Path(path).is_file()]
    if not paths:
        raise ValueError("feature manifest contains no readable .pt files")
    return paths


def _load(path: str, device):
    item = torch.load(path, map_location="cpu")
    return item["input_features"].float().to(device), item["teacher_features"].float().to(device)


def _load_speaker_probe(path, device):
    if not path:
        return None
    checkpoint = torch.load(path, map_location="cpu")
    speakers = checkpoint.get("speakers")
    if not speakers:
        raise ValueError("speaker probe checkpoint has no speaker labels")
    feature_dim = int(checkpoint["feature_dim"])
    hidden_dim = int(checkpoint["state_dict"]["network.0.weight"].size(0))
    probe = SpeakerAdversarialProbe(feature_dim, len(speakers), hidden_dim).to(device)
    probe.load_state_dict(checkpoint["state_dict"])
    return probe.eval().requires_grad_(False)


def _speaker_logits(probe, features):
    if probe is None:
        return None
    pooled = torch.cat([features.mean(1), features.std(1)], dim=-1)
    return probe.network(pooled)


@torch.no_grad()
def _validate(model, paths, device, probe=None, leakage_weight=0.05):
    model.eval()
    values = []
    for path in paths:
        source, teacher = _load(path, device)
        student = model(source)
        loss, _ = student_distillation_loss(
            student, teacher, _speaker_logits(probe, student), leakage_weight
        )
        values.append(float(loss.cpu()))
    model.train()
    return sum(values) / len(values)


def run(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = StreamingConfig(
        content_dim=args.content_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        kernel_size=args.kernel_size,
    )
    model = CausalContentStudent(input_dim=80, config=config).to(device)
    speaker_probe = _load_speaker_probe(args.speaker_probe, device)
    train_paths = _read_manifest(args.train_manifest)
    validation_paths = _read_manifest(args.validation_manifest)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best, stale, history = float("inf"), 0, []
    best_path = output_dir / "content_student_best.pt"

    for step in tqdm(range(1, args.steps + 1)):
        source, teacher = _load(random.choice(train_paths), device)
        source, teacher = dynamic_chunk(source, teacher, args.min_chunk_frames, args.max_chunk_frames)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
            prediction = model(source)
            loss, parts = student_distillation_loss(
                prediction,
                teacher,
                _speaker_logits(speaker_probe, prediction),
                args.speaker_leakage_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if step % args.validate_every == 0 or step == args.steps:
            validation = _validate(
                model, validation_paths, device, speaker_probe, args.speaker_leakage_weight
            )
            checkpoint = output_dir / f"content_student_step_{step:06d}.pt"
            torch.save({"config": config.__dict__, "state_dict": model.state_dict()}, checkpoint)
            improved = validation < best - args.min_delta
            if improved:
                best = validation
                shutil.copy2(checkpoint, best_path)
                stale = 0
            else:
                stale += 1
            row = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                **{f"train_{key}": float(value.detach().cpu()) for key, value in parts.items()},
                "validation_loss": validation,
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
    print(f"Saved content student: {best_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill P11 teacher features into a causal content student.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--speaker-probe", help="Frozen external multi-speaker P11 probe checkpoint")
    parser.add_argument("--speaker-leakage-weight", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--content-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--min-chunk-frames", type=int, default=2)
    parser.add_argument("--max-chunk-frames", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
