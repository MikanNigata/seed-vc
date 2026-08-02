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
from mosaic_svc.r16.streaming_modules import CausalContentStudent, StreamingAcousticConverter, StreamingConfig


def _paths(manifest):
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paths = [row.get("feature_path") or row.get("path") for row in rows]
    return [path for path in paths if path and Path(path).is_file()]


def _load(path, device, frames, random_crop, student=None):
    item = torch.load(path, map_location="cpu")
    content = item["teacher_features"].float()
    student_input = item["input_features"].float()
    prosody = item["prosody"].float()
    mel = item["target_mel"].float()
    length = min(content.size(1), student_input.size(1), prosody.size(1), mel.size(1))
    start = random.randint(0, length - frames) if random_crop and length > frames else 0
    stop = min(length, start + frames)
    content, student_input, prosody, mel = content[:, start:stop], student_input[:, start:stop], prosody[:, start:stop], mel[:, start:stop]
    if student is not None:
        with torch.no_grad():
            content = student(student_input.to(device)).cpu()
    return content.to(device), prosody.to(device), mel.to(device)


def _load_student(path, device):
    if not path:
        return None
    checkpoint = torch.load(path, map_location="cpu")
    model = CausalContentStudent(config=StreamingConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False).to(device)


def _loss(prediction, target):
    l1 = (prediction - target).abs().mean()
    convergence = torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target).clamp_min(1e-6)
    delta = ((prediction[:, 1:] - prediction[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs().mean()
    return l1 + 0.1 * convergence + 0.25 * delta, {"mel_l1": l1, "spectral_convergence": convergence, "mel_delta": delta}


@torch.no_grad()
def _validate(model, paths, style, device, frames, student):
    model.eval()
    values = []
    for path in paths:
        content, prosody, mel = _load(path, device, frames, False, student)
        prediction, _ = model(content, prosody, style)
        loss, _ = _loss(prediction, mel)
        values.append(float(loss.cpu()))
    model.train()
    return sum(values) / len(values)


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = StreamingConfig(hidden_dim=args.hidden_dim, layers=args.layers, kernel_size=args.kernel_size)
    model = StreamingAcousticConverter(config).to(device)
    student = _load_student(args.student, device)
    profile = torch.load(args.identity_profile, map_location="cpu")
    style = profile["centroid"].float().view(1, -1).to(device)
    train_paths, validation_paths = _paths(args.train_manifest), _paths(args.validation_manifest)
    if not train_paths or not validation_paths:
        raise ValueError("converter manifests require acoustic feature files")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best, stale, history = float("inf"), 0, []
    best_path = output_dir / "streaming_converter_best.pt"
    for step in tqdm(range(1, args.steps + 1)):
        content, prosody, mel = _load(random.choice(train_paths), device, args.segment_frames, True, student)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
            prediction, _ = model(content, prosody, style)
            loss, parts = _loss(prediction, mel)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if step % args.validate_every == 0 or step == args.steps:
            validation = _validate(model, validation_paths, style, device, args.segment_frames, student)
            checkpoint = output_dir / f"streaming_converter_step_{step:06d}.pt"
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
    print(f"Saved streaming converter: {best_path}")
    return best_path


def build_parser():
    parser = argparse.ArgumentParser(description="Train the P14 streaming acoustic converter.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--identity-profile", required=True)
    parser.add_argument("--student", help="Frozen P13 student checkpoint; recommended to match runtime inputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--segment-frames", type=int, default=150)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
