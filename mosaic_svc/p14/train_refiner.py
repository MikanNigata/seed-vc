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
from mosaic_svc.p14.refiner import CausalAcousticRefiner, RefinerConfig, save_refiner
from mosaic_svc.p14.train_converter import _load, _load_student, _loss, _paths
from mosaic_svc.r16.streaming_modules import StreamingAcousticConverter, StreamingConfig
from mosaic_svc.r16.style_conditioning import load_conditioned_style


def _load_converter(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = StreamingAcousticConverter(StreamingConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False).to(device)


def _predict(converter, refiner, content, prosody, style):
    with torch.no_grad():
        latent = converter.encode(content, prosody, style)
        mel = converter.mel(latent)
    return refiner(latent, mel, prosody, style)


@torch.no_grad()
def _validate(converter, refiner, paths, style, device, frames, student):
    refiner.eval()
    values = []
    for path in paths:
        content, prosody, target = _load(path, device, frames, False, student)
        values.append(float(_loss(_predict(converter, refiner, content, prosody, style), target)[0].cpu()))
    refiner.train()
    return sum(values) / len(values)


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    converter = _load_converter(args.converter, device)
    student = _load_student(args.student, device)
    style = load_conditioned_style(
        args.identity_profile,
        device,
        args.prototype_bank,
        args.prototype_strength,
        args.prototype_max_norm_ratio,
        args.prototype_max_gate,
    )
    config = RefinerConfig(
        latent_dim=converter.config.hidden_dim,
        mel_dim=converter.config.mel_dim,
        style_dim=converter.config.style_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        kernel_size=args.kernel_size,
        max_scale=args.max_scale,
    )
    model = CausalAcousticRefiner(config).to(device)
    train_paths, validation_paths = _paths(args.train_manifest), _paths(args.validation_manifest)
    if not train_paths or not validation_paths:
        raise ValueError("refiner manifests require acoustic feature files")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best, stale, history = float("inf"), 0, []
    best_path = output / "acoustic_refiner_best.pt"
    for step in tqdm(range(1, args.steps + 1)):
        content, prosody, target = _load(random.choice(train_paths), device, args.segment_frames, True, student)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
            prediction = _predict(converter, model, content, prosody, style)
            loss, parts = _loss(prediction, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if step % args.validate_every == 0 or step == args.steps:
            validation = _validate(converter, model, validation_paths, style, device, args.segment_frames, student)
            checkpoint = output / f"acoustic_refiner_step_{step:06d}.pt"
            save_refiner(model, checkpoint)
            improved = validation < best - args.min_delta
            if improved:
                best, stale = validation, 0
                shutil.copy2(checkpoint, best_path)
            else:
                stale += 1
            row = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                **{f"train_{key}": float(value.detach().cpu()) for key, value in parts.items()},
                "validation_loss": validation,
                "scale": float(model.scale.detach().cpu()),
                "best": improved,
            }
            history.append(row)
            print(json.dumps(row))
            if args.patience > 0 and stale >= args.patience:
                break
    with (output / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved acoustic refiner: {best_path}")
    return best_path


def build_parser():
    parser = argparse.ArgumentParser(description="Train the bounded P14 acoustic residual refiner.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--identity-profile", required=True)
    parser.add_argument("--student")
    parser.add_argument("--converter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prototype-bank")
    parser.add_argument("--prototype-strength", type=float, default=1.0)
    parser.add_argument("--prototype-max-norm-ratio", type=float, default=0.10)
    parser.add_argument("--prototype-max-gate", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--segment-frames", type=int, default=150)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--max-scale", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
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
