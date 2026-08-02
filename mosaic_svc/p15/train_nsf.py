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
from mosaic_svc.p15.losses import nsf_loss
from mosaic_svc.p15.nsf import NSFConfig, StreamingHarmonicNoiseNSF, save_nsf


def _paths(manifest):
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [value for row in rows if (value := row.get("feature_path") or row.get("path")) and Path(value).is_file()]


def _load(path, device, frames, random_crop):
    item = torch.load(path, map_location="cpu")
    mel, prosody, ap = (item[key].float() for key in ("target_mel", "prosody", "target_ap"))
    waveform = item["waveform"].float()
    hop = int(item.get("hop_length", 640))
    length = min(mel.size(1), prosody.size(1), ap.size(1), waveform.size(1) // hop)
    start = random.randint(0, length - frames) if random_crop and length > frames else 0
    stop = min(length, start + frames)
    return mel[:, start:stop].to(device), prosody[:, start:stop].to(device), ap[:, start:stop].to(device), waveform[:, start * hop:stop * hop].unsqueeze(1).to(device)


@torch.no_grad()
def _validate(model, paths, device, frames):
    model.eval()
    values = []
    for path in paths:
        mel, prosody, ap, waveform = _load(path, device, frames, False)
        prediction, _ = model(mel, prosody, ap)
        values.append(float(nsf_loss(prediction, waveform)[0].cpu()))
    model.train()
    return sum(values) / len(values)


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = NSFConfig(hidden_dim=args.hidden_dim, harmonics=args.harmonics, blocks=args.blocks)
    model = StreamingHarmonicNoiseNSF(config).to(device)
    train_paths, validation_paths = _paths(args.train_manifest), _paths(args.validation_manifest)
    if not train_paths or not validation_paths:
        raise ValueError("NSF manifests require P14 acoustic feature files")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best, stale, history = float("inf"), 0, []
    best_path = output / "streaming_nsf_best.pt"
    for step in tqdm(range(1, args.steps + 1)):
        mel, prosody, ap, waveform = _load(random.choice(train_paths), device, args.segment_frames, True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
            prediction, _ = model(mel, prosody, ap)
            loss, parts = nsf_loss(prediction, waveform)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if step % args.validate_every == 0 or step == args.steps:
            validation = _validate(model, validation_paths, device, args.segment_frames)
            checkpoint = output / f"streaming_nsf_step_{step:06d}.pt"
            save_nsf(model, checkpoint)
            improved = validation < best - args.min_delta
            if improved:
                best, stale = validation, 0
                shutil.copy2(checkpoint, best_path)
            else:
                stale += 1
            row = {"step": step, "train_loss": float(loss.detach().cpu()), **{f"train_{k}": float(v.detach().cpu()) for k, v in parts.items()}, "validation_loss": validation, "best": improved}
            history.append(row)
            print(json.dumps(row))
            if args.patience > 0 and stale >= args.patience:
                break
    with (output / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved NSF vocoder: {best_path}")
    return best_path


def build_parser():
    parser = argparse.ArgumentParser(description="Train the P15 harmonic-noise NSF vocoder.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--segment-frames", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--harmonics", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
