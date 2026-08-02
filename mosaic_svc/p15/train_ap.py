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
from mosaic_svc.p14.refiner import load_refiner
from mosaic_svc.p15.ap_head import APHeadConfig, TargetAPHead
from mosaic_svc.r16.streaming_modules import StreamingAcousticConverter, StreamingConfig
from mosaic_svc.r16.style_conditioning import load_conditioned_style


def _paths(manifest):
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [value for row in rows if (value := row.get("feature_path") or row.get("path")) and Path(value).is_file()]


def _load_converter(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = StreamingAcousticConverter(StreamingConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().requires_grad_(False)
    return model.to(device)


def _load_item(path, device, frames, random_crop):
    item = torch.load(path, map_location="cpu")
    keys = ("teacher_features", "prosody", "target_mel", "target_ap")
    tensors = [item[key].float() for key in keys]
    length = min(tensor.size(1) for tensor in tensors)
    start = random.randint(0, length - frames) if random_crop and length > frames else 0
    stop = min(length, start + frames)
    return [tensor[:, start:stop].to(device) for tensor in tensors]


def _ap_loss(prediction, target):
    probability = prediction[..., :-2].float().clamp(1e-6, 1.0 - 1e-6)
    truth = target[..., :-2].float()
    bands = -(truth * probability.log() + (1.0 - truth) * (1.0 - probability).log()).mean()
    summary = torch.nn.functional.l1_loss(prediction[..., -2:], target[..., -2:])
    temporal = torch.nn.functional.l1_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1])
    return bands + summary + 0.25 * temporal, {"band_bce": bands, "summary_l1": summary, "temporal_l1": temporal}


def _predict(converter, head, item, style, refiner=None):
    content, prosody, target_mel, target_ap = item
    latent = converter.encode(content, prosody, style)
    mel = converter.mel(latent)
    if refiner is not None:
        mel = refiner(latent, mel, prosody, style)
    return head(latent, mel, prosody, style), target_ap


@torch.no_grad()
def _validate(converter, head, paths, style, device, frames, refiner=None):
    head.eval()
    values = []
    for path in paths:
        prediction, target = _predict(converter, head, _load_item(path, device, frames, False), style, refiner)
        values.append(float(_ap_loss(prediction, target)[0].cpu()))
    head.train()
    return sum(values) / len(values)


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    converter = _load_converter(args.converter, device)
    refiner = load_refiner(args.refiner, device).eval().requires_grad_(False) if args.refiner else None
    converter_config = converter.config
    config = APHeadConfig(latent_dim=converter_config.hidden_dim, bands=args.bands)
    head = TargetAPHead(config).to(device)
    style = load_conditioned_style(
        args.identity_profile,
        device,
        args.prototype_bank,
        args.prototype_strength,
        args.prototype_max_norm_ratio,
        args.prototype_max_gate,
    )
    train_paths, validation_paths = _paths(args.train_manifest), _paths(args.validation_manifest)
    if not train_paths or not validation_paths:
        raise ValueError("AP manifests require P14 acoustic feature files")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best, stale, history = float("inf"), 0, []
    best_path = output / "ap_head_best.pt"
    for step in tqdm(range(1, args.steps + 1)):
        item = _load_item(random.choice(train_paths), device, args.segment_frames, True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
            prediction, target = _predict(converter, head, item, style, refiner)
            loss, parts = _ap_loss(prediction, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if step % args.validate_every == 0 or step == args.steps:
            validation = _validate(converter, head, validation_paths, style, device, args.segment_frames, refiner)
            checkpoint = output / f"ap_head_step_{step:06d}.pt"
            torch.save({"config": config.__dict__, "state_dict": head.state_dict()}, checkpoint)
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
    print(f"Saved AP head: {best_path}")
    return best_path


def build_parser():
    parser = argparse.ArgumentParser(description="Train P15 AP Head with the acoustic converter frozen.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--identity-profile", required=True)
    parser.add_argument("--prototype-bank")
    parser.add_argument("--prototype-strength", type=float, default=1.0)
    parser.add_argument("--prototype-max-norm-ratio", type=float, default=0.10)
    parser.add_argument("--prototype-max-gate", type=float, default=0.25)
    parser.add_argument("--converter", required=True)
    parser.add_argument("--refiner")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--segment-frames", type=int, default=150)
    parser.add_argument("--bands", type=int, default=8)
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
