from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

import inference as seed_inference
from modules.commons import str2bool
from mosaic_svc.p0.audio_features import extract_campplus_style, load_audio_tensor
from mosaic_svc.p4.train_prompt_mel_lora import _crop, _features, _read_paths
from mosaic_svc.p5.kv_lora import KVLoRAConfig, install_kv_lora, save_kv_lora, trainable_parameters


@torch.no_grad()
def _validate(args, model, semantic_fn, f0_fn, mel_fn, sr, device, paths, styles) -> float:
    model.cfm.eval()
    losses = []
    for path_index, path in enumerate(paths):
        audio = load_audio_tensor(path, sr=sr, device=device, max_seconds=None)
        audio = _crop(audio, sr, args.segment_seconds, random_crop=False, min_rms=args.min_rms)
        mel, condition, lengths = _features(model, semantic_fn, f0_fn, mel_fn, audio, sr)
        prompt_len = torch.LongTensor([max(1, min(int(lengths.item() * args.prompt_ratio), lengths.item() - 1))]).to(device)
        for style_index, style in enumerate(styles):
            torch.manual_seed(args.seed + path_index * 97 + style_index)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
                loss, _ = model.cfm(mel, lengths, prompt_len, condition, style)
            losses.append(float(loss.detach().cpu()))
    model.cfm.train()
    model.cfm.estimator.class_dropout_prob = 0.0
    return sum(losses) / len(losses)


def run(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    seed_args = argparse.Namespace(f0_condition=True, checkpoint=args.checkpoint, config=args.config, fp16=args.fp16)
    model, semantic_fn, f0_fn, _, campplus_model, mel_fn, mel_fn_args = seed_inference.load_models(seed_args)
    device = seed_inference.device
    sr = mel_fn_args["sampling_rate"]
    layer_indices = [int(item) for item in args.layers.split(",") if item.strip()]
    adapters = install_kv_lora(
        model,
        layer_indices=layer_indices,
        config=KVLoRAConfig(
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            initial_scale=args.initial_scale,
            max_scale=args.max_scale,
        ),
        trainable=True,
    )
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
    parameters = list(trainable_parameters(adapters))
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    history, recent_losses = [], []
    best_validation = float("inf")
    best_path = output_dir / "kv_lora_best.pt"
    stale_validations = 0

    for step in tqdm(range(1, args.steps + 1)):
        audio = load_audio_tensor(random.choice(train_paths), sr=sr, device=device, max_seconds=None)
        audio = _crop(audio, sr, args.segment_seconds, random_crop=True, min_rms=args.min_rms)
        mel, condition, lengths = _features(model, semantic_fn, f0_fn, mel_fn, audio, sr)
        prompt_len = torch.LongTensor([max(1, min(int(lengths.item() * args.prompt_ratio), lengths.item() - 1))]).to(device)
        style = random.choice(styles)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
            reconstruction, _ = model.cfm(mel, lengths, prompt_len, condition, style)
            regularization = torch.stack([adapter.up.weight.float().square().mean() for adapter in adapters.values()]).mean()
            loss = reconstruction + args.adapter_l2 * regularization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        recent_losses.append(float(reconstruction.detach().cpu()))

        if step % args.validate_every == 0 or step == args.steps:
            validation = _validate(args, model, semantic_fn, f0_fn, mel_fn, sr, device, validation_paths, styles)
            train_loss = sum(recent_losses) / len(recent_losses)
            checkpoint = output_dir / f"kv_lora_step_{step:06d}.pt"
            save_kv_lora(adapters, checkpoint)
            improved = validation < best_validation - args.min_delta
            if improved:
                best_validation = validation
                shutil.copy2(checkpoint, best_path)
                stale_validations = 0
            else:
                stale_validations += 1
            row = {
                "step": step,
                "train_loss": train_loss,
                "validation_loss": validation,
                "gate": sum(float(item.scale.detach().cpu()) for item in adapters.values()) / len(adapters),
                "adapter_rms": sum(item.last_delta_rms for item in adapters.values()) / len(adapters),
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
        "layers": layer_indices,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "best_validation_loss": best_validation,
        "best_checkpoint": str(best_path),
        "canonical": args.canonical,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved best K/V LoRA: {best_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train middle-block K/V-only LoRA over frozen Seed-VC.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--canonical", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="8")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--canonical-seconds", type=float, default=12.0)
    parser.add_argument("--prompt-ratio", type=float, default=0.30)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--initial-scale", type=float, default=0.01)
    parser.add_argument("--max-scale", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adapter-l2", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-rms", type=float, default=0.005)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
