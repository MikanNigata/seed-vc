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
from mosaic_svc.p0.f0_block_adapter import (
    F0BlockAdapterConfig,
    install_f0_block_adapter,
    save_f0_block_adapter,
    set_f0_block_schedule,
)
from mosaic_svc.p0.train_f0_condition_adapter import _freeze_seed_model
from mosaic_svc.p0.train_f0_embedding_adapter import _acoustic_inputs, _condition
from mosaic_svc.p4.train_prompt_mel_lora import _crop, _read_paths


@torch.no_grad()
def _validate(args, model, adapter, wrappers, semantic_fn, f0_fn, mel_fn, sr, device, paths, styles):
    adapter.eval()
    losses = []
    for path_index, path in enumerate(paths):
        audio = load_audio_tensor(path, sr=sr, device=device, max_seconds=None)
        audio = _crop(audio, sr, args.segment_seconds, random_crop=False, min_rms=args.min_rms)
        mel, semantic, f0, lengths = _acoustic_inputs(model, semantic_fn, f0_fn, mel_fn, audio, sr)
        condition, lengths = _condition(model, semantic, f0, lengths)
        frames = int(lengths.item())
        mel = mel[:, :, :frames]
        prompt_frames = max(1, min(int(frames * args.prompt_ratio), frames - 1))
        prompt_len = torch.LongTensor([prompt_frames]).to(device)
        set_f0_block_schedule(adapter, wrappers, f0, frames, prompt_frames)
        for style_index, style in enumerate(styles):
            torch.manual_seed(args.seed + path_index * 97 + style_index)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
                reconstruction, _ = model.cfm(mel, lengths, prompt_len, condition, style)
            losses.append(float(reconstruction.cpu()))
    set_f0_block_schedule(adapter, wrappers, None)
    adapter.train()
    return sum(losses) / len(losses)


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
    _freeze_seed_model(model)
    adapter, wrappers = install_f0_block_adapter(
        model,
        config=F0BlockAdapterConfig(
            rank=args.rank,
            hidden_dim=args.hidden_dim,
            layer_indices=tuple(args.layers),
            alpha=args.alpha,
            dropout=args.dropout,
            initial_scale=args.initial_scale,
            max_scale=args.max_scale,
        ),
        trainable=True,
    )

    styles = []
    for canonical in args.canonical:
        audio = load_audio_tensor(canonical, sr=sr, device=device, max_seconds=args.canonical_seconds)
        styles.append(extract_campplus_style(campplus_model, audio, sr, device).detach())
    train_paths = _read_paths(args.train_manifest)
    validation_paths = _read_paths(args.validation_manifest)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    history: list[dict] = []
    recent = []
    best_loss = float("inf")
    best_path = output_dir / "f0_block_adapter_best.pt"
    stale = 0

    for step in tqdm(range(1, args.steps + 1)):
        audio = load_audio_tensor(random.choice(train_paths), sr=sr, device=device, max_seconds=None)
        audio = _crop(audio, sr, args.segment_seconds, random_crop=True, min_rms=args.min_rms)
        mel, semantic, f0, lengths = _acoustic_inputs(model, semantic_fn, f0_fn, mel_fn, audio, sr)
        condition, lengths = _condition(model, semantic, f0, lengths)
        frames = int(lengths.item())
        mel = mel[:, :, :frames]
        prompt_frames = max(1, min(int(frames * args.prompt_ratio), frames - 1))
        prompt_len = torch.LongTensor([prompt_frames]).to(device)
        style = random.choice(styles)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
            set_f0_block_schedule(adapter, wrappers, f0, frames, prompt_frames)
            reconstruction, _ = model.cfm(mel, lengths, prompt_len, condition, style)
            regularization = torch.stack(
                [projection.weight.float().square().mean() for projection in adapter.projections.values()]
            ).mean()
            loss = reconstruction + args.adapter_l2 * regularization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        recent.append(float(reconstruction.detach().cpu()))

        if step % args.validate_every == 0 or step == args.steps:
            validation_loss = _validate(
                args,
                model,
                adapter,
                wrappers,
                semantic_fn,
                f0_fn,
                mel_fn,
                sr,
                device,
                validation_paths,
                styles,
            )
            checkpoint = output_dir / f"f0_block_adapter_step_{step:06d}.pt"
            metadata = {"step": step, "validation_reconstruction": validation_loss}
            save_f0_block_adapter(adapter, checkpoint, metadata)
            improved = validation_loss < best_loss - args.min_delta
            if improved:
                best_loss = validation_loss
                shutil.copy2(checkpoint, best_path)
                stale = 0
            else:
                stale += 1
            row = {
                "step": step,
                "train_reconstruction": sum(recent) / len(recent),
                "validation_reconstruction": validation_loss,
                "gates": ",".join(
                    f"{index}:{float(adapter.scale(index).detach().cpu()):.6f}"
                    for index in adapter.config.layer_indices
                ),
                "adapter_rms": float(
                    torch.stack(
                        [projection.weight.float().square().mean() for projection in adapter.projections.values()]
                    ).mean().sqrt().detach().cpu()
                ),
                "best": improved,
                "checkpoint": str(checkpoint),
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))
            recent.clear()
            if args.patience > 0 and stale >= args.patience:
                print(f"Early stopping at step {step}")
                break

    set_f0_block_schedule(adapter, wrappers, None)
    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved best F0 block adapter: {best_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train layer-wise RMVPE F0 conditioning.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--canonical", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--canonical-seconds", type=float, default=12.0)
    parser.add_argument("--prompt-ratio", type=float, default=0.30)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--initial-scale", type=float, default=0.03)
    parser.add_argument("--max-scale", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adapter-l2", type=float, default=0.001)
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
