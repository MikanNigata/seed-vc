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
from mosaic_svc.p0.f0_condition_adapter import (
    F0ConditionAdapter,
    F0ConditionAdapterConfig,
    save_f0_condition_adapter,
)
from mosaic_svc.p0.pitch_probe import load_pitch_probe
from mosaic_svc.p0.train_f0_embedding_adapter import (
    _acoustic_inputs,
    _condition,
    _pitch_labels,
    _pitch_loss,
)
from mosaic_svc.p4.train_prompt_mel_lora import _crop, _read_paths


def _freeze_seed_model(model) -> None:
    for value in model.values():
        if isinstance(value, torch.nn.Module):
            value.eval()
            for parameter in value.parameters():
                parameter.requires_grad = False
    model.cfm.estimator.class_dropout_prob = 0.0


@torch.no_grad()
def _validate(args, model, adapter, probe, semantic_fn, f0_fn, mel_fn, sr, device, paths, styles):
    adapter.eval()
    reconstruction_losses = []
    pitch_losses = []
    for path_index, path in enumerate(paths):
        audio = load_audio_tensor(path, sr=sr, device=device, max_seconds=None)
        audio = _crop(audio, sr, args.segment_seconds, random_crop=False, min_rms=args.min_rms)
        mel, semantic, f0, lengths = _acoustic_inputs(model, semantic_fn, f0_fn, mel_fn, audio, sr)
        condition, lengths = _condition(model, semantic, f0, lengths)
        condition = adapter(condition, f0)
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
            reconstruction_losses.append(float(reconstruction.cpu()))
            pitch_losses.append(float(pitch.cpu()))
    adapter.train()
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
    _freeze_seed_model(model)
    probe = load_pitch_probe(args.pitch_probe, device, trainable=False)
    adapter = F0ConditionAdapter(
        F0ConditionAdapterConfig(
            rank=args.rank,
            output_dim=args.output_dim,
            alpha=args.alpha,
            dropout=args.dropout,
            initial_scale=args.initial_scale,
            max_scale=args.max_scale,
        )
    ).to(device)

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
    recent: list[tuple[float, float]] = []
    best_objective = float("inf")
    best_path = output_dir / "f0_condition_adapter_best.pt"
    stale = 0

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
            adapted = adapter(condition, f0)
            reconstruction, predicted_mel = model.cfm(
                mel, lengths, prompt_len, adapted, style
            )
            pitch = _pitch_loss(
                probe,
                predicted_mel,
                labels,
                int(prompt_len.item()),
                args.unvoiced_weight,
            )
            pitch_weight = args.pitch_loss_weight * min(
                1.0, step / max(1, args.pitch_warmup_steps)
            )
            regularization = adapter.up.weight.float().square().mean()
            loss = reconstruction + pitch_weight * pitch + args.adapter_l2 * regularization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        recent.append((float(reconstruction.detach().cpu()), float(pitch.detach().cpu())))

        if step % args.validate_every == 0 or step == args.steps:
            validation_reconstruction, validation_pitch = _validate(
                args,
                model,
                adapter,
                probe,
                semantic_fn,
                f0_fn,
                mel_fn,
                sr,
                device,
                validation_paths,
                styles,
            )
            objective = validation_reconstruction + args.pitch_loss_weight * validation_pitch
            checkpoint = output_dir / f"f0_condition_adapter_step_{step:06d}.pt"
            metadata = {"step": step, "selection_objective": objective}
            save_f0_condition_adapter(adapter, checkpoint, metadata)
            improved = objective < best_objective - args.min_delta
            if improved:
                best_objective = objective
                shutil.copy2(checkpoint, best_path)
                stale = 0
            else:
                stale += 1
            row = {
                "step": step,
                "train_reconstruction": sum(item[0] for item in recent) / len(recent),
                "train_pitch_loss": sum(item[1] for item in recent) / len(recent),
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
            recent.clear()
            if args.patience > 0 and stale >= args.patience:
                print(f"Early stopping at step {step}")
                break

    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved best F0 condition adapter: {best_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a post-regulator F0 condition adapter.")
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
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=768)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--initial-scale", type=float, default=0.05)
    parser.add_argument("--max-scale", type=float, default=0.20)
    parser.add_argument("--pitch-loss-weight", type=float, default=0.10)
    parser.add_argument("--pitch-warmup-steps", type=int, default=100)
    parser.add_argument("--unvoiced-weight", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=8e-5)
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
