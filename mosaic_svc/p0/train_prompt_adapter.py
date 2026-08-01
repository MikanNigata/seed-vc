from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

import inference as seed_inference
from modules.commons import str2bool
from mosaic_svc.p0.audio_features import extract_campplus_style, load_audio_tensor
from mosaic_svc.p0.prompt_adapter import PromptAdapterConfig, install_prompt_adapter, save_prompt_adapter


def _read_paths(manifest: str) -> list[str]:
    with open(manifest, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    paths = [row["path"] for row in rows if row.get("approved", "true").lower() in {"1", "true", "yes", "y"}]
    if not paths:
        raise ValueError("manifest has no approved rows")
    return paths


def _crop(audio: torch.Tensor, sr: int, seconds: float) -> torch.Tensor:
    length = int(sr * seconds)
    if audio.size(-1) <= length:
        return audio
    start = random.randint(0, audio.size(-1) - length)
    return audio[:, start : start + length]


@torch.no_grad()
def _semantic(semantic_fn, audio: torch.Tensor, sr: int) -> torch.Tensor:
    return semantic_fn(torchaudio.functional.resample(audio, sr, 16000))


def run(args: argparse.Namespace) -> Path:
    seed_args = argparse.Namespace(
        f0_condition=True,
        checkpoint=args.checkpoint,
        config=args.config,
        fp16=args.fp16,
    )
    model, semantic_fn, f0_fn, _, campplus_model, mel_fn, mel_fn_args = seed_inference.load_models(seed_args)
    device = seed_inference.device
    sr = mel_fn_args["sampling_rate"]

    adapter = install_prompt_adapter(
        model,
        config=PromptAdapterConfig(
            rank=args.rank,
            dropout=args.dropout,
            initial_scale=args.initial_scale,
            max_scale=args.max_scale,
            source_only=args.source_only,
        ),
        trainable=True,
    )
    model.cfm.train()
    model.cfm.estimator.class_dropout_prob = 0.0
    adapter.train()

    canonical_audio = load_audio_tensor(args.canonical, sr=sr, device=device, max_seconds=args.canonical_seconds)
    canonical_style = extract_campplus_style(campplus_model, canonical_audio, sr, device).detach()
    paths = _read_paths(args.manifest)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    losses = []
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in tqdm(range(1, args.steps + 1)):
        path = random.choice(paths)
        audio = load_audio_tensor(path, sr=sr, device=device, max_seconds=None)
        audio = _crop(audio, sr, args.segment_seconds)
        mel = mel_fn(audio.float())
        target_lengths = torch.LongTensor([mel.size(2)]).to(device)

        with torch.no_grad():
            S = _semantic(semantic_fn, audio, sr)
            audio_16k = torchaudio.functional.resample(audio, sr, 16000)
            F0 = torch.from_numpy(f0_fn(audio_16k[0], thred=0.03)).to(device)[None]
            cond, *_ = model.length_regulator(S, ylens=target_lengths, n_quantizers=3, f0=F0)
            common = min(mel.size(2), cond.size(1))
            mel_train = mel[:, :, :common]
            cond = cond[:, :common]
            target_lengths = torch.LongTensor([common]).to(device)
            min_prompt = max(1, int(common * args.min_prompt_ratio))
            max_prompt = max(min_prompt, min(int(common * args.max_prompt_ratio), common - 1))
            prompt_len = torch.LongTensor([random.randint(min_prompt, max_prompt)]).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
            loss, _ = model.cfm(mel_train, target_lengths, prompt_len, cond, canonical_style)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))

        if step % args.log_every == 0:
            avg = sum(losses[-args.log_every :]) / min(len(losses), args.log_every)
            print(f"step={step} loss={avg:.6f} gate={float(adapter.scale.detach().cpu()):.6f}")
        if step % args.save_every == 0:
            save_prompt_adapter(adapter, output_dir / f"prompt_adapter_step_{step:06d}.pt")

    out = output_dir / "prompt_adapter_final.pt"
    save_prompt_adapter(adapter, out)
    print(f"Saved prompt adapter: {out}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Mosaic-SVC M2 prompt-aware adapter over frozen Seed-VC.")
    parser.add_argument("--manifest", required=True, help="CSV with at least path and approved columns.")
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--canonical-seconds", type=float, default=12.0)
    parser.add_argument("--min-prompt-ratio", type=float, default=0.20)
    parser.add_argument("--max-prompt-ratio", type=float, default=0.45)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--initial-scale", type=float, default=0.03)
    parser.add_argument("--max-scale", type=float, default=0.20)
    parser.add_argument("--source-only", type=str2bool, default=True)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
