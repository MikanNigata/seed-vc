from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import torch

import inference as seed_inference
from modules.commons import str2bool
from mosaic_svc.p0.audio_features import extract_campplus_style


def _robust_centroid(embeddings: torch.Tensor, keep_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    center = torch.nn.functional.normalize(normalized.mean(dim=0, keepdim=True), dim=-1)
    sims = (normalized @ center.T).squeeze(1)
    keep = max(1, int(len(sims) * keep_ratio))
    top = sims.topk(keep).indices
    centroid = torch.nn.functional.normalize(embeddings[top].mean(dim=0, keepdim=True), dim=-1)
    return centroid, top.cpu()


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    seed_args = argparse.Namespace(
        f0_condition=True,
        checkpoint=args.checkpoint,
        config=args.config,
        fp16=args.fp16,
    )
    _, _, _, _, campplus_model, _, mel_fn_args = seed_inference.load_models(seed_args)
    device = seed_inference.device
    sr = mel_fn_args["sampling_rate"]

    wav, _ = librosa.load(args.input, sr=sr, mono=True)
    chunk_len = int(args.chunk_seconds * sr)
    hop_len = int(args.hop_seconds * sr)
    starts = list(range(0, max(1, len(wav) - chunk_len + 1), hop_len))
    if args.max_segments > 0:
        stride = max(1, len(starts) // args.max_segments)
        starts = starts[::stride][: args.max_segments]

    embeddings = []
    rows = []
    for idx, start in enumerate(starts, start=1):
        clip = wav[start : start + chunk_len]
        if clip.size < int(args.min_seconds * sr):
            continue
        audio = torch.tensor(clip).unsqueeze(0).float().to(device)
        emb = extract_campplus_style(campplus_model, audio, sr, device).squeeze(0).detach().cpu()
        embeddings.append(emb)
        rows.append(
            {
                "index": idx,
                "start_seconds": round(start / sr, 3),
                "duration_seconds": round(clip.size / sr, 3),
            }
        )

    if not embeddings:
        raise ValueError("no usable dialogue segments")

    emb_tensor = torch.stack(embeddings, dim=0)
    centroid, kept = _robust_centroid(emb_tensor, args.keep_ratio)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source": str(Path(args.input)),
            "sample_rate": sr,
            "chunk_seconds": args.chunk_seconds,
            "hop_seconds": args.hop_seconds,
            "keep_ratio": args.keep_ratio,
            "embeddings": emb_tensor,
            "centroid": centroid.squeeze(0),
            "kept_indices": kept,
        },
        output,
    )

    report = output.with_suffix(".csv")
    kept_set = set(int(i) for i in kept.tolist())
    for idx, row in enumerate(rows):
        row["kept_for_centroid"] = idx in kept_set
    with open(report, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved speaker profile: {output}")
    print(f"Saved segment report: {report}")
    print(f"Segments: {len(embeddings)} kept: {len(kept)}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a CAMPPlus speaker profile from dialogue audio.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    parser.add_argument("--hop-seconds", type=float, default=16.0)
    parser.add_argument("--min-seconds", type=float, default=4.0)
    parser.add_argument("--max-segments", type=int, default=96)
    parser.add_argument("--keep-ratio", type=float, default=0.70)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
