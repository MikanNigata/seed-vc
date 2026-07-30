from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

import inference as seed_inference
from modules.commons import str2bool
from mosaic_svc.p0.audio_features import extract_campplus_style, load_audio_tensor


def _read_manifest(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(row: dict[str, str], key: str, default: float) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


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
    profile = torch.load(args.profile, map_location="cpu")
    centroid = profile["centroid"].float().to(device).view(1, -1)
    centroid = torch.nn.functional.normalize(centroid, dim=-1)

    rows = []
    for row in _read_manifest(args.manifest):
        audio = load_audio_tensor(row["path"], sr=sr, device=device, max_seconds=args.max_seconds)
        emb = extract_campplus_style(campplus_model, audio, sr, device)
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        speaker_similarity = float((emb @ centroid.T).item())
        quality = _as_float(row, "quality_score", 0.5)
        combined = args.speaker_weight * speaker_similarity + args.quality_weight * quality
        out = dict(row)
        out["speaker_similarity_to_profile"] = f"{speaker_similarity:.6f}"
        out["combined_profile_score"] = f"{combined:.6f}"
        rows.append(out)

    rows.sort(key=lambda r: float(r["combined_profile_score"]), reverse=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved ranked prompts: {output}")
    for idx, row in enumerate(rows[: args.print_top], start=1):
        print(
            f"{idx:02d} {row.get('name', Path(row['path']).stem)} "
            f"sim={row['speaker_similarity_to_profile']} score={row['combined_profile_score']}"
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank high-quality prompt clips by similarity to a dialogue speaker profile.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--speaker-weight", type=float, default=0.70)
    parser.add_argument("--quality-weight", type=float, default=0.30)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--print-top", type=int, default=8)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
