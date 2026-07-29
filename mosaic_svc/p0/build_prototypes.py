from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

import inference as seed_inference
from mosaic_svc.p0.audio_features import extract_campplus_style, load_audio_tensor
from mosaic_svc.p0.prototype_bank import PrototypeMeta, build_bank
from mosaic_svc.p0.quality import audit_audio


def run(args: argparse.Namespace) -> Path:
    seed_args = argparse.Namespace(
        f0_condition=True,
        checkpoint=args.checkpoint,
        config=args.config,
        fp16=True,
    )
    _, _, _, _, campplus_model, _, mel_fn_args = seed_inference.load_models(seed_args)
    device = seed_inference.device
    sr = mel_fn_args["sampling_rate"]

    canonical_audio = load_audio_tensor(args.canonical, sr=sr, device=device, max_seconds=args.max_seconds)
    canonical = extract_campplus_style(campplus_model, canonical_audio, sr, device).detach().cpu()

    rows = []
    items = []
    with open(args.manifest, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["path"]
            quality = audit_audio(path, sr=sr)
            approved = row.get("approved", "true").lower() in {"1", "true", "yes", "y"}
            approved = approved and quality.quality_score >= args.min_quality_score
            audio = load_audio_tensor(path, sr=sr, device=device, max_seconds=args.max_seconds)
            emb = extract_campplus_style(campplus_model, audio, sr, device).detach().cpu()
            meta = PrototypeMeta(
                name=row.get("name") or Path(path).stem,
                path=path,
                quality=float(row.get("quality") or quality.quality_score),
                category=row.get("category") or "neutral_mid",
                approved=approved,
            )
            items.append((emb, meta))
            rows.append({**row, **quality.to_dict(), "approved_final": approved})

    bank = build_bank(items, canonical=canonical)
    out = Path(args.output)
    bank.save(out)
    audit_path = out.with_suffix(".audit.csv")
    with open(audit_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved prototype bank: {out}")
    print(f"Saved audit: {audit_path}")
    print(f"Approved prototypes: {len(bank.metas)}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Mosaic-SVC P0 CAMPPlus prototype bank.")
    parser.add_argument("--manifest", required=True, help="CSV with path,name,category,quality,approved columns.")
    parser.add_argument("--canonical", required=True, help="Canonical prompt audio used as S0.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--min-quality-score", type=float, default=0.75)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
