from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import soundfile as sf

from mosaic_svc.p0.quality import audit_audio


def run(args: argparse.Namespace) -> Path:
    wav, sr = librosa.load(args.input, sr=args.sr, mono=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    length = int(args.seconds * sr)
    hop = int(args.hop_seconds * sr)
    starts = list(range(0, max(1, len(wav) - length + 1), hop))
    for idx, start in enumerate(starts[: args.max_candidates], start=1):
        clip = wav[start : start + length]
        if clip.size < int(args.min_seconds * sr):
            continue
        out_path = out_dir / f"prompt_{idx:02d}_{start / sr:06.2f}s.wav"
        sf.write(out_path, clip, sr)
        quality = audit_audio(out_path, sr=sr)
        rows.append(
            {
                "path": str(out_path),
                "name": out_path.stem,
                "start_seconds": start / sr,
                "duration": quality.duration,
                "quality_score": quality.quality_score,
                "peak": quality.peak,
                "rms_db": quality.rms_db,
                "silence_ratio": quality.silence_ratio,
                "approved": quality.quality_score >= args.min_quality_score,
            }
        )

    manifest = out_dir / "prompt_candidates.csv"
    if rows:
        with open(manifest, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {len(rows)} prompt candidates: {manifest}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cut target audio into prompt candidates for Seed-VC prompt sweep.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--hop-seconds", type=float, default=10.0)
    parser.add_argument("--min-seconds", type=float, default=8.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--min-quality-score", type=float, default=0.55)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
