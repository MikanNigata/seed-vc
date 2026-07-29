from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mosaic_svc.p0.quality import audit_audio


def iter_audio(root: Path):
    for ext in ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.webm"):
        yield from root.rglob(ext)


def run(args: argparse.Namespace) -> Path:
    root = Path(args.input)
    rows = []
    for path in iter_audio(root):
        try:
            q = audit_audio(path, sr=args.sr)
            row = q.to_dict()
            row["admission_pass"] = (
                q.quality_score >= args.min_quality_score
                and q.clipping_ratio <= args.max_clipping_ratio
                and q.duration >= args.min_duration
            )
            rows.append(row)
        except Exception as exc:
            rows.append({"path": str(path), "error": str(exc), "admission_pass": False})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Audited {len(rows)} files: {out}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit candidate Mosaic-SVC target audio.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--min-quality-score", type=float, default=0.75)
    parser.add_argument("--max-clipping-ratio", type=float, default=0.0005)
    parser.add_argument("--min-duration", type=float, default=3.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
