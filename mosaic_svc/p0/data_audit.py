from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mosaic_svc.p0.quality import audit_audio


def iter_audio(root: Path):
    for extension in ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.webm"):
        yield from root.rglob(extension)


def run(args: argparse.Namespace) -> Path:
    rows = []
    for path in iter_audio(Path(args.input)):
        try:
            quality = audit_audio(path, sr=args.sr)
            row = quality.to_dict()
            row["admission_pass"] = (
                quality.quality_score >= args.min_quality_score
                and quality.clipping_ratio <= args.max_clipping_ratio
                and quality.duration >= args.min_duration
            )
            rows.append(row)
        except Exception as error:
            rows.append({"path": str(path), "error": str(error), "admission_pass": False})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Audited {len(rows)} files: {output}")
    return output


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
