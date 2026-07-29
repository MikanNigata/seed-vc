from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def loudnorm(input_path: str | Path, output_path: str | Path, integrated_lufs: float = -20.0) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-af",
        f"loudnorm=I={integrated_lufs}:TP=-1.5:LRA=11",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize audio with ffmpeg loudnorm.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lufs", type=float, default=-20.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loudnorm(args.input, args.output, args.lufs)


if __name__ == "__main__":
    main()
