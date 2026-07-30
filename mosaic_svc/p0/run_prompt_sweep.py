from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from mosaic_svc.p0.normalize import loudnorm


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed, see log: {log_path}")


def _latest_wav(directory: Path) -> Path:
    wavs = sorted(directory.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs:
        raise FileNotFoundError(f"no wav files in {directory}")
    return wavs[0]


def _read_manifest(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if row.get("approved", "true").lower() in {"1", "true", "yes"}]
    if not rows:
        raise ValueError("manifest has no approved prompt candidates")
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    python = sys.executable
    out_root = Path(args.output)
    raw_dir = out_root / "raw"
    norm_dir = out_root / "lufs_norm"
    log_path = out_root / "prompt_sweep.log"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, prompt in enumerate(_read_manifest(args.manifest), start=1):
        condition = f"P{idx:02d}_{Path(prompt['path']).stem}"
        cond_dir = raw_dir / condition
        _run(
            [
                python,
                "-m",
                "mosaic_svc.p0.infer_p0",
                "--source",
                args.source,
                "--prompt",
                prompt["path"],
                "--output",
                str(cond_dir),
                "--f0-condition",
                "True",
                "--fp16",
                "True",
                "--diffusion-steps",
                str(args.diffusion_steps),
                "--inference-cfg-rate",
                str(args.inference_cfg_rate),
            ],
            log_path,
        )
        raw = _latest_wav(cond_dir)
        norm = loudnorm(raw, norm_dir / f"{condition}.wav", args.lufs)
        rows.append(
            {
                "condition": condition,
                "prompt": prompt["path"],
                "prompt_quality": prompt.get("quality_score", ""),
                "raw": str(raw),
                "lufs_norm": str(norm),
            }
        )

    outputs = out_root / "prompt_sweep_outputs.csv"
    _write_csv(rows, outputs)
    _run(
        [
            python,
            "-m",
            "mosaic_svc.r16.eval_audio",
            "--reference",
            args.source,
            "--candidates",
            *[row["lufs_norm"] for row in rows],
            "--output",
            str(out_root / "prompt_sweep_eval.csv"),
        ],
        log_path,
    )
    print(f"Saved prompt sweep summary: {outputs}")
    print(f"Saved prompt sweep metrics: {out_root / 'prompt_sweep_eval.csv'}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Seed-VC prompt candidate sweep for practical reference selection.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--lufs", type=float, default=-20.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
