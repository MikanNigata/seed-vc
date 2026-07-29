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


def _write_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    root = Path(__file__).resolve().parents[2]
    python = sys.executable
    out_root = Path(args.output)
    raw_dir = out_root / "raw"
    norm_dir = out_root / "lufs_norm"
    log_path = out_root / "p0_suite.log"
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    conditions = [
        ("B", []),
    ]
    if args.style_adapter:
        conditions.append(("C", ["--style-adapter", args.style_adapter]))
    if args.prototype_bank:
        proto_args = ["--prototype-bank", args.prototype_bank]
        if args.style_adapter:
            proto_args = ["--style-adapter", args.style_adapter] + proto_args
        conditions.append(("D0", proto_args))

    # A uses upstream inference.py directly. With one available target reference, A and B are expected to be close.
    a_dir = raw_dir / "A_seed_vc"
    _run(
        [
            python,
            "inference.py",
            "--source",
            args.source,
            "--target",
            args.canonical,
            "--output",
            str(a_dir),
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
    a_wav = _latest_wav(a_dir)
    a_norm = loudnorm(a_wav, norm_dir / f"A_{a_wav.name}", args.lufs)
    results.append({"condition": "A", "raw": str(a_wav), "lufs_norm": str(a_norm)})

    for condition, extra in conditions:
        cond_dir = raw_dir / condition
        cmd = [
            python,
            "-m",
            "mosaic_svc.p0.infer_p0",
            "--source",
            args.source,
            "--prompt",
            args.canonical,
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
        ] + extra
        _run(cmd, log_path)
        wav = _latest_wav(cond_dir)
        norm = loudnorm(wav, norm_dir / f"{condition}_{wav.name}", args.lufs)
        results.append({"condition": condition, "raw": str(wav), "lufs_norm": str(norm)})

    summary_path = out_root / "p0_outputs.csv"
    _write_summary(results, summary_path)

    eval_cmd = [
        python,
        "-m",
        "mosaic_svc.r16.eval_audio",
        "--reference",
        args.source,
        "--candidates",
        *[row["lufs_norm"] for row in results],
        "--output",
        str(out_root / "p0_eval.csv"),
    ]
    _run(eval_cmd, log_path)
    print(f"Saved P0 output summary: {summary_path}")
    print(f"Saved P0 metrics: {out_root / 'p0_eval.csv'}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Mosaic-SVC P0 A/B/C/D0 comparison suite.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--style-adapter", default=None)
    parser.add_argument("--prototype-bank", default=None)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--lufs", type=float, default=-20.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
