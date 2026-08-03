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


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    python = sys.executable
    out_root = Path(args.output)
    raw_dir = out_root / "raw"
    norm_dir = out_root / "lufs_norm"
    log_path = out_root / "path_ablation.log"
    out_root.mkdir(parents=True, exist_ok=True)

    conditions = [
        ("DD_prompt_dadadada_style_dadadada", args.prompt_a, args.style_a),
        ("DM_prompt_dadadada_style_maneki", args.prompt_a, args.style_b),
        ("MD_prompt_maneki_style_dadadada", args.prompt_b, args.style_a),
        ("MM_prompt_maneki_style_maneki", args.prompt_b, args.style_b),
    ]

    rows = []
    for condition, prompt, style_audio in conditions:
        cond_dir = raw_dir / condition
        cmd = [
            python,
            "-m",
            "mosaic_svc.p0.infer_p0",
            "--source",
            args.source,
            "--prompt",
            prompt,
            "--style-audio",
            style_audio,
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
        ]
        _run(cmd, log_path)
        raw = _latest_wav(cond_dir)
        norm = loudnorm(raw, norm_dir / f"{condition}.wav", args.lufs)
        rows.append(
            {
                "condition": condition,
                "prompt": prompt,
                "style_audio": style_audio,
                "raw": str(raw),
                "lufs_norm": str(norm),
            }
        )

    summary_path = out_root / "path_ablation_outputs.csv"
    _write_csv(rows, summary_path)
    _run(
        [
            python,
            "-m",
            "mosaic_svc.p0.eval_audio",
            "--reference",
            args.source,
            "--candidates",
            *[row["lufs_norm"] for row in rows],
            "--output",
            str(out_root / "path_ablation_eval.csv"),
        ],
        log_path,
    )
    print(f"Saved path ablation summary: {summary_path}")
    print(f"Saved path ablation metrics: {out_root / 'path_ablation_eval.csv'}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prompt-vs-style conditioning path ablation.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--prompt-a", required=True, help="Reference A prompt mel/semantic, usually target voice.")
    parser.add_argument("--style-a", required=True, help="Reference A CAMPPlus style audio.")
    parser.add_argument("--prompt-b", required=True, help="Reference B prompt mel/semantic, intentionally different speaker.")
    parser.add_argument("--style-b", required=True, help="Reference B CAMPPlus style audio.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--lufs", type=float, default=-20.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
