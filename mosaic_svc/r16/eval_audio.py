from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np


def rms_lufs_proxy(wav: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(wav)) + 1e-12)
    return float(20 * np.log10(rms + 1e-12) - 0.691)


def f0_metrics(reference: np.ndarray, candidate: np.ndarray, sr: int) -> dict:
    f0_ref, _, _ = librosa.pyin(reference, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    f0_cand, _, _ = librosa.pyin(candidate, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    n = min(len(f0_ref), len(f0_cand))
    f0_ref = f0_ref[:n]
    f0_cand = f0_cand[:n]
    mask = np.isfinite(f0_ref) & np.isfinite(f0_cand)
    if mask.sum() < 5:
        return {"f0_corr": np.nan, "cent_rmse": np.nan, "uv_mismatch": float(1.0 - mask.mean())}
    cents = 1200 * np.log2((f0_cand[mask] + 1e-6) / (f0_ref[mask] + 1e-6))
    return {
        "f0_corr": float(np.corrcoef(f0_ref[mask], f0_cand[mask])[0, 1]),
        "cent_rmse": float(np.sqrt(np.mean(np.square(cents)))),
        "uv_mismatch": float(np.mean(np.isfinite(f0_ref[:n]) != np.isfinite(f0_cand[:n]))),
    }


def run(args: argparse.Namespace) -> Path:
    ref, sr = librosa.load(args.reference, sr=args.sr, mono=True)
    rows = []
    for candidate_path in args.candidates:
        cand, _ = librosa.load(candidate_path, sr=sr, mono=True)
        n = min(ref.size, cand.size)
        row = {
            "candidate": candidate_path,
            "duration": n / sr,
            "lufs_proxy": rms_lufs_proxy(cand[:n]),
            "peak": float(np.max(np.abs(cand[:n]))),
        }
        row.update(f0_metrics(ref[:n], cand[:n], sr))
        rows.append(row)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved evaluation: {out}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal audio metrics for Mosaic-SVC A/B/C/D comparisons.")
    parser.add_argument("--reference", required=True, help="Source vocal for F0-retention comparison.")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sr", type=int, default=44100)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
