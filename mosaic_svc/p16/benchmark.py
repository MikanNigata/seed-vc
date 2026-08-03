from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import librosa
import torch

from mosaic_svc.p16.runtime import MosaicStreamingRuntime
from mosaic_svc.retired import reject_r16


def run(args):
    reject_r16()
    waveform, sample_rate = librosa.load(args.input, sr=None, mono=True, duration=args.max_seconds)
    duration = len(waveform) / sample_rate
    if duration <= 0:
        raise ValueError("benchmark input is empty")
    runtime = MosaicStreamingRuntime(
        student=args.student,
        converter=args.converter,
        ap_head=args.ap_head,
        nsf=args.nsf,
        identity_profile=args.identity_profile,
        mode=args.mode,
        device=args.device,
        prototype_bank=args.prototype_bank,
        prototype_strength=args.prototype_strength,
        prototype_max_norm_ratio=args.prototype_max_norm_ratio,
        prototype_max_gate=args.prototype_max_gate,
        refiner=args.refiner,
    )
    for _ in range(args.warmup):
        runtime.convert_waveform(waveform, sample_rate)
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
        torch.cuda.synchronize(runtime.device)
    elapsed = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        runtime.convert_waveform(waveform, sample_rate)
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
        elapsed.append(time.perf_counter() - start)
    report = {
        "mode": args.mode,
        "device": str(runtime.device),
        "input_seconds": duration,
        "repetitions": args.repetitions,
        "elapsed_seconds_mean": statistics.mean(elapsed),
        "elapsed_seconds_median": statistics.median(elapsed),
        "rtf_mean": statistics.mean(elapsed) / duration,
        "rtf_median": statistics.median(elapsed) / duration,
        "realtime_capable": statistics.median(elapsed) / duration < 1.0,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated(runtime.device) / (1024**2) if runtime.device.type == "cuda" else None
        ),
        "prototype_enabled": bool(args.prototype_bank),
        "refiner_enabled": bool(args.refiner and runtime.mode.refinement),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return output


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark Mosaic-SVC P16 end-to-end RTF and peak VRAM.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--converter", required=True)
    parser.add_argument("--ap-head", required=True)
    parser.add_argument("--nsf", required=True)
    parser.add_argument("--identity-profile", required=True)
    parser.add_argument("--prototype-bank")
    parser.add_argument("--prototype-strength", type=float, default=1.0)
    parser.add_argument("--prototype-max-norm-ratio", type=float, default=0.10)
    parser.add_argument("--prototype-max-gate", type=float, default=0.25)
    parser.add_argument("--refiner")
    parser.add_argument("--mode", choices=("live-fast", "live-quality", "render"), default="live-quality")
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
