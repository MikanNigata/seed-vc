from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from modules.commons import str2bool
from mosaic_svc.p0.eval_audio import f0_metrics


def composite_score(row: dict[str, float]) -> float:
    cent_score = math.exp(-max(0.0, row["cent_rmse"]) / 200.0)
    correlation = float(np.clip(row["f0_corr"], 0.0, 1.0))
    uv_score = 1.0 - float(np.clip(row["uv_mismatch"], 0.0, 1.0))
    identity = float(np.clip(row["identity_similarity"], -1.0, 1.0))
    return 0.45 * cent_score + 0.20 * correlation + 0.15 * uv_score + 0.20 * identity


def rank_candidates(rows: list[dict]) -> list[dict]:
    ranked = []
    for row in rows:
        candidate = dict(row)
        candidate["selection_score"] = composite_score(candidate)
        ranked.append(candidate)
    return sorted(ranked, key=lambda item: item["selection_score"], reverse=True)


def select_probe(audio: np.ndarray, sr: int, seconds: float, hop_seconds: float = 5.0) -> tuple[np.ndarray, float]:
    samples = min(len(audio), max(1, int(round(seconds * sr))))
    if len(audio) <= samples:
        return audio, 0.0
    f0 = librosa.yin(audio, fmin=65.0, fmax=1600.0, sr=sr, frame_length=2048, hop_length=512)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    window_frames = max(1, int(round(seconds * sr / 512)))
    hop_frames = max(1, int(round(hop_seconds * sr / 512)))
    best_score = -float("inf")
    best_frame = 0
    for start in range(0, max(1, len(f0) - window_frames + 1), hop_frames):
        stop = min(len(f0), start + window_frames)
        pitch = f0[start:stop]
        energy = rms[start:stop]
        voiced = np.isfinite(pitch) & (pitch > 65.0) & (energy > np.percentile(rms, 20))
        if voiced.sum() < max(5, int(0.15 * len(pitch))):
            continue
        log_pitch = np.log2(pitch[voiced])
        score = (
            0.45 * float(np.percentile(log_pitch, 90))
            + 0.35 * float(np.percentile(log_pitch, 90) - np.percentile(log_pitch, 10))
            + 0.20 * float(np.mean(voiced))
        )
        if score > best_score:
            best_score = score
            best_frame = start
    start_sample = min(int(best_frame * 512), len(audio) - samples)
    return audio[start_sample : start_sample + samples], start_sample / sr


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")


def _identity_scores(outputs: list[Path], profile: Path, output_dir: Path, args) -> dict[str, float]:
    manifest = output_dir / "probe_outputs.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for path in outputs:
            handle.write(json.dumps({"output_path": str(path.resolve())}) + "\n")
    scores_path = output_dir / "identity_scores.csv"
    command = [
        sys.executable,
        "-m",
        "mosaic_svc.p0.score_outputs_by_profile",
        "--manifest",
        str(manifest),
        "--profile",
        str(profile),
        "--output",
        str(scores_path),
        "--max-seconds",
        str(args.probe_seconds),
        "--fp16",
        str(args.fp16),
    ]
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    if args.config:
        command.extend(["--config", args.config])
    _run(command, output_dir / "logs" / "identity.log")
    with scores_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(Path(row["output_path"]).resolve()): float(row["identity_similarity"])
            for row in csv.DictReader(handle)
        }


def run(args: argparse.Namespace) -> Path:
    source = Path(args.source).resolve()
    profile = Path(args.profile).resolve()
    prompts = sorted(
        {Path(path).resolve() for value in args.prompt for path in glob.glob(value)}
    )
    if not prompts:
        raise ValueError("no prompt candidates matched --prompt")
    output_dir = Path(args.output).resolve()
    probe_dir = output_dir / "probe"
    generated_dir = output_dir / "generated"
    probe_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = librosa.load(source, sr=44100, mono=True)
    probe, probe_start = select_probe(audio, sr, args.probe_seconds)
    probe_path = probe_dir / "source_probe.wav"
    sf.write(probe_path, probe, sr, subtype="PCM_24")

    outputs = []
    for index, prompt in enumerate(prompts, start=1):
        candidate_dir = generated_dir / f"{index:02d}_{prompt.stem}"
        command = [
            sys.executable,
            "-m",
            "mosaic_svc.p0.infer_p0",
            "--source",
            str(probe_path),
            "--prompt",
            str(prompt),
            "--output",
            str(candidate_dir),
            "--diffusion-steps",
            str(args.probe_steps),
            "--inference-cfg-rate",
            str(args.inference_cfg_rate),
            "--seed",
            str(args.seed),
            "--fp16",
            str(args.fp16),
        ]
        if args.checkpoint:
            command.extend(["--checkpoint", args.checkpoint])
        if args.config:
            command.extend(["--config", args.config])
        _run(command, output_dir / "logs" / f"probe_{index:02d}.log")
        generated = list(candidate_dir.glob("*.wav"))
        if len(generated) != 1:
            raise RuntimeError(f"expected one generated probe in {candidate_dir}")
        outputs.append(generated[0].resolve())

    identities = _identity_scores(outputs, profile, output_dir, args)
    rows = []
    reference, _ = librosa.load(probe_path, sr=sr, mono=True)
    for prompt, output in zip(prompts, outputs):
        candidate, _ = librosa.load(output, sr=sr, mono=True)
        frames = min(len(reference), len(candidate))
        metrics = f0_metrics(reference[:frames], candidate[:frames], sr)
        rows.append(
            {
                "prompt": str(prompt),
                "probe_output": str(output),
                "identity_similarity": identities[str(output.resolve())],
                **metrics,
            }
        )
    ranked = rank_candidates(rows)
    acceptable = [
        row
        for row in ranked
        if row["cent_rmse"] <= args.max_probe_cent_rmse
        and row["f0_corr"] >= args.min_probe_f0_corr
        and row["uv_mismatch"] <= args.max_probe_uv_mismatch
    ]
    accepted = bool(acceptable)
    winner = acceptable[0] if accepted else ranked[0]
    ranking_path = output_dir / "prompt_ranking.csv"
    with ranking_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0].keys()))
        writer.writeheader()
        writer.writerows(ranked)

    winner_output = None
    if args.render_winner and accepted:
        final_dir = output_dir / "final"
        command = [
            sys.executable,
            "-m",
            "mosaic_svc.p0.infer_p0",
            "--source",
            str(source),
            "--prompt",
            winner["prompt"],
            "--output",
            str(final_dir),
            "--diffusion-steps",
            str(args.final_steps),
            "--inference-cfg-rate",
            str(args.inference_cfg_rate),
            "--seed",
            str(args.seed),
            "--fp16",
            str(args.fp16),
        ]
        if args.checkpoint:
            command.extend(["--checkpoint", args.checkpoint])
        if args.config:
            command.extend(["--config", args.config])
        _run(command, output_dir / "logs" / "final.log")
        final_outputs = list(final_dir.glob("*.wav"))
        if len(final_outputs) != 1:
            raise RuntimeError(f"expected one final output in {final_dir}")
        winner_output = str(final_outputs[0].resolve())

    summary = {
        "source": str(source),
        "probe_start_seconds": probe_start,
        "probe_seconds": len(probe) / sr,
        "winner_prompt": winner["prompt"],
        "winner_score": winner["selection_score"],
        "winner_output": winner_output,
        "accepted": accepted,
        "rejection_reason": None
        if accepted
        else (
            "no prompt passed probe gate: "
            f"cent_rmse<={args.max_probe_cent_rmse}, "
            f"f0_corr>={args.min_probe_f0_corr}, "
            f"uv_mismatch<={args.max_probe_uv_mismatch}"
        ),
        "ranking": ranked,
    }
    summary_path = output_dir / "selection.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if accepted:
        print(f"Selected prompt: {winner['prompt']}")
    else:
        print("No prompt passed the probe quality gate; final render skipped")
    print(f"Ranking: {ranking_path}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automatically select a Seed-VC prompt for the source song.")
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Prompt path or glob. Repeat for multiple patterns.",
    )
    parser.add_argument("--profile", required=True, help="Target CAMPPlus speaker profile.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-seconds", type=float, default=30.0)
    parser.add_argument("--probe-steps", type=int, default=20)
    parser.add_argument("--render-winner", type=str2bool, default=True)
    parser.add_argument("--final-steps", type=int, default=60)
    parser.add_argument("--max-probe-cent-rmse", type=float, default=250.0)
    parser.add_argument("--min-probe-f0-corr", type=float, default=0.85)
    parser.add_argument("--max-probe-uv-mismatch", type=float, default=0.20)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
