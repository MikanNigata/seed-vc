from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class F0GuardConfig:
    min_confidence: float = 0.80
    min_error_semitones: float = 7.0
    max_correction_semitones: float = 12.0
    min_region_seconds: float = 0.15
    fade_seconds: float = 0.05
    strength: float = 0.75
    fmin_hz: float = 65.0
    fmax_hz: float = 2100.0


def _regions(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def build_correction_curve(
    reference_f0: np.ndarray,
    candidate_f0: np.ndarray,
    reference_probability: np.ndarray,
    candidate_probability: np.ndarray,
    *,
    frame_seconds: float,
    config: F0GuardConfig,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return a bounded semitone correction and accepted frame regions."""
    n = min(
        len(reference_f0),
        len(candidate_f0),
        len(reference_probability),
        len(candidate_probability),
    )
    ref = np.asarray(reference_f0[:n], dtype=np.float64)
    cand = np.asarray(candidate_f0[:n], dtype=np.float64)
    ref_prob = np.nan_to_num(reference_probability[:n], nan=0.0)
    cand_prob = np.nan_to_num(candidate_probability[:n], nan=0.0)
    voiced = np.isfinite(ref) & np.isfinite(cand) & (ref > 0.0) & (cand > 0.0)
    confident = (ref_prob >= config.min_confidence) & (cand_prob >= config.min_confidence)

    error = np.zeros(n, dtype=np.float64)
    error[voiced] = 12.0 * np.log2(ref[voiced] / cand[voiced])
    eligible = voiced & confident & (np.abs(error) >= config.min_error_semitones)
    eligible &= np.abs(error) <= config.max_correction_semitones * 1.5

    minimum_frames = max(1, int(round(config.min_region_seconds / frame_seconds)))
    fade_frames = max(1, int(round(config.fade_seconds / frame_seconds)))
    curve = np.zeros(n, dtype=np.float64)
    accepted: list[tuple[int, int]] = []
    for start, end in _regions(eligible):
        if end - start < minimum_frames:
            continue
        segment = np.clip(
            error[start:end],
            -config.max_correction_semitones,
            config.max_correction_semitones,
        )
        if segment.size >= 3:
            kernel = min(9, segment.size if segment.size % 2 else segment.size - 1)
            segment = median_filter(segment, size=max(1, kernel), mode="nearest")
        segment *= config.strength
        edge = min(fade_frames, max(1, segment.size // 2))
        ramp = np.ones(segment.size, dtype=np.float64)
        ramp[:edge] = np.linspace(0.0, 1.0, edge, endpoint=False)
        ramp[-edge:] = np.linspace(1.0, 0.0, edge, endpoint=False)
        curve[start:end] = segment * ramp
        accepted.append((start, end))
    return curve, accepted


def _analyze_f0(audio: np.ndarray, sr: int, config: F0GuardConfig):
    hop_length = 512
    f0, voiced, probability = librosa.pyin(
        audio,
        fmin=config.fmin_hz,
        fmax=config.fmax_hz,
        sr=sr,
        hop_length=hop_length,
    )
    probability = np.where(voiced, probability, 0.0)
    return f0, probability, hop_length


def _pitch_resynthesis(
    candidate: np.ndarray,
    sr: int,
    candidate_f0: np.ndarray,
    correction: np.ndarray,
    candidate_probability: np.ndarray,
    hop_length: int,
    config: F0GuardConfig,
) -> np.ndarray:
    import parselmouth
    from parselmouth.praat import call

    sound = parselmouth.Sound(candidate, sampling_frequency=sr)
    manipulation = call(sound, "To Manipulation", 0.01, config.fmin_hz, config.fmax_hz)
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Remove points between", 0.0, sound.duration)

    target = candidate_f0 * np.power(2.0, correction / 12.0)
    times = librosa.frames_to_time(np.arange(target.size), sr=sr, hop_length=hop_length)
    keep = np.isfinite(target) & (target > 0.0) & (candidate_probability >= 0.20)
    for time_value, f0_value in zip(times[keep], target[keep]):
        if 0.0 < time_value < sound.duration:
            call(pitch_tier, "Add point", float(time_value), float(f0_value))
    call([pitch_tier, manipulation], "Replace pitch tier")
    wet = np.asarray(call(manipulation, "Get resynthesis (overlap-add)").values[0])

    n = min(candidate.size, wet.size)
    frame_blend = np.clip(
        np.abs(correction) / max(config.min_error_semitones, 1e-6),
        0.0,
        1.0,
    )
    sample_times = np.arange(n) / sr
    blend = np.interp(sample_times, times, frame_blend, left=0.0, right=0.0)
    output = candidate[:n] * (1.0 - blend) + wet[:n] * blend
    peak = np.max(np.abs(output)) + 1e-9
    original_peak = np.max(np.abs(candidate[:n])) + 1e-9
    if peak > original_peak:
        output *= original_peak / peak
    return output.astype(np.float32)


def process(reference_path: str, candidate_path: str, output_path: str, config: F0GuardConfig) -> Path:
    reference, sr = librosa.load(reference_path, sr=None, mono=True)
    candidate, _ = librosa.load(candidate_path, sr=sr, mono=True)
    n = min(reference.size, candidate.size)
    reference = reference[:n]
    candidate = candidate[:n]

    ref_f0, ref_prob, hop_length = _analyze_f0(reference, sr, config)
    cand_f0, cand_prob, _ = _analyze_f0(candidate, sr, config)
    curve, regions = build_correction_curve(
        ref_f0,
        cand_f0,
        ref_prob,
        cand_prob,
        frame_seconds=hop_length / sr,
        config=config,
    )
    if regions:
        output = _pitch_resynthesis(
            candidate,
            sr,
            cand_f0,
            curve,
            cand_prob,
            hop_length,
            config,
        )
    else:
        output = candidate.astype(np.float32)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, output, sr, subtype="PCM_24")
    report = {
        "reference": str(Path(reference_path).resolve()),
        "candidate": str(Path(candidate_path).resolve()),
        "output": str(destination.resolve()),
        "config": asdict(config),
        "analyzed_frames": int(curve.size),
        "corrected_frames": int(np.count_nonzero(np.abs(curve) > 1e-4)),
        "corrected_ratio": float(np.mean(np.abs(curve) > 1e-4)) if curve.size else 0.0,
        "mean_abs_correction_semitones": float(np.mean(np.abs(curve[np.abs(curve) > 1e-4])))
        if np.any(np.abs(curve) > 1e-4)
        else 0.0,
        "max_abs_correction_semitones": float(np.max(np.abs(curve))) if curve.size else 0.0,
        "regions": [
            {
                "start_seconds": round(start * hop_length / sr, 4),
                "end_seconds": round(end * hop_length / sr, 4),
            }
            for start, end in regions
        ],
    }
    destination.with_suffix(destination.suffix + ".f0guard.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Confidence-gated post-conversion F0 correction.")
    parser.add_argument("--reference", required=True, help="Source vocal containing the desired F0 trajectory.")
    parser.add_argument("--candidate", required=True, help="Converted vocal to correct.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--min-error-semitones", type=float, default=7.0)
    parser.add_argument("--max-correction-semitones", type=float, default=12.0)
    parser.add_argument("--min-region-seconds", type=float, default=0.15)
    parser.add_argument("--fade-seconds", type=float, default=0.05)
    parser.add_argument("--strength", type=float, default=0.75)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = F0GuardConfig(
        min_confidence=args.min_confidence,
        min_error_semitones=args.min_error_semitones,
        max_correction_semitones=args.max_correction_semitones,
        min_region_seconds=args.min_region_seconds,
        fade_seconds=args.fade_seconds,
        strength=args.strength,
    )
    output = process(args.reference, args.candidate, args.output, config)
    print(output)


if __name__ == "__main__":
    main()
