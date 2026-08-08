from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class F0ConsensusConfig:
    min_anchor_probability: float = 0.80
    octave_tolerance_semitones: float = 2.0
    min_region_seconds: float = 0.05
    max_octaves: int = 2
    fmin_hz: float = 65.0
    fmax_hz: float = 1600.0
    frame_length: int = 2048
    hop_length: int = 512

    def validate(self) -> None:
        if not 0.0 <= self.min_anchor_probability <= 1.0:
            raise ValueError("min_anchor_probability must be between 0 and 1")
        if self.octave_tolerance_semitones < 0.0:
            raise ValueError("octave_tolerance_semitones must be non-negative")
        if self.min_region_seconds < 0.0:
            raise ValueError("min_region_seconds must be non-negative")
        if self.max_octaves <= 0:
            raise ValueError("max_octaves must be positive")


def _regions(values: np.ndarray) -> list[tuple[int, int, int]]:
    regions: list[tuple[int, int, int]] = []
    start = 0
    while start < values.size:
        step = int(values[start])
        end = start + 1
        while end < values.size and int(values[end]) == step:
            end += 1
        if step != 0:
            regions.append((start, end, step))
        start = end
    return regions


def build_octave_correction(
    rmvpe_f0: np.ndarray,
    anchor_f0: np.ndarray,
    anchor_probability: np.ndarray,
    *,
    frame_seconds: float,
    config: F0ConsensusConfig | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Return conservative integer-octave corrections for an RMVPE track."""
    config = config or F0ConsensusConfig()
    config.validate()
    n = min(len(rmvpe_f0), len(anchor_f0), len(anchor_probability))
    rmvpe = np.asarray(rmvpe_f0[:n], dtype=np.float64)
    anchor = np.asarray(anchor_f0[:n], dtype=np.float64)
    probability = np.nan_to_num(anchor_probability[:n], nan=0.0)
    valid = (
        np.isfinite(rmvpe)
        & np.isfinite(anchor)
        & (rmvpe > 0.0)
        & (anchor > 0.0)
        & (probability >= config.min_anchor_probability)
    )

    delta = np.zeros(n, dtype=np.float64)
    delta[valid] = 12.0 * np.log2(anchor[valid] / rmvpe[valid])
    rounded = np.rint(delta / 12.0).astype(np.int8)
    eligible = valid & (rounded != 0) & (np.abs(rounded) <= config.max_octaves)
    eligible &= np.abs(delta - rounded * 12.0) <= config.octave_tolerance_semitones
    proposed = np.where(eligible, rounded, 0).astype(np.int8)

    minimum_frames = max(1, int(round(config.min_region_seconds / max(frame_seconds, 1e-6))))
    correction = np.zeros(n, dtype=np.int8)
    accepted: list[tuple[int, int, int]] = []
    for start, end, step in _regions(proposed):
        if end - start < minimum_frames:
            continue
        correction[start:end] = step
        accepted.append((start, end, step))
    return correction, accepted


def lock_rmvpe_with_pyin(
    rmvpe_f0: np.ndarray,
    audio: np.ndarray,
    sr: int,
    config: F0ConsensusConfig | None = None,
) -> tuple[np.ndarray, dict]:
    config = config or F0ConsensusConfig()
    config.validate()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    rmvpe = np.asarray(rmvpe_f0, dtype=np.float64).reshape(-1)
    if audio.size == 0 or rmvpe.size == 0:
        raise ValueError("audio and rmvpe_f0 must not be empty")

    anchor, voiced, probability = librosa.pyin(
        audio,
        fmin=config.fmin_hz,
        fmax=config.fmax_hz,
        sr=sr,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
    )
    probability = np.where(voiced, probability, 0.0)
    anchor_times = librosa.frames_to_time(
        np.arange(anchor.size), sr=sr, hop_length=config.hop_length
    )
    duration = audio.size / sr
    rmvpe_times = np.arange(rmvpe.size) * duration / rmvpe.size
    right = np.searchsorted(anchor_times, rmvpe_times, side="left")
    right = np.clip(right, 0, anchor.size - 1)
    left = np.clip(right - 1, 0, anchor.size - 1)
    use_left = np.abs(anchor_times[left] - rmvpe_times) <= np.abs(
        anchor_times[right] - rmvpe_times
    )
    nearest = np.where(use_left, left, right)
    aligned_anchor = anchor[nearest]
    aligned_probability = probability[nearest]

    frame_seconds = duration / rmvpe.size
    correction, regions = build_octave_correction(
        rmvpe,
        aligned_anchor,
        aligned_probability,
        frame_seconds=frame_seconds,
        config=config,
    )
    corrected = rmvpe * np.power(2.0, correction.astype(np.float64))
    report = {
        "schema_version": 1,
        "config": asdict(config),
        "frames": int(rmvpe.size),
        "corrected_frames": int(np.count_nonzero(correction)),
        "corrected_ratio": float(np.mean(correction != 0)),
        "regions": [
            {
                "start_seconds": round(start * frame_seconds, 4),
                "end_seconds": round(end * frame_seconds, 4),
                "octave_shift": step,
            }
            for start, end, step in regions
        ],
    }
    return corrected.astype(np.float32), report


def save_f0_consensus_report(report: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
