from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import librosa
import numpy as np


@dataclass
class AudioQuality:
    path: str
    duration: float
    rms_db: float
    peak: float
    clipping_ratio: float
    silence_ratio: float
    quality_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def audit_audio(path: str | Path, sr: int = 44100) -> AudioQuality:
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    if wav.size == 0:
        raise ValueError(f"empty audio: {path}")
    peak = float(np.max(np.abs(wav)))
    rms = float(np.sqrt(np.mean(np.square(wav)) + 1e-12))
    rms_db = float(20 * np.log10(rms + 1e-12))
    clipping_ratio = float(np.mean(np.abs(wav) >= 0.995))
    frame = max(1, int(sr * 0.05))
    usable = wav[: wav.size - (wav.size % frame)] if wav.size >= frame else wav
    if usable.size >= frame:
        frames = usable.reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
        silence_ratio = float(np.mean(frame_rms < 10 ** (-45 / 20)))
    else:
        silence_ratio = 1.0
    score = 1.0
    score -= min(0.5, clipping_ratio * 200)
    score -= min(0.2, max(0.0, peak - 0.98) * 5)
    score -= min(0.3, silence_ratio * 0.3)
    if rms_db < -35:
        score -= 0.2
    return AudioQuality(
        path=str(path),
        duration=float(wav.size / sr),
        rms_db=rms_db,
        peak=peak,
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        quality_score=float(np.clip(score, 0.0, 1.0)),
    )
