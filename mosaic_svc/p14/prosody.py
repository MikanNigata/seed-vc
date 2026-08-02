from __future__ import annotations

import librosa
import numpy as np
import torch


def extract_prosody(waveform: np.ndarray, sr: int = 32000, hop_length: int = 640) -> torch.Tensor:
    f0, voiced, probability = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )
    energy = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=2048, hop_length=hop_length)[0]
    length = min(len(f0), len(energy), len(zcr))
    f0 = f0[:length]
    voiced = voiced[:length].astype(np.float32)
    probability = np.nan_to_num(probability[:length], nan=0.0)
    log_f0 = np.log2(np.nan_to_num(f0, nan=1.0)).astype(np.float32)
    log_f0 = np.where(voiced > 0, log_f0, 0.0)
    slope = np.diff(log_f0, prepend=log_f0[:1])
    log_energy = np.log(np.maximum(energy[:length], 1e-6))
    phonation = np.clip(1.0 - zcr[:length] * 8.0, 0.0, 1.0)
    features = np.stack([log_f0, voiced, probability, slope, log_energy, phonation], axis=-1)
    return torch.from_numpy(features).float().unsqueeze(0)


def mel_spectrogram(waveform: np.ndarray, sr: int = 32000, hop_length: int = 640) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=2048,
        hop_length=hop_length,
        win_length=2048,
        n_mels=128,
        fmin=20,
        fmax=sr / 2,
        power=1.0,
    )
    return torch.from_numpy(np.log(np.maximum(mel, 1e-5))).float().transpose(0, 1).unsqueeze(0)


def aperiodicity_targets(waveform: np.ndarray, sr: int = 32000, hop_length: int = 640, bands: int = 8) -> torch.Tensor:
    magnitude = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=hop_length, win_length=2048)).T
    edges = np.linspace(0, magnitude.shape[1], bands + 1, dtype=int)
    flatness = []
    for low, high in zip(edges[:-1], edges[1:]):
        band = np.maximum(magnitude[:, low:high], 1e-7)
        geometric = np.exp(np.mean(np.log(band), axis=1))
        arithmetic = np.mean(band, axis=1)
        flatness.append(np.clip(geometric / np.maximum(arithmetic, 1e-7), 0.0, 1.0))
    band_ap = np.stack(flatness, axis=-1).astype(np.float32)
    noise_ratio = band_ap.mean(axis=-1, keepdims=True)
    harmonicity = 1.0 - noise_ratio
    return torch.from_numpy(np.concatenate([band_ap, harmonicity, noise_ratio], axis=-1)).float().unsqueeze(0)
