from __future__ import annotations

from pathlib import Path

import librosa
import torch
import torchaudio


def load_audio_tensor(path: str | Path, sr: int, device: torch.device, max_seconds: float | None = None) -> torch.Tensor:
    wav = librosa.load(str(path), sr=sr)[0]
    if max_seconds is not None:
        wav = wav[: int(sr * max_seconds)]
    return torch.tensor(wav).unsqueeze(0).float().to(device)


@torch.no_grad()
def extract_campplus_style(campplus_model, audio: torch.Tensor, sr: int, device: torch.device) -> torch.Tensor:
    waves_16k = torchaudio.functional.resample(audio, sr, 16000)
    feat = torchaudio.compliance.kaldi.fbank(
        waves_16k,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    return campplus_model(feat.unsqueeze(0)).float().to(device)
