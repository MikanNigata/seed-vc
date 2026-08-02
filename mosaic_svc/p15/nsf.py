from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class NSFConfig:
    sample_rate: int = 32000
    hop_length: int = 640
    mel_dim: int = 128
    prosody_dim: int = 6
    ap_dim: int = 10
    hidden_dim: int = 192
    harmonics: int = 8
    blocks: int = 8


class HarmonicNoiseSource(nn.Module):
    def __init__(self, config: NSFConfig):
        super().__init__()
        self.config = config
        self.harmonic_logits = nn.Parameter(torch.linspace(0.0, -2.0, config.harmonics))

    def forward(self, prosody: torch.Tensor, ap: torch.Tensor, phase: torch.Tensor | None = None):
        log_f0, voiced = prosody[..., 0], prosody[..., 1]
        f0 = torch.where(voiced > 0.5, torch.pow(2.0, log_f0), torch.zeros_like(log_f0))
        f0 = nn.functional.interpolate(f0[:, None], scale_factor=self.config.hop_length, mode="linear", align_corners=False)[:, 0]
        voiced_samples = nn.functional.interpolate(voiced[:, None], scale_factor=self.config.hop_length, mode="nearest")[:, 0]
        increments = 2.0 * torch.pi * f0 / self.config.sample_rate
        initial = torch.zeros(f0.size(0), 1, device=f0.device, dtype=f0.dtype) if phase is None else phase[:, None]
        accumulated = torch.cumsum(increments, dim=-1) + initial
        weights = torch.softmax(self.harmonic_logits, dim=0)
        harmonic = sum(
            weights[index] * torch.sin(accumulated * float(index + 1))
            for index in range(self.config.harmonics)
        )
        harmonic = harmonic * voiced_samples
        noise_ratio = ap[..., -1]
        noise_ratio = nn.functional.interpolate(
            noise_ratio[:, None], scale_factor=self.config.hop_length, mode="linear", align_corners=False
        )[:, 0]
        # Spectral flatness overestimates AP in high-frequency voiced bands. Keep
        # the excitation harmonic-dominant while retaining noise for unvoiced frames.
        voiced_noise = noise_ratio.clamp(0.0, 1.0) * 0.15
        unvoiced_noise = noise_ratio.clamp(0.5, 1.0)
        excitation_noise = torch.where(voiced_samples > 0.5, voiced_noise, unvoiced_noise)
        noise = torch.randn_like(harmonic) * excitation_noise
        source = harmonic * (1.0 - excitation_noise) + noise
        return source[:, None], torch.remainder(accumulated[:, -1], 2.0 * torch.pi)


class ResidualFilterBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.filter = nn.Conv1d(channels, channels * 2, 3, padding=dilation, dilation=dilation)
        self.output = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        gate, value = self.filter(x).chunk(2, dim=1)
        return x + self.output(torch.tanh(value) * torch.sigmoid(gate))


class StreamingHarmonicNoiseNSF(nn.Module):
    def __init__(self, config: NSFConfig | None = None):
        super().__init__()
        self.config = config or NSFConfig()
        self.source = HarmonicNoiseSource(self.config)
        condition_dim = self.config.mel_dim + self.config.prosody_dim + self.config.ap_dim
        self.condition = nn.Conv1d(condition_dim, self.config.hidden_dim, 1)
        self.source_projection = nn.Conv1d(1, self.config.hidden_dim, 7, padding=3)
        self.blocks = nn.ModuleList(
            [ResidualFilterBlock(self.config.hidden_dim, 2 ** (index % 6)) for index in range(self.config.blocks)]
        )
        self.output = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.config.hidden_dim, self.config.hidden_dim // 2, 7, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.config.hidden_dim // 2, 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel, prosody, ap, phase: torch.Tensor | None = None):
        source, next_phase = self.source(prosody, ap, phase)
        condition = torch.cat([mel, prosody, ap], dim=-1).transpose(1, 2)
        condition = nn.functional.interpolate(condition, scale_factor=self.config.hop_length, mode="linear", align_corners=False)
        x = self.condition(condition) + self.source_projection(source)
        for block in self.blocks:
            x = block(x)
        return self.output(x), next_phase


def save_nsf(model: StreamingHarmonicNoiseNSF, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_nsf(path: str | Path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu")
    model = StreamingHarmonicNoiseNSF(NSFConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)
