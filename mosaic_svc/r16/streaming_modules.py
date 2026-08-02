from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class StreamingConfig:
    content_dim: int = 768
    hidden_dim: int = 384
    style_dim: int = 192
    mel_dim: int = 128
    ap_bands: int = 8
    layers: int = 6
    kernel_size: int = 5


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.05):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size)
        # Per-frame normalization preserves causality; GroupNorm would leak future frames.
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.pad(x.transpose(1, 2), (self.pad, 0))
        y = self.conv(y)
        y = self.norm(y.transpose(1, 2))
        y = nn.functional.silu(y)
        return x + self.dropout(y)


class CausalContentStudent(nn.Module):
    """Small causal encoder distilled from the offline teacher content bus."""

    def __init__(self, input_dim: int = 80, config: StreamingConfig | None = None):
        super().__init__()
        self.config = config or StreamingConfig()
        self.in_proj = nn.Linear(input_dim, self.config.hidden_dim)
        self.blocks = nn.ModuleList(
            [CausalConvBlock(self.config.hidden_dim, self.config.kernel_size) for _ in range(self.config.layers)]
        )
        self.out_proj = nn.Linear(self.config.hidden_dim, self.config.content_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(features)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)


class PrototypeStyleMemory(nn.Module):
    """Level-1 memory: bounded correction to a global style vector."""

    def __init__(self, style_dim: int = 192, max_norm_ratio: float = 0.10, max_gate: float = 0.25):
        super().__init__()
        self.max_norm_ratio = max_norm_ratio
        self.max_gate = max_gate
        self.query = nn.Linear(style_dim, style_dim, bias=False)

    def forward(self, canonical: torch.Tensor, prototypes: torch.Tensor, quality: torch.Tensor | None = None) -> torch.Tensor:
        q = nn.functional.normalize(self.query(canonical), dim=-1)
        k = nn.functional.normalize(prototypes, dim=-1)
        weights = torch.softmax(q @ k.transpose(-1, -2), dim=-1)
        if quality is not None:
            weights = weights * quality
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        centroid = weights @ prototypes
        delta = centroid - canonical
        max_norm = canonical.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.max_norm_ratio
        delta = delta * torch.minimum(torch.ones_like(max_norm), max_norm / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return canonical + self.max_gate * delta


class StreamingAcousticConverter(nn.Module):
    def __init__(self, config: StreamingConfig | None = None):
        super().__init__()
        self.config = config or StreamingConfig()
        in_dim = self.config.content_dim + self.config.style_dim + 6
        self.in_proj = nn.Linear(in_dim, self.config.hidden_dim)
        self.blocks = nn.ModuleList(
            [CausalConvBlock(self.config.hidden_dim, self.config.kernel_size) for _ in range(self.config.layers)]
        )
        self.mel = nn.Linear(self.config.hidden_dim, self.config.mel_dim)
        self.ap = nn.Linear(self.config.hidden_dim, self.config.ap_bands + 2)

    def encode(self, content: torch.Tensor, prosody: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        style_t = style[:, None, :].expand(-1, content.size(1), -1)
        x = torch.cat([content, prosody, style_t], dim=-1)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, content: torch.Tensor, prosody: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encode(content, prosody, style)
        return self.mel(x), self.ap(x)


class HarmonicNoiseNSFStub(nn.Module):
    """Trainable interface placeholder for the future streaming NSF vocoder.

    This intentionally does not replace BigVGAN yet. It fixes the conditioning
    contract used by R1.6 training code: mel, F0/UV/phonation, and target AP.
    """

    def __init__(self, mel_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(mel_dim + 4, hidden_dim, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel: torch.Tensor, excitation: torch.Tensor, ap: torch.Tensor) -> torch.Tensor:
        x = torch.cat([mel, excitation, ap[..., :2]], dim=-1).transpose(1, 2)
        return self.net(x)
