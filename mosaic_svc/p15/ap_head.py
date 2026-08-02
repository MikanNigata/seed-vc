from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class APHeadConfig:
    latent_dim: int = 384
    mel_dim: int = 128
    prosody_dim: int = 6
    style_dim: int = 192
    hidden_dim: int = 128
    bands: int = 8
    layers: int = 3
    kernel_size: int = 5


class TargetAPHead(nn.Module):
    def __init__(self, config: APHeadConfig | None = None):
        super().__init__()
        self.config = config or APHeadConfig()
        input_dim = self.config.latent_dim + self.config.mel_dim + self.config.prosody_dim + self.config.style_dim
        self.input_projection = nn.Linear(input_dim, self.config.hidden_dim)
        blocks = []
        for _ in range(self.config.layers):
            blocks.extend(
                [
                    CausalConv1d(self.config.hidden_dim, self.config.hidden_dim, self.config.kernel_size),
                    nn.SiLU(),
                ]
            )
        self.network = nn.Sequential(*blocks)
        self.output = nn.Linear(self.config.hidden_dim, self.config.bands + 2)

    def forward(self, latent, predicted_mel, prosody, style):
        style_t = style[:, None, :].expand(-1, latent.size(1), -1)
        x = torch.cat([latent, predicted_mel.detach(), prosody, style_t], dim=-1)
        x = self.input_projection(x).transpose(1, 2)
        return torch.sigmoid(self.output(self.network(x).transpose(1, 2)))


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(in_channels, out_channels, kernel_size)
        self.left_padding = kernel_size - 1

    def forward(self, x):
        return super().forward(nn.functional.pad(x, (self.left_padding, 0)))


def load_ap_head(path: str | Path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu")
    model = TargetAPHead(APHeadConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)
