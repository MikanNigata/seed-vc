from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class PitchProbeConfig:
    mel_bins: int = 128
    hidden_dim: int = 128
    f0_bins: int = 256
    kernel_size: int = 5
    layers: int = 3
    dropout: float = 0.05


class PitchProbe(nn.Module):
    def __init__(self, config: PitchProbeConfig):
        super().__init__()
        padding = config.kernel_size // 2
        blocks: list[nn.Module] = [
            nn.Conv1d(config.mel_bins, config.hidden_dim, config.kernel_size, padding=padding),
            nn.GELU(),
        ]
        for _ in range(config.layers - 1):
            blocks.extend(
                [
                    nn.Conv1d(
                        config.hidden_dim,
                        config.hidden_dim,
                        config.kernel_size,
                        padding=padding,
                    ),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
        self.backbone = nn.Sequential(*blocks)
        self.classifier = nn.Conv1d(config.hidden_dim, config.f0_bins, 1)
        self.config = config

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(mel))


def save_pitch_probe(probe: PitchProbe, path: str | Path, metadata: dict | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(probe.config),
            "probe": probe.state_dict(),
            "metadata": metadata or {},
        },
        destination,
    )


def load_pitch_probe(path: str | Path, device: torch.device, *, trainable: bool = False) -> PitchProbe:
    checkpoint = torch.load(str(path), map_location="cpu")
    probe = PitchProbe(PitchProbeConfig(**checkpoint["config"]))
    probe.load_state_dict(checkpoint["probe"], strict=True)
    probe.to(device)
    probe.train(mode=trainable)
    for parameter in probe.parameters():
        parameter.requires_grad = trainable
    return probe
