from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from mosaic_svc.p15.ap_head import CausalConv1d


@dataclass
class RefinerConfig:
    latent_dim: int = 384
    mel_dim: int = 128
    prosody_dim: int = 6
    style_dim: int = 192
    hidden_dim: int = 192
    layers: int = 3
    kernel_size: int = 5
    max_scale: float = 0.15


class CausalAcousticRefiner(nn.Module):
    """Bounded residual mel refiner for quality/render modes."""

    def __init__(self, config: RefinerConfig | None = None):
        super().__init__()
        self.config = config or RefinerConfig()
        input_dim = self.config.latent_dim + self.config.mel_dim + self.config.prosody_dim + self.config.style_dim
        self.input_projection = nn.Linear(input_dim, self.config.hidden_dim)
        blocks = []
        for _ in range(self.config.layers):
            blocks.extend(
                [CausalConv1d(self.config.hidden_dim, self.config.hidden_dim, self.config.kernel_size), nn.SiLU()]
            )
        self.network = nn.Sequential(*blocks)
        self.output = nn.Linear(self.config.hidden_dim, self.config.mel_dim)
        self.scale_logit = nn.Parameter(torch.tensor(-4.0))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @property
    def scale(self) -> torch.Tensor:
        return torch.sigmoid(self.scale_logit) * self.config.max_scale

    def forward(self, latent, mel, prosody, style):
        style_t = style[:, None, :].expand(-1, latent.size(1), -1)
        x = torch.cat([latent, mel, prosody, style_t], dim=-1)
        x = self.input_projection(x).transpose(1, 2)
        residual = torch.tanh(self.output(self.network(x).transpose(1, 2)))
        return mel + self.scale * residual


def save_refiner(model: CausalAcousticRefiner, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_refiner(path: str | Path, device="cpu") -> CausalAcousticRefiner:
    checkpoint = torch.load(path, map_location="cpu")
    model = CausalAcousticRefiner(RefinerConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)
