from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from modules.length_regulator import f0_to_coarse


@dataclass(frozen=True)
class F0ConditionAdapterConfig:
    f0_bins: int = 256
    rank: int = 8
    output_dim: int = 768
    alpha: float = 8.0
    dropout: float = 0.05
    initial_scale: float = 0.05
    max_scale: float = 0.20


class F0ConditionAdapter(nn.Module):
    """Inject a bounded F0 residual after Seed-VC's length regulator."""

    def __init__(self, config: F0ConditionAdapterConfig):
        super().__init__()
        self.config = config
        self.down = nn.Embedding(config.f0_bins, config.rank)
        self.temporal = nn.Conv1d(
            config.rank,
            config.rank,
            kernel_size=3,
            padding=1,
            groups=config.rank,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.up = nn.Linear(config.rank, config.output_dim, bias=False)
        ratio = min(max(config.initial_scale / max(config.max_scale, 1e-6), 1e-6), 1.0 - 1e-6)
        self.gate_logit = nn.Parameter(torch.logit(torch.tensor(ratio)))
        self.runtime_strength = 1.0
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.dirac_(self.temporal.weight)
        nn.init.zeros_(self.temporal.bias)
        nn.init.zeros_(self.up.weight)

    @property
    def scale(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit) * self.config.max_scale

    def forward(self, condition: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        if condition.size(-1) != self.config.output_dim:
            raise ValueError(
                f"Expected condition dim {self.config.output_dim}, got {condition.size(-1)}"
            )
        coarse = f0_to_coarse(f0, self.config.f0_bins).clamp(0, self.config.f0_bins - 1)
        hidden = self.down(coarse.long()).transpose(1, 2)
        hidden = self.temporal(hidden)
        hidden = F.interpolate(hidden, size=condition.size(1), mode="nearest").transpose(1, 2)
        delta = self.up(self.dropout(hidden))
        delta = (
            delta
            * (self.config.alpha / max(1, self.config.rank))
            * self.scale
            * self.runtime_strength
        )
        return condition + delta.to(dtype=condition.dtype)


def save_f0_condition_adapter(
    adapter: F0ConditionAdapter,
    path: str | Path,
    metadata: dict | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(adapter.config),
            "adapter": adapter.state_dict(),
            "metadata": metadata or {},
        },
        destination,
    )


def load_f0_condition_adapter(
    path: str | Path,
    device: torch.device,
    *,
    trainable: bool = False,
    strength: float = 1.0,
) -> F0ConditionAdapter:
    if strength < 0.0:
        raise ValueError("F0 condition adapter strength must be non-negative")
    checkpoint = torch.load(str(path), map_location="cpu")
    adapter = F0ConditionAdapter(F0ConditionAdapterConfig(**checkpoint["config"]))
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    adapter.runtime_strength = strength
    adapter.to(device)
    adapter.train(mode=trainable)
    for parameter in adapter.parameters():
        parameter.requires_grad = trainable
    return adapter
