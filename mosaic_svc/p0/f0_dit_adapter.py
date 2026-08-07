from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from modules.length_regulator import f0_to_coarse


@dataclass(frozen=True)
class F0DiTAdapterConfig:
    f0_bins: int = 256
    rank: int = 8
    hidden_dim: int = 768
    acoustic_channels: int = 128
    alpha: float = 8.0
    dropout: float = 0.05
    initial_scale: float = 0.05
    max_scale: float = 0.20


class F0DiTAdapter(nn.Module):
    def __init__(self, config: F0DiTAdapterConfig):
        super().__init__()
        self.config = config
        self.down = nn.Embedding(config.f0_bins, config.rank)
        self.temporal = nn.Conv1d(
            config.rank, config.rank, kernel_size=3, padding=1, groups=config.rank
        )
        self.dropout = nn.Dropout(config.dropout)
        self.up = nn.Linear(config.rank, config.hidden_dim, bias=False)
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

    def schedule(self, f0: torch.Tensor, frames: int) -> torch.Tensor:
        coarse = f0_to_coarse(f0, self.config.f0_bins).clamp(0, self.config.f0_bins - 1)
        hidden = self.down(coarse.long()).transpose(1, 2)
        hidden = self.temporal(hidden)
        hidden = F.interpolate(hidden, size=frames, mode="nearest").transpose(1, 2)
        delta = self.up(self.dropout(hidden))
        return (
            delta
            * (self.config.alpha / max(1, self.config.rank))
            * self.scale
            * self.runtime_strength
        )


class F0DiTMerge(nn.Module):
    def __init__(self, base: nn.Module, adapter: F0DiTAdapter):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self._schedule: torch.Tensor | None = None
        for parameter in base.parameters():
            parameter.requires_grad = False

    def set_schedule(self, schedule: torch.Tensor | None) -> None:
        if schedule is not None and (schedule.ndim != 3 or schedule.size(-1) != self.adapter.config.hidden_dim):
            raise ValueError("F0 DiT schedule must be [B, T, hidden_dim]")
        self._schedule = schedule

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        merged = self.base(x_in)
        if self._schedule is None:
            return merged
        schedule = self._schedule.to(device=merged.device, dtype=merged.dtype)
        if schedule.size(1) != merged.size(1):
            schedule = F.interpolate(
                schedule.transpose(1, 2), size=merged.size(1), mode="linear", align_corners=True
            ).transpose(1, 2)
        if schedule.size(0) == 1 and merged.size(0) > 1:
            schedule = schedule.expand(merged.size(0), -1, -1)
        conditioned = (
            x_in[..., self.adapter.config.acoustic_channels :]
            .abs()
            .sum(dim=(1, 2), keepdim=True)
            > 1e-8
        )
        return merged + torch.where(conditioned, schedule, torch.zeros_like(schedule))


def install_f0_dit_adapter(
    seed_model: Any,
    *,
    config: F0DiTAdapterConfig | None = None,
    state_path: str | Path | None = None,
    trainable: bool = False,
    strength: float = 1.0,
) -> tuple[F0DiTAdapter, F0DiTMerge]:
    if strength < 0.0:
        raise ValueError("F0 DiT adapter strength must be non-negative")
    checkpoint = torch.load(str(state_path), map_location="cpu") if state_path else None
    if checkpoint:
        config = F0DiTAdapterConfig(**checkpoint["config"])
    estimator = seed_model.cfm.estimator
    current = estimator.cond_x_merge_linear
    if isinstance(current, F0DiTMerge):
        wrapper = current
        adapter = wrapper.adapter
    else:
        config = config or F0DiTAdapterConfig(hidden_dim=current.out_features)
        adapter = F0DiTAdapter(config)
        wrapper = F0DiTMerge(current, adapter)
        estimator.cond_x_merge_linear = wrapper
    if checkpoint:
        adapter.load_state_dict(checkpoint["adapter"], strict=True)
    adapter.runtime_strength = strength
    adapter.to(next(wrapper.base.parameters()).device)
    adapter.train(mode=trainable)
    for parameter in adapter.parameters():
        parameter.requires_grad = trainable
    return adapter, wrapper


def save_f0_dit_adapter(adapter: F0DiTAdapter, path: str | Path, metadata: dict | None = None) -> None:
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
