from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class StyleAdapterConfig:
    input_dim: int = 192
    rank: int = 4
    output_dim: int = 768
    alpha: float = 4.0
    dropout: float = 0.10
    max_scale: float = 0.20
    initial_scale: float = 0.05


class StyleSliceAdapter(nn.Module):
    """Low-rank residual branch that only sees the CAMPPlus style vector."""

    def __init__(self, config: StyleAdapterConfig):
        super().__init__()
        self.config = config
        self.down = nn.Linear(config.input_dim, config.rank, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.up = nn.Linear(config.rank, config.output_dim, bias=False)
        ratio = min(max(config.initial_scale / max(config.max_scale, 1e-6), 1e-6), 1.0 - 1e-6)
        self.gate_logit = nn.Parameter(torch.logit(torch.tensor(ratio)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    @property
    def scale(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit) * self.config.max_scale

    def forward(self, style: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.dropout(self.down(style)))
        return delta * (self.config.alpha / max(1, self.config.rank)) * self.scale


class StyleConditionedMerge(nn.Module):
    """Wrap Seed-VC's cond_x_merge_linear without changing its public interface."""

    def __init__(self, base: nn.Linear, adapter: StyleSliceAdapter, style_dim: int):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.style_dim = style_dim
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x_in)
        if x_in.size(-1) < self.style_dim:
            return base_out
        style = x_in[:, 0, -self.style_dim :]
        delta = self.adapter(style).unsqueeze(1)
        return base_out + delta


def install_style_slice_adapter(
    seed_model: Any,
    config: StyleAdapterConfig | None = None,
    state_path: str | Path | None = None,
    trainable: bool = False,
) -> StyleSliceAdapter:
    """Install the P0 style-only residual branch on a loaded Seed-VC model."""

    checkpoint = torch.load(str(state_path), map_location="cpu") if state_path else None
    if checkpoint and "config" in checkpoint:
        config = StyleAdapterConfig(**checkpoint["config"])
    estimator = seed_model.cfm.estimator
    base = estimator.cond_x_merge_linear
    if isinstance(base, StyleConditionedMerge):
        adapter = base.adapter
    else:
        config = config or StyleAdapterConfig(output_dim=base.out_features)
        adapter = StyleSliceAdapter(config)
        estimator.cond_x_merge_linear = StyleConditionedMerge(base, adapter, config.input_dim)

    if checkpoint:
        adapter.load_state_dict(checkpoint["adapter"] if "adapter" in checkpoint else checkpoint, strict=True)

    for module in _iter_modules(seed_model):
        for param in module.parameters():
            param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = trainable

    adapter.to(base.weight.device)
    return adapter


def save_style_adapter(adapter: StyleSliceAdapter, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": adapter.config.__dict__, "adapter": adapter.state_dict()}, path)


def _iter_modules(seed_model: Any):
    if isinstance(seed_model, nn.Module):
        yield seed_model
        return
    if hasattr(seed_model, "values"):
        for value in seed_model.values():
            if isinstance(value, nn.Module):
                yield value
