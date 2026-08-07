from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from modules.length_regulator import f0_to_coarse


@dataclass(frozen=True)
class F0BlockAdapterConfig:
    f0_bins: int = 256
    rank: int = 8
    hidden_dim: int = 768
    layer_indices: tuple[int, ...] = (4, 8, 12, 16)
    alpha: float = 8.0
    dropout: float = 0.05
    initial_scale: float = 0.03
    max_scale: float = 0.15


class F0BlockAdapter(nn.Module):
    def __init__(self, config: F0BlockAdapterConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.f0_bins, config.rank)
        self.temporal = nn.Conv1d(
            config.rank, config.rank, kernel_size=3, padding=1, groups=config.rank
        )
        self.dropout = nn.Dropout(config.dropout)
        self.projections = nn.ModuleDict(
            {str(index): nn.Linear(config.rank, config.hidden_dim, bias=False) for index in config.layer_indices}
        )
        ratio = min(max(config.initial_scale / max(config.max_scale, 1e-6), 1e-6), 1.0 - 1e-6)
        self.gate_logits = nn.ParameterDict(
            {str(index): nn.Parameter(torch.logit(torch.tensor(ratio))) for index in config.layer_indices}
        )
        self.runtime_strength = 1.0
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.dirac_(self.temporal.weight)
        nn.init.zeros_(self.temporal.bias)
        for projection in self.projections.values():
            nn.init.zeros_(projection.weight)

    def scale(self, layer_index: int) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits[str(layer_index)]) * self.config.max_scale

    def features(self, f0: torch.Tensor, frames: int) -> torch.Tensor:
        coarse = f0_to_coarse(f0, self.config.f0_bins).clamp(0, self.config.f0_bins - 1)
        hidden = self.embedding(coarse.long()).transpose(1, 2)
        hidden = self.temporal(hidden)
        return F.interpolate(hidden, size=frames, mode="nearest").transpose(1, 2)

    def residual(self, features: torch.Tensor, layer_index: int) -> torch.Tensor:
        return (
            self.projections[str(layer_index)](self.dropout(features))
            * (self.config.alpha / max(1, self.config.rank))
            * self.scale(layer_index)
            * self.runtime_strength
        )


class F0BlockInjection(nn.Module):
    def __init__(self, base: nn.Module, adapter: F0BlockAdapter, layer_index: int):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.layer_index = layer_index
        self._features: torch.Tensor | None = None
        for parameter in base.parameters():
            parameter.requires_grad = False

    def set_features(self, features: torch.Tensor | None) -> None:
        if features is not None and features.ndim != 3:
            raise ValueError("F0 block features must be [B, T, rank]")
        self._features = features

    def forward(self, *args, **kwargs) -> torch.Tensor:
        output = self.base(*args, **kwargs)
        if self._features is None:
            return output
        features = self._features.to(device=output.device, dtype=output.dtype)
        if features.size(1) != output.size(1):
            features = F.interpolate(
                features.transpose(1, 2), size=output.size(1), mode="linear", align_corners=True
            ).transpose(1, 2)
        residual = self.adapter.residual(features, self.layer_index)
        if output.size(0) == features.size(0) * 2:
            residual = torch.cat([residual, torch.zeros_like(residual)], dim=0)
        elif output.size(0) != features.size(0):
            raise ValueError("F0 feature batch does not match the Transformer batch")
        return output + residual


def install_f0_block_adapter(
    seed_model: Any,
    *,
    config: F0BlockAdapterConfig | None = None,
    state_path: str | Path | None = None,
    trainable: bool = False,
    strength: float = 1.0,
) -> tuple[F0BlockAdapter, list[F0BlockInjection]]:
    if strength < 0.0:
        raise ValueError("F0 block adapter strength must be non-negative")
    checkpoint = torch.load(str(state_path), map_location="cpu") if state_path else None
    if checkpoint:
        saved = dict(checkpoint["config"])
        saved["layer_indices"] = tuple(saved["layer_indices"])
        config = F0BlockAdapterConfig(**saved)
    transformer = seed_model.cfm.estimator.transformer
    config = config or F0BlockAdapterConfig(hidden_dim=transformer.config.dim)
    adapter = F0BlockAdapter(config)
    wrappers = []
    for index in config.layer_indices:
        if index < 0 or index >= len(transformer.layers):
            raise ValueError(f"F0 block layer index {index} is outside the Transformer")
        current = transformer.layers[index]
        if isinstance(current, F0BlockInjection):
            wrapper = current
        else:
            wrapper = F0BlockInjection(current, adapter, index)
            transformer.layers[index] = wrapper
        wrappers.append(wrapper)
    if checkpoint:
        adapter.load_state_dict(checkpoint["adapter"], strict=True)
    adapter.runtime_strength = strength
    adapter.to(next(transformer.parameters()).device)
    adapter.train(mode=trainable)
    for parameter in adapter.parameters():
        parameter.requires_grad = trainable
    return adapter, wrappers


def set_f0_block_schedule(
    adapter: F0BlockAdapter,
    wrappers: list[F0BlockInjection],
    f0: torch.Tensor | None,
    frames: int | None = None,
    prompt_frames: int = 0,
) -> None:
    if f0 is None:
        for wrapper in wrappers:
            wrapper.set_features(None)
        return
    if frames is None:
        raise ValueError("frames is required when setting an F0 block schedule")
    features = adapter.features(f0, frames)
    if prompt_frames:
        features[:, :prompt_frames] = 0.0
    for wrapper in wrappers:
        wrapper.set_features(features)


def save_f0_block_adapter(adapter: F0BlockAdapter, path: str | Path, metadata: dict | None = None) -> None:
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
