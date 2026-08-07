from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class F0EmbeddingAdapterConfig:
    rank: int = 4
    alpha: float = 4.0
    dropout: float = 0.05
    initial_scale: float = 0.05
    max_scale: float = 0.20


class F0EmbeddingAdapter(nn.Module):
    """Frozen F0 embedding plus a small target-speaker residual table."""

    def __init__(self, base: nn.Embedding, config: F0EmbeddingAdapterConfig):
        super().__init__()
        self.base = base
        self.config = config
        self.down = nn.Embedding(base.num_embeddings, config.rank)
        self.dropout = nn.Dropout(config.dropout)
        self.up = nn.Linear(config.rank, base.embedding_dim, bias=False)
        ratio = min(max(config.initial_scale / max(config.max_scale, 1e-6), 1e-6), 1.0 - 1e-6)
        self.gate_logit = nn.Parameter(torch.logit(torch.tensor(ratio)))
        self.runtime_strength = 1.0
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    @property
    def scale(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit) * self.config.max_scale

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        base = self.base(indices)
        delta = self.up(self.dropout(self.down(indices)))
        delta = (
            delta
            * (self.config.alpha / max(1, self.config.rank))
            * self.scale
            * self.runtime_strength
        )
        return base + delta.to(dtype=base.dtype)


def install_f0_embedding_adapter(
    seed_model: Any,
    *,
    config: F0EmbeddingAdapterConfig | None = None,
    state_path: str | Path | None = None,
    trainable: bool = False,
    strength: float = 1.0,
) -> F0EmbeddingAdapter:
    if strength < 0.0:
        raise ValueError("F0 embedding adapter strength must be non-negative")
    checkpoint = torch.load(str(state_path), map_location="cpu") if state_path else None
    if checkpoint and "config" in checkpoint:
        config = F0EmbeddingAdapterConfig(**checkpoint["config"])
    config = config or F0EmbeddingAdapterConfig()

    current = seed_model.length_regulator.f0_embedding
    if isinstance(current, F0EmbeddingAdapter):
        adapter = current
    else:
        adapter = F0EmbeddingAdapter(current, config)
        seed_model.length_regulator.f0_embedding = adapter
    if checkpoint:
        state = checkpoint["adapter"] if "adapter" in checkpoint else checkpoint
        missing, unexpected = adapter.load_state_dict(state, strict=False)
        if unexpected or set(missing) - {"base.weight"}:
            raise ValueError(f"Invalid F0 adapter state: missing={missing}, unexpected={unexpected}")

    for module in _iter_modules(seed_model):
        for parameter in module.parameters():
            parameter.requires_grad = False
    for name, parameter in adapter.named_parameters():
        parameter.requires_grad = trainable and not name.startswith("base.")
    adapter.to(adapter.base.weight.device)
    adapter.runtime_strength = strength
    adapter.train(mode=trainable)
    return adapter


def save_f0_embedding_adapter(adapter: F0EmbeddingAdapter, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value for key, value in adapter.state_dict().items() if not key.startswith("base.")}
    torch.save({"config": asdict(adapter.config), "adapter": state}, destination)


def _iter_modules(seed_model: Any):
    if isinstance(seed_model, nn.Module):
        yield seed_model
        return
    if hasattr(seed_model, "values"):
        for value in seed_model.values():
            if isinstance(value, nn.Module):
                yield value
