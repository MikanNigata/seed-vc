from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


@dataclass
class KVLoRAConfig:
    rank: int = 4
    alpha: float = 4.0
    dropout: float = 0.05
    max_scale: float = 0.05
    initial_scale: float = 0.01


class FusedQKVLoRA(nn.Module):
    """Keep Q frozen and add a low-rank residual to fused K/V outputs."""

    def __init__(self, base: nn.Linear, config: KVLoRAConfig):
        super().__init__()
        if base.out_features % 3:
            raise ValueError(f"expected equal fused Q/K/V dimensions, got {base.out_features}")
        self.base = base
        self.config = config
        self.q_dim = base.out_features // 3
        self.down = nn.Linear(base.in_features, config.rank, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.up = nn.Linear(config.rank, self.q_dim * 2, bias=False)
        ratio = min(max(config.initial_scale / max(config.max_scale, 1e-6), 1e-6), 1.0 - 1e-6)
        self.gate_logit = nn.Parameter(torch.logit(torch.tensor(ratio)))
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.last_delta_rms = 0.0
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def scale(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit) * self.config.max_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        kv_delta = self.up(self.dropout(self.down(x)))
        kv_delta = kv_delta * (self.config.alpha / max(1, self.config.rank)) * self.scale
        self.last_delta_rms = float(kv_delta.detach().float().square().mean().sqrt().cpu())
        q_delta = torch.zeros_like(kv_delta[..., : self.q_dim])
        return base + torch.cat([q_delta, kv_delta], dim=-1)

    def adaptation_state(self) -> dict[str, torch.Tensor]:
        return {
            "gate_logit": self.gate_logit.detach().cpu(),
            "down.weight": self.down.weight.detach().cpu(),
            "up.weight": self.up.weight.detach().cpu(),
        }

    def load_adaptation_state(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.gate_logit.copy_(state["gate_logit"])
            self.down.weight.copy_(state["down.weight"])
            self.up.weight.copy_(state["up.weight"])


def install_kv_lora(
    seed_model: Any,
    layer_indices: Iterable[int] = (8,),
    config: KVLoRAConfig | None = None,
    state_path: str | Path | None = None,
    trainable: bool = False,
) -> dict[int, FusedQKVLoRA]:
    checkpoint = torch.load(str(state_path), map_location="cpu") if state_path else None
    if checkpoint and "config" in checkpoint:
        config = KVLoRAConfig(**checkpoint["config"])
        layer_indices = [int(index) for index in checkpoint["layers"]]
    config = config or KVLoRAConfig()
    layers = seed_model.cfm.estimator.transformer.layers
    adapters: dict[int, FusedQKVLoRA] = {}
    for index in sorted(set(int(item) for item in layer_indices)):
        if not 0 <= index < len(layers):
            raise ValueError(f"layer index {index} is outside 0..{len(layers) - 1}")
        current = layers[index].attention.wqkv
        if isinstance(current, FusedQKVLoRA):
            adapter = current
        else:
            adapter = FusedQKVLoRA(current, config)
            layers[index].attention.wqkv = adapter
        if checkpoint:
            adapter.load_adaptation_state(checkpoint["layers"][str(index)])
        adapters[index] = adapter

    for module in _iter_modules(seed_model):
        for parameter in module.parameters():
            parameter.requires_grad = False
    for adapter in adapters.values():
        for parameter in adapter.down.parameters():
            parameter.requires_grad = trainable
        for parameter in adapter.up.parameters():
            parameter.requires_grad = trainable
        adapter.gate_logit.requires_grad = trainable
        adapter.to(adapter.base.weight.device)
    return adapters


def save_kv_lora(adapters: dict[int, FusedQKVLoRA], path: str | Path) -> None:
    if not adapters:
        raise ValueError("no K/V LoRA adapters to save")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first = next(iter(adapters.values()))
    torch.save(
        {
            "config": asdict(first.config),
            "layers": {str(index): adapter.adaptation_state() for index, adapter in adapters.items()},
        },
        path,
    )


def trainable_parameters(adapters: dict[int, FusedQKVLoRA]):
    for adapter in adapters.values():
        yield from (parameter for parameter in adapter.parameters() if parameter.requires_grad)


def _iter_modules(seed_model: Any):
    if isinstance(seed_model, nn.Module):
        yield seed_model
    elif hasattr(seed_model, "values"):
        for value in seed_model.values():
            if isinstance(value, nn.Module):
                yield value
