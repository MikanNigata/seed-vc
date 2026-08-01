from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class PromptMelLoRAConfig:
    mel_dim: int = 128
    rank: int = 8
    output_dim: int = 768
    alpha: float = 8.0
    dropout: float = 0.05
    max_scale: float = 0.10
    initial_scale: float = 0.02


class PromptMelLoRA(nn.Module):
    """Low-rank update driven only by Seed-VC prompt-mel frames."""

    def __init__(self, config: PromptMelLoRAConfig):
        super().__init__()
        self.config = config
        self.down = nn.Linear(config.mel_dim, config.rank, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.up = nn.Linear(config.rank, config.output_dim, bias=False)
        ratio = min(max(config.initial_scale / max(config.max_scale, 1e-6), 1e-6), 1.0 - 1e-6)
        self.gate_logit = nn.Parameter(torch.logit(torch.tensor(ratio)))
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    @property
    def scale(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit) * self.config.max_scale

    def forward(self, prompt_mel: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.dropout(self.down(prompt_mel)))
        return delta * (self.config.alpha / max(1, self.config.rank)) * self.scale


class PromptMelConditionedMerge(nn.Module):
    def __init__(self, base: nn.Linear, adapter: PromptMelLoRA, mel_dim: int):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.mel_dim = mel_dim
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x_in)
        if x_in.size(-1) < self.mel_dim * 2:
            return base_output
        prompt_mel = x_in[..., self.mel_dim : self.mel_dim * 2]
        return base_output + self.adapter(prompt_mel)


def install_prompt_mel_lora(
    seed_model: Any,
    config: PromptMelLoRAConfig | None = None,
    state_path: str | Path | None = None,
    trainable: bool = False,
) -> PromptMelLoRA:
    estimator = seed_model.cfm.estimator
    base = estimator.cond_x_merge_linear
    if isinstance(base, PromptMelConditionedMerge):
        adapter = base.adapter
    else:
        config = config or PromptMelLoRAConfig(output_dim=base.out_features)
        config.output_dim = base.out_features
        adapter = PromptMelLoRA(config)
        estimator.cond_x_merge_linear = PromptMelConditionedMerge(base, adapter, config.mel_dim)
    if state_path:
        checkpoint = torch.load(str(state_path), map_location="cpu")
        if "config" in checkpoint:
            adapter.config = PromptMelLoRAConfig(**checkpoint["config"])
        adapter.load_state_dict(checkpoint.get("adapter", checkpoint), strict=True)
    for module in _iter_modules(seed_model):
        for parameter in module.parameters():
            parameter.requires_grad = False
    for parameter in adapter.parameters():
        parameter.requires_grad = trainable
    adapter.to(base.weight.device)
    return adapter


def save_prompt_mel_lora(adapter: PromptMelLoRA, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": adapter.config.__dict__, "adapter": adapter.state_dict()}, path)


def _iter_modules(seed_model: Any):
    if isinstance(seed_model, nn.Module):
        yield seed_model
    elif hasattr(seed_model, "values"):
        for value in seed_model.values():
            if isinstance(value, nn.Module):
                yield value
