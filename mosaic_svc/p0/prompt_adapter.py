from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class PromptAdapterConfig:
    mel_dim: int = 128
    cond_dim: int = 768
    style_dim: int = 192
    rank: int = 8
    output_dim: int = 768
    alpha: float = 8.0
    dropout: float = 0.05
    max_scale: float = 0.20
    initial_scale: float = 0.03
    prompt_eps: float = 1e-5
    source_only: bool = True


class PromptSliceAdapter(nn.Module):
    """Low-rank residual branch that only sees prompt mel, prompt condition, and CAMPPlus style."""

    def __init__(self, config: PromptAdapterConfig):
        super().__init__()
        self.config = config
        input_dim = config.mel_dim + config.cond_dim + config.style_dim
        self.down = nn.Linear(input_dim, config.rank, bias=False)
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

    def forward(self, prompt_mel: torch.Tensor, prompt_cond: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        features = torch.cat([prompt_mel, prompt_cond, style], dim=-1)
        delta = self.up(self.dropout(self.down(features)))
        return delta * (self.config.alpha / max(1, self.config.rank)) * self.scale


class PromptConditionedMerge(nn.Module):
    """Wrap Seed-VC's cond_x_merge_linear with a prompt-aware residual branch."""

    def __init__(self, base: nn.Module, adapter: PromptSliceAdapter, config: PromptAdapterConfig):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.config = config
        self.strength = 1.0
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x_in)
        cfg = self.config
        min_dim = cfg.mel_dim * 2 + cfg.cond_dim
        if x_in.size(-1) < min_dim:
            return base_out

        prompt_mel = x_in[..., cfg.mel_dim : cfg.mel_dim * 2]
        prompt_cond = x_in[..., cfg.mel_dim * 2 : cfg.mel_dim * 2 + cfg.cond_dim]
        if x_in.size(-1) >= min_dim + cfg.style_dim:
            style = x_in[:, 0, -cfg.style_dim :]
        else:
            style = x_in.new_zeros(x_in.size(0), cfg.style_dim)

        prompt_mask = prompt_mel.abs().mean(dim=-1, keepdim=True).gt(cfg.prompt_eps).to(prompt_mel.dtype)
        denom = prompt_mask.sum(dim=1).clamp_min(1.0)
        prompt_mel_summary = (prompt_mel * prompt_mask).sum(dim=1) / denom
        prompt_cond_summary = (prompt_cond * prompt_mask).sum(dim=1) / denom
        delta = self.adapter(prompt_mel_summary, prompt_cond_summary, style).unsqueeze(1) * self.strength

        if cfg.source_only:
            delta = delta * (1.0 - prompt_mask)
        return base_out + delta


def install_prompt_adapter(
    seed_model: Any,
    config: PromptAdapterConfig | None = None,
    state_path: str | Path | None = None,
    trainable: bool = False,
    strength: float = 1.0,
) -> PromptSliceAdapter:
    """Install the M2 prompt-aware residual branch on a loaded Seed-VC model."""

    estimator = seed_model.cfm.estimator
    base = estimator.cond_x_merge_linear
    if isinstance(base, PromptConditionedMerge):
        adapter = base.adapter
        wrapper = base
    else:
        output_dim = _output_dim(base)
        config = config or PromptAdapterConfig(output_dim=output_dim)
        if config.output_dim != output_dim:
            config.output_dim = output_dim
        adapter = PromptSliceAdapter(config)
        wrapper = PromptConditionedMerge(base, adapter, config)
        estimator.cond_x_merge_linear = wrapper
    wrapper.strength = strength

    if state_path:
        checkpoint = torch.load(str(state_path), map_location="cpu")
        if "config" in checkpoint:
            saved = PromptAdapterConfig(**checkpoint["config"])
            wrapper.config = saved
            adapter.config = saved
        adapter.load_state_dict(checkpoint["adapter"] if "adapter" in checkpoint else checkpoint, strict=True)

    for module in _iter_modules(seed_model):
        for param in module.parameters():
            param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = trainable

    adapter.to(_module_device(wrapper.base))
    return adapter


def save_prompt_adapter(adapter: PromptSliceAdapter, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": adapter.config.__dict__, "adapter": adapter.state_dict()}, path)


def _output_dim(module: nn.Module) -> int:
    if hasattr(module, "out_features"):
        return int(module.out_features)
    if hasattr(module, "base"):
        return _output_dim(module.base)
    raise TypeError(f"cannot infer output dim from {module.__class__.__name__}")


def _module_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    return torch.device("cpu")


def _iter_modules(seed_model: Any):
    if isinstance(seed_model, nn.Module):
        yield seed_model
        return
    if hasattr(seed_model, "values"):
        for value in seed_model.values():
            if isinstance(value, nn.Module):
                yield value
