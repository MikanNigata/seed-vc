from __future__ import annotations

import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, strength):
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def gradient_reverse(value: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReversal.apply(value, strength)


class SpeakerAdversarialProbe(nn.Module):
    def __init__(self, feature_dim: int, speakers: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, speakers),
        )

    def forward(self, features: torch.Tensor, strength: float = 1.0):
        pooled = torch.cat([features.mean(1), features.std(1)], dim=-1)
        return self.network(gradient_reverse(pooled, strength))


def grl_strength(progress: float, maximum: float) -> float:
    if progress < 0.10:
        return 0.0
    if progress >= 0.40:
        return maximum
    local = (progress - 0.10) / 0.30
    return maximum * (1.0 - torch.cos(torch.tensor(local * torch.pi))).item() / 2.0
