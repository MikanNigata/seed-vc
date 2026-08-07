from __future__ import annotations

import torch


def amplify_f0_condition(
    conditioned: torch.Tensor,
    unconditioned: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Apply classifier-free-style guidance to the learned F0 condition."""
    if scale < 0.0:
        raise ValueError("F0 guidance scale must be non-negative")
    if conditioned.shape != unconditioned.shape:
        raise ValueError("Conditioned and unconditioned tensors must have the same shape")
    if scale == 1.0:
        return conditioned
    return unconditioned + scale * (conditioned - unconditioned)
