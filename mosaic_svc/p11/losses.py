from __future__ import annotations

import torch


def temporal_delta(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:] - x[:, :-1]


def detimbre_loss(
    original: torch.Tensor,
    perturbed: torch.Tensor,
    fused_reference: torch.Tensor,
    *,
    consistency_weight: float = 1.0,
    retention_weight: float = 0.5,
    delta_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    consistency = (original - perturbed).abs().mean()
    retention = (original - fused_reference.detach()).abs().mean()
    delta = (temporal_delta(original) - temporal_delta(fused_reference.detach())).abs().mean()
    total = consistency_weight * consistency + retention_weight * retention + delta_weight * delta
    return total, {"consistency": consistency, "retention": retention, "delta": delta}
