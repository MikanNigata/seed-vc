from __future__ import annotations

import torch
import torch.nn.functional as F


def delta(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:] - x[:, :-1]


def delta2(x: torch.Tensor) -> torch.Tensor:
    return delta(delta(x))


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = (pred - target).abs()
    if mask is None:
        return loss.mean()
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def content_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    phoneme_logits: torch.Tensor | None = None,
    phoneme_targets: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    frame_l1: float = 1.0,
    delta_w: float = 0.5,
    delta2_w: float = 0.1,
    phoneme_w: float = 0.5,
) -> torch.Tensor:
    loss = frame_l1 * masked_l1(student, teacher, mask)
    loss = loss + delta_w * masked_l1(delta(student), delta(teacher), mask[:, 1:] if mask is not None else None)
    loss = loss + delta2_w * masked_l1(delta2(student), delta2(teacher), mask[:, 2:] if mask is not None else None)
    if phoneme_logits is not None and phoneme_targets is not None:
        ce = F.cross_entropy(phoneme_logits.transpose(1, 2), phoneme_targets, reduction="none")
        if mask is not None:
            ce = (ce * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            ce = ce.mean()
        loss = loss + phoneme_w * ce
    return loss


def grl_warmup_lambda(progress: float, lambda_max: float) -> float:
    if progress < 0.10:
        return 0.0
    if progress < 0.40:
        local = (progress - 0.10) / 0.30
        return float(lambda_max * (1.0 - torch.cos(torch.tensor(local * torch.pi))) / 2.0)
    return float(lambda_max)
