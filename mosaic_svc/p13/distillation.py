from __future__ import annotations

import random

import torch

from mosaic_svc.r16.losses import delta, delta2, masked_l1


def dynamic_chunk(x: torch.Tensor, target: torch.Tensor, min_frames: int = 2, max_frames: int = 16):
    total = x.size(1)
    if total <= min_frames or random.random() < 0.2:
        return x, target
    length = random.randint(min_frames, min(max_frames, total))
    start = random.randint(0, total - length)
    return x[:, start : start + length], target[:, start : start + length]


def student_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    frame = masked_l1(student, teacher)
    velocity = masked_l1(delta(student), delta(teacher)) if student.size(1) > 1 else frame.new_zeros(())
    acceleration = masked_l1(delta2(student), delta2(teacher)) if student.size(1) > 2 else frame.new_zeros(())
    teacher_speed = torch.linalg.vector_norm(delta(teacher), dim=-1) if teacher.size(1) > 1 else None
    if teacher_speed is not None:
        threshold = torch.quantile(teacher_speed.detach(), 0.35)
        vowel_mask = teacher_speed <= threshold
        vowel = masked_l1(student[:, 1:], teacher[:, 1:], vowel_mask)
        boundary_threshold = torch.quantile(teacher_speed.detach(), 0.80)
        boundary_mask = teacher_speed >= boundary_threshold
        boundary = masked_l1(delta(student), delta(teacher), boundary_mask)
    else:
        vowel = frame.new_zeros(())
        boundary = frame.new_zeros(())
    total = frame + 0.5 * velocity + 0.1 * acceleration + 0.6 * vowel + 0.4 * boundary
    return total, {
        "frame": frame,
        "delta": velocity,
        "delta2": acceleration,
        "long_vowel": vowel,
        "boundary": boundary,
    }
