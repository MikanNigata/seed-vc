from __future__ import annotations

import pytest
import torch

from mosaic_svc.f0_guidance import amplify_f0_condition


def test_f0_guidance_preserves_baseline_at_one() -> None:
    conditioned = torch.tensor([[[2.0, 4.0]]])
    unconditioned = torch.tensor([[[1.0, 1.0]]])
    torch.testing.assert_close(
        amplify_f0_condition(conditioned, unconditioned, 1.0),
        conditioned,
    )


def test_f0_guidance_amplifies_only_the_delta() -> None:
    conditioned = torch.tensor([[[2.0, 4.0]]])
    unconditioned = torch.tensor([[[1.0, 1.0]]])
    expected = torch.tensor([[[2.5, 5.5]]])
    torch.testing.assert_close(
        amplify_f0_condition(conditioned, unconditioned, 1.5),
        expected,
    )


def test_f0_guidance_rejects_negative_scale() -> None:
    with pytest.raises(ValueError):
        amplify_f0_condition(torch.ones(1), torch.zeros(1), -0.1)
