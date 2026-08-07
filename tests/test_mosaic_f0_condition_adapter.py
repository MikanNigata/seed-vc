from __future__ import annotations

import torch

from mosaic_svc.p0.f0_condition_adapter import F0ConditionAdapter, F0ConditionAdapterConfig


def test_f0_condition_adapter_is_exact_at_zero_initialization() -> None:
    adapter = F0ConditionAdapter(F0ConditionAdapterConfig(rank=4, output_dim=16))
    condition = torch.randn(2, 30, 16)
    f0 = torch.full((2, 40), 220.0)
    torch.testing.assert_close(adapter(condition, f0), condition)


def test_f0_condition_adapter_changes_condition_after_update() -> None:
    adapter = F0ConditionAdapter(
        F0ConditionAdapterConfig(rank=4, output_dim=16, dropout=0.0)
    )
    with torch.no_grad():
        adapter.up.weight.fill_(0.5)
    condition = torch.zeros(1, 25, 16)
    low = adapter(condition, torch.full((1, 40), 110.0))
    high = adapter(condition, torch.full((1, 40), 440.0))
    assert low.shape == condition.shape
    assert not torch.equal(low, high)
