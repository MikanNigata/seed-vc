from __future__ import annotations

import torch
from torch import nn

from mosaic_svc.p0.f0_dit_adapter import F0DiTAdapter, F0DiTAdapterConfig, F0DiTMerge


def test_f0_dit_merge_is_exact_at_zero_initialization() -> None:
    config = F0DiTAdapterConfig(rank=4, hidden_dim=16, acoustic_channels=4)
    adapter = F0DiTAdapter(config)
    base = nn.Linear(12, 16)
    wrapper = F0DiTMerge(base, adapter)
    values = torch.randn(1, 20, 12)
    wrapper.set_schedule(adapter.schedule(torch.full((1, 30), 220.0), 20))
    torch.testing.assert_close(wrapper(values), base(values))


def test_f0_dit_merge_preserves_cfg_null_branch() -> None:
    config = F0DiTAdapterConfig(rank=2, hidden_dim=8, acoustic_channels=2, dropout=0.0)
    adapter = F0DiTAdapter(config)
    with torch.no_grad():
        adapter.up.weight.fill_(1.0)
    base = nn.Linear(6, 8, bias=False)
    wrapper = F0DiTMerge(base, adapter)
    conditioned = torch.ones(1, 10, 6)
    null = conditioned.clone()
    null[..., 2:] = 0.0
    values = torch.cat([conditioned, null], dim=0)
    wrapper.set_schedule(adapter.schedule(torch.full((1, 12), 220.0), 10))
    output = wrapper(values)
    torch.testing.assert_close(output[1], base(null)[0])
    assert not torch.equal(output[0], base(conditioned)[0])
