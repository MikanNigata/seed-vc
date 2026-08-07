from __future__ import annotations

import torch
from torch import nn

from mosaic_svc.p0.f0_block_adapter import (
    F0BlockAdapter,
    F0BlockAdapterConfig,
    F0BlockInjection,
)


class IdentityBlock(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def test_f0_block_injection_is_exact_at_zero_initialization() -> None:
    config = F0BlockAdapterConfig(rank=4, hidden_dim=16, layer_indices=(0,))
    adapter = F0BlockAdapter(config)
    wrapper = F0BlockInjection(IdentityBlock(), adapter, 0)
    values = torch.randn(1, 20, 16)
    wrapper.set_features(adapter.features(torch.full((1, 30), 220.0), 20))
    torch.testing.assert_close(wrapper(values), values)


def test_f0_block_injection_zeros_cfg_null_half() -> None:
    config = F0BlockAdapterConfig(
        rank=2, hidden_dim=8, layer_indices=(0,), dropout=0.0
    )
    adapter = F0BlockAdapter(config)
    with torch.no_grad():
        adapter.projections["0"].weight.fill_(1.0)
    wrapper = F0BlockInjection(IdentityBlock(), adapter, 0)
    values = torch.randn(2, 10, 8)
    wrapper.set_features(adapter.features(torch.full((1, 12), 220.0), 10))
    output = wrapper(values)
    torch.testing.assert_close(output[1], values[1])
    assert not torch.equal(output[0], values[0])
