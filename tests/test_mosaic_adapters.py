from __future__ import annotations

import torch
from torch import nn

from mosaic_svc.p0.style_adapter import StyleAdapterConfig, StyleConditionedMerge, StyleSliceAdapter
from mosaic_svc.p5.kv_lora import FusedQKVLoRA, KVLoRAConfig


def test_style_slice_is_noop_at_zero_initialization() -> None:
    torch.manual_seed(1)
    base = nn.Linear(32, 16)
    adapter = StyleSliceAdapter(StyleAdapterConfig(input_dim=8, rank=4, output_dim=16, dropout=0.0))
    wrapped = StyleConditionedMerge(base, adapter, style_dim=8)
    x = torch.randn(2, 5, 32)

    torch.testing.assert_close(wrapped(x), base(x))


def test_fused_qkv_lora_preserves_query_slice() -> None:
    torch.manual_seed(2)
    base = nn.Linear(12, 36)
    adapter = FusedQKVLoRA(base, KVLoRAConfig(rank=4, dropout=0.0))
    nn.init.normal_(adapter.up.weight)
    x = torch.randn(2, 4, 12)

    expected = base(x)
    actual = adapter(x)
    torch.testing.assert_close(actual[..., :12], expected[..., :12])
    assert not torch.allclose(actual[..., 12:], expected[..., 12:])
