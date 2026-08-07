from __future__ import annotations

import torch
from torch import nn

from mosaic_svc.p0.f0_embedding_adapter import F0EmbeddingAdapter, F0EmbeddingAdapterConfig


def test_f0_embedding_adapter_is_exact_at_zero_initialization() -> None:
    base = nn.Embedding(16, 8)
    adapter = F0EmbeddingAdapter(base, F0EmbeddingAdapterConfig(rank=2))
    indices = torch.tensor([[1, 3, 7]])
    torch.testing.assert_close(adapter(indices), base(indices))


def test_f0_embedding_adapter_changes_only_selected_indices_after_update() -> None:
    base = nn.Embedding(8, 4)
    adapter = F0EmbeddingAdapter(base, F0EmbeddingAdapterConfig(rank=2, dropout=0.0))
    with torch.no_grad():
        adapter.up.weight.fill_(0.5)
    first = adapter(torch.tensor([1]))
    second = adapter(torch.tensor([2]))
    assert not torch.equal(first - base(torch.tensor([1])), second - base(torch.tensor([2])))
    assert not base.weight.requires_grad


def test_f0_embedding_adapter_runtime_strength_scales_delta() -> None:
    base = nn.Embedding(8, 4)
    adapter = F0EmbeddingAdapter(base, F0EmbeddingAdapterConfig(rank=2, dropout=0.0))
    with torch.no_grad():
        adapter.up.weight.fill_(0.5)
    indices = torch.tensor([3])
    adapter.runtime_strength = 1.0
    delta_one = adapter(indices) - base(indices)
    adapter.runtime_strength = 10.0
    delta_ten = adapter(indices) - base(indices)
    torch.testing.assert_close(delta_ten, delta_one * 10.0)
