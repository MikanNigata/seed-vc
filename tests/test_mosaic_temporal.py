from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from mosaic_svc.temporal.style_schedule import (
    TemporalStyleConfig,
    TemporalStyleMerge,
    build_temporal_style_schedule,
)


class CaptureBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.last_input = value
        return value


def _write_temporal_files(root: Path, confidences: list[float]) -> tuple[Path, Path]:
    memory = root / "memory"
    memory.mkdir()
    (memory / "memory.json").write_text(
        json.dumps({"schema_version": 1, "memory_type": "mosaic_temporal_timbre_memory"}),
        encoding="utf-8",
    )
    records = [
        {"patch_id": "patch_a", "audio_path": "patch_a.wav"},
        {"patch_id": "patch_b", "audio_path": "patch_b.wav"},
    ]
    (memory / "memory.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    query = root / "query.jsonl"
    rows = []
    for index, confidence in enumerate(confidences):
        rows.append(
            {
                "frame_index": index,
                "source_time_seconds": index * 0.1,
                "selected_patch_id": "patch_a",
                "retrieval_confidence": confidence,
                "candidates": [
                    {"patch_id": "patch_a", "soft_weight": 0.75},
                    {"patch_id": "patch_b", "soft_weight": 0.25},
                ],
            }
        )
    query.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return query, memory


def test_temporal_merge_preserves_cfg_null_style() -> None:
    base = CaptureBase()
    wrapped = TemporalStyleMerge(base, style_dim=2)
    wrapped.set_schedule(torch.tensor([[[2.0, 3.0], [4.0, 5.0]]]))
    inputs = torch.tensor(
        [
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        ]
    )
    output = wrapped(inputs)

    torch.testing.assert_close(output[0, :, -2:], torch.tensor([[2.0, 3.0], [4.0, 5.0]]))
    torch.testing.assert_close(output[1, :, -2:], torch.zeros(2, 2))


def test_temporal_schedule_is_bounded_and_resampled(tmp_path: Path) -> None:
    query, memory = _write_temporal_files(tmp_path, [0.8, 0.9, 0.7])
    canonical = torch.ones(1, 4)
    embeddings = {
        "patch_a": torch.tensor([[10.0, -8.0, 6.0, -4.0]]),
        "patch_b": torch.tensor([[-9.0, 7.0, -5.0, 3.0]]),
    }
    schedule, summary = build_temporal_style_schedule(
        query,
        memory,
        canonical,
        lambda record: embeddings[record["patch_id"]],
        source_frames=17,
        config=TemporalStyleConfig(
            style_dim=4,
            max_gate=0.25,
            max_norm_ratio=0.10,
            min_confidence=0.45,
            smoothing_seconds=0.2,
        ),
    )

    assert schedule.shape == (1, 17, 4)
    assert torch.isfinite(schedule).all()
    ratios = (schedule - canonical[:, None, :]).norm(dim=-1) / canonical.norm(dim=-1, keepdim=True)
    assert float(ratios.max()) <= 0.02501
    assert summary["active_query_frames"] == 3
    assert summary["unique_patches_embedded"] == 2


def test_low_confidence_schedule_is_canonical(tmp_path: Path) -> None:
    query, memory = _write_temporal_files(tmp_path, [0.1, 0.2])
    canonical = torch.arange(4, dtype=torch.float32).unsqueeze(0)
    schedule, summary = build_temporal_style_schedule(
        query,
        memory,
        canonical,
        lambda record: torch.full((1, 4), 100.0),
        source_frames=8,
        config=TemporalStyleConfig(style_dim=4, min_confidence=0.5),
    )

    torch.testing.assert_close(schedule, canonical[:, None, :].expand(-1, 8, -1))
    assert summary["active_query_frames"] == 0
    assert summary["mean_gate"] == 0.0


def test_f0_gate_falls_back_to_canonical(tmp_path: Path) -> None:
    query, memory = _write_temporal_files(tmp_path, [0.9])
    row = json.loads(query.read_text(encoding="utf-8"))
    row["source_features"] = {
        "f0_valid": True,
        "f0_confidence": 0.20,
        "relative_register": 0.20,
        "voiced_ratio": 0.90,
    }
    row["candidates"][0]["target_features"] = {
        "f0_valid": True,
        "f0_confidence": 0.90,
        "relative_register": 0.22,
        "voiced_ratio": 0.92,
    }
    query.write_text(json.dumps(row) + "\n", encoding="utf-8")
    canonical = torch.arange(4, dtype=torch.float32).unsqueeze(0)
    schedule, summary = build_temporal_style_schedule(
        query,
        memory,
        canonical,
        lambda record: torch.full((1, 4), 100.0),
        source_frames=8,
        config=TemporalStyleConfig(style_dim=4),
    )

    torch.testing.assert_close(schedule, canonical[:, None, :].expand(-1, 8, -1))
    assert summary["active_query_frames"] == 0
    assert summary["gate_rejections"]["source_f0_confidence"] == 1
