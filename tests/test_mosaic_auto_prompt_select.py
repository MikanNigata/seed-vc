from __future__ import annotations

import numpy as np

from mosaic_svc.p0.auto_prompt_select import composite_score, rank_candidates, select_probe


def test_rank_candidates_prefers_large_f0_gain_with_near_equal_identity() -> None:
    rows = [
        {"name": "old", "cent_rmse": 242.0, "f0_corr": 0.94, "uv_mismatch": 0.11, "identity_similarity": 0.705},
        {"name": "new", "cent_rmse": 97.0, "f0_corr": 0.99, "uv_mismatch": 0.12, "identity_similarity": 0.698},
    ]
    ranked = rank_candidates(rows)
    assert ranked[0]["name"] == "new"
    assert composite_score(rows[1]) > composite_score(rows[0])


def test_select_probe_returns_requested_duration() -> None:
    sr = 8000
    time = np.arange(sr * 12) / sr
    audio = np.sin(2 * np.pi * 220 * time).astype(np.float32)
    probe, start = select_probe(audio, sr, seconds=5.0, hop_seconds=1.0)
    assert len(probe) == sr * 5
    assert 0.0 <= start <= 7.0


def test_composite_score_does_not_hide_a_probe_gate_failure() -> None:
    rows = [
        {"name": "bad", "cent_rmse": 440.0, "f0_corr": 0.54, "uv_mismatch": 0.09, "identity_similarity": 0.72},
    ]
    ranked = rank_candidates(rows)
    assert ranked[0]["selection_score"] > 0.0
    assert ranked[0]["cent_rmse"] > 250.0
