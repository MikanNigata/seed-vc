from __future__ import annotations

import numpy as np

from mosaic_svc.p0.f0_consensus import F0ConsensusConfig, build_octave_correction


def test_sustained_high_confidence_octave_error_is_corrected() -> None:
    rmvpe = np.full(20, 220.0)
    anchor = np.full(20, 440.0)
    probability = np.ones(20)
    correction, regions = build_octave_correction(
        rmvpe,
        anchor,
        probability,
        frame_seconds=0.01,
        config=F0ConsensusConfig(min_region_seconds=0.05),
    )
    np.testing.assert_array_equal(correction, np.ones(20, dtype=np.int8))
    assert regions == [(0, 20, 1)]


def test_short_or_low_confidence_mismatch_is_rejected() -> None:
    rmvpe = np.full(20, 220.0)
    anchor = rmvpe.copy()
    anchor[3:6] = 440.0
    anchor[10:20] = 440.0
    probability = np.ones(20)
    probability[10:20] = 0.4
    correction, regions = build_octave_correction(
        rmvpe,
        anchor,
        probability,
        frame_seconds=0.01,
        config=F0ConsensusConfig(min_region_seconds=0.05, min_anchor_probability=0.8),
    )
    assert not correction.any()
    assert regions == []


def test_non_octave_pitch_difference_is_preserved() -> None:
    rmvpe = np.full(20, 220.0)
    anchor = np.full(20, 330.0)
    correction, _ = build_octave_correction(
        rmvpe,
        anchor,
        np.ones(20),
        frame_seconds=0.01,
    )
    assert not correction.any()
