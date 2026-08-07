from __future__ import annotations

import numpy as np

from mosaic_svc.f0_guard import F0GuardConfig, build_correction_curve


def test_f0_guard_rejects_short_and_low_confidence_errors() -> None:
    reference = np.full(40, 220.0)
    candidate = reference.copy()
    candidate[5:8] = 110.0
    candidate[15:30] = 110.0
    confidence = np.ones(40)
    confidence[15:30] = 0.4
    curve, regions = build_correction_curve(
        reference,
        candidate,
        np.ones(40),
        confidence,
        frame_seconds=0.01,
        config=F0GuardConfig(min_region_seconds=0.1),
    )
    assert regions == []
    assert np.count_nonzero(curve) == 0


def test_f0_guard_accepts_and_bounds_sustained_octave_error() -> None:
    reference = np.full(50, 440.0)
    candidate = reference.copy()
    candidate[10:40] = 220.0
    curve, regions = build_correction_curve(
        reference,
        candidate,
        np.ones(50),
        np.ones(50),
        frame_seconds=0.01,
        config=F0GuardConfig(
            min_region_seconds=0.1,
            fade_seconds=0.03,
            max_correction_semitones=12.0,
            strength=0.85,
        ),
    )
    assert regions == [(10, 40)]
    assert np.max(curve) <= 10.2 + 1e-6
    assert np.max(curve) > 10.0
    assert curve[10] == 0.0
    assert curve[25] > 10.0
