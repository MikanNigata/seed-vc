from __future__ import annotations

import torch

from mosaic_svc.p0.pitch_probe import PitchProbe, PitchProbeConfig


def test_pitch_probe_preserves_frame_count_and_outputs_classes() -> None:
    probe = PitchProbe(PitchProbeConfig(mel_bins=8, hidden_dim=16, f0_bins=32, layers=2))
    mel = torch.randn(2, 8, 41)
    assert probe(mel).shape == (2, 32, 41)
