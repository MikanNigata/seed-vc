from __future__ import annotations

import random

import torch
import torchaudio


@torch.no_grad()
def timbre_perturb(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Pitch-preserving timbre perturbations safe for content consistency training."""

    output = waveform.float()
    low_gain = random.uniform(-5.0, 5.0)
    high_gain = random.uniform(-6.0, 6.0)
    output = torchaudio.functional.bass_biquad(output, sr, gain=low_gain, central_freq=180.0, Q=0.7)
    output = torchaudio.functional.treble_biquad(output, sr, gain=high_gain, central_freq=3500.0, Q=0.7)
    if random.random() < 0.7:
        cutoff = random.uniform(6500.0, min(11000.0, sr * 0.45))
        output = torchaudio.functional.lowpass_biquad(output, sr, cutoff_freq=cutoff, Q=0.707)
    if random.random() < 0.5:
        coefficient = random.uniform(0.88, 0.97)
        delayed = nn_pad(output[..., :-1])
        output = output - coefficient * delayed
    output = output * random.uniform(0.75, 1.20)
    return output / output.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)


def nn_pad(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(x, (1, 0))
