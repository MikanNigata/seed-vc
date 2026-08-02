from __future__ import annotations

import torch


def multi_resolution_stft_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.squeeze(1)
    target = target.squeeze(1)
    losses = []
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        window = torch.hann_window(n_fft, device=prediction.device, dtype=prediction.dtype)
        pred = torch.stft(prediction, n_fft, hop, n_fft, window, return_complex=True).abs().clamp_min(1e-5)
        truth = torch.stft(target, n_fft, hop, n_fft, window, return_complex=True).abs().clamp_min(1e-5)
        convergence = torch.linalg.vector_norm(pred - truth) / torch.linalg.vector_norm(truth).clamp_min(1e-6)
        log_magnitude = (pred.log() - truth.log()).abs().mean()
        losses.append(convergence + log_magnitude)
    return torch.stack(losses).mean()


def nsf_loss(prediction: torch.Tensor, target: torch.Tensor):
    waveform = (prediction - target).abs().mean()
    stft = multi_resolution_stft_loss(prediction.float(), target.float())
    return waveform + stft, {"waveform_l1": waveform, "mrstft": stft}
