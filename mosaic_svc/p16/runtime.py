from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

from mosaic_svc.p14.prosody import extract_prosody
from mosaic_svc.p15.ap_head import load_ap_head
from mosaic_svc.p15.nsf import load_nsf
from mosaic_svc.r16.streaming_modules import CausalContentStudent, StreamingAcousticConverter, StreamingConfig


@dataclass(frozen=True)
class RuntimeMode:
    chunk_frames: int
    left_context_frames: int
    refinement: bool = False


MODES = {
    "live-fast": RuntimeMode(4, 32, False),
    "live-quality": RuntimeMode(8, 32, True),
    "render": RuntimeMode(50, 64, True),
}


def _load_module(path, model_type, device):
    checkpoint = torch.load(path, map_location="cpu")
    config = StreamingConfig(**checkpoint["config"])
    model = model_type(config=config)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False).to(device)


class MosaicStreamingRuntime:
    def __init__(self, student, converter, ap_head, nsf, identity_profile, mode="live-fast", device=None):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mode = MODES[mode]
        self.student = _load_module(student, CausalContentStudent, self.device)
        self.converter = _load_module(converter, StreamingAcousticConverter, self.device)
        self.ap_head = load_ap_head(ap_head, self.device).eval().requires_grad_(False)
        self.nsf = load_nsf(nsf, self.device).eval().requires_grad_(False)
        profile = torch.load(identity_profile, map_location="cpu")
        self.style = profile["centroid"].float().view(1, -1).to(self.device)
        self.reset()

    def reset(self):
        self.feature_cache = None
        self.content_cache = None
        self.prosody_cache = None
        self.phase = None
        self.last_sample = None

    @torch.inference_mode()
    def process_features(self, acoustic, prosody):
        acoustic = acoustic.to(self.device)
        prosody = prosody.to(self.device)
        if self.feature_cache is not None:
            acoustic = torch.cat([self.feature_cache, acoustic], dim=1)
        cache_frames = min(self.mode.left_context_frames, acoustic.size(1))
        self.feature_cache = acoustic[:, -cache_frames:].detach()
        content = self.student(acoustic)
        current = prosody.size(1)
        content = content[:, -current:]
        converter_content = content
        converter_prosody = prosody
        if self.content_cache is not None:
            converter_content = torch.cat([self.content_cache, content], dim=1)
            converter_prosody = torch.cat([self.prosody_cache, prosody], dim=1)
        converter_cache = min(self.mode.left_context_frames, converter_content.size(1))
        self.content_cache = converter_content[:, -converter_cache:].detach()
        self.prosody_cache = converter_prosody[:, -converter_cache:].detach()
        latent = self.converter.encode(converter_content, converter_prosody, self.style)[:, -current:]
        mel = self.converter.mel(latent)
        ap = self.ap_head(latent, mel, prosody, self.style)
        waveform, self.phase = self.nsf(mel, prosody, ap, self.phase)
        if self.last_sample is not None:
            ramp_length = min(int(self.nsf.config.sample_rate * 0.005), waveform.size(-1))
            ramp = torch.linspace(0.0, 1.0, ramp_length, device=waveform.device, dtype=waveform.dtype)
            waveform[..., :ramp_length] = self.last_sample + (waveform[..., :ramp_length] - self.last_sample) * ramp
        self.last_sample = waveform[..., -1:].detach()
        return waveform

    @torch.inference_mode()
    def process_audio_chunk(self, waveform, sample_rate):
        audio16 = librosa.resample(np.asarray(waveform, dtype=np.float32), orig_sr=sample_rate, target_sr=16000)
        audio32 = librosa.resample(np.asarray(waveform, dtype=np.float32), orig_sr=sample_rate, target_sr=32000)
        if audio16.size < 400:
            audio16 = np.pad(audio16, (0, 400 - audio16.size))
        if audio32.size < 2048:
            audio32 = np.pad(audio32, (0, 2048 - audio32.size))
        acoustic = torchaudio.compliance.kaldi.fbank(
            torch.from_numpy(audio16).unsqueeze(0), num_mel_bins=80, dither=0,
            sample_frequency=16000, frame_length=25.0, frame_shift=20.0, snip_edges=False,
        ).unsqueeze(0)
        prosody = extract_prosody(audio32, 32000, 640)
        length = min(acoustic.size(1), prosody.size(1))
        acoustic, prosody = acoustic[:, :length], prosody[:, :length]
        chunks = []
        for start in range(0, length, self.mode.chunk_frames):
            stop = min(length, start + self.mode.chunk_frames)
            chunks.append(self.process_features(acoustic[:, start:stop], prosody[:, start:stop]).cpu())
        return torch.cat(chunks, dim=-1).squeeze().numpy(), self.nsf.config.sample_rate

    @torch.inference_mode()
    def convert_waveform(self, waveform, sample_rate):
        self.reset()
        output, output_rate = self.process_audio_chunk(waveform, sample_rate)
        peak = float(np.max(np.abs(output))) if output.size else 0.0
        if peak > 0.98:
            output = output * (0.98 / peak)
        return output, output_rate
