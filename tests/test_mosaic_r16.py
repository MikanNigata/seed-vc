from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import torch

from mosaic_svc.p11.content_teacher import ContentTeacher, ContentTeacherConfig, GatedContentFusion
from mosaic_svc.p12.evaluate_leakage import _read_manifest
from mosaic_svc.p15.ap_head import APHeadConfig, TargetAPHead
from mosaic_svc.p15.nsf import NSFConfig, StreamingHarmonicNoiseNSF
from mosaic_svc.r16.losses import masked_l1
from mosaic_svc.r16.streaming_modules import CausalContentStudent, StreamingAcousticConverter, StreamingConfig


def test_content_fusion_gate_is_bounded():
    model = GatedContentFusion(ContentTeacherConfig(input_dim=8, max_whisper_gate=0.3))
    output = model(torch.randn(2, 10, 8), torch.randn(2, 5, 8))
    assert output.shape == (2, 10, 8)
    assert 0.0 <= float(model.whisper_gate) <= 0.3


def test_detimbre_zero_initialization_is_no_harm():
    config = ContentTeacherConfig(input_dim=8, hidden_dim=8, bottleneck_dim=4, layers=1, heads=2)
    model = ContentTeacher(config).eval()
    contentvec, whisper = torch.randn(1, 10, 8), torch.randn(1, 5, 8)
    fused = model.fusion(contentvec, whisper)
    assert torch.allclose(model(contentvec, whisper), fused, atol=1e-6)


def test_single_speaker_leakage_manifest_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("path", "speaker_id"))
            writer.writeheader()
            writer.writerows(({"path": "a.wav", "speaker_id": "one"}, {"path": "b.wav", "speaker_id": "one"}))
        try:
            _read_manifest(path)
        except ValueError as error:
            assert "at least two" in str(error)
        else:
            raise AssertionError("single-speaker leakage evaluation must fail")


def test_masked_l1_averages_feature_dimensions():
    prediction = torch.ones(1, 2, 4)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False]])
    assert torch.isclose(masked_l1(prediction, target, mask), torch.tensor(1.0))


def test_student_and_converter_are_causal():
    config = StreamingConfig(content_dim=8, hidden_dim=12, style_dim=4, mel_dim=6, layers=2, kernel_size=3)
    student = CausalContentStudent(input_dim=5, config=config).eval()
    converter = StreamingAcousticConverter(config).eval()
    source = torch.randn(1, 12, 5)
    changed = source.clone()
    changed[:, 7:] += 20.0
    assert torch.allclose(student(source)[:, :7], student(changed)[:, :7], atol=1e-5)
    content, prosody, style = torch.randn(1, 12, 8), torch.randn(1, 12, 6), torch.randn(1, 4)
    changed_content = content.clone()
    changed_content[:, 7:] += 20.0
    assert torch.allclose(converter(content, prosody, style)[0][:, :7], converter(changed_content, prosody, style)[0][:, :7], atol=1e-5)


def test_ap_head_is_causal():
    head = TargetAPHead(APHeadConfig(latent_dim=8, mel_dim=6, prosody_dim=6, style_dim=4, hidden_dim=12, layers=2, kernel_size=3)).eval()
    latent, mel, prosody, style = torch.randn(1, 12, 8), torch.randn(1, 12, 6), torch.randn(1, 12, 6), torch.randn(1, 4)
    changed = latent.clone()
    changed[:, 7:] += 20.0
    assert torch.allclose(head(latent, mel, prosody, style)[:, :7], head(changed, mel, prosody, style)[:, :7], atol=1e-5)


def test_nsf_output_length_and_phase_state():
    config = NSFConfig(hop_length=16, mel_dim=6, prosody_dim=6, ap_dim=4, hidden_dim=8, harmonics=2, blocks=1)
    model = StreamingHarmonicNoiseNSF(config).eval()
    mel, prosody, ap = torch.randn(1, 5, 6), torch.zeros(1, 5, 6), torch.zeros(1, 5, 4)
    prosody[..., 0] = torch.log2(torch.tensor(220.0))
    prosody[..., 1] = 1.0
    audio, phase = model(mel, prosody, ap)
    continued, next_phase = model(mel, prosody, ap, phase)
    assert audio.shape == continued.shape == (1, 1, 80)
    assert phase.shape == next_phase.shape == (1,)
    assert torch.isfinite(audio).all() and torch.isfinite(next_phase).all()
