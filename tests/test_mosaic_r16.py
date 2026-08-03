from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import torch

from mosaic_svc.p11.content_teacher import ContentTeacher, ContentTeacherConfig, GatedContentFusion
from mosaic_svc.p13.distillation import student_distillation_loss
from mosaic_svc.p12.evaluate_leakage import _read_manifest
from mosaic_svc.p12.evaluate_leakage import _retention_metrics, _verification_metrics
from mosaic_svc.p14.refiner import CausalAcousticRefiner, RefinerConfig
from mosaic_svc.p15.ap_head import APHeadConfig, TargetAPHead
from mosaic_svc.p15.nsf import NSFConfig, StreamingHarmonicNoiseNSF
from mosaic_svc.r16.losses import masked_l1
from mosaic_svc.r16.streaming_modules import CausalContentStudent, StreamingAcousticConverter, StreamingConfig
from mosaic_svc.r16.style_conditioning import load_conditioned_style
from mosaic_svc.retired import RetiredPipelineError
from mosaic_svc.p0.prototype_bank import PrototypeBank, PrototypeMeta


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


def test_student_leakage_loss_prefers_uniform_probe_output():
    features = torch.zeros(1, 4, 3)
    uniform = torch.zeros(1, 3)
    confident = torch.tensor([[8.0, -4.0, -4.0]])
    uniform_loss, uniform_parts = student_distillation_loss(features, features, uniform)
    confident_loss, confident_parts = student_distillation_loss(features, features, confident)
    assert uniform_parts["speaker_leakage"] < confident_parts["speaker_leakage"]
    assert uniform_loss < confident_loss


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


def test_refiner_is_zero_init_bounded_and_causal():
    config = RefinerConfig(latent_dim=8, mel_dim=6, prosody_dim=6, style_dim=4, hidden_dim=12, layers=2, kernel_size=3)
    model = CausalAcousticRefiner(config).eval()
    latent, mel = torch.randn(1, 12, 8), torch.randn(1, 12, 6)
    prosody, style = torch.randn(1, 12, 6), torch.randn(1, 4)
    assert torch.allclose(model(latent, mel, prosody, style), mel, atol=1e-6)
    with torch.no_grad():
        model.output.weight.normal_()
    changed = latent.clone()
    changed[:, 7:] += 20.0
    original = model(latent, mel, prosody, style)
    modified = model(changed, mel, prosody, style)
    assert torch.allclose(original[:, :7], modified[:, :7], atol=1e-5)
    assert float((original - mel).abs().max()) <= config.max_scale + 1e-6


def test_prototype_conditioning_is_bounded_and_optional():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        profile_path = directory / "identity.pt"
        bank_path = directory / "bank.pt"
        canonical = torch.ones(1, 4)
        torch.save({"centroid": canonical}, profile_path)
        bank = PrototypeBank(
            torch.stack([torch.full((4,), 10.0), torch.full((4,), 8.0)]),
            [PrototypeMeta("a", "a.wav"), PrototypeMeta("b", "b.wav")],
            canonical,
        )
        bank.save(bank_path)
        unchanged = load_conditioned_style(profile_path, "cpu")
        corrected = load_conditioned_style(profile_path, "cpu", bank_path)
        assert torch.equal(unchanged, canonical)
        assert not torch.equal(corrected, canonical)
        assert float((corrected - canonical).norm()) <= float(canonical.norm() * 0.10 * 0.25) + 1e-6


def test_leakage_metrics_report_separable_speakers_and_retention():
    train_x = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    train_y = torch.tensor([0, 0, 1, 1])
    test_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    test_y = torch.tensor([0, 1])
    metrics = _verification_metrics(train_x, train_y, test_x, test_y, 2)
    assert metrics["nearest_centroid_accuracy"] == 1.0
    assert metrics["verification_eer"] == 0.0
    retention = _retention_metrics(test_x, test_x.clone())
    assert abs(retention["contentvec_cosine"] - 1.0) < 1e-6
    assert abs(retention["linear_cka"] - 1.0) < 1e-6


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


def test_r16_runtime_is_retired_before_loading_checkpoints():
    from mosaic_svc.p16.runtime import MosaicStreamingRuntime

    try:
        MosaicStreamingRuntime(None, None, None, None, None)
    except RetiredPipelineError as error:
        assert "retired" in str(error)
    else:
        raise AssertionError("R16 runtime must reject execution")
