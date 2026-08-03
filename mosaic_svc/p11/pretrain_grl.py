from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import librosa
import torch
from tqdm import tqdm

from modules.commons import str2bool
from mosaic_svc.retired import reject_r16
from mosaic_svc.p11.content_teacher import load_content_teacher, save_content_teacher
from mosaic_svc.p11.encoders import FrozenTeacherEncoders
from mosaic_svc.p11.grl import SpeakerAdversarialProbe, grl_strength
from mosaic_svc.p11.losses import detimbre_loss
from mosaic_svc.p11.perturb import timbre_perturb


def _manifest(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("path") and row.get("speaker_id")]
    labels = sorted({row["speaker_id"] for row in rows})
    if len(labels) < 2:
        raise ValueError("GRL pretraining requires external data with at least two speakers")
    return rows, {label: index for index, label in enumerate(labels)}


def _crop(path, seconds):
    audio, _ = librosa.load(path, sr=16000, mono=True)
    samples = int(seconds * 16000)
    if len(audio) > samples:
        start = random.randint(0, len(audio) - samples)
        audio = audio[start:start + samples]
    elif len(audio) < samples:
        audio = torch.nn.functional.pad(torch.from_numpy(audio), (0, samples - len(audio))).numpy()
    return torch.from_numpy(audio).float().unsqueeze(0)


def run(args):
    reject_r16()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, labels = _manifest(args.manifest)
    teacher = load_content_teacher(args.teacher, device).train()
    encoders = FrozenTeacherEncoders(args.contentvec, args.whisper, device=device, fp16=args.fp16)
    probe = SpeakerAdversarialProbe(teacher.config.input_dim, len(labels), args.probe_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(list(teacher.parameters()) + list(probe.parameters()), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for step in tqdm(range(1, args.steps + 1)):
        row = random.choice(rows)
        waveform = _crop(row["path"], args.segment_seconds).to(device)
        perturbed = timbre_perturb(waveform, 16000)
        with torch.no_grad():
            cv, whisper = encoders(waveform)
            cv_alt, whisper_alt = encoders(perturbed)
        strength = grl_strength(step / args.steps, args.max_grl)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
            fused = teacher.fusion(cv, whisper)
            content = teacher.detimbre(fused)
            content_alt = teacher(cv_alt, whisper_alt)
            retention, parts = detimbre_loss(content, content_alt, fused)
            logits = probe(content, strength)
            speaker_loss = torch.nn.functional.cross_entropy(logits, torch.tensor([labels[row["speaker_id"]]], device=device))
            loss = retention + args.speaker_weight * speaker_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(teacher.parameters()) + list(probe.parameters()), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if step % args.save_every == 0 or step == args.steps:
            save_content_teacher(teacher, output / f"content_teacher_grl_step_{step:06d}.pt")
            torch.save({"speakers": labels, "state_dict": probe.state_dict(), "feature_dim": teacher.config.input_dim}, output / f"speaker_probe_step_{step:06d}.pt")
            row_metrics = {"step": step, "loss": float(loss.detach().cpu()), "speaker_loss": float(speaker_loss.detach().cpu()), "grl_strength": strength, **{key: float(value.detach().cpu()) for key, value in parts.items()}}
            history.append(row_metrics)
            print(json.dumps(row_metrics))
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description="Pretrain De-Timbre GRL on external multi-speaker data only.")
    parser.add_argument("--manifest", required=True, help="CSV with path,speaker_id")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--contentvec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--whisper", default="openai/whisper-small")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--probe-hidden-dim", type=int, default=256)
    parser.add_argument("--max-grl", type=float, default=0.05)
    parser.add_argument("--speaker-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
