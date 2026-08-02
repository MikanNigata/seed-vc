from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import torch
import torchaudio

from modules.commons import str2bool
from mosaic_svc.p11.content_teacher import load_content_teacher
from mosaic_svc.p11.encoders import FrozenTeacherEncoders


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = FrozenTeacherEncoders(args.contentvec, args.whisper, device=device, fp16=args.fp16)
    teacher = load_content_teacher(args.teacher, device).eval()
    audio, _ = librosa.load(args.input, sr=16000, mono=True)
    chunks = []
    chunk_samples = int(args.chunk_seconds * 16000)
    for start in range(0, len(audio), chunk_samples):
        waveform = torch.from_numpy(audio[start : start + chunk_samples]).float().unsqueeze(0).to(device)
        if waveform.size(-1) < 320:
            continue
        contentvec, whisper = encoders(waveform)
        chunks.append(teacher(contentvec, whisper).cpu())
    features = torch.cat(chunks, dim=1)
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    acoustic = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
        frame_length=25.0,
        frame_shift=20.0,
        snip_edges=False,
    ).unsqueeze(0)
    length = min(features.size(1), acoustic.size(1))
    features = features[:, :length]
    acoustic = acoustic[:, :length]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source": str(Path(args.input).resolve()),
            "frame_hz": 50,
            "input_features": acoustic,
            "teacher_features": features,
        },
        output,
    )
    print(f"Saved teacher features: {output} {tuple(features.shape)}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract P11 teacher content features.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--contentvec", required=True)
    parser.add_argument("--whisper", default="openai/whisper-small")
    parser.add_argument("--chunk-seconds", type=float, default=20.0)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
