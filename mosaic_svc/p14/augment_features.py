from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import torch

from mosaic_svc.retired import reject_r16

from mosaic_svc.p14.prosody import aperiodicity_targets, extract_prosody, mel_spectrogram


def run(args: argparse.Namespace) -> Path:
    reject_r16()
    item = torch.load(args.input, map_location="cpu")
    waveform, _ = librosa.load(item["source"], sr=args.sample_rate, mono=True)
    prosody = extract_prosody(waveform, args.sample_rate, args.hop_length)
    mel = mel_spectrogram(waveform, args.sample_rate, args.hop_length)
    ap = aperiodicity_targets(waveform, args.sample_rate, args.hop_length)
    teacher = item["teacher_features"]
    student_input = item["input_features"]
    length = min(teacher.size(1), student_input.size(1), prosody.size(1), mel.size(1), ap.size(1))
    item.update(
        {
            "sample_rate": args.sample_rate,
            "hop_length": args.hop_length,
            "input_features": student_input[:, :length],
            "teacher_features": teacher[:, :length],
            "prosody": prosody[:, :length],
            "target_mel": mel[:, :length],
            "target_ap": ap[:, :length],
            "waveform": torch.from_numpy(waveform[: length * args.hop_length]).float().unsqueeze(0),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(item, output)
    print(f"Saved acoustic features: {output} frames={length}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add P14 prosody, mel, and waveform targets to P11 features.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--hop-length", type=int, default=640)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
