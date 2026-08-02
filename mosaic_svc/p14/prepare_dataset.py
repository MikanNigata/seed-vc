from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import torch
import torchaudio
from tqdm import tqdm

from modules.commons import str2bool
from mosaic_svc.p11.content_teacher import load_content_teacher
from mosaic_svc.p11.encoders import FrozenTeacherEncoders
from mosaic_svc.p14.prosody import aperiodicity_targets, extract_prosody, mel_spectrogram


def _rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    valid = {"train", "validation", "test"}
    if not rows or any(not row.get("path") or row.get("split") not in valid for row in rows):
        raise ValueError("manifest requires path and explicit split=train|validation|test for every row")
    return rows


@torch.inference_mode()
def _teacher_features(audio16, encoders, teacher, device, chunk_seconds):
    tensors = []
    chunk_samples = int(chunk_seconds * 16000)
    for start in range(0, len(audio16), chunk_samples):
        waveform = torch.from_numpy(audio16[start:start + chunk_samples]).float().unsqueeze(0).to(device)
        if waveform.size(1) < 400:
            waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.size(1)))
        contentvec, whisper = encoders(waveform)
        tensors.append(teacher(contentvec, whisper).cpu())
    return torch.cat(tensors, dim=1)


def run(args):
    rows = _rows(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = FrozenTeacherEncoders(args.contentvec, args.whisper, device=device, fp16=args.fp16)
    teacher = load_content_teacher(args.teacher, device).eval()
    output = Path(args.output)
    feature_dir = output / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    outputs = {split: [] for split in ("train", "validation", "test")}
    for index, row in enumerate(tqdm(rows)):
        path = Path(row["path"])
        audio16, _ = librosa.load(path, sr=16000, mono=True)
        audio32, _ = librosa.load(path, sr=32000, mono=True)
        teacher_features = _teacher_features(audio16, encoders, teacher, device, args.chunk_seconds)
        acoustic = torchaudio.compliance.kaldi.fbank(
            torch.from_numpy(audio16).float().unsqueeze(0), num_mel_bins=80, dither=0,
            sample_frequency=16000, frame_length=25.0, frame_shift=20.0, snip_edges=False,
        ).unsqueeze(0)
        prosody = extract_prosody(audio32, 32000, 640)
        mel = mel_spectrogram(audio32, 32000, 640)
        ap = aperiodicity_targets(audio32, 32000, 640)
        length = min(teacher_features.size(1), acoustic.size(1), prosody.size(1), mel.size(1), ap.size(1), len(audio32) // 640)
        item = {
            "source": str(path.resolve()), "split": row["split"], "session": row.get("session", ""),
            "sample_rate": 32000, "hop_length": 640,
            "input_features": acoustic[:, :length], "teacher_features": teacher_features[:, :length],
            "prosody": prosody[:, :length], "target_mel": mel[:, :length], "target_ap": ap[:, :length],
            "waveform": torch.from_numpy(audio32[:length * 640]).float().unsqueeze(0),
        }
        feature_path = feature_dir / f"{index:05d}_{path.stem[:48]}.pt"
        torch.save(item, feature_path)
        outputs[row["split"]].append({"feature_path": str(feature_path.resolve()), "source": str(path.resolve()), "session": row.get("session", "")})
    for split, split_rows in outputs.items():
        with (output / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("feature_path", "source", "session"))
            writer.writeheader()
            writer.writerows(split_rows)
    print(f"Prepared {len(rows)} clips in {output}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare the full P13-P15 training dataset in one pass.")
    parser.add_argument("--manifest", required=True, help="CSV: path,split[,session]")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--contentvec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--whisper", default="openai/whisper-small")
    parser.add_argument("--chunk-seconds", type=float, default=20.0)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
