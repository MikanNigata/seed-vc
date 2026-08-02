from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import librosa
import torch
from torch import nn

from modules.commons import str2bool
from mosaic_svc.p11.content_teacher import load_content_teacher
from mosaic_svc.p11.encoders import FrozenTeacherEncoders


def _pool(features: torch.Tensor) -> torch.Tensor:
    return torch.cat([features.mean(dim=1), features.std(dim=1)], dim=-1)


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("path") and row.get("speaker_id")]
    speakers = Counter(row["speaker_id"] for row in rows)
    if len(speakers) < 2:
        raise ValueError("speaker leakage evaluation requires at least two speaker_id values")
    if min(speakers.values()) < 2:
        raise ValueError("each speaker requires at least two clips")
    return rows


def _split(rows: list[dict[str, str]], seed: int):
    random.seed(seed)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["speaker_id"], []).append(row)
    train, test = [], []
    for items in grouped.values():
        random.shuffle(items)
        count = max(1, int(len(items) * 0.8))
        count = min(count, len(items) - 1)
        train.extend(items[:count])
        test.extend(items[count:])
    return train, test


@torch.no_grad()
def _extract(rows, encoders, teacher, device, seconds: float):
    labels = sorted({row["speaker_id"] for row in rows})
    label_map = {label: index for index, label in enumerate(labels)}
    stages = {name: [] for name in ("contentvec", "whisper", "fusion", "detimbre")}
    targets = []
    for row in rows:
        audio, _ = librosa.load(row["path"], sr=16000, mono=True, duration=seconds)
        waveform = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        contentvec, whisper = encoders(waveform)
        if whisper.size(1) != contentvec.size(1):
            whisper_aligned = nn.functional.interpolate(
                whisper.transpose(1, 2), size=contentvec.size(1), mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            whisper_aligned = whisper
        fusion = teacher.fusion(contentvec, whisper)
        output = teacher.detimbre(fusion)
        for name, features in {
            "contentvec": contentvec,
            "whisper": whisper_aligned,
            "fusion": fusion,
            "detimbre": output,
        }.items():
            stages[name].append(_pool(features).cpu())
        targets.append(label_map[row["speaker_id"]])
    return {name: torch.cat(values) for name, values in stages.items()}, torch.tensor(targets), label_map


class Probe(nn.Module):
    def __init__(self, input_dim: int, speakers: int, nonlinear: bool):
        super().__init__()
        self.network = (
            nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, speakers))
            if nonlinear
            else nn.Linear(input_dim, speakers)
        )

    def forward(self, x):
        return self.network(x)


def _fit_probe(train_x, train_y, test_x, test_y, speakers, nonlinear, epochs, device):
    mean = train_x.mean(0, keepdim=True)
    std = train_x.std(0, keepdim=True).clamp_min(1e-5)
    train_x = ((train_x - mean) / std).to(device)
    test_x = ((test_x - mean) / std).to(device)
    train_y, test_y = train_y.to(device), test_y.to(device)
    model = Probe(train_x.size(1), speakers, nonlinear).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return float((model(test_x).argmax(-1) == test_y).float().mean().cpu())


def run(args: argparse.Namespace) -> Path:
    rows = _read_manifest(args.manifest)
    train_rows, test_rows = _split(rows, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = FrozenTeacherEncoders(args.contentvec, args.whisper, device=device, fp16=args.fp16)
    teacher = load_content_teacher(args.teacher, device).eval()
    train_stages, train_y, labels = _extract(train_rows, encoders, teacher, device, args.clip_seconds)
    test_stages, test_y, _ = _extract(test_rows, encoders, teacher, device, args.clip_seconds)
    report = {
        "speakers": len(labels),
        "train_clips": len(train_rows),
        "test_clips": len(test_rows),
        "chance_accuracy": 1.0 / len(labels),
        "stages": {},
    }
    for name in train_stages:
        report["stages"][name] = {
            "linear_accuracy": _fit_probe(
                train_stages[name], train_y, test_stages[name], test_y, len(labels), False, args.epochs, device
            ),
            "mlp_accuracy": _fit_probe(
                train_stages[name], train_y, test_stages[name], test_y, len(labels), True, args.epochs, device
            ),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate speaker leakage at each P11 content stage.")
    parser.add_argument("--manifest", required=True, help="CSV containing path and speaker_id")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--contentvec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--whisper", default="openai/whisper-small")
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
