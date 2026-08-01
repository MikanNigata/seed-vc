from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

import inference as seed_inference
from modules.commons import str2bool
from mosaic_svc.p0.audio_features import extract_campplus_style, load_audio_tensor


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    seed_args = argparse.Namespace(f0_condition=True, checkpoint=args.checkpoint, config=args.config, fp16=args.fp16)
    _, _, _, _, campplus_model, _, mel_fn_args = seed_inference.load_models(seed_args)
    device = seed_inference.device
    sr = mel_fn_args["sampling_rate"]
    profile = torch.load(args.profile, map_location="cpu")
    centroid = torch.nn.functional.normalize(profile["centroid"].float().to(device).view(1, -1), dim=-1)

    rows = []
    for record in _read_jsonl(args.manifest):
        output_path = record.get("output_path")
        if not output_path or not Path(output_path).is_file():
            continue
        audio = load_audio_tensor(output_path, sr=sr, device=device, max_seconds=args.max_seconds)
        embedding = extract_campplus_style(campplus_model, audio, sr, device)
        embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
        rows.append({"output_path": str(Path(output_path).resolve()), "identity_similarity": float((embedding @ centroid.T).item())})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output_path", "identity_similarity"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved identity scores: {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score generated audio against a CAMPPlus Identity Memory profile.")
    parser.add_argument("--manifest", required=True, help="JSONL records containing output_path.")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
