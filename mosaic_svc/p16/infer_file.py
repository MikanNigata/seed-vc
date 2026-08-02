from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import soundfile as sf

from mosaic_svc.p16.runtime import MosaicStreamingRuntime


def run(args):
    waveform, sample_rate = librosa.load(args.input, sr=None, mono=True)
    runtime = MosaicStreamingRuntime(
        args.student, args.converter, args.ap_head, args.nsf, args.identity_profile, args.mode, args.device
    )
    output_audio, output_rate = runtime.convert_waveform(waveform, sample_rate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, output_audio, output_rate, subtype="PCM_24")
    print(f"Saved {output} mode={args.mode} sr={output_rate}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description="Run the Mosaic-SVC P16 streaming stack on an audio file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--converter", required=True)
    parser.add_argument("--ap-head", required=True)
    parser.add_argument("--nsf", required=True)
    parser.add_argument("--identity-profile", required=True)
    parser.add_argument("--mode", choices=("live-fast", "live-quality", "render"), default="render")
    parser.add_argument("--device")
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
