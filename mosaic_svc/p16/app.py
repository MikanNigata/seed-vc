from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gradio as gr
import librosa
import soundfile as sf

from mosaic_svc.p16.runtime import MosaicStreamingRuntime


def build_app(args):
    runtimes = {}

    def convert(audio_path, mode):
        if not audio_path:
            raise gr.Error("Input audio is required.")
        if mode not in runtimes:
            runtimes[mode] = MosaicStreamingRuntime(
                args.student, args.converter, args.ap_head, args.nsf, args.identity_profile, mode, args.device
            )
        waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        output, output_rate = runtimes[mode].convert_waveform(waveform, sample_rate)
        path = Path(tempfile.mkdtemp(prefix="mosaic_svc_")) / "converted.wav"
        sf.write(path, output, output_rate, subtype="PCM_24")
        return str(path)

    with gr.Blocks(title="Mosaic-SVC R1.6") as app:
        gr.Markdown("# Mosaic-SVC R1.6\nStreaming Student + AP Head + harmonic-noise NSF")
        source = gr.Audio(type="filepath", label="Source singing")
        mode = gr.Radio(("live-fast", "live-quality", "render"), value="render", label="Mode")
        button = gr.Button("Convert", variant="primary")
        output = gr.Audio(type="filepath", label="Converted audio")
        button.click(convert, (source, mode), output)
    return app


def build_parser():
    parser = argparse.ArgumentParser(description="Launch the Mosaic-SVC P16 GUI.")
    parser.add_argument("--student", required=True)
    parser.add_argument("--converter", required=True)
    parser.add_argument("--ap-head", required=True)
    parser.add_argument("--nsf", required=True)
    parser.add_argument("--identity-profile", required=True)
    parser.add_argument("--device")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    return parser


def main():
    args = build_parser().parse_args()
    build_app(args).launch(server_name=args.host, server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
