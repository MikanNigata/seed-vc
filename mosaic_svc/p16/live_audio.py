from __future__ import annotations

import argparse
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from mosaic_svc.p16.runtime import MODES, MosaicStreamingRuntime
from mosaic_svc.retired import reject_r16


def run(args):
    reject_r16()
    sample_rate = 32000
    mode = MODES[args.mode]
    blocksize = mode.chunk_frames * 640
    runtime = MosaicStreamingRuntime(
        student=args.student,
        converter=args.converter,
        ap_head=args.ap_head,
        nsf=args.nsf,
        identity_profile=args.identity_profile,
        mode=args.mode,
        device=args.device,
        prototype_bank=args.prototype_bank,
        prototype_strength=args.prototype_strength,
        prototype_max_norm_ratio=args.prototype_max_norm_ratio,
        prototype_max_gate=args.prototype_max_gate,
        refiner=args.refiner,
    )
    inputs: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
    outputs: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
    stopped = threading.Event()
    underruns = 0

    def input_callback(indata, frames, timing, status):
        del frames, timing
        if status:
            print(f"input status: {status}")
        try:
            inputs.put_nowait(indata[:, 0].copy())
        except queue.Full:
            print("input queue overrun; dropping one block")

    def output_callback(outdata, frames, timing, status):
        nonlocal underruns
        del timing
        if status:
            print(f"output status: {status}")
        try:
            block = outputs.get_nowait()
        except queue.Empty:
            underruns += 1
            outdata.fill(0)
            return
        if len(block) < frames:
            block = np.pad(block, (0, frames - len(block)))
        outdata[:, 0] = block[:frames]

    def worker():
        while not stopped.is_set():
            try:
                block = inputs.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                converted, _ = runtime.process_audio_chunk(block, sample_rate)
                if len(converted) < blocksize:
                    converted = np.pad(converted, (0, blocksize - len(converted)))
                outputs.put(converted[:blocksize], timeout=1.0)
            except Exception as error:
                print(f"conversion error: {type(error).__name__}: {error}")
                outputs.put(np.zeros(blocksize, dtype=np.float32))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print(f"Mosaic-SVC live mode={args.mode} block={blocksize / sample_rate * 1000:.0f}ms. Press Ctrl+C to stop.")
    try:
        with sd.InputStream(device=args.input_device, channels=1, samplerate=sample_rate, blocksize=blocksize, callback=input_callback), sd.OutputStream(device=args.output_device, channels=1, samplerate=sample_rate, blocksize=blocksize, callback=output_callback):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        thread.join(timeout=2.0)
        print(f"Stopped. output underruns={underruns}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run Mosaic-SVC with live microphone input.")
    parser.add_argument("--student")
    parser.add_argument("--converter")
    parser.add_argument("--ap-head")
    parser.add_argument("--nsf")
    parser.add_argument("--identity-profile")
    parser.add_argument("--prototype-bank")
    parser.add_argument("--prototype-strength", type=float, default=1.0)
    parser.add_argument("--prototype-max-norm-ratio", type=float, default=0.10)
    parser.add_argument("--prototype-max-gate", type=float, default=0.25)
    parser.add_argument("--refiner", help="Optional bounded P14 refiner; used only in live-quality mode")
    parser.add_argument("--mode", choices=("live-fast", "live-quality"), default="live-quality")
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    parser.add_argument("--device")
    parser.add_argument("--list-devices", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    missing = [name for name in ("student", "converter", "ap_head", "nsf", "identity_profile") if not getattr(args, name)]
    if missing:
        parser.error("required for live conversion: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    run(args)


if __name__ == "__main__":
    main()
