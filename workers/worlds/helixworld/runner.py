"""Invoke the pinned offline CLI, then select its actual audio-visual output."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def completed_clip(directory):
    directory = Path(directory).resolve()
    marker = directory / "release/native/INFERENCE_COMPLETE.json"
    if not marker.is_file() or marker.stat().st_size > 1024 * 1024:
        raise RuntimeError("HelixWorld did not produce its completion record")
    record = json.loads(marker.read_text())
    outputs = record.get("outputs", {})
    if outputs.get("audio_muxed") is not True:
        raise RuntimeError("HelixWorld did not produce its required generated audio")
    raw = outputs.get("clean_av_mp4")
    if not isinstance(raw, str):
        raise RuntimeError("HelixWorld did not produce a clean audio-visual clip")
    path = Path(raw).resolve()
    if not path.is_relative_to(directory) or not path.is_file() or not 1 <= path.stat().st_size <= 256 * 1024 * 1024:
        raise RuntimeError("HelixWorld returned an invalid clip path or size")
    return path


def main():
    parser = argparse.ArgumentParser()
    for flag in ("source", "output", "image", "prompts"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--action", choices=("W", "S", "A", "D", "left", "right", "up", "down", "stop"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    command = [sys.executable, str(args.source / "infer.py"), "--image", str(args.image), "--prompt-file", str(args.prompts),
               "--actions", args.action, "--perspective", "first_person", "--num-frames", "121", "--output-dir", str(args.output),
               "--seed", str(args.seed), "--attention-backend", "sdpa_flash"]
    # Inherit the gateway's process group so cancellation reaches every child.
    subprocess.run(command, cwd=args.source, check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.copyfile(completed_clip(args.output), args.output / "chunk.mp4")


if __name__ == "__main__":
    main()
