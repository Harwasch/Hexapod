"""Offline JSONL worker; upstream output never shares the protocol stream."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ltx25.planning import validate_request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--residency", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 1)
        os.dup2(null.fileno(), 2)

    def emit(value):
        protocol.write(json.dumps(value, separators=(",", ":")) + "\n")
        protocol.flush()

    for package in ("ltx-core", "ltx-pipelines"):
        sys.path.insert(0, str(args.source.resolve() / "packages" / package / "src"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    emit({"type": "ready", "protocolVersion": 1})
    engine = None
    load_seconds = 0.0
    try:
        for line in iter(lambda: sys.stdin.buffer.readline(131073), b""):
            try:
                if len(line) > 131072:
                    raise ValueError("Request limit")
                request = json.loads(line)
                if isinstance(request, dict) and request.get("op") == "reset":
                    if engine is not None:
                        engine.reset()
                    emit({"type": "reset", "sessionStateCleared": True})
                    continue
                validate_request(request)
                if request.get("image") and not Path(request["image"]).is_file():
                    raise ValueError("Missing conditioning image")
                if engine is None:
                    emit({"type": "phase", "phase": "loading-model"})
                    started = time.perf_counter()
                    import torch

                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA required")
                    from ltx25.engine import Engine

                    torch.cuda.reset_peak_memory_stats()
                    engine = Engine(args.source, args.weights, args.residency)
                    torch.cuda.synchronize()
                    load_seconds = time.perf_counter() - started
                emit({"type": "phase", "phase": "generating"})
                started = time.perf_counter()
                result = engine.generate(request)
                torch.cuda.synchronize()
                seconds = max(time.perf_counter() - started, 0.000001)
                result.update(
                    generationSeconds=seconds,
                    generatedFPS=result["generatedFrames"] / max(seconds, 0.001),
                    loadSeconds=load_seconds,
                    loadCount=1,
                    peakVRAMBytes=int(torch.cuda.max_memory_allocated()),
                )
                emit({"type": "result", "result": result})
            except Exception as error:  # noqa: BLE001 - keep upstream details off the wire
                emit(
                    {
                        "type": "error",
                        "code": "gpu-out-of-memory"
                        if type(error).__name__ == "OutOfMemoryError"
                        else "inference-failed",
                    }
                )
                break
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main()
