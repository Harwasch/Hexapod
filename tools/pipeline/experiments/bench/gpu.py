"""Best-effort container GPU samples; no CUDA or nvidia-smi dependency on the client."""

from __future__ import annotations

import math
import statistics
import subprocess
import threading
import time
from collections.abc import Iterable


def summarise_samples(samples: Iterable[str]) -> dict[str, float | None]:
    """Parse nvidia-smi CSV rows (percent, MiB), ignoring errors and unsupported fields.

    Bench containers have one GPU. If more are present, each valid GPU row is a sample;
    memory is the largest observed device usage, not the sum across devices.
    """
    utilisation = []
    memory = []
    for sample in samples:
        for line in sample.splitlines():
            fields = line.split(",")
            if len(fields) != 2:
                continue
            try:
                util, mem = map(float, fields)
            except ValueError:
                continue
            if not (math.isfinite(util) and math.isfinite(mem) and 0 <= util <= 100 and mem >= 0):
                continue
            utilisation.append(util)
            memory.append(mem)
    return {
        "gpuUtilMean": statistics.mean(utilisation) if utilisation else None,
        "gpuUtilP50": statistics.median(utilisation) if utilisation else None,
        "gpuMemMaxMiB": max(memory) if memory else None,
    }


class GpuSampler:
    """Sample immediately and every five seconds until stop; probe failures are harmless.

    Each probe has a two-second timeout. Event.wait makes shutdown interrupt the interval,
    and joining the thread reaps its final subprocess before evaluation starts.
    """

    def __init__(self, executable: str | None, *, interval: float = 5.0) -> None:
        self.executable = executable
        self.interval = interval
        self.samples: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.executable is None:
            return
        thread = threading.Thread(target=self._sample, name="bench-gpu", daemon=True)
        try:
            thread.start()
        except (RuntimeError, OSError):
            return
        self._thread = thread

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _sample(self) -> None:
        assert self.executable is not None
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                result = subprocess.run(  # noqa: S603 - resolved nvidia-smi, fixed argv
                    [
                        self.executable,
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2,
                )
                self.samples.append(result.stdout)
            except Exception:  # noqa: S110 - optional telemetry must not fail training
                # Telemetry must never abort training, even on a driver/probe failure.
                pass
            if self._stop.wait(max(0.0, self.interval - (time.monotonic() - started))):
                break
