"""Offline survey RSS probe; fixture generation runs outside the measured process.

From tools/pipeline:
  heavy uv run python tests/density_memory_probe.py --points 1000000 --observations 30

Uses TMPDIR for the synthetic COLMAP model and deletes it afterwards. Reports the
full gb.plan survey, not just footprint reduction, in a fresh process. No GPU/network.
"""

from __future__ import annotations

import argparse
import json
import resource
import struct
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gaussian_budget as gb


def make(directory: Path, points: int, observations: int) -> None:
    """Write consistent fixed-track synthetic data in bounded batches."""
    with (directory / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<QIiQQdddd", 1, 1, 1, 1080, 1920, 1000, 1000, 540, 960))
    with (directory / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", observations))
        for image in range(observations):
            handle.write(struct.pack("<IdddddddI", image + 1, 1, 0, 0, 0, 0, 0, image + 2, 1))
            handle.write(f"frame_{image:03d}.jpg".encode() + b"\0")
            handle.write(struct.pack("<Q", points))
            for start in range(0, points, 4096):
                data = np.zeros((min(4096, points - start), 3), dtype="<u8")
                data[:, 2] = np.arange(start + 1, start + 1 + len(data), dtype=np.uint64)
                handle.write(data.tobytes())
    dtype = np.dtype(
        [
            ("id", "<u8"),
            ("xyz", "<f8", (3,)),
            ("rgb", "u1", (3,)),
            ("error", "<f8"),
            ("length", "<u8"),
            ("track", "<u4", (observations, 2)),
        ]
    )
    with (directory / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", points))
        for start in range(0, points, 4096):
            count = min(4096, points - start)
            rows = np.arange(start, start + count)
            point_data = np.zeros(count, dtype=dtype)
            point_data["id"] = rows + 1
            point_data["xyz"][:, 0] = (rows % 1000) * 0.02
            point_data["xyz"][:, 1] = (rows // 1000) * 0.02
            point_data["rgb"] = 128
            point_data["error"] = 0.5
            point_data["length"] = observations
            point_data["track"][:, :, 0] = np.arange(1, observations + 1)
            point_data["track"][:, :, 1] = rows[:, None]
            handle.write(point_data.tobytes())


def measure(directory: Path, trace: bool) -> None:
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if trace:
        tracemalloc.start()
    started = time.monotonic()
    budget = gb.plan("auto", model=directory, train_size=(1080, 1920), pixel_scale=1.0)
    assert budget is not None and budget.mode == "auto"
    allocated = tracemalloc.get_traced_memory()[1] / 1024**2 if trace else None
    if trace:
        tracemalloc.stop()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result = {
        "baselineRssMiB": round(baseline, 2),
        "peakRssMiB": round(peak, 2),
        "tracemallocPeakMiB": None if allocated is None else round(allocated, 2),
        "seconds": round(time.monotonic() - started, 2),
        "surface": budget.inside.to_dict(),
        "densityTarget": budget.raw,
    }
    sys.stdout.write(json.dumps(result) + "\n")
    assert peak < 600, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=1_000_000)
    parser.add_argument("--observations", type=int, default=30)
    parser.add_argument("--measure", type=Path)
    parser.add_argument("--trace", action="store_true", help="also track allocations (slower)")
    args = parser.parse_args()
    if args.measure:
        measure(args.measure, args.trace)
        return
    with tempfile.TemporaryDirectory(prefix="density-survey-") as tmp:
        directory = Path(tmp)
        make(directory, args.points, args.observations)
        argv = [sys.executable, __file__, "--measure", str(directory)]
        if args.trace:
            argv.append("--trace")
        subprocess.run(argv, check=True, timeout=1800)


if __name__ == "__main__":
    main()
