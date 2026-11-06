"""Selection equivalence and a subprocess RSS gate independent of the pytest heap."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from live_sample_reference import snapshot_splat as reference

from captures_bridge import ply_layout
from live_sample import snapshot_splat


def write_fixture(path: Path, count: int, *, tied: bool = False) -> None:
    names = ["x", "y", "z", "opacity"]
    names += [f"scale_{i}" for i in range(3)] + [f"f_dc_{i}" for i in range(3)]
    names += [f"rot_{i}" for i in range(4)] + [f"f_rest_{i}" for i in range(45)]
    rng = np.random.default_rng(123)
    with path.open("wb") as handle:
        handle.write(
            (
                f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n"
                + "".join(f"property float {name}\n" for name in names)
                + "end_header\n"
            ).encode()
        )
        for start in range(0, count, 16_384):
            rows = np.zeros(min(16_384, count - start), dtype=[(k, "<f4") for k in names])
            for name in ("x", "y", "z"):
                rows[name] = (
                    rng.integers(-20, 20, len(rows)) if tied else rng.normal(0, 4, len(rows))
                )
            rows["opacity"] = 3 if tied else rng.uniform(-5, 5, len(rows))
            for i in range(3):
                rows[f"scale_{i}"] = -3 if tied else rng.uniform(-7, 1, len(rows))
                rows[f"f_dc_{i}"] = rng.normal(size=len(rows))
            rows["rot_0"] = 1
            rows.tofile(handle)


@pytest.mark.parametrize("tied", [False, True])
def test_windowed_thumbnail_is_byte_identical(tmp_path: Path, tied: bool) -> None:
    ply = tmp_path / "medium.ply"
    write_fixture(ply, 150_003, tied=tied)
    # Invalid rows and a very dense cell exercise filtering and rank cutoffs.
    layout = ply_layout(ply)
    rows = np.memmap(ply, dtype=layout.dtype, offset=layout.offset, mode="r+")
    rows["x"][:17] = np.nan
    rows["y"][17:30] = 3000
    rows["scale_1"][30:40] = np.inf
    del rows
    cameras = None if tied else np.array([[0.0, 0.0, 0.0], [2.0, -1.0, 3.0]])
    for cap in (113, 10_000, 200_000):
        expected, actual = tmp_path / "old.spz", tmp_path / "new.spz"
        with np.errstate(invalid="ignore"):
            reference(ply, expected, max_gaussians=cap, camera_centres=cameras, focal_px=500.0)
            snapshot_splat(ply, actual, max_gaussians=cap, camera_centres=cameras, focal_px=500.0)
        assert actual.read_bytes() == expected.read_bytes()


def test_thumbnail_peak_memory_is_bounded(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "test_live_sample", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join(
                (str(Path(__file__).parent), str(Path(__file__).parent.parent))
            ),
        },
    )
    measured = json.loads(result.stdout)
    # One million SH3 rows are 236 MB on disk. The old reader alone copies all of
    # those columns twice; this bound includes interpreter, imports and SPZ packing.
    assert measured["peakMiB"] < 256, measured
    assert measured["kept"] == 100_000


if __name__ == "__main__":
    from memory_probe import peak_mb

    directory = Path(sys.argv[1])
    ply = directory / "million.ply"
    write_fixture(ply, 1_000_000)
    kept, _, _ = snapshot_splat(
        ply, directory / "out.spz", camera_centres=np.zeros((8, 3)), focal_px=500.0
    )
    # VmHWM belongs to this address space; Linux ru_maxrss can inherit pytest's
    # earlier peak across exec and falsely attribute the entire suite to this probe.
    sys.stdout.write(json.dumps({"kept": kept, "peakMiB": peak_mb()}))
