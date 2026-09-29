"""The packer's memory does not grow with the splat it packs (splat_tiles.convert, "Out of core").

A trained splat is a 3DGS PLY of 62 floats a gaussian (248 bytes, with its SH degree 3,
which the tiles carry): 8M of them is a 2 GB file, and the worker that packages it has 2 GB
in all. So `convert` reads the PLY in windows and sorts through disk, and what it holds for
the whole scan is a few arrays of 8-16 bytes a gaussian. Carrying the SH (as the bytes SPZ
will hold) makes each sorted record 109 bytes rather than 64, and the windows and buckets
shrink to match, so the peak barely moves: 1.5M gaussians at ~210 MB with SH degree 3,
~207 MB without.

Measured in a child process, by its own peak resident set (`VmHWM`), which counts the
pages of every window it maps as well as everything numpy allocates. Not `ru_maxrss`: Linux
carries the pre-exec high-water mark of the forked parent into the child's (`exec_mmap`
saves the old mm's hiwater into `signal->maxrss`), so a child of this test process reported
the test's own ~380 MB, not its own ~200. The default test packs
1.5M gaussians (a 370 MB PLY, two buckets of the disk sort) and holds the packer to a
bound that the whole-file reader it replaced could not meet; `PIPELINE_BENCH=1` runs the
8M-gaussian case the plan's Phase 2 names, against its 1.5 GB limit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

CAPTURES = Path(__file__).resolve().parents[1]
REST = [f"f_rest_{i}" for i in range(45)]
PROPERTIES = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", *REST]
PROPERTIES += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def write_trained_like_ply(path: Path, count: int, seed: int = 1) -> Path:
    """A lumpy scene as a full 3DGS PLY, written in chunks so the test itself stays small."""
    dtype = np.dtype([(name, "<f4") for name in PROPERTIES])
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header += [f"property float {name}" for name in PROPERTIES] + ["end_header"]
    rng = np.random.default_rng(seed)
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        for start in range(0, count, 500_000):
            n = min(500_000, count - start)
            rows = np.zeros(n, dtype=dtype)
            kind = rng.integers(0, 3, n)
            ground = np.c_[rng.uniform(-40, 40, (n, 2)), rng.normal(0, 0.05, n)]
            ball = rng.normal(size=(n, 3))
            ball = ball / np.linalg.norm(ball, axis=1, keepdims=True) * 6 + [5, -3, 6]
            haze = rng.uniform(-20, 20, (n, 3)) + [0, 0, 20]
            xyz = np.where(kind[:, None] == 0, ground, np.where(kind[:, None] == 1, ball, haze))
            for axis, name in enumerate("xyz"):
                rows[name] = xyz[:, axis]
            for name in ("f_dc_0", "f_dc_1", "f_dc_2", *REST):
                rows[name] = rng.normal(0, 0.3, n)
            rows["opacity"] = rng.uniform(-2, 4, n)
            for i in range(3):
                rows[f"scale_{i}"] = rng.normal(-4, 0.6, n)
            quat = rng.normal(size=(n, 4))
            quat /= np.linalg.norm(quat, axis=1, keepdims=True)
            for i in range(4):
                rows[f"rot_{i}"] = quat[:, i]
            handle.write(rows.tobytes())
        # Out of the page cache, as a PLY the worker downloaded a while ago would be: a
        # file written a moment ago sits in large dirty folios, and mapping a window of
        # one maps (and counts in the child's RSS) more of it than the window touches.
        handle.flush()
        os.fsync(handle.fileno())
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return path


def package_in_a_child(ply: Path, out: Path) -> dict[str, float]:
    """`convert` in a fresh interpreter; its stats plus its own peak RSS in MB."""
    code = textwrap.dedent(
        f"""
        import json
        from pathlib import Path
        import splat_tiles
        stats = splat_tiles.convert(Path({str(ply)!r}), Path({str(out)!r}), 46.84, -91.99, 0.0)
        status = Path("/proc/self/status").read_text().splitlines()
        peak_kb = next(int(line.split()[1]) for line in status if line.startswith("VmHWM:"))
        stats["peak_mb"] = peak_kb / 1024
        print(json.dumps(stats))
        """
    )
    # One BLAS thread: the packer uses none, and idle thread arenas only blur the number.
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=CAPTURES,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_packing_1_5m_gaussians_holds_a_fraction_of_the_file(tmp_path: Path) -> None:
    count = 1_500_000
    ply = write_trained_like_ply(tmp_path / "big.ply", count)
    size_mb = ply.stat().st_size / 2**20
    stats = package_in_a_child(ply, tmp_path / "splat")
    assert stats["gaussians"] + stats["dropped"] == count
    assert stats["tiles"] > 15
    assert stats["storage_overhead"] <= 0.2
    # The reader this replaced held the whole file and float64 copies of it: past 2x the
    # file (5M gaussians of 14 floats: 1.6 GB). Out of core it is ~120 MB of interpreter,
    # numpy and one 64 MB window, plus a few tens of bytes a gaussian: measured 1.5M at
    # ~200 MB and 8M (a 2 GB PLY) at ~380 MB.
    assert stats["peak_mb"] < 120 + 100 * count / 2**20, stats
    assert stats["peak_mb"] < size_mb, stats


@pytest.mark.skipif(
    os.environ.get("PIPELINE_BENCH") != "1",
    reason="8M gaussians is a 2 GB PLY and a few minutes; PIPELINE_BENCH=1 runs it",
)
def test_packing_8m_gaussians_fits_in_1_5_gb(tmp_path: Path) -> None:
    """The plan's Phase 2 bound: an 8M-gaussian splat packages under 1.5 GB."""
    count = 8_000_000
    ply = write_trained_like_ply(tmp_path / "big.ply", count)
    stats = package_in_a_child(ply, tmp_path / "splat")
    print(json.dumps(stats))
    assert stats["gaussians"] + stats["dropped"] == count
    assert stats["storage_overhead"] <= 0.2
    assert stats["peak_mb"] < 1536, stats
