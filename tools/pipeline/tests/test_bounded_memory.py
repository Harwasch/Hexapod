"""The stages after training run in memory that does not grow with the splat.

Each measurement is a process of its own (`memory_probe.py`), so its peak resident set
(`ru_maxrss`) is the stages' and nothing else's. Measured on this code, 2026-09-27, the
synthetic orbit with held-out arrays, 16 cameras, default chunking:

    gaussians   whole-splat quality + place    chunked quality + place + thumbnail + ground
    250k        122 MB                          162 MB
    1M          365 MB                          182 MB
    4M          1,288 MB                        188 MB
    8M          fails at a 1.5 GB limit         197 MB (81 s)

The whole-splat stages grew ~0.3 GB a million gaussians on this 14-property input (and
~0.75 GB a million on SH3 input, apps/api/app/worker/README.md); the chunked ones hold
fixed-size chunks, histograms, partitions and buffers, whose caps (`outofcore.BUDGET`,
`outofcore.COLLECT`, `quality.OCCLUDER_BLOCK`, `quality.SAMPLE_BYTES`,
`quality.SEEN_BYTES`) the defaults reach by a few million.

The always-on test shrinks those caps and the chunk (`--small --chunk`) so that 150k and
600k gaussians are both past them, and requires the peak not to grow between the two
while the whole-splat path's does. The 8M run is opt-in (PIPELINE_BENCH=1): it is the
plan's pass criterion -- quality and place of an 8M-gaussian splat under a 1.5 GB limit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import splat_io

PROBE = Path(__file__).resolve().parent / "memory_probe.py"


def _probe(*args: str) -> dict[str, Any]:
    done = subprocess.run(
        [sys.executable, str(PROBE), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    last = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else "{}"
    result: dict[str, Any] = json.loads(last)
    return result


def test_memory_does_not_grow_with_the_splat(tmp_path: Path) -> None:
    peaks: dict[int, float] = {}
    whole: dict[int, float] = {}
    for count in (150_000, 600_000):
        root = tmp_path / f"n{count}"
        _probe("make", str(root), str(count), "8", "--held")
        stages = ("quality", "place", "thumbnail", "ground")
        chunked = _probe("chunked", str(root), *stages, "--small", "--chunk", "32768")
        peaks[count] = float(chunked["peakMb"])
        whole[count] = float(_probe("whole", str(root))["peakMb"])
    growth = peaks[600_000] - peaks[150_000]
    grown = whole[600_000] - whole[150_000]
    sys.stdout.write(f"\nchunked peaks {peaks} (+{growth:.1f} MB); whole {whole} (+{grown:.1f})\n")
    # Four times the gaussians: the whole-splat path pays ~0.3 GB a million for them,
    # the chunked one nothing beyond allocator noise.
    assert grown > 80
    assert growth < 15


@pytest.mark.skipif(
    os.environ.get("PIPELINE_BENCH") != "1",
    reason="8M gaussians through the stages takes a few minutes; PIPELINE_BENCH=1 runs it",
)
def test_eight_million_gaussians_fit_in_a_fixed_budget(tmp_path: Path) -> None:
    root = tmp_path / "n8m"
    _probe("make", str(root), "8000000", "16", "--held")
    limit = 1536
    result = _probe(
        "chunked", str(root), "quality", "place", "thumbnail", "ground", "--limit-mb", str(limit)
    )
    sys.stdout.write(f"\n8M gaussians, quality + place + thumbnail + ground: {result}\n")
    assert result["peakMb"] < limit
    run = next(root.glob("run-*"))
    gated = splat_io.read_layout(run / "stages" / "quality" / "out" / "gated.ply")
    canonical = splat_io.read_layout(run / "stages" / "place" / "out" / "canonical.ply")
    # Most of the orbit is kept, and every kept gaussian is placed.
    assert canonical.count == gated.count > 4_000_000
