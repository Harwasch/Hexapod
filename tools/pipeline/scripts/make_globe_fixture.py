"""Four simulated gaussians, using the existing production KHR/SPZ packager. No GPU."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import gaussians
from captures_bridge import splat_tiles_convert

OUTPUT = Path(__file__).resolve().parents[3] / "apps/web/public/fixtures/globe-splat"


def generate(output: Path) -> None:
    """Write tiny, georeferenced content with the production pipeline seam."""
    columns = {name: np.zeros(4, dtype=np.float32) for name in gaussians.CANONICAL_PROPERTIES}
    columns["x"][:] = [-1, 1, -1, 1]
    columns["y"][:] = [-1, -1, 1, 1]
    columns["z"][:] = [0, 1, 2, 3]
    columns["opacity"][:] = 4
    columns["rot_0"][:] = 1
    for axis in range(3):
        columns[f"scale_{axis}"][:] = -1
    columns["f_dc_1"][:] = 1
    with tempfile.TemporaryDirectory() as scratch:
        ply = Path(scratch) / "fixture.ply"
        gaussians.write_ply(ply, columns)
        splat_tiles_convert(ply, output, lat=47.6405, lon=-122.1395, height=10)


if __name__ == "__main__":
    generate(OUTPUT)
