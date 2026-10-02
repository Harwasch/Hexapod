"""`experiments/sh_compare.py`: one splat packed at SH 0, 1 and 3, side by side.

On a synthetic SH-3 splat in a COLMAP-like frame, placed with `--place` (the real `place`
stage, SH turned), the three tilesets must be what the `package` stage would publish at
each degree: every tile at its degree as CesiumJS counts it, the same gaussians, bytes
that grow with the degree, and -- with `--site` -- three site documents the seeder reads,
each a scan-width and a half east of the last. Nothing here is a measurement of the
viewers; the manifest says what to measure on them.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest
from test_harmonics import columns_with_sh, write_raw_ply

import gaussians
import harmonics
from experiments import sh_compare


def _glb_degree(path: Path) -> int:
    blob = path.read_bytes()
    length, _ = struct.unpack_from("<II", blob, 12)
    document = json.loads(blob[20 : 20 + length])
    names = document["meshes"][0]["primitives"][0]["attributes"]
    return {3: 1, 8: 2, 15: 3}.get(sum("SH_DEGREE_" in name for name in names), 0)


def _scene(tmp_path: Path) -> tuple[Path, Path]:
    """A 30k-gaussian SH-3 splat with trained-looking SH (mostly small, some shine), and a
    georef.json whose frame turns it upright and recentres it, as an EXIF run's would."""
    columns = columns_with_sh(30_000, 3, seed=21)
    g = np.random.default_rng(22)
    for name in harmonics.rest_names(3):
        shine = g.laplace(0.0, 0.03, size=columns["x"].shape[0])
        columns[name] = np.asarray(shine, dtype=np.float32)
    ply = write_raw_ply(tmp_path / "trained.ply", columns)
    angle = 0.4
    frame = {
        "source": "test",
        "rotation": [
            [math.cos(angle), 0.0, math.sin(angle)],
            [math.sin(angle), 0.0, -math.cos(angle)],
            [0.0, 1.0, 0.0],
        ],
        "scale": 1.0,
        "recentre": True,
    }
    georef = tmp_path / "georef.json"
    georef.write_text(json.dumps({"lat": 44.9, "lon": -93.4, "height": 250.0, "frame": frame}))
    return ply, georef


def test_one_site_packed_three_ways_as_three_sites(tmp_path: Path) -> None:
    ply, georef = _scene(tmp_path)
    out = tmp_path / "tiles"

    # Small tiles, so there are merged parents: they carry the variant's degree too.
    argv = [str(ply), str(out), "--georef", str(georef), "--place", "--site", "shiny"]
    sh_compare.main([*argv, "--tile-gaussians", "8000"])

    manifest = json.loads((out / "sh_compare.json").read_text())
    variants = {variant["shDegree"]: variant for variant in manifest["variants"]}
    assert sorted(variants) == [0, 1, 3]
    assert manifest["source"]["shDegree"] == 3
    # The same gaussians at every degree; only the colour bytes differ, and grow with it.
    assert len({v["gaussians"] for v in variants.values()}) == 1
    assert len({v["tiles"] for v in variants.values()}) == 1
    assert variants[0]["tileBytesRatio"] == 1.0
    assert 1.0 < variants[1]["tileBytesRatio"] < variants[3]["tileBytesRatio"]
    for degree, variant in variants.items():
        folder = out / variant["dir"]
        assert folder == out / f"shiny-sh{degree}" / "splat"
        tiles = [p for p in folder.glob("*.glb")]
        assert len(tiles) == variant["tiles"]
        # Every tile at the variant's degree: CesiumJS draws a tileset at one degree.
        assert {_glb_degree(tile) for tile in tiles} == {degree}
        site = json.loads((folder.parent / "site.json").read_text())
        assert site["slug"] == f"shiny-sh{degree}"
        assert site["assets"][0]["path"] == "splat/tileset.json"
        assert site["boundary"][0] == site["boundary"][-1]
    # Side by side on the globe: each a scan-width and a half east of the one before.
    assert variants[0]["eastOffsetM"] == 0.0
    assert variants[3]["eastOffsetM"] == pytest.approx(2 * variants[1]["eastOffsetM"], abs=2e-3)
    assert variants[1]["lon"] > variants[0]["lon"]
    # Placed by the real stage: the canonical.ply carries degree 3, turned.
    placed = out / "sh-compare-place" / "stages" / "place" / "out" / "canonical.ply"
    assert gaussians.read_splat(placed, sh_degree=3).sh_degree == 3


def test_a_splat_without_its_sh_is_refused_by_name(tmp_path: Path) -> None:
    columns = columns_with_sh(500, 0)
    ply = tmp_path / "canonical.ply"
    gaussians.write_ply(ply, columns)
    with pytest.raises(SystemExit, match="sh_degree: 3"):
        sh_compare.main([str(ply), str(tmp_path / "out"), "--lat", "1", "--lon", "2"])
