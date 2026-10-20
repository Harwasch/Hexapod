"""`lod_parity.py`: every parent tile scored against its own leaves, reproducibly, in the
root's frame."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import lod_parity
import splat_tiles

ROOT = Path(__file__).resolve().parents[3]
TREE_PLY = ROOT / "data" / "tiles" / "synthetic-tree" / "source" / "splat.ply"
FAR = ROOT / "data" / "tiles" / "synthetic-far" / "splat"


def _parents(tileset_dir: Path) -> list[str]:
    document = json.loads((tileset_dir / "tileset.json").read_text(encoding="utf-8"))
    out: list[str] = []
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("children"):
            out.append(str(tile["content"]["uri"]))
            stack.extend(tile["children"])
    return out


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("lod") / "splat"
    splat_tiles.convert(TREE_PLY, out, 28.0389, -82.6966, 0.0, 0.02, 1500)
    return out


def test_every_parent_is_scored(packed: Path) -> None:
    report, sheet = lod_parity.parity(packed, size=64, azimuths=1)
    assert sorted(row["uri"] for row in report["tiles"]) == sorted(_parents(packed))
    assert len(report["tiles"]) > 1
    for row in report["tiles"]:
        assert row["leafGaussians"] > row["gaussians"] > 0
        for view in row["views"]:
            assert 0.0 <= view["holes"] <= 1.0
            assert view["coverage"] > 0.0
            assert math.isfinite(view["psnr"])
    assert len(sheet) == 1 and sheet[0][0].shape == (64, 64, 3)


def test_the_report_is_deterministic(packed: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    for out in (first, second):
        assert (
            lod_parity.main([str(packed), "--size", "48", "--azimuths", "2", "--out", str(out)])
            == 0
        )
    assert first.read_bytes() == second.read_bytes()


def test_a_far_site_is_measured_in_the_root_frame() -> None:
    """synthetic-far's tiles carry their own origins. Composed, every tile's gaussians sit
    in its box, both in the root's frame -- so a parent and its leaves are drawn from one
    camera where the viewer draws them."""
    document = json.loads((FAR / "tileset.json").read_text(encoding="utf-8"))
    offsets = splat_tiles.tile_offsets(document)
    assert any(np.any(o) for o in offsets.values()), "the fixture is meant to have origins"
    for tile in lod_parity.tiles_of(document):
        xyz = lod_parity.tile_splats(FAR, tile["uri"], offsets).positions
        assert np.all(xyz >= tile["low"] - 1e-3), tile["uri"]
        assert np.all(xyz <= tile["high"] + 1e-3), tile["uri"]
    report, _ = lod_parity.parity(FAR, size=48, azimuths=2)
    assert len(report["tiles"]) == len(_parents(FAR))
    root = report["tiles"][0]
    assert root["uri"] == "splat.glb" and root["mean"]["coverage"] > 0.0
