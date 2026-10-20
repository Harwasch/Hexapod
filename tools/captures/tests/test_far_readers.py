"""Every reader that binds by tile checksum, held to the 10.4 km fixture
(data/tiles/synthetic-far): its moved tiles are written in frames of their own, and each
reader must compose their transforms (docs/DECISIONS/0011-splat-tile-origins.md), so that
what it binds is keyed by expected.json's site-frame checksums. Each test fails when its
reader drops a tile's offset (review's mutations of 589947d0)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import synthetic_far

REPO_ROOT = Path(__file__).resolve().parents[3]
FAR = REPO_ROOT / "data" / "tiles" / "synthetic-far"
TILES = FAR / "splat"


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads((FAR / "source" / "expected.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def checksums(expected: dict) -> dict[str, str]:
    """Per uri, its site-frame checksum, and at least four of them moved tiles'."""
    out = {uri: row["checksum"] for uri, row in expected["tiles"].items()}
    assert sum(any(row["offset"]) for row in expected["tiles"].values()) >= 4
    return out


def _leaves(tileset: dict) -> set[str]:
    out: set[str] = set()
    stack = [tileset["root"]]
    while stack:
        tile = stack.pop()
        stack.extend(tile.get("children", []))
        if not tile.get("children"):
            out.add(tile["content"]["uri"])
    return out


def _tileset() -> dict:
    return json.loads((TILES / "tileset.json").read_text(encoding="utf-8"))


def _instances(expected: dict, label: int = 1) -> dict:
    """An instances.json bound to the fixture: every gaussian of every tile is `label`."""
    tiles = {row["checksum"]: [label, row["gaussians"]] for row in expected["tiles"].values()}
    return {
        "format": "hexapod.instances",
        "version": 1,
        "instances": [
            {
                "id": label,
                "parent": None,
                "level": 0,
                "splats": 1,
                "bounds": {"min": [-1, -1, -1], "max": [1, 1, 1]},
                "tags": [{"label": "plot", "score": 1.0}],
                "category": "trees",
            }
        ],
        "tiles": tiles,
    }


def _site_leaves() -> tuple[np.ndarray, np.ndarray]:
    """The scene's gaussians (the leaves in the site frame) and an id for each: 2 within
    50 m of the north-east plot, else 1."""
    positions = synthetic_far.scene().positions.astype(np.float64)
    plot = np.asarray(synthetic_far.PLOTS["north-east"], np.float64)
    ids = np.where(np.linalg.norm(positions - plot, axis=1) < 50.0, 2, 1).astype(np.int64)
    assert (ids == 2).any()
    return positions, ids


# ---------------------------------------------------------------------- binding checks


def test_check_binding_accepts_site_frame_checksums_and_refuses_local_ones(
    expected: dict,
) -> None:
    """publish-instances.yml's check, which once called tile_positions with no offset."""
    import rebind_instances
    import splat_tiles
    from rig_tiles import glb_spz
    from synthetic_tree import checksum_positions

    rebind_instances.check_binding(TILES, _instances(expected))
    local = {}
    for uri, row in expected["tiles"].items():
        columns = splat_tiles.unpack_spz(glb_spz(TILES / uri))
        xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype("<f4")
        xyz[xyz == 0.0] = 0.0
        local[checksum_positions(xyz)] = [1, row["gaussians"]]
    with pytest.raises(ValueError, match="not in instances.json's binding"):
        rebind_instances.check_binding(TILES, {"tiles": local})
    short = _instances(expected)
    first = next(iter(short["tiles"]))
    short["tiles"][first] = [1, short["tiles"][first][1] - 1]
    with pytest.raises(ValueError, match="do not cover"):
        rebind_instances.check_binding(TILES, short)
