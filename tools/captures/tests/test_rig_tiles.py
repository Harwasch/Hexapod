"""rig_tiles: per-tile identity for a level-of-detail tileset's motion rig."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import rig_tiles
import splat_tiles
from synthetic_tree import checksum_positions

REPO_ROOT = Path(__file__).resolve().parents[3]
TREE = REPO_ROOT / "data" / "tiles" / "synthetic-tree"
LOD = REPO_ROOT / "data" / "tiles" / "synthetic-tree-lod"


def _sorted_rows(positions: np.ndarray) -> np.ndarray:
    """Rows of raw float32 bits, sorted, so two orders of the same set compare equal."""
    bits = np.ascontiguousarray(positions, dtype="<f4").view("<u4").reshape(-1, 3)
    return bits[np.lexsort(bits.T[::-1])]


def test_a_decoded_tile_digests_as_the_runtime_unbakes_it() -> None:
    """The single-tile fixture's GLB, decoded here, carries the rig's own checksum.

    The runtime's digest is over its un-baked positions (apps/web e2e asserts
    fnv1a32:12000:8b008bc0 against a real CesiumJS); matching it from the SPZ bytes is what
    makes a per-tile checksum computed offline mean the same thing in the browser.
    """
    rig = json.loads((TREE / "source" / "rig.json").read_text(encoding="utf-8"))
    positions = rig_tiles.tile_positions(TREE / "splat" / "splat.glb")
    assert checksum_positions(positions) == rig["canonicalChecksum"]


def test_leaves_hold_every_gaussian_once_and_parents_are_new() -> None:
    tileset = json.loads((LOD / "tileset.json").read_text(encoding="utf-8"))
    canonical = np.fromfile(TREE / "source" / "positions.f32", dtype="<f4").reshape(-1, 3)
    leaves: list[np.ndarray] = []
    parents: list[np.ndarray] = []

    def walk(tile: dict) -> None:
        positions = rig_tiles.tile_positions(LOD / tile["content"]["uri"])
        (parents if tile.get("children") else leaves).append(positions)
        for child in tile.get("children", []):
            walk(child)

    walk(tileset["root"])
    leaf = np.concatenate(leaves)
    assert leaf.shape == canonical.shape
    assert np.array_equal(_sorted_rows(leaf), _sorted_rows(canonical))
    assert sum(len(p) for p in parents) > 0


def test_the_committed_lod_rig_is_the_stamped_tree_rig() -> None:
    """Not hand-edited, not stale: the tree's rig plus this tileset's tile checksums."""
    tree_rig = json.loads((TREE / "source" / "rig.json").read_text(encoding="utf-8"))
    committed = json.loads((LOD / "rig.json").read_text(encoding="utf-8"))
    assert committed == rig_tiles.stamp(tree_rig, LOD)
    tileset = json.loads((LOD / "tileset.json").read_text(encoding="utf-8"))
    assert len(committed["tileChecksums"]) == len(rig_tiles.tile_uris(tileset))


def test_stamping_a_fresh_pack(tmp_path: Path) -> None:
    """Any tile budget: every tile's digest is in the set, and a changed gaussian is not."""
    source = TREE / "source" / "splat.ply"
    splat_tiles.convert(source, tmp_path / "lod", 28.0389, -82.6966, 0.0, 0.02, 3000)
    checksums = set(rig_tiles.tile_checksums(tmp_path / "lod"))
    tileset = json.loads((tmp_path / "lod" / "tileset.json").read_text(encoding="utf-8"))
    for uri in rig_tiles.tile_uris(tileset):
        positions = rig_tiles.tile_positions(tmp_path / "lod" / uri)
        assert checksum_positions(positions) in checksums
        nudged = positions.copy()
        nudged[0, 0] += 1 / 4096
        assert checksum_positions(nudged) not in checksums
