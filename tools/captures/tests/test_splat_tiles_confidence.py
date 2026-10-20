"""The packer ships an epoch's confidence grid and each tile's least confident cell.

Opt-in: without a grid every byte is what it was (the synthetic-tree gates hold that for
the committed fixtures; this holds it for a hierarchy). With one, the grid's file is
copied beside the tiles, its descriptor goes on the root last, and every tile states the
lowest score of the scored cells under it -- a leaf from its own gaussians, a parent the
least of its children's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_splat_tiles_lod import glb_positions, scene, tiles_of, write_ply

import cell_grid
import splat_tiles

COUNT = 6_000
BUDGET = 400


def _grid(root: Path, *, level: int = 1, seed: int = 4, kind: str = "confidence") -> Path:
    """A confidence grid over the scene's ground (+/-8 m): random scores, some cells empty."""
    rng = np.random.default_rng(seed)
    cell = cell_grid.Lattice(level).cell_m
    i0 = j0 = int(np.floor(-9.0 / cell))
    n = int(np.ceil(18.0 / cell))
    data = np.zeros((n, n, 3), dtype=np.uint8)
    data[..., 0] = np.where(rng.random((n, n)) < 0.85, rng.integers(1, 256, (n, n)), 0)
    data[..., 1] = rng.integers(0, 8, (n, n))
    data[..., 2] = rng.integers(0, 40, (n, n))
    planes = cell_grid.min_pyramid(cell_grid.CellPlane(level, i0, j0, n, n, data), 8)
    grid = cell_grid.CellGrid(
        kind, cell_grid.CONFIDENCE_CHANNELS, tuple(planes), {"policy": cell_grid.WEAK_POLICY}
    )
    folder = root / "confidence"
    descriptor = cell_grid.encode(grid, folder, "confidence.bin")
    (folder / "confidence.json").write_text(json.dumps(descriptor, indent=1))
    return folder / "confidence.json"


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("confidence")
    ply = write_ply(root / "scene.ply", scene(COUNT, seed=5))
    plain, measured = root / "plain", root / "measured"
    grid = _grid(root)
    stats_plain = splat_tiles.convert(ply, plain, 46.8, -91.9, 180.0, tile_gaussians=BUDGET)
    stats = splat_tiles.convert(
        ply, measured, 46.8, -91.9, 180.0, tile_gaussians=BUDGET, confidence=grid
    )
    return {
        "plain": plain,
        "measured": measured,
        "grid": grid,
        "stats": stats,
        "stats_plain": stats_plain,
    }


def _strip(tile: dict[str, Any]) -> None:
    tile["extras"].pop("confidenceMin", None)
    for child in tile.get("children", []):
        _strip(child)


def test_without_a_grid_nothing_changes_and_with_one_only_the_extras_do(
    packed: dict[str, Any],
) -> None:
    plain, measured = packed["plain"], packed["measured"]
    tiles = sorted(p.name for p in plain.glob("*.glb"))
    assert len(tiles) > 7  # a real hierarchy, several levels deep
    # Every GLB, the collision grid and the view cones are the bytes they were.
    for name in [*tiles, "collision.bin", "viewcones.bin"]:
        assert (measured / name).read_bytes() == (plain / name).read_bytes(), name
    assert {p.name for p in measured.iterdir()} == {p.name for p in plain.iterdir()} | {
        "confidence.bin"
    }
    # And tileset.json is the plain one plus the two keys, written where they go: take
    # them away and the text is the plain text, byte for byte.
    document = json.loads((measured / "tileset.json").read_text())
    descriptor = document["root"]["extras"].pop("confidence")
    _strip(document["root"])
    assert json.dumps(document, indent=1) == (plain / "tileset.json").read_text()
    assert descriptor == json.loads(packed["grid"].read_text())
    assert (measured / "confidence.bin").read_bytes() == (
        packed["grid"].parent / "confidence.bin"
    ).read_bytes()
    assert set(packed["stats"]) - set(packed["stats_plain"]) == {
        "confidence_level",
        "confidence_min",
    }


def test_the_root_extras_keep_their_order_and_the_grid_comes_last(
    packed: dict[str, Any],
) -> None:
    root = json.loads((packed["measured"] / "tileset.json").read_text())["root"]
    assert list(root["extras"]) == [
        "gaussians",
        "confidenceMin",
        "viewCones",
        "collision",
        "confidence",
    ]
    for tile, _, _ in tiles_of({"root": root}):
        keys = list(tile["extras"])
        assert keys[0] == "gaussians"
        if "confidenceMin" in keys:
            assert keys[1] == "confidenceMin"
    grids = cell_grid.from_tileset(packed["measured"] / "tileset.json")
    assert list(grids) == ["confidence"]


def test_a_leaf_states_its_least_confident_cell_and_a_parent_its_childrens(
    packed: dict[str, Any],
) -> None:
    measured = packed["measured"]
    document = json.loads((measured / "tileset.json").read_text())
    grid = cell_grid.from_tileset(measured / "tileset.json")["confidence"]
    offsets = splat_tiles.tile_offsets(document)
    leaves = parents = 0
    for tile, _, _ in tiles_of(document):
        uri = tile["content"]["uri"]
        if tile.get("children"):
            below = [c["extras"].get("confidenceMin") for c in tile["children"]]
            known = [v for v in below if v is not None]
            assert tile["extras"].get("confidenceMin") == (min(known) if known else None), uri
            parents += 1
            continue
        _, local = glb_positions(measured / uri)
        site = splat_tiles.site_positions(local, offsets[uri])
        # Brute force: every gaussian's cell, one at a time, at the grid's finest level.
        plane = grid.planes[0]
        scores = []
        for x, y, _ in site.astype(np.float64):
            i = int(np.floor(x / plane.cell_m)) - plane.i0
            j = int(np.floor(y / plane.cell_m)) - plane.j0
            if 0 <= i < plane.ni and 0 <= j < plane.nj and plane.data[j, i, 0] > 0:
                scores.append(int(plane.data[j, i, 0]))
        assert tile["extras"].get("confidenceMin") == (min(scores) if scores else None), uri
        leaves += 1
    assert leaves > 4 and parents > 1
    assert document["root"]["extras"]["confidenceMin"] == packed["stats"]["confidence_min"]


def test_a_grid_that_is_not_a_confidence_grid_or_not_its_bytes_is_refused(
    tmp_path: Path,
) -> None:
    ply = write_ply(tmp_path / "scene.ply", scene(500, seed=6))
    surface = _grid(tmp_path / "s", kind="surface")
    with pytest.raises(cell_grid.CellGridError, match="not a confidence grid"):
        splat_tiles.convert(ply, tmp_path / "a", 0.0, 0.0, 0.0, confidence=surface)
    tampered = _grid(tmp_path / "t")
    descriptor = json.loads(tampered.read_text())
    descriptor["sha256"] = "0" * 64
    tampered.write_text(json.dumps(descriptor))
    with pytest.raises(cell_grid.CellGridError, match="sha256"):
        splat_tiles.convert(ply, tmp_path / "b", 0.0, 0.0, 0.0, confidence=tampered)
    for uri in ("collision.bin", "../confidence.bin", "splat_0.glb", "tileset.json"):
        clash = _grid(tmp_path / f"u{len(uri)}")
        descriptor = json.loads(clash.read_text())
        descriptor["uri"] = uri
        clash.write_text(json.dumps(descriptor))
        with pytest.raises(cell_grid.CellGridError, match="file name"):
            splat_tiles.convert(ply, tmp_path / "c", 0.0, 0.0, 0.0, confidence=clash)
    assert not (tmp_path / "a").exists() and not (tmp_path / "c").exists()
