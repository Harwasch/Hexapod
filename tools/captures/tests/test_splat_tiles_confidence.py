"""The packer ships an epoch's confidence grid and each tile's least confident cell.

Opt-in: without a grid every byte is what it was (the synthetic-tree gates hold that for
the committed fixtures; this holds it for a hierarchy). With one, the grid's file is
copied beside the tiles, its descriptor goes on the root last, and every tile states the
lowest score of the cells its box covers in plan: scored cells at their score, unseen ones
at 0, whether or not any of its gaussians is in them -- strict gating removes exactly the
gaussians of a weak cell, so the tile drawn over it holds none of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_splat_tiles_lod import box_bounds, scene, tiles_of, write_ply

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
    # Half the empty cells are unseen: expected and not seen, so weak.
    unseen = (data[..., 0] == 0) & (rng.random((n, n)) < 0.5)
    data[..., 1] |= np.where(unseen, cell_grid.REASON_BITS["unseen"], 0).astype(np.uint8)
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


def _covered_min(plane: cell_grid.CellPlane, low: np.ndarray, high: np.ndarray) -> int | None:
    """Brute force, one cell at a time: the least score of the cells a plan box touches,
    an unseen cell at 0, an empty cell that is not unseen not counted."""
    unseen = cell_grid.REASON_BITS["unseen"]
    found = []
    for j in range(plane.nj):
        for i in range(plane.ni):
            x0, y0 = (plane.i0 + i) * plane.cell_m, (plane.j0 + j) * plane.cell_m
            if x0 + plane.cell_m <= low[0] or x0 > high[0]:
                continue
            if y0 + plane.cell_m <= low[1] or y0 > high[1]:
                continue
            score, reasons = int(plane.data[j, i, 0]), int(plane.data[j, i, 1])
            if reasons & unseen:
                found.append(0)
            elif score > 0:
                found.append(score)
    return min(found) if found else None


def test_every_tile_states_the_least_confident_cell_under_its_box(
    packed: dict[str, Any],
) -> None:
    measured = packed["measured"]
    document = json.loads((measured / "tileset.json").read_text())
    grid = cell_grid.from_tileset(measured / "tileset.json")["confidence"]
    offsets = splat_tiles.tile_offsets(document)
    plane = grid.planes[0]
    leaves = parents = unseen = 0
    for tile, _, _ in tiles_of(document):
        uri = tile["content"]["uri"]
        low, high = box_bounds(tile)
        offset = np.asarray(offsets[uri], dtype=np.float64)
        expected = _covered_min(plane, low + offset, high + offset)
        assert tile["extras"].get("confidenceMin") == expected, uri
        unseen += expected == 0
        if tile.get("children"):
            # A parent's box holds its children's: never more confident than they are.
            below = [c["extras"].get("confidenceMin") for c in tile["children"]]
            known = [v for v in below if v is not None]
            assert not known or tile["extras"]["confidenceMin"] <= min(known), uri
            parents += 1
        else:
            leaves += 1
    assert leaves > 4 and parents > 1 and unseen > 0
    assert document["root"]["extras"]["confidenceMin"] == packed["stats"]["confidence_min"]


def _one_tile(root: Path, mark: tuple[int, int] | None) -> int | None:
    """A single tile of ground over (0..4 m)^2 with no gaussian in the cell (4, 4) at level 1
    (x, y in [2, 2.5)), and a grid strong everywhere but that cell, which `mark` sets."""
    root.mkdir(parents=True)
    columns = scene(2_000, seed=7)
    rng = np.random.default_rng(8)
    xy = rng.uniform(0.0, 4.0, (2_000, 2))
    hole = (xy >= 2.0).all(axis=1) & (xy < 2.5).all(axis=1)
    xy[hole] += 1.0  # out of the cell, still inside the box
    columns["x"] = (np.round(xy[:, 0] * 4096) / 4096).astype(np.float32)
    columns["y"] = (np.round(xy[:, 1] * 4096) / 4096).astype(np.float32)
    columns["z"] = np.zeros(2_000, dtype=np.float32)
    ply = write_ply(root / "ground.ply", columns)
    data = np.zeros((12, 12, 3), dtype=np.uint8)  # i, j in [-2, 10): -1 m to 5 m
    data[..., 0] = 255
    if mark is not None:
        data[4 + 2, 4 + 2, :2] = mark
    plane = cell_grid.CellPlane(1, -2, -2, 12, 12, data)
    grid = cell_grid.CellGrid(
        "confidence", cell_grid.CONFIDENCE_CHANNELS, (plane,), {"policy": cell_grid.WEAK_POLICY}
    )
    folder = root / "confidence"
    descriptor = cell_grid.encode(grid, folder, "confidence.bin")
    (folder / "confidence.json").write_text(json.dumps(descriptor))
    out = root / "tiles"
    splat_tiles.convert(ply, out, 0.0, 0.0, 0.0, confidence=folder / "confidence.json")
    document = json.loads((out / "tileset.json").read_text())
    assert "children" not in document["root"]  # one tile: its box is the whole patch
    value = document["root"]["extras"].get("confidenceMin")
    return None if value is None else int(value)


def test_a_cell_gating_emptied_or_nobody_saw_lowers_the_tile_over_it(tmp_path: Path) -> None:
    """Review R3 #54 (medium): strict gating removes a weak cell's gaussians, so the tile
    over it holds none of them; it must still say the cell is weak. An unseen cell under
    it counts as 0; an empty cell nobody aimed at is not this epoch's and says nothing."""
    unseen = cell_grid.REASON_BITS["unseen"]
    assert _one_tile(tmp_path / "gated", (1, cell_grid.REASON_BITS["views"])) == 1
    assert _one_tile(tmp_path / "unseen", (0, unseen)) == 0
    assert _one_tile(tmp_path / "empty", (0, 0)) == 255
    assert _one_tile(tmp_path / "plain", None) == 255


def test_a_grid_that_is_not_a_confidence_grid_or_not_its_bytes_is_refused(
    tmp_path: Path,
) -> None:
    ply = write_ply(tmp_path / "scene.ply", scene(500, seed=6))
    heights = cell_grid.CellGrid(
        "surface",
        cell_grid.SURFACE_CHANNELS,
        (cell_grid.CellPlane(1, -18, -18, 36, 36, np.zeros((36, 36, 5), dtype=np.int16)),),
    )
    descriptor = cell_grid.encode(heights, tmp_path / "s", "surface.bin")
    surface = tmp_path / "s" / "surface.json"
    surface.write_text(json.dumps(descriptor))
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
