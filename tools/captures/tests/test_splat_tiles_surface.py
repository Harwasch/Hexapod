"""The packer ships an epoch's height surfaces beside its confidence grid (ADR 0018).

Opt-in, as the confidence grid is: without one every byte is what it was (the synthetic-tree
gates hold that for the committed fixtures; this holds it for a hierarchy). With one, its
file is copied beside the tiles and its descriptor goes on the root last -- after the
confidence grid, or after the collision grid when there is none. Nothing per tile.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_splat_tiles_confidence import BUDGET, COUNT
from test_splat_tiles_confidence import _grid as confidence_grid
from test_splat_tiles_lod import scene, write_ply

import cell_grid
import splat_tiles


def surface_grid(root: Path, *, level: int = 1, seed: int = 9, uri: str = "surface.bin") -> Path:
    """A surface grid over the scene's ground (+/-9 m): random heights, some cells empty."""
    rng = np.random.default_rng(seed)
    cell = cell_grid.Lattice(level).cell_m
    i0 = j0 = int(np.floor(-9.0 / cell))
    n = int(np.ceil(18.0 / cell))
    data = np.zeros((n, n, 5), dtype=np.int16)
    has = rng.random((n, n)) < 0.8
    top = rng.integers(-300, 3000, (n, n))
    data[..., 0] = np.where(has, top, cell_grid.NO_HEIGHT)
    data[..., 1] = np.where(has, top - rng.integers(0, 300, (n, n)), cell_grid.NO_HEIGHT)
    data[..., 2] = np.where(has, rng.integers(1, 256, (n, n)), 0)
    data[..., 3] = rng.integers(0, 256, (n, n))
    data[..., 4] = np.where(has, rng.integers(1, 256, (n, n)), 0)
    grid = cell_grid.CellGrid(
        "surface",
        cell_grid.SURFACE_CHANNELS,
        (cell_grid.CellPlane(level, i0, j0, n, n, data),),
        {"policy": "surface-v1", "noData": cell_grid.NO_HEIGHT},
    )
    folder = root / "surface"
    descriptor = cell_grid.encode(grid, folder, uri)
    (folder / "surface.json").write_text(json.dumps(descriptor, indent=1))
    return folder / "surface.json"


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("surface")
    ply = write_ply(root / "scene.ply", scene(COUNT, seed=5))
    grid, heights = confidence_grid(root), surface_grid(root)
    out = {"grid": grid, "heights": heights}
    for name, kwargs in (
        ("plain", {}),
        ("graded", {"confidence": grid}),
        ("both", {"confidence": grid, "surface": heights}),
        ("alone", {"surface": heights}),
    ):
        out[name] = root / name
        out[f"{name}_stats"] = splat_tiles.convert(
            ply, root / name, 46.8, -91.9, 180.0, tile_gaussians=BUDGET, **kwargs
        )
    return out


def test_without_a_surface_nothing_changes_and_with_one_only_the_root_extras_do(
    packed: dict[str, Any],
) -> None:
    for before, after in (("graded", "both"), ("plain", "alone")):
        old, new = packed[before], packed[after]
        assert {p.name for p in new.iterdir()} == {p.name for p in old.iterdir()} | {"surface.bin"}
        for path in old.iterdir():
            if path.name != "tileset.json":
                assert (new / path.name).read_bytes() == path.read_bytes(), path.name
        # tileset.json is the one before plus the key, appended last: take it away and the
        # text is the text before, byte for byte.
        document = json.loads((new / "tileset.json").read_text())
        descriptor = document["root"]["extras"].pop("surface")
        assert json.dumps(document, indent=1) == (old / "tileset.json").read_text()
        assert descriptor == json.loads(packed["heights"].read_text())
        assert (new / "surface.bin").read_bytes() == (
            packed["heights"].parent / "surface.bin"
        ).read_bytes()
        assert set(packed[f"{after}_stats"]) - set(packed[f"{before}_stats"]) == {"surface_level"}


def test_the_surface_comes_after_the_confidence_grid_or_the_collision_grid(
    packed: dict[str, Any],
) -> None:
    both = json.loads((packed["both"] / "tileset.json").read_text())["root"]["extras"]
    assert list(both) == [
        "gaussians",
        "confidenceMin",
        "viewCones",
        "collision",
        "confidence",
        "surface",
    ]
    alone = json.loads((packed["alone"] / "tileset.json").read_text())["root"]["extras"]
    assert list(alone) == ["gaussians", "viewCones", "collision", "surface"]
    grids = cell_grid.from_tileset(packed["both"] / "tileset.json")
    assert list(grids) == ["confidence", "surface"]
    plane = grids["surface"].planes[0]
    source = json.loads(packed["heights"].read_text())
    original = cell_grid.decode(source, (packed["heights"].parent / "surface.bin").read_bytes())
    np.testing.assert_array_equal(plane.data, original.planes[0].data)
    assert packed["both_stats"]["surface_level"] == 1


def test_a_grid_that_is_not_a_surface_grid_or_takes_a_taken_name_is_refused(
    tmp_path: Path,
) -> None:
    ply = write_ply(tmp_path / "scene.ply", scene(500, seed=6))
    graded = confidence_grid(tmp_path / "g")
    with pytest.raises(splat_tiles.GridRefused, match="not a surface grid") as refused:
        splat_tiles.convert(ply, tmp_path / "a", 0.0, 0.0, 0.0, surface=graded)
    assert refused.value.key == "surface"
    # The confidence grid's own file name, when both are given.
    clash = surface_grid(tmp_path / "c", uri="confidence.bin")
    splat_tiles.convert(ply, tmp_path / "alone", 0.0, 0.0, 0.0, surface=clash)  # alone: fine
    with pytest.raises(splat_tiles.GridRefused, match="file name") as refused:
        splat_tiles.convert(ply, tmp_path / "b", 0.0, 0.0, 0.0, confidence=graded, surface=clash)
    assert refused.value.key == "surface"
    tampered = surface_grid(tmp_path / "t")
    descriptor = json.loads(tampered.read_text())
    descriptor["sha256"] = "0" * 64
    tampered.write_text(json.dumps(descriptor))
    with pytest.raises(cell_grid.CellGridError, match="sha256"):
        splat_tiles.convert(ply, tmp_path / "d", 0.0, 0.0, 0.0, surface=tampered)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
