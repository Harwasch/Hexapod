"""Optimised parents: the tree the GPU sees is the tree the packer writes, and only the
parents' content changes when optimised ones are applied.

The GPU stage (tools/pipeline lod_optimise.py) builds the hierarchy with
`splat_tiles.hierarchy` on the same `canonical.ply`, optimises the parent gaussians
against the photos (Hierarchical 3DGS Sec. 5.1) and hands them back as a
`ParentOverrides` file keyed by tile and cell. What has to hold for that to be sound:

* `hierarchy` emits exactly the tiles, boxes, errors and gaussians `convert` writes;
* applying overrides changes each parent tile's gaussians to the overrides and nothing
  else -- not a leaf, not a box, not an error;
* overrides for another PLY or other packing parameters are refused before anything is
  written, and a tile whose cells differ is refused by name;
* the file round-trips.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import splat_tiles
from test_splat_tiles_lod import glb_splats, scene, tiles_of, write_ply

BUDGET = 700
COUNT = 8_000


def collect(ply: Path, work: Path) -> tuple[splat_tiles.Tile, dict[str, Any]]:
    emitted: dict[str, Any] = {}

    def emit(tile: splat_tiles.Tile, gaussians: splat_tiles.Gaussians, keys: Any) -> None:
        emitted[tile.uri] = (gaussians, keys)

    root = splat_tiles.hierarchy(ply, 0.02, BUDGET, work, emit)
    return root, emitted


def shifted(emitted: dict[str, Any], sha: str) -> splat_tiles.ParentOverrides:
    """Every parent moved 1 cm up and made greener, as an "optimised" set."""
    uris, offsets, keys, parts = [], [0], [], []
    for uri, (gaussians, cell_keys) in emitted.items():
        if cell_keys is None:
            continue
        uris.append(uri)
        offsets.append(offsets[-1] + cell_keys.size)
        keys.append(cell_keys)
        parts.append(gaussians)
    cat = {
        name: np.concatenate([getattr(part, name) for part in parts])
        for name in ("xyz", "sh0", "opacity_logit", "log_scales", "quat_xyzw")
    }
    cat["xyz"] = cat["xyz"] + np.float32([0.0, 0.0, 0.01])
    cat["sh0"] = cat["sh0"] + np.float32([0.0, 0.3, 0.0])
    return splat_tiles.ParentOverrides(
        source_sha256=sha,
        opacity_min=0.02,
        tile_gaussians=BUDGET,
        uris=uris,
        offsets=np.asarray(offsets, dtype=np.int64),
        keys=np.concatenate(keys),
        gaussians=splat_tiles.Gaussians(**cat),
    )


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("parents")
    ply = write_ply(root / "scene.ply", scene(COUNT, seed=11))
    plain = root / "plain"
    splat_tiles.convert(ply, plain, 0.0, 0.0, 0.0, tile_gaussians=BUDGET)
    tree, emitted = collect(ply, root / "work")
    overrides = shifted(emitted, splat_tiles.file_sha256(ply))
    path = root / "lod_parents.npz"
    overrides.save(path)
    optimised = root / "optimised"
    stats = splat_tiles.convert(
        ply,
        optimised,
        0.0,
        0.0,
        0.0,
        tile_gaussians=BUDGET,
        parents=splat_tiles.ParentOverrides.load(path),
    )
    return {
        "ply": ply,
        "plain": plain,
        "optimised": optimised,
        "tree": tree,
        "emitted": emitted,
        "overrides": overrides,
        "path": path,
        "stats": stats,
    }


def test_hierarchy_is_the_tree_convert_writes(setup: dict[str, Any]) -> None:
    tileset = json.loads((setup["plain"] / "tileset.json").read_text())
    written = {tile["content"]["uri"]: tile for tile, _, _ in tiles_of(tileset)}
    tree = {tile.uri: tile for tile in setup["tree"].walk()}
    assert set(written) == set(tree) == set(setup["emitted"])
    assert any(tile.children for tile in tree.values())
    for uri, tile in tree.items():
        entry = written[uri]
        assert entry["geometricError"] == tile.geometric_error
        assert entry["extras"]["gaussians"] == tile.count
        assert entry["boundingVolume"]["box"] == splat_tiles._box(tile.low, tile.high)
        _, decoded = glb_splats(setup["plain"] / uri)
        gaussians, keys = setup["emitted"][uri]
        assert (keys is None) == (not tile.children)
        xyz = np.stack([decoded["x"], decoded["y"], decoded["z"]], axis=1)
        # SPZ positions are 1/4096 fixed point.
        assert np.abs(xyz - gaussians.xyz).max() <= 0.5 / 4096 + 1e-6


def test_overrides_replace_parent_content_and_nothing_else(setup: dict[str, Any]) -> None:
    plain = json.loads((setup["plain"] / "tileset.json").read_text())
    optimised = json.loads((setup["optimised"] / "tileset.json").read_text())
    # The tree -- boxes, errors, counts, children -- is the merge's, unchanged.
    assert plain == optimised
    overrides: splat_tiles.ParentOverrides = setup["overrides"]
    assert setup["stats"]["optimised_parent_gaussians"] == int(overrides.offsets[-1]) > 0
    for tile, _, _ in tiles_of(plain):
        uri = tile["content"]["uri"]
        before = (setup["plain"] / uri).read_bytes()
        after = (setup["optimised"] / uri).read_bytes()
        if not tile.get("children"):
            assert before == after, uri
            continue
        assert before != after, uri
        _, a = glb_splats(setup["plain"] / uri)
        _, b = glb_splats(setup["optimised"] / uri)
        assert np.allclose(b["z"] - a["z"], 0.01, atol=1.0 / 4096)
        assert np.allclose(b["x"], a["x"]) and np.allclose(b["y"], a["y"])
        assert np.all(b["f_dc_1"] >= a["f_dc_1"])


def test_the_file_round_trips(setup: dict[str, Any]) -> None:
    loaded = splat_tiles.ParentOverrides.load(setup["path"])
    original: splat_tiles.ParentOverrides = setup["overrides"]
    assert loaded.uris == original.uris
    assert loaded.source_sha256 == original.source_sha256
    assert loaded.tile_gaussians == BUDGET and loaded.opacity_min == pytest.approx(0.02)
    assert np.array_equal(loaded.keys, original.keys)
    assert np.array_equal(loaded.gaussians.xyz, original.gaussians.xyz)


def test_overrides_for_another_packing_are_refused(setup: dict[str, Any], tmp_path: Path) -> None:
    overrides: splat_tiles.ParentOverrides = setup["overrides"]
    ply: Path = setup["ply"]
    with pytest.raises(splat_tiles.ParentOverrideError, match="tiles of 700"):
        splat_tiles.convert(ply, tmp_path / "a", 0, 0, 0, tile_gaussians=500, parents=overrides)
    with pytest.raises(splat_tiles.ParentOverrideError, match="opacity_min"):
        splat_tiles.convert(ply, tmp_path / "b", 0, 0, 0, 0.05, BUDGET, parents=overrides)
    other = write_ply(tmp_path / "other.ply", scene(COUNT, seed=12))
    with pytest.raises(splat_tiles.ParentOverrideError, match="sha256"):
        splat_tiles.convert(other, tmp_path / "c", 0, 0, 0, tile_gaussians=BUDGET, parents=overrides)
    # Refused before a byte was written.
    assert not any((tmp_path / name).exists() for name in ("a", "b", "c"))


def test_a_tile_whose_cells_differ_is_refused_by_name(setup: dict[str, Any], tmp_path: Path) -> None:
    overrides: splat_tiles.ParentOverrides = setup["overrides"]
    keys = overrides.keys.copy()
    keys[0] += 1
    broken = splat_tiles.ParentOverrides(
        overrides.source_sha256,
        overrides.opacity_min,
        overrides.tile_gaussians,
        overrides.uris,
        overrides.offsets,
        keys,
        overrides.gaussians,
    )
    with pytest.raises(splat_tiles.ParentOverrideError, match=overrides.uris[0]):
        splat_tiles.convert(
            setup["ply"], tmp_path / "x", 0, 0, 0, tile_gaussians=BUDGET, parents=broken
        )
