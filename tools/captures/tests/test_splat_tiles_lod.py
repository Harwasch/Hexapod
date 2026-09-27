"""The level-of-detail tileset `splat_tiles.convert` writes, checked from the files it writes.

What the hierarchy promises, and what CesiumJS and the Spark viewer lean on:

* nothing that passes the filters is lost or doubled -- refinement is ADD, so the tiles'
  contents partition the kept gaussians and the full-detail view is exactly the input;
* every tile's box holds its own gaussians and its children's boxes (Cesium culls and
  measures distance by them);
* geometric error falls, never rises, from the root to the leaves, and leaves are 0;
* no tile holds more than the budget;
* `tileset.json` names every tile written and nothing else, and says how many gaussians
  each holds (the viewer budgets on it before fetching anything);
* a scan within one tile's budget is the single-tile layout it always was.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import splat_tiles

GRID = 1 << splat_tiles.SPZ_FRACTIONAL_BITS
PROPERTIES = [
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
]


def scene(count: int, seed: int = 3) -> dict[str, np.ndarray]:
    """A lumpy scene: a ground patch, a dense ball and a sparse haze, unevenly populated.

    Positions are snapped to SPZ's 1/4096 m grid so a decoded tile compares exactly.
    """
    rng = np.random.default_rng(seed)
    ground = np.c_[rng.uniform(-8, 8, count // 2), rng.uniform(-8, 8, count // 2)]
    ground = np.c_[ground, rng.normal(0.0, 0.02, count // 2)]
    ball = rng.normal(size=(count // 3, 3))
    ball = ball / np.linalg.norm(ball, axis=1, keepdims=True) * 1.5 + [2.0, -1.0, 2.0]
    haze = rng.uniform(-6, 6, size=(count - ground.shape[0] - ball.shape[0], 3)) + [0, 0, 6]
    xyz = np.round(np.r_[ground, ball, haze] * GRID) / GRID
    quat = rng.normal(size=(count, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    columns = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    columns.update({f"f_dc_{i}": rng.normal(0, 0.5, count) for i in range(3)})
    # Logits from 0.12 to 0.98 opaque: all above the default opacity_min.
    columns["opacity"] = rng.uniform(-2.0, 4.0, count)
    columns.update({f"scale_{i}": rng.normal(-4.0, 0.6, count) for i in range(3)})
    columns.update({f"rot_{i}": quat[:, i] for i in range(4)})
    return {name: np.asarray(value, dtype=np.float32) for name, value in columns.items()}


def write_ply(path: Path, columns: dict[str, np.ndarray]) -> Path:
    count = columns["x"].shape[0]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header += [f"property float {name}" for name in PROPERTIES]
    header.append("end_header")
    rows = np.empty(count, dtype=[(name, "<f4") for name in PROPERTIES])
    for name in PROPERTIES:
        rows[name] = columns[name]
    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + rows.tobytes())
    return path


def glb_positions(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    """The glTF JSON and the decoded positions of one tile."""
    raw = path.read_bytes()
    assert raw[:4] == b"glTF"
    offset, chunks = 12, {}
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        chunks[kind] = raw[offset + 8 : offset + 8 + length]
        offset += 8 + length
    gltf = json.loads(chunks[0x4E4F534A])
    view = gltf["bufferViews"][0]
    start = view.get("byteOffset", 0)
    spz = chunks[0x004E4942][start : start + view["byteLength"]]
    decoded = splat_tiles.unpack_spz(spz)
    return gltf, np.stack([decoded["x"], decoded["y"], decoded["z"]], axis=1)


def tiles_of(tileset: dict[str, Any]) -> list[tuple[dict[str, Any], int, dict[str, Any] | None]]:
    """(tile, depth, parent) for every tile, parents first."""
    out: list[tuple[dict[str, Any], int, dict[str, Any] | None]] = []
    stack: list[tuple[dict[str, Any], int, dict[str, Any] | None]] = [(tileset["root"], 0, None)]
    while stack:
        tile, depth, parent = stack.pop(0)
        out.append((tile, depth, parent))
        stack.extend((child, depth + 1, tile) for child in tile.get("children", []))
    return out


def box_bounds(tile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    box = tile["boundingVolume"]["box"]
    centre = np.array(box[:3])
    # Axis-aligned by construction: the half-axes are diagonal.
    assert box[4:7] == [0, 0, 0] and box[8:9] == [0] and box[10] == 0
    half = np.array([box[3], box[7], box[11]])
    return centre - half, centre + half


BUDGET = 700
COUNT = 20_000


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any], dict]:
    root = tmp_path_factory.mktemp("lod")
    columns = scene(COUNT)
    ply = write_ply(root / "scene.ply", columns)
    out = root / "splat"
    stats = splat_tiles.convert(ply, out, 46.84, -91.99, 0.0, tile_gaussians=BUDGET)
    return out, json.loads((out / "tileset.json").read_text()), stats | {"columns": columns}


def test_it_is_a_hierarchy_with_add_refinement(packed: tuple[Path, dict, dict]) -> None:
    _, tileset, stats = packed
    assert tileset["asset"]["version"] == "1.1"
    assert tileset["root"]["refine"] == "ADD"
    assert len(tileset["root"]["transform"]) == 16
    assert stats["tiles"] == len(tiles_of(tileset)) > 10
    assert stats["depth"] >= 2
    # Only the root is placed; children inherit its frame (Cesium takes the SH frame from
    # the first selected tile and assumes the rest share it).
    assert all("transform" not in tile for tile, depth, _ in tiles_of(tileset) if depth > 0)


def test_every_kept_gaussian_is_in_exactly_one_tile(packed: tuple[Path, dict, dict]) -> None:
    out, tileset, stats = packed
    columns = stats["columns"]
    expected = np.stack([columns["x"], columns["y"], columns["z"]], axis=1)
    assert stats["dropped"] == 0 and stats["gaussians"] == COUNT
    found = np.concatenate(
        [glb_positions(out / tile["content"]["uri"])[1] for tile, _, _ in tiles_of(tileset)]
    )
    assert found.shape == expected.shape
    # As multisets: sort both by row. Exact, because the scene is on the SPZ grid.
    order = np.lexsort(found.T[::-1])
    target = np.lexsort(expected.T[::-1])
    assert np.array_equal(found[order], expected[target])


def test_the_hierarchy_partitions_its_input() -> None:
    """The same promise one level down, where the tile members are PLY row numbers."""
    columns = scene(9_000, seed=11)
    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1)
    scales = np.stack([columns[f"scale_{i}"] for i in range(3)], axis=1)
    root = splat_tiles.build_hierarchy(xyz, scales, splat_tiles.sigmoid(columns["opacity"]), 500)
    members = np.concatenate([tile.members for tile in root.walk()])
    assert np.array_equal(np.sort(members), np.arange(9_000))
    for tile in root.walk():
        assert np.all(np.diff(tile.members) > 0), "PLY order is kept inside a tile"


def test_no_tile_exceeds_the_budget(packed: tuple[Path, dict, dict]) -> None:
    out, tileset, _ = packed
    for tile, _, _ in tiles_of(tileset):
        gltf, positions = glb_positions(out / tile["content"]["uri"])
        assert positions.shape[0] <= BUDGET
        assert tile["extras"]["gaussians"] == positions.shape[0]
        assert gltf["accessors"][0]["count"] == positions.shape[0]


def test_boxes_hold_their_gaussians_and_their_children(packed: tuple[Path, dict, dict]) -> None:
    out, tileset, _ = packed
    for tile, _, parent in tiles_of(tileset):
        low, high = box_bounds(tile)
        _, positions = glb_positions(out / tile["content"]["uri"])
        assert np.all(positions >= low - 1e-9) and np.all(positions <= high + 1e-9)
        if parent is not None:
            parent_low, parent_high = box_bounds(parent)
            assert np.all(low >= parent_low - 1e-9) and np.all(high <= parent_high + 1e-9)


def test_geometric_error_falls_with_depth(packed: tuple[Path, dict, dict]) -> None:
    _, tileset, _ = packed
    for tile, _, parent in tiles_of(tileset):
        if "children" in tile:
            assert tile["geometricError"] > 0
        else:
            assert tile["geometricError"] == 0, "a leaf is the data at full resolution"
        if parent is not None:
            assert tile["geometricError"] <= parent["geometricError"]
    assert tileset["geometricError"] > tileset["root"]["geometricError"]
    by_depth: dict[int, list[float]] = {}
    for tile, depth, _ in tiles_of(tileset):
        if "children" in tile:
            by_depth.setdefault(depth, []).append(tile["geometricError"])
    levels = [max(by_depth[depth]) for depth in sorted(by_depth)]
    assert levels == sorted(levels, reverse=True) and levels[0] > levels[-1]


def test_the_error_is_derived_from_the_data_not_fixed() -> None:
    """Half the scene's size at the same density: the root's error halves with it."""
    columns = scene(8_000, seed=5)
    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1)
    scales = np.stack([columns[f"scale_{i}"] for i in range(3)], axis=1)
    opacity = splat_tiles.sigmoid(columns["opacity"])
    full = splat_tiles.build_hierarchy(xyz, scales, opacity, 400)
    half = splat_tiles.build_hierarchy(xyz / 2, scales, opacity, 400)
    assert half.geometric_error == pytest.approx(full.geometric_error / 2)
    coarser = splat_tiles.build_hierarchy(xyz, scales, opacity, 100)
    assert coarser.geometric_error > full.geometric_error


def test_tileset_json_names_exactly_the_tiles_written(packed: tuple[Path, dict, dict]) -> None:
    out, tileset, _ = packed
    uris = [tile["content"]["uri"] for tile, _, _ in tiles_of(tileset)]
    assert len(set(uris)) == len(uris)
    assert tileset["root"]["content"]["uri"] == "splat.glb"
    written = sorted(path.name for path in out.iterdir())
    assert written == sorted([*uris, "tileset.json"])


def test_a_small_scan_is_one_tile_in_ply_order(tmp_path: Path) -> None:
    columns = scene(3_000)
    ply = write_ply(tmp_path / "small.ply", columns)
    stats = splat_tiles.convert(ply, tmp_path / "splat", 0.0, 0.0, 0.0)
    assert stats["tiles"] == 1
    assert sorted(path.name for path in (tmp_path / "splat").iterdir()) == [
        "splat.glb",
        "tileset.json",
    ]
    tileset = json.loads((tmp_path / "splat" / "tileset.json").read_text())
    assert "children" not in tileset["root"]
    assert tileset["root"]["geometricError"] == 0
    _, positions = glb_positions(tmp_path / "splat" / "splat.glb")
    expected = np.stack([columns["x"], columns["y"], columns["z"]], axis=1)
    assert np.array_equal(positions, expected)


def test_no_budget_means_one_tile(tmp_path: Path) -> None:
    """What the Living Survey tools ask for: everything in one tile, whatever the count."""
    ply = write_ply(tmp_path / "scene.ply", scene(5_000))
    stats = splat_tiles.convert(ply, tmp_path / "splat", 0.0, 0.0, 0.0, tile_gaussians=None)
    assert stats["tiles"] == 1 and stats["gaussians"] == 5_000


def test_the_output_is_byte_stable(tmp_path: Path) -> None:
    ply = write_ply(tmp_path / "scene.ply", scene(6_000))
    splat_tiles.convert(ply, tmp_path / "a", 1.0, 2.0, 3.0, tile_gaussians=500)
    splat_tiles.convert(ply, tmp_path / "b", 1.0, 2.0, 3.0, tile_gaussians=500)
    first = {path.name: path.read_bytes() for path in (tmp_path / "a").iterdir()}
    second = {path.name: path.read_bytes() for path in (tmp_path / "b").iterdir()}
    assert first == second and len(first) > 3


def test_coincident_gaussians_end_in_an_over_budget_leaf_not_a_loop() -> None:
    """The one case the budget cannot be kept: more copies of one point than it allows."""
    count = 300
    xyz = np.zeros((count, 3), dtype=np.float32)
    root = splat_tiles.build_hierarchy(
        xyz, np.full((count, 3), -4.0, np.float32), np.full(count, 0.9), 50
    )
    members = np.concatenate([tile.members for tile in root.walk()])
    assert np.array_equal(np.sort(members), np.arange(count))
