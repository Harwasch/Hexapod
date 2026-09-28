"""The level-of-detail tileset `splat_tiles.convert` writes, checked from the files it writes.

What the hierarchy promises, and what CesiumJS and the Spark viewer lean on:

* refinement is REPLACE: the leaves hold every kept gaussian exactly once, as it was, and
  a parent holds its subtree *merged* (Hierarchical 3DGS moment matching), one gaussian a
  cell -- checked here against a brute-force merge written the way ClusterMerger.cpp is;
* every tile's box holds its own gaussians and its children's boxes (Cesium culls and
  measures distance by them);
* geometric error falls, never rises, from the root to the leaves, and leaves are 0;
* no leaf holds more than the budget, and the parents add at most ~1/7 to the storage;
* `tileset.json` names every tile written and nothing else, and says how many gaussians
  each holds (the viewers budget on it before fetching anything);
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


def glb_splats(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The glTF JSON and the decoded SPZ columns of one tile."""
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
    return gltf, splat_tiles.unpack_spz(chunks[0x004E4942][start : start + view["byteLength"]])


def glb_positions(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    gltf, decoded = glb_splats(path)
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


def columns_table(columns: dict[str, np.ndarray]) -> np.ndarray:
    """Every decoded attribute side by side, one row a gaussian, for multiset comparison."""
    names = ["x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    table = np.stack([columns[name] for name in names], axis=1)
    return table[np.lexsort(table.T[::-1])]


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


def test_it_is_a_hierarchy_with_replace_refinement(packed: tuple[Path, dict, dict]) -> None:
    _, tileset, stats = packed
    assert tileset["asset"]["version"] == "1.1"
    assert tileset["root"]["refine"] == "REPLACE"
    assert len(tileset["root"]["transform"]) == 16
    assert stats["tiles"] == len(tiles_of(tileset)) > 10
    assert stats["depth"] >= 2
    # Only the root is placed; children inherit its frame (Cesium takes the SH frame from
    # the first selected tile and assumes the rest share it).
    assert all("transform" not in tile for tile, depth, _ in tiles_of(tileset) if depth > 0)
    # Nothing below restates refinement: REPLACE is inherited everywhere.
    assert all("refine" not in tile for tile, depth, _ in tiles_of(tileset) if depth > 0)


def test_every_kept_gaussian_is_in_exactly_one_leaf_unchanged(
    packed: tuple[Path, dict, dict],
) -> None:
    """The leaves are the scan: every attribute of every gaussian, once, as packed alone."""
    out, tileset, stats = packed
    columns = stats["columns"]
    assert stats["dropped"] == 0 and stats["gaussians"] == COUNT
    leaves = [tile for tile, _, _ in tiles_of(tileset) if "children" not in tile]
    decoded = [glb_splats(out / tile["content"]["uri"])[1] for tile in leaves]
    found = {name: np.concatenate([part[name] for part in decoded]) for name in decoded[0]}
    # The same gaussians through SPZ on their own: quantisation is per gaussian, so a leaf
    # must decode to exactly these rows, whatever tile they landed in.
    alone = splat_tiles.unpack_spz(
        splat_tiles.pack_spz(
            np.stack([columns["x"], columns["y"], columns["z"]], axis=1),
            np.stack([columns[f"f_dc_{i}"] for i in range(3)], axis=1),
            columns["opacity"],
            np.stack([columns[f"scale_{i}"] for i in range(3)], axis=1),
            np.stack([columns[f"rot_{i}"] for i in (1, 2, 3, 0)], axis=1),
        )
    )
    assert np.array_equal(columns_table(found), columns_table(alone))


def test_the_plan_partitions_its_input() -> None:
    """The same promise one level down: the leaves' runs of Morton positions tile 0..n."""
    columns = scene(9_000, seed=11)
    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1)
    origin = xyz.min(axis=0).astype(np.float64)
    codes = np.sort(splat_tiles.morton_codes(xyz, origin, float(np.ptp(xyz, axis=0).max())))
    root = splat_tiles.plan_tree(codes, 500)
    runs = sorted(run for tile in root.walk() if not tile.children for run in tile.ranges)
    covered = np.concatenate([np.arange(a, b) for a, b in runs])
    assert np.array_equal(covered, np.arange(9_000))
    for tile in root.walk():
        assert tile.size <= 500 or tile.children, "only a parent may be over the budget"
        if tile.children:
            # A parent's allowance: the budget, and an eighth of what its children hold.
            below = sum(child.count for child in tile.children)
            assert tile.count <= min(500, max(1, below // splat_tiles.PARENT_RATIO))
            assert all(tile.level <= child.level for child in tile.children if child.children)


def test_no_tile_exceeds_the_budget(packed: tuple[Path, dict, dict]) -> None:
    out, tileset, _ = packed
    for tile, _, _ in tiles_of(tileset):
        gltf, positions = glb_positions(out / tile["content"]["uri"])
        assert positions.shape[0] <= BUDGET
        assert tile["extras"]["gaussians"] == positions.shape[0]
        assert gltf["accessors"][0]["count"] == positions.shape[0]


def test_parents_add_little_storage(packed: tuple[Path, dict, dict]) -> None:
    """About 1/(8 - 1) at most, the octree's own ratio (PARENT_RATIO); well under 20%."""
    _, tileset, stats = packed
    parents = sum(t["extras"]["gaussians"] for t, _, _ in tiles_of(tileset) if "children" in t)
    assert parents == stats["parent_gaussians"] > 0
    assert stats["storage_overhead"] == pytest.approx(parents / COUNT, abs=1e-4)
    assert stats["storage_overhead"] <= 1 / 7 + 1e-9


def test_boxes_hold_their_gaussians_and_their_children(packed: tuple[Path, dict, dict]) -> None:
    out, tileset, _ = packed
    for tile, _, parent in tiles_of(tileset):
        low, high = box_bounds(tile)
        _, positions = glb_positions(out / tile["content"]["uri"])
        # A merged centre may round to the SPZ grid outside a box built from float centres.
        slack = 1.0 / GRID
        assert np.all(positions >= low - slack) and np.all(positions <= high + slack)
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


def test_the_error_is_derived_from_the_data_not_fixed(tmp_path: Path) -> None:
    """The same scene at half the size, gaussians and all: every error halves with it."""
    columns = scene(8_000, seed=5)
    half = dict(columns)
    for axis in "xyz":
        half[axis] = columns[axis] / 2
    for i in range(3):
        half[f"scale_{i}"] = columns[f"scale_{i}"] - np.float32(np.log(2))
    splat_tiles.convert(write_ply(tmp_path / "a.ply", columns), tmp_path / "a", 0, 0, 0, 0.02, 400)
    splat_tiles.convert(write_ply(tmp_path / "b.ply", half), tmp_path / "b", 0, 0, 0, 0.02, 400)
    full = json.loads((tmp_path / "a" / "tileset.json").read_text())
    halved = json.loads((tmp_path / "b" / "tileset.json").read_text())
    assert halved["root"]["geometricError"] == pytest.approx(
        full["root"]["geometricError"] / 2, rel=1e-3
    )


def test_tileset_json_names_exactly_the_tiles_written(packed: tuple[Path, dict, dict]) -> None:
    out, tileset, _ = packed
    uris = [tile["content"]["uri"] for tile, _, _ in tiles_of(tileset)]
    assert len(set(uris)) == len(uris)
    assert tileset["root"]["content"]["uri"] == "splat.glb"
    written = sorted(path.name for path in out.iterdir())
    assert written == sorted([*uris, "tileset.json"]), "and no working files are left behind"


def test_a_small_scan_is_one_tile_in_ply_order(tmp_path: Path) -> None:
    columns = scene(3_000)
    ply = write_ply(tmp_path / "small.ply", columns)
    stats = splat_tiles.convert(ply, tmp_path / "splat", 0.0, 0.0, 0.0)
    assert stats["tiles"] == 1 and stats["parent_gaussians"] == 0
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


REPO_ROOT = Path(__file__).resolve().parents[3]
#: The level-of-detail fixture apps/web's e2e/splatLod.spec.ts renders in CesiumJS: the
#: committed synthetic tree, packed at 1,500 gaussians a tile. Regenerated by:
#:   uv run python splat_tiles.py ../../data/tiles/synthetic-tree/source/splat.ply \
#:     ../../data/tiles/synthetic-tree-lod --lat 28.0389 --lon -82.6966 --tile-gaussians 1500
LOD_FIXTURE = REPO_ROOT / "data" / "tiles" / "synthetic-tree-lod"


def test_committed_lod_fixture_matches_the_packer(tmp_path: Path) -> None:
    """Not hand-edited, not stale: the packer writes it again byte for byte."""
    source = REPO_ROOT / "data" / "tiles" / "synthetic-tree" / "source" / "splat.ply"
    splat_tiles.convert(source, tmp_path / "lod", 28.0389, -82.6966, 0.0, 0.02, 1500)
    fresh = {path.name: path.read_bytes() for path in (tmp_path / "lod").iterdir()}
    # rig.json and its motion.json sit beside the tiles (renderConfig.rigUrl resolves relative
    # to the tileset) but are not the packer's: rig_tiles.py writes them, test_rig_tiles.py pins
    # the rig.
    committed = {
        path.name: path.read_bytes()
        for path in LOD_FIXTURE.iterdir()
        if path.name not in {"rig.json", "motion.json"}
    }
    assert sorted(fresh) == sorted(committed)
    assert fresh == committed
    tileset = json.loads(fresh["tileset.json"])
    assert tileset["root"]["refine"] == "REPLACE" and len(tiles_of(tileset)) > 10


def test_the_disk_sort_is_the_in_memory_sort(tmp_path: Path, monkeypatch) -> None:
    """Several buckets or one, the same files: bucketing only bounds memory."""
    ply = write_ply(tmp_path / "scene.ply", scene(6_000, seed=9))
    splat_tiles.convert(ply, tmp_path / "one", 0.0, 0.0, 0.0, tile_gaussians=500)
    monkeypatch.setattr(splat_tiles, "BUCKET_RECORDS", 700)
    monkeypatch.setattr(splat_tiles, "CHUNK_BYTES", 56 * 999)
    splat_tiles.convert(ply, tmp_path / "many", 0.0, 0.0, 0.0, tile_gaussians=500)
    one = {path.name: path.read_bytes() for path in (tmp_path / "one").iterdir()}
    many = {path.name: path.read_bytes() for path in (tmp_path / "many").iterdir()}
    assert one == many


def test_coincident_gaussians_end_in_an_over_budget_leaf_not_a_loop() -> None:
    """The one case the budget cannot be kept: more copies of one point than it allows."""
    codes = np.zeros(300, dtype=np.uint64)
    root = splat_tiles.plan_tree(codes, 50)
    runs = [run for tile in root.walk() if not tile.children for run in tile.ranges]
    assert sum(b - a for a, b in runs) == 300


# ----------------------------------------------------------------------- the merge itself


def brute_force_merge(
    xyz: np.ndarray, sh0: np.ndarray, opacity: np.ndarray, log_scales: np.ndarray, quat: np.ndarray
) -> dict[str, np.ndarray]:
    """One cell merged the way gaussian-hierarchy ClusterMerger.cpp `mergeRec` does it.

    Deliberately naive -- one gaussian at a time, normalised weights, a 3x3 matrix each --
    so it shares nothing with the vectorised code it checks but the paper's equations.
    """
    scales = np.exp(np.clip(log_scales.astype(np.float64), *splat_tiles.SPZ_LOG_SCALE_RANGE))
    weights = []
    covs = []
    for s, q, o in zip(scales, quat.astype(np.float64), opacity, strict=True):
        w, x, y, z = q / np.linalg.norm(q)
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        covs.append(rot @ np.diag(s * s) @ rot.T)
        # ellipseSurface: s0 s1 + s0 s2 + s1 s2, times opacity.
        weights.append(
            splat_tiles.sigmoid(np.float64(o)) * (s[0] * s[1] + s[0] * s[2] + s[1] * s[2])
        )
    total = float(sum(weights))
    a = np.array(weights) / total
    mean = sum(ai * p for ai, p in zip(a, xyz.astype(np.float64), strict=True))
    colour = sum(ai * c for ai, c in zip(a, sh0.astype(np.float64), strict=True))
    cov = sum(
        ai * (c + np.outer(p - mean, p - mean))
        for ai, c, p in zip(a, covs, xyz.astype(np.float64), strict=True)
    )
    return {"mean": mean, "colour": colour, "cov": cov, "weight": np.float64(total)}


def records_of(columns: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    out = np.empty(rows.size, dtype=splat_tiles.RECORD)
    out["row"] = rows
    out["xyz"] = np.stack([columns[a][rows] for a in "xyz"], axis=1)
    out["f_dc"] = np.stack([columns[f"f_dc_{i}"][rows] for i in range(3)], axis=1)
    out["opacity"] = columns["opacity"][rows]
    out["scale"] = np.stack([columns[f"scale_{i}"][rows] for i in range(3)], axis=1)
    out["rot"] = np.stack([columns[f"rot_{i}"][rows] for i in range(4)], axis=1)
    return out


def test_merging_merged_cells_is_merging_their_gaussians() -> None:
    """Eqs. 3-4 hierarchically (fine cells, then coarsened, as parents are built) against
    the brute-force merge of every gaussian in each coarse cell at once."""
    columns = scene(4_000, seed=21)
    xyz = np.stack([columns[a] for a in "xyz"], axis=1)
    origin = xyz.min(axis=0).astype(np.float64)
    edge = float(np.ptp(xyz, axis=0).max())
    codes = splat_tiles.morton_codes(xyz, origin, edge)
    order = np.argsort(codes, kind="stable")
    codes = codes[order]
    shift = 3 * (splat_tiles.GRID_BITS - 5)
    fine = splat_tiles.merge_cells(
        splat_tiles.gaussian_moments(records_of(columns, order), (codes >> np.uint64(shift)))
    )
    coarse = splat_tiles.coarsen(fine, 2)  # level 5 -> level 3
    keys = (codes >> np.uint64(3 * (splat_tiles.GRID_BITS - 3))).astype(np.int64)
    assert np.array_equal(coarse.key, np.unique(keys))
    for index, key in enumerate(coarse.key[:40]):
        rows = order[keys == key]
        expected = brute_force_merge(
            xyz[rows],
            np.stack([columns[f"f_dc_{i}"][rows] for i in range(3)], axis=1),
            columns["opacity"][rows],
            np.stack([columns[f"scale_{i}"][rows] for i in range(3)], axis=1),
            np.stack([columns[f"rot_{i}"][rows] for i in range(4)], axis=1),
        )
        c = coarse.cov[index]
        got = np.array([[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]])
        assert coarse.weight[index] == pytest.approx(expected["weight"], rel=1e-9)
        assert np.allclose(coarse.mean[index], expected["mean"], rtol=0, atol=1e-9)
        assert np.allclose(coarse.colour[index], expected["colour"], rtol=0, atol=1e-9)
        assert np.allclose(got, expected["cov"], rtol=1e-9, atol=1e-12)


def covariance_of(log_scales: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    rot = splat_tiles._rotation_matrices(quat_xyzw[:, [3, 0, 1, 2]])
    axes = rot * np.exp(log_scales.astype(np.float64))[:, None, :]
    return axes @ axes.transpose(0, 2, 1)


def test_a_merged_cell_becomes_the_gaussian_its_moments_describe() -> None:
    """Axes and scales from the eigendecomposition, opacity from H3DGS's falloff."""
    rng = np.random.default_rng(4)
    quat = rng.normal(size=(1, 4))
    one = {
        "xyz": np.array([[1.0, 2.0, 3.0]]),
        "f_dc": np.array([[0.1, 0.2, 0.3]]),
        "opacity": np.array([np.log(0.6 / 0.4)]),
        "scale": np.log(np.array([[0.02, 0.05, 0.3]])),
        "rot": quat,
    }
    record = np.empty(1, dtype=splat_tiles.RECORD)
    for name, value in one.items():
        record[name] = value
    alone = splat_tiles.to_gaussians(
        splat_tiles.gaussian_moments(record, np.zeros(1, dtype=np.int64))
    )
    # A cell of one is that gaussian: same place, shape, colour and opacity (falloff o S / S).
    assert np.allclose(alone.xyz, one["xyz"], atol=1e-6)
    assert np.allclose(alone.sh0, one["f_dc"], atol=1e-6)
    assert splat_tiles.sigmoid(alone.opacity_logit[0]) == pytest.approx(0.6, rel=1e-5)
    wanted = covariance_of(one["scale"], quat[:, [1, 2, 3, 0]])
    assert np.allclose(covariance_of(alone.log_scales, alone.quat_xyzw), wanted, rtol=1e-5)
    # Two coincident copies: the same covariance, twice the weight, so falloff 1.2 --
    # cut to 0.99 with the surface widened by 1.2 / 0.99 (opacity x surface is conserved).
    two = splat_tiles.to_gaussians(
        splat_tiles.merge_cells(
            splat_tiles.gaussian_moments(np.r_[record, record], np.zeros(2, dtype=np.int64))
        )
    )
    assert splat_tiles.sigmoid(two.opacity_logit[0]) == pytest.approx(0.99, rel=1e-5)
    widened = covariance_of(two.log_scales, two.quat_xyzw)
    assert np.allclose(widened, wanted * (1.2 / 0.99), rtol=1e-5)


def test_quaternions_round_trip_through_rotation_matrices() -> None:
    rng = np.random.default_rng(8)
    quat = rng.normal(size=(2_000, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    back = splat_tiles._quaternions(splat_tiles._rotation_matrices(quat))
    # q and -q are the same rotation.
    assert np.allclose(np.abs(np.sum(back * quat, axis=1)), 1.0, atol=1e-9)


def test_a_parent_tile_is_its_subtree_merged(packed: tuple[Path, dict, dict]) -> None:
    """End to end: the root's decoded gaussians against a brute-force merge of the scan.

    Through SPZ, so within its quantisation: positions to half a 1/4096 m step, alpha to a
    byte, colour to a byte, covariance to the 1/16 log-scale and 8-bit rotation steps.
    """
    out, tileset, stats = packed
    columns = stats["columns"]
    xyz = np.stack([columns[a] for a in "xyz"], axis=1)
    origin = xyz.min(axis=0).astype(np.float64)
    edge = max(float((xyz.max(axis=0).astype(np.float64) - origin).max()), 1e-3)
    codes = splat_tiles.morton_codes(xyz, origin, edge)
    order = np.argsort(codes, kind="stable")
    root_plan = splat_tiles.plan_tree(codes[order], BUDGET)
    keys = codes >> np.uint64(3 * (splat_tiles.GRID_BITS - root_plan.level))
    cells = np.unique(keys)
    _, decoded = glb_splats(out / "splat.glb")
    assert decoded["x"].size == cells.size == tileset["root"]["extras"]["gaussians"]
    decoded_cov = covariance_of(
        np.stack([decoded[f"scale_{i}"] for i in range(3)], axis=1),
        np.stack([decoded[f"rot_{i}"] for i in (1, 2, 3, 0)], axis=1),
    )
    for index, key in enumerate(cells):
        rows = np.flatnonzero(keys == key)
        expected = brute_force_merge(
            xyz[rows],
            np.stack([columns[f"f_dc_{i}"][rows] for i in range(3)], axis=1),
            columns["opacity"][rows],
            np.stack([columns[f"scale_{i}"][rows] for i in range(3)], axis=1),
            np.stack([columns[f"rot_{i}"][rows] for i in range(4)], axis=1),
        )
        position = np.array([decoded[a][index] for a in "xyz"])
        assert np.all(np.abs(position - expected["mean"]) <= 0.5 / GRID + 1e-6)
        colour = np.array([decoded[f"f_dc_{i}"][index] for i in range(3)])
        assert np.all(np.abs(colour - expected["colour"]) <= 1.0 / (0.15 * 255) + 1e-6)
        values = np.linalg.eigvalsh(expected["cov"])
        falloff = expected["weight"] / (
            np.sqrt(values[0] * values[1]) + np.sqrt(values[0] * values[2])
            + np.sqrt(values[1] * values[2])
        )  # fmt: skip
        widen = min(max(falloff / splat_tiles.MERGED_OPACITY_MAX, 1.0), 4.0)
        alpha = min(falloff / widen, splat_tiles.MERGED_OPACITY_MAX)
        assert splat_tiles.sigmoid(decoded["opacity"][index]) == pytest.approx(alpha, abs=1 / 255)
        wanted = expected["cov"] * widen
        error = np.linalg.norm(decoded_cov[index] - wanted) / np.linalg.norm(wanted)
        assert error < 0.25, f"cell {key}: covariance off by {error:.0%}"
