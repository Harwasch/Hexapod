"""`lod_maths`: CesiumJS's REPLACE cut on screen-space error, as `experiments/lod_compare.py`
draws it, on a hand-made tree and on a packed tileset read back from `tileset.json`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_placement import _splat_columns

import gaussians
import lod_maths
import splat_tiles

BUDGET = 400


def _tile(uri: str, low: list[float], high: list[float], error: float, kids: list[Any]) -> Any:
    return SimpleNamespace(
        uri=uri, low=np.array(low), high=np.array(high), geometric_error=error, children=kids
    )


def _walk(tile: Any) -> list[Any]:
    out = [tile]
    for child in tile.children:
        out.extend(_walk(child))
    return out


def small_tree() -> lod_maths.Tree:
    """root (error 1) -> [a (error 0.25) -> [a0, a1], b]; boxes along x."""
    a0 = _tile("a0", [0, 0, 0], [1, 1, 1], 0.0, [])
    a1 = _tile("a1", [1, 0, 0], [2, 1, 1], 0.0, [])
    a = _tile("a", [0, 0, 0], [2, 1, 1], 0.25, [a0, a1])
    b = _tile("b", [2, 0, 0], [4, 1, 1], 0.0, [])
    root = _tile("root", [0, 0, 0], [4, 1, 1], 1.0, [a, b])
    tiles = _walk(root)
    return lod_maths.tree_from(tiles, [10, 5, 30, 30, 40])


def test_the_tree_lays_leaves_first_then_parents() -> None:
    tree = small_tree()
    assert tree.uris == ("root", "a", "a0", "a1", "b")
    assert tree.parents == (0, 1) and tree.leaves == (2, 3, 4)
    # Leaves 30 + 30 + 40 from row 0, then the parents' 10 + 5.
    assert tree.rows.tolist() == [[100, 110], [110, 115], [0, 30], [30, 60], [60, 100]]


def test_the_cut_is_cesiums_replace_traversal() -> None:
    """Refine a tile when geometricError * f / distance-to-box > tau; REPLACE by all
    children, each of which decides again. Checked against the rule written out."""
    tree = small_tree()
    focal = 100.0
    eye = np.array([-10.0, 0.5, 0.5])  # 10 from the root's and a's boxes
    # root: 1 * 100 / 10 = 10 px; a: 0.25 * 100 / 10 = 2.5 px.
    assert lod_maths.screen_space_error(tree, 0, eye, focal) == pytest.approx(10.0)
    assert lod_maths.cut(tree, eye, focal, 16.0) == [0]
    assert lod_maths.cut(tree, eye, focal, 10.0) == [0]  # not over tau: drawn
    assert lod_maths.cut(tree, eye, focal, 5.0) == [1, 4]
    assert lod_maths.cut(tree, eye, focal, 2.0) == [2, 3, 4]
    # A camera inside a box is infinite error: always refined.
    inside = np.array([1.0, 0.5, 0.5])
    assert lod_maths.cut(tree, inside, focal, 1e9) == [2, 3, 4]


def test_the_packed_tileset_and_the_in_memory_tree_cut_alike(tmp_path: Path) -> None:
    """`experiments/lod_compare.py` reads the tree back from `tileset.json`;
    `splat_tiles.hierarchy` builds it in memory. Same tiles, same boxes, same errors --
    so the same cut from anywhere."""
    from experiments.lod_compare import Tileset, summarise

    ply = tmp_path / "scene.ply"
    gaussians.write_ply(ply, _splat_columns(3000, seed=8))
    splat_tiles.convert(ply, tmp_path / "splat", 0, 0, 0, 0.02, BUDGET)
    packed = Tileset(tmp_path / "splat")

    def emit(tile: Any, gaussians_: Any, keys: Any) -> None:
        pass

    walked = splat_tiles.hierarchy(ply, 0.02, BUDGET, tmp_path, emit).walk()
    tree = lod_maths.tree_from(walked, [tile.count for tile in walked])
    assert sorted(packed.uris) == sorted(tree.uris)
    order = [packed.uris.index(uri) for uri in tree.uris]
    np.testing.assert_allclose(packed.tree.error[order], tree.error)
    np.testing.assert_allclose(packed.tree.low[order], tree.low, atol=1e-9)
    rng = np.random.default_rng(3)
    for _ in range(50):
        eye = rng.normal(size=3) * 20
        tau = float(np.exp(rng.uniform(np.log(3.0), np.log(64.0))))
        a = {packed.uris[t] for t in lod_maths.cut(packed.tree, eye, 500.0, tau)}
        b = {tree.uris[t] for t in lod_maths.cut(tree, eye, 500.0, tau)}
        assert a == b
    rows = [
        {"test": "root-at-switch", "tau": 16.0, "tileset": name, "lpips": v, "psnr": 20.0,
         "ssim": 0.9, "holes": 0.1}
        for name, v in (("a", 0.3), ("a", 0.32), ("b", 0.2))
    ]  # fmt: skip
    lines = summarise(rows)
    assert len(lines) == 2 and "lpips 0.310" in lines[0] and " b " in lines[1]


def test_orbit_eye_is_at_the_asked_distance_from_the_box() -> None:
    rng = np.random.default_rng(5)
    low, high = np.array([-1.0, -2.0, 0.0]), np.array([3.0, 1.0, 0.5])
    target = 0.5 * (low + high)
    for _ in range(20):
        direction = rng.normal(size=3)
        distance = float(rng.uniform(0.01, 500.0))
        eye = lod_maths.orbit_eye(target, direction, low, high, distance)
        got = lod_maths.box_distance(eye, low, high)
        assert abs(got - distance) <= 1e-9 * max(1.0, distance)
        # On the ray from the target, in the asked direction.
        along = (eye - target) / np.linalg.norm(eye - target)
        np.testing.assert_allclose(along, direction / np.linalg.norm(direction), atol=1e-12)


def test_look_at_is_a_rigid_opencv_camera() -> None:
    eye, target = np.array([4.0, -3.0, 2.0]), np.array([0.5, 0.5, 0.0])
    view = lod_maths.look_at(eye, target)
    rotation = view[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    # The eye maps to the camera's origin; the target lies straight ahead (+z).
    np.testing.assert_allclose(view @ np.append(eye, 1.0), [0, 0, 0, 1], atol=1e-12)
    ahead = view @ np.append(target, 1.0)
    np.testing.assert_allclose(ahead[:2], [0.0, 0.0], atol=1e-12)
    assert ahead[2] > 0
    # World up is camera up, i.e. -y (OpenCV's y points down).
    assert (rotation @ np.array([0.0, 0.0, 1.0]))[1] < 0
    # Looking straight down still gives a camera.
    down = lod_maths.look_at(np.array([0.0, 0.0, 10.0]), np.zeros(3))
    np.testing.assert_allclose(down[:3, :3] @ down[:3, :3].T, np.eye(3), atol=1e-12)


def test_a_far_tileset_is_read_in_the_root_frame(tmp_path: Path) -> None:
    """A site wider than SPZ's +/-2 km gives its far tiles translation-only transforms
    (ADR 0011). `lod_compare.Tileset` composes them, so its boxes and positions are the
    in-memory tree's (site frame) and its cut is the same from anywhere."""
    from experiments.lod_compare import Tileset

    columns = _splat_columns(3000, seed=8)
    for axis in "xy":
        columns[axis] = np.ascontiguousarray(columns[axis] * np.float32(2500.0))
    ply = tmp_path / "far.ply"
    gaussians.write_ply(ply, columns)
    splat_tiles.convert(ply, tmp_path / "splat", 0, 0, 0, 0.02, BUDGET)
    packed = Tileset(tmp_path / "splat")
    shifted = [uri for uri in packed.uris if packed.offsets[uri].any()]
    assert shifted, "the scene is meant to need tile origins"

    def emit(tile: Any, gaussians_: Any, keys: Any) -> None:
        pass

    walked = splat_tiles.hierarchy(ply, 0.02, BUDGET, tmp_path / "work", emit).walk()
    tree = lod_maths.tree_from(walked, [tile.count for tile in walked])
    order = [packed.uris.index(uri) for uri in tree.uris]
    np.testing.assert_allclose(packed.tree.low[order], tree.low, atol=1e-6)
    np.testing.assert_allclose(packed.tree.high[order], tree.high, atol=1e-6)
    # Each tile's decoded gaussians sit in its (composed) box.
    for uri in shifted:
        t = packed.uris.index(uri)
        xyz = packed.content([t])["xyz"].astype(np.float64)
        assert np.all(xyz >= packed.tree.low[t] - 1e-3) and np.all(
            xyz <= packed.tree.high[t] + 1e-3
        )
    rng = np.random.default_rng(4)
    for _ in range(30):
        eye = rng.normal(size=3) * 4000
        a = {packed.uris[t] for t in lod_maths.cut(packed.tree, eye, 500.0, 16.0)}
        b = {tree.uris[t] for t in lod_maths.cut(tree, eye, 500.0, 16.0)}
        assert a == b
