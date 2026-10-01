"""`view_cones`: from where each part of a scan was seen, estimated from its own detail.

Three capture shapes, each built from gaussians whose size is what a camera at a known place
would have fitted, check that the estimate follows the observers and not the scene:

* walked **inside** a clearing (fine ground, a ring of trees and a canopy far coarser): the
  clearing is seen from everywhere, the ring from inside it, the canopy from below;
* an **orbit** of one subject: the subject and the ground beside it are as fine as each
  other, and nothing gets a cone;
* a **drone grid** over even ground: nothing gets a cone.

And the format: octahedral axes within the quantisation, the bytes the same however the
scan is windowed, a point outside the grid taking its edge cell, `build_view_cones`
declaring itself on a tileset.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

import view_cones as vc


def _cloud(
    rng: np.random.Generator, n: int, low: list[float], high: list[float], size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """`n` gaussians in a box, each with a middle axis of about `size_m`."""
    xyz = rng.uniform(low, high, size=(n, 3))
    middle = np.log(size_m) + rng.normal(0.0, 0.15, size=n)
    return xyz, middle


def _ring(
    rng: np.random.Generator, n: int, r0: float, r1: float, z0: float, z1: float, size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(0, 2 * math.pi, n)
    radius = rng.uniform(r0, r1, n)
    xyz = np.stack([radius * np.cos(angle), radius * np.sin(angle), rng.uniform(z0, z1, n)], 1)
    return xyz, np.log(size_m) + rng.normal(0.0, 0.15, size=n)


def _grid(parts: list[tuple[np.ndarray, np.ndarray]], windows: int = 3) -> vc.ConeGrid:
    xyz = np.concatenate([p[0] for p in parts])
    middle = np.concatenate([p[1] for p in parts])

    def chunks() -> list[tuple[np.ndarray, np.ndarray]]:
        bounds = np.linspace(0, xyz.shape[0], windows + 1).astype(int)
        return [(xyz[a:b], middle[a:b]) for a, b in itertools.pairwise(bounds)]

    return vc.cone_grid_from_chunks(chunks(), chunks(), xyz.shape[0])


def _seen(grid: vc.ConeGrid, point: list[float], viewer: list[float]) -> float:
    p = np.asarray([point], np.float64)
    direction = p - np.asarray(viewer, np.float64)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    texel = vc.lookup(grid.texels, grid.origin, grid.cell, grid.dims, p)
    return float(vc.visibility(texel, direction)[0])


@pytest.fixture(scope="module")
def clearing() -> vc.ConeGrid:
    """A walk inside a 16 m clearing: ground 3 mm, trees at 30 m 8 cm, canopy 25 m up 6 cm."""
    rng = np.random.default_rng(3)
    return _grid(
        [
            _ring(rng, 60_000, 0.0, 8.0, 0.0, 2.0, 0.003),
            _ring(rng, 60_000, 30.0, 31.0, 0.0, 6.0, 0.08),
            _ring(rng, 30_000, 0.0, 10.0, 25.0, 26.0, 0.06),
        ]
    )


def test_the_clearing_is_seen_from_everywhere(clearing: vc.ConeGrid) -> None:
    for viewer in ([0, 0, 200], [150, 0, 40], [-3, 2, 1.6]):
        assert _seen(clearing, [1.0, 1.0, 0.5], viewer) == 1.0


def test_the_ring_is_seen_from_inside_and_not_from_outside(clearing: vc.ConeGrid) -> None:
    tree = [30.5, 0.0, 3.0]
    assert _seen(clearing, tree, [0.0, 0.0, 1.6]) == 1.0  # from the clearing, looking out
    assert _seen(clearing, tree, [120.0, 0.0, 3.0]) == 0.0  # from outside, looking in
    assert _seen(clearing, tree, [30.5, 0.0, 200.0]) == 0.0  # from straight above


def test_the_canopy_is_seen_from_below_and_not_from_above(clearing: vc.ConeGrid) -> None:
    canopy = [2.0, 0.0, 25.5]
    assert _seen(clearing, canopy, [0.0, 0.0, 1.6]) == 1.0
    assert _seen(clearing, canopy, [0.0, 0.0, 150.0]) == 0.0


def test_the_cones_point_from_the_clearing(clearing: vc.ConeGrid) -> None:
    probes = np.array([[30.5, 0.0, 3.0], [0.0, -30.5, 3.0], [2.0, 0.0, 25.5]])
    expected = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    texels = vc.lookup(clearing.texels, clearing.origin, clearing.cell, clearing.dims, probes)
    assert np.all(texels[:, 2] != vc.OMNI)
    axes = vc.decode_axis(texels[:, :2])
    assert np.all((axes * expected).sum(axis=1) > 0.8)


def test_a_viewer_fades_a_splat_over_the_cones_edge(clearing: vc.ConeGrid) -> None:
    texel = vc.lookup(
        clearing.texels, clearing.origin, clearing.cell, clearing.dims, np.array([[30.5, 0, 3]])
    )
    half = texel[0, 2] * 180.0 / 254.0
    axis = vc.decode_axis(texel[:, :2])[0]
    side = np.cross(axis, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)

    def at(angle_deg: float) -> float:
        a = math.radians(angle_deg)
        d = np.array([math.cos(a) * axis + math.sin(a) * side])
        return float(vc.visibility(texel, d)[0])

    assert at(half - 1) == 1.0
    assert at(half + vc.FADE_DEG / 2) == pytest.approx(0.5, abs=0.05)
    assert at(half + vc.FADE_DEG + 1) == 0.0


def test_an_orbit_gives_its_subject_and_its_ground_no_cone() -> None:
    """The subject is as fine as the ground at its feet: the orbit saw both from outside."""
    rng = np.random.default_rng(5)
    grid = _grid(
        [
            _ring(rng, 40_000, 0.0, 2.0, 0.0, 5.0, 0.004),
            _ring(rng, 20_000, 0.0, 6.0, -0.1, 0.1, 0.008),
        ]
    )
    assert grid.stats["directionalShare"] == 0.0


def test_a_drone_grid_over_even_ground_gives_no_cone() -> None:
    rng = np.random.default_rng(9)
    grid = _grid([_cloud(rng, 50_000, [-60, -60, -0.5], [60, 60, 3.0], 0.02)])
    assert grid.stats["directionalShare"] == 0.0


def test_the_grid_does_not_depend_on_how_the_scan_is_windowed() -> None:
    rng = np.random.default_rng(11)
    parts = [
        _ring(rng, 20_000, 0.0, 6.0, 0.0, 2.0, 0.003),
        _ring(rng, 3_000, 25.0, 28.0, 0.0, 10.0, 0.08),
    ]
    one = _grid(parts, windows=1)
    seven = _grid(parts, windows=7)
    assert one.dims == seven.dims and one.origin == seven.origin
    assert np.array_equal(one.texels, seven.texels)
    assert vc.encode_view_cones(one) == vc.encode_view_cones(seven)


def test_octahedral_axes_round_trip_within_the_quantisation() -> None:
    rng = np.random.default_rng(1)
    v = rng.normal(size=(5000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    back = vc.decode_axis(vc.encode_axis(v))
    worst = np.degrees(np.arccos(np.clip((v * back).sum(axis=1), -1, 1))).max()
    assert worst < 1.5


def test_a_point_outside_the_grid_takes_its_edge_cell(clearing: vc.ConeGrid) -> None:
    far = np.array([[1000.0, 0.0, 3.0]])
    edge = np.array([[clearing.origin[0] + (clearing.dims[0] - 0.5) * clearing.cell, 0.0, 3.0]])
    args = (clearing.texels, clearing.origin, clearing.cell, clearing.dims)
    assert np.array_equal(vc.lookup(*args, far), vc.lookup(*args, edge))


def test_the_file_round_trips_and_is_declared_on_a_tileset(
    clearing: vc.ConeGrid, tmp_path: Path
) -> None:
    extras = vc.write_view_cones(clearing, tmp_path)
    blob = (tmp_path / vc.URI).read_bytes()
    assert np.array_equal(vc.decode_view_cones(blob, clearing.dims), clearing.texels)
    assert extras["format"] == vc.FORMAT and extras["version"] == vc.VERSION
    assert extras["dims"] == list(clearing.dims) and extras["fadeDeg"] == vc.FADE_DEG
    with pytest.raises(ValueError, match="bytes"):
        vc.decode_view_cones(blob, (1, 1, 1))


def test_the_packer_declares_the_grid_beside_the_collision_grid() -> None:
    root = Path(__file__).resolve().parents[3] / "data" / "tiles" / "synthetic-tree" / "splat"
    extras = json.loads((root / "tileset.json").read_text())["root"]["extras"]
    cones = extras["viewCones"]
    assert cones["uri"] == vc.URI and (root / cones["uri"]).is_file()
    texels = vc.decode_view_cones((root / cones["uri"]).read_bytes(), tuple(cones["dims"]))
    assert texels.shape == (math.prod(cones["dims"]), 4)


def test_the_cli_backfills_a_published_tileset(tmp_path: Path) -> None:
    """`splat_tiles.py viewcones` on a tileset packed without the grid -- from its PLY, or
    from its own leaves when the PLY is gone -- writes what `convert` writes today."""
    import shutil
    import subprocess
    import sys

    import splat_tiles

    captures = Path(__file__).resolve().parents[1]
    ply = captures.parents[1] / "data" / "tiles" / "synthetic-tree" / "source" / "splat.ply"
    fresh = tmp_path / "fresh"
    splat_tiles.convert(ply, fresh, 28.0389, -82.6966, 0.0, 0.02, 1500)
    for source in (ply, None):
        old = tmp_path / ("old-ply" if source else "old-leaves")
        shutil.copytree(fresh, old)
        (old / vc.URI).unlink()
        tileset = json.loads((old / "tileset.json").read_text())
        del tileset["root"]["extras"]["viewCones"]
        (old / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")
        given = str(source) if source else str(old / "tileset.json")
        subprocess.run(
            [sys.executable, "splat_tiles.py", "viewcones", given, str(old)]
            + ["--tileset", str(old / "tileset.json")],
            cwd=captures,
            capture_output=True,
            text=True,
            check=True,
        )
        if source:
            for name in (vc.URI, "tileset.json"):
                assert (old / name).read_bytes() == (fresh / name).read_bytes(), name
        else:
            # The leaves hold every kept gaussian, SPZ-quantised: the same grid within it.
            cones = json.loads((old / "tileset.json").read_text())["root"]["extras"]["viewCones"]
            assert (
                cones["dims"]
                == json.loads((fresh / "tileset.json").read_text())["root"]["extras"]["viewCones"][
                    "dims"
                ]
            )
