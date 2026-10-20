"""`height_surface`: an epoch's top surface, bare ground and their spread, per cell of its
confidence grid's plane (surface.py, ADR 0018)."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import confidence_scene as scene
import gaussians
import surface
from captures_bridge import cell_grid
from conftest import make_recipe
from executor import execute
from runners import RunnerSet
from workdir import Workdir

NO = cell_grid.NO_HEIGHT
IDENTITY = {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]}


def _splat(xyz: np.ndarray, opacity: float | np.ndarray = 2.0) -> dict[str, np.ndarray]:
    """canonical.ply's fourteen columns for centres `xyz` (tileset frame)."""
    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = points.shape[0]
    zeros, ones = np.zeros(n, np.float32), np.ones(n, np.float32)
    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "f_dc_0": zeros,
        "f_dc_1": zeros,
        "f_dc_2": zeros,
        "opacity": np.broadcast_to(np.asarray(opacity, np.float32), (n,)).copy(),
        "scale_0": ones * -3,
        "scale_1": ones * -3,
        "scale_2": ones * -3,
        "rot_0": ones,
        "rot_1": zeros,
        "rot_2": zeros,
        "rot_3": zeros,
    }


def _plane(spacing: float = 0.1, half_m: float = 8.0, z: float = 0.0) -> np.ndarray:
    """A grid of points over a `2 half_m` square at height `z`, off every cell edge."""
    ticks = np.arange(-half_m + spacing / 2, half_m, spacing)
    xx, yy = np.meshgrid(ticks, ticks, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)], axis=1)


def _box(
    centre: tuple[float, float], side: float, height: float, spacing: float = 0.1
) -> np.ndarray:
    """A box's top and four sides, as points `spacing` apart."""
    half = side / 2
    ticks = np.arange(-half + spacing / 2, half, spacing)
    rises = np.arange(spacing / 2, height, spacing)
    top = np.stack(np.meshgrid(ticks, ticks, indexing="ij"), axis=-1).reshape(-1, 2)
    out = [np.column_stack([top, np.full(len(top), height)])]
    uu, zz = (a.ravel() for a in np.meshgrid(ticks, rises, indexing="ij"))
    for fixed, axis in ((half, 0), (-half, 0), (half, 1), (-half, 1)):
        face = np.zeros((uu.size, 3))
        face[:, axis] = fixed
        face[:, 1 - axis] = uu
        face[:, 2] = zz
        out.append(face)
    points = np.concatenate(out)
    points[:, 0] += centre[0]
    points[:, 1] += centre[1]
    return points


def _off_box(points: np.ndarray, centre: tuple[float, float], side: float) -> np.ndarray:
    inside = (np.abs(points[:, 0] - centre[0]) < side / 2) & (
        np.abs(points[:, 1] - centre[1]) < side / 2
    )
    return points[~inside]


def _confidence(folder: Path, level: int, i0: int, j0: int, ni: int, nj: int) -> Path:
    """A confidence grid of that plane (its scores do not matter to the surface)."""
    data = np.full((nj, ni, 3), 200, dtype=np.uint8)
    plane = cell_grid.CellPlane(level, i0, j0, ni, nj, data)
    grid = cell_grid.CellGrid(
        "confidence",
        cell_grid.CONFIDENCE_CHANNELS,
        tuple(cell_grid.min_pyramid(plane, 8)),
        {"policy": cell_grid.WEAK_POLICY, "summary": {"metric": True}},
    )
    descriptor = cell_grid.encode(grid, folder, "confidence.bin")
    (folder / "confidence.json").write_text(json.dumps(descriptor), encoding="utf-8")
    return folder


def _run(
    root: Path,
    splat: np.ndarray | None,
    *,
    opacity: float | np.ndarray = 2.0,
    sfm: np.ndarray | None = None,
    placement: dict[str, Any] | None = None,
    window: tuple[int, int, int, int, int] = (2, -8, -8, 16, 16),
    params: dict[str, Any] | None = None,
) -> Workdir:
    """`height_surface` alone over a hand-made epoch: canonical.ply of `splat` (tileset
    frame), poses/ with `sfm` as its points (COLMAP frame) when given, a placement, and a
    confidence grid of `window` (level, i0, j0, ni, nj)."""
    workdir = Workdir.create(root)
    inputs = workdir.root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    centres = np.zeros((0, 3)) if splat is None else splat
    gaussians.write_ply(inputs / "canonical.ply", _splat(centres, opacity))
    (inputs / "poses").mkdir()
    if sfm is not None:
        scene.write_points3d(inputs / "poses", sfm, track=3)
    (inputs / "placement.json").write_text(json.dumps(placement or IDENTITY), encoding="utf-8")
    _confidence(inputs / "confidence", *window)
    (inputs / "georef.json").write_text(json.dumps({"lat": 46.8, "lon": -91.9, "height": 180.0}))
    execute(
        make_recipe(
            [{"id": "surface", "impl": "height_surface", "params": params or {}}],
            inputs=["canonical.ply", "poses", "placement.json", "confidence", "georef.json"],
        ),
        workdir,
        RunnerSet.local(),
    )
    return workdir


def _grid(workdir: Workdir) -> tuple[dict[str, Any], cell_grid.CellGrid]:
    folder = workdir.out_dir("surface") / "surface"
    descriptor = json.loads((folder / "surface.json").read_text())
    return descriptor, cell_grid.decode(descriptor, (folder / "surface.bin").read_bytes())


def _at(grid: cell_grid.CellGrid, x: float, y: float) -> dict[str, int]:
    row = cell_grid.lookup(grid, [[x, y, 0.0]])[0]
    return dict(zip(("dsm", "ground", "sigma", "support"), (int(v) for v in row), strict=True))


BOX = ((0.0, 0.0), 2.0, 1.5)


@pytest.fixture(scope="module")
def box(tmp_path_factory: pytest.TempPathFactory) -> Workdir:
    centre, side, height = BOX
    points = np.concatenate([_off_box(_plane(), centre, side), _box(centre, side, height)])
    return _run(tmp_path_factory.mktemp("box") / "run", points)


def test_a_box_on_a_plane_is_its_top_on_the_plane_and_the_ground_runs_under_it(
    box: Workdir,
) -> None:
    descriptor, grid = _grid(box)
    plane = grid.planes[0]
    assert (plane.level, plane.i0, plane.j0, plane.ni, plane.nj) == (2, -8, -8, 16, 16)
    # The box's own cells (1 m: i, j in -1..0) stand 1.5 m up; their top is flat.
    for x, y in ((-0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        cell = _at(grid, x, y)
        assert cell["dsm"] == 150 and cell["sigma"] == 1, cell
    # The plane is at 0, flat, everywhere away from the box.
    for x, y in ((-6.5, 4.5), (5.5, -7.5), (3.5, 3.5)):
        cell = _at(grid, x, y)
        assert cell["dsm"] == 0 and cell["sigma"] == 1, cell
    # The ground is the plane everywhere -- under the box too, which nothing saw: a
    # neighbour within 2.5 m did.
    ground = plane.data[..., 1]
    assert (ground == 0).all()
    assert descriptor["estimators"]["groundWindowCells"] == 5
    # Never above the surface.
    assert (plane.data[..., 1] <= plane.data[..., 0]).all()
    assert descriptor["summary"]["surfaced"] == plane.cells
    assert descriptor["status"] == "uncalibrated" and descriptor["noData"] == NO
    assert descriptor["vertical"]["originHeightM"] == 180.0


def test_the_surface_is_written_on_the_confidence_grids_own_plane(box: Workdir) -> None:
    descriptor, grid = _grid(box)
    folder = box.input_path("confidence")
    graded = json.loads((folder / "confidence.json").read_text())
    finest = graded["levels"][0]
    plane = grid.planes[0]
    assert [plane.level, plane.i0, plane.j0, plane.ni, plane.nj] == [
        finest[k] for k in ("level", "i0", "j0", "ni", "nj")
    ]
    assert descriptor["lattice"] == graded["lattice"] and descriptor["cellM"] == graded["cellM"]
    assert descriptor["alignedTo"] == {
        "kind": "confidence",
        "level": finest["level"],
        "sha256": graded["sha256"],
    }
    assert len(grid.planes) == 1  # no pyramid: A5 resamples (ADR 0018)


def test_one_floater_moves_neither_surface_but_widens_sigma(tmp_path: Path) -> None:
    flat = _plane()
    high = np.array([[3.5, 3.5, 30.0]])  # one sky floater in cell (3, 3)
    low = np.array([[-4.5, 2.5, -5.0]])  # one sample under the ground in (-5, 2)
    _, grid = _grid(_run(tmp_path / "one", np.concatenate([flat, high, low])))
    cell = _at(grid, 3.5, 3.5)
    assert cell["dsm"] == 0
    # The floater is in z1: half the top four's spread is 15 m, past u1 -- "at least 2.55 m".
    assert cell["sigma"] == 255
    assert _at(grid, -4.5, 2.5)["ground"] == 0  # the 2nd lowest is the plane's
    assert (grid.planes[0].data[..., 1] == 0).all()  # and it did not spread

    # Two floaters in one cell are a surface as far as this rule can tell (ADR 0018).
    two = np.concatenate([flat, high, high + np.array([0.1, 0.0, 0.0])])
    descriptor, grid = _grid(_run(tmp_path / "two", two))
    assert _at(grid, 3.5, 3.5)["dsm"] == 3000
    assert descriptor["summary"]["sigmaSaturated"] == 1


def test_sfm_points_are_placed_by_the_placement_and_marked_as_sfm(tmp_path: Path) -> None:
    """The pose model's points are in COLMAP's frame: placed by `placement.json`'s
    similarity they land where the splat's would. A turned, scaled, moved placement."""
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    scale, shift = 2.0, np.array([3.0, -1.0, 0.5])
    placement = {"scale": scale, "rotation": turn.tolist(), "translation": shift.tolist()}
    flat = _plane()
    # A 2 m step the SfM cloud alone saw, over x in [4, 6), y in [-6, -4) (tileset frame).
    step = _plane(0.2, 1.0, 2.0) + np.array([5.0, -5.0, 0.0])
    colmap = ((step - shift) @ turn) / scale  # R^T (p - t) / s, row by row
    splat_only = _off_box(flat, (5.0, -5.0), 2.0)
    run = _run(tmp_path / "run", splat_only, sfm=colmap, placement=placement)
    descriptor, grid = _grid(run)
    for x, y in ((4.5, -5.5), (5.5, -4.5)):
        cell = _at(grid, x, y)
        assert cell["dsm"] == 200 and cell["support"] & 0x0F == surface.SFM, cell
        assert cell["ground"] == 0  # a neighbour within 2.5 m saw the plane
    assert _at(grid, -3.5, 2.5)["support"] & 0x0F == surface.SPLAT
    assert descriptor["summary"]["samples"]["sfm"] == len(step)
    assert descriptor["summary"]["samples"]["splat"] == len(splat_only)


def test_support_counts_the_samples_by_powers_of_two_and_says_where_from(tmp_path: Path) -> None:
    centre = np.array([0.5, 0.5, 0.0])
    splat = np.repeat(centre[None], 100, axis=0)  # 100 in cell (0, 0) at 1 m
    sfm = np.repeat(centre[None], 28, axis=0)  # and 28 more from the pose model
    lone = np.repeat([[2.5, 2.5, 0.0]], 2, axis=0)  # two only: too thin for a surface
    run = _run(tmp_path / "run", np.concatenate([splat, lone]), sfm=sfm)
    descriptor, grid = _grid(run)
    both = _at(grid, 0.5, 0.5)
    assert both["support"] == (7 << 4) | surface.SFM | surface.SPLAT  # 128 samples: 2^7
    thin = _at(grid, 2.5, 2.5)
    assert thin["support"] == (1 << 4) | surface.SPLAT
    assert (thin["dsm"], thin["ground"], thin["sigma"]) == (NO, NO, 0)
    assert descriptor["summary"]["thin"] == 1
    empty = _at(grid, -5.5, -5.5)
    assert empty == {"dsm": NO, "ground": NO, "sigma": 0, "support": 0}
    np.testing.assert_array_equal(
        surface.support_of(np.array([0, 1, 2, 3, 4, 1 << 15, 1 << 20]), np.zeros(7, np.uint8)),
        np.array([0, 0, 1, 1, 2, 15, 15]) << 4,
    )


def test_faint_gaussians_and_non_finite_samples_are_not_samples(tmp_path: Path) -> None:
    flat = _plane()
    haze = _plane(z=3.0)
    opacity = np.concatenate([np.full(len(flat), 2.0), np.full(len(haze), -0.01)])
    broken = np.array([[0.5, 0.5, np.nan], [np.inf, 0.5, 1.0]])
    run = _run(tmp_path / "run", np.concatenate([flat, haze]), opacity=opacity, sfm=broken)
    descriptor, grid = _grid(run)
    assert (grid.planes[0].data[..., 0] == 0).all()  # alpha just under a half: not a surface
    assert descriptor["summary"]["nonFinite"] == 2
    assert descriptor["summary"]["samples"]["splat"] == len(flat)


def test_a_height_i2_centimetres_cannot_hold_is_no_surface_not_a_clipped_one(
    tmp_path: Path,
) -> None:
    flat = _plane()
    edge = np.repeat([[0.5, 0.5, 327.67]], 4, axis=0)
    past = np.repeat([[1.5, 0.5, 327.68]], 4, axis=0)
    deep = np.repeat([[2.5, 0.5, -400.0]], 4, axis=0)
    cells = _off_box(flat, (1.5, 0.5), 3.0)
    run = _run(tmp_path / "run", np.concatenate([cells, edge, past, deep]))
    descriptor, grid = _grid(run)
    assert _at(grid, 0.5, 0.5)["dsm"] == 32767
    assert _at(grid, 1.5, 0.5)["dsm"] == NO and _at(grid, 1.5, 0.5)["sigma"] == 0
    assert _at(grid, 2.5, 0.5)["ground"] == NO
    assert descriptor["summary"]["outOfRange"] == 2
    # The out-of-range cell takes no part in its neighbours' ground either.
    assert (grid.planes[0].data[..., 1][grid.planes[0].data[..., 1] != NO] >= 0).all()


def test_samples_outside_the_confidence_plane_are_counted_not_binned(tmp_path: Path) -> None:
    flat = _plane()
    far = np.array([[500.0, 0.0, 0.0], [0.0, -900.0, 1.0]])
    descriptor, grid = _grid(_run(tmp_path / "run", np.concatenate([flat, far])))
    assert descriptor["summary"]["outside"] == 2
    assert grid.planes[0].ni == 16


def test_the_bytes_depend_on_neither_chunk_nor_band_and_repeat(
    box: Workdir, tmp_path: Path
) -> None:
    centre, side, height = BOX
    points = np.concatenate([_off_box(_plane(), centre, side), _box(centre, side, height)])
    sfm = points[::7] + np.array([0.0, 0.0, 0.003])
    a = _run(tmp_path / "a", points, sfm=sfm)
    b = _run(tmp_path / "b", points, sfm=sfm, params={"chunk_gaussians": 97, "band_cells": 16})
    c = _run(tmp_path / "c", points, sfm=sfm)
    first = a.out_dir("surface") / "surface"
    for run in (b, c):
        folder = run.out_dir("surface") / "surface"
        for name in ("surface.bin", "surface.json"):
            assert (folder / name).read_bytes() == (first / name).read_bytes(), name


def test_the_extremes_are_held_a_band_at_a_time(tmp_path: Path) -> None:
    """Memory: the band's extremes (73 bytes a cell), not the plane's; the plane the same
    whatever the band."""
    import tracemalloc

    rng = np.random.default_rng(5)
    xyz = np.column_stack([rng.uniform(0, 128, (200_000, 2)), rng.normal(0, 0.01, 200_000)])

    def samples() -> Any:
        for start in range(0, len(xyz), 1 << 14):
            yield xyz[start : start + (1 << 14)], surface.SPLAT

    window = surface.Window(0, 0, 0, 512, 512)
    planes, peaks = {}, {}
    for band in (1 << 12, 1 << 20):
        tracemalloc.start()
        planes[band] = surface.surface_plane(samples, window, band_cells=band).plane
        peaks[band] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    np.testing.assert_array_equal(planes[1 << 12].data, planes[1 << 20].data)
    whole = 73 * window.cells  # 19 MB: every cell's extremes at once
    assert peaks[1 << 20] > whole
    assert peaks[1 << 12] < whole, (peaks, whole)


@pytest.mark.parametrize("cell_m", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_the_ground_filter_is_the_least_over_the_square_of_cells_within_half_the_window(
    cell_m: float,
) -> None:
    half = surface.ground_half_cells(cell_m)
    assert half == math.floor(2.5 / cell_m)
    assert (2 * half) * cell_m <= surface.GROUND_WINDOW_M  # centres within 2.5 m along each axis
    rng = np.random.default_rng(int(cell_m * 8))
    ground = rng.integers(-500, 500, (23, 31)).astype(np.int32)
    ground[rng.random(ground.shape) < 0.3] = np.iinfo(np.int32).max
    small = min(half, 3)
    out = surface.ground_filter(ground, small)
    none = np.iinfo(np.int32).max
    for j in range(ground.shape[0]):
        for i in range(ground.shape[1]):
            square = ground[max(0, j - small) : j + small + 1, max(0, i - small) : i + small + 1]
            want = none if ground[j, i] == none else int(square.min())
            assert int(out[j, i]) == want, (j, i)


def test_too_few_samples_to_rank_is_refused() -> None:
    window = surface.Window(0, 0, 0, 1, 1)
    with pytest.raises(ValueError, match="at least 3"):
        surface.surface_plane(lambda: iter(()), window, min_samples=2)


def test_the_window_must_be_a_confidence_grid_in_the_tileset_frame(tmp_path: Path) -> None:
    folder = _confidence(tmp_path / "c", 2, -3, 4, 5, 6)
    descriptor = json.loads((folder / "confidence.json").read_text())
    assert surface.read_window(descriptor) == surface.Window(2, -3, 4, 5, 6, descriptor["sha256"])
    for key, value in (("kind", "surface"), ("frame", "colmap"), ("format", "x")):
        with pytest.raises(ValueError, match="not a confidence grid"):
            surface.read_window(dict(descriptor, **{key: value}))
    with pytest.raises(ValueError, match="lattice"):
        surface.read_window(dict(descriptor, lattice={"baseM": 0.25, "level": 3}))


# --- in a run ----------------------------------------------------------------------------


def test_after_place_and_confidence_the_crate_and_the_tree_stand_on_the_ground(
    tmp_path: Path,
) -> None:
    """The shipped stage order over a synthetic epoch with a sparse cloud: the surface
    plane is the confidence plane, the tree box and a crate stand at their heights, and
    the ground runs under both."""
    workdir = Workdir.create(tmp_path / "run")
    columns = scene.with_crate(scene.field(20_000, seed=1), centre=(-3.0, -4.5))
    rng = np.random.default_rng(0)
    cloud = np.column_stack([columns["x"], columns["y"], columns["z"]])
    cloud = cloud[rng.random(len(cloud)) < 0.05]
    names = scene.write_epoch(
        workdir.root / "inputs", columns, scene.orbit(24) + scene.obliques(8), points=cloud
    )
    execute(
        make_recipe(
            [
                {"id": "quality", "impl": "support_gate", "params": {"bar": "strict"}},
                {"id": "place", "impl": "place_splat"},
                {"id": "confidence", "impl": "confidence_grid"},
                {"id": "surface", "impl": "height_surface"},
            ],
            inputs=names,
        ),
        workdir,
        RunnerSet.local(),
    )
    descriptor, grid = _grid(workdir)
    graded = json.loads(
        (workdir.out_dir("confidence") / "confidence" / "confidence.json").read_text()
    )
    assert descriptor["levels"][0]["ni"] == graded["levels"][0]["ni"]
    assert _at(grid, 0.2, 0.2)["dsm"] == 300  # the tree box's top
    assert _at(grid, -3.0, -4.5)["dsm"] == 150  # the crate's
    assert _at(grid, -3.0, -4.5)["ground"] == 0 and _at(grid, 0.2, 0.2)["ground"] == 0
    assert descriptor["summary"]["samples"]["sfm"] > 0
    assert descriptor["summary"]["metric"] is True


def test_a_stub_poses_without_points_is_the_splat_alone(tmp_path: Path) -> None:
    run = _run(tmp_path / "run", _plane())
    descriptor, _ = _grid(run)
    assert descriptor["summary"]["samples"]["sfm"] == 0
    assert "no points3D.bin" in run.log_path("surface").read_text()


def test_copy_of_box_inputs_reruns_identically(box: Workdir, tmp_path: Path) -> None:
    """A re-run from the same artifacts (a Refine from `surface`) writes the same bytes."""
    again = Workdir.create(tmp_path / "again")
    shutil.copytree(box.root / "inputs", again.root / "inputs", dirs_exist_ok=True)
    execute(
        make_recipe(
            [{"id": "surface", "impl": "height_surface"}],
            inputs=["canonical.ply", "poses", "placement.json", "confidence", "georef.json"],
        ),
        again,
        RunnerSet.local(),
    )
    for name in ("surface.bin", "surface.json"):
        first = (box.out_dir("surface") / "surface" / name).read_bytes()
        assert (again.out_dir("surface") / "surface" / name).read_bytes() == first

