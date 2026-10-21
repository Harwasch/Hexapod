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
    names = [channel.name for channel in cell_grid.SURFACE_CHANNELS]
    return dict(zip(names, (int(v) for v in row), strict=True))


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
    # The floater is in z1: the top eight's spread is 30 m, so sigma is past u1 -- "at least
    # 2.55 m".
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
    assert empty == {"dsm": NO, "ground": NO, "sigma": 0, "support": 0, "groundSigma": 0}
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
    """Memory: the band's extremes (137 bytes a cell), not the plane's; the plane the same
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
    whole = 137 * window.cells  # 36 MB: every cell's extremes at once
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


# --- package and catalog -----------------------------------------------------------------


def _package(root: Path, epoch: Workdir, *, heights: Path | None, grid: bool = True) -> Workdir:
    """package -> register over the box epoch's canonical.ply, with its confidence grid
    (unless not `grid`) and the surface folder `heights` (when given)."""
    workdir = Workdir.create(root)
    inputs = workdir.root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copy(epoch.input_path("canonical.ply"), inputs / "canonical.ply")
    shutil.copy(epoch.input_path("georef.json"), inputs / "georef.json")
    names = ["canonical.ply", "georef.json"]
    if grid:
        shutil.copytree(epoch.input_path("confidence"), inputs / "confidence")
        names.append("confidence")
    if heights is not None:
        shutil.copytree(heights, inputs / "surface")
        names.append("surface")
    execute(
        make_recipe(
            [
                {"id": "package", "impl": "splat_tiles", "params": {"tile_gaussians": 4000}},
                {"id": "register", "impl": "catalog"},
            ],
            inputs=names,
        ),
        workdir,
        RunnerSet.local(),
    )
    return workdir


def test_package_ships_the_surface_after_the_confidence_grid_and_catalog_registers_it(
    box: Workdir, tmp_path: Path
) -> None:
    heights = box.out_dir("surface") / "surface"
    shipped = _package(tmp_path / "with", box, heights=heights)
    plain = _package(tmp_path / "without", box, heights=None)
    splat = shipped.out_dir("package") / "splat"
    tileset = json.loads((splat / "tileset.json").read_text())
    extras = tileset["root"]["extras"]
    descriptor = json.loads((heights / "surface.json").read_text())
    assert list(extras)[-2:] == ["confidence", "surface"]
    assert extras["surface"] == descriptor
    assert (splat / "surface.bin").read_bytes() == (heights / "surface.bin").read_bytes()
    registration = json.loads((shipped.out_dir("register") / "registration.json").read_text())
    assert registration["surface"] == descriptor
    assert "surface.bin" in registration["artifacts"]
    # Not measured: no file, no key, null in the registration; every other byte the same.
    plain_splat = plain.out_dir("package") / "splat"
    plain_tiles = json.loads((plain_splat / "tileset.json").read_text())
    assert "surface" not in plain_tiles["root"]["extras"]
    plain_registration = json.loads((plain.out_dir("register") / "registration.json").read_text())
    assert plain_registration["surface"] is None
    assert plain_registration["confidence"] == registration["confidence"] is not None
    del extras["surface"]
    assert tileset == plain_tiles
    for path in plain_splat.iterdir():
        assert (splat / path.name).read_bytes() == path.read_bytes() or path.name == "tileset.json"
    grids = cell_grid.from_tileset(splat / "tileset.json")
    assert set(grids) == {"confidence", "surface"}


def test_a_surface_without_a_confidence_grid_follows_the_collision_grid(
    box: Workdir, tmp_path: Path
) -> None:
    run = _package(tmp_path / "run", box, heights=box.out_dir("surface") / "surface", grid=False)
    extras = json.loads((run.out_dir("package") / "splat" / "tileset.json").read_text())["root"][
        "extras"
    ]
    assert list(extras)[-2:] == ["collision", "surface"]


@pytest.mark.parametrize("damage", ["sha", "kind", "uri-taken", "stub"])
def test_a_surface_package_refuses_is_left_out_and_the_confidence_grid_still_ships(
    box: Workdir, tmp_path: Path, damage: str
) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(box.out_dir("surface") / "surface", broken)
    descriptor = json.loads((broken / "surface.json").read_text())
    if damage == "sha":
        descriptor["sha256"] = "0" * 64
    elif damage == "kind":
        descriptor["kind"] = "confidence"
    elif damage == "uri-taken":
        # The confidence grid's file name: two grids cannot both be confidence.bin.
        shutil.copy(broken / "surface.bin", broken / "confidence.bin")
        descriptor["uri"] = "confidence.bin"
    if damage == "stub":
        (broken / "surface.json").write_bytes(b"\x00stub")
    else:
        (broken / "surface.json").write_text(json.dumps(descriptor))
    run = _package(tmp_path / "run", box, heights=broken)
    splat = run.out_dir("package") / "splat"
    extras = json.loads((splat / "tileset.json").read_text())["root"]["extras"]
    assert "surface" not in extras and "confidence" in extras
    assert not (splat / "surface.bin").exists()
    log = run.log_path("package").read_text()
    if damage == "stub":
        assert "no surface grid in surface/" in log
    else:
        assert "surface grid was not shipped" in log
        assert "confidence grid was not shipped" not in log
    registration = json.loads((run.out_dir("register") / "registration.json").read_text())
    assert registration["surface"] is None and registration["confidence"] is not None


# --- the demo ----------------------------------------------------------------------------


def test_the_change_demo_finds_the_crate_raised_and_the_noise_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`confidence_demo.py --change`: a 1.5 m crate in e2 on ground both epochs saw well,
    each scan with its own sample, 1 cm noise and SfM cloud. Every cell of the crate is
    raised by about its height, and nothing that does not touch it changes."""
    from experiments import confidence_demo

    assert confidence_demo.main(["--out", str(tmp_path / "demo"), "--change"]) == 0
    text = capsys.readouterr().out
    report = json.loads((tmp_path / "demo" / "report.json").read_text())
    found = report["change"]
    crate = found["crate"]
    assert crate["cellsInside"] > 20
    assert crate["raisedInside"] == crate["cellsInside"]
    assert crate["changedOutside"] == 0
    assert found["counts"]["lowered"] == 0
    assert found["counts"]["unchanged"] > 10 * found["counts"]["raised"]
    on = [
        row
        for row in found["changed"]
        if abs(row["eastM"] + 3.1) <= 0.75 and abs(row["northM"] - 3.4) <= 0.75
    ]
    assert on and all(abs(row["deltaM"] - 1.5) < 0.1 for row in on)
    # The threshold is the frame-relative one: e1 made the frame (0), e2 aligned at 0.062.
    assert (found["uE1M"], found["uE2M"]) == (0.0, 0.062)
    assert 0.18 < found["minDetectableM"] < 0.2
    assert found["unchangedMaxDeltaOverThreshold"] < 1.0
    assert "Derived from height surfaces, not observed" in text
    for name in ("e1-splat", "e2-splat"):
        tileset = json.loads((tmp_path / "demo" / name / "tileset.json").read_text())
        assert list(tileset["root"]["extras"])[-2:] == ["confidence", "surface"]


def test_coarsening_a_surface_takes_the_highest_top_the_lowest_ground_the_widest_sigma() -> None:
    from experiments import confidence_demo

    data = np.zeros((2, 2, 5), dtype=np.int64)
    data[..., 0] = [[10, NO], [30, 20]]
    data[..., 1] = [[5, NO], [-7, 0]]
    data[..., 2] = [[1, 0], [4, 2]]
    data[..., 3] = [[0x12, 0x00], [0x31, 0x22]]
    data[..., 4] = [[6, 0], [9, 3]]
    coarse = confidence_demo.coarsen_surface(cell_grid.CellPlane(0, 0, 0, 2, 2, data), 1)
    assert coarse.data[0, 0].tolist() == [30, -7, 4, 0x33, 9]
    empty = np.zeros((2, 2, 5), dtype=np.int64)
    empty[..., :2] = NO
    nothing = confidence_demo.coarsen_surface(cell_grid.CellPlane(0, 0, 0, 2, 2, empty), 1)
    assert nothing.data[0, 0].tolist() == [NO, NO, 0, 0, 0]
    # A child whose ground was never observed (NO_HEIGHT under a surface) gives no ground:
    # the lowest is of the children that have one, and none when no child has.
    unseen = data.copy()
    unseen[0, 1, :] = [50, NO, 3, 0x12, 0]
    unseen[1, 0, 1], unseen[1, 0, 4] = NO, 0
    coarse = confidence_demo.coarsen_surface(cell_grid.CellPlane(0, 0, 0, 2, 2, unseen), 1)
    assert coarse.data[0, 0].tolist() == [50, 0, 4, 0x33, 6]
    unseen[..., 1], unseen[..., 4] = NO, 0
    coarse = confidence_demo.coarsen_surface(cell_grid.CellPlane(0, 0, 0, 2, 2, unseen), 1)
    assert coarse.data[0, 0].tolist() == [50, NO, 4, 0x33, 0]


# --- boundaries (review) -----------------------------------------------------------------


def _one_cell(points: list[list[float]], **kwargs: Any) -> tuple[list[int], surface.Surfaced]:
    """`surface_plane` over a 3x1 window at 1 m with nothing else in it: cell 0's channels."""
    xyz = np.asarray(points, dtype=np.float64)
    window = surface.Window(level=2, i0=0, j0=0, ni=3, nj=1)
    found = surface.surface_plane(lambda: iter([(xyz, surface.SPLAT)]), window, **kwargs)
    return [int(v) for v in found.plane.data[0, 0]], found


def test_exactly_three_samples_are_a_surface_and_sigma_spans_all_three() -> None:
    """`count >= min_samples`, not `>`; and with three, z8 does not exist, so sigma is
    read from the spread of all three: (z1 - z3) RMS2(3) / SPAN(3)."""
    cell, found = _one_cell([[0.5, 0.5, 0.0], [0.5, 0.5, 0.1], [0.5, 0.5, 0.3]])
    # dsm z2, ground z2 (no neighbours), 0.3 m * 0.6699 / 1.6925 = 11.9 cm, rounded up.
    assert cell[:3] == [10, 10, 12]
    assert found.thin == 0


def test_sigma_is_rounded_up_to_the_centimetre() -> None:
    """0.0369 m * RMS2(4) / SPAN(4) = 1.2 cm is stored as 2 cm, never 1; the 1 cm floor
    stays exactly 1."""
    up, _ = _one_cell([[0.5, 0.5, z] for z in (0.0369, 0.01, 0.005, 0.0)])
    assert up[2] == 2
    exact, _ = _one_cell([[0.5, 0.5, z] for z in (0.02, 0.01, 0.005, 0.0)])
    assert exact[2] == 1


def test_a_height_too_large_for_int64_is_out_of_range_not_a_wrapped_surface() -> None:
    """Review finding: 1e20 m wrapped in the int64 cast and passed the i2 range check as a
    plausible surface, with a RuntimeWarning. It is out of range, quietly."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cell, found = _one_cell([[0.5, 0.5, 1e20]] * 3)
        low, below = _one_cell([[0.5, 0.5, 0.0]] * 2 + [[0.5, 0.5, -1e19]] * 2)
    assert cell == [NO, NO, 0, cell[3], 0] and found.out_of_range == 1
    assert low[:3] == [NO, NO, 0] and below.out_of_range == 1


def test_a_gaussian_exactly_half_opaque_is_a_sample(tmp_path: Path) -> None:
    """`opacity >= 0` in the logit: alpha exactly 0.5 is in."""
    descriptor, grid = _grid(_run(tmp_path / "run", _plane(), opacity=0.0))
    assert descriptor["summary"]["samples"]["splat"] == len(_plane())
    assert (grid.planes[0].data[..., 2] > 0).all()


# --- the noise floor (review B5a) --------------------------------------------------------


def _flat_epoch(per_cell: int, noise_m: float, seed: int, side: int = 20) -> cell_grid.CellPlane:
    """Unchanged flat ground at 0 over `side` x `side` cells of 0.25 m: `per_cell` samples
    in every cell, each with Gaussian height noise `noise_m`. One epoch's scan of it."""
    rng = np.random.default_rng(seed)
    cells = side * side
    i, j = np.divmod(np.repeat(np.arange(cells), per_cell), side)
    xy = (np.column_stack([i, j]) + rng.uniform(0.01, 0.99, (cells * per_cell, 2))) * 0.25
    xyz = np.column_stack([xy, rng.normal(0.0, noise_m, cells * per_cell)])
    window = surface.Window(level=0, i0=0, j0=0, ni=side, nj=side)
    return surface.surface_plane(lambda: iter([(xyz, surface.SPLAT)]), window).plane


def _false_changes(a: cell_grid.CellPlane, b: cell_grid.CellPlane, ua: float, ub: float) -> int:
    """Cells of an unchanged scene the change rule (ADR 0018 section 4, the demo's and
    A5's) would call raised or lowered: |dsm_b - dsm_a| > 3 sqrt(sa^2 + sb^2 + ua^2 + ub^2)."""
    unit = cell_grid.HEIGHT_UNIT_M
    da, db = (p.data[..., 0].astype(np.float64) * unit for p in (a, b))
    sa, sb = (p.data[..., 2].astype(np.float64) * unit for p in (a, b))
    assert (sa > 0).all() and (sb > 0).all()
    threshold = 3.0 * np.sqrt(sa**2 + sb**2 + ua**2 + ub**2)
    return int((np.abs(db - da) > threshold).sum())


@pytest.mark.parametrize(
    ("a", "b", "ub"),
    [
        ((20, 0.10), (2000, 0.10), 0.0),  # the review's: 198 of 400 cells raised before
        ((100, 0.01), (100, 0.10), 0.03),  # the review's: 332 of 400 before
        ((3, 0.05), (500, 0.05), 0.0),
        ((100, 0.01), (100, 0.01), 0.0),
    ],
)
def test_no_change_control_unchanged_ground_is_unchanged_whatever_the_samples_and_noise(
    a: tuple[int, float], b: tuple[int, float], ub: float
) -> None:
    """Review B5a, the no-change control: the same unchanged ground scanned twice, with
    very different sample counts or noise. `dsm` is the 2nd highest sample, which reads
    higher the more samples a cell has and the noisier they are; its sigma must carry
    that bias, or the denser or noisier epoch reads as raised everywhere. At the rule's
    3 sigma no more than about one cell in 370 of Gaussian noise may be called changed:
    at most 1 of 400 here."""
    first = _flat_epoch(*a, seed=1)
    second = _flat_epoch(*b, seed=2)
    assert _false_changes(first, second, 0.0, ub) <= 1


def _normal_order_moments(k: int, n: int) -> tuple[float, float]:
    """E[Z(k:n)] and E[Z(k:n)^2], the k-th highest of n standard normals, by integrating
    its density on a fine grid in log space (math.erfc alone: independent of the Beta
    quadrature the tables were computed by)."""
    x = np.linspace(-14.0, 14.0, 280_001)
    erfc = np.frompyfunc(math.erfc, 1, 1)
    upper = np.asarray(erfc(x / math.sqrt(2.0)), dtype=np.float64) / 2.0  # 1 - Phi(x)
    lower = np.asarray(erfc(-x / math.sqrt(2.0)), dtype=np.float64) / 2.0  # Phi(x), no cancelling
    with np.errstate(divide="ignore"):
        log_density = (
            math.lgamma(n + 1)
            - math.lgamma(k)
            - math.lgamma(n - k + 1)
            + ((n - k) * np.log(lower) if n > k else 0.0)
            + ((k - 1) * np.log(upper) if k > 1 else 0.0)
            - x * x / 2.0
            - 0.5 * math.log(2.0 * math.pi)
        )
    weight = np.exp(log_density)
    weight /= weight.sum()
    return float((weight * x).sum()), float((weight * x * x).sum())


@pytest.mark.parametrize("n", [3, 4, 5, 8, 9, 20, 64, 65, 95, 96, 127, 1000, 1 << 16])
def test_the_noise_model_tables_are_the_normal_order_statistics(n: int) -> None:
    """RMS2 and SPAN (surface.py, the noise model) against an independent integration:
    exact to 64 samples; above, the bucket's value is its top's, so never a smaller sigma
    than the count's own (RMS2 at least, SPAN at most), and within the half-octave."""
    count = np.array([n], dtype=np.int64)
    rms, span = float(surface.rms_second(count)[0]), float(surface.top_span(count)[0])
    m = min(n, surface.TOP_K)
    _, second = _normal_order_moments(2, n)
    span_n = _normal_order_moments(1, n)[0] - _normal_order_moments(m, n)[0]
    if n <= 64:
        # Rounded at the 4th decimal: up for RMS2, down for SPAN.
        assert 0.0 <= rms - math.sqrt(second) < 2e-4, (rms, math.sqrt(second))
        assert 0.0 <= span_n - span < 2e-4, (span, span_n)
    else:
        e = n.bit_length() - 1  # n in [2^e, 2^(e+1)); halves split at 1.5 2^e
        top = (3 << (e - 1)) - 1 if n < 3 << (e - 1) else (1 << (e + 1)) - 1
        _, top_second = _normal_order_moments(2, top)
        top_span = _normal_order_moments(1, top)[0] - _normal_order_moments(m, top)[0]
        assert 0.0 <= rms - math.sqrt(top_second) < 2e-4
        assert 0.0 <= top_span - span < 2e-4
        assert rms >= math.sqrt(second) and span <= span_n


# --- where the ground was not seen (review B5a) ------------------------------------------


def _surface_of(xyz: np.ndarray, level: int, half_m: float) -> surface.Surfaced:
    """`surface_plane` over the square `+/- half_m` at `level`, from these samples."""
    cells = round(2 * half_m / cell_grid.Lattice(level).cell_m)
    window = surface.Window(level=level, i0=-cells // 2, j0=-cells // 2, ni=cells, nj=cells)
    return surface.surface_plane(lambda: iter([(xyz, surface.SPLAT)]), window)


def test_ground_deep_under_a_canopy_wider_than_the_window_is_unobserved_not_the_canopy() -> None:
    """Review B5a: a 10 m canopy at 4 m, dense enough that nothing under it was sampled,
    with ground seen only around it. Its cells within 2.5 m of its edge take the ground
    their window sees; the cells further in see only canopy, and their ground was never
    observed -- `NO_HEIGHT`, with the canopy's dsm still there -- never 4 m of "ground"."""
    centre, side = (0.0, 0.0), 10.0
    canopy = _plane(spacing=0.1, half_m=5.0, z=4.0)
    points = np.concatenate([_off_box(_plane(spacing=0.1, half_m=8.0), centre, side), canopy])
    found = _surface_of(points, level=0, half_m=8.0)
    plane = found.plane
    dsm, ground = plane.data[..., 0], plane.data[..., 1]
    east = (np.arange(plane.ni) + plane.i0 + 0.5) * plane.cell_m
    north = (np.arange(plane.nj) + plane.j0 + 0.5) * plane.cell_m
    inside = np.maximum(np.abs(east)[None, :], np.abs(north)[:, None])  # L-inf from centre
    deep = inside < 5.0 - 2.5 - 0.25  # more than 2.5 m (and a cell) inside the edge
    near = (inside > 5.0 - 2.5 + 0.25) & (inside < 5.0 - 0.25)
    assert (dsm[deep] == 400).all() and (dsm[near] == 400).all()
    assert (ground[deep] == NO).all(), np.unique(ground[deep])
    assert (ground[near] == 0).all()  # within the window's reach: the ground beside it
    assert (ground[inside > 5.25] == 0).all()  # open ground
    assert found.ground_unobserved == int((ground == NO).sum()) > 0


def test_ground_on_a_steep_even_slope_is_never_called_unobserved() -> None:
    """The lower envelope reads low on a slope (2.5 g m along each axis); a slope up to the
    grade the unobserved test allows (`GROUND_MAX_GRADE`, 1 in 2) is ground everywhere."""
    points = _plane(spacing=0.1, half_m=8.0)
    for grade in (0.2, surface.GROUND_MAX_GRADE):
        tilted = points.copy()
        tilted[:, 2] = grade * (tilted[:, 0] + 0.6 * tilted[:, 1])
        found = _surface_of(tilted, level=0, half_m=8.0)
        assert found.ground_unobserved == 0, grade
        assert (found.plane.data[..., 1] != NO).all()


def _channel(plane: cell_grid.CellPlane, name: str) -> np.ndarray:
    names = [channel.name for channel in cell_grid.SURFACE_CHANNELS]
    return plane.data[..., names.index(name)].astype(np.float64)


@pytest.mark.parametrize(
    ("a", "b", "ub"),
    [
        ((20, 0.10), (2000, 0.10), 0.0),
        ((100, 0.01), (100, 0.10), 0.03),
        ((3, 0.05), (500, 0.05), 0.0),
        ((100, 0.01), (100, 0.01), 0.0),
    ],
)
def test_no_change_control_for_the_ground_its_own_sigma_carries_the_envelopes_bias(
    a: tuple[int, float], b: tuple[int, float], ub: float
) -> None:
    """Review B5a, the no-change control for the ground (the change map's primary layer):
    the ground is the least 2nd-lowest sample over a 5 m window, so it reads low by more
    the more samples the window holds and the noisier they are -- by far more than dsm
    reads high. `groundSigma` must carry that, or unchanged ground reads as lowered."""
    first = _flat_epoch(*a, seed=1)
    second = _flat_epoch(*b, seed=2)
    unit = cell_grid.HEIGHT_UNIT_M
    ga, gb = (_channel(p, "ground") * unit for p in (first, second))
    sa, sb = (_channel(p, "groundSigma") * unit for p in (first, second))
    assert (sa > 0).all() and (sb > 0).all()
    threshold = 3.0 * np.sqrt(sa**2 + sb**2 + ub**2)
    assert int((np.abs(gb - ga) > threshold).sum()) <= 1


# --- the demo's comparison (review B5a) ---------------------------------------------------


def _heights(level: int, cells: dict[tuple[int, int], tuple[int, int]], n: int = 2) -> Any:
    """A surface grid at `level` over cells `0 .. n-1` square: `cells` maps (i, j) to
    (dsm, sigma) in centimetres (ground at 0, groundSigma 1); every other cell has none."""
    data = np.zeros((n, n, len(cell_grid.SURFACE_CHANNELS)), dtype=np.int64)
    data[..., 0] = data[..., 1] = NO
    for (i, j), (top, spread) in cells.items():
        data[j, i] = [top, 0, spread, 0x42, 1]
    plane = cell_grid.CellPlane(level, 0, 0, n, n, data)
    meta = {"vertical": {"originHeightM": 180.0}, "summary": {"metric": True}}
    return cell_grid.CellGrid("surface", cell_grid.SURFACE_CHANNELS, (plane,), meta)


def _vouched(level: int, n: int = 2) -> Any:
    """A confidence grid at `level` that calls every cell strong."""
    plane = cell_grid.CellPlane(level, 0, 0, n, n, np.full((n, n, 3), 200, dtype=np.uint8))
    return cell_grid.CellGrid(
        "confidence",
        cell_grid.CONFIDENCE_CHANNELS,
        tuple(cell_grid.min_pyramid(plane, 4)),
        {"policy": cell_grid.WEAK_POLICY},
    )


def test_the_demo_compares_a_coarse_cell_only_with_every_child_the_finer_epoch_measured() -> None:
    """Review B5a: the demo coarsened each epoch on its own, so a finer epoch that measured
    only a 2 m child of a cell stood for the whole cell against the coarser epoch's ground
    there: "lowered 2 m", a difference in what was measured. A coarse cell is compared only
    when the finer epoch measured all of its children; otherwise it is insufficient."""
    from experiments import confidence_demo

    fine = _heights(0, {(0, 0): (200, 1)}, n=2)  # one child of coarse cell (0, 0), 2 m up
    coarse = _heights(1, {(0, 0): (0, 1)}, n=1)  # the whole coarse cell, at the ground
    found = confidence_demo.surface_change(coarse, fine, _vouched(1, 1), _vouched(0), 0.0, 0.0)
    assert found["counts"]["insufficient"] == 1 and found["changed"] == []
    # All four children measured: compared, and the 2 m child is the coarse cell's top.
    whole = _heights(
        0, {(i, j): (200 if (i, j) == (0, 0) else 0, 1) for i in (0, 1) for j in (0, 1)}
    )
    found = confidence_demo.surface_change(coarse, whole, _vouched(1, 1), _vouched(0), 0.0, 0.0)
    assert found["counts"]["raised"] == 1


def test_the_demo_refuses_heights_it_cannot_put_in_metres_on_one_origin() -> None:
    """Review B5a (low): the demo compared heights above different origins (100 m and 99 m
    read the same ground as "raised 1 m") and of runs whose scale was never resolved
    (`summary.metric` false: not metres at all). It refuses both, as the change map does."""
    from experiments import confidence_demo

    a, b = _heights(0, {(0, 0): (0, 1)}), _heights(0, {(0, 0): (100, 1)})
    a.meta["vertical"]["originHeightM"] = 100.0
    b.meta["vertical"]["originHeightM"] = 99.0
    with pytest.raises(ValueError, match="origin"):
        confidence_demo.surface_change(a, b, _vouched(0), _vouched(0), 0.0, 0.0)
    c, d = _heights(0, {(0, 0): (0, 1)}), _heights(0, {(0, 0): (100, 1)})
    d.meta["summary"]["metric"] = False
    with pytest.raises(ValueError, match="metric"):
        confidence_demo.surface_change(c, d, _vouched(0), _vouched(0), 0.0, 0.0)


def test_the_demo_takes_a_saturated_sigma_as_unbounded_not_as_2_55_m() -> None:
    """Review B5a (low): sigma 255 means at least 2.55 m, so its threshold is only a lower
    bound -- the review's two 30 m floaters made a 30 m dsm read as raised against 7.65 m.
    A cell with a saturated sigma on either side cannot be classified: insufficient."""
    from experiments import confidence_demo

    a = _heights(0, {(0, 0): (0, 1), (1, 0): (0, 1)})
    b = _heights(0, {(0, 0): (3000, 255), (1, 0): (3000, 254)})
    found = confidence_demo.surface_change(a, b, _vouched(0), _vouched(0), 0.0, 0.0)
    assert found["counts"]["insufficient"] == 2 + 1  # (0, 1) and (1, 1) have none; (0, 0)
    assert [(row["i"], row["class"]) for row in found["changed"]] == [(1, "raised")]


# --- review round 2: spatially uneven noise under the ground (B5a, round 2) --------------


def _uneven_epoch(
    per_cell: int, quiet_m: float, loud_m: float, every: int, seed: int, side: int = 48
) -> cell_grid.CellPlane:
    """Unchanged flat ground at 0 over `side` x `side` cells of 0.25 m, `per_cell` samples
    a cell with `quiet_m` of noise, except the cells `[every//2::every, every//2::every]`
    (none when `every` is 0), whose noise is `loud_m`."""
    rng = np.random.default_rng(seed)
    cells = side * side
    j, i = np.divmod(np.repeat(np.arange(cells), per_cell), side)
    noise = np.full((side, side), quiet_m)
    if every:
        noise[every // 2 :: every, every // 2 :: every] = loud_m
    xy = (np.column_stack([i, j]) + rng.uniform(0.01, 0.99, (cells * per_cell, 2))) * 0.25
    z = rng.normal(0.0, 1.0, cells * per_cell) * noise[j, i]
    window = surface.Window(level=0, i0=0, j0=0, ni=side, nj=side)
    xyz = np.column_stack([xy, z])
    return surface.surface_plane(lambda: iter([(xyz, surface.SPLAT)]), window).plane


def _false_ground(a: cell_grid.CellPlane, b: cell_grid.CellPlane, ua: float, ub: float) -> int:
    """Cells of an unchanged scene whose ground the rule would call changed, of those where
    both epochs observed a ground: |g_b - g_a| > 3 sqrt(gs_a^2 + gs_b^2 + ua^2 + ub^2)."""
    unit = cell_grid.HEIGHT_UNIT_M
    ga, gb = (_channel(p, "ground") for p in (a, b))
    sa, sb = (_channel(p, "groundSigma") * unit for p in (a, b))
    both = (ga != NO) & (gb != NO)
    threshold = 3.0 * np.sqrt(sa**2 + sb**2 + ua**2 + ub**2)
    return int((both & (np.abs(gb - ga) * unit > threshold)).sum())


@pytest.mark.parametrize(
    ("per_cell", "every", "seeds"),
    [
        (2000, 8, (90, 91)),  # the review's: 775 of 2,304 cells lowered
        (2000, 8, (92, 93)),  # 619
        (2000, 8, (94, 95)),  # 592
        (2000, 16, (90, 91)),  # 1 noisy cell in 256: 693 of 6,912 over three seeds
        (50, 8, (90, 91)),  # fewer samples: 1 of 6,912
        (2000, 64, (96, 97)),  # one noisy cell in the plane: 2 of 23,040
    ],
)
def test_ground_sigma_is_the_noise_of_the_sample_that_set_the_minimum_not_an_average(
    per_cell: int, every: int, seeds: tuple[int, int]
) -> None:
    """Review B5a round 2: epoch a has 1 cm of noise everywhere; epoch b the same but 10 cm
    in a sparse lattice of cells. The ground is a window's least 2nd-lowest, so in b it is
    set by the noisy cells' deep lows (some 32 cm down); an average of the window's noise
    scales gave it a sigma of 5-7 cm and 28.7% of cells read as lowered. The sigma of the
    cell that set the minimum carries it: no more than the 3 sigma rule's one cell in 370."""
    a = _uneven_epoch(per_cell, 0.01, 0.01, 0, seeds[0])
    b = _uneven_epoch(per_cell, 0.01, 0.10, every, seeds[1])
    assert _false_ground(a, b, 0.03, 0.03) <= math.ceil(48 * 48 / 370)


# --- review round 2: a canopy wider than the window's reach ------------------------------


def _crown_epoch(
    side_m: float, crown_m: float, tall_m: float, seed: int, per_cell: int = 50
) -> cell_grid.CellPlane:
    """Flat ground at 0 over a `side_m` square of 0.25 m cells centred on the origin, 1 cm
    of noise, and a dense square crown `crown_m` wide and `tall_m` high in the middle (0:
    none) under which nothing was sampled: every sample there is on its top."""
    rng = np.random.default_rng(seed)
    side = round(side_m / 0.25)
    cells = side * side
    j, i = np.divmod(np.repeat(np.arange(cells), per_cell), side)
    xy = (
        np.column_stack([i, j]) - side / 2 + rng.uniform(0.01, 0.99, (cells * per_cell, 2))
    ) * 0.25
    under = (np.abs(xy[:, 0]) < crown_m / 2) & (np.abs(xy[:, 1]) < crown_m / 2)
    z = np.where(under, tall_m, 0.0) + rng.normal(0.0, 0.01, cells * per_cell)
    window = surface.Window(level=0, i0=-side // 2, j0=-side // 2, ni=side, nj=side)
    xyz = np.column_stack([xy, z])
    return surface.surface_plane(lambda: iter([(xyz, surface.SPLAT)]), window).plane


@pytest.mark.parametrize(
    ("side_m", "crown_m", "before_m", "after_m"),
    [
        (32.0, 20.0, 0.0, 4.0),  # a 20 m crown appears: 324 false ground changes
        (32.0, 20.0, 4.0, 0.0),  # and is cut down
        (32.0, 20.0, 4.0, 6.0),  # grows
        (56.0, 40.0, 4.0, 6.0),  # a 40 m crown grows 4 -> 6 m: 4,356 false ground raises
        (16.0, 6.0, 4.0, 6.0),  # a 6 m crown (its core already unobserved): 0
    ],
)
def test_ground_deep_under_a_wide_canopy_is_unobserved_and_never_changes_with_it(
    side_m: float, crown_m: float, before_m: float, after_m: float
) -> None:
    """Review B5a round 2: deep inside a crown wider than about 2 H / grade, the slope
    test let the canopy itself stand as ground (397 cm at the centre of a 4 m crown), so
    its growth read as the ground rising. The canopy's edge is a step no slope of 1 in 2
    can make; every cell reached from that step across the canopy's top, whose window
    ground stands that far above the ground the step's foot showed, has no ground: never
    canopy height. Ground within the window's reach of the edge is still observed."""
    a = _crown_epoch(side_m, crown_m, before_m, seed=11)
    b = _crown_epoch(side_m, crown_m, after_m, seed=12)
    assert _false_ground(a, b, 0.03, 0.03) == 0
    for plane, tall in ((a, before_m), (b, after_m)):
        ground = _channel(plane, "ground")
        if tall:
            centre = ground[
                plane.nj // 2 - 2 : plane.nj // 2 + 2, plane.ni // 2 - 2 : plane.ni // 2 + 2
            ]
            assert (centre == NO).all(), np.unique(centre)
        seen = ground[ground != NO]
        assert seen.max() < 50, seen.max()  # what ground there is is the ground


@pytest.mark.parametrize("grade", [0.49, 0.50, 0.51])
def test_ground_on_a_slope_near_the_grade_limit_is_all_observed(grade: float) -> None:
    """Review B5a round 2's check: 64 x 64 cells of 0.25 m, 1 cm of noise, grades just
    under, at and over 1 in 2 along one axis: no cell's ground is missing."""
    rng = np.random.default_rng(int(grade * 100))
    side, per_cell = 64, 20
    cells = side * side
    j, i = np.divmod(np.repeat(np.arange(cells), per_cell), side)
    xy = (np.column_stack([i, j]) + rng.uniform(0.01, 0.99, (cells * per_cell, 2))) * 0.25
    z = grade * xy[:, 0] + rng.normal(0.0, 0.01, cells * per_cell)
    window = surface.Window(level=0, i0=0, j0=0, ni=side, nj=side)
    found = surface.surface_plane(lambda: iter([(np.column_stack([xy, z]), surface.SPLAT)]), window)
    assert found.ground_unobserved == 0
