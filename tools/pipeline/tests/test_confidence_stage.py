"""`confidence_grid`: an epoch's evidence binned into its site's cells, capped by its alignment."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import confidence
import confidence_scene as scene
import quality
import splat_io
from captures_bridge import cell_grid, sigmoid
from conftest import make_recipe
from executor import execute
from runners import RunnerSet
from workdir import Workdir

N = 20_000
ARC = {"degrees": 120.0, "height": 2.5, "radius": 7.0}


def _epoch(
    root: Path, cameras: list[tuple[np.ndarray, np.ndarray]], *, seed: int = 1, **context: Any
) -> Workdir:
    """quality -> place -> confidence over one synthetic epoch."""
    workdir = Workdir.create(root)
    names = scene.write_epoch(
        workdir.root / "inputs", scene.field(N, seed=seed), cameras, **context
    )
    execute(
        make_recipe(
            [
                {"id": "quality", "impl": "support_gate", "params": {"bar": "strict"}},
                {"id": "place", "impl": "place_splat"},
                {"id": "confidence", "impl": "confidence_grid"},
            ],
            inputs=names,
        ),
        workdir,
        RunnerSet.local(),
    )
    return workdir


def _again(
    root: Path,
    epoch: Workdir,
    *,
    prior_epochs: int | None = None,
    alignment: dict[str, Any] | None = None,
    georef: bool = True,
    params: dict[str, Any] | None = None,
) -> Workdir:
    """The confidence stage alone, over another run's support and placement, with the
    site context and alignment asked for."""
    workdir = Workdir.create(root)
    inputs = workdir.root / "inputs"
    shutil.copytree(epoch.out_dir("quality") / "support", inputs / "support")
    shutil.copy(epoch.out_dir("place") / "placement.json", inputs / "placement.json")
    context: dict[str, Any] = {"version": 1, "reference": None}
    if prior_epochs is not None:
        context["priorEpochs"] = prior_epochs
    (inputs / "site_context").mkdir()
    (inputs / "site_context" / "site_context.json").write_text(json.dumps(context))
    names = ["support", "placement.json", "site_context"]
    if georef:
        shutil.copy(epoch.input_path("georef.json"), inputs / "georef.json")
        names.append("georef.json")
    if alignment is not None:
        (inputs / "epoch_alignment.json").write_text(json.dumps(alignment))
        names.append("epoch_alignment.json")
    execute(
        make_recipe(
            [{"id": "confidence", "impl": "confidence_grid", "params": params or {}}], inputs=names
        ),
        workdir,
        RunnerSet.local(),
    )
    return workdir


def _grid(workdir: Workdir) -> tuple[dict[str, Any], cell_grid.CellGrid]:
    folder = workdir.out_dir("confidence") / "confidence"
    descriptor = json.loads((folder / "confidence.json").read_text())
    return descriptor, cell_grid.decode(descriptor, (folder / "confidence.bin").read_bytes())


def _cells(
    plane: cell_grid.CellPlane, x: tuple[float, float], y: tuple[float, float]
) -> np.ndarray:
    """The (j, i) indices of the plane's cells whose centres are inside the rectangle."""
    cell = plane.cell_m
    cx = (np.arange(plane.i0, plane.i0 + plane.ni) + 0.5) * cell
    cy = (np.arange(plane.j0, plane.j0 + plane.nj) + 0.5) * cell
    jj, ii = np.meshgrid(np.arange(plane.nj), np.arange(plane.ni), indexing="ij")
    inside = (cx[ii] >= x[0]) & (cx[ii] <= x[1]) & (cy[jj] >= y[0]) & (cy[jj] <= y[1])
    return np.argwhere(inside)


@pytest.fixture(scope="module")
def arc(tmp_path_factory: pytest.TempPathFactory) -> Workdir:
    return _epoch(tmp_path_factory.mktemp("arc") / "run", scene.arc(24, **ARC))


@pytest.fixture(scope="module")
def full(tmp_path_factory: pytest.TempPathFactory) -> Workdir:
    cameras = scene.orbit(24) + scene.obliques(8)
    return _epoch(tmp_path_factory.mktemp("full") / "run", cameras)


def test_a_half_ring_leaves_the_far_side_weak_and_the_near_side_strong(
    arc: Workdir, full: Workdir
) -> None:
    descriptor, grid = _grid(arc)
    plane = grid.planes[0]
    assert descriptor["frame"] == "tileset" and descriptor["policy"] == "weak-v1"
    assert plane.cell_m == 0.5  # 78 gaussians a square metre: 16 or more in a 0.5 m cell
    weak = cell_grid.weak_mask(plane)
    reasons = plane.data[..., 1]
    # Behind the tree from the cameras' side: few views, and all from one side.
    far = _cells(plane, (-0.6, 0.6), (1.1, 2.4))
    assert weak[far[:, 0], far[:, 1]].mean() >= 0.75
    shy = reasons[far[:, 0], far[:, 1]] & (
        cell_grid.REASON_BITS["views"] | cell_grid.REASON_BITS["spread"]
    )
    assert (shy[weak[far[:, 0], far[:, 1]]] > 0).all()
    # Between the cameras and the tree (left of the pond): seen well, from every side.
    near = _cells(plane, (-2.0, 1.0), (-3.5, -1.5))
    assert weak[near[:, 0], near[:, 1]].mean() <= 0.1
    assert (plane.data[near[:, 0], near[:, 1], 0] > 0).all()
    # The re-scan all the way round sees behind the tree.
    _, grid_full = _grid(full)
    weak_full = cell_grid.weak_mask(grid_full.planes[0])
    far_full = _cells(grid_full.planes[0], (-0.6, 0.6), (1.1, 2.4))
    assert weak_full[far_full[:, 0], far_full[:, 1]].mean() <= 0.25
    summary = descriptor["summary"]
    assert summary["gaussians"] == N and summary["outside"] == 0
    assert summary["cells"] == plane.cells and summary["metric"] is True
    assert summary["weak"] == int(weak.sum())


def test_cells_inside_the_region_with_no_evidence_are_unseen(arc: Workdir) -> None:
    descriptor, grid = _grid(arc)
    plane = grid.planes[0]
    disc = descriptor["roiDisc"]
    np.testing.assert_allclose(disc["centerM"], [0.0, 0.0], atol=1e-3)
    px, py, pr = scene.POND
    cell = plane.cell_m
    unseen = (plane.data[..., 1] & cell_grid.REASON_BITS["unseen"]) > 0
    assert unseen.any()
    for j, i in np.argwhere(unseen):
        cx, cy = (plane.i0 + i + 0.5) * cell, (plane.j0 + j + 0.5) * cell
        # Inside the pond (no gaussians) and inside the disc the cameras pointed at.
        assert np.hypot(cx - px, cy - py) < pr + cell
        assert np.hypot(cx - disc["centerM"][0], cy - disc["centerM"][1]) <= disc["radiusM"]
        assert plane.data[j, i, 0] == 0
    assert cell_grid.weak_mask(plane)[unseen].all()
    assert descriptor["summary"]["unseen"] == int(unseen.sum())
    # An empty cell past the disc is not this epoch's to call unseen.
    empty = plane.data[..., 0] == 0
    assert (empty & ~unseen).any() or not empty.any()


def test_an_unaligned_epoch_is_weak_everywhere_and_says_why(arc: Workdir, tmp_path: Path) -> None:
    capped = _again(
        tmp_path / "capped", arc, prior_epochs=1, alignment=scene.alignment_of(10.0, "unaligned")
    )
    descriptor, grid = _grid(capped)
    plane = grid.planes[0]
    scored = plane.data[..., 0] > 0
    cap = descriptor["alignmentCap"]
    assert cap == {
        "uncertaintyM": 10.0,
        "priorEpochs": 1,
        "factor": 0.025,  # a 0.5 m cell over twice 10 m
        "source": "epoch_alignment.json",
    }
    assert cell_grid.weak_mask(plane)[scored].all()
    assert (plane.data[..., 0][scored] <= 1 + round(254 * 0.025)).all()
    assert ((plane.data[..., 1][scored] & cell_grid.REASON_BITS["alignment"]) > 0).all()
    assert descriptor["summary"]["reasons"]["alignment"] == int(scored.sum())
    # The coarse levels carry it up: no level of an unaligned epoch is anything but weak.
    for coarse in grid.planes[1:]:
        assert cell_grid.weak_mask(coarse)[coarse.data[..., 0] > 0].all()


def test_an_epoch_aligned_well_under_half_a_cell_is_not_capped(
    arc: Workdir, tmp_path: Path
) -> None:
    aligned = _again(tmp_path / "aligned", arc, prior_epochs=1, alignment=scene.alignment_of(0.062))
    first = _again(tmp_path / "first", arc, prior_epochs=0, alignment=scene.alignment_of(10.0))
    d_aligned, g_aligned = _grid(aligned)
    d_first, g_first = _grid(first)
    assert d_aligned["alignmentCap"]["factor"] == 1.0
    assert d_aligned["alignmentCap"]["uncertaintyM"] == 0.062
    # The epoch that starts the frame is not capped whatever its own GPS says: the frame
    # is where it is.
    assert d_first["alignmentCap"] == {
        "uncertaintyM": 0.0,
        "priorEpochs": 0,
        "factor": 1.0,
        "source": "site_context.json",
    }
    for a, b in zip(g_aligned.planes, g_first.planes, strict=True):
        np.testing.assert_array_equal(a.data, b.data)
    assert not ((g_first.planes[0].data[..., 1] & cell_grid.REASON_BITS["alignment"]) > 0).any()
    # The arc run itself had no priorEpochs key and no reference: also the first epoch.
    assert _grid(arc)[0]["alignmentCap"]["factor"] == 1.0


def test_with_no_alignment_the_georeference_caps_a_later_epoch(
    arc: Workdir, tmp_path: Path
) -> None:
    later = _again(tmp_path / "later", arc, prior_epochs=2)
    assert _grid(later)[0]["alignmentCap"] == {
        "uncertaintyM": 5.0,
        "priorEpochs": 2,
        "factor": 0.05,
        "source": "georef.json",
    }
    unknown = _again(tmp_path / "unknown", arc, prior_epochs=1, georef=False)
    assert _grid(unknown)[0]["alignmentCap"]["source"] == "unknown"
    assert _grid(unknown)[0]["alignmentCap"]["uncertaintyM"] == confidence.UNKNOWN_UNCERTAINTY_M


def test_two_epochs_at_different_levels_nest(arc: Workdir, tmp_path: Path) -> None:
    coarse_run = _again(tmp_path / "coarse", arc, params={"min_cell_gaussians": 64})
    _, fine = _grid(arc)
    _, coarse = _grid(coarse_run)
    assert coarse.level == fine.level + 1
    # The finer epoch's pyramid at the coarser one's level is exactly the min over its own
    # finest cells beneath each coarse cell: the same cells, no resampling.
    up = fine.plane(coarse.level)
    base = fine.planes[0]
    for jj in range(up.nj):
        for ii in range(up.ni):
            i, j = up.i0 + ii, up.j0 + jj
            children = [
                int(base.data[cj - base.j0, ci - base.i0, 0])
                for cj in (2 * j, 2 * j + 1)
                for ci in (2 * i, 2 * i + 1)
                if 0 <= ci - base.i0 < base.ni and 0 <= cj - base.j0 < base.nj
            ]
            scored = [s for s in children if s > 0]
            assert int(up.data[jj, ii, 0]) == (min(scored) if scored else 0)
    # Every point of the field falls in a coarse cell and the four fine cells under it.
    rng = np.random.default_rng(0)
    points = rng.uniform(-8, 8, (500, 3))
    i_fine, j_fine = cell_grid.Lattice(fine.level).index(points)
    i_coarse, j_coarse = cell_grid.Lattice(coarse.level).index(points)
    np.testing.assert_array_equal(i_fine >> 1, i_coarse)
    np.testing.assert_array_equal(j_fine >> 1, j_coarse)


def test_the_map_does_not_depend_on_chunks_or_bands_and_is_deterministic(
    arc: Workdir, tmp_path: Path
) -> None:
    """Memory is a chunk of support and a band of the plane: the same bytes whatever the
    chunk or the band, and the same bytes twice (integer sums, no BLAS)."""
    small = _again(tmp_path / "small", arc, params={"chunk_gaussians": 997, "band_cells": 64})
    again = _again(tmp_path / "again", arc)
    first = arc.out_dir("confidence") / "confidence"
    for run in (small, again):
        folder = run.out_dir("confidence") / "confidence"
        for name in ("confidence.bin", "confidence.json"):
            assert (folder / name).read_bytes() == (first / name).read_bytes(), name


def test_a_floater_is_counted_outside_and_does_not_coarsen_the_grid(
    arc: Workdir, tmp_path: Path
) -> None:
    epoch = Workdir.create(tmp_path / "floater")
    support = epoch.root / "support"
    shutil.copytree(arc.out_dir("quality") / "support", support)
    # Move one visible gaussian 3 km east: one sky floater.
    x = np.fromfile(support / "x.bin", dtype="<f4")
    alpha = np.fromfile(support / "alpha.bin", dtype="u1")
    row = int(np.flatnonzero(alpha >= confidence.VISIBLE_ALPHA)[0])
    x[row] = 3000.0
    x.tofile(support / "x.bin")
    moved = _again_from(tmp_path / "moved", support, arc)
    descriptor, grid = _grid(moved)
    _, original = _grid(arc)
    assert descriptor["summary"]["outside"] == 1
    assert grid.level == original.level
    assert grid.planes[0].ni <= original.planes[0].ni + 1


def _again_from(root: Path, support: Path, epoch: Workdir) -> Workdir:
    workdir = Workdir.create(root)
    inputs = workdir.root / "inputs"
    shutil.copytree(support, inputs / "support")
    shutil.copy(epoch.out_dir("place") / "placement.json", inputs / "placement.json")
    execute(
        make_recipe(
            [{"id": "confidence", "impl": "confidence_grid"}], inputs=["support", "placement.json"]
        ),
        workdir,
        RunnerSet.local(),
    )
    return workdir


# --- the pure parts --------------------------------------------------------------------


def test_the_alignment_factor_is_a_cell_over_twice_the_uncertainty() -> None:
    assert confidence.alignment_factor(1.0, 0.062) == 1.0
    assert confidence.alignment_factor(1.0, 10.0) == pytest.approx(0.05)
    assert confidence.alignment_factor(2.0, 1.0) == 1.0
    assert confidence.alignment_factor(0.5, 0.0) == 1.0


def test_prior_epochs_default_from_the_reference() -> None:
    assert confidence.prior_epochs(None) == 0
    assert confidence.prior_epochs({"reference": None}) == 0
    assert confidence.prior_epochs({"reference": {"epochId": "e1"}}) == 1
    assert confidence.prior_epochs({"reference": {"epochId": "e1"}, "priorEpochs": 3}) == 3
    assert confidence.prior_epochs({"reference": None, "priorEpochs": True}) == 0


def test_the_level_is_the_finest_whose_median_cell_holds_sixteen() -> None:
    counts = np.zeros((4, 4), dtype=np.int64)
    counts[:2, :2] = 5  # 4 occupied cells of 5: one level up, one cell of 20
    assert confidence.choose_level(counts, 0, 0, 0) == 1
    counts[:] = 16
    assert confidence.choose_level(counts, 0, 0, 0) == 0
    assert confidence.choose_level(np.zeros((3, 3), dtype=np.int64), 2, -1, -1) == 2


def _binned(alpha: int, keep: int, failed: list[int], z: tuple[float, float]) -> confidence.Binned:
    return confidence.Binned(
        level=2,
        i0=0,
        j0=0,
        ni=1,
        nj=1,
        count=np.array([10], dtype=np.int64),
        alpha=np.array([alpha], dtype=np.int64),
        keep=np.array([keep], dtype=np.int64),
        failed=np.array([failed], dtype=np.int64),
        zmin=np.array([z[0]]),
        zmax=np.array([z[1]]),
        outside=0,
        binned=10,
    )


def test_a_cell_is_weak_below_a_keep_majority_and_says_the_majoritys_reasons() -> None:
    none = np.zeros((1, 1), dtype=bool)
    half = confidence.score_plane(_binned(200, 100, [51, 50, 0, 0], (0.0, 1.0)), 1.0, none)
    assert half.data[0, 0, 0] == 128  # 1 + round(254 x 0.5): exactly half is not weak
    assert not cell_grid.weak_mask(half)[0, 0]
    # More than half of the non-keep alpha failed views; exactly half failed spread: not a reason.
    assert half.data[0, 0, 1] == cell_grid.REASON_BITS["views"]
    assert half.data[0, 0, 2] == 4  # a metre of relief in 0.25 m
    short = confidence.score_plane(_binned(200, 98, [0, 0, 60, 0], (0.0, 0.0)), 1.0, none)
    assert short.data[0, 0, 0] == 125 and cell_grid.weak_mask(short)[0, 0]
    assert short.data[0, 0, 1] == cell_grid.REASON_BITS["gsd"]
    capped = confidence.score_plane(_binned(200, 200, [0, 0, 0, 0], (0.0, 0.0)), 0.25, none)
    assert capped.data[0, 0, 0] == 1 + round(254 * 0.25)
    assert capped.data[0, 0, 1] == cell_grid.REASON_BITS["alignment"]


def test_the_support_reason_bits_are_the_grids() -> None:
    for name, bit in quality.SUPPORT_REASON_BITS.items():
        assert cell_grid.REASON_BITS[name] == bit


# --- package and catalog -----------------------------------------------------------------


def _package(root: Path, epoch: Workdir, *, grid: Path | None) -> Workdir:
    """package -> register over an epoch's canonical.ply, with or without its grid."""
    workdir = Workdir.create(root)
    inputs = workdir.root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copy(epoch.out_dir("place") / "canonical.ply", inputs / "canonical.ply")
    shutil.copy(epoch.input_path("georef.json"), inputs / "georef.json")
    names = ["canonical.ply", "georef.json"]
    if grid is not None:
        shutil.copytree(grid, inputs / "confidence")
        names.append("confidence")
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


def test_package_ships_the_grid_and_catalog_registers_it(arc: Workdir, tmp_path: Path) -> None:
    grid_dir = arc.out_dir("confidence") / "confidence"
    shipped = _package(tmp_path / "with", arc, grid=grid_dir)
    plain = _package(tmp_path / "without", arc, grid=None)

    splat = shipped.out_dir("package") / "splat"
    tileset = json.loads((splat / "tileset.json").read_text())
    descriptor = json.loads((grid_dir / "confidence.json").read_text())
    assert tileset["root"]["extras"]["confidence"] == descriptor
    assert (splat / "confidence.bin").read_bytes() == (grid_dir / "confidence.bin").read_bytes()
    assert "confidenceMin" in tileset["root"]["extras"]
    registration = json.loads((shipped.out_dir("register") / "registration.json").read_text())
    assert registration["confidence"] == descriptor
    assert "confidence.bin" in registration["artifacts"]
    # Not measured: no grid, no key on any tile, and the registration says null.
    plain_tiles = json.loads((plain.out_dir("package") / "splat" / "tileset.json").read_text())
    assert "confidence" not in plain_tiles["root"]["extras"]
    assert "confidenceMin" not in json.dumps(plain_tiles)
    plain_registration = json.loads((plain.out_dir("register") / "registration.json").read_text())
    assert plain_registration["confidence"] is None
    for name in ("splat.glb", "collision.bin", "viewcones.bin"):
        assert (splat / name).read_bytes() == (
            plain.out_dir("package") / "splat" / name
        ).read_bytes()


def test_a_grid_package_cannot_read_is_left_out_not_fatal(arc: Workdir, tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(arc.out_dir("confidence") / "confidence", broken)
    raw = bytearray((broken / "confidence.bin").read_bytes())
    raw[-12] ^= 0xFF  # inside the deflate stream: the planes no longer hash to the sha256
    (broken / "confidence.bin").write_bytes(bytes(raw))
    run = _package(tmp_path / "run", arc, grid=broken)
    splat = run.out_dir("package") / "splat"
    assert "confidence" not in json.loads((splat / "tileset.json").read_text())["root"]["extras"]
    assert not (splat / "confidence.bin").exists()
    log = run.log_path("package").read_text()
    assert "confidence grid was not shipped" in log
    registration = json.loads((run.out_dir("register") / "registration.json").read_text())
    assert registration["confidence"] is None


def test_the_demo_runs_and_reports_the_rescan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from experiments import confidence_demo

    monkeypatch.setattr(confidence_demo, "GAUSSIANS", 20_000)
    assert confidence_demo.main(["--out", str(tmp_path / "demo")]) == 0

    text = capsys.readouterr().out
    report = json.loads((tmp_path / "demo" / "report.json").read_text())
    names = [row["epoch"] for row in report["epochs"]]
    assert names == ["e1 (south arc)", "e2 (re-scan)", "e2 if unaligned"]
    unaligned = report["epochs"][2]
    assert unaligned["weakShare"] == 1.0 and "alignment" in unaligned["reasons"]
    change = report["rescan"]
    assert change["strongAfterRescan"] > 0 and change["stillWeak"] > 0
    assert change["coveredByE2"] <= change["weakInE1"]
    assert "Every capture here is synthetic" in text
    for name in ("e1-splat", "e2-splat"):
        tileset = json.loads((tmp_path / "demo" / name / "tileset.json").read_text())
        assert tileset["root"]["extras"]["confidence"]["kind"] == "confidence"


def test_the_grid_is_in_the_frame_canonical_ply_and_the_tiles_are_in(tmp_path: Path) -> None:
    """A recentred, rotated, scaled epoch moved by a site-frame offset: every gaussian
    `place` published lands in a scored cell of the grid, so the grid is in the frame the
    tileset's root transform places on the globe, not the trainer's."""
    import gaussians

    workdir = Workdir.create(tmp_path / "moved")
    names = scene.write_epoch(workdir.root / "inputs", scene.field(N, seed=2), scene.orbit(24))
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    georef = json.loads(workdir.input_path("georef.json").read_text())
    georef["frame"] = {
        "source": "camera-up",
        "scale": 2.0,
        "rotation": turn.tolist(),
        "translationM": None,
        "recentre": True,
    }
    georef.pop("scaleSource")
    georef["siteFrame"] = {"offsetM": [30.0, -12.0, 1.0]}
    workdir.input_path("georef.json").write_text(json.dumps(georef))
    execute(
        make_recipe(
            [
                {"id": "quality", "impl": "support_gate", "params": {"bar": "balanced"}},
                {"id": "place", "impl": "place_splat"},
                {"id": "confidence", "impl": "confidence_grid"},
            ],
            inputs=names,
        ),
        workdir,
        RunnerSet.local(),
    )
    _, grid = _grid(workdir)
    placed = gaussians.read_splat(workdir.out_dir("place") / "canonical.ply")
    visible = sigmoid(placed.columns["opacity"]) >= 0.05
    xyz = placed.xyz[visible]
    assert abs(float(np.median(xyz[:, 0])) - 30.0) < 3.0  # really moved
    scores = cell_grid.lookup(grid, xyz)[:, 0]
    # A gaussian exactly on a cell edge may round into its neighbour (ADR 0016).
    assert (scores > 0).mean() > 0.999
    assert grid.planes[0].cell_m == 1.0  # twice the scale: a quarter of the density


@pytest.mark.parametrize("damage", ["mid-stream", "missing-file", "no-score", "not-json"])
def test_any_broken_grid_is_left_out_by_package(arc: Workdir, tmp_path: Path, damage: str) -> None:
    """A grid damaged any way at all -- a deflate stream zlib cannot read, a file that is
    gone, a descriptor of the wrong shape -- is left out, and the tiles still ship."""
    broken = tmp_path / "broken"
    shutil.copytree(arc.out_dir("confidence") / "confidence", broken)
    if damage == "mid-stream":
        raw = bytearray((broken / "confidence.bin").read_bytes())
        raw[40] ^= 0xFF
        raw[41] ^= 0xFF
        (broken / "confidence.bin").write_bytes(bytes(raw))
    elif damage == "missing-file":
        (broken / "confidence.bin").unlink()
    elif damage == "no-score":
        descriptor = json.loads((broken / "confidence.json").read_text())
        descriptor["channels"][0]["name"] = "points"
        (broken / "confidence.json").write_text(json.dumps(descriptor))
    else:
        (broken / "confidence.json").write_text('{"format": "hexapod.cellgrid", "levels": ')
    run = _package(tmp_path / "run", arc, grid=broken)
    splat = run.out_dir("package") / "splat"
    assert (splat / "splat.glb").is_file()
    assert "confidence" not in json.loads((splat / "tileset.json").read_text())["root"]["extras"]
    registration = json.loads((run.out_dir("register") / "registration.json").read_text())
    assert registration["confidence"] is None


def _flat_support(folder: Path, n: int, side_m: float) -> None:
    """A `support/` of `n` visible gaussians spread evenly over a `side_m` square."""
    rng = np.random.default_rng(3)
    store = splat_io.ColumnStore(folder)
    columns = {
        "x": rng.uniform(0, side_m, n).astype(np.float32),
        "y": rng.uniform(0, side_m, n).astype(np.float32),
        "z": rng.uniform(0, 2, n).astype(np.float32),
        "alpha": np.full(n, 200, np.uint8),
        "tier": rng.integers(0, 3, n).astype(np.uint8),
        "reasons": rng.integers(0, 16, n).astype(np.uint8),
        "views": np.full(n, 9, np.uint16),
        "spreadDeg": np.full(n, 50, np.uint8),
        "gsdMm": np.full(n, np.nan, np.float16),
    }
    for name, dtype in quality.SUPPORT_COLUMNS:
        store.create(name, dtype).append(columns[name])
    store.finish()
    document = {
        "version": 1,
        "frame": "colmap",
        "rows": n,
        "status": "measured",
        "columns": {
            name: {"dtype": d, "file": f"{name}.bin"} for name, d in quality.SUPPORT_COLUMNS
        },
        "roi": None,
    }
    (folder / quality.SUPPORT_FILE).write_text(json.dumps(document))


def test_the_sums_are_held_a_band_at_a_time_not_for_the_whole_plane(tmp_path: Path) -> None:
    """Review finding: the band loop once bounded nothing (every accumulator was the whole
    plane's, 72 bytes a cell). Each band now allocates its own; the peak is the finest
    plane's counts plus one band, and the plane is the same bytes whatever the band."""
    import tracemalloc

    _flat_support(tmp_path / "support", 300_000, 128.0)
    support = confidence.read_support(tmp_path / "support", chunk=1 << 14)
    identity = confidence.Placement(1.0, np.eye(3), np.zeros(3))
    planes = {}
    peaks = {}
    for band in (1 << 12, 1 << 20):
        tracemalloc.start()
        scored = confidence.bin_support(
            support, identity, None, 0.0, min_cell_gaussians=1, max_cells=1 << 18, band_cells=band
        )
        peaks[band] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert scored is not None
        planes[band] = scored.plane
    cells = planes[1 << 12].cells
    assert cells == 512 * 512  # 0.25 m cells over 128 m: the whole 2^18
    np.testing.assert_array_equal(planes[1 << 12].data, planes[1 << 20].data)
    # The whole plane's accumulators alone would be 72 bytes a cell (18.9 MB here). With
    # one band that big the peak passes it (measured 42 MB); with small bands the whole
    # stage stays under it (measured 11 MB: the counts plane and outofcore's fixed
    # selection buffer, not the sums).
    whole = 72 * cells
    assert peaks[1 << 20] > whole
    assert peaks[1 << 12] < whole, (peaks, whole)
