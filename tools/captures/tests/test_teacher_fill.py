"""`teacher_fill` (Teacher B) on the committed synthetic yard, with the CPU stand-in filler.

What must hold whatever the image model is: a dropped region is masked where it was, the
fill is scored against what was there, a fill that repaints what it was told is refused,
lifted gaussians sit at the scan's depth facing their camera, and the inferred layer is its
own tileset -- in the measured frame, with view cones from its virtual cameras and
`extras.evidence`. The fill from outside is gated (`rescan_gate`): every `fill_scan` here
is given a real gate, built from two confidence grids (an earlier epoch A weak everywhere,
its re-scan B weak in part) the way the API builds one, and fills only B's still-weak cells.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import gate_fixtures as gf
import numpy as np
import pytest

import splat_tiles
import teacher_fill as tf
import view_cones as vc
from rescan_gate import RescanGate, RescanGateError, verify_layer
from splat_render import Camera, Splats, load_ply, load_tileset, save_ply

YARD = Path(__file__).resolve().parents[3] / "data" / "tiles" / "synthetic-yard"


@pytest.fixture(scope="module")
def yard(tmp_path_factory: pytest.TempPathFactory) -> tuple[Splats, vc.ConeGrid]:
    """The yard's source PLY: the checkout's when generated (it is gitignored), else a fresh
    one, as `test_segment_scene` does."""
    ply = YARD / "source" / "splat.ply"
    if not ply.exists():
        import synthetic_yard
        from synthetic_tree import write_ply

        ply = tmp_path_factory.mktemp("yard") / "splat.ply"
        write_ply(ply, synthetic_yard.generate_yard()[0])
    layout = splat_tiles.ply_layout(ply)
    grid = vc.cone_grid(layout, np.ones(layout.count, bool))
    return load_ply(ply), grid


@pytest.fixture(scope="module")
def centre(yard: tuple[Splats, vc.ConeGrid]) -> np.ndarray:
    """A patch of what the capture saw: the densest gaussians near its main observer."""
    splats, grid = yard
    observer = grid.observers[int(np.argmax(grid.observer_weights))]
    near = np.argsort(np.linalg.norm(splats.positions - observer, axis=1))[:500]
    return splats.positions[near].mean(axis=0)


@pytest.fixture(scope="module")
def report(yard: tuple[Splats, vc.ConeGrid], centre: np.ndarray, tmp_path_factory) -> tf.DropReport:
    splats, grid = yard
    out = tmp_path_factory.mktemp("drop")
    return tf.drop_and_fill(
        splats,
        centre,
        0.4,
        tf.InpaintFiller(),
        grid,
        views=2,
        width=200,
        height=150,
        save_dir=out,
    )


def test_drop_and_fill_scores_the_fill_against_what_was_there(report: tf.DropReport) -> None:
    doc = report.to_json()
    assert doc["filler"] == "opencv-telea"
    assert doc["region"]["dropped"] > 0
    assert report.views, "the dropped region must show in the fill views"
    for view in report.views:
        assert view["maskPx"] > 0
        assert view["gatePsnr"] >= tf.GATE_PSNR_DB  # inpainting leaves unmasked pixels be
        assert np.isfinite(view["psnrFill"]) and np.isfinite(view["psnrHole"])
    assert report.lifted > 0
    json.dumps(doc)


class _Repaint:
    """A filler that ignores the mask and repaints the whole frame."""

    name = "repaint"

    def fill(self, rgb: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
        return [255 - rgb]


def test_the_gate_refuses_a_fill_that_repaints_what_it_was_told(
    yard: tuple[Splats, vc.ConeGrid], centre: np.ndarray
) -> None:
    splats, grid = yard
    camera = tf.plan_views(grid, centre[None], count=1, width=120, height=90, distance_m=3.0)[0]
    mask = np.zeros((90, 120), bool)
    mask[40:50, 55:65] = True
    cond = tf.condition(splats, camera, None, mask=mask)
    refused = tf.fill_views([cond], _Repaint())[0]
    kept = tf.fill_views([cond], tf.InpaintFiller())[0]
    assert not refused.accepted and kept.accepted
    lifted, _ = tf.lift([refused])
    assert len(lifted) == 0


class _FullRender:
    """A filler shown the full render (as Fixer is): `invert` repaints it, else it softens it,
    as a model that re-renders the frame does."""

    name = "full"
    reads_full_render = True

    def __init__(self, invert: bool) -> None:
        self.invert = invert

    def fill(self, rgb: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
        import cv2

        return [255 - rgb if self.invert else cv2.GaussianBlur(rgb, (0, 0), 1.0)]


def test_a_full_render_filler_is_gated_on_the_frames_layout(
    yard: tuple[Splats, vc.ConeGrid], centre: np.ndarray
) -> None:
    splats, grid = yard
    camera = tf.plan_views(grid, centre[None], count=1, width=120, height=90, distance_m=3.0)[0]
    mask = np.zeros((90, 120), bool)
    mask[40:50, 55:65] = True
    cond = tf.condition(splats, camera, None, mask=mask)
    softened = tf.fill_views([cond], _FullRender(invert=False))[0]
    repainted = tf.fill_views([cond], _FullRender(invert=True))[0]
    assert softened.accepted and not repainted.accepted
    assert repainted.gate_psnr_db < 10.0


def test_lift_puts_discs_at_the_scan_depth_facing_the_camera() -> None:
    camera = Camera.look_at([0.0, -5.0, 1.0], [0.0, 0.0, 1.0], width=40, height=30)
    h, w = 30, 40
    mask = np.zeros((h, w), bool)
    mask[10:20, 10:30] = True
    depth = np.full((h, w), 5.0)
    distance = np.zeros((h, w), np.float32)
    frame = tf.Frame(np.zeros((h, w, 3)), depth, np.ones((h, w)), np.zeros((h, w)), np.ones((h, w)))
    cond = tf.Conditioning(camera, frame, frame, mask, depth, distance)
    rgb = np.full((h, w, 3), 200, np.uint8)
    lifted, confidence = tf.lift([tf.Filled(cond, rgb, 40.0, True)], stride=2)
    assert len(lifted) == 5 * 10
    # On the plane five metres in front of the camera.
    assert np.allclose(lifted.positions[:, 1], 0.0, atol=1e-9)
    # Discs: thin along the normal, which points back at the camera.
    assert np.all(lifted.scales[:, 2] < lifted.scales[:, 0])
    w_, x, y, z = lifted.rotations.T
    normal = np.column_stack([2 * (x * z + w_ * y), 2 * (y * z - w_ * x), 1 - 2 * (x * x + y * y)])
    to_camera = camera.centre - lifted.positions
    to_camera /= np.linalg.norm(to_camera, axis=1, keepdims=True)
    assert np.allclose(np.sum(normal * to_camera, axis=1), 1.0, atol=1e-9)
    assert np.allclose(lifted.colours, 200 / 255)
    assert np.allclose(confidence, 1.0) and np.allclose(lifted.opacities, 0.95)


def _wall(y: float, colour: tuple[float, float, float], n: int, seed: int) -> Splats:
    rng = np.random.default_rng(seed)
    positions = np.column_stack(
        [rng.uniform(-1.5, 1.5, n), y + rng.normal(0, 0.005, n), rng.uniform(-0.5, 2.5, n)]
    )
    return Splats(
        positions,
        np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        np.full((n, 3), 0.04),
        np.tile(colour, (n, 1)),
        np.full(n, 0.95),
    )


def test_a_dropped_region_is_lifted_where_it_was_not_behind_it() -> None:
    """A hole cut in a near wall shows the far wall through it: the drop test's fill must
    land on the near wall (interpolated from around the hole), not on the far one."""
    near, far = _wall(0.0, (0.8, 0.2, 0.2), 12000, 1), _wall(3.0, (0.2, 0.2, 0.8), 12000, 2)
    scene = Splats.concat([near, far])
    inside = np.all(np.abs(scene.positions - [0.0, 0.0, 1.0]) <= 0.3, axis=1)
    kept, dropped = scene.take(np.flatnonzero(~inside)), scene.take(np.flatnonzero(inside))
    camera = Camera.look_at([0.0, -5.0, 1.0], [0.0, 0.0, 1.0], width=80, height=60)
    hole = tf._hole(tf.render(dropped, camera).alpha)
    assert hole.sum() > 50
    ones = np.ones(len(kept))
    behind = tf.condition(kept, camera, None, mask=hole, seen_opacity=ones, hole_depth="scan")
    around = tf.condition(kept, camera, None, mask=hole, seen_opacity=ones, hole_depth="surround")
    assert np.nanmedian(behind.depth[hole]) == pytest.approx(8.0, abs=0.3)
    assert np.all(np.isfinite(around.depth[hole]))
    assert np.allclose(around.depth[hole], 5.0, atol=0.3)
    # Narrowed to the scan just outside the region: its surface, even where the image
    # around the hole is mostly the far wall.
    reach = np.all(np.abs(scene.positions - [0.0, 0.0, 1.0]) <= 0.6, axis=1)
    shell = scene.take(np.flatnonzero(reach & ~inside))
    hole_far = hole.copy()
    shelled = tf.condition(
        far,
        camera,
        None,
        mask=hole_far,
        seen_opacity=np.ones(len(far)),
        hole_depth="surround",
        surround=shell,
    )
    assert np.allclose(shelled.depth[hole], 5.0, atol=0.3)
    rgb = np.full((60, 80, 3), 200, np.uint8)
    lifted, _ = tf.lift([tf.Filled(around, rgb, 40.0, True)])
    assert np.allclose(lifted.positions[:, 1], 0.0, atol=0.3)
    with pytest.raises(ValueError):
        tf.condition(kept, camera, None, mask=hole, seen_opacity=ones, hole_depth="nope")


def test_make_renderer_is_the_cpu_unless_asked() -> None:
    assert tf.make_renderer("cpu") is tf.render
    with pytest.raises(ValueError):
        tf.make_renderer("opengl")


def test_confidence_falls_away_from_what_was_measured() -> None:
    camera = Camera.look_at([0.0, -5.0, 1.0], [0.0, 0.0, 1.0], width=40, height=30)
    h, w = 30, 40
    mask = np.ones((h, w), bool)
    distance = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    depth = np.full((h, w), 5.0)
    frame = tf.Frame(np.zeros((h, w, 3)), depth, np.ones((h, w)), np.zeros((h, w)), np.ones((h, w)))
    cond = tf.Conditioning(camera, frame, frame, mask, depth, distance)
    _, confidence = tf.lift([tf.Filled(cond, np.zeros((h, w, 3), np.uint8), 40.0, True)], 1)
    row = confidence[:w]
    assert row[0] == pytest.approx(1.0) and row[12] == pytest.approx(0.5, rel=1e-6)
    assert np.all(np.diff(row) < 0)


def test_plan_views_spreads_cameras_and_rings_the_target() -> None:
    target = np.array([[0.0, 0.0, 0.0]])
    observers = [np.array([5.0, 0.0, 1.0]), np.array([5.1, 0.0, 1.0]), np.array([0.0, 5.0, 1.0])]
    near = tf.plan_views(None, target, cameras=observers, count=2, distance_m=2.0)
    centres = np.array([c.centre for c in near])
    assert np.allclose(np.linalg.norm(centres, axis=1), 2.0)
    assert np.linalg.norm(centres[0] - centres[1]) > 2.0  # the spread pair, not the twins
    ring = tf.plan_views(None, target, count=6, mode="ring", ring_radius_m=4.0)
    assert len(ring) == 6
    for camera in ring:
        uv, _ = camera.project(target)
        assert np.allclose(uv[0], [camera.width / 2, camera.height / 2], atol=1e-6)
    with pytest.raises(ValueError):
        tf.plan_views(None, target)


def test_package_inferred_is_its_own_layer_in_the_measured_frame(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 400
    positions = rng.uniform([-1, -1, 0], [1, 1, 0.1], size=(n, 3))
    lifted = Splats(
        positions,
        np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        np.full((n, 3), 0.02),
        rng.uniform(0, 1, size=(n, 3)),
        np.full(n, 0.8),
    )
    cameras = [Camera.look_at([4.0, 0.0, 2.0], [0.0, 0.0, 0.0], width=64, height=48)]
    measured = YARD / "splat" / "tileset.json"
    out = tmp_path / "inferred"
    evidence = tf.package_inferred(lifted, np.full(n, 0.7), cameras, measured, out, "test")
    doc = json.loads((out / "tileset.json").read_text())
    root = doc["root"]
    assert root["transform"] == json.loads(measured.read_text())["root"]["transform"]
    assert root["extras"]["evidence"] == evidence
    assert evidence["kind"] == "inferred" and evidence["gaussians"] == n
    assert evidence["meanConfidence"] == pytest.approx(0.7)
    assert (out / vc.URI).exists() and root["extras"]["viewCones"]["uri"] == vc.URI
    assert not any(p.name.startswith(".") for p in tmp_path.iterdir())
    # Seen from its one camera's side only: the cones written point back at it.
    meta = root["extras"]["viewCones"]
    assert meta["observers"] == "cameras"
    dims = tuple(meta["dims"])
    texels = vc.decode_view_cones((out / vc.URI).read_bytes(), dims)
    at = vc.lookup(texels, np.array(meta["origin"]), meta["cell"], dims, positions)
    seen = vc.visibility(at, positions - [4.0, 0.0, 2.0])
    behind = vc.visibility(at, positions - [-4.0, 0.0, 2.0])
    assert seen.mean() > 0.9 and behind.mean() < 0.1


def _measured_and_gate(
    tmp_path: Path,
    ply: Path,
    positions: np.ndarray,
    still_weak: Callable[[np.ndarray, np.ndarray], np.ndarray],
    name: str = "measured",
) -> tuple[Path, RescanGate]:
    """`ply` packed as epoch B's measured tileset with B's confidence grid (1 m cells over
    the scene), and B's fill targets against an earlier epoch A that was weak everywhere:
    B is weak (so the targets are) where `still_weak(i, j)`, strong elsewhere."""
    low = np.floor(positions[:, :2].min(axis=0)).astype(int) - 1
    high = np.floor(positions[:, :2].max(axis=0)).astype(int) + 1
    (i0, j0), (ni, nj) = low, high - low + 1
    jj, ii = np.meshgrid(np.arange(j0, j0 + nj), np.arange(i0, i0 + ni), indexing="ij")
    weak = still_weak(ii, jj)
    a = gf.confidence_grid(2, int(i0), int(j0), np.full((nj, ni), 60), np.ones((nj, ni)))
    b = gf.confidence_grid(2, int(i0), int(j0), np.where(weak, 60, 200), np.where(weak, 1, 0))
    measured = tmp_path / name
    grid = gf.grid_json(tmp_path / f"{name}-grid", b)
    splat_tiles.convert(ply, measured, 0.0, 0.0, 0.0, opacity_min=0.0, confidence=grid)
    targets = gf.write_payload(tmp_path / f"{name}-targets.json", gf.still_weak_payload(b, [a]))
    return measured, RescanGate.from_json(targets)


def _north(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return j >= 0


def _everywhere(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return np.ones(i.shape, dtype=bool)


def _nowhere(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return np.zeros(i.shape, dtype=bool)


def _photographed_wall(n: int = 6000) -> Splats:
    """A wall in the y-z plane at x = 0, 2 m long north-south and 2 m high."""
    rng = np.random.default_rng(1)
    positions = np.column_stack(
        [rng.normal(0, 0.01, n), rng.uniform(-1, 1, n), rng.uniform(0, 2, n)]
    )
    return Splats(
        positions,
        np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        np.full((n, 3), 0.03),
        np.tile([0.6, 0.3, 0.2], (n, 1)),
        np.full(n, 0.9),
    )


def test_fill_scan_fills_the_side_a_capture_never_saw(tmp_path: Path) -> None:
    """A wall photographed from +x only: from -x it is faded, so the ring's far cameras
    are asked to fill it, and the inferred layer is drawn only from their side. Every cell
    of it is still weak after a re-scan, so the gate leaves all of it to fill."""
    wall = _photographed_wall()
    ply = tmp_path / "wall.ply"
    save_ply(ply, wall, comment="test wall")
    measured, gate = _measured_and_gate(tmp_path, ply, wall.positions, _everywhere)
    layout = splat_tiles.ply_layout(ply)
    eye = np.array([[6.0, 0.0, 1.0]])
    grid = vc.cone_grid(layout, np.ones(layout.count, bool), observers=eye)
    out = tmp_path / "inferred"
    evidence = tf.fill_scan(
        wall,
        grid,
        tf.InpaintFiller(),
        measured / "tileset.json",
        out,
        gate=gate,
        views=4,
        width=96,
        height=72,
        stride=2,
    )
    assert evidence["gaussians"] > 0, evidence
    assert evidence["gated"]["targets"] == evidence["gated"]["faded"] > 0
    accepted = [v for v in evidence["perView"] if v["accepted"]]
    assert accepted and all(v["maskPx"] > 0 for v in accepted)
    assert evidence["views"] == len(accepted)
    lifted = load_tileset(out / "tileset.json")
    # On the wall, and only in its cells. (Ungated, a few dozen discs at the silhouette's
    # edge were lifted metres off it towards the cameras; their mean was what once said
    # "towards the cameras". The gate drops them: they are in no target cell.)
    assert abs(float(np.median(lifted.positions[:, 0]))) < 0.05
    assert gate.mask(lifted.positions).all()
    assert evidence["gated"]["liftedOutside"] > 0


class _Painter:
    """Paints what it is asked in one colour; chains its views and reads the void."""

    name = "painter"
    chain_views = True
    reads_void = True

    def __init__(self) -> None:
        self.context: dict | None = None
        self.asked: list[int] = []
        self.voids: list[int] = []

    def fill(self, rgb: np.ndarray, mask: np.ndarray, void: np.ndarray) -> list[np.ndarray]:
        assert not (void & mask).any()
        self.asked.append(int(mask.sum()))
        self.voids.append(int(void.sum()))
        return [np.where(mask[..., None], np.array([30, 200, 30], np.uint8), rgb)]


def test_fill_scan_chains_a_filler_that_asks_and_tells_it_the_void(tmp_path: Path) -> None:
    wall = _photographed_wall()
    ply = tmp_path / "wall.ply"
    save_ply(ply, wall, comment="test wall")
    measured, gate = _measured_and_gate(tmp_path, ply, wall.positions, _north)
    layout = splat_tiles.ply_layout(ply)
    grid = vc.cone_grid(layout, np.ones(layout.count, bool), observers=np.array([[6.0, 0, 1]]))
    painter = _Painter()
    evidence = tf.fill_scan(
        wall, grid, painter, measured / "tileset.json", tmp_path / "inferred",
        gate=gate, views=6, width=96, height=72, stride=2,
    )  # fmt: skip
    asked_total = [v["maskPx"] for v in evidence["perView"]]
    assert len(painter.asked) == 6 and all(v > 0 for v in painter.voids)
    # Later views reuse what earlier ones filled: asked less than they lift.
    assert sum(painter.asked) < sum(asked_total)
    assert evidence["gaussians"] > 0


def test_fill_scan_can_condition_without_floaters(tmp_path: Path) -> None:
    """`max_scale_m` leaves gaussians larger than that out of every view it renders (a
    floater gsplat would draw over the frame; the CPU renderer barely shows it)."""
    wall = _wall(0.0, (0.6, 0.3, 0.2), 4000, 3)
    floater = Splats(
        np.array([[0.0, -2.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0, 0.0]]),
        np.full((1, 3), 3.0),
        np.array([[0.9, 0.9, 0.9]]),
        np.array([0.9]),
    )
    scene = Splats.concat([wall, floater])
    ply = tmp_path / "scene.ply"
    save_ply(ply, scene, comment="test floater")
    # The wall runs east-west at y = 0: its east half is still weak.
    measured, gate = _measured_and_gate(tmp_path, ply, scene.positions, lambda i, j: i >= 0)
    layout = splat_tiles.ply_layout(ply)
    grid = vc.cone_grid(layout, np.ones(layout.count, bool), observers=np.array([[0.0, -6.0, 1.0]]))
    largest: list[float] = []

    def spy(splats: Splats, camera: Camera, **kwargs: object) -> tf.Frame:
        largest.append(float(splats.scales.max()))
        return tf.render(splats, camera, **kwargs)  # type: ignore[arg-type]

    evidence = tf.fill_scan(
        scene, grid, tf.InpaintFiller(), measured / "tileset.json", tmp_path / "inferred",
        gate=gate, views=4, width=64, height=48, renderer=spy, max_scale_m=1.0,
    )  # fmt: skip
    assert evidence["gaussians"] > 0
    assert largest and max(largest) < 1.0


def _wall_case(tmp_path: Path, still_weak: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    wall = _photographed_wall()
    ply = tmp_path / "wall.ply"
    save_ply(ply, wall, comment="test wall")
    measured, gate = _measured_and_gate(tmp_path, ply, wall.positions, still_weak)
    layout = splat_tiles.ply_layout(ply)
    grid = vc.cone_grid(layout, np.ones(layout.count, bool), observers=np.array([[6.0, 0, 1]]))
    return wall, measured / "tileset.json", gate, grid


def test_fill_scan_will_not_run_without_a_gate(tmp_path: Path) -> None:
    wall, tileset, _, grid = _wall_case(tmp_path, _north)
    with pytest.raises(TypeError, match="gate"):
        tf.fill_scan(wall, grid, tf.InpaintFiller(), tileset, tmp_path / "inferred")  # type: ignore[call-arg]
    assert not (tmp_path / "inferred").exists()


def test_fill_scan_fills_only_the_cells_a_re_scan_left_weak(tmp_path: Path) -> None:
    """The wall's north half is still weak, its south half was fixed by the re-scan: the
    fill is asked only for the north, every gaussian of the layer is in a target cell,
    and its evidence names the gate it was held to."""
    wall, tileset, gate, grid = _wall_case(tmp_path, _north)
    out = tmp_path / "inferred"
    evidence = tf.fill_scan(
        wall, grid, tf.InpaintFiller(), tileset, out,
        gate=gate, views=4, width=96, height=72, stride=2,
    )  # fmt: skip
    gated = evidence["gated"]
    assert 0 < gated["targets"] < gated["faded"]
    assert gated["targets"] == int((gate.mask(wall.positions)).sum())
    assert evidence["gaussians"] > 0, evidence
    layer = load_tileset(out / "tileset.json")
    assert gate.mask(layer.positions).all()
    assert (layer.positions[:, 1] >= 0).all()
    stored = json.loads((out / "tileset.json").read_text())["root"]["extras"]["evidence"]
    confidence = json.loads(tileset.read_text())["root"]["extras"]["confidence"]
    assert stored["rescanGate"] == gate.evidence(confidence_sha256=confidence["sha256"])
    assert stored["rescanGate"]["cells"] == gate.cells > 0
    assert set(stored["rescanGate"]) >= {"policy", "epochId", "weakIn", "cells", "maskSha256"}
    # What publish asks of it: its tiles' grid and gate hold, against targets fetched now.
    asset = {"epochId": gate.epoch_id, "siteId": gate.site_id, "validTo": None}
    current = json.loads(tileset.read_text())
    a, b = (
        gf.confidence_grid(
            2, gate.i0, gate.j0, np.full(gate.targets.shape, score), np.ones(gate.targets.shape)
        )
        for score in (60, 60)
    )
    then = json.loads((tmp_path / "measured-targets.json").read_text())
    verify_layer(stored, asset=asset, tileset=current, current=then, positions=layer.positions)
    now = gf.still_weak_payload(b, [a])  # the targets grew since: every cell is weak now
    with pytest.raises(RescanGateError, match="changed since"):
        verify_layer(stored, asset=asset, tileset=current, current=now, positions=layer.positions)


def test_fill_scan_with_no_cell_still_weak_is_skipped_not_refused(tmp_path: Path) -> None:
    wall, tileset, gate, grid = _wall_case(tmp_path, _nowhere)
    assert gate.cells == 0
    out = tmp_path / "inferred"
    evidence = tf.fill_scan(wall, grid, tf.InpaintFiller(), tileset, out, gate=gate, views=4)
    assert evidence["kind"] == "inferred" and evidence["gaussians"] == 0
    assert evidence["skipped"] == "no cell is still weak after a re-scan"
    assert evidence["gated"]["targets"] == 0 and evidence["rescanGate"]["cells"] == 0
    assert not out.exists()


def test_fill_scan_refuses_another_tileset_s_targets(tmp_path: Path) -> None:
    """Targets computed for another epoch (another window) or for a tileset never measured
    are refused before anything is rendered."""
    wall, tileset, gate, grid = _wall_case(tmp_path, _north)
    other = gf.confidence_grid(2, 40, 40, np.full((3, 3), 60), np.ones((3, 3)))
    elsewhere = RescanGate.from_payload(gf.still_weak_payload(other, [other]))
    with pytest.raises(RescanGateError, match="window"):
        tf.fill_scan(wall, grid, tf.InpaintFiller(), tileset, tmp_path / "x", gate=elsewhere)
    plain = tmp_path / "plain"
    splat_tiles.convert(tmp_path / "wall.ply", plain, 0.0, 0.0, 0.0, opacity_min=0.0)
    with pytest.raises(RescanGateError, match="no confidence grid"):
        tf.fill_scan(
            wall, grid, tf.InpaintFiller(), plain / "tileset.json", tmp_path / "y", gate=gate
        )
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


def test_the_fill_command_needs_the_targets_and_drop_says_it_is_not_gated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, tileset, _, _ = _wall_case(tmp_path, _north)
    with pytest.raises(SystemExit) as exited:
        tf.main(["fill", str(tileset), str(tmp_path / "inferred")])
    assert exited.value.code == 2
    assert "--still-weak" in capsys.readouterr().err
    other = gf.confidence_grid(2, 40, 40, np.full((3, 3), 60), np.ones((3, 3)))
    targets = gf.write_payload(tmp_path / "t.json", gf.still_weak_payload(other, [other]))
    code = tf.main(["fill", str(tileset), str(tmp_path / "inferred"), "--still-weak", str(targets)])
    assert code == 2 and "refused" in capsys.readouterr().err
    assert not (tmp_path / "inferred").exists()
    with pytest.raises(SystemExit):
        tf.main(["drop", "--help"])
    assert "not gated" in capsys.readouterr().out


def test_make_filler_names_the_stand_in_and_imports_the_rest() -> None:
    assert tf.make_filler("telea").name == "opencv-telea"
    assert isinstance(tf.make_filler("teacher_fill:InpaintFiller"), tf.InpaintFiller)
    assert tf.make_filler("teacher_fill:InpaintFiller?radius=9&name=wide").radius == 9
    with pytest.raises(ValueError):
        tf.make_filler("fixer")


def test_refine_sends_the_accepted_views_and_the_scan_near_the_fill() -> None:
    import distill_fill as df

    camera = Camera.look_at([0.0, -5.0, 1.0], [0.0, 0.0, 1.0], width=40, height=30)
    h, w = 30, 40
    mask = np.zeros((h, w), bool)
    mask[10:20, 10:30] = True
    depth = np.full((h, w), 5.0)
    frame = tf.Frame(np.zeros((h, w, 3)), depth, np.ones((h, w)), np.zeros((h, w)), np.ones((h, w)))
    cond = tf.Conditioning(camera, frame, frame, mask, depth, np.zeros((h, w), np.float32))
    rgb = np.full((h, w, 3), 90, np.uint8)
    filled = [tf.Filled(cond, rgb, 40.0, True), tf.Filled(cond, rgb, 10.0, False)]
    lifted, _ = tf.lift(filled)
    far = Splats(
        np.array([[0.0, 0.0, 1.0], [50.0, 0.0, 1.0]]),
        np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        np.full((2, 3), 0.05),
        np.full((2, 3), 0.5),
        np.full(2, 0.9),
    )
    sent: dict = {}

    def runner(request: dict) -> dict:
        sent.update(request)
        return {"inferred": request["init"], "report": {"iterations": request["iterations"]}}

    out, report = tf.refine(lifted, far, filled, 7, runner)
    assert report == {"iterations": 7}
    assert np.allclose(out.positions, lifted.positions, atol=1e-5)
    assert len(df.unpack_scan(sent["measured"])["positions"]) == 1  # the far one cropped
    cameras, images, masks = df.unpack_views(sent["views"])
    assert len(cameras) == 1 and images.shape == (1, h, w, 3) and masks[0].sum() == mask.sum()


def test_link_declares_the_layer_where_the_viewer_looks(tmp_path: Path) -> None:
    measured = tmp_path / "site" / "splat" / "tileset.json"
    inferred = tmp_path / "site" / "inferred" / "tileset.json"
    for path in (measured, inferred):
        path.parent.mkdir(parents=True)
    measured.write_text(json.dumps({"root": {"extras": {"viewCones": {}}}}))
    evidence = {"kind": "inferred", "filler": "x", "views": 2, "gaussians": 9}
    inferred.write_text(json.dumps({"root": {"extras": {"evidence": evidence}}}))
    tf.link_inferred(measured, inferred)
    layers = tf.link_inferred(measured, inferred)  # again: replaced, not doubled
    assert layers == [{"uri": "../inferred/tileset.json", "evidence": evidence}]
    root = json.loads(measured.read_text())["root"]
    assert list(root["extras"]) == ["viewCones", "inferredLayers"]
    inferred.write_text(json.dumps({"root": {"extras": {}}}))
    with pytest.raises(ValueError):
        tf.link_inferred(measured, inferred)
