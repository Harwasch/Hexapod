"""`generative_fill`: unknown pixels, paths that start at a real camera, the lift, carving and
the whole loop on a small table nobody saw from above (CPU renderer, Telea stand-in)."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

import generative_fill as gf
import video_fill_models as vfm
from splat_render import Camera, Splats, load_tileset, render

# --- a table on a lawn, filmed from below its top ---------------------------------------------


def _splats(positions: np.ndarray, colour: list[float], size: float = 0.05) -> Splats:
    n = len(positions)
    return Splats(
        np.asarray(positions, np.float64),
        np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        np.full((n, 3), size),
        np.tile(colour, (n, 1)),
        np.full(n, 0.9),
    )


def table_scene() -> tuple[Splats, np.ndarray]:
    """Ground (z = 0, 2.4 m square), a cylinder 0.4 m across and 0.7 m high, and its top.
    Returns the gaussians and which are the top's."""
    g = np.linspace(-1.2, 1.2, 49)
    gx, gy = np.meshgrid(g, g)
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    a = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    z = np.linspace(0.03, 0.67, 14)
    aa, zz = np.meshgrid(a, z)
    side = np.column_stack([0.4 * np.cos(aa.ravel()), 0.4 * np.sin(aa.ravel()), zz.ravel()])
    r = np.linspace(0, 0.38, 8)
    rr, ta = np.meshgrid(r, np.linspace(0, 2 * math.pi, 40, endpoint=False))
    top = np.column_stack(
        [rr.ravel() * np.cos(ta.ravel()), rr.ravel() * np.sin(ta.ravel()), np.full(rr.size, 0.7)]
    )
    parts = [
        _splats(ground, [0.2, 0.5, 0.2]),
        _splats(side, [0.5, 0.3, 0.1]),
        _splats(top, [0.9, 0.8, 0.6]),
    ]
    is_top = np.concatenate([np.zeros(len(ground) + len(side), bool), np.ones(len(top), bool)])
    return Splats.concat(parts), is_top


def ring_views(n: int = 8, height: float = 0.45, radius: float = 2.2) -> list[gf.RealView]:
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        eye = [radius * math.cos(a), radius * math.sin(a), height]
        cam = Camera.look_at(eye, [0.0, 0.0, 0.35], fov_deg=60, width=96, height=64)
        out.append(gf.RealView(f"v{k}", cam))
    return out


ROI = (np.array([-0.45, -0.45, 0.0]), np.array([0.45, 0.45, 0.75]))


def table(views: list[gf.RealView] | None = None) -> gf.Scene:
    splats, _ = table_scene()
    views = ring_views() if views is None else views
    known = gf.seen_directions(splats, views, render, width=96)
    return gf.Scene("table", splats, known, ROI, views=views)


# --- latent masks ----------------------------------------------------------------------------------


def test_latent_known_marks_any_generated_pixel_and_grows() -> None:
    masks = np.zeros((9, 32, 64), bool)
    masks[5, 10, 40] = True  # pixel frame 5 -> latent frame 2 (frames 5..8)
    known = vfm.latent_known(masks, 4, 8, patch=1, grow=0)
    assert known.shape == (3, 4, 8)
    assert not known[2, 1, 5] and known[2].sum() == 31 and known[:2].all()
    grown = vfm.latent_known(masks, 4, 8, patch=1, grow=1)
    assert (~grown[2]).sum() == 9
    tokens = vfm.latent_known(masks, 4, 8, patch=2, grow=0)
    assert (~tokens[2]).sum() == 4  # the 2x2 token holding the cell


def test_latent_frames_need_4k_plus_1() -> None:
    assert [s.start for s in vfm.latent_frames(9)] == [0, 1, 5]
    with pytest.raises(ValueError):
        vfm.latent_frames(8)


def test_clip_round_trip_and_grey() -> None:
    frames = np.random.default_rng(0).integers(0, 255, (5, 4, 6, 3), dtype=np.uint8)
    masks = frames[..., 0] > 128
    back, m = vfm.unpack_clip(vfm.pack_clip(frames, masks))
    assert np.array_equal(back, frames) and np.array_equal(m, masks)
    assert (vfm.greyed(frames, masks)[masks] == 127).all()


# --- real cameras -----------------------------------------------------------------------------------


def _write_colmap(folder: Path, r: np.ndarray, t: np.ndarray) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<IiQQ", 1, 2, 64, 48))  # SIMPLE_RADIAL
        f.write(struct.pack("<4d", 50.0, 32.0, 24.0, 0.0))
    q = gf._quaternion(r)
    with (folder / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<IdddddddI", 1, *q, *t, 1))
        f.write(b"frame0001.jpg\x00")
        f.write(struct.pack("<Q", 0))


def test_colmap_cameras_land_in_the_tiles_frame(tmp_path: Path) -> None:
    look = Camera.look_at([2.0, -1.0, 1.0], [0.0, 0.0, 0.0])
    r, centre = look.rotation, look.centre
    _write_colmap(tmp_path, r, -r @ centre)
    angle = math.radians(30)
    rp = np.array(
        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    )
    placement = {"scale": 2.5, "rotation": rp.tolist(), "translation": [1.0, 2.0, -0.5]}
    (view,) = gf.real_views(tmp_path, placement)
    assert view.name == "frame0001.jpg" and view.intrinsics.model == "SIMPLE_RADIAL"
    point = np.array([[0.3, 0.2, -0.1]])
    raw = Camera(r, centre, 50.0, 64, 48)
    placed = 2.5 * point @ rp.T + np.array([1.0, 2.0, -0.5])
    uv0, _ = raw.project(point)
    uv1, _ = view.camera.project(placed)
    assert np.allclose(uv0, uv1, atol=1e-6)


# --- what is known -------------------------------------------------------------------------------------


def test_the_top_is_unknown_and_the_side_is_known() -> None:
    splats, is_top = table_scene()
    known = gf.seen_directions(splats, ring_views(), render, width=96)
    inner = is_top & (np.linalg.norm(splats.positions[:, :2], axis=1) < 0.3)
    assert (known.counts[inner] < gf.SUPPORT_MIN).mean() > 0.95  # (its rim shows)
    side = np.flatnonzero(~is_top & (splats.positions[:, 2] > 0.2))
    assert (known.counts[side] >= gf.SUPPORT_MIN).mean() > 0.8
    # From where a camera stood, its side is known; from straight above, the top is not.
    w_side = known.weights(splats.positions, np.array([2.2, 0.0, 0.45]))
    facing = side[splats.positions[side, 0] > 0.3]
    assert w_side[facing].mean() > 0.7
    w_top = known.weights(splats.positions, np.array([0.0, 0.0, 3.0]))
    assert w_top[is_top].max() == 0.0


def test_direction_bins_cover_the_sphere() -> None:
    d = np.random.default_rng(1).normal(size=(5000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    bins = gf.direction_bins(d)
    assert bins.min() >= 0 and bins.max() < gf.DIR_BINS**2
    assert np.unique(bins).size == gf.DIR_BINS**2


# --- paths ----------------------------------------------------------------------------------------------


def test_the_path_starts_at_a_real_camera_and_climbs_to_the_unknown_top() -> None:
    scene = table()
    (path,), scores = gf.plan_paths(scene, 1, 9, render, azimuths=4, probe=(96, 54))
    assert path.start is not None
    assert np.allclose(path.centres[0], path.start.camera.centre)
    assert np.allclose(path.rotations[0], path.start.camera.rotation, atol=1e-6)
    assert path.end_direction[2] > 0.5  # it ends looking down on the top
    best = max(scores, key=lambda s: s["unknownPx"])
    assert best["direction"][2] > 0.5
    # The last frame looks at the region's centre.
    last = path.camera(8, 96, 54)
    uv, z = last.project(scene.centre[None])
    assert z[0] > 0 and abs(uv[0, 0] - 48) < 2 and abs(uv[0, 1] - 27) < 2


def test_frame_masks_lift_only_the_region() -> None:
    scene = table()
    cam = Camera.look_at([0.0, -1.6, 1.2], [0.0, 0.0, 0.4], fov_deg=60, width=96, height=64)
    m = gf.frame_masks(scene, cam, render)
    assert m.lift.any() and (m.lift <= m.generate).all()
    assert (gf.ray_box(cam, *scene.roi) | ~m.lift).all()
    assert np.isfinite(m.depth[~m.generate]).mean() > 0.5


# --- depth, lift, seeds, carving -----------------------------------------------------------------------


def test_align_depth_recovers_metric_depth() -> None:
    z = np.linspace(1.0, 4.0, 64)[None, :].repeat(48, axis=0)
    disparity = 2.0 / z + 0.3
    known = np.ones_like(z, bool)
    known[10:20, 10:30] = False
    out, info = gf.align_depth(disparity, np.where(known, z, np.nan), known, ~known)
    assert out is not None and info["residual"] < 1e-6
    assert np.allclose(out[~known], z[~known], rtol=1e-6)


def test_seed_agreement() -> None:
    rgb = np.full((4, 4, 3), 100, np.uint8)
    depth = np.ones((4, 4))
    assert np.allclose(gf.seed_agreement(rgb, depth, rgb, depth), 1.0)
    other = rgb + 80
    assert gf.seed_agreement(rgb, depth, other, depth).max() < 0.5
    assert gf.seed_agreement(rgb, depth, rgb, depth * 1.2).max() < 0.2


def test_carving_deletes_what_a_camera_saw_through() -> None:
    grid = [[2.0, y, z] for y in np.linspace(-1, 1, 21) for z in np.linspace(-1, 1, 21)]
    wall = _splats(np.array(grid), [0.5, 0.5, 0.5], size=0.2)
    cam = Camera.look_at([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], fov_deg=60, width=64, height=64)
    points = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 5.0, 0.0]])
    keep = gf.carve(points, wall, [cam], render, region_size=1.0)
    assert keep.tolist() == [False, True, True]


def test_the_video_classes_are_the_modal_apps() -> None:
    """The class and method names the client calls are the ones infra/modal/fill.py defines."""
    import ast

    source = Path(__file__).resolve().parents[3] / "infra" / "modal" / "fill.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    methods = {
        node.name: {f.name for f in node.body if isinstance(f, ast.FunctionDef)}
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    for key, cls in gf.VIDEO_CLASSES.items():
        assert "fill_clip" in methods[cls], cls
        assert key in vfm.MODELS


# --- the whole loop -------------------------------------------------------------------------------------


class OracleDepth:
    """The test's depth model: the true scene rendered from the frame's camera."""

    name = "oracle"

    def __init__(self, splats: Splats) -> None:
        self.splats = splats

    def disparity(self, frames: np.ndarray, cameras=()) -> np.ndarray:
        out = []
        for cam in cameras:
            d = render(self.splats, cam).depth
            out.append(np.where(np.isfinite(d), 1.0 / np.where(np.isfinite(d), d, 1.0), 0.05))
        return np.stack(out)


def _measured_tileset(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "tileset.json"
    path.write_text(
        json.dumps(
            {
                "asset": {"version": "1.1"},
                "geometricError": 1.0,
                "root": {
                    "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                    "boundingVolume": {"box": [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]},
                    "geometricError": 1.0,
                    "extras": {},
                },
            }
        )
    )
    return path


def test_the_loop_fills_the_top_and_carves_nothing_it_keeps(tmp_path: Path) -> None:
    splats, is_top = table_scene()
    # The scan without its top: what the generator must make.
    scene = table()
    scene.dropped = is_top
    filler = gf.TeleaClipFiller(size=(96, 54))
    options = gf.Options(
        paths=1, seeds=2, rounds=2, round2_paths=1, frames=9, distill=0, azimuths=4, probe=(96, 54)
    )
    report = gf.run_bakeoff(
        scene,
        [filler],
        OracleDepth(splats),
        render,
        _measured_tileset(tmp_path / "scan"),
        tmp_path / "out",
        options,
        log=lambda s: None,
    )
    entry = report["candidates"][filler.name]
    assert entry.get("evidence"), entry
    ev = entry["evidence"]
    assert ev["kind"] == "inferred" and ev["gaussians"] > 0 and ev["seeds"] == 2
    assert ev["rounds"] == 2 and ev["clips"] == 3
    assert 0 < ev["meanConfidence"] <= 1
    layer = load_tileset(tmp_path / "out" / gf.slug(filler.name) / "inferred" / "tileset.json")
    # Every inferred gaussian sits on or under the table's top, near it, not in the air.
    assert np.all(np.abs(layer.positions[:, :2]) <= 0.45 + 0.25 * 0.9 + 0.05)
    assert np.percentile(layer.positions[:, 2], 90) <= 0.78
    # Carving is final: nothing it keeps lies where a real camera saw through.
    present = scene.measured.take(scene.present_index())
    cams = gf.carving_cameras(scene)
    keep = gf.carve(layer.positions, present, cams, render, 2 * scene.radius)
    assert keep.mean() > 0.98  # (the packed tiles round positions)
    renders = tmp_path / "out" / "renders"
    assert (renders / "before-after-table.png").exists()
    assert any(p.name.startswith("clip-table-") for p in renders.iterdir())


def test_held_out_cameras_are_the_highest() -> None:
    views = ring_views(6, 0.45) + ring_views(2, 2.5, 1.0)
    for k, v in enumerate(views):
        v.name = f"c{k}"
    kept, held = gf.split_held_out(views, np.array([0.0, 0.0, 0.35]), 0.25)
    assert {v.name for v in held} == {"c6", "c7"}
    assert len(kept) == 6
