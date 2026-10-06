"""The object round of the inferred fill (`object_fill`) on a synthetic scan, on the CPU: two
squashed balls on a ground, their bottoms never captured, filmed from 40 and 60 degrees; the
stand-in generator returns an ellipsoid in its own frame, turned and at unit size."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
from fill_scenes import photos, ring_cameras, splats

import generative_fill as gf
import object_fill as of
import object_models as om
from splat_render import Camera, Splats, load_tileset

BALLS = [((0.0, 0.0), 0.5), ((1.15, 0.7), 0.3)]
SQUASH = 0.6


def _ball(centre: tuple[float, float], radius: float, n: int, seed: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = math.pi * (1 + 5**0.5) * k + seed
    unit = np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
    p = unit * [radius, radius, radius * SQUASH] + [centre[0], centre[1], radius * SQUASH]
    # The bottom was never captured: nothing below a fifth of the height.
    return p[p[:, 2] > 0.2 * radius * SQUASH * 2]


def scene() -> tuple[Splats, np.ndarray]:
    """The ground (id 1, hay) and the balls (ids 2, 3, pumpkins), and each gaussian's id."""
    g = np.linspace(-1.6, 2.2, 70)
    gx, gy = np.meshgrid(g, g)
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    for (cx, cy), r in BALLS:
        ground = ground[np.hypot(ground[:, 0] - cx, ground[:, 1] - cy) > 0.8 * r]
    parts = [splats(ground, [0.75, 0.68, 0.4], size=0.05, seed=1)]
    ids = [np.ones(len(ground), np.int64)]
    for k, ((cx, cy), r) in enumerate(BALLS):
        p = _ball((cx, cy), r, int(2400 * r / 0.5), k)
        shade = 0.6 + 0.4 * (p[:, 2] / (2 * r * SQUASH))
        colour = np.column_stack([0.95 * shade, 0.5 * shade, 0.1 * shade])
        parts.append(splats(p, colour, size=0.025, seed=2 + k))
        ids.append(np.full(len(p), 2 + k, np.int64))
    return Splats.concat(parts), np.concatenate(ids)


def _instances(tiles: Path, original: Splats, ids: np.ndarray) -> dict:
    """An instances.json for the tiles: every leaf's ids by nearest original gaussian."""
    import rebind_instances as ri
    from rig_tiles import tile_positions
    from scipy.spatial import cKDTree
    from synthetic_tree import checksum_positions

    tree = cKDTree(original.positions)
    runs = {}
    for uri in of.leaf_order(tiles / "tileset.json"):
        at = tile_positions(tiles / uri)
        _, near = tree.query(at.astype(np.float64))
        runs[checksum_positions(at)] = ri.encode_runs(ids[near])
    items = [
        {"id": 1, "parent": None, "level": 0, "name": "Hay", "kind": "ground"},
        {"id": 2, "parent": None, "level": 0, "name": "Pumpkin", "concept": "pumpkin"},
        {"id": 3, "parent": None, "level": 0, "name": "Pumpkin", "concept": "pumpkin"},
    ]
    return {"format": "hexapod.instances", "instances": items, "tiles": runs,
            "variant": {"name": "test"}}  # fmt: skip


def _write_colmap(folder: Path, cams: list[Camera], names: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for k, c in enumerate(cams):
            f.write(struct.pack("<IiQQ", k + 1, 1, c.width, c.height))  # PINHOLE
            f.write(struct.pack("<4d", c.focal, c.focal, c.width / 2, c.height / 2))
    with (folder / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for k, (c, name) in enumerate(zip(cams, names, strict=True)):
            q = gf._quaternion(c.rotation)
            t = -c.rotation @ c.centre
            f.write(struct.pack("<IdddddddI", k + 1, *q, *t, k + 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))


def test_registration_recovers_scale_yaw_and_place() -> None:
    rng = np.random.default_rng(0)
    d = rng.standard_normal((3000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    # An egg (not symmetric about z), so the yaw is defined.
    gen = d * [0.5, 0.35, 0.25]
    gen[:, 0] += 0.08 * (d[:, 0] > 0)
    truth = of.Similarity(2.0, of._rz(70.0), np.array([3.0, -1.0, 0.6]))
    measured = truth.apply(gen)
    measured = measured[measured[:, 2] > 0.4]  # the underside unseen
    t, info = of.register(gen, measured)
    assert info["error"] < 0.01
    assert abs(t.scale - 2.0) < 0.05
    assert np.allclose(t.apply(gen[:5]), truth.apply(gen[:5]), atol=0.05)


def test_quaternions_follow_the_rotation() -> None:
    r = of._rz(90.0)
    q = of._matrix_quaternion(r)
    s = Splats(
        np.zeros((1, 3)), np.array([[1.0, 0, 0, 0]]), np.ones((1, 3)), np.ones((1, 3)), np.ones(1)
    )
    moved = of.Similarity(1.0, r, np.zeros(3)).splats(s)
    assert np.allclose(moved.rotations[0], q)
    assert np.allclose(q, [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])


def test_arrays_survive_the_wire() -> None:
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert np.array_equal(om.unpack_array(om.pack_array(a)), a)
    g = om.pack_gaussians(a[:, :3], np.ones((4, 4)), a[:, :3], a[:, :3] / 12, a[:, 0] / 12)
    s = of.splats_from_arrays(g)
    assert len(s) == 4 and np.allclose(s.colours, a / 12, atol=1e-3)


def test_the_pointmap_is_the_scan_in_the_camera_frame() -> None:
    from splat_render import render

    original, _ = scene()
    cam = ring_cameras(1, 40.0, 3.2, target=(0.4, 0.25, 0.25), size=(80, 60), fov=55.0)[0]
    frame = render(original, cam)
    points = of.pointmap(frame.depth, frame.alpha, cam)
    ok = np.isfinite(points[..., 2])
    assert ok.mean() > 0.5
    # Back to the world (OpenCV camera: x and y negated back), every point is near the scan.
    p_cv = points[ok] * [-1, -1, 1]
    world = p_cv @ cam.rotation + cam.centre
    from scipy.spatial import cKDTree

    d, _ = cKDTree(original.positions).query(world)
    assert np.median(d) < 0.05


def test_the_smoke_request_makes_an_object() -> None:
    (result,) = of.StandInGenerator().start([of.smoke_request()])()
    assert result.key == "smoke" and result.splats is not None and len(result.splats) > 100
    assert np.allclose(result.splats.colours[0], [230 / 255, 120 / 255, 30 / 255], atol=0.02)


def test_the_whole_run_completes_the_undersides(tmp_path: Path, monkeypatch) -> None:
    import splat_tiles
    from splat_render import save_ply

    for name, value in (("MASK_WIDTH", 160), ("STRICT_WIDTH", 160), ("GRADE_WIDTH", 200),
                        ("CARVE_WIDTH", 96), ("TILE", (96, 72)), ("SMALL_MIN", 20)):  # fmt: skip
        monkeypatch.setattr(of, name, value)
    original, ids = scene()
    ply = tmp_path / "scan.ply"
    save_ply(ply, original)
    splat_tiles.convert(ply, tmp_path / "tiles", 0.0, 0.0, 0.0, opacity_min=0.0)
    tileset = tmp_path / "tiles" / "tileset.json"
    measured = load_tileset(tileset)
    instances = _instances(tmp_path / "tiles", original, ids)
    (tmp_path / "instances.json").write_text(json.dumps(instances))
    target = (0.4, 0.25, 0.25)
    cams = ring_cameras(14, 40.0, 3.2, target=target, size=(160, 120), fov=55.0)
    cams += ring_cameras(6, 62.0, 3.0, target=target, size=(160, 120), fov=55.0, phase=0.3)
    paths = photos(measured, cams, tmp_path / "frames")
    _write_colmap(tmp_path / "poses" / "sparse" / "0", cams, [p.name for p in paths])
    argv = [
        "run", str(tileset), str(tmp_path / "out"), "--instances", str(tmp_path / "instances.json"),
        "--poses", str(tmp_path / "poses"), "--frames", str(tmp_path / "frames"),
        "--scan", "balls", "--method", "standin", "--renderer", "cpu", "--perceptual", "standin",
        "--quality-width", "96", "--seeds", "1", "--min-frame-psnr", "8",
    ]  # fmt: skip
    assert of.main(argv) == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert [t["instance"] for t in report["targets"]] == [2, 3]
    assert report["instances"]["matched"] == report["instances"]["leaves"]
    for part in report["parts"]["full"]:
        assert part["kept"] > 0, part
        chosen = [c for c in part["candidates"] if c["key"] == part["chosen"]][0]
        assert chosen["registration"]["error"] < 0.03
    assert report["freeSpace"]["violations"] == 0
    layer = load_tileset(tmp_path / "out" / "objects-standin" / "inferred" / "tileset.json")
    assert len(layer) == report["evidence"]["gaussians"] > 0
    # What is kept is the underside: below the lowest measured point of each ball, mostly.
    low = layer.positions[:, 2] < 0.25
    assert low.mean() > 0.6
    for sil in report["silhouette"].values():
        assert sil["meanIoU"] > 0.9 and sil["maxGrowth"] < 0.05
    assert "mean" in report["heldOut"]
    assert (tmp_path / "out" / "renders" / "sheet-objects-standin.png").exists()
