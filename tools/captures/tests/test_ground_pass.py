"""`ground_pass` on scenes whose ground is known: synthetic slopes with objects, a hidden
patch under a canopy and floaters beneath, and the synthetic yard."""

from __future__ import annotations

import numpy as np
import pytest

import ground_pass as gp
import synthetic_yard


def _terrain(x: np.ndarray, y: np.ndarray, tilt: float) -> np.ndarray:
    return tilt * x + 0.3 * np.exp(-((x - 6.0) ** 2 + (y - 4.0) ** 2) / 4.0)


def _slope_scene(tilt: float = 0.25, seed: int = 3) -> dict[str, np.ndarray]:
    """12 x 10 m of tilted ground with a hill, a 1 m box, a 0.8 m drum, a canopy over a patch
    whose ground nobody saw, and floaters under the ground. Per splat its kind (0 ground,
    1 object, 2 floater, 3 canopy) and its true height above the terrain."""
    rng = np.random.default_rng(seed)
    parts, kinds = [], []

    def add(points: np.ndarray, kind: int) -> None:
        parts.append(points)
        kinds.append(np.full(len(points), kind))

    n = 12 * 10 * 400
    x, y = rng.random(n) * 12, rng.random(n) * 10
    box = (np.abs(x - 3) < 0.5) & (np.abs(y - 3) < 0.5)
    drum = np.hypot(x - 8, y - 7) < 0.4
    hidden = np.hypot(x - 9.5, y - 2.5) < 1.2
    keep = ~(box | drum | hidden)
    x, y = x[keep], y[keep]
    tuft = 0.04 * rng.random(x.size) ** 2
    add(np.stack([x, y, _terrain(x, y, tilt) + tuft + 0.005 * rng.standard_normal(x.size)], 1), 0)
    # The box: five faces (no bottom), 1 m.
    m = 6000
    u, v = rng.random(m) - 0.5, rng.random(m)
    face = rng.integers(0, 5, m)
    bx = np.where(face == 0, -0.5, np.where(face == 1, 0.5, u))
    by = np.where(
        face == 2, -0.5, np.where(face == 3, 0.5, np.where(face < 2, u, rng.random(m) - 0.5))
    )
    bz = np.where(face == 4, 1.0, v)
    base = _terrain(np.full(m, 3.0), np.full(m, 3.0), tilt)
    add(np.stack([3 + bx, 3 + by, base + bz], 1), 1)
    # The drum: a cylinder's side and top, 0.8 m.
    a, h = rng.random(m) * 2 * np.pi, rng.random(m) * 0.8
    top = rng.random(m) < 0.3
    r = np.where(top, 0.4 * np.sqrt(rng.random(m)), 0.4)
    base = _terrain(np.full(m, 8.0), np.full(m, 7.0), tilt)
    add(np.stack([8 + r * np.cos(a), 7 + r * np.sin(a), base + np.where(top, 0.8, h)], 1), 1)
    # The canopy: a slab 3-4 m up over the hidden patch, and its trunk.
    c = 8000
    a, r = rng.random(c) * 2 * np.pi, 1.6 * np.sqrt(rng.random(c))
    cx, cy = 9.5 + r * np.cos(a), 2.5 + r * np.sin(a)
    add(np.stack([cx, cy, _terrain(cx, cy, tilt) + 3 + rng.random(c)], 1), 3)
    t = 1500
    a = rng.random(t) * 2 * np.pi
    tx, ty = 9.5 + 0.12 * np.cos(a), 2.5 + 0.12 * np.sin(a)
    add(np.stack([tx, ty, _terrain(tx, ty, tilt) + 0.3 + 2.7 * rng.random(t)], 1), 1)
    # Floaters 0.3-1 m under the ground.
    f = 400
    fx, fy = 0.5 + rng.random(f) * 11, 0.5 + rng.random(f) * 9
    add(np.stack([fx, fy, _terrain(fx, fy, tilt) - 0.3 - 0.7 * rng.random(f)], 1), 2)
    points = np.concatenate(parts)
    kind = np.concatenate(kinds)
    truth = points[:, 2] - _terrain(points[:, 0], points[:, 1], tilt)
    return {"points": points, "kind": kind, "height": truth}


@pytest.fixture(scope="module")
def slopes() -> tuple[dict[str, np.ndarray], gp.GroundResult]:
    scene = _slope_scene()
    return scene, gp.ground_pass(scene["points"])


def test_terrain_follows_the_ground_and_lies_flat_under_objects(slopes) -> None:
    scene, result = slopes
    p, kind = scene["points"], scene["kind"]
    ground = kind == 0
    error = result.terrain.at(p[ground, :2]) - _terrain(p[ground, 0], p[ground, 1], 0.25)
    assert abs(float(np.median(error))) < 0.015
    assert float(np.percentile(np.abs(error), 95)) < 0.04
    # Under the box and the drum the terrain is interpolated, not raised.
    for cx, cy in ((3.0, 3.0), (8.0, 7.0)):
        xy = np.array([[cx, cy], [cx + 0.2, cy - 0.2]])
        under = result.terrain.at(xy) - _terrain(xy[:, 0], xy[:, 1], 0.25)
        assert np.all(np.abs(under) < 0.06), under


def test_labels(slopes) -> None:
    scene, result = slopes
    kind, height, label = scene["kind"], scene["height"], result.label
    assert (label[kind == 0] == gp.GROUND).mean() > 0.97
    standing = (kind == 1) & (height > 0.15)
    assert (label[standing] == gp.ABOVE).mean() > 0.99
    assert (label[kind == 2] == gp.BELOW).mean() > 0.9
    assert (label[kind == 3] == gp.ABOVE).all()
    assert result.ground.sum() == (label == gp.GROUND).sum()
    assert 0.0 < result.layer_m < 0.15


def test_hidden_ground_is_unknown_not_guessed(slopes) -> None:
    _, result = slopes
    centre = np.array([[9.5, 2.5]])
    assert not result.terrain.seen_at(centre)[0]
    assert result.terrain.seen_at(np.array([[1.0, 8.0]]))[0]
    # A splat in the layer there is unknown: the height under it was interpolated.
    probe = np.concatenate([centre, result.terrain.at(centre)[:, None] + 0.01], 1)
    again = gp.ground_pass(np.concatenate([_slope_scene()["points"], probe]))
    assert again.label[-1] == gp.UNKNOWN


def test_a_steep_slope_stays_ground() -> None:
    rng = np.random.default_rng(5)
    x, y = rng.random(40000) * 10, rng.random(40000) * 10
    z = 0.55 * x + 0.004 * rng.standard_normal(x.size)  # 29 degrees
    result = gp.ground_pass(np.stack([x, y, z], 1))
    assert (result.label == gp.GROUND).mean() > 0.97


def test_transparent_splats_and_blobs_do_not_shape_the_terrain() -> None:
    scene = _slope_scene()
    p = scene["points"]
    n = len(p)
    # A transparent, blobby sheet 0.5 m under everything: it would be the lowest surface.
    sheet = p[scene["kind"] == 0][::10].copy()
    sheet[:, 2] -= 0.5
    points = np.concatenate([p, sheet])
    opacities = np.concatenate([np.full(n, 0.9), np.full(len(sheet), 0.03)])
    scales = np.concatenate([np.full((n, 3), 0.01), np.full((len(sheet), 3), 0.01)])
    result = gp.ground_pass(points, opacities, scales)
    assert (result.label[:n][scene["kind"] == 0] == gp.GROUND).mean() > 0.97
    assert (result.label[n:] == gp.BELOW).all()


def test_deterministic_and_saved(slopes, tmp_path) -> None:
    scene, result = slopes
    again = gp.ground_pass(scene["points"])
    assert np.array_equal(again.hag, result.hag)
    assert np.array_equal(again.label, result.label)
    result.save(tmp_path / "ground.npz")
    loaded = gp.load(tmp_path / "ground.npz")
    assert np.array_equal(loaded.label, result.label)
    assert np.array_equal(loaded.hag, result.hag)
    assert loaded.layer_m == result.layer_m
    assert loaded.stats == result.stats
    xy = scene["points"][:50, :2]
    assert np.allclose(loaded.terrain.at(xy), result.terrain.at(xy))


def test_split_cells_never_mixes_ground_and_the_rest() -> None:
    cell = np.array([0, 0, 0, 1, 1, 2])
    label = np.array([gp.GROUND, gp.ABOVE, gp.BELOW, gp.ABOVE, gp.UNKNOWN, gp.GROUND])
    new, count = gp.split_cells(cell, label)
    assert count == 4
    assert new[0] == new[2] != new[1]
    assert new[3] == new[4] != new[5]
    assert sorted(set(new.tolist())) == list(range(count))


def test_the_yard() -> None:
    data, klass, _, _ = synthetic_yard.generate_yard()
    p = data["position"].astype(np.float64)
    opacity = 1 / (1 + np.exp(-data["opacity_logit"]))
    result = gp.ground_pass(p, opacity, np.exp(data["log_scale"]))
    names = synthetic_yard.CLASSES
    lawn = (klass == names.index("grass/low")) | (klass == names.index("ground"))
    assert (result.label[lawn] == gp.GROUND).mean() > 0.97
    height = p[:, 2] - synthetic_yard.ground_height(p[:, 0], p[:, 1])
    standing = ~lawn & (height > 0.3)
    assert (result.label[standing] == gp.ABOVE).mean() > 0.99
    error = result.terrain.at(p[lawn, :2]) - synthetic_yard.ground_height(p[lawn, 0], p[lawn, 1])
    assert float(np.percentile(np.abs(error), 95)) < 0.05
