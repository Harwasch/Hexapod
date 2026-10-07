"""Small synthetic scenes for the inferred-fill round-2 tests (CPU renderer).

* `table_scene`: a lawn, a cylinder 0.4 m across and 0.7 m high, and its top, filmed by a
  ring of cameras only a little above the top (the spool: its top seen at grazing angles).
* `ball_scene`: a ball on a lawn filmed from high above (the pumpkin: its lower belt seen at
  grazing angles).

The gaussians are isotropic discs, so their own shortest axis says nothing: the surface
normal comes from their neighbours, as on a real scan seen badly.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from splat_render import Camera, Splats, render


def splats(positions: np.ndarray, colour, size: float = 0.05, seed: int = 0) -> Splats:
    n = len(positions)
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    colours = np.tile(np.asarray(colour, np.float64), (n, 1)) if np.ndim(colour) == 1 else colour
    return Splats(
        np.asarray(positions, np.float64),
        q,  # orientations say nothing (isotropic)
        np.full((n, 3), size),
        np.asarray(colours, np.float64),
        np.full(n, 0.9),
    )


def table_scene() -> tuple[Splats, dict[str, np.ndarray]]:
    """Ground (z = 0, 2.4 m square), a cylinder's side (0.4 m radius, 0.7 m high) and its top
    (with a darker ring on it, so a fill can be scored). Returns the gaussians and boolean
    masks `top`, `side`, `ground`."""
    g = np.linspace(-1.2, 1.2, 41)
    gx, gy = np.meshgrid(g, g)
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    ground = ground[np.linalg.norm(ground[:, :2], axis=1) > 0.42]
    a = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    z = np.linspace(0.03, 0.67, 14)
    aa, zz = np.meshgrid(a, z)
    side = np.column_stack([0.4 * np.cos(aa.ravel()), 0.4 * np.sin(aa.ravel()), zz.ravel()])
    xs = np.linspace(-0.39, 0.39, 17)
    tx, ty = np.meshgrid(xs, xs)
    top = np.column_stack([tx.ravel(), ty.ravel(), np.full(tx.size, 0.7)])
    top = top[np.linalg.norm(top[:, :2], axis=1) <= 0.39]
    r = np.linalg.norm(top[:, :2], axis=1)
    top_colour = np.where((np.abs(r - 0.22) < 0.05)[:, None], [0.4, 0.3, 0.2], [0.9, 0.8, 0.6])
    parts = [
        splats(ground, [0.2, 0.5, 0.2], seed=1),
        splats(side, [0.5, 0.3, 0.1], seed=2),
        splats(top, top_colour, seed=3),
    ]
    n = [len(p) for p in parts]
    masks = {
        "ground": np.r_[np.ones(n[0], bool), np.zeros(n[1] + n[2], bool)],
        "side": np.r_[np.zeros(n[0], bool), np.ones(n[1], bool), np.zeros(n[2], bool)],
        "top": np.r_[np.zeros(n[0] + n[1], bool), np.ones(n[2], bool)],
    }
    return Splats.concat(parts), masks


def flanged_spool_scene() -> tuple[Splats, dict[str, np.ndarray]]:
    """A cable spool on a lawn: a bottom flange (0.6 m radius, its top at z = 0.06), a drum
    (0.3 m radius) and a top flange (0.6 m radius, 0.02 m thick, its top at z = 0.6). As the
    capture from above left it, the drum's top band (z 0.4 to 0.58) and the top flange's
    underside are missing. Masks: `ground`, `drum`, `top`."""
    g = np.linspace(-1.2, 1.2, 49)
    gx, gy = np.meshgrid(g, g)
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    xs = np.linspace(-0.6, 0.6, 25)
    fx, fy = np.meshgrid(xs, xs)
    disc = np.column_stack([fx.ravel(), fy.ravel()])
    disc = disc[np.linalg.norm(disc, axis=1) <= 0.6]
    bottom = np.column_stack([disc, np.full(len(disc), 0.06)])
    top = np.column_stack([disc, np.full(len(disc), 0.6)])
    a = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    rim = np.concatenate(
        [
            np.column_stack([0.6 * np.cos(a), 0.6 * np.sin(a), np.full(a.size, z)])
            for z in (0.58, 0.59)
        ]
    )
    a2 = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    zs = np.arange(0.08, 0.401, 0.02)
    aa, zz = np.meshgrid(a2, zs)
    drum = np.column_stack([0.3 * np.cos(aa.ravel()), 0.3 * np.sin(aa.ravel()), zz.ravel()])
    parts = [
        splats(ground, [0.2, 0.5, 0.2], size=0.04, seed=6),
        splats(bottom, [0.55, 0.45, 0.3], size=0.035, seed=7),
        splats(drum, [0.5, 0.35, 0.2], size=0.02, seed=8),
        splats(np.concatenate([top, rim]), [0.6, 0.5, 0.35], size=0.025, seed=9),
    ]
    n = [len(p) for p in parts]
    ends = np.cumsum([0, *n])
    masks = {}
    for name, k in (("ground", 0), ("drum", 2), ("top", 3)):
        m = np.zeros(ends[-1], bool)
        m[ends[k] : ends[k + 1]] = True
        masks[name] = m
    return Splats.concat(parts), masks


def ring_cameras(
    n: int,
    elevation_deg: float,
    distance: float,
    target=(0.0, 0.0, 0.35),
    size: tuple[int, int] = (96, 64),
    fov: float = 60.0,
    phase: float = 0.0,
) -> list[Camera]:
    out = []
    t = np.asarray(target, np.float64)
    e = math.radians(elevation_deg)
    for k in range(n):
        a = 2 * math.pi * k / n + phase
        eye = t + distance * np.array(
            [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
        )
        out.append(Camera.look_at(eye, t, fov_deg=fov, width=size[0], height=size[1]))
    return out


def spool_cameras() -> list[Camera]:
    """The spool's capture: two rings, the top seen at about 8-20 degrees."""
    return ring_cameras(12, 18.0, 2.4) + ring_cameras(10, 28.0, 2.2, phase=0.3)


def ball_scene() -> tuple[Splats, dict[str, np.ndarray]]:
    """A ball of radius 0.35 resting on a lawn (centre at z = 0.35). Masks: `ball`,
    `lower` (the ball below its equator), `ground`."""
    g = np.linspace(-1.2, 1.2, 41)
    gx, gy = np.meshgrid(g, g)
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    ground = ground[np.linalg.norm(ground[:, :2], axis=1) > 0.2]
    k = np.arange(900) + 0.5
    phi = np.arccos(1 - 2 * k / 900)
    theta = math.pi * (1 + 5**0.5) * k
    unit = np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
    ball = unit * 0.35 + [0.0, 0.0, 0.35]
    ball = ball[ball[:, 2] > 0.04]
    colour = np.where((ball[:, 2] < 0.3)[:, None], [0.8, 0.4, 0.1], [0.9, 0.55, 0.15])
    parts = [splats(ground, [0.6, 0.55, 0.3], seed=4), splats(ball, colour, size=0.04, seed=5)]
    n = [len(p) for p in parts]
    is_ball = np.r_[np.zeros(n[0], bool), np.ones(n[1], bool)]
    lower = np.zeros(sum(n), bool)
    lower[n[0] :] = ball[:, 2] < 0.3
    return Splats.concat(parts), {"ball": is_ball, "lower": lower, "ground": ~is_ball}


def pumpkin_cameras() -> list[Camera]:
    """Filmed from high above: the ball's lower belt is seen only at grazing angles."""
    return ring_cameras(14, 58.0, 2.4) + ring_cameras(8, 72.0, 2.2, phase=0.2)


def photos(scene: Splats, cameras: list[Camera], folder: Path) -> list[Path]:
    """Each camera's view of the true scene, as a PNG (the "real photos")."""
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for k, cam in enumerate(cameras):
        f = render(scene, cam)
        rgb = np.clip(f.rgb, 0, 1)
        path = folder / f"frame_{k:04d}.png"
        Image.fromarray((rgb * 255).round().astype(np.uint8)).save(path)
        out.append(path)
    return out
