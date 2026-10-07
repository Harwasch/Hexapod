"""Free space, the pockets no camera saw, and the shape under an overhang (inferred fill round 2).

**Free space.** A voxel grid over the scan's focus. Each camera's rays run from the camera to
the first measured gaussian they meet (a depth buffer of gaussian fronts, `first_hits`):
- the voxels a ray crossed before it are *free*;
- the few between there and that gaussian are *front* (seen, but not empty);
- the voxels holding a measured gaussian are *surface*.
Nothing else was observed: those voxels are *unseen*.

**Pockets.** Unseen voxels next to a measured surface that a straight line joins to free
space without crossing a surface, so a camera in the open could have looked into them. This
excludes the space under the ground and inside a scanned solid. These are the places a fill
must invent: under the spool's top flange, behind a low wall. For each connected group, the
report gives its size, the share of it under an overhang, and the directions it can be seen
from through free space (low and below views included), so that generative views can be
aimed at it.

**Shape first under an overhang** (`shape_under_overhang`). When a pocket sits between a
plane above (the overhang: a flange, a table top) and a cylinder below (a drum, a leg),
continue both into it: the cylinder up to the plane's underside, and the underside out to the
overhang's rim.
- Samples a camera saw through (free voxels) are carved away.
- Each sample is coloured from the nearest wood a camera did see, at its azimuth on the
  cylinder just below the pocket.
- The cylinder and the underside are both darkened, because they are in shade.
The result is opaque discs in an inferred layer of their own.

**Pass tests** (`overhang_tests`):
- the top face: the 0.9 R disc on the fitted plane, from the 17 headline directions;
- the pocket: the continued band and the underside must be opaque wherever a low view
  (elevation -5 to 15, every azimuth) sees them;
- the open air beside the cylinder must stay as clear as it was.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from splat_render import Camera, Splats

#: Voxels across the grid's longest side.
DIVISIONS = 128
#: Width the cameras' depth buffers are taken at (pixels).
DEPTH_WIDTH = 160
#: A measured gaussian at least this opaque blocks a ray and marks its voxel as surface.
SOLID_OPACITY = 0.1
#: A ray's free run stops this many voxels short of the first gaussian it meets.
FREE_MARGIN = 1.5
#: Unseen space within this many voxels of a measured surface is a pocket candidate.
POCKET_REACH = 4
#: Smaller clusters are not reported.
MIN_POCKET = 20
#: Directions tried around a pocket for the views that can see it.
VIEW_DIRECTIONS = 400

UNSEEN, FREE, SURFACE, FRONT = 0, 1, 2, 3


@dataclass
class Voxels:
    origin: np.ndarray
    size: float
    state: np.ndarray  # uint8 (nx, ny, nz): UNSEEN, FREE or SURFACE

    @property
    def dims(self) -> tuple[int, int, int]:
        return tuple(self.state.shape)  # type: ignore[return-value]

    def cell(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integer voxel coordinates (n, 3) of `points`, and which are inside the grid."""
        ijk = np.floor((np.asarray(points, np.float64) - self.origin) / self.size).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.array(self.dims)), axis=1)
        return ijk, inside

    def at(self, points: np.ndarray) -> np.ndarray:
        """The state at `points` (UNSEEN outside the grid)."""
        ijk, inside = self.cell(points)
        out = np.full(len(ijk), UNSEEN, np.uint8)
        i = ijk[inside]
        out[inside] = self.state[i[:, 0], i[:, 1], i[:, 2]]
        return out

    def centres(self, ijk: np.ndarray) -> np.ndarray:
        return self.origin + (np.asarray(ijk, np.float64) + 0.5) * self.size


def grid(lo: np.ndarray, hi: np.ndarray, divisions: int = DIVISIONS) -> Voxels:
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    size = float((hi - lo).max()) / divisions
    dims = np.maximum(1, np.ceil((hi - lo) / size).astype(int))
    return Voxels(lo, size, np.zeros(tuple(dims), np.uint8))


def first_hits(splats: Splats, camera: Camera, max_radius: int = 4) -> np.ndarray:
    """Per pixel, the camera depth (along its axis) of the nearest front of a measured
    gaussian covering it (its centre less its largest scale), inf where none does. Each
    gaussian stamps a square of twice its largest scale's footprint, at most `max_radius`
    pixels out."""
    w, h = camera.width, camera.height
    depth = np.full(h * w, np.inf)
    p = (splats.positions - camera.centre) @ camera.rotation.T
    z = p[:, 2]
    ok = z > 0.05
    if not ok.any():
        return depth.reshape(h, w)
    p, z = p[ok], z[ok]
    s = splats.scales[ok].max(axis=1)
    u = camera.focal * p[:, 0] / z + w / 2
    v = camera.focal * p[:, 1] / z + h / 2
    front = np.maximum(z - s, 0.05)
    radius = np.clip(np.ceil(2 * s * camera.focal / z - 0.5), 0, max_radius).astype(int)
    on = (u > -max_radius) & (u < w + max_radius) & (v > -max_radius) & (v < h + max_radius)
    for r in range(max_radius + 1):
        k = on & (radius == r)
        if not k.any():
            continue
        uk, vk, fk = np.floor(u[k]).astype(int), np.floor(v[k]).astype(int), front[k]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = uk + dx, vk + dy
                valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                np.minimum.at(depth, y[valid] * w + x[valid], fk[valid])
    return depth.reshape(h, w)


def _ray_box(origin: np.ndarray, dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Per ray (origin + t dirs), the t range inside the box (t0 > t1: none)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        a = (lo - origin) * inv
        b = (hi - origin) * inv
    t0 = np.nanmax(np.minimum(a, b), axis=1)
    t1 = np.nanmin(np.maximum(a, b), axis=1)
    return np.maximum(t0, 0.0), t1


def carve(vox: Voxels, camera: Camera, depth: np.ndarray, margin: float = FREE_MARGIN) -> int:
    """Marks free every voxel a pixel's ray crosses before its first hit (`depth`, camera z)
    less `margin` voxels; returns how many samples it marked."""
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    d_cam = np.stack(
        [(xs + 0.5 - w / 2) / camera.focal, (ys + 0.5 - h / 2) / camera.focal, np.ones((h, w))],
        axis=-1,
    ).reshape(-1, 3)
    dirs = d_cam @ camera.rotation  # world, scaled so that camera z is the parameter
    lo = vox.origin
    hi = vox.origin + np.array(vox.dims) * vox.size
    t0, t1 = _ray_box(camera.centre, dirs, lo, hi)
    # Step about half a voxel along each ray (its length per unit camera z varies).
    length = np.linalg.norm(dirs, axis=1)
    step = 0.5 * vox.size / length
    flat = depth.reshape(-1)
    free_end = np.minimum(t1, flat - margin * vox.size / length)
    # Up to half a voxel past the surface's front: the voxels between where the free run
    # stopped and the surface it met were seen too (observed, not free, never a pocket).
    seen_end = np.minimum(t1, flat + step)
    count = np.where(seen_end > t0, np.ceil((seen_end - t0) / step), 0).astype(np.int64)
    total = int(count.sum())
    if total == 0:
        return 0
    ray = np.repeat(np.arange(len(count)), count)
    first = np.repeat(np.cumsum(count) - count, count)
    k = np.arange(total) - first
    t = t0[ray] + (k + 0.5) * step[ray]
    points = camera.centre + t[:, None] * dirs[ray]
    ijk, inside = vox.cell(points)
    is_free = (t < free_end[ray])[inside]
    ijk = ijk[inside]
    cells = vox.state[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    to_free = is_free & (cells != SURFACE)
    vox.state[ijk[to_free, 0], ijk[to_free, 1], ijk[to_free, 2]] = FREE
    to_front = ~is_free & (cells == UNSEEN)
    vox.state[ijk[to_front, 0], ijk[to_front, 1], ijk[to_front, 2]] = FRONT
    return total


def free_space(
    measured: Splats,
    cameras: list[Camera],
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    divisions: int = DIVISIONS,
    width: int = DEPTH_WIDTH,
    log: Any = None,
) -> Voxels:
    """The grid over [lo, hi]: surface where a measured gaussian at least `SOLID_OPACITY`
    opaque sits, free where a camera's ray crossed before meeting one, unseen elsewhere."""
    from fill_views import scaled

    vox = grid(lo, hi, divisions)
    solid = measured.opacities >= SOLID_OPACITY
    ijk, inside = vox.cell(measured.positions[solid])
    ijk = ijk[inside]
    vox.state[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = SURFACE
    blockers = measured.take(np.flatnonzero(solid))
    for k, cam in enumerate(cameras):
        small = scaled(cam, min(width, cam.width))
        carve(vox, small, first_hits(blockers, small))
        if log and (k + 1) % 50 == 0:
            log(f"free space: {k + 1}/{len(cameras)} cameras")
    return vox


@dataclass
class Pocket:
    cluster: int
    voxels: int
    volume: float
    centre: np.ndarray
    low: np.ndarray
    high: np.ndarray
    under: float  # share of its voxels with a measured surface above (an overhang)
    edge: bool  # touches the grid's side: partly outside what was looked at
    views: list[dict[str, float]] = field(default_factory=list)

    def json(self) -> dict[str, Any]:
        return {
            "cluster": self.cluster,
            "voxels": self.voxels,
            "volumeM3": round(self.volume, 4),
            "centre": np.round(self.centre, 3).tolist(),
            "low": np.round(self.low, 3).tolist(),
            "high": np.round(self.high, 3).tolist(),
            "under": round(self.under, 3),
            "edge": self.edge,
            "views": self.views,
        }


def _sees_free_space(vox: Voxels, cells: np.ndarray, sight_m: float) -> np.ndarray:
    """Per voxel, whether a straight line from it, along one of the 26 grid directions, meets
    space a camera saw (a free voxel) within `sight_m` before any surface: whether a camera in
    the open could have looked into it. Under the ground or inside a scanned solid, none can."""
    from scipy import ndimage

    steps = max(1, round(sight_m / vox.size))
    dims = np.array(vox.dims)
    out = np.zeros(len(cells), bool)
    # Sight is blocked by the surface with its one-voxel gaps closed (a sparse lawn).
    blocked = ndimage.binary_closing(vox.state == SURFACE, structure=np.ones((3, 3, 3)))
    state_of = np.where(blocked, SURFACE, vox.state).astype(np.uint8)
    for d in np.argwhere(np.ones((3, 3, 3), bool)) - 1:
        if not d.any():
            continue
        todo = np.flatnonzero(~out)
        if not todo.size:
            break
        met = np.zeros(todo.size, np.int8)  # 0 nothing yet, 1 free, 2 surface
        for k in range(1, steps + 1):
            live = met == 0
            if not live.any():
                break
            p = cells[todo[live]] + k * d
            inside = np.all((p >= 0) & (p < dims), axis=1)
            state = np.full(len(p), SURFACE, np.uint8)  # leaving the grid: no sight
            q = p[inside]
            state[inside] = state_of[q[:, 0], q[:, 1], q[:, 2]]
            sub = met[live]
            sub[state == FREE] = 1
            sub[state == SURFACE] = 2
            met[live] = sub
        out[todo[met == 1]] = True
    return out


def pockets(
    vox: Voxels,
    *,
    reach: int = POCKET_REACH,
    min_voxels: int = MIN_POCKET,
    overhang_m: float = 0.6,
    sight_m: float = 1.0,
) -> tuple[np.ndarray, list[Pocket]]:
    """The pocket labels (0: none; 1.. by size) and the clusters, largest first: unseen
    voxels within `reach` of a measured surface that a line of sight joins to free space
    (`_sees_free_space`)."""
    from scipy import ndimage

    # A surface's one-voxel gaps (between a sparse lawn's gaussians) are surface too.
    surface = ndimage.binary_closing(vox.state == SURFACE, structure=np.ones((3, 3, 3)))
    unseen = (vox.state == UNSEEN) & ~surface
    near = ndimage.distance_transform_edt(~surface) <= reach
    candidate = unseen & near
    candidate[candidate] = _sees_free_space(vox, np.argwhere(candidate), sight_m)
    groups, n = ndimage.label(candidate, structure=np.ones((3, 3, 3)))
    if n == 0:
        return np.zeros(vox.dims, np.int32), []
    sizes = np.bincount(groups.ravel())[1:]
    # Surface above within `overhang_m`: a cumulative look up each column.
    reach_up = max(1, round(overhang_m / vox.size))
    above = np.zeros(vox.dims, bool)
    for k in range(1, reach_up + 1):
        above[:, :, :-k] |= surface[:, :, k:]
    order = np.argsort(-sizes)
    out_labels = np.zeros(vox.dims, np.int32)
    found: list[Pocket] = []
    for rank, g in enumerate(order, start=1):
        if sizes[g] < min_voxels:
            break
        cells = np.argwhere(groups == g + 1)
        out_labels[cells[:, 0], cells[:, 1], cells[:, 2]] = rank
        pts = vox.centres(cells)
        edge = bool(
            (cells.min(axis=0) == 0).any() or (cells.max(axis=0) == np.array(vox.dims) - 1).any()
        )
        found.append(
            Pocket(
                rank,
                int(sizes[g]),
                float(sizes[g]) * vox.size**3,
                pts.mean(axis=0),
                pts.min(axis=0),
                pts.max(axis=0),
                float(above[cells[:, 0], cells[:, 1], cells[:, 2]].mean()),
                edge,
            )
        )
    return out_labels, found


def fibonacci_directions(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = math.pi * (1 + 5**0.5) * k
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def pocket_views(
    vox: Voxels,
    labels: np.ndarray,
    pocket: Pocket,
    *,
    directions: int = VIEW_DIRECTIONS,
    distance: float | None = None,
    best: int = 3,
) -> list[dict[str, float]]:
    """Directions from which most of `pocket` is seen through space no surface blocks: per
    direction, the share of its voxels (up to 200 of them) whose ray out of the grid meets no
    surface voxel. The `best` well apart (at least 30 degrees), lowest first among equals."""
    cells = np.argwhere(labels == pocket.cluster)
    rng = np.random.default_rng(pocket.cluster)
    if len(cells) > 200:
        cells = cells[rng.choice(len(cells), 200, replace=False)]
    starts = vox.centres(cells)
    dirs = fibonacci_directions(directions)
    span = distance or float(np.array(vox.dims).max() * vox.size)
    steps = np.arange(1.0, span / (0.5 * vox.size)) * 0.5 * vox.size
    seen = np.zeros(len(dirs))
    for j, d in enumerate(dirs):
        pts = starts[:, None, :] + steps[None, :, None] * d
        state = vox.at(pts.reshape(-1, 3)).reshape(len(starts), len(steps))
        seen[j] = float((state != SURFACE).all(axis=1).mean())
    elev = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1, 1)))
    order = np.lexsort((elev, -np.round(seen, 2)))
    chosen: list[int] = []
    for j in order:
        if seen[j] <= 0:
            break
        if all(np.degrees(np.arccos(np.clip(dirs[j] @ dirs[c], -1, 1))) >= 30 for c in chosen):
            chosen.append(int(j))
        if len(chosen) == best:
            break
    return [
        {
            "azimuthDeg": round(float(np.degrees(np.arctan2(dirs[j, 1], dirs[j, 0]))) % 360, 1),
            "elevationDeg": round(float(elev[j]), 1),
            "seenShare": round(float(seen[j]), 3),
        }
        for j in chosen
    ]


# --- shape first under an overhang ------------------------------------------------------------


@dataclass
class Overhang:
    """A plane above (its point, unit normal pointing up, out to `rim` from the axis) and a
    cylinder below it (axis through `axis_point` along the normal, `radius`), with the
    heights along the normal (from the plane) where the measured cylinder stops (`drum_top`)
    and the plane's underside (`underside`)."""

    point: np.ndarray
    normal: np.ndarray
    axis_point: np.ndarray
    radius: float
    rim: float
    underside: float
    drum_top: float
    residual: float
    report: dict[str, Any] = field(default_factory=dict)
    #: The overhang's own centre, in the plane's (u, v) from the cylinder's axis: a table's
    #: top need not sit centred on its leg.
    rim_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def frame(self) -> tuple[np.ndarray, np.ndarray]:
        u = np.cross(self.normal, [1.0, 0.0, 0.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(self.normal, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        return u, np.cross(self.normal, u)

    def local(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`pts` as (height along the normal, distance from the cylinder's axis, distance
        from the overhang's centre), heights from the plane."""
        u, v = self.frame()
        d = np.asarray(pts, np.float64) - self.axis_point
        h = d @ self.normal
        x, y = d @ u, d @ v
        return h, np.hypot(x, y), np.hypot(x - self.rim_offset[0], y - self.rim_offset[1])

    def plane_grid(self, height: float, step: float, inner: float, outer: float) -> np.ndarray:
        """Points at `height` on a grid of `step`: at least `inner` from the cylinder's axis
        and within `outer` of the overhang's centre."""
        u, v = self.frame()
        reach = outer + float(np.linalg.norm(self.rim_offset)) + step
        g = np.arange(-reach, reach + 1e-9, step)
        gx, gy = np.meshgrid(g, g)
        x, y = gx.ravel(), gy.ravel()
        keep = (np.hypot(x, y) >= inner) & (
            np.hypot(x - self.rim_offset[0], y - self.rim_offset[1]) <= outer
        )
        x, y = x[keep], y[keep]
        return self.axis_point + height * self.normal + x[:, None] * u + y[:, None] * v


def _circle(x: np.ndarray, y: np.ndarray, rounds: int = 4) -> tuple[float, float, float, float]:
    """Kasa's circle fit, refitted on the closest 70 % `rounds` times: centre, radius and
    the median residual."""
    cx = cy = r = 0.0
    res = np.zeros(1)
    for _ in range(rounds):
        a = np.column_stack([x, y, np.ones_like(x)])
        sol = np.linalg.lstsq(a, x * x + y * y, rcond=None)[0]
        cx, cy = sol[0] / 2, sol[1] / 2
        r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
        res = np.abs(np.hypot(x - cx, y - cy) - r)
        keep = res <= np.quantile(res, 0.7)
        if keep.sum() < 10:
            break
        x, y = x[keep], y[keep]
    return float(cx), float(cy), r, float(np.median(res))


def _rim(radial: np.ndarray, inner: float, step: float = 0.02) -> float:
    """The overhang's rim: going out from `inner` in rings of `step`, where the gaussians'
    areal density first stays under 30 % of its median inside for three rings (the top's
    frayed edge, or leaves at its height beyond it, do not move it much)."""
    far = float(np.quantile(radial, 0.99)) + step
    edges = np.arange(0.0, far + step, step)
    counts, _ = np.histogram(radial, bins=edges)
    mids = 0.5 * (edges[:-1] + edges[1:])
    density = counts / (2 * math.pi * mids * step)
    inside = (mids > inner) & (mids < 0.8 * float(np.quantile(radial, 0.9)))
    if not inside.any():
        return float(np.quantile(radial, 0.9))
    floor = 0.3 * float(np.median(density[inside]))
    low = density < floor
    for k in np.flatnonzero(mids > inner):
        if k + 3 <= len(low) and low[k : k + 3].all():
            return float(edges[k])
    return float(edges[-1])


def fit_overhang(
    measured: Splats,
    pocket: Pocket,
    top_points: np.ndarray,
    *,
    min_opacity: float = 0.2,
    band: float = 0.1,
) -> Overhang | None:
    """The plane through `top_points` (the overhang's top: a rebuilt surface's discs, or its
    measured gaussians) and the cylinder under it, fitted to the measured gaussians below
    the pocket in bands along the plane's normal. None when no band fits a circle well."""
    if len(top_points) < 50:
        return None
    a = np.column_stack([top_points[:, 0], top_points[:, 1], np.ones(len(top_points))])
    coef = np.linalg.lstsq(a, top_points[:, 2], rcond=None)[0]
    res = np.abs(a @ coef - top_points[:, 2])
    keep = res <= np.quantile(res, 0.9)
    coef = np.linalg.lstsq(a[keep], top_points[keep, 2], rcond=None)[0]
    normal = np.array([-coef[0], -coef[1], 1.0])
    normal /= np.linalg.norm(normal)
    point = top_points[keep].mean(axis=0)
    point[2] = coef[0] * point[0] + coef[1] * point[1] + coef[2]
    probe = Overhang(point, normal, point, 0.0, 0.0, 0.0, 0.0, 0.0)
    u, v = probe.frame()
    m = measured.positions[measured.opacities >= min_opacity]
    d = m - point
    h = d @ normal
    x, y = d @ u, d @ v
    top = np.abs(h) < 0.06
    rim_guess = float(np.quantile(np.hypot(x[top], y[top]), 0.9)) if top.sum() > 20 else np.inf
    depth_low = float((pocket.low - point) @ normal)
    fits = []
    for k in range(math.ceil(max(2.0, -depth_low + 1.0) / band)):
        hi_, lo_ = -0.05 - k * band, -0.05 - (k + 1) * band
        sel = (h <= hi_) & (h > lo_) & (np.hypot(x, y) < 1.2 * rim_guess)
        if sel.sum() < 100:
            continue
        cx, cy, r, e = _circle(x[sel], y[sel])
        if r < 0.8 * rim_guess and e / max(r, 1e-6) < 0.05:
            fits.append((int(sel.sum()), cx, cy, r, e, hi_))
    if not fits:
        return None
    # The cylinder: the largest set of bands that agree on the radius (within 5 %).
    radii = np.array([f[3] for f in fits])
    agree = [[f for f in fits if abs(f[3] - r) < 0.05 * r] for r in radii]
    good = max(agree, key=lambda group: sum(f[0] for f in group))
    cx = float(np.median([f[1] for f in good]))
    cy = float(np.median([f[2] for f in good]))
    radius = float(np.median([f[3] for f in good]))
    axis_point = point + cx * u + cy * v
    radial = np.hypot(x - cx, y - cy)
    on_drum = np.abs(radial - radius) < 0.05 * radius + 0.01
    # Where the measured cylinder stops: going up from its fitted bands in 2 cm steps, the
    # first step with under a fifth of their density (a fuzzy overhang's own gaussians at the
    # cylinder's radius, a few above, do not count as cylinder).
    lowest = min(f[5] for f in good) - band
    edges = np.arange(lowest, -0.02 + 1e-9, 0.02)
    counts, _ = np.histogram(h[on_drum], bins=edges)
    fitted = counts[: max(1, round((max(f[5] for f in good) - lowest) / 0.02))]
    dense = 0.2 * float(np.median(fitted)) if fitted.size else 0.0
    drum_top = float(edges[-1])
    # From the top of the lowest fitted band (the one surely on the cylinder) upwards.
    start = round(band / 0.02)
    for k in range(start, len(counts)):
        if counts[k] < dense:
            drum_top = float(edges[k])
            break
    top = np.abs(h) < 0.06
    rim = _rim(radial[top], radius) if top.any() else radius
    # The overhang's own circle, from its outer ring of gaussians (a table's top may sit off
    # its leg's axis): kept when it fits well, else the rim about the axis.
    offset = np.zeros(2)
    ring = top & (radial > 0.6 * rim) & (radial < 1.1 * rim)
    if ring.sum() >= 200:
        fx, fy, fr, fe = _circle(x[ring] - cx, y[ring] - cy)
        if fe < 0.05 * fr and radius < fr < 1.2 * rim:
            offset, rim = np.array([fx, fy]), fr
    rim_dist = np.hypot(x - cx - offset[0], y - cy - offset[1])
    near_rim = (rim_dist > rim * 0.85) & (rim_dist < rim * 1.05) & (h < 0.05) & (h > drum_top)
    underside = -float(np.quantile(-h[near_rim], 0.9)) if near_rim.sum() > 20 else -0.05
    underside = min(underside, -0.02)
    return Overhang(
        point,
        normal,
        axis_point,
        radius,
        rim,
        underside,
        drum_top,
        float(np.median([f[4] for f in good])),
        {
            "tiltDeg": round(float(np.degrees(np.arccos(np.clip(normal[2], -1, 1)))), 2),
            "radius": round(radius, 4),
            "rim": round(rim, 4),
            "underside": round(underside, 4),
            "drumTop": round(drum_top, 4),
            "bands": len(good),
            "residual": round(float(np.median([f[4] for f in good])), 4),
            "rimOffset": np.round(offset, 4).tolist(),
        },
        offset,
    )


SHADE = {"drum": 0.7, "underside": 0.5}


def shape_under_overhang(
    fit: Overhang,
    measured: Splats,
    vox: Voxels,
    spacing: float,
    *,
    supported: float = 1.5,
    min_opacity: float = 0.2,
) -> tuple[Splats, dict[str, Any]]:
    """Opaque discs continuing the cylinder from where the measured one stops up to the
    plane's underside, and the underside from the cylinder out to the rim; none where a
    measured gaussian already is (within `supported` spacings) or where a camera saw through
    (a free voxel). Coloured from the measured wood at the same azimuth, darkened (`SHADE`)."""
    from scipy.spatial import cKDTree

    from teacher_fill import _disc_rotations

    u, v = fit.frame()
    n = fit.normal
    ring = max(8, round(2 * math.pi * fit.radius / spacing))
    az = np.arange(ring) * 2 * math.pi / ring
    heights = np.arange(fit.drum_top, fit.underside, spacing)
    drum = []
    for hh in heights:
        drum.append(
            fit.axis_point
            + hh * n
            + fit.radius * (np.cos(az)[:, None] * u + np.sin(az)[:, None] * v)
        )
    drum_pts = np.concatenate(drum) if drum else np.zeros((0, 3))
    drum_nrm = drum_pts - fit.axis_point
    drum_nrm -= (drum_nrm @ n)[:, None] * n
    drum_nrm /= np.maximum(np.linalg.norm(drum_nrm, axis=1, keepdims=True), 1e-9)
    under_pts = fit.plane_grid(
        fit.underside, spacing, fit.radius + spacing / 2, fit.rim - spacing / 2
    )
    under_nrm = np.tile(-n, (len(under_pts), 1))
    pts = np.concatenate([drum_pts, under_pts])
    nrm = np.concatenate([drum_nrm, under_nrm])
    part = np.r_[np.zeros(len(drum_pts), int), np.ones(len(under_pts), int)]
    solid = measured.opacities >= min_opacity
    mpos = measured.positions[solid]
    tree = cKDTree(mpos)
    # Already measured: a gaussian within `supported` spacings that is not behind the sample's
    # face (the flange's top is no underside, however thin the flange).
    dist, idx = tree.query(pts, k=8, distance_upper_bound=supported * spacing)
    found = np.isfinite(dist)
    ahead = np.zeros(found.shape, bool)
    rows, cols = np.nonzero(found)
    ahead[rows, cols] = ((mpos[idx[rows, cols]] - pts[rows]) * nrm[rows]).sum(
        axis=1
    ) >= -0.3 * spacing
    new = ~ahead.any(axis=1)
    carved = vox.at(pts) == FREE
    keep = new & ~carved
    # Colour: the measured gaussians at the same azimuth, on the cylinder just below where
    # it stops (for both: the underside is the same wood, in deeper shade).
    d = mpos - fit.axis_point
    h = d @ n
    x, y = d @ u, d @ v
    radial = np.hypot(x, y)
    wood = (
        (np.abs(radial - fit.radius) < 0.08 * fit.radius + 0.02)
        & (h < fit.drum_top)
        & (h > fit.drum_top - 0.3)
    )
    colours = np.tile([0.35, 0.3, 0.25], (len(pts), 1))
    if wood.sum() >= 20:
        wood_az = np.arctan2(y[wood], x[wood])
        wood_col = measured.colours[solid][wood]
        dp = pts - fit.axis_point
        p_az = np.arctan2(dp @ v, dp @ u)
        bins = 72
        idx_w = ((wood_az + math.pi) / (2 * math.pi) * bins).astype(int) % bins
        sums = np.zeros((bins, 3))
        counts = np.zeros(bins)
        np.add.at(sums, idx_w, wood_col)
        np.add.at(counts, idx_w, 1)
        mean_all = wood_col.mean(axis=0)
        per_bin = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], mean_all)
        idx_p = ((p_az + math.pi) / (2 * math.pi) * bins).astype(int) % bins
        colours = per_bin[idx_p]
    shade = np.where(part == 0, SHADE["drum"], SHADE["underside"])
    colours = np.clip(colours * shade[:, None], 0, 1)
    size = 0.75 * spacing
    m = int(keep.sum())
    discs = Splats(
        pts[keep],
        _disc_rotations(nrm[keep]),
        np.column_stack([np.full(m, size), np.full(m, size), np.full(m, 0.1 * size)]),
        colours[keep],
        np.full(m, 0.95),
    )
    report = {
        **fit.report,
        "spacing": round(spacing, 4),
        "drumSamples": len(drum_pts),
        "undersideSamples": len(under_pts),
        "alreadyMeasured": int((~new).sum()),
        "carved": int((new & carved).sum()),
        "discs": m,
        "drumDiscs": int((keep & (part == 0)).sum()),
        "undersideDiscs": int((keep & (part == 1)).sum()),
    }
    return discs, report


def in_free_space(vox: Voxels, splats: Splats) -> int:
    """How many of `splats` sit in voxels a camera saw through (the gate: none)."""
    return int((vox.at(splats.positions) == FREE).sum())


# --- pass tests ---------------------------------------------------------------------------------

#: The 17 headline directions (anchor_fill.HEADLINE_VIEWS): straight above, then 8 azimuths at
#: 15 and at 45 degrees. The low views that look under an overhang: 8 azimuths at -5 to 15.
TOP_VIEWS = [(90.0, 0.0)] + [(e, float(a)) for e in (15.0, 45.0) for a in range(0, 360, 45)]
LOW_VIEWS = [(e, float(a)) for e in (-5.0, 0.0, 5.0, 10.0, 15.0) for a in range(0, 360, 45)]
SEE_ALPHA = 0.5
TOP_FACE_SHARE = 0.9


def _view(fit: Overhang, target: np.ndarray, elev: float, az: float, size=(240, 180)) -> Camera:
    """A camera at `elev`, `az` (degrees, about the plane's normal) looking at `target`."""
    u, v = fit.frame()
    e, a = math.radians(elev), math.radians(az)
    d = math.cos(e) * (math.cos(a) * u + math.sin(a) * v) + math.sin(e) * fit.normal
    up = v if elev >= 89.0 else fit.normal
    return Camera.look_at(
        target + 3.2 * fit.rim * d, target, fov_deg=40, width=size[0], height=size[1], up=tuple(up)
    )


def _inside_solid(fit: Overhang, pts: np.ndarray) -> np.ndarray:
    """Whether `pts` are inside the fitted overhang (between its underside and top, out to its
    rim) or inside the cylinder under it."""
    h, r, r_rim = fit.local(pts)
    slab = (h <= 0.0) & (h >= fit.underside) & (r_rim <= fit.rim)
    drum = (h < fit.underside) & (h > fit.drum_top - 1.0) & (r < fit.radius)
    return slab | drum


def _visible(fit: Overhang, cam: Camera, pts: np.ndarray, step: float = 0.01) -> np.ndarray:
    """Whether the segment from each point towards the camera leaves the fitted solid's
    neighbourhood without entering it (starting a little off the point's own surface)."""
    if not len(pts):
        return np.zeros(0, bool)
    to_cam = cam.centre - pts
    dirs = to_cam / np.linalg.norm(to_cam, axis=1, keepdims=True)
    ok = np.ones(len(pts), bool)
    for t in np.arange(2 * step, 3.0 * fit.rim, step):
        live = np.flatnonzero(ok)
        if not live.size:
            break
        ok[live[_inside_solid(fit, pts[live] + t * dirs[live])]] = False
    return ok


def _pixels(cam: Camera, pts: np.ndarray) -> np.ndarray:
    if not len(pts):
        return np.zeros((0, 2), int)
    uv, z = cam.project(pts)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width)
    ok &= (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    return np.unique(uv[ok].astype(int), axis=0)


def _slab(fit: Overhang, splats: Splats) -> Splats:
    h, _, r_rim = fit.local(splats.positions)
    keep = (h < 0.3) & (h > fit.drum_top - 0.4) & (r_rim < 1.4 * fit.rim)
    return splats.take(np.flatnonzero(keep))


def overhang_tests(
    fit: Overhang,
    states: dict[str, Splats],
    vox: Voxels,
    spacing: float,
    *,
    size: tuple[int, int] = (240, 180),
    low_states: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The split pass test, per state (each a whole scene, as it would be drawn; the first is
    the reference for the air):
    - `top`: per headline direction, the share of the top face (the 0.9 rim disc on the
      fitted plane) below alpha 0.5, and its mean alpha. Pass: at most 5 % from every
      direction, mean at least 0.95.
    - `low`: per low view, the share of the continued band and underside that the view sees
      (front-facing, not hidden by the solid) below alpha 0.5 (pass: at most 5 %); and the
      share of the pixels looking through the carved open air beside the cylinder below 0.5,
      which should not drop."""
    from fill_surface import ewa_alpha

    u, v = fit.frame()
    n = fit.normal
    ring = max(16, round(2 * math.pi * fit.radius / spacing))
    az = np.arange(ring) * 2 * math.pi / ring
    radial = np.cos(az)[:, None] * u + np.sin(az)[:, None] * v
    heights = np.arange(fit.drum_top, fit.underside, spacing)
    band = np.concatenate([fit.axis_point + h * n + fit.radius * radial for h in heights])
    band_n = np.tile(radial, (len(heights), 1))
    under = fit.plane_grid(fit.underside, spacing, fit.radius + spacing, fit.rim - spacing)
    pocket_pts = np.concatenate([band, under])
    pocket_n = np.concatenate([band_n, np.tile(-n, (len(under), 1))])
    air = np.concatenate(
        [
            fit.plane_grid(h, 2 * spacing, fit.radius + 3 * spacing, fit.rim - 2 * spacing)
            for h in np.arange(fit.drum_top - 0.3, fit.underside - 2 * spacing, 2 * spacing)
        ]
    )
    air = air[vox.at(air) == FREE]
    slabs = {name: _slab(fit, s) for name, s in states.items()}
    target_low = fit.axis_point + 0.5 * (fit.drum_top + fit.underside) * n
    out: dict[str, Any] = {"states": list(states), "top": [], "low": []}
    for elev, azim in TOP_VIEWS:
        cam = _view(fit, fit.axis_point, elev, azim, size)
        ys, xs = np.mgrid[0 : cam.height, 0 : cam.width]
        dc = np.stack(
            [
                (xs + 0.5 - cam.width / 2) / cam.focal,
                (ys + 0.5 - cam.height / 2) / cam.focal,
                np.ones(xs.shape),
            ],
            -1,
        )
        dw = dc @ cam.rotation
        denom = dw @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((fit.axis_point - cam.centre) @ n) / denom
        hits = (cam.centre + t[..., None] * dw).reshape(-1, 3)
        rr = fit.local(hits)[2].reshape(t.shape)
        face = (t > 0) & (rr <= TOP_FACE_SHARE * fit.rim) & (denom < 0)
        row: dict[str, Any] = {"el": elev, "az": azim, "pixels": int(face.sum())}
        for name, s in slabs.items():
            alpha = ewa_alpha(s, cam)
            row[name] = {
                "seeThrough": round(float((alpha[face] < SEE_ALPHA).mean()), 4)
                if face.any()
                else 0.0,
                "meanAlpha": round(float(alpha[face].mean()), 4) if face.any() else 1.0,
            }
        out["top"].append(row)
    low_slabs = {k: v for k, v in slabs.items() if low_states is None or k in low_states}
    for elev, azim in LOW_VIEWS:
        cam = _view(fit, target_low, elev, azim, size)
        facing = ((cam.centre - pocket_pts) * pocket_n).sum(axis=1) > 0
        seen = np.zeros(len(pocket_pts), bool)
        seen[facing] = _visible(fit, cam, pocket_pts[facing])
        px = _pixels(cam, pocket_pts[seen])
        apx = _pixels(cam, air[_visible(fit, cam, air)])
        if len(px) and len(apx):
            # A pixel the pocket's surface covers is the surface's, not the air's.
            taken = {tuple(p) for p in px.tolist()}
            apx = np.array([p for p in apx.tolist() if tuple(p) not in taken], int).reshape(-1, 2)
        row = {"el": elev, "az": azim, "pocketPixels": len(px), "airPixels": len(apx)}
        for name, s in low_slabs.items():
            alpha = ewa_alpha(s, cam)
            row[name] = {
                "pocketSeeThrough": round(float((alpha[px[:, 1], px[:, 0]] < SEE_ALPHA).mean()), 4)
                if len(px)
                else 0.0,
                "airSeeThrough": round(float((alpha[apx[:, 1], apx[:, 0]] < SEE_ALPHA).mean()), 4)
                if len(apx)
                else 1.0,
            }
        out["low"].append(row)
    summary: dict[str, Any] = {}
    for name in states:
        top = [r[name] for r in out["top"] if r["pixels"]]
        low = [r[name] for r in out["low"] if r["pocketPixels"] and name in r]
        airs = [r[name]["airSeeThrough"] for r in out["low"] if r["airPixels"] and name in r]
        s = {
            "topWorst": max((t["seeThrough"] for t in top), default=0.0),
            "topMinMeanAlpha": min((t["meanAlpha"] for t in top), default=1.0),
            "pocketWorst": max((t["pocketSeeThrough"] for t in low), default=0.0),
            "pocketMean": round(float(np.mean([t["pocketSeeThrough"] for t in low])), 4)
            if low
            else 0.0,
            "airMean": round(float(np.mean(airs)), 4) if airs else 1.0,
        }
        s["topPass"] = s["topWorst"] <= 0.05 and s["topMinMeanAlpha"] >= 0.95
        s["pocketPass"] = s["pocketWorst"] <= 0.05
        summary[name] = s
    first = next(iter(low_slabs))
    for name in low_slabs:
        summary[name]["airKept"] = summary[name]["airMean"] >= summary[first]["airMean"] - 0.02
    out["summary"] = summary
    return out
