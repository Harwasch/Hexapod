"""How well each gaussian of a scan was seen: the inferred-fill bake-off's "where to fill".

Round 1 (`generative_fill.SeenDirections`) counted cameras: a gaussian three cameras saw was
known. The spool's top was seen by plenty of cameras, but only at grazing angles from below
about 23 degrees, so it counted as known, was never filled, and looks smeared from above.
Here each gaussian gets the **best supervision quality** any real camera gave it:

    q = cos(angle between the view ray and the surface normal)
        x footprint (the camera's pixels per unit length there, against the capture's own)
        x sharpness (the photo's detail against the other photos')

and is classed by it (`classify`):

* **known** -- some camera saw it squarely and close (`Q_KNOWN`), and at least `MIN_GOOD`
  cameras saw it at all well: frozen, never repainted;
* **weak** -- seen, but only badly (`Q_WEAK` <= q < `Q_KNOWN`): repainted at partial
  denoising strength, so the generator keeps what was seen and sharpens it;
* **unknown** -- nobody saw it usefully: generated.

The surface normal is the local plane of the gaussians around it (`surface_normals`: the
wider neighbourhood's where that is flat, else the nearer one's), else the gaussian's own
shortest axis. A splat optimised only from grazing views is often not flat along its
surface, so its own axis can point anywhere; its neighbours still lie in the surface.

What to fill is then grouped without picking regions by hand (`hole_clusters`): the weak
and unknown gaussians near what the cameras were filming (`Focus`), joined where they touch
(voxel connectivity). Nothing here knows what the object is.

Numpy and scipy; renders through any `splat_render`-style renderer (the CPU one in tests,
gsplat on a GPU).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from splat_render import Camera, Frame, Splats

KNOWN, WEAK, UNKNOWN = 0, 1, 2
CLASS_NAMES = ("known", "weak", "unknown")
#: Best quality at or above this is known (cos 0.5 is 60 degrees from the normal, at the
#: capture's usual resolution and sharpness).
Q_KNOWN = 0.5
#: Below this a gaussian was not usefully seen at all: unknown.
Q_WEAK = 0.12
#: Known also needs at least this many cameras at `Q_WEAK` or better.
MIN_GOOD = 2
#: A camera sees a gaussian when its centre is no farther than the rendered surface at its
#: pixel by this share of the depth (and about its own size): `generative_fill`'s test.
DEPTH_TOLERANCE = 0.03
#: Neighbours for the local surface normal, and the flatness (smallest over middle
#: eigenvalue) below which the neighbourhood counts as a surface.
NORMAL_NEIGHBOURS = 16
PLANAR = 0.35
#: Which side of a surface a camera is on is judged on the plane of this many neighbours
#: (smoother than the normal used for the viewing angle: a lawn or straw is rough), and a
#: camera within this cosine past edge-on still counts as on the seen side.
FACING_NEIGHBOURS = 64
FACING_MARGIN = 0.1
#: The fill works within this share of the median camera distance of what the cameras
#: converge on (the object the capture is of, and its surroundings).
FOCUS_SHARE = 0.55
#: A leave-out withholds a gaussian a held-out camera saw this much better (factor and
#: margin over the kept cameras' best quality).
HELD_BETTER = 1.2
HELD_MARGIN = 0.02
#: ... decided by neighbourhood: withheld where this share of its nearest neighbours is.
WITHHELD_NEIGHBOURS = 32
WITHHELD_SHARE = 0.35
#: Hole clusters: voxels this many median spacings across join; smaller clusters are noise.
CLUSTER_SPACING = 2.5
MIN_CLUSTER = 30

Renderer = Callable[..., Frame]


@dataclass(frozen=True)
class Focus:
    """What the real cameras were filming: the point nearest all their optical axes, their
    median distance from it, their median horizontal field of view, and the capture's
    reference resolution (median focal over distance: pixels per unit length there)."""

    centre: np.ndarray
    distance: float
    hfov_deg: float
    density: float
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    @property
    def radius(self) -> float:
        return FOCUS_SHARE * self.distance

    def inside(self, positions: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.asarray(positions) - self.centre, axis=1) <= self.radius

    def elevation(self, point: np.ndarray) -> float:
        d = np.asarray(point, np.float64) - self.centre
        return math.degrees(math.asin(np.clip(d @ self.up / max(np.linalg.norm(d), 1e-12), -1, 1)))

    def to_json(self) -> dict[str, Any]:
        return {
            "centre": np.round(self.centre, 4).tolist(),
            "distance": round(self.distance, 4),
            "hfovDeg": round(self.hfov_deg, 2),
            "density": round(self.density, 3),
            "radius": round(self.radius, 4),
        }


def capture_focus(cameras: Sequence[Camera], up: np.ndarray | None = None) -> Focus:
    """The least-squares point nearest every camera's optical axis (an orbit's centre),
    and the cameras' median distance from it, field of view and resolution there."""
    if not cameras:
        raise ValueError("no cameras to find the focus of")
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for cam in cameras:
        d = cam.rotation[2] / np.linalg.norm(cam.rotation[2])
        p = np.eye(3) - np.outer(d, d)
        a += p
        b += p @ cam.centre
    try:
        centre = np.linalg.solve(a + 1e-9 * np.eye(3), b)
    except np.linalg.LinAlgError:
        centre = np.mean([c.centre for c in cameras], axis=0)
    dist = np.array([np.linalg.norm(c.centre - centre) for c in cameras])
    hfov = np.array([math.degrees(2 * math.atan(c.width / (2 * c.focal))) for c in cameras])
    density = np.array([c.focal for c in cameras]) / np.maximum(dist, 1e-9)
    return Focus(
        centre,
        float(np.median(dist)),
        float(np.median(hfov)),
        float(np.median(density)),
        np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, np.float64),
    )


# --- normals ------------------------------------------------------------------------------------


def _rotation_matrices(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )


def shortest_axes(splats: Splats) -> np.ndarray:
    """Each gaussian's shortest axis (unit, world frame): a flat splat's normal."""
    rot = _rotation_matrices(np.asarray(splats.rotations, np.float64))
    k = np.argmin(splats.scales, axis=1)
    return rot[np.arange(len(splats)), :, k]


def local_normals(
    positions: np.ndarray, k: int = NORMAL_NEIGHBOURS
) -> tuple[np.ndarray, np.ndarray]:
    """The normal of the plane through each point's `k` nearest neighbours and how flat they
    are (smallest over middle eigenvalue of their covariance: 0 is a perfect plane)."""
    from scipy.spatial import cKDTree

    p = np.asarray(positions, np.float64)
    n = len(p)
    if n < 4:
        return np.tile([0.0, 0.0, 1.0], (n, 1)), np.ones(n)
    k = min(k, n)
    _, idx = cKDTree(p).query(p, k=k)
    normals = np.empty((n, 3))
    flat = np.empty(n)
    for start in range(0, n, 65536):
        stop = min(start + 65536, n)
        nb = p[idx[start:stop]]
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb) / k
        values, vectors = np.linalg.eigh(cov)
        normals[start:stop] = vectors[:, :, 0]
        flat[start:stop] = values[:, 0] / np.maximum(values[:, 1], 1e-18)
    return normals, flat


def surface_normals(
    splats: Splats,
    k: int = NORMAL_NEIGHBOURS,
    coarse: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Per gaussian the surface normal: the flatter of two planes, through its `k` nearest
    neighbours or its wider neighbourhood (`FACING_NEIGHBOURS`, or `coarse` = (normals,
    flatness) already computed), where that is flat (`PLANAR`), else its own shortest axis.
    Unit, unsigned (only |cos| is used). On a large flat surface the 16-neighbour planes of
    a splat's layered gaussians tilt every way (a quarter of the spool top's came out more
    than 60 degrees off vertical) and a tilted normal makes a camera at a grazing angle look
    head-on; the wider plane averages the layer out. Where the wider one reaches round an
    edge the nearer one is flatter and wins."""
    plane, flat = local_normals(splats.positions, k)
    wide, wide_flat = (
        coarse if coarse is not None else local_normals(splats.positions, FACING_NEIGHBOURS)
    )
    own = shortest_axes(splats)
    out = np.where((flat <= PLANAR)[:, None], plane, own)
    use_wide = (wide_flat <= PLANAR) & (wide_flat < flat)
    out = np.where(use_wide[:, None], wide, out)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


# --- sharpness of the photos ---------------------------------------------------------------------


def photo_sharpness(photos: Sequence[np.ndarray | None]) -> np.ndarray:
    """Per photo a sharpness factor in (0, 1]: the variance of its Laplacian (grey, a fixed
    small size) against the median photo's, square-rooted and capped at 1. A motion-blurred
    frame scores low; a missing photo scores 1 (nothing to say against it)."""
    import cv2

    raw = []
    for photo in photos:
        if photo is None:
            raw.append(np.nan)
            continue
        grey = cv2.cvtColor(np.asarray(photo, np.uint8), cv2.COLOR_RGB2GRAY)
        h, w = grey.shape
        scale = 256.0 / max(w, 1)
        grey = cv2.resize(grey, (256, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        raw.append(float(cv2.Laplacian(grey.astype(np.float32), cv2.CV_32F).var()))
    values = np.asarray(raw, np.float64)
    ok = np.isfinite(values) & (values > 0)
    out = np.ones(len(values))
    if ok.any():
        med = float(np.median(values[ok]))
        out[ok] = np.clip(np.sqrt(values[ok] / max(med, 1e-12)), 0.05, 1.0)
    return out


# --- quality --------------------------------------------------------------------------------------


@dataclass
class Quality:
    """Per gaussian: the best quality over the cameras, how many cameras saw it well
    (`Q_WEAK` or better), how many saw it at all, and the class. `per_camera` (cameras x
    gaussians, sparse) keeps every camera's quality of every gaussian it saw (plus a tiny
    epsilon, so a seen gaussian is stored), for subsets and context retrieval. `facing`:
    the normal turned towards the side the cameras saw it from (`observed`: by any)."""

    best: np.ndarray
    good: np.ndarray
    seen: np.ndarray
    classes: np.ndarray
    per_camera: Any
    normals: np.ndarray
    sharpness: np.ndarray
    centres: np.ndarray
    facing: np.ndarray
    observed: np.ndarray
    positions: np.ndarray
    side_normals: np.ndarray

    def subset(self, cameras: Sequence[int]) -> Quality:
        """The same measured with only these cameras (rows of `per_camera`)."""
        idx = np.asarray(list(cameras), np.int64)
        return _summarise(
            self.per_camera[idx],
            self.normals,
            self.sharpness[idx],
            self.centres[idx],
            self.positions,
            self.side_normals,
        )

    def summary(self, mask: np.ndarray | None = None) -> dict[str, Any]:
        c = self.classes if mask is None else self.classes[mask]
        return {
            "gaussians": int(c.size),
            **{name: int((c == k).sum()) for k, name in enumerate(CLASS_NAMES)},
            "medianBest": round(
                float(np.median(self.best if mask is None else self.best[mask])) if c.size else 0.0,
                4,
            ),
        }


def classify(best: np.ndarray, good: np.ndarray) -> np.ndarray:
    out = np.full(best.shape, UNKNOWN, np.int8)
    out[best >= Q_WEAK] = WEAK
    out[(best >= Q_KNOWN) & (good >= MIN_GOOD)] = KNOWN
    return out


def _summarise(
    per_camera: Any,
    normals: np.ndarray,
    sharpness: np.ndarray,
    centres: np.ndarray,
    positions: np.ndarray,
    side_normals: np.ndarray,
) -> Quality:
    best = np.asarray(per_camera.max(axis=0).toarray()).ravel().astype(np.float32)
    good = np.asarray((per_camera >= Q_WEAK).sum(axis=0)).ravel().astype(np.int32)
    seen = np.asarray((per_camera > 0).sum(axis=0)).ravel().astype(np.int32)
    # The side each gaussian was seen from: the quality-weighted pull towards the cameras
    # that saw it (unnormalised: only its sign against the normal is used).
    weight = np.asarray(per_camera.sum(axis=0)).ravel()
    toward = (
        np.asarray(per_camera.T @ np.asarray(centres, np.float64)) - weight[:, None] * positions
    )
    sign = np.where(np.einsum("ij,ij->i", side_normals, toward) < 0, -1.0, 1.0)
    return Quality(
        best,
        good,
        seen,
        classify(best, good),
        per_camera,
        normals,
        sharpness,
        np.asarray(centres, np.float64),
        side_normals * sign[:, None],
        seen > 0,
        positions,
        side_normals,
    )


def facing(quality: Quality, eye: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Per gaussian, whether `eye` is on the side the real cameras saw it from (a gaussian
    nobody saw counts as facing: there is no side to be wrong about)."""
    toward = np.asarray(eye, np.float64) - positions
    toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-12)
    return ~quality.observed | (np.einsum("ij,ij->i", quality.facing, toward) >= -FACING_MARGIN)


def depth_spread(depth: np.ndarray, cap: float = 0.1) -> np.ndarray:
    """Per pixel the depth range over its 3x3 neighbourhood where all of it is a surface
    (0 at a silhouette), at most `cap` of the depth: how uncertain a pixel's depth is on a
    surface seen at a grazing angle, where it ramps steeply across pixels."""
    from scipy.ndimage import maximum_filter, minimum_filter

    finite = np.isfinite(depth)
    d = np.where(finite, depth, 0.0)
    whole = minimum_filter(finite.astype(np.uint8), size=3, mode="nearest") > 0
    spread = maximum_filter(d, size=3, mode="nearest") - minimum_filter(d, size=3, mode="nearest")
    return np.where(whole, np.minimum(spread, cap * d), 0.0)


def nearest_surface(depth: np.ndarray) -> np.ndarray:
    """Per pixel the rendered surface, or the median of its 3x3 neighbourhood where that is
    nearer (inf where none): a renderer's sampling gap in a front surface (one pixel showing
    what is behind it) does not let what is behind count as seen, while a surface seen at a
    grazing angle (depth ramping steeply across pixels) still is."""
    from scipy.ndimage import median_filter

    d = np.where(np.isfinite(depth), depth, np.inf)
    big = np.where(np.isfinite(d), d, 1e30)
    med = median_filter(big, size=3, mode="nearest")
    out = np.minimum(big, med)
    return np.where(out >= 1e29, np.inf, out)


def measure_quality(
    splats: Splats,
    cameras: Sequence[Camera],
    renderer: Renderer,
    focus: Focus,
    *,
    width: int = 320,
    sharpness: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> Quality:
    """Every camera's quality of every gaussian it sees (depth-tested at `width` px), and the
    per-gaussian summary. `cameras` are the real ones at their photo size (their focal sets
    the footprint); `sharpness` per camera (`photo_sharpness`, default 1)."""
    from scipy.sparse import csr_matrix

    n = len(splats)
    side, side_flat = local_normals(splats.positions, FACING_NEIGHBOURS)
    if normals is None:
        normals = surface_normals(splats, coarse=(side, side_flat))
    sharp = np.ones(len(cameras)) if sharpness is None else np.asarray(sharpness, np.float64)
    reach = 2.0 * splats.scales.max(axis=1)
    rows_all, cols_all, vals_all = [], [], []
    for c, full in enumerate(cameras):
        small = Camera(
            full.rotation,
            full.centre,
            full.focal * width / full.width,
            width,
            max(1, round(full.height * width / full.width)),
            full.far,
        )
        depth = renderer(splats, small).depth
        surface_map = nearest_surface(depth)
        spread_map = depth_spread(depth)
        uv, z = small.project(splats.positions)
        u = np.floor(uv[:, 0]).astype(np.int64)
        v = np.floor(uv[:, 1]).astype(np.int64)
        inside = (z > 1e-3) & (u >= 0) & (u < small.width) & (v >= 0) & (v < small.height)
        rows = np.flatnonzero(inside)
        surface = surface_map[v[rows], u[rows]]
        limit = (
            surface * (1 + DEPTH_TOLERANCE)
            + np.minimum(reach[rows], 0.05 * surface)
            + spread_map[v[rows], u[rows]]
        )
        rows = rows[np.isfinite(surface) & (z[rows] <= limit)]
        if rows.size == 0:
            continue
        d = splats.positions[rows] - full.centre
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        cos = np.abs(np.einsum("ij,ij->i", d, normals[rows]))
        footprint = np.clip((full.focal / np.maximum(z[rows], 1e-9)) / focus.density, 0.0, 1.0)
        q = cos * footprint * sharp[c] + 1e-6
        rows_all.append(np.full(rows.size, c, np.int32))
        cols_all.append(rows.astype(np.int32))
        vals_all.append(q.astype(np.float32))
    if rows_all:
        matrix = csr_matrix(
            (np.concatenate(vals_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
            shape=(len(cameras), n),
        )
    else:
        matrix = csr_matrix((len(cameras), n), dtype=np.float32)
    centres = np.array([c.centre for c in cameras], np.float64).reshape(-1, 3)
    return _summarise(
        matrix, normals, sharp, centres, np.asarray(splats.positions, np.float64), side
    )


def withheld_by_holdout(
    kept: Quality, everyone: Quality, held: Quality | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The leave-out check: (withheld, dropped). Withheld: not known with the kept cameras,
    and either known only with the held-out ones added or seen clearly better by a held-out
    camera (`HELD_BETTER`) -- the scan learned their look mostly from the held-out photos,
    so it must not be shown (their look is withheld and they count as unknown). A surface
    the kept cameras saw only at a grazing angle and the held-out ones from above (the
    spool's top) is weak both ways, and only the second test catches it. Decided over each
    gaussian's neighbourhood (`WITHHELD_SHARE` of its `WITHHELD_NEIGHBOURS`). Dropped: seen
    only by held-out cameras -- their very shape came from them, so they leave the scene."""
    dropped = (kept.seen == 0) & (everyone.seen > 0)
    learned = everyone.classes == KNOWN
    if held is not None:
        learned |= (held.best >= Q_WEAK) & (held.best > HELD_BETTER * kept.best + HELD_MARGIN)
    learned &= kept.classes != KNOWN
    # Per gaussian the call is noisy (a few of a surface's gaussians come out known with
    # the kept cameras, and their learned look would leak into the scored region), so a
    # surface is withheld where enough of its neighbourhood is.
    n = len(learned)
    if n > 1 and learned.any():
        from scipy.spatial import cKDTree

        k = min(WITHHELD_NEIGHBOURS, n)
        _, idx = cKDTree(kept.positions).query(kept.positions, k=k)
        learned = learned[idx].mean(axis=1) >= WITHHELD_SHARE
    return learned & ~dropped, dropped


# --- hole clusters ----------------------------------------------------------------------------------


@dataclass
class Clusters:
    """Connected groups of the gaussians to fill: `labels` per gaussian (-1: not in one),
    and per cluster its size, centre, bounds and fill weight (area x deficit)."""

    labels: np.ndarray
    info: list[dict[str, Any]]

    def members(self, k: int) -> np.ndarray:
        return np.flatnonzero(self.labels == k)


def median_spacing(positions: np.ndarray, sample: int = 20000) -> float:
    """The median distance to the nearest other point. On a subsample, scaled back as for
    points on a surface (spacing goes with the square root of the density)."""
    from scipy.spatial import cKDTree

    p = np.asarray(positions, np.float64)
    if len(p) < 2:
        return 1.0
    pick = p
    if len(p) > sample:
        pick = p[np.random.default_rng(0).choice(len(p), sample, replace=False)]
    d, _ = cKDTree(pick).query(pick, k=2)
    return float(np.median(d[:, 1])) * math.sqrt(len(pick) / len(p))


def voxel_components(positions: np.ndarray, voxel: float) -> np.ndarray:
    """Connected components of points through occupied voxels (26-neighbourhood)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    p = np.asarray(positions, np.float64)
    if len(p) == 0:
        return np.zeros(0, np.int64)
    keys = np.floor((p - p.min(axis=0)) / voxel).astype(np.int64)
    span = keys.max(axis=0) + 3
    flat = (keys[:, 0] + 1) * span[1] * span[2] + (keys[:, 1] + 1) * span[2] + (keys[:, 2] + 1)
    cells, inverse = np.unique(flat, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    src, dst = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                other = cells + dx * span[1] * span[2] + dy * span[2] + dz
                pos = np.searchsorted(cells, other)
                pos = np.clip(pos, 0, len(cells) - 1)
                hit = cells[pos] == other
                src.append(np.flatnonzero(hit))
                dst.append(pos[hit])
    s = np.concatenate(src) if src else np.zeros(0, np.int64)
    d = np.concatenate(dst) if dst else np.zeros(0, np.int64)
    graph = coo_matrix((np.ones(s.size), (s, d)), shape=(len(cells), len(cells)))
    _, cell_label = connected_components(graph, directed=False)
    return cell_label[inverse]


def hole_clusters(
    splats: Splats,
    classes: np.ndarray,
    focus: Focus,
    *,
    best: np.ndarray | None = None,
    min_size: int = MIN_CLUSTER,
) -> Clusters:
    """The weak and unknown gaussians within the focus, grouped by voxel connectivity
    (`CLUSTER_SPACING` median spacings); clusters smaller than `min_size` are dropped.
    Clusters are numbered by fill weight, largest first."""
    todo = (classes != KNOWN) & focus.inside(splats.positions)
    labels = np.full(len(splats), -1, np.int64)
    rows = np.flatnonzero(todo)
    if rows.size == 0:
        return Clusters(labels, [])
    voxel = CLUSTER_SPACING * median_spacing(splats.positions[rows])
    comp = voxel_components(splats.positions[rows], max(voxel, 1e-9))
    weights = fill_weights(splats, classes, best)
    groups = []
    for c in np.unique(comp):
        members = rows[comp == c]
        if members.size >= min_size:
            groups.append((float(weights[members].sum()), members))
    groups.sort(key=lambda g: -g[0])
    info = []
    for k, (w, members) in enumerate(groups):
        labels[members] = k
        pos = splats.positions[members]
        info.append(
            {
                "cluster": k,
                "gaussians": int(members.size),
                "unknown": int((classes[members] == UNKNOWN).sum()),
                "weak": int((classes[members] == WEAK).sum()),
                "weight": round(w, 6),
                "centre": pos.mean(axis=0).round(4).tolist(),
                "low": pos.min(axis=0).round(4).tolist(),
                "high": pos.max(axis=0).round(4).tolist(),
            }
        )
    return Clusters(labels, info)


def fill_weights(splats: Splats, classes: np.ndarray, best: np.ndarray | None = None) -> np.ndarray:
    """What filling each gaussian is worth: its area (two largest scales) times its deficit
    (1 for unknown; for weak, how far its best quality falls short of known)."""
    s = np.sort(np.asarray(splats.scales, np.float64), axis=1)
    area = s[:, 2] * s[:, 1]
    deficit = np.where(classes == UNKNOWN, 1.0, 0.0)
    if best is not None:
        weak = classes == WEAK
        deficit = np.where(weak, np.clip(1.0 - best / Q_KNOWN, 0.1, 1.0), deficit)
    else:
        deficit = np.where(classes == WEAK, 0.5, deficit)
    return area * deficit
