"""Which virtual views to fill, in what order, and which real photos to show with each.

The inferred-fill bake-off's round 2 fills a scan at the poses it needs rather than along
camera paths. This module chooses them, with nothing picked by hand:

1. **Candidates** (`candidate_cameras`): rings about what the cameras filmed (`Focus`) at
   elevations -15, 15, 40, 65 and 90 degrees, 24 azimuths each, at the real cameras' median
   distance and field of view.
2. **Coverage** (`coverage`): a candidate covers a weak or unknown gaussian when the
   gaussian passes a depth test there, faces it (|cos| > 0.5 to its surface normal) and is
   seen at enough resolution (`RES_MIN` of the capture's own). Each candidate's pixels are
   classed (`pixel_classes`: known, weak, unknown, void; a hole enclosed by the scan is
   unknown, open background is not), and only views with something to fill (`DEFICIT_MIN`
   of their pixels weak or unknown) that keep context (`UNKNOWN_MAX` unknown at most) and
   do not look out from under or behind a surface (`BACK_MAX`) are eligible.
3. **Greedy weighted max coverage** (`select_views`): anchors first, each adding the most
   not-yet-seen fill weight (area x deficit); then propagation views until about
   `COVER_TARGET` of what can be covered is seen well by `MULTIPLICITY` views. Views too close
   in direction to one already chosen are suppressed (`NMS_DEG`). Greedy max coverage keeps
   the (1 - 1/e) guarantee.
4. **Order** (`propagation_order`, `tour`): propagation views from the one nearest what is
   already filled outwards; the joint (video) arm walks all views in a nearest-neighbour
   tour so consecutive frames are close.
5. **Context** (`retrieve_context`, VMem's idea): the known gaussians on a hole's border
   vote for the real cameras that saw them (by their quality); the top two, suppressed by
   direction, then one wide shot and one close-up of the same surface. The editor takes
   1-3 images, so a fill gets the render and two of these, rotated across seeds
   (`context_for_seed`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import fill_quality as fq
from splat_render import Camera, Frame, Splats

ELEVATIONS = (-15.0, 15.0, 40.0, 65.0, 90.0)
AZIMUTHS = 24
#: A candidate covers a gaussian it sees at |cos| above this...
COVER_COS = 0.5
#: ...and at this share of the capture's resolution or better (at the fill's size).
RES_MIN = 0.5
#: A view is eligible with at least this share of its pixels to fill (weak or unknown) and
#: at most this share unknown (the rest is context).
DEFICIT_MIN = 0.10
UNKNOWN_MAX = 0.50
#: ...and when the scan covers at least this share of the frame...
SHOWN_MIN = 0.2
#: ...and at most this share of what it shows is the back of a surface the cameras saw: an
#: eye under the ground or behind a bush sees mostly backs (the spool's -15 degree ring,
#: at ground level, 0.34-0.95; the pumpkin's 15 degree ring at most 0.25).
BACK_MAX = 0.25
#: View directions closer than this to a chosen one are suppressed.
NMS_DEG = 20.0
COVER_TARGET = 0.95
MULTIPLICITY = 2
ANCHORS = (4, 8)
PROPAGATION = (12, 24)
#: Elements of the coverage problem (gaussians to fill), at most this many (sampled).
MAX_ELEMENTS = 30000
#: Front-surface test of the pixel classes: a class's render counts at a pixel when it is no
#: farther than the full render there by this share.
FRONT_TOLERANCE = 0.03


def _direction(elevation: float, azimuth: float) -> np.ndarray:
    e, a = math.radians(elevation), math.radians(azimuth)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def candidate_directions(
    elevations: Sequence[float] = ELEVATIONS, azimuths: int = AZIMUTHS
) -> list[tuple[float, float, np.ndarray]]:
    """(elevation, azimuth, unit direction from the focus to the eye); one at each pole."""
    out = []
    for elev in elevations:
        count = 1 if abs(abs(elev) - 90.0) < 1e-6 else azimuths
        for k in range(count):
            az = 360.0 * k / azimuths
            out.append((float(elev), az, _direction(elev, az)))
    return out


def view_camera(
    focus: fq.Focus, direction: np.ndarray, size: tuple[int, int], fov_deg: float | None = None
) -> Camera:
    """A camera at the focus' distance in `direction`, looking at its centre."""
    eye = focus.centre + focus.distance * np.asarray(direction, np.float64)
    up = tuple(focus.up)
    if abs(float(np.dot(direction, focus.up))) > 0.999:
        up = (0.0, 1.0, 0.0)
    return Camera.look_at(
        eye, focus.centre, fov_deg=fov_deg or focus.hfov_deg, width=size[0], height=size[1], up=up
    )


def scaled(camera: Camera, width: int, height: int | None = None) -> Camera:
    h = height if height is not None else max(1, round(camera.height * width / camera.width))
    return Camera(
        camera.rotation, camera.centre, camera.focal * width / camera.width, width, h, camera.far
    )


# --- pixel classes ---------------------------------------------------------------------------------


@dataclass
class PixelClasses:
    """One view's pixels, a partition: known, weak, unknown (to generate, including holes
    the scan encloses), void (open background). `colour` is the scan's own colour (un-
    premultiplied, sampling gaps smoothed), `depth` the front surface's depth (inf where
    nothing), `full` the frame drawn."""

    known: np.ndarray
    weak: np.ndarray
    unknown: np.ndarray
    void: np.ndarray
    colour: np.ndarray
    depth: np.ndarray
    full: Frame
    #: Pixels whose front surface is the back of what the cameras saw (none without facing).
    back: np.ndarray | None = None

    @property
    def shown(self) -> np.ndarray:
        return ~self.void

    def shares(self, region: np.ndarray | None = None) -> dict[str, float]:
        """The frame's share the scan shows, and of the shown pixels (within `region`, the
        pixels near the holes, when given) the unknown, weak and deficit (either) shares;
        and of all shown pixels the share that shows the back of a seen surface."""
        total = self.known.size
        shown = self.shown if region is None else self.shown & region
        n = max(int(shown.sum()), 1)
        back = 0.0
        if self.back is not None:
            back = float((self.back & self.shown).sum() / max(int(self.shown.sum()), 1))
        return {
            "shown": float(self.shown.sum() / total),
            "local": float(shown.sum() / total),
            "unknown": float((self.unknown & shown).sum() / n),
            "weak": float((self.weak & shown).sum() / n),
            "deficit": float(((self.unknown | self.weak) & shown).sum() / n),
            "back": back,
        }


def _covered(alpha: np.ndarray) -> np.ndarray:
    import teacher_fill as tf

    return tf._covered(alpha)


def unpremultiply(frame: Frame) -> np.ndarray:
    """A frame's colour divided by its coverage, its sampling gaps filled from around."""
    import cv2

    a = frame.alpha.astype(np.float32)
    w = cv2.GaussianBlur(a, (0, 0), 1.0)
    smooth = (
        cv2.GaussianBlur(frame.rgb.astype(np.float32), (0, 0), 1.0) / np.maximum(w, 1e-4)[..., None]
    )
    colour = np.where((a >= 0.5)[..., None], frame.rgb / np.maximum(a, 1e-6)[..., None], smooth)
    return np.clip(colour, 0.0, 1.0)


def enclosed(void: np.ndarray) -> np.ndarray:
    """The void pixels that do not reach the frame's edge: holes the scan surrounds."""
    import cv2

    count, labels = cv2.connectedComponents(void.astype(np.uint8), connectivity=4)
    if count <= 1:
        return np.zeros_like(void)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    inside = np.isin(labels, edge, invert=True) & void
    return inside


def pixel_classes(
    splats: Splats,
    camera: Camera,
    renderer: Any,
    known: np.ndarray,
    weak: np.ndarray,
    facing: np.ndarray | None = None,
) -> PixelClasses:
    """`camera`'s pixels classed by the front surface: known where the known gaussians
    alone draw a surface there no farther than everything drawn (`FRONT_TOLERANCE`), weak
    likewise for known and weak together, unknown where something else is in front, where
    the front surface is seen from the side no real camera saw it from (`facing`, per
    gaussian: `fill_quality.facing` for this camera), or in a hole the scan encloses; void
    elsewhere. `known` and `weak` are per gaussian of `splats`."""
    import teacher_fill as tf

    full = renderer(splats, camera)
    kw = renderer(splats, camera, opacity_scale=(known | weak).astype(np.float64))
    k = renderer(splats, camera, opacity_scale=known.astype(np.float64))
    covered = _covered(full.alpha)
    front = np.where(np.isfinite(full.depth), full.depth, np.inf)

    def in_front(f: Frame) -> np.ndarray:
        d = np.where(np.isfinite(f.depth), f.depth, np.inf)
        return _covered(f.alpha) & (d <= front * (1 + FRONT_TOLERANCE) + 1e-6)

    k_front = in_front(k) & covered
    kw_front = (in_front(kw) & covered) | k_front
    back = None
    if facing is not None and not np.all(facing):
        # The front surface's facing share, drawn as a colour: below half, it is the back of
        # what the cameras saw (the underside of a lawn from below).
        f = np.repeat(np.asarray(facing, np.float64)[:, None], 3, axis=1)
        drawn = renderer(
            Splats(splats.positions, splats.rotations, splats.scales, f, splats.opacities), camera
        )
        import cv2

        # Smoothed like the colour (`unpremultiply`): a sampling gap is not a back.
        share = cv2.GaussianBlur(drawn.rgb[..., 0].astype(np.float32), (0, 0), 1.0)
        alpha = cv2.GaussianBlur(drawn.alpha.astype(np.float32), (0, 0), 1.0)
        back = covered & (alpha > 0.05) & (share < 0.5 * alpha)
        k_front &= ~back
        kw_front &= ~back
    holes = enclosed(~covered)
    unknown = tf.clean_mask((covered & ~kw_front) | holes)
    known_px = k_front & ~unknown
    # Weak: the weak front surface, and specks of unknown too small to fill (the scan's
    # look is kept there and refined).
    weak_px = covered & ~known_px & ~unknown
    void = ~(known_px | weak_px | unknown)
    return PixelClasses(known_px, weak_px, unknown, void, unpremultiply(full), front, full, back)


# --- coverage --------------------------------------------------------------------------------------


@dataclass
class Candidate:
    index: int
    elevation: float
    azimuth: float
    direction: np.ndarray
    covers: np.ndarray  # (elements,) bool
    shares: dict[str, float]
    eligible: bool
    clusters: dict[int, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "elevation": self.elevation,
            "azimuth": round(self.azimuth, 2),
            "covers": int(self.covers.sum()),
            "eligible": self.eligible,
            **{k: round(v, 3) for k, v in self.shares.items()},
        }


def sample_elements(
    clusters: fq.Clusters, weights: np.ndarray, limit: int = MAX_ELEMENTS
) -> tuple[np.ndarray, np.ndarray]:
    """The gaussians the coverage problem is over (those in a cluster), at most `limit`
    sampled, and their weights (scaled so each cluster keeps its total)."""
    rows = np.flatnonzero(clusters.labels >= 0)
    w = np.asarray(weights, np.float64)[rows]
    if rows.size > limit:
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(rows.size, limit, replace=False))
        scale = np.zeros(len(clusters.info))
        for k in range(len(clusters.info)):
            mine = clusters.labels[rows] == k
            got = clusters.labels[rows[pick]] == k
            scale[k] = w[mine].sum() / max(w[pick][got].sum(), 1e-18)
        rows, w = rows[pick], w[pick] * scale[clusters.labels[rows[pick]]]
    return rows, w


def cluster_spheres(clusters: fq.Clusters, grow: float = 1.5) -> list[tuple[np.ndarray, float]]:
    """A sphere about each hole cluster (its bounds' half diagonal, grown): the
    neighbourhood whose pixels a view's shares are counted over."""
    out = []
    for c in clusters.info:
        low, high = np.asarray(c["low"], float), np.asarray(c["high"], float)
        out.append(((low + high) / 2, grow * 0.5 * float(np.linalg.norm(high - low)) + 1e-6))
    return out


def near_holes(
    camera: Camera,
    depth: np.ndarray,
    holes: np.ndarray,
    spheres: Sequence[tuple[np.ndarray, float]],
) -> np.ndarray:
    """The pixels whose front surface lies in one of the `spheres` (and, for the pixels of
    an enclosed hole, which have no surface, whose ray passes through one)."""
    if not spheres:
        return np.ones(depth.shape, bool)
    rays = camera.rays()
    forward = rays @ camera.rotation[2]
    t = np.where(np.isfinite(depth), depth / np.maximum(forward, 1e-9), np.nan)
    points = camera.centre + rays * t[..., None]
    out = np.zeros(depth.shape, bool)
    for centre, radius in spheres:
        d = np.linalg.norm(points - centre, axis=-1)
        out |= np.nan_to_num(d, nan=np.inf) <= radius
        # Ray to sphere: the distance from its centre to the ray.
        along = (centre - camera.centre) @ rays.reshape(-1, 3).T
        closest = camera.centre + rays * along.reshape(depth.shape)[..., None]
        miss = np.linalg.norm(closest - centre, axis=-1)
        out |= holes & (miss <= radius) & (along.reshape(depth.shape) > 0)
    return out


def coverage(
    splats: Splats,
    probe: Camera,
    fill_focal: float,
    elements: np.ndarray,
    normals: np.ndarray,
    focus: fq.Focus,
    renderer: Any,
    known: np.ndarray,
    weak: np.ndarray,
    labels: np.ndarray | None = None,
    quality: fq.Quality | None = None,
    spheres: Sequence[tuple[np.ndarray, float]] = (),
) -> tuple[np.ndarray, dict[str, float], dict[int, float]]:
    """Which `elements` (gaussian indices) the view covers (depth test at the probe's size;
    |cos| > `COVER_COS`; resolution at the fill's focal >= `RES_MIN` of the capture's; from
    the side the real cameras saw it, with `quality`), the view's pixel shares (near the
    holes: `spheres`), and per cluster how many of its elements it covers."""
    face = None if quality is None else fq.facing(quality, probe.centre, splats.positions)
    pc = pixel_classes(splats, probe, renderer, known, weak, face)
    uv, z = probe.project(splats.positions[elements])
    u = np.floor(uv[:, 0]).astype(np.int64)
    v = np.floor(uv[:, 1]).astype(np.int64)
    inside = (z > 1e-3) & (u >= 0) & (u < probe.width) & (v >= 0) & (v < probe.height)
    covers = np.zeros(elements.size, bool)
    rows = np.flatnonzero(inside)
    if rows.size:
        surface = fq.nearest_surface(pc.depth)[v[rows], u[rows]]
        reach = 2.0 * splats.scales[elements[rows]].max(axis=1)
        limit = surface * (1 + fq.DEPTH_TOLERANCE) + np.minimum(reach, 0.05 * surface)
        ok = np.isfinite(surface) & (z[rows] <= limit)
        if face is not None:
            ok &= face[elements[rows]]
        d = splats.positions[elements[rows]] - probe.centre
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        cos = np.abs(np.einsum("ij,ij->i", d, normals[elements[rows]]))
        res = (fill_focal / np.maximum(z[rows], 1e-9)) / focus.density
        covers[rows] = ok & (cos > COVER_COS) & (res >= RES_MIN)
    per_cluster: dict[int, float] = {}
    if labels is not None and covers.any():
        ids, counts = np.unique(labels[elements[covers]], return_counts=True)
        per_cluster = {int(i): float(c) for i, c in zip(ids, counts, strict=True) if i >= 0}
    region = near_holes(probe, pc.depth, pc.unknown & ~np.isfinite(pc.depth), spheres)
    shares = pc.shares(region if spheres else None)
    shares["frameUnknown"] = float((pc.unknown & pc.shown).sum() / max(int(pc.shown.sum()), 1))
    return covers, shares, per_cluster


def eligible(shares: dict[str, float], *, relaxed: bool = False) -> bool:
    """A view with something to fill near the holes (`DEFICIT_MIN`, unless `relaxed`) that
    keeps context there and in the frame (`UNKNOWN_MAX`), from an eye in the open
    (`BACK_MAX`)."""
    return (
        shares["shown"] >= SHOWN_MIN
        and shares.get("local", 1.0) > 0.0
        and (relaxed or shares["deficit"] >= DEFICIT_MIN)
        and shares["unknown"] <= UNKNOWN_MAX
        and shares.get("frameUnknown", 0.0) <= UNKNOWN_MAX
        and shares.get("back", 0.0) <= BACK_MAX
    )


# --- greedy selection ---------------------------------------------------------------------------------


@dataclass
class Selection:
    anchors: list[int]
    propagation: list[int]
    report: dict[str, Any]


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def select_views(
    directions: np.ndarray,
    covers: np.ndarray,
    weights: np.ndarray,
    usable: np.ndarray,
    *,
    anchors: tuple[int, int] = ANCHORS,
    propagation: tuple[int, int] = PROPAGATION,
    nms_deg: float = NMS_DEG,
    target: float = COVER_TARGET,
    multiplicity: int = MULTIPLICITY,
    padding: np.ndarray | None = None,
) -> Selection:
    """Greedy weighted max coverage over candidate views (`covers`: candidates x elements).

    Anchors: each pick adds the most weight not yet seen by any chosen view, until
    `target` of the coverable weight is seen once (at least `anchors[0]`, at most
    `anchors[1]`). Propagation: each adds the most weight not yet seen `multiplicity` times,
    until `target` of what can be is (at least `propagation[0]`, padded with the views that
    best re-cover the least-covered weight, from `padding` -- a wider set -- when given; at
    most `propagation[1]`). Only `usable` candidates, none within `nms_deg` of one chosen."""
    raw = np.asarray(covers, bool)
    covers = raw & np.asarray(usable, bool)[:, None]
    pad_covers = covers if padding is None else raw & np.asarray(padding, bool)[:, None]
    w = np.asarray(weights, np.float64)
    # What any view that may be chosen covers (and so may be counted).
    raw = covers | pad_covers
    reach = raw.sum(axis=0)
    possible = np.minimum(reach, multiplicity).astype(np.float64)
    once_total = float(w[reach > 0].sum())
    multi_total = float((w * possible).sum())
    count = np.zeros(w.size, np.int64)
    chosen: list[int] = []
    steps: list[dict[str, Any]] = []

    def blocked(c: int) -> bool:
        return any(_angle(directions[c], directions[o]) < nms_deg for o in chosen)

    def pick(gains: np.ndarray) -> int | None:
        order = np.argsort(-gains, kind="stable")
        for c in order:
            if gains[c] <= 0:
                return None
            if c in chosen or blocked(int(c)):
                continue
            return int(c)
        return None

    def once() -> float:
        return float(w[count > 0].sum()) / max(once_total, 1e-18)

    def multi() -> float:
        return float((w * np.minimum(count, possible)).sum()) / max(multi_total, 1e-18)

    def take(c: int, phase: str, gain: float) -> None:
        chosen.append(c)
        count[raw[c]] += 1
        steps.append(
            {"candidate": c, "phase": phase, "gain": gain, "once": once(), "multi": multi()}
        )

    # Anchors.
    while len(chosen) < anchors[1]:
        if len(chosen) >= anchors[0] and once() >= target:
            break
        gains = (covers & (count == 0)[None, :]) @ w
        if pick(gains) is None:
            # Everything coverable is seen once: the second sight of what one view has seen.
            gains = (covers & (count < possible)[None, :]) @ w
        if pick(gains) is None and len(chosen) < anchors[0]:
            # ...and twice: still the minimum, re-covering the least-covered weight.
            gains = covers @ (w / (1.0 + count))
        c = pick(gains)
        if c is None:
            break
        take(c, "anchor", float(gains[c]))
    n_anchor = len(chosen)
    # Propagation.
    while len(chosen) - n_anchor < propagation[1]:
        if len(chosen) - n_anchor >= propagation[0] and multi() >= target:
            break
        gains = (covers & (count < possible)[None, :]) @ w
        c = pick(gains)
        if c is None:
            if len(chosen) - n_anchor >= propagation[0]:
                break
            # Padding: the view that best re-covers the least-covered weight.
            pad = pad_covers @ (w / (1.0 + count))
            c = pick(pad)
            if c is None:
                break
            take(c, "pad", float(pad[c]))
            continue
        take(c, "propagation", float(gains[c]))
    report = {
        "candidates": len(directions),
        "usable": int(np.asarray(usable).sum()),
        "elements": int(w.size),
        "coverable": round(float(w[reach > 0].sum() / max(w.sum(), 1e-18)), 4),
        "seenOnce": round(once(), 4),
        "seenTwice": round(multi(), 4),
        "steps": [{**s, "gain": round(s["gain"], 8)} for s in steps],
    }
    return Selection(chosen[:n_anchor], chosen[n_anchor:], report)


def propagation_order(
    directions: np.ndarray, anchors: Sequence[int], rest: Sequence[int]
) -> list[int]:
    """`rest` ordered outwards from what is filled: each next view the one nearest (in
    direction) to the anchors or a view already ordered."""
    done = list(anchors)
    left = list(rest)
    out: list[int] = []
    while left:
        if done:
            k = min(left, key=lambda c: min(_angle(directions[c], directions[d]) for d in done))
        else:
            k = left[0]
        out.append(k)
        done.append(k)
        left.remove(k)
    return out


def tour(directions: np.ndarray, views: Sequence[int], start: int | None = None) -> list[int]:
    """A nearest-neighbour walk over `views` (by direction), from `start` (default the
    first)."""
    left = list(views)
    if not left:
        return []
    k = start if start is not None and start in left else left[0]
    out = [k]
    left.remove(k)
    while left:
        k = min(left, key=lambda c: _angle(directions[c], directions[out[-1]]))
        out.append(k)
        left.remove(k)
    return out


def choose_views(
    splats: Splats,
    classes: np.ndarray,
    normals: np.ndarray,
    best: np.ndarray | None,
    clusters: fq.Clusters,
    focus: fq.Focus,
    renderer: Any,
    *,
    fill_size: tuple[int, int],
    probe_size: tuple[int, int],
    elevations: Sequence[float] = ELEVATIONS,
    azimuths: int = AZIMUTHS,
    anchors: tuple[int, int] = ANCHORS,
    propagation: tuple[int, int] = PROPAGATION,
    quality: fq.Quality | None = None,
) -> tuple[list[Candidate], Selection, dict[int, Camera]]:
    """Every candidate's coverage of the hole clusters, the greedy selection, and the
    chosen views' cameras at the fill size (by candidate index). When no candidate is
    eligible, every candidate that covers something is (`report["relaxed"]`)."""
    known = classes == fq.KNOWN
    weak = classes == fq.WEAK
    weights = fq.fill_weights(splats, classes, best)
    elements, w = sample_elements(clusters, weights)
    spheres = cluster_spheres(clusters)
    candidates: list[Candidate] = []
    cameras: dict[int, Camera] = {}
    for k, (elev, az, d) in enumerate(candidate_directions(elevations, azimuths)):
        cam = view_camera(focus, d, fill_size)
        probe = scaled(cam, probe_size[0], probe_size[1])
        covers, shares, per_cluster = coverage(
            splats,
            probe,
            cam.focal,
            elements,
            normals,
            focus,
            renderer,
            known,
            weak,
            clusters.labels,
            quality,
            spheres,
        )
        candidates.append(Candidate(k, elev, az, d, covers, shares, eligible(shares), per_cluster))
        cameras[k] = cam
    dirs = np.array([c.direction for c in candidates])
    matrix = np.array([c.covers for c in candidates]).reshape(len(candidates), elements.size)
    usable = np.array([c.eligible for c in candidates])
    wider = np.array([eligible(c.shares, relaxed=True) for c in candidates])
    relaxed = False
    if not (usable & matrix.any(axis=1)).any():
        usable = wider if (wider & matrix.any(axis=1)).any() else matrix.any(axis=1)
        relaxed = True
    selection = select_views(
        dirs, matrix, w, usable, anchors=anchors, propagation=propagation, padding=wider | usable
    )
    selection.report["relaxed"] = relaxed
    chosen = [*selection.anchors, *selection.propagation]
    return candidates, selection, {k: cameras[k] for k in chosen}


# --- context --------------------------------------------------------------------------------------------


def border_gaussians(
    positions: np.ndarray, members: np.ndarray, candidates: np.ndarray, radius: float
) -> np.ndarray:
    """Of `candidates` (indices, typically the known gaussians), those within `radius` of a
    member of the hole."""
    from scipy.spatial import cKDTree

    if members.size == 0 or candidates.size == 0:
        return np.zeros(0, np.int64)
    p = np.asarray(positions, np.float64)
    low = p[members].min(axis=0) - radius
    high = p[members].max(axis=0) + radius
    near = candidates[np.all((p[candidates] >= low) & (p[candidates] <= high), axis=1)]
    if near.size == 0:
        return near
    d, _ = cKDTree(p[members]).query(p[near], distance_upper_bound=radius)
    return near[np.isfinite(d)]


def retrieve_context(
    per_camera: Any,
    voters: np.ndarray,
    camera_centres: np.ndarray,
    centre: np.ndarray,
    *,
    nms_deg: float = NMS_DEG,
    share: float = 0.25,
) -> dict[str, Any]:
    """The real photos to show a fill of the hole whose border is `voters` (gaussian
    indices): every camera's vote is the sum of its quality over them. `top`: the two best,
    the second at least `nms_deg` from the first in direction (from the hole). `wide` and
    `close`: of the cameras with at least `share` of the best vote, not already taken, the
    farthest from and the nearest to the hole."""
    votes = np.asarray(per_camera[:, voters].sum(axis=1)).ravel() if voters.size else None
    if votes is None or not np.any(votes > 0):
        return {"top": [], "wide": None, "close": None, "votes": {}}
    away = camera_centres - centre
    dist = np.linalg.norm(away, axis=1)
    dirs = away / np.maximum(dist[:, None], 1e-12)
    top: list[int] = []
    for c in np.argsort(-votes, kind="stable"):
        if votes[c] <= 0 or len(top) == 2:
            break
        if all(_angle(dirs[c], dirs[t]) >= nms_deg for t in top):
            top.append(int(c))
    strong = [int(c) for c in np.flatnonzero(votes >= share * votes.max()) if int(c) not in top]
    wide = max(strong, key=lambda c: dist[c]) if strong else None
    close = min((c for c in strong if c != wide), key=lambda c: dist[c], default=None)
    ranked = np.argsort(-votes, kind="stable")[:8]
    return {
        "top": top,
        "wide": wide,
        "close": close,
        "votes": {int(c): round(float(votes[c]), 4) for c in ranked if votes[c] > 0},
    }


#: The pair of context photos each seed is shown (the editor's optimum is 1-3 images: the
#: render and two photos); the wide shot and the close-up rotate in.
SEED_PAIRS = (("top0", "top1"), ("top0", "wide"), ("top1", "close"), ("wide", "close"))


def context_roles(context: dict[str, Any]) -> dict[str, int]:
    roles: dict[str, int] = {}
    for k, c in enumerate(context.get("top", [])):
        roles[f"top{k}"] = int(c)
    for name in ("wide", "close"):
        if context.get(name) is not None:
            roles[name] = int(context[name])
    return roles


def context_for_seed(context: dict[str, Any], seed_index: int) -> list[tuple[str, int]]:
    """The (role, camera) pair shown with seed `seed_index` (`SEED_PAIRS`, skipping roles
    the hole has none for; at most two, never the same camera twice)."""
    roles = context_roles(context)
    if not roles:
        return []
    want = SEED_PAIRS[seed_index % len(SEED_PAIRS)]
    out: list[tuple[str, int]] = []
    for name in (*want, "top0", "top1", "wide", "close"):
        if name in roles and roles[name] not in [c for _, c in out]:
            out.append((name, roles[name]))
        if len(out) == 2:
            break
    return out
