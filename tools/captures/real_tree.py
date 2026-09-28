"""A trained splat of a real tree to a Living Survey site: metres, one tree, a rig, site.json.

The post-train half of the Minnetonka tree (docs/CAPTURES.md "The real tree"; the GPU half
is tools/pipeline/experiments/minnetonka.py and infra/modal/minnetonka.py). CPU only, and
every step writes what it measured into ``source/real_tree.json`` so a human can check it
without re-running anything:

1. **Into metres.** The trained PLY is in the pose set's own frame. ``frame.json`` (the
   pose gate's output) holds the similarity into metres, east/north/up: up from the gimbal
   pitch of every photo, scale from their barometric heights, heading from their compass.
   Positions, scales and rotations are all moved -- a gaussian is an ellipsoid, not a point.
2. **The ground** is the densest 5 cm layer of opaque splats under the orbit: a lawn
   reconstructs as a sheet, and a sheet is a histogram spike.
3. **The trunk**, measured: in three horizontal slabs 0.3-1.2 m above that ground, a
   circle is fitted (RANSAC over three-point circles, then least squares on its inliers)
   to the opaque splats near the axis the cameras look at -- a trunk is a hollow shell of
   splats, so its cross-section is a ring. The median centre becomes the site's origin and
   the median diameter is reported. It is a *check* on the barometric scale (a mature
   tree's trunk is tens of centimetres, not metres or millimetres), and an explicit
   ``--trunk-diameter-m`` turns it into the scale: whatever diameter is measured is scaled
   to the one given, about the trunk's foot. Nothing published gives this tree's diameter,
   so that stays an input rather than a default.
4. **Isolate** the tree: a cylinder of the crown's radius about the trunk (from the splats
   above head height inside the camera ring, or ``--crown-radius-m``), everything
   ``--ground-clearance-m`` above the ground, opaque, not isolated haze -- and then the
   packer's own floater rule, run to a fixed point so the packer drops nothing (the rig's
   checksum is over exactly the splats written).
5. **At most** ``--max-splats`` (400k, the viewer's Standard Detail budget): a single tile
   cannot be trimmed by the viewer's budget, so the tile itself has to fit it. The least
   significant splats go first (opacity x projected area).
6. **Rig** with ``skeleton.extract`` -- single tile (``tile_gaussians=None``), because the
   Living Survey deformer refuses anything else until M5 -- and the deformer's own
   ``upright`` refusal checked here first (``uprightness``, a port of ``treeUprightness``
   in apps/web/src/cesium/splatFrames.ts).
7. **site.json**, the shape ``build_site.py`` writes and ``app.seed.captures`` reads, with
   the splat asset's ``rig: ../source/rig.json``.

Usage::

    uv run python real_tree.py trained.ply ../../data/tiles/minnetonka-tree \\
        --frame frame.json [--train train.json] [--pose pose.json] \\
        [--trunk-diameter-m 0.35 | --metres-per-unit 0.41] [--crown-radius-m 4.5]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import skeleton
import splat_tiles
from build_site import write_site_document
from synthetic_tree import write_ply

SLUG = "minnetonka-tree"
LATITUDE = 44.944565
LONGITUDE = -93.425903
#: EGM2008 geoid height near Minneapolis, approximately (+-2 m). Only the site's recorded
#: centre height uses it: the splat asset is clamped to the viewer's terrain.
GEOID_UNDULATION_M = -27.6

#: Where the trunk is measured: slabs above the ground, metres. Below the first limbs of a
#: mature street tree, above the mown grass and whatever is at its foot.
TRUNK_SLABS_M = ((0.3, 0.6), (0.6, 0.9), (0.9, 1.2))
#: How far from the orbit's axis the trunk may be, metres.
TRUNK_SEARCH_M = 2.5
#: What makes a fitted circle a trunk: splats round at least half of it, enough of them,
#: and several times denser than the slab's splats spread evenly would put there.
RING_MIN_ARC = 0.5
RING_MIN_INLIERS = 20
RING_MIN_CONTRAST = 4.0
#: The narrowest crown search, metres from the trunk: a mature tree's crown radius.
CROWN_SEARCH_M = 8.0
#: Opaque enough to be surface: what the ground and trunk measurements count.
SURFACE_OPACITY = 0.5
#: The viewer's Standard Detail budget (apps/web/src/lib/detail.ts DEFAULT_SPLAT_BUDGET).
DEFAULT_MAX_SPLATS = 400_000

ATTRIBUTION = "Matthew Guertin"
ATTRIBUTION_URL = "https://mattguertin.com"
LICENSE_NAME = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
SOURCE_URL = "https://github.com/Matt1Up/tree-photogrammetry-dataset"
CAPTURED = "2020-07-20"


# ------------------------------------------------------------------------------ into metres


def _matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    """A proper rotation matrix to a w-first unit quaternion."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    quat = np.asarray(q, dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, w-first; `a` a single quaternion, `b` an (n, 4) array."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


def similarity(matrix: Any) -> tuple[float, np.ndarray, np.ndarray]:
    """(scale, rotation, translation) of a 4x4 uniform-scale similarity; refuses anything
    else, since a shear or a mirror would turn a gaussian into something it is not."""
    m = np.asarray(matrix, dtype=np.float64)
    linear = m[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if not scale > 0:
        raise ValueError("the frame's matrix is a reflection or degenerate, not a similarity")
    rotation = linear / scale
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError("the frame's matrix is not a uniform-scale similarity")
    return scale, rotation, m[:3, 3].copy()


def transform_splats(data: dict[str, np.ndarray], matrix: Any) -> dict[str, np.ndarray]:
    """Every gaussian through `x' = s R x + t`: centre moved, log-scales grown by ln s,
    rotation composed with R. Columns not listed (SH rest, normals) pass through."""
    scale, rotation, translation = similarity(matrix)
    out = dict(data)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    moved = xyz @ (scale * rotation).T + translation
    out["x"], out["y"], out["z"] = (moved[:, i].astype(np.float32) for i in range(3))
    for axis in ("scale_0", "scale_1", "scale_2"):
        out[axis] = (data[axis].astype(np.float64) + math.log(scale)).astype(np.float32)
    quats = np.stack([data[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    composed = _quat_multiply(_matrix_to_quat(rotation), quats)
    for i in range(4):
        out[f"rot_{i}"] = composed[:, i].astype(np.float32)
    return out


def take(data: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    return {name: column[keep] for name, column in data.items()}


def positions(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


# ---------------------------------------------------------------------------- measurements


def ground_height(
    xyz: np.ndarray,
    opacity: np.ndarray,
    radius: float,
    bin_m: float = 0.05,
    *,
    below: float | None = None,
) -> float:
    """The densest thin layer of opaque splats within `radius` of the axis, in the lower
    half of what is there: a lawn is a sheet, and a sheet is the histogram's spike.

    `below` is a height the ground must be under -- the lowest camera, since the drone did
    not fly underground. Without it a thick trunk can out-count the lawn: on the Minnetonka
    tree (m0) the densest layer was 3.5 m up the trunk, above the drone's low hover pass, and
    the "trunk" then measured was the 2.1 m fork where the limbs part."""
    near = (np.hypot(xyz[:, 0], xyz[:, 1]) <= radius) & (opacity >= SURFACE_OPACITY)
    if below is not None:
        near &= xyz[:, 2] < below
    z = xyz[near, 2]
    if z.size < 50:
        where = f"within {radius:g} m of the axis"
        if below is not None:
            where += f" below {below:.2f} m (the lowest camera)"
        raise ValueError(f"only {z.size} opaque splats {where}")
    lo, hi = float(np.percentile(z, 0.5)), float(np.percentile(z, 50))
    edges = np.arange(lo, hi + bin_m, bin_m)
    if edges.size < 3:
        return float(np.median(z))
    counts, edges = np.histogram(z, bins=edges)
    peak = int(np.argmax(counts))
    layer = z[(z >= edges[peak]) & (z < edges[peak + 1])]
    return float(np.median(layer))


def fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic (Kasa) least-squares circle: centre and radius."""
    a = np.column_stack([2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))])
    b = (xy**2).sum(axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    radius = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    return np.array([cx, cy]), radius


def ring_fit(
    xy: np.ndarray,
    *,
    min_radius: float = 0.03,
    max_radius: float = 1.0,
    trials: int = 3000,
    seed: int = 0,
    area: float | None = None,
) -> dict[str, Any] | None:
    """The circle most splats lie on: RANSAC over three-point circles, refined on its
    inliers. The inlier band scales with the circle (15 % of its radius, at least 2 cm),
    since bark splats sit on the surface with a scatter proportional to what they cover."""
    if len(xy) < 12:
        return None
    rng = np.random.default_rng(seed)
    best: tuple[int, np.ndarray, float] | None = None
    for _ in range(trials):
        sample = xy[rng.choice(len(xy), 3, replace=False)]
        centre, radius = fit_circle(sample)
        if not (min_radius <= radius <= max_radius):
            continue
        band = max(0.02, 0.15 * radius)
        inliers = np.abs(np.hypot(*(xy - centre).T) - radius) <= band
        count = int(inliers.sum())
        if best is None or count > best[0]:
            best = (count, centre, radius)
    if best is None:
        return None
    _, centre, radius = best
    for _ in range(3):
        band = max(0.02, 0.15 * radius)
        inliers = np.abs(np.hypot(*(xy - centre).T) - radius) <= band
        if inliers.sum() < 6:
            return None
        centre, radius = fit_circle(xy[inliers])
    angles = np.arctan2(*(xy[inliers] - centre).T[::-1])
    covered = np.unique(np.floor((angles + math.pi) / (2 * math.pi) * 12).astype(int)).size
    # How much denser the ring is than the points it was found among would be, spread
    # evenly over the area they were searched in: a bark shell is many times that, a
    # chance circle through haze is not.
    if area is None:
        area = math.pi * float(np.max(np.hypot(*(xy - xy.mean(axis=0)).T))) ** 2
    band = max(0.02, 0.15 * radius)
    expected = len(xy) * (4.0 * math.pi * radius * band) / max(area, 1e-9)
    return {
        "centre": [float(centre[0]), float(centre[1])],
        "radius": float(radius),
        "inliers": int(inliers.sum()),
        "of": len(xy),
        "arcCoverage": round(covered / 12, 3),
        "contrast": round(float(inliers.sum()) / max(expected, 1e-9), 2),
    }


def measure_trunk(
    xyz: np.ndarray, opacity: np.ndarray, ground: float, *, search: float = TRUNK_SEARCH_M
) -> dict[str, Any]:
    """Trunk centre and diameter from ring fits in `TRUNK_SLABS_M` above `ground`."""
    near = (np.hypot(xyz[:, 0], xyz[:, 1]) <= search) & (opacity >= SURFACE_OPACITY)
    slabs = []
    for lo, hi in TRUNK_SLABS_M:
        inside = near & (xyz[:, 2] >= ground + lo) & (xyz[:, 2] < ground + hi)
        fit = ring_fit(xyz[inside, :2], area=math.pi * search * search)
        if (
            fit is not None
            and fit["arcCoverage"] >= RING_MIN_ARC
            and fit["inliers"] >= RING_MIN_INLIERS
            and fit["contrast"] >= RING_MIN_CONTRAST
        ):
            slabs.append({"fromM": lo, "toM": hi, **fit})
    if not slabs:
        raise ValueError(
            f"no trunk-shaped ring within {search:g} m of the axis between "
            f"{TRUNK_SLABS_M[0][0]} and {TRUNK_SLABS_M[-1][1]} m above the ground"
        )
    centres = np.asarray([s["centre"] for s in slabs])
    centre = np.median(centres, axis=0)
    diameter = 2.0 * float(np.median([s["radius"] for s in slabs]))
    spread = float(np.max(np.hypot(*(centres - centre).T))) if len(slabs) > 1 else 0.0
    return {
        "centre": [float(centre[0]), float(centre[1])],
        "diameterM": round(diameter, 4),
        "slabs": slabs,
        "centreSpreadM": round(spread, 4),
    }


def crown_radius(
    xyz: np.ndarray, opacity: np.ndarray, cameras: np.ndarray | None
) -> dict[str, Any]:
    """How far the crown reaches from the trunk, from the splats above head height inside
    the camera ring: where the drone flew round the tree, what is inside the ring above
    2.5 m is the crown. 98th percentile of their distance, plus 0.3 m.

    Where it flew under the crown instead (Minnetonka: hover passes 2.7-4 m from the trunk,
    looking up), the ring is inside the crown and would cut it off, so the search is never
    narrower than `CROWN_SEARCH_M`."""
    ring = None
    if cameras is not None and len(cameras):
        ring = float(np.percentile(np.hypot(cameras[:, 0], cameras[:, 1]), 10))
    reach = np.hypot(xyz[:, 0], xyz[:, 1])
    limit = max(0.9 * ring, CROWN_SEARCH_M) if ring else CROWN_SEARCH_M
    crown = (xyz[:, 2] > 2.5) & (opacity >= SURFACE_OPACITY) & (reach <= limit)
    if crown.sum() < 50:
        raise ValueError(f"only {int(crown.sum())} opaque splats above 2.5 m inside {limit:.1f} m")
    radius = float(np.percentile(reach[crown], 98)) + 0.3
    return {"radiusM": round(min(radius, limit), 3), "cameraRingP10M": ring, "searchM": limit}


def profile(xyz: np.ndarray, opacity: np.ndarray, cameras: np.ndarray | None) -> dict[str, Any]:
    """Opaque splat counts by height (1 m rows) and distance from the axis (1 m columns),
    plus where the cameras are: enough to see where a tree is without the PLY."""
    opaque = opacity >= SURFACE_OPACITY
    z = xyz[opaque, 2]
    r = np.hypot(xyz[opaque, 0], xyz[opaque, 1])
    rows: dict[str, Any] = {}
    if z.size:
        lo, hi = math.floor(float(np.percentile(z, 0.5))), math.ceil(float(np.percentile(z, 99.5)))
        for h in range(lo, hi):
            band = (z >= h) & (z < h + 1)
            counts, _ = np.histogram(r[band], bins=np.arange(0.0, 13.0, 1.0))
            rows[f"{h}..{h + 1}"] = [int(c) for c in counts]
    out: dict[str, Any] = {
        "opaque": int(opaque.sum()),
        "of": len(xyz),
        "zPercentiles_1_50_99": [round(float(v), 2) for v in np.percentile(z, [1, 50, 99])]
        if z.size
        else None,
        "radiusColumnsM": "0..1, 1..2, ... 11..12",
        "countsByHeight": rows,
    }
    if cameras is not None and len(cameras):
        cr = np.hypot(cameras[:, 0], cameras[:, 1])
        out["cameras"] = {
            "zPercentiles_0_50_100": [
                round(float(v), 2) for v in np.percentile(cameras[:, 2], [0, 50, 100])
            ],
            "radiusPercentiles_0_10_50_100": [
                round(float(v), 2) for v in np.percentile(cr, [0, 10, 50, 100])
            ],
        }
    return out


def floater_fixpoint(xyz: np.ndarray, rounds: int = 10) -> np.ndarray:
    """splat_tiles' floater rule (drop beyond 1.5x the 99.5th-percentile radius about the
    median centre) applied until it drops nothing, so the packer drops nothing either."""
    keep = np.ones(len(xyz), dtype=bool)
    for _ in range(rounds):
        kept = xyz[keep].astype(np.float32)
        centre = np.median(kept, axis=0)
        radius = np.linalg.norm(kept - centre, axis=1)
        near = radius <= np.percentile(radius, 99.5) * 1.5
        if near.all():
            return keep
        keep[np.flatnonzero(keep)[~near]] = False
    return keep


def significance(data: dict[str, np.ndarray]) -> np.ndarray:
    """Opacity x projected area: what a splat contributes to a picture, roughly."""
    opacity = splat_tiles.sigmoid(data["opacity"].astype(np.float64))
    log_volume = sum(data[f"scale_{i}"].astype(np.float64) for i in range(3))
    return opacity * np.exp(2.0 / 3.0 * log_volume)


def cap_splats(data: dict[str, np.ndarray], limit: int) -> np.ndarray:
    """The `limit` most significant splats, in their original order (stable ties)."""
    count = len(data["x"])
    keep = np.ones(count, dtype=bool)
    if limit <= 0 or count <= limit:
        return keep
    order = np.argsort(-significance(data), kind="stable")
    keep[:] = False
    keep[order[:limit]] = True
    return keep


def uprightness(xyz: np.ndarray) -> dict[str, Any]:
    """`treeUprightness` (apps/web/src/cesium/splatFrames.ts), the deformer's `upright`
    refusal, on the positions it will see: crown spread over base spread, 90th-percentile
    radii of the bottom 15 % and the top half; a standing tree is at least 3."""

    def spread(points: np.ndarray) -> float:
        if len(points) < 8:
            return math.nan
        radii = np.hypot(*(points[:, :2] - points[:, :2].mean(axis=0)).T)
        return float(np.sort(radii)[min(len(radii) - 1, int(len(radii) * 0.9))])

    low, high = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    extent = high - low
    base = spread(xyz[xyz[:, 2] <= low + 0.15 * extent])
    crown = spread(xyz[xyz[:, 2] >= low + 0.5 * extent])
    ratio = crown / base if base > 0 else math.inf
    return {
        "baseSpreadM": round(base, 4),
        "crownSpreadM": round(crown, 4),
        "crownToBase": round(ratio, 3) if math.isfinite(ratio) else None,
        "upright": bool(math.isfinite(crown) and ratio >= 3.0),
    }


# ---------------------------------------------------------------------------------- the site


def _offset_lonlat(lat: float, lon: float, east: float, north: float) -> tuple[float, float]:
    return (
        lon + east / (111_320.0 * math.cos(math.radians(lat))),
        lat + north / 111_132.0,
    )


def site_document(
    *,
    lat: float,
    lon: float,
    ground_ellipsoid_m: float | None,
    crown_m: float,
    height_m: float,
    splats: int,
    photos: int | None,
    gsd_m: float | None,
    spacing_m: float,
    nodes: int,
    trained: dict[str, Any] | None,
) -> dict[str, Any]:
    ring = []
    for k in range(17):
        angle = 2 * math.pi * (k % 16) / 16
        east, north = crown_m * math.sin(angle), crown_m * math.cos(angle)
        x, y = _offset_lonlat(lat, lon, east, north)
        ring.append([round(x, 7), round(y, 7)])
    height = ground_ellipsoid_m if ground_ellipsoid_m is not None else 0.0
    bookmark_lon, bookmark_lat = _offset_lonlat(lat, lon, 0.0, -max(18.0, 2.5 * height_m))
    photos_text = f"{photos} drone photographs" if photos else "drone photographs"
    psnr = (trained or {}).get("psnr")
    gsd_text = (
        f"about {gsd_m * 1000:.0f} mm per pixel of the 1600 px training frames, estimated from "
        f"how close the cameras came to the crown"
        if gsd_m
        else None
    )
    return {
        "slug": SLUG,
        "name": "Minnetonka tree (real capture, Living Survey)",
        "description": (
            "A photogrammetric reconstruction of a real tree: a Gaussian splat trained on "
            f"{photos_text} of one mature deciduous tree in Minnetonka, Minnesota, flown by "
            "Matthew Guertin on 20 July 2020 in still air (Single Tree -- High-Density "
            "Photogrammetry Dataset, CC BY 4.0). Its skeleton was extracted from the splat's "
            "geometry alone. The motion is simulated: nothing here recorded this tree moving; "
            "the sway is the Living Survey's wind model driving that skeleton."
        ),
        "boundary": ring,
        "center": [round(lon, 7), round(lat, 7), round(height, 2)],
        "attribution": ATTRIBUTION,
        "attribution_url": ATTRIBUTION_URL,
        "license_name": LICENSE_NAME,
        "license_url": LICENSE_URL,
        "source_url": SOURCE_URL,
        "captured": CAPTURED,
        "pipeline": (
            "tools/pipeline pose (COLMAP) + train (gsplat MCMC) on Modal via "
            "infra/modal/minnetonka.py; tools/captures/real_tree.py + skeleton.py"
            + (f"; held-out PSNR {psnr:.2f} dB" if isinstance(psnr, int | float) else "")
        ),
        "images": photos,
        "assets": [
            {
                "representation": "gaussian-splat",
                "name": "Gaussian splat (photogrammetric)",
                "path": "splat/tileset.json",
                "ground_sample_distance_m": round(gsd_m, 4) if gsd_m else None,
                "point_spacing_m": round(spacing_m, 4),
                "description": (
                    f"{splats:,} gaussians of one tree, a single tile, rigged with {nodes} "
                    "skeleton nodes" + (f"; {gsd_text}" if gsd_text else "")
                ),
                "maximum_screen_space_error": 1,
                "clamp_to_ground": True,
                "default": True,
                "rig": "../source/rig.json",
            }
        ],
        "bookmark": {
            "name": "The tree",
            "longitude": round(bookmark_lon, 7),
            "latitude": round(bookmark_lat, 7),
            "height": round(height + max(4.0, 0.6 * height_m), 2),
            "heading": 0.0,
            "pitch": -8.0,
            "is_default": True,
        },
    }


def ground_sample_distance(
    cameras: np.ndarray | None, tree_xyz: np.ndarray, focal_px: float | None
) -> float | None:
    """Median over cameras of (distance to the nearest splat of the tree) / focal: the
    size of a training pixel on the part of the tree each camera saw closest."""
    if cameras is None or not len(cameras) or not focal_px:
        return None
    from scipy.spatial import cKDTree

    distance, _ = cKDTree(tree_xyz).query(cameras, k=1)
    return float(np.median(distance) / focal_px)


def build(
    ply: Path,
    out_dir: Path,
    frame: dict[str, Any],
    *,
    train: dict[str, Any] | None = None,
    pose: dict[str, Any] | None = None,
    metres_per_unit: float | None = None,
    trunk_diameter_m: float | None = None,
    crown_radius_m: float | None = None,
    ground_clearance_m: float = 0.2,
    max_splats: int = DEFAULT_MAX_SPLATS,
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
    tile: bool = True,
) -> dict[str, Any]:
    """Everything in the module docstring, in order; returns (and writes) the report."""
    if metres_per_unit is not None and trunk_diameter_m is not None:
        raise ValueError("give --metres-per-unit or --trunk-diameter-m, not both")
    matrix = np.asarray(frame["matrix"], dtype=np.float64)
    fitted_scale = float(frame["metresPerUnit"])
    report: dict[str, Any] = {
        "input": str(ply),
        "gate": frame.get("verdict"),
        "scale": {"barometricMetresPerUnit": fitted_scale, "method": "barometric"},
    }
    if metres_per_unit is not None:
        # About the frame's origin: the similarity's own scale replaced.
        factor = metres_per_unit / fitted_scale
        matrix = np.diag([factor, factor, factor, 1.0]) @ matrix
        report["scale"].update(method="explicit", metresPerUnit=metres_per_unit)
    cameras = np.asarray(frame["camerasM"], dtype=np.float64) if frame.get("camerasM") else None
    if cameras is not None and metres_per_unit is not None:
        cameras = cameras * (metres_per_unit / fitted_scale)

    data = transform_splats(splat_tiles.read_ply(ply), matrix)
    report["trained"] = len(data["x"])
    xyz = positions(data)
    opacity = splat_tiles.sigmoid(data["opacity"].astype(np.float64))
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity)
    data, xyz, opacity = take(data, finite), xyz[finite], opacity[finite]

    report["profileAboutFrameAxis"] = profile(xyz, opacity, cameras)
    try:
        lowest_camera = float(cameras[:, 2].min()) if cameras is not None else None
        ground = ground_height(xyz, opacity, TRUNK_SEARCH_M * 2, below=lowest_camera)
        trunk = measure_trunk(xyz, opacity, ground)
        report["groundM"] = round(ground, 4)
        report["trunk"] = trunk
        if trunk_diameter_m is not None:
            factor = trunk_diameter_m / trunk["diameterM"]
            report["scale"].update(
                method="trunk",
                trunkDiameterM=trunk_diameter_m,
                metresPerUnit=fitted_scale * factor,
                factorOnBarometric=round(factor, 4),
            )
        else:
            factor = 1.0
        report["scale"].setdefault("metresPerUnit", fitted_scale)
        # Origin at the trunk's foot, then any trunk-derived rescale about it.
        shift = np.array([trunk["centre"][0], trunk["centre"][1], ground])
        about = np.eye(4)
        about[:3, :3] *= factor
        about[:3, 3] = -factor * shift
        data = transform_splats(data, about)
        xyz = positions(data)
        if cameras is not None:
            cameras = (cameras - shift) * factor
        report["trunk"]["diameterScaledM"] = round(trunk["diameterM"] * factor, 4)

        crown = (
            {"radiusM": crown_radius_m, "method": "given"}
            if crown_radius_m is not None
            else {**crown_radius(xyz, opacity, cameras), "method": "splats inside the camera ring"}
        )
        report["crown"] = crown
    except ValueError as error:
        # What the measurements saw, so a failed run can be read without the PLY.
        if "trunk" in report:
            report["profileAboutTrunk"] = profile(xyz, opacity, cameras)
        error.add_note("partial report: " + json.dumps(report, indent=1))
        raise
    radius = float(crown["radiusM"])
    region = (np.hypot(xyz[:, 0], xyz[:, 1]) <= radius) & (xyz[:, 2] >= ground_clearance_m)
    region &= opacity >= 0.02
    index = np.flatnonzero(region)
    dense = skeleton.dense_mask(xyz[index], 8, 3.0)
    region[index[~dense]] = False
    index = np.flatnonzero(region)
    settled = floater_fixpoint(xyz[index])
    region[index[~settled]] = False
    tree_data = take(data, region)
    capped = cap_splats(tree_data, max_splats)
    tree_data = take(tree_data, capped)
    report["isolation"] = {
        "inRegion": int(region.sum()),
        "keptAfterCap": int(capped.sum()),
        "maxSplats": max_splats,
        "groundClearanceM": ground_clearance_m,
    }
    tree_xyz = positions(tree_data)
    upright = uprightness(tree_xyz)
    report["uprightness"] = upright
    if not upright["upright"]:
        raise SystemExit(
            f"the isolated tree does not stand up (crown/base {upright['crownToBase']}, the "
            "deformer wants >= 3): the ground cut or the crown radius is wrong -- see "
            "real_tree.json"
        )

    source = out_dir / "source"
    source.mkdir(parents=True, exist_ok=True)
    isolated = source / "isolated.ply"
    write_ply(
        isolated,
        {
            "position": tree_xyz.astype(np.float32),
            "rgb": np.stack([tree_data[f"f_dc_{i}"] for i in range(3)], axis=1) * splat_tiles.SH_C0
            + 0.5,
            "opacity_logit": tree_data["opacity"],
            "log_scale": np.stack([tree_data[f"scale_{i}"] for i in range(3)], axis=1),
            "quat_wxyz": np.stack([tree_data[f"rot_{i}"] for i in range(4)], axis=1),
        },
    )
    # pose.json (COLMAP's run) says how many registered; prepare.json how many were fetched.
    photos = (pose or {}).get("registered") or (pose or {}).get("photos")
    takeoff_msl = frame.get("takeoffMslM")
    # frame.json's z = 0 is the take-off; the splat's ground is `ground` above it.
    ground_ellipsoid = (
        None if takeoff_msl is None else float(takeoff_msl) + ground + GEOID_UNDULATION_M
    )
    rig_report = skeleton.extract(
        isolated,
        out_dir,
        lat=lat,
        lon=lon,
        height=ground_ellipsoid or 0.0,
        # Isolation is done above, so extract keeps every splat it is given: no region,
        # no ground cut past the lowest splat, no second denoise.
        ground_percentile=0.0,
        denoise_k=0,
        source_note=(
            f"geometric skeleton extraction from a gsplat reconstruction of the Minnetonka "
            f"tree ({SOURCE_URL}, CC BY 4.0): {{nodes}} nodes over {len(tree_xyz):,} splats "
            "(tools/captures/real_tree.py, skeleton.py)"
        ),
        tile=tile,
    )
    rig_path = source / "rig.json"
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    if "{nodes}" in rig.get("sourceNote", ""):
        rig["sourceNote"] = rig["sourceNote"].replace("{nodes}", str(len(rig["nodes"])))
        rig_path.write_text(json.dumps(rig, separators=(",", ":")) + "\n", encoding="utf-8")
    isolated.unlink()
    report["rig"] = rig_report
    gsd = ground_sample_distance(cameras, tree_xyz, frame.get("focalPx"))
    report["gsdM"] = gsd
    report["placement"] = {
        "lat": lat,
        "lon": lon,
        "groundEllipsoidM": ground_ellipsoid,
        "geoidUndulationM": GEOID_UNDULATION_M,
        "clampToGround": True,
    }
    document = site_document(
        lat=lat,
        lon=lon,
        ground_ellipsoid_m=ground_ellipsoid,
        crown_m=radius,
        height_m=float(rig_report["height_m"]),
        splats=int(rig_report["splats"]),
        photos=photos,
        gsd_m=gsd,
        spacing_m=float(rig_report["spacing_m"]),
        nodes=int(rig_report["nodes"]),
        trained=train,
    )
    write_site_document(out_dir, document)
    (source / "real_tree.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("ply", type=Path, help="the trainer's PLY, in the pose set's frame")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--frame", type=Path, required=True, help="the pose gate's frame.json")
    parser.add_argument("--train", type=Path, help="train.json, for the site's pipeline line")
    parser.add_argument("--pose", type=Path, help="pose.json or prepare.json: the photo count")
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument("--metres-per-unit", type=float, help="replace the barometric scale")
    scale.add_argument("--trunk-diameter-m", type=float, help="scale to a measured trunk")
    parser.add_argument("--crown-radius-m", type=float)
    parser.add_argument("--ground-clearance-m", type=float, default=0.2)
    parser.add_argument("--max-splats", type=int, default=DEFAULT_MAX_SPLATS)
    parser.add_argument("--lat", type=float, default=LATITUDE)
    parser.add_argument("--lon", type=float, default=LONGITUDE)
    parser.add_argument("--allow-failed-gate", action="store_true")
    args = parser.parse_args()

    frame = json.loads(args.frame.read_text(encoding="utf-8"))
    if frame.get("verdict") != "pass" and not args.allow_failed_gate:
        raise SystemExit(f"{args.frame} failed the pose gate ({frame.get('failed')})")

    def optional(path: Path | None) -> dict[str, Any] | None:
        return json.loads(path.read_text(encoding="utf-8")) if path else None

    report = build(
        args.ply,
        args.out_dir,
        frame,
        train=optional(args.train),
        pose=optional(args.pose),
        metres_per_unit=args.metres_per_unit,
        trunk_diameter_m=args.trunk_diameter_m,
        crown_radius_m=args.crown_radius_m,
        ground_clearance_m=args.ground_clearance_m,
        max_splats=args.max_splats,
        lat=args.lat,
        lon=args.lon,
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
