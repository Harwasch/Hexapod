"""The quality bar: keep what the capture actually supports, and say what it did not.

A trained splat has no idea which of its gaussians were well observed. The middle of an
orbit is seen by every frame from every side; the fringes -- the floor past the table,
the wall behind it -- are seen by a handful of frames from nearly one direction, and
the trainer fills them with spiky, see-through gaussians that fit those few frames and
nothing else. This stage measures, per gaussian, how much data stands behind it and
sorts it into three tiers:

* **keep** -- seen (not occluded) in at least `keep_min_views` frames, from directions at
  least `keep_min_spread_deg` apart, at a ground sampling distance no coarser than
  `keep_max_gsd_ratio` times the region of interest's median;
* **context** -- short of that, but still seen `context_min_views` times over
  `context_min_spread_deg`, at no worse than `context_max_gsd_ratio` times the median;
* **drop** -- everything else, and anything more transparent than `min_opacity`.

`bar` decides what goes on: `strict` keeps only *keep*; `balanced` also keeps *context*,
faded to `context_fade` of its opacity; `everything` is the behaviour before this stage
existed and passes the splat through untouched. The output, `gated.ply`, has exactly
`trained.ply`'s schema, so `place` reads either.

What is measured, and how, in the order it is computed (`measure_support`):

1. **occlusion** -- a z-buffer per camera at `zbuffer_side` cells on the long side, filled
   with the depth of the nearest gaussian at least `occluder_opacity` opaque. A gaussian
   counts as seen by a camera when it is in that camera's frame and no deeper than its
   cell's surface plus a tolerance (`depth_tolerance` of the depth, plus two cells'
   footprint, which is what a surface slanted up to about 60 degrees varies by across one
   cell). Centres only: a gaussian's extent is not splatted into the buffer.
2. **views** -- how many cameras see it.
3. **angular spread** -- the widest angle between two of the directions it is seen from.
   Exact maximum pairwise angle is quadratic in views, so it is the farthest-point
   estimate instead: find the view farthest from the mean direction, then the view
   farthest from *that* one. Never more than the true spread and at least half of it;
   on real orbits it is the true value or within a few degrees of it.
4. **ground sampling distance** -- `depth / focal` in the sharpest view that sees it, in
   the reconstruction's own units per pixel at the pose model's resolution.

**GSD units.** A COLMAP model has no scale until `georeference` gives it one, which is
why this stage runs after `georeference` and reads `georef.json` when it can: an EXIF
alignment's similarity carries metres per model unit, and the GSD is then reported in
mm/px too. The *criterion* is relative either way -- a ratio to the region of interest's
median -- because an absolute millimetre threshold means different things for a
building and for a teacup. `keep_max_gsd_mm` adds an absolute ceiling when, and only
when, the scale is metric. `quality.json` says which applied.

**The region of interest** is where the cameras were pointed: the least-squares point
nearest every camera's optical axis, which on an orbit is the thing being orbited. When
the axes are nearly parallel (a walk, not an orbit) that point is ill-defined, and it
falls back to the mean of each camera's median visible depth along its axis. Its radius
is `roi_radius_factor` of the median camera distance to it. It is written to
`quality.json` in the **COLMAP frame** -- the frame of `trained.ply` -- because that is
what a refine run's `train: {roi: ...}` crops in, over the same poses.

**Measured accuracy.** Coverage says a gaussian could be right; the `holdout` artifact
(`holdout.py`), when the `train` stage wrote one, says whether it is: each gaussian's mean
error on gsplat's held-out frames, which training never saw. Where a gaussian has at least
`holdout_min_weight` pixels of held-out evidence, keep also needs that error at most
`keep_max_holdout_error_ratio` times the median of the measured coverage-keep (and at most
`keep_max_holdout_error`, when set); one that fails drops to context. A gaussian no
held-out frame saw is judged by coverage alone. `heldOut` in quality.json has the counts
-- verified keep against geometry-only keep, per-tier error -- and `keepVerifiedPct` is
`keepPct` counting verified keep only. The support mask is built from verified keep when
there is enough of it.

The stage never fails a run for being strict: a bar that would leave fewer than
`min_gaussians` falls back to the next looser one and says so (`barApplied`).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import gaussians
import holdout
import sfm
import support_mask
from artifacts import ArtifactDecl
from contracts import MetricValue, StageContext, StageOutcome
from registry import stage_impl

__all__ = [
    "BARS",
    "COVERAGE_PLY",
    "GATED_PLY",
    "MODES",
    "QUALITY_JSON",
    "Cameras",
    "Roi",
    "Support",
    "Thresholds",
    "assign_tiers",
    "estimate_roi",
    "gate",
    "measure_support",
    "read_coverage",
    "write_coverage",
]

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
I32 = npt.NDArray[np.int32]
U8 = npt.NDArray[np.uint8]
Bools = npt.NDArray[np.bool_]

GATED_PLY = ArtifactDecl(
    "gated.ply",
    content_type="application/octet-stream",
    summary="trained.ply with what the capture does not support removed or faded, by `bar`",
    stub_bytes=1024,
)
QUALITY_JSON = ArtifactDecl(
    "quality.json",
    content_type="application/json",
    summary="tier counts, the region of interest (COLMAP frame), held-out PSNR, capture tips",
)
COVERAGE_PLY = ArtifactDecl(
    "coverage.ply",
    content_type="application/octet-stream",
    summary="a sample of gaussian centres coloured by tier, plus the camera path, for viewers",
    stub_bytes=512,
)

#: What `bar` may be, loosest last. The fallback walks this order.
BARS: tuple[str, ...] = ("strict", "balanced", "everything")
#: What `mode` may be: a cheap preview that forecasts, or the full-quality refine.
MODES: tuple[str, ...] = ("preview", "refine")

TIER_DROP = 0
TIER_CONTEXT = 1
TIER_KEEP = 2
#: Not a gaussian: a camera centre, appended to `coverage.ply` so a viewer can draw the path.
TIER_CAMERA = 3
TIER_NAMES: Mapping[int, str] = {
    TIER_DROP: "drop",
    TIER_CONTEXT: "context",
    TIER_KEEP: "keep",
    TIER_CAMERA: "camera",
}
#: sRGB per tier in `coverage.ply`: green, amber, grey, and the pages' accent for cameras.
TIER_COLOURS: Mapping[int, tuple[int, int, int]] = {
    TIER_KEEP: (64, 196, 108),
    TIER_CONTEXT: (238, 170, 52),
    TIER_DROP: (132, 136, 130),
    TIER_CAMERA: (127, 216, 192),
}

_UNSEEN_GSD = np.float32(np.inf)

#: The support mask is built from held-out-verified keep only when at least this many
#: gaussians, and this share of keep, were verified; otherwise from all of keep, and
#: `heldOut.supportMaskFrom` says which.
MIN_VERIFIED_FOR_MASK = 64
VERIFIED_MASK_SHARE = 0.25


# ---------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """The tier rules. Every field is a stage parameter of the same name."""

    keep_min_views: int = 8
    keep_min_spread_deg: float = 45.0
    keep_max_gsd_ratio: float = 2.0
    context_min_views: int = 3
    context_min_spread_deg: float = 15.0
    context_max_gsd_ratio: float = 6.0
    min_opacity: float = 0.02
    #: Context is a backdrop for the subject, so it must be near it: within this many ROI
    #: radii of the ROI's centre. Without it, a scattered far fringe that a few frames
    #: happened to see from a spread of angles was kept (at half opacity) as "context" --
    #: exactly the noise the bar exists to remove.
    context_max_roi_radii: float = 2.0
    #: An absolute ceiling on the keep tier's GSD, applied only when the scale is metric.
    keep_max_gsd_mm: float | None = None
    #: Measured accuracy (`holdout.py`), applied where held-out frames saw the gaussian:
    #: keep needs its mean held-out error at most this many times the median of the
    #: measured coverage-keep gaussians...
    keep_max_holdout_error_ratio: float | None = 2.0
    #: ...and at most this, absolutely (the per-pixel `0.8 L1 + 0.2 (1 - SSIM)`, 0..1).
    keep_max_holdout_error: float | None = None
    #: Pixels of held-out evidence (summed blending weight) a gaussian needs to be judged
    #: by accuracy at all; with less it is judged by coverage alone.
    holdout_min_weight: float = 2.0

    @staticmethod
    def from_params(params: Mapping[str, Any]) -> Thresholds:
        defaults = Thresholds()
        values: dict[str, Any] = {}
        for name, default in asdict(defaults).items():
            raw = params.get(name, default)
            if raw is None:
                values[name] = None
            elif name.endswith("_views"):
                values[name] = int(raw)
            else:
                values[name] = float(raw)
        return Thresholds(**values)

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


# ---------------------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cameras:
    """Every registered frame as arrays: world-to-camera `R`, `t`, and pinhole intrinsics.

    COLMAP's convention -- `x_cam = R @ x_world + t`, looking down +z -- so a camera's
    centre is `-R^T t` and its optical axis in the world is `R[2]`. Lens distortion is
    ignored: at the z-buffer's resolution it moves a point by a fraction of a cell.
    """

    names: tuple[str, ...]
    rotations: F64
    translations: F64
    fx: F64
    fy: F64
    cx: F64
    cy: F64
    width: F64
    height: F64

    @property
    def count(self) -> int:
        return len(self.names)

    @property
    def centres(self) -> F64:
        return np.asarray(
            -np.einsum("mji,mj->mi", self.rotations, self.translations), dtype=np.float64
        )

    @property
    def axes(self) -> F64:
        return self.rotations[:, 2, :].copy()

    @staticmethod
    def from_model(model: sfm.Model) -> Cameras:
        by_id = {camera.id: camera for camera in model.cameras}
        names: list[str] = []
        rotations: list[F64] = []
        translations: list[F64] = []
        intrinsics: list[tuple[float, float, float, float, float, float]] = []
        for image in model.images:
            camera = by_id.get(image.camera_id)
            if camera is None:
                continue
            names.append(image.name)
            rotations.append(image.rotation)
            translations.append(np.asarray(image.tvec, dtype=np.float64))
            intrinsics.append(_intrinsics(camera))
        if not names:
            raise ValueError("the pose model has no registered frames to measure support from")
        k = np.asarray(intrinsics, dtype=np.float64)
        return Cameras(
            names=tuple(names),
            rotations=np.stack(rotations),
            translations=np.stack(translations),
            fx=k[:, 0],
            fy=k[:, 1],
            cx=k[:, 2],
            cy=k[:, 3],
            width=k[:, 4],
            height=k[:, 5],
        )


def _intrinsics(camera: sfm.Camera) -> tuple[float, float, float, float, float, float]:
    named = dict(zip(camera.param_names, camera.params, strict=False))
    f = float(named.get("f", named.get("fx", camera.focal_px)))
    fx = float(named.get("fx", f))
    fy = float(named.get("fy", f))
    cx = float(named.get("cx", camera.width / 2))
    cy = float(named.get("cy", camera.height / 2))
    return fx, fy, cx, cy, float(camera.width), float(camera.height)


# ---------------------------------------------------------------------------------------
# Support: occlusion-aware views, angular spread, GSD
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Support:
    """Per gaussian: how many cameras see it, over what angle, at what sampling distance."""

    views: I32
    spread_deg: F32
    #: Model units per pixel in the sharpest view that sees it; +inf where none does.
    gsd: F32
    #: Per camera: the median depth of the opaque gaussians in its frame, or NaN.
    camera_depth: F64


def measure_support(
    xyz: F32,
    alpha: F32,
    cameras: Cameras,
    *,
    zbuffer_side: int = 160,
    occluder_opacity: float = 0.5,
    depth_tolerance: float = 0.05,
    footprint: F32 | None = None,
    max_footprint_cells: int = 3,
    chunk: int = 1 << 18,
) -> Support:
    """Views, spread and GSD for every gaussian, in bounded memory.

    `footprint` is each gaussian's radius in world units (its largest standard
    deviation, for a splat). An occluder covers every z-buffer cell within that radius
    of its centre, up to `max_footprint_cells` cells: a surface made of a few large
    gaussians must occlude like the surface it renders as, not like a sieve of centres.

    Memory is `cameras x chunk` booleans plus a few `chunk`-long columns at a time, so a
    1.5-million-gaussian splat over 300 frames costs what 260k over 300 does. Positions
    are handled as three contiguous columns rather than an (n, 3) array: the per-camera
    projection is then a dozen streaming float32 operations, which measured 5x faster
    than a matmul and a strided gather here.
    """
    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    count = int(points.shape[0])
    finite = np.isfinite(points).all(axis=1)
    columns = _columns(np.where(finite[:, None], points, np.float32(0.0)))
    opaque = finite & (np.nan_to_num(alpha, nan=0.0) >= occluder_opacity)
    centres = cameras.centres
    scale = float(np.median(np.linalg.norm(centres - centres.mean(axis=0), axis=1))) or 1.0
    near = 1e-3 * scale

    radius = (
        np.zeros(count, dtype=np.float32)
        if footprint is None
        else np.nan_to_num(np.asarray(footprint, dtype=np.float32), nan=0.0, posinf=0.0)
    )
    buffers, depths = _zbuffers(
        columns, radius, opaque, cameras, zbuffer_side, near, max_footprint_cells, chunk
    )

    views = np.zeros(count, dtype=np.int32)
    spread = np.zeros(count, dtype=np.float32)
    gsd = np.full(count, _UNSEEN_GSD, dtype=np.float32)
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        block = _slice(columns, start, stop)
        size = stop - start
        seen = np.zeros((cameras.count, size), dtype=np.bool_)
        direction_sum = np.zeros((3, size), dtype=np.float32)
        block_gsd = np.full(size, _UNSEEN_GSD, dtype=np.float32)
        for j in range(cameras.count):
            index, depth = _visible(
                block, cameras, j, buffers[j], zbuffer_side, near, depth_tolerance
            )
            if index.size == 0:
                continue
            seen[j, index] = True
            toward = _unit_towards(centres[j], _take(block, index))
            for axis in range(3):
                direction_sum[axis, index] += toward[axis]
            focal = float(min(cameras.fx[j], cameras.fy[j]))
            # `index` is unique within one camera, so a plain gather-min-scatter is exact.
            block_gsd[index] = np.minimum(block_gsd[index], depth / np.float32(focal))
        block_views = seen.sum(axis=0).astype(np.int32)
        block_views[~finite[start:stop]] = 0
        views[start:stop] = block_views
        gsd[start:stop] = np.where(block_views > 0, block_gsd, _UNSEEN_GSD)
        spread[start:stop] = _spread(block, centres, seen, direction_sum, block_views)
    return Support(views=views, spread_deg=spread, gsd=gsd, camera_depth=depths)


Columns = tuple[F32, F32, F32]


def _columns(points: npt.ArrayLike) -> Columns:
    array = np.asarray(points, dtype=np.float32)
    return (
        np.ascontiguousarray(array[:, 0]),
        np.ascontiguousarray(array[:, 1]),
        np.ascontiguousarray(array[:, 2]),
    )


def _slice(columns: Columns, start: int, stop: int) -> Columns:
    return columns[0][start:stop], columns[1][start:stop], columns[2][start:stop]


def _take(columns: Columns, index: npt.NDArray[np.intp] | Bools) -> Columns:
    return columns[0][index], columns[1][index], columns[2][index]


def _grid(cameras: Cameras, j: int, side: int) -> tuple[int, int]:
    w, h = float(cameras.width[j]), float(cameras.height[j])
    long_side = max(w, h, 1.0)
    return max(1, round(side * w / long_side)), max(1, round(side * h / long_side))


def _project(
    block: Columns, cameras: Cameras, j: int, near: float
) -> tuple[npt.NDArray[np.intp], F32, F32, F32]:
    """Indices in `block` that land inside camera `j`'s frame, with their u, v and depth."""
    r = cameras.rotations[j].astype(np.float32)
    t = cameras.translations[j].astype(np.float32)
    x, y, z = block
    depth = r[2, 0] * x + r[2, 1] * y + r[2, 2] * z + t[2]
    front = np.flatnonzero(depth > near)
    if front.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return front, empty, empty, empty
    xf, yf, zf = x[front], y[front], z[front]
    inverse = np.float32(1.0) / depth[front]
    u = (r[0, 0] * xf + r[0, 1] * yf + r[0, 2] * zf + t[0]) * inverse
    u *= np.float32(cameras.fx[j])
    u += np.float32(cameras.cx[j])
    v = (r[1, 0] * xf + r[1, 1] * yf + r[1, 2] * zf + t[1]) * inverse
    v *= np.float32(cameras.fy[j])
    v += np.float32(cameras.cy[j])
    inside = np.flatnonzero(
        (u >= 0)
        & (u < np.float32(cameras.width[j]))
        & (v >= 0)
        & (v < np.float32(cameras.height[j]))
    )
    return front[inside], u[inside], v[inside], depth[front[inside]]


def _rows_cols(
    u: F32, v: F32, cameras: Cameras, j: int, side: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    gw, gh = _grid(cameras, j, side)
    col = np.minimum((u * np.float32(gw / cameras.width[j])).astype(np.intp), gw - 1)
    row = np.minimum((v * np.float32(gh / cameras.height[j])).astype(np.intp), gh - 1)
    return row, col


def _cells(u: F32, v: F32, cameras: Cameras, j: int, side: int) -> npt.NDArray[np.intp]:
    row, col = _rows_cols(u, v, cameras, j, side)
    return row * _grid(cameras, j, side)[0] + col


def _zbuffers(
    columns: Columns,
    radius: F32,
    opaque: Bools,
    cameras: Cameras,
    side: int,
    near: float,
    max_cells: int,
    chunk: int,
) -> tuple[list[F32], F64]:
    """The nearest opaque depth per cell, per camera; and each camera's median depth."""
    buffers: list[F32] = []
    depths = np.full(cameras.count, np.nan, dtype=np.float64)
    occluders = _take(columns, opaque)
    occluder_radius = radius[opaque]
    total = int(occluders[0].shape[0])
    for j in range(cameras.count):
        gw, gh = _grid(cameras, j, side)
        buffer = np.full(gw * gh, np.inf, dtype=np.float32)
        cell_world = np.float32(
            float(cameras.width[j]) / gw / float(min(cameras.fx[j], cameras.fy[j]))
        )
        samples: list[F32] = []
        for start in range(0, total, chunk):
            index, u, v, z = _project(_slice(occluders, start, start + chunk), cameras, j, near)
            if z.size == 0:
                continue
            row, col = _rows_cols(u, v, cameras, j, side)
            np.minimum.at(buffer, row * gw + col, z)
            # The footprint, in cells: world radius over what one cell spans at that depth.
            reach = np.minimum(
                (occluder_radius[start + index] / (z * cell_world)).astype(np.intp), max_cells
            )
            for ring_ in range(1, max_cells + 1):
                wide = np.flatnonzero(reach >= ring_)
                if wide.size == 0:
                    break
                r0, c0, zw = row[wide], col[wide], z[wide]
                for dr in range(-ring_, ring_ + 1):
                    for dc in range(-ring_, ring_ + 1):
                        if max(abs(dr), abs(dc)) != ring_:
                            continue
                        rr, cc = r0 + dr, c0 + dc
                        ok = (rr >= 0) & (rr < gh) & (cc >= 0) & (cc < gw)
                        np.minimum.at(buffer, rr[ok] * gw + cc[ok], zw[ok])
            samples.append(z[:: max(1, z.size // 20_000)])
        if samples:
            depths[j] = float(np.median(np.concatenate(samples)))
        buffers.append(buffer)
    return buffers, depths


def _visible(
    block: Columns,
    cameras: Cameras,
    j: int,
    buffer: F32,
    side: int,
    near: float,
    tolerance: float,
) -> tuple[npt.NDArray[np.intp], F32]:
    index, u, v, z = _project(block, cameras, j, near)
    if index.size == 0:
        return index, z
    surface = buffer[_cells(u, v, cameras, j, side)]
    gw, _ = _grid(cameras, j, side)
    cell_px = float(cameras.width[j]) / gw
    focal = float(min(cameras.fx[j], cameras.fy[j])) or 1.0
    # A cell with no opaque gaussian in it is +inf, which admits everything behind it.
    limit = surface * np.float32(1.0 + tolerance + 2.0 * cell_px / focal)
    keep = z <= limit
    return index[keep], z[keep]


def _unit_towards(centre: F64, block: Columns) -> Columns:
    """Unit vectors from each point toward `centre`, in float32 throughout."""
    dx: F32 = np.float32(centre[0]) - block[0]
    dy: F32 = np.float32(centre[1]) - block[1]
    dz: F32 = np.float32(centre[2]) - block[2]
    length: F32 = np.sqrt(dx * dx + dy * dy + dz * dz)
    inverse: F32 = np.reciprocal(np.maximum(length, np.float32(1e-12)))
    return dx * inverse, dy * inverse, dz * inverse


def _spread(block: Columns, centres: F64, seen: Bools, direction_sum: F32, views: I32) -> F32:
    """The farthest-point estimate of each gaussian's widest pair of viewing directions.

    Only the (gaussian, camera) pairs that see each other are touched, so the cost is
    the number of observations, not gaussians x cameras.
    """
    count = block[0].shape[0]
    length = np.maximum(np.sqrt((direction_sum * direction_sum).sum(axis=0)), np.float32(1e-12))
    mean = direction_sum / length
    farthest = np.zeros(count, dtype=np.intp)
    lowest = np.full(count, 2.0, dtype=np.float32)
    for j in range(centres.shape[0]):
        index = np.flatnonzero(seen[j])
        if index.size == 0:
            continue
        dx, dy, dz = _unit_towards(centres[j], _take(block, index))
        dots = dx * mean[0, index] + dy * mean[1, index] + dz * mean[2, index]
        better = dots < lowest[index]
        lowest[index[better]] = dots[better]
        farthest[index[better]] = j
    far = centres[farthest].astype(np.float32)
    # Toward the farthest camera: `_unit_towards` of the origin from (point - camera).
    anchor = _unit_towards(
        np.zeros(3), (block[0] - far[:, 0], block[1] - far[:, 1], block[2] - far[:, 2])
    )
    widest = np.full(count, 1.0, dtype=np.float32)
    for j in range(centres.shape[0]):
        index = np.flatnonzero(seen[j])
        if index.size == 0:
            continue
        dx, dy, dz = _unit_towards(centres[j], _take(block, index))
        dots = dx * anchor[0][index] + dy * anchor[1][index] + dz * anchor[2][index]
        widest[index] = np.minimum(widest[index], dots)
    degrees = np.degrees(np.arccos(np.clip(widest, -1.0, 1.0))).astype(np.float32)
    degrees[views < 2] = 0.0
    return degrees


# ---------------------------------------------------------------------------------------
# The region of interest
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Roi:
    """A sphere in the COLMAP frame: where the capture was pointed, and how big it is."""

    centre: F64
    radius: float
    method: str
    camera_distance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "center": [round(float(v), 6) for v in self.centre],
            "radius": round(self.radius, 6),
            "frame": "colmap",
            "method": self.method,
            "medianCameraDistance": round(self.camera_distance, 6),
        }

    def contains(self, xyz: F32) -> Bools:
        offset = np.asarray(xyz, dtype=np.float64) - self.centre
        inside: Bools = np.einsum("ij,ij->i", offset, offset) <= self.radius**2
        return inside


def supported_extent(xyz: F32, tiers: U8, fallback: Roi, quantile: float = 99.0) -> Roi:
    """The sphere that holds the well-supported part of the scene: the keep tier's extent.

    This, not where the cameras pointed, is what a Refine trains inside. It is decided by
    the data alone -- a wall that was seen well enough is inside it wherever it stands --
    and needs nobody to say what the subject was. The centre is the keep tier's median
    (robust to a stray keep splat); the radius its `quantile` distance from there. With no
    keep splats at all, the camera-pointed region stands in, and `method` says so.
    """
    kept = np.asarray(xyz, dtype=np.float64)[tiers == TIER_KEEP]
    kept = kept[np.isfinite(kept).all(axis=1)]
    if kept.shape[0] < 32:
        return replace(fallback, method=f"{fallback.method} (too little keep for an extent)")
    centre = np.median(kept, axis=0)
    radius = float(np.percentile(np.linalg.norm(kept - centre, axis=1), quantile))
    return Roi(
        centre=centre,
        radius=max(radius, 1e-6),
        method="keep-extent",
        camera_distance=fallback.camera_distance,
    )


def estimate_roi(
    centres: F64,
    axes: F64,
    camera_depth: F64 | None = None,
    *,
    radius_factor: float = 0.5,
    max_condition: float = 1e4,
) -> Roi:
    """The point nearest every optical axis, in least squares, and a radius around it.

    Minimises the sum of squared distances from `x` to each line `C_j + s a_j`, whose
    normal equations are `sum(I - a a^T) x = sum(I - a a^T) C`. It is refused -- and the
    median-depth fallback used -- when that system is ill-conditioned (nearly parallel
    axes: a walk rather than an orbit) or when the point it gives is behind most of the
    cameras (axes diverging: a panorama from one spot).
    """
    a = axes / np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-12)
    projector = np.eye(3)[None, :, :] - a[:, :, None] * a[:, None, :]
    lhs = projector.sum(axis=0)
    rhs = np.einsum("mij,mj->i", projector, centres)
    centre: F64 | None = None
    method = "optical-axes"
    if np.linalg.cond(lhs) < max_condition:
        candidate = np.linalg.solve(lhs, rhs)
        ahead = np.einsum("mi,mi->m", candidate[None, :] - centres, a) > 0
        if ahead.mean() >= 0.5:
            centre = candidate
    if centre is None:
        method = "median-depth"
        depth = np.asarray(camera_depth if camera_depth is not None else [], dtype=np.float64)
        usable = np.isfinite(depth) if depth.size == len(centres) else np.zeros(0, dtype=bool)
        if usable.any():
            centre = (centres[usable] + a[usable] * depth[usable, None]).mean(axis=0)
        else:
            method = "camera-centroid"
            centre = centres.mean(axis=0)
    distances = np.linalg.norm(centres - centre, axis=1)
    median = float(np.median(distances)) if distances.size else 0.0
    if median <= 0.0:
        median = 1.0
    return Roi(
        centre=np.asarray(centre, dtype=np.float64),
        radius=radius_factor * median,
        method=method,
        camera_distance=median,
    )


# ---------------------------------------------------------------------------------------
# Tiers and the gate
# ---------------------------------------------------------------------------------------


def assign_tiers(
    views: I32,
    spread_deg: F32,
    gsd_ratio: F32,
    alpha: F32,
    thresholds: Thresholds,
    gsd_mm: F32 | None = None,
    roi_radii: F32 | None = None,
) -> U8:
    """keep / context / drop per gaussian. Pure, so the rules are testable on their own.

    `roi_radii` is each gaussian's distance from the ROI's centre in ROI radii; given, it
    limits *context* to `context_max_roi_radii`. Keep is not limited: a well-supported
    splat is kept wherever it is.
    """
    opaque_enough = np.nan_to_num(alpha, nan=0.0) >= thresholds.min_opacity
    keep = (
        (views >= thresholds.keep_min_views)
        & (spread_deg >= thresholds.keep_min_spread_deg)
        & (gsd_ratio <= thresholds.keep_max_gsd_ratio)
        & opaque_enough
    )
    if gsd_mm is not None and thresholds.keep_max_gsd_mm is not None:
        keep &= gsd_mm <= thresholds.keep_max_gsd_mm
    context = (
        ~keep
        & (views >= thresholds.context_min_views)
        & (spread_deg >= thresholds.context_min_spread_deg)
        & (gsd_ratio <= thresholds.context_max_gsd_ratio)
        & opaque_enough
    )
    if roi_radii is not None:
        context &= roi_radii <= thresholds.context_max_roi_radii
    tiers = np.full(views.shape[0], TIER_DROP, dtype=np.uint8)
    tiers[context] = TIER_CONTEXT
    tiers[keep] = TIER_KEEP
    return tiers


def gate(
    columns: Mapping[str, F32], tiers: U8, bar: str, *, context_fade: float = 0.5
) -> dict[str, F32]:
    """The splat the bar lets through, with the same columns it came in with."""
    if bar not in BARS:
        raise ValueError(f"bar must be one of {', '.join(BARS)}, not {bar!r}")
    if bar == "everything":
        return {
            name: np.array(values, dtype=np.float32, copy=True) for name, values in columns.items()
        }
    floor = TIER_KEEP if bar == "strict" else TIER_CONTEXT
    selected = tiers >= floor
    out = {
        name: np.ascontiguousarray(values[selected], dtype=np.float32)
        for name, values in columns.items()
    }
    if bar == "balanced" and "opacity" in out:
        faded = tiers[selected] == TIER_CONTEXT
        if faded.any():
            alpha = 1.0 / (1.0 + np.exp(-out["opacity"][faded].astype(np.float64)))
            alpha = np.clip(alpha * context_fade, gaussians.ALPHA_EPS, 1.0 - gaussians.ALPHA_EPS)
            out["opacity"][faded] = np.log(alpha / (1.0 - alpha)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------------------
# coverage.ply
# ---------------------------------------------------------------------------------------

_COVERAGE_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("tier", "u1"),
    ]
)


def write_coverage(path: Path, xyz: npt.ArrayLike, tiers: npt.ArrayLike) -> int:
    """A point cloud any PLY viewer opens: centres, an sRGB colour per tier, and the tier.

    Binary little-endian with a fixed header, so a browser can read it with a DataView
    and no PLY library.
    """
    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    labels = np.asarray(tiers, dtype=np.uint8).reshape(-1)
    record = np.empty(points.shape[0], dtype=_COVERAGE_DTYPE)
    record["x"], record["y"], record["z"] = points[:, 0], points[:, 1], points[:, 2]
    palette = np.zeros((256, 3), dtype=np.uint8)
    for tier, colour in TIER_COLOURS.items():
        palette[tier] = colour
    record["red"], record["green"], record["blue"] = (palette[labels, i] for i in range(3))
    record["tier"] = labels
    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment tier: 0 drop, 1 context, 2 keep, 3 camera (in capture order)",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property uchar tier",
        "end_header",
    ]
    payload = ("\n".join(header) + "\n").encode("ascii") + record.tobytes()
    path.write_bytes(payload)
    return len(payload)


def read_coverage(path: Path) -> tuple[F32, U8]:
    """The inverse of `write_coverage`: (n, 3) positions and n tiers."""
    raw = path.read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        raise ValueError(f"{path.name} is not a PLY file: no end_header")
    header = raw[:end].decode("ascii", errors="replace").splitlines()
    count = next(
        (int(line.split()[2]) for line in header if line.startswith("element vertex")), None
    )
    if count is None:
        raise ValueError(f"{path.name} declares no vertex element")
    record = np.frombuffer(raw, dtype=_COVERAGE_DTYPE, count=count, offset=end + len(marker))
    xyz = np.stack([record["x"], record["y"], record["z"]], axis=1).astype(np.float32)
    return xyz, record["tier"].astype(np.uint8)


def _coverage_sample(xyz: F32, alpha: F32, limit: int, seed: int = 0) -> npt.NDArray[np.intp]:
    """Up to `limit` indices of visible gaussians, the same ones on every run."""
    candidates = np.flatnonzero(np.isfinite(xyz).all(axis=1) & (alpha >= 0.05))
    if candidates.size <= limit:
        return candidates
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, size=limit, replace=False))


# ---------------------------------------------------------------------------------------
# What the capture could have done better
# ---------------------------------------------------------------------------------------


def capture_tips(
    *,
    centres: F64,
    roi: Roi,
    up: F64 | None,
    roi_views: I32,
    roi_spread: F32,
    roi_gsd_ratio: F32,
    roi_tiers: U8,
    thresholds: Thresholds,
    keep_pct: float | None,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Advice read off this capture's own numbers, most useful first; and those numbers."""
    tips: list[dict[str, str]] = []
    geometry: dict[str, object] = {"cameras": int(centres.shape[0])}
    offsets = centres - roi.centre
    lengths = np.maximum(np.linalg.norm(offsets, axis=1), 1e-12)
    if up is not None and np.linalg.norm(up) > 0:
        axis = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
        elevation = np.degrees(np.arcsin(np.clip(offsets @ axis / lengths, -1.0, 1.0)))
        bins = {
            "below": int((elevation < -10).sum()),
            "level": int(((elevation >= -10) & (elevation < 20)).sum()),
            "raised": int(((elevation >= 20) & (elevation < 45)).sum()),
            "above": int((elevation >= 45).sum()),
        }
        geometry["elevationDeg"] = {
            "min": round(float(elevation.min()), 1),
            "max": round(float(elevation.max()), 1),
            "bins": bins,
        }
        if bins["above"] == 0 and bins["raised"] < 0.2 * len(elevation):
            tips.append(
                {
                    "id": "from-above",
                    "text": (
                        f"Add frames from above: the highest one looked down at only "
                        f"{max(0.0, float(elevation.max())):.0f}°. A second loop, higher, "
                        f"pointing down at about 45°, fills in the tops of things."
                    ),
                }
            )
        if bins["level"] + bins["below"] == 0:
            tips.append(
                {
                    "id": "from-level",
                    "text": (
                        "Add a lower loop, at the subject's own height: every frame looked "
                        "down on it, so its sides are thin."
                    ),
                }
            )
        # Azimuth about the up axis, in eight sectors.
        east = np.cross(axis, [1.0, 0.0, 0.0])
        if np.linalg.norm(east) < 1e-6:
            east = np.cross(axis, [0.0, 1.0, 0.0])
        east /= np.linalg.norm(east)
        north = np.cross(axis, east)
        azimuth = np.degrees(np.arctan2(offsets @ north, offsets @ east)) % 360.0
        # The uncovered arc is the widest gap between neighbouring camera directions.
        # (Counting occupied 45-degree sectors undercounted it: a half-circle walk touches
        # six of eight sectors at their edges and read as "90 degrees missing".)
        ordered = np.sort(azimuth)
        gaps = np.diff(np.concatenate([ordered, ordered[:1] + 360.0]))
        gap = float(gaps.max()) if gaps.size else 360.0
        missing_deg = int(round(gap / 15.0) * 15)
        geometry["azimuthGapDeg"] = round(gap, 1)
        if missing_deg >= 60:
            tips.append(
                {
                    "id": "all-around",
                    "text": (
                        f"Walk all the way around: about {missing_deg}° of the circle around "
                        f"it has no frames, so that side is guessed."
                    ),
                }
            )
    if roi_views.size:
        short = roi_tiers != TIER_KEEP
        failing = {
            "views": float((short & (roi_views < thresholds.keep_min_views)).mean()),
            "spread": float((short & (roi_spread < thresholds.keep_min_spread_deg)).mean()),
            "gsd": float((short & (roi_gsd_ratio > thresholds.keep_max_gsd_ratio)).mean()),
        }
        geometry["shortOfKeepBy"] = {key: round(value, 3) for key, value in failing.items()}
        median_views = int(np.median(roi_views))
        if failing["views"] >= 0.15:
            tips.append(
                {
                    "id": "more-frames",
                    "text": (
                        f"Move more slowly or take more photos: a typical point near the "
                        f"subject was in {median_views} frames, and {thresholds.keep_min_views} "
                        f"is what high quality needs."
                    ),
                }
            )
        if failing["spread"] >= 0.15:
            tips.append(
                {
                    "id": "more-angles",
                    "text": (
                        "Move around it, not only toward it: much of it was seen from too "
                        f"narrow a range of angles (under {thresholds.keep_min_spread_deg:.0f}°)."
                    ),
                }
            )
        if failing["gsd"] >= 0.15:
            tips.append(
                {
                    "id": "get-closer",
                    "text": (
                        "Get closer to the parts you care about: some were only ever seen "
                        "from far away, so they have little detail."
                    ),
                }
            )
    if not tips and keep_pct is not None and keep_pct >= 80:
        tips.append({"id": "good", "text": "Well covered. Nothing to add."})
    return tips[:4], geometry


# ---------------------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------------------


def _occupied_share(
    xyz: F32, tiers: U8, alpha: F32, roi: Roi, floor: int, cells: int = 24
) -> float | None:
    """The share of occupied voxels in the ROI whose gaussians mostly reach `floor`.

    "Volume that is keep" measured over the volume that has anything in it: an orbit's
    ROI sphere is mostly air, and a percentage of air would say nothing.
    """
    inside = roi.contains(xyz) & (np.nan_to_num(alpha, nan=0.0) >= 0.1)
    if not inside.any():
        return None
    size = 2.0 * roi.radius / cells
    ijk = np.floor((xyz[inside].astype(np.float64) - (roi.centre - roi.radius)) / size)
    ijk = np.clip(ijk, 0, cells - 1).astype(np.int64)
    voxel = (ijk[:, 0] * cells + ijk[:, 1]) * cells + ijk[:, 2]
    total = np.bincount(voxel, minlength=cells**3)
    good = np.bincount(
        voxel, weights=(tiers[inside] >= floor).astype(np.float64), minlength=cells**3
    )
    occupied = total > 0
    return round(100.0 * float((good[occupied] >= 0.5 * total[occupied]).mean()), 1)


def _metres_per_unit(georef: Mapping[str, Any] | None) -> float | None:
    """Metres per model unit, when georeference measured one; None otherwise."""
    if not georef or georef.get("scaleSource") != "exif-gps":
        return None
    frame = georef.get("frame")
    scale = frame.get("scale") if isinstance(frame, Mapping) else None
    try:
        value = float(scale) if scale is not None else math.nan
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


@stage_impl(
    "support_gate",
    consumes=("trained.ply", "poses"),
    optional_consumes=("georef.json", "train_metrics.json", holdout.HOLDOUT.name),
    produces=(GATED_PLY, QUALITY_JSON, COVERAGE_PLY),
    summary="the quality bar: keep what enough frames saw well, and say what to capture next",
)
def support_gate(ctx: StageContext) -> StageOutcome:
    """Measure per-gaussian support, tier it, gate the splat by `bar`. See the module."""
    started = time.perf_counter()
    mode = str(ctx.param("mode", "refine"))
    if mode not in MODES:
        raise ValueError(f"quality.mode must be one of {', '.join(MODES)}, not {mode!r}")
    bar = str(ctx.param("bar", "balanced"))
    if bar not in BARS:
        raise ValueError(f"quality.bar must be one of {', '.join(BARS)}, not {bar!r}")
    thresholds = Thresholds.from_params(ctx.params)

    splat = gaussians.read_splat(ctx.input(GATED_PLY_SOURCE))
    model = sfm.read_model(ctx.input("poses"))
    cameras = Cameras.from_model(model)
    xyz, alpha = splat.xyz, splat.alpha
    support = measure_support(
        xyz,
        alpha,
        cameras,
        zbuffer_side=int(ctx.param("zbuffer_side", 160)),
        occluder_opacity=float(ctx.param("occluder_opacity", 0.5)),
        depth_tolerance=float(ctx.param("depth_tolerance", 0.05)),
        footprint=_footprint(splat.columns),
    )
    measured = time.perf_counter() - started
    ctx.log(
        f"support: {splat.count} gaussians x {cameras.count} cameras in {measured:.1f} s; "
        f"median views {int(np.median(support.views))}"
    )

    roi = estimate_roi(
        cameras.centres,
        cameras.axes,
        support.camera_depth,
        radius_factor=float(ctx.param("roi_radius_factor", 0.5)),
    )
    in_roi = roi.contains(xyz) & (support.views > 0)
    seen = support.views > 0
    reference = in_roi if in_roi.any() else seen
    median_gsd = float(np.median(support.gsd[reference])) if reference.any() else math.nan
    gsd_ratio = (
        (support.gsd / np.float32(median_gsd)).astype(np.float32)
        if math.isfinite(median_gsd) and median_gsd > 0
        else np.where(seen, np.float32(1.0), np.float32(np.inf)).astype(np.float32)
    )
    georef = _read_json(ctx.input("georef.json")) if ctx.has_input("georef.json") else None
    metres = _metres_per_unit(georef)
    gsd_mm = (
        None if metres is None else (support.gsd * np.float32(metres * 1000.0)).astype(np.float32)
    )
    roi_radii = (
        np.linalg.norm(xyz.astype(np.float64) - roi.centre, axis=1) / max(roi.radius, 1e-12)
    ).astype(np.float32)
    tiers = assign_tiers(
        support.views, support.spread_deg, gsd_ratio, alpha, thresholds, gsd_mm, roi_radii
    )
    # Measured accuracy, where held-out frames saw the splat: keep must also match them.
    held, held_status = (
        holdout.load(ctx.input(holdout.HOLDOUT.name), splat.count)
        if ctx.has_input(holdout.HOLDOUT.name)
        else (None, {"status": "missing", "reason": "the train stage wrote no holdout/"})
    )
    accuracy: holdout.Accuracy | None = None
    coverage_keep = tiers == TIER_KEEP
    if held is not None:
        accuracy = holdout.judge(
            tiers,
            held,
            max_ratio=thresholds.keep_max_holdout_error_ratio,
            max_abs=thresholds.keep_max_holdout_error,
            min_weight=thresholds.holdout_min_weight,
            keep=TIER_KEEP,
            demote_to=TIER_CONTEXT,
        )
        tiers = accuracy.tiers
        ctx.log(
            f"held-out: {int(accuracy.measured.sum())} of {splat.count} gaussians measured; "
            f"{int(accuracy.demoted.sum())} of {int(coverage_keep.sum())} coverage-keep over "
            f"the limit {accuracy.limit}; {int(accuracy.verified.sum())} keep verified"
        )
    else:
        ctx.log(
            f"held-out: not used ({held_status['status']}: {held_status.get('reason')}); "
            f"keep is judged by coverage alone"
        )

    min_gaussians = int(ctx.param("min_gaussians", 1000))
    applied = bar
    kept = gate(splat.columns, tiers, applied, context_fade=float(ctx.param("context_fade", 0.5)))
    while int(kept["x"].shape[0]) < min(min_gaussians, splat.count) and applied != "everything":
        applied = BARS[BARS.index(applied) + 1]
        ctx.log(f"WARNING: bar {bar!r} leaves too little; falling back to {applied!r}")
        kept = gate(
            splat.columns, tiers, applied, context_fade=float(ctx.param("context_fade", 0.5))
        )
    written = gaussians.write_ply(ctx.output(GATED_PLY.name), kept)

    limit = int(ctx.param("coverage_points", 150_000))
    sample = _coverage_sample(xyz, alpha, max(0, limit - cameras.count))
    path_order = np.argsort(np.asarray(cameras.names))
    coverage_xyz = np.concatenate([xyz[sample], cameras.centres[path_order].astype(np.float32)])
    coverage_tiers = np.concatenate(
        [tiers[sample], np.full(cameras.count, TIER_CAMERA, dtype=np.uint8)]
    )
    write_coverage(ctx.output(COVERAGE_PLY.name), coverage_xyz, coverage_tiers)

    counts = {
        name: int((tiers == tier).sum()) for tier, name in TIER_NAMES.items() if tier != TIER_CAMERA
    }
    # What a Refine trains inside, and what the percentages are over: the extent the data
    # supports, not the region the cameras pointed at (`roi`, which stays the reference
    # for pixel size and the capture tips).
    extent = supported_extent(xyz, tiers, roi)
    # The region a Refine trains in: the voxels holding the keep tier, in whatever shape
    # they make -- the keep that held-out frames verified, when there is enough of it to
    # be a region (a capture whose held-out frames saw only part of the subject would
    # otherwise refine only that part). `extent` is only the frame the percentages below
    # are measured over.
    mask_source = "keep"
    mask_from = tiers == TIER_KEEP
    if accuracy is not None and int(accuracy.verified.sum()) >= max(
        MIN_VERIFIED_FOR_MASK, VERIFIED_MASK_SHARE * int(mask_from.sum())
    ):
        mask_source, mask_from = "verified-keep", accuracy.verified
    mask = support_mask.build(xyz[mask_from])
    scene = replace(extent, radius=extent.radius * 1.5)
    keep_pct = _occupied_share(xyz, tiers, alpha, scene, TIER_KEEP)
    context_pct = _occupied_share(xyz, tiers, alpha, scene, TIER_CONTEXT)
    # The same share, counting only keep that held-out frames confirmed.
    keep_verified_pct = (
        None
        if accuracy is None
        else _occupied_share(
            xyz,
            np.where(accuracy.verified, TIER_KEEP, TIER_DROP).astype(np.uint8),
            alpha,
            scene,
            TIER_KEEP,
        )
    )
    metrics_doc = (
        _read_json(ctx.input("train_metrics.json")) if ctx.has_input("train_metrics.json") else {}
    )
    psnr = _round(_float_or_none(metrics_doc.get("psnr")), 2)
    up_estimate = sfm.camera_up(model)
    roi_index = np.flatnonzero(in_roi)
    tips, geometry = capture_tips(
        centres=cameras.centres,
        roi=roi,
        up=None if up_estimate is None else np.asarray(up_estimate.up, dtype=np.float64),
        roi_views=support.views[roi_index],
        roi_spread=support.spread_deg[roi_index],
        roi_gsd_ratio=gsd_ratio[roi_index],
        roi_tiers=tiers[roi_index],
        thresholds=thresholds,
        keep_pct=keep_pct,
    )
    held_report: dict[str, object] = dict(held_status)
    if accuracy is not None and held is not None:
        held_report = holdout.tier_report(
            accuracy,
            held,
            {tier: name for tier, name in TIER_NAMES.items() if tier != TIER_CAMERA},
            max_ratio=thresholds.keep_max_holdout_error_ratio,
            max_abs=thresholds.keep_max_holdout_error,
            min_weight=thresholds.holdout_min_weight,
        )
        near_keep = int(coverage_keep[roi_index].sum())
        demoted_share = float(accuracy.demoted[roi_index].sum()) / near_keep if near_keep else 0.0
        held_report["demotedShareNearSubject"] = round(demoted_share, 3)
        tip = holdout.accuracy_tip(held.summary, demoted_share)
        if tip is not None:
            # Accuracy is what failed, so it leads, and "well covered" is no longer true.
            tips = [tip, *(t for t in tips if t["id"] != "good")][:4]
    held_report["supportMaskFrom"] = mask_source
    roi_gsd_mm = None if metres is None else _round(median_gsd * metres * 1000.0, 2)
    document: dict[str, object] = {
        "version": 1,
        "mode": mode,
        "bar": bar,
        "barApplied": applied,
        # `roi` is what a Refine crops training to: the supported extent. `pointedRoi` is
        # where the cameras converged, kept for the record and for the tips.
        "roi": extent.to_dict(),
        "pointedRoi": roi.to_dict(),
        # What a Refine crops training to, when present: any shape, decided by the data.
        "supportMask": None if mask is None else mask.to_dict(),
        "gaussians": {"in": splat.count, "out": int(kept["x"].shape[0]), **counts},
        "keepPct": keep_pct,
        "contextPct": context_pct,
        "keepPctNote": (
            "share of the occupied 1/24-diameter voxels within 1.5x the supported extent whose "
            "gaussians are mostly keep (contextPct: keep or context)"
        ),
        "heldOutPsnr": psnr,
        "heldOutPsnrNote": "gsplat's own held-out frames (every 8th), from train_metrics.json",
        # keepPct's measure, counting only keep that held-out frames confirmed; null when
        # accuracy was not measured (then every keep is on coverage alone).
        "keepVerifiedPct": keep_verified_pct,
        "heldOut": held_report,
        "views": {"medianRoi": _median_int(support.views[roi_index])},
        "spreadDeg": {"medianRoi": _round(_median(support.spread_deg[roi_index]), 1)},
        "gsd": {
            "metric": metres is not None,
            "medianRoiModelUnits": _round(median_gsd, 6),
            "medianRoiMm": roi_gsd_mm,
            "criterion": (
                f"relative: keep at most {thresholds.keep_max_gsd_ratio:g}x and context at "
                f"most {thresholds.context_max_gsd_ratio:g}x the region of interest's median"
                + (
                    f", and keep at most {thresholds.keep_max_gsd_mm:g} mm/px"
                    if metres is not None and thresholds.keep_max_gsd_mm is not None
                    else ""
                )
            ),
            "note": (
                "depth / focal in the sharpest view, at the pose model's resolution"
                + (
                    "; mm/px from georeference's EXIF-GPS scale"
                    if metres is not None
                    else "; no metric scale (georeference did not align to GPS), so "
                    "model units only and the criterion is relative"
                )
            ),
        },
        "thresholds": thresholds.to_dict(),
        "capture": geometry,
        "tips": tips,
        "coverage": {
            "file": COVERAGE_PLY.name,
            "points": int(coverage_xyz.shape[0]),
            "cameras": cameras.count,
            "colours": {TIER_NAMES[tier]: list(rgb) for tier, rgb in TIER_COLOURS.items()},
        },
    }
    path = ctx.output(QUALITY_JSON.name)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    seconds = time.perf_counter() - started
    ctx.log(
        f"quality: {counts['keep']} keep, {counts['context']} context, {counts['drop']} drop; "
        f"bar {applied}; {int(kept['x'].shape[0])} gaussians out ({written} bytes); "
        f"keep {keep_pct}% of the ROI; {len(tips)} tip(s); {seconds:.1f} s"
    )
    metrics: dict[str, MetricValue] = {
        "mode": mode,
        "bar": applied,
        "gaussiansOut": int(kept["x"].shape[0]),
        "keep": counts["keep"],
        "context": counts["context"],
        "drop": counts["drop"],
        "roiRadius": round(roi.radius, 6),
        "tips": len(tips),
        "seconds": round(seconds, 2),
    }
    if keep_pct is not None:
        metrics["keepPct"] = keep_pct
    if keep_verified_pct is not None:
        metrics["keepVerifiedPct"] = keep_verified_pct
    if accuracy is not None:
        metrics["keepVerified"] = int(accuracy.verified.sum())
        metrics["demotedByHeldOut"] = int(accuracy.demoted.sum())
    if psnr is not None:
        metrics["heldOutPsnr"] = psnr
    summary = (
        f"{keep_pct:g}% of the scene met the high-quality bar"
        if keep_pct is not None
        else "nothing met the high-quality bar"
    )
    return StageOutcome(metrics=metrics, summary=summary)


def _footprint(columns: Mapping[str, F32]) -> F32:
    """Each gaussian's middle standard deviation, in world units (scales are logs).

    The middle axis, not the largest: a trained splat's surface gaussians are flat discs,
    which cover about their middle axis of the image face-on and less at a glance, and
    its fringe is needles, whose length is not what they hide. The largest would make a
    table top seen at 20 degrees occlude most of itself.
    """
    logs = np.stack([columns["scale_0"], columns["scale_1"], columns["scale_2"]])
    middle = np.median(np.nan_to_num(logs, nan=-30.0), axis=0)
    return np.exp(np.clip(middle, -30.0, 10.0)).astype(np.float32)


#: `gated.ply` is made from this. Named so the one place it is read says so.
GATED_PLY_SOURCE = "trained.ply"


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _median(values: npt.NDArray[Any]) -> float | None:
    return float(np.median(values)) if values.size else None


def _median_int(values: I32) -> int | None:
    return int(np.median(values)) if values.size else None
