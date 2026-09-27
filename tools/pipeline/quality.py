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

**Resolution.** The support mask's voxels, and the voxels `keepPct` counts, are sized
from the keep tier's pixel footprint (its GSD; `support_mask` derives the multiple), not
a fixed number of cells across the scene -- which made a building's voxels tens of times
coarser than a table's, and "X% of the scene met the bar" mean something different for
each. `supportMask.sizing` and `keepPctVoxel` say what size was used and why.

The stage never fails a run for being strict: a bar that would leave fewer than
`min_gaussians` falls back to the next looser one and says so (`barApplied`).
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import gaussians
import holdout
import outofcore
import sfm
import splat_io
import splat_stream
import support_mask
from artifacts import ArtifactDecl
from captures_bridge import sigmoid
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
    "RoiStats",
    "Support",
    "Thresholds",
    "assign_tiers",
    "coverage_positions",
    "estimate_roi",
    "extent_of",
    "gate",
    "measure_support",
    "occupied_share_of",
    "read_coverage",
    "support_pass",
    "supported_extent",
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


#: The visibility matrix of one support chunk (cameras x rows, a byte a pair) is held to
#: this: the rows are cut to fit, so 300 frames take 2^18 rows at a time (75 MB) and 1,000
#: take 84k. Views, spread and GSD are each a gaussian's own, so the cut changes nothing.
SEEN_BYTES = 80 << 20
#: Occluders are z-buffered in blocks of exactly this many opaque rows, as they always
#: were: each camera's median depth (the ROI's fallback for a walk) samples every block's
#: depths at a stride its size sets, so the block is part of the answer, not a tuning knob.
OCCLUDER_BLOCK = 1 << 18
#: Depth samples held in memory, over all cameras, before they are spilled to disk: at most
#: ~40k a camera per block, which at 300 cameras and a 30M-gaussian scene would be ~1.4 GB.
SAMPLE_BYTES = 32 << 20
#: Cells per axis a chunk is split into for culling cameras against, and the fewest rows
#: worth splitting (below it the culling costs more than the projections it saves).
CULL_CELLS = 8
CULL_MIN_ROWS = 4096

#: A stream of rows: per chunk, in row order, x, y and z (float32, as stored), alpha and
#: the occluder radius (`footprint`). Called once per pass.
Rows = Callable[[], Iterable[tuple[F32, F32, F32, npt.NDArray[Any], F32]]]


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
    chunk: int = OCCLUDER_BLOCK,
) -> Support:
    """Views, spread and GSD for every gaussian of arrays in memory (`support_pass` over
    them, `chunk` rows at a time).

    `footprint` is each gaussian's radius in world units (its largest standard
    deviation, for a splat). An occluder covers every z-buffer cell within that radius
    of its centre, up to `max_footprint_cells` cells: a surface made of a few large
    gaussians must occlude like the surface it renders as, not like a sieve of centres.
    """
    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    count = int(points.shape[0])
    opacity = np.asarray(alpha).reshape(-1)
    radius = (
        np.zeros(count, dtype=np.float32)
        if footprint is None
        else np.asarray(footprint, dtype=np.float32).reshape(-1)
    )

    def rows() -> Iterator[tuple[F32, F32, F32, npt.NDArray[Any], F32]]:
        for start, stop in splat_io.ranges(count, chunk):
            part = points[start:stop]
            yield part[:, 0], part[:, 1], part[:, 2], opacity[start:stop], radius[start:stop]

    views = np.zeros(count, dtype=np.int32)
    spread = np.zeros(count, dtype=np.float32)
    gsd = np.full(count, _UNSEEN_GSD, dtype=np.float32)

    def emit(start: int, block_views: I32, block_spread: F32, block_gsd: F32) -> None:
        stop = start + block_views.shape[0]
        views[start:stop], spread[start:stop], gsd[start:stop] = (
            block_views,
            block_spread,
            block_gsd,
        )

    depths = support_pass(
        rows,
        cameras,
        emit,
        zbuffer_side=zbuffer_side,
        occluder_opacity=occluder_opacity,
        depth_tolerance=depth_tolerance,
        max_footprint_cells=max_footprint_cells,
        block=chunk,
    )
    return Support(views=views, spread_deg=spread, gsd=gsd, camera_depth=depths)


def support_pass(
    rows: Rows,
    cameras: Cameras,
    emit: Callable[[int, I32, F32, F32], None],
    *,
    zbuffer_side: int = 160,
    occluder_opacity: float = 0.5,
    depth_tolerance: float = 0.05,
    max_footprint_cells: int = 3,
    block: int = OCCLUDER_BLOCK,
    spill: Path | None = None,
) -> F64:
    """Views, spread and GSD of every row `rows` yields, in two passes over them and in
    memory that does not grow with their number; returns each camera's median depth.

    Pass one fills a z-buffer per camera from the opaque rows, `block` of them at a time
    (`_ZBuffers`). Pass two measures each chunk of rows against the buffers and hands the
    results to `emit(start, views, spread, gsd)`, in row order. Both split a chunk into
    spatial cells and test each camera against the cells first (`_Cull`), so a camera
    projects only the rows that could be in its frame -- which leaves every result as it
    was, and on a scene larger than any one frame saves most of the work.

    Memory is the cameras' buffers (cameras x `zbuffer_side`^2 floats), one occluder
    block, and a chunk's `SEEN_BYTES` visibility matrix. Positions are handled as three
    contiguous columns rather than an (n, 3) array: the per-camera projection is then a
    dozen streaming float32 operations, which measured 5x faster than a matmul and a
    strided gather here.
    """
    centres = cameras.centres
    scale = float(np.median(np.linalg.norm(centres - centres.mean(axis=0), axis=1))) or 1.0
    near = 1e-3 * scale
    zbuffers = _ZBuffers(cameras, zbuffer_side, near, max_footprint_cells, block, spill)
    for x, y, z, alpha, radius in rows():
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        opaque = finite & (np.nan_to_num(alpha, nan=0.0) >= occluder_opacity)
        reach = np.nan_to_num(np.asarray(radius, dtype=np.float32), nan=0.0, posinf=0.0)
        zbuffers.add((x[opaque], y[opaque], z[opaque]), reach[opaque])
    buffers, depths = zbuffers.finish()
    step = max(1, min(block, SEEN_BYTES // max(1, cameras.count)))
    start = 0
    for x, y, z, _, _ in rows():
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        zero = np.float32(0.0)
        columns = (
            np.where(finite, x, zero).astype(np.float32),
            np.where(finite, y, zero).astype(np.float32),
            np.where(finite, z, zero).astype(np.float32),
        )
        size = int(finite.shape[0])
        for low in range(0, size, step):
            high = min(size, low + step)
            views, spread, gsd = _support_rows(
                _slice(columns, low, high),
                finite[low:high],
                cameras,
                centres,
                buffers,
                zbuffer_side,
                near,
                depth_tolerance,
            )
            emit(start + low, views, spread, gsd)
        start += size
    return depths


def _support_rows(
    block: Columns,
    finite: Bools,
    cameras: Cameras,
    centres: F64,
    buffers: list[F32],
    side: int,
    near: float,
    tolerance: float,
) -> tuple[I32, F32, F32]:
    """One chunk of the support pass: views, spread and GSD of each of its rows."""
    size = int(block[0].shape[0])
    seen = np.zeros((cameras.count, size), dtype=np.bool_)
    direction_sum = np.zeros((3, size), dtype=np.float32)
    block_gsd = np.full(size, _UNSEEN_GSD, dtype=np.float32)
    cull = _Cull(block)
    for j in range(cameras.count):
        subset = cull.rows(cameras, j, near)
        if subset is not None and subset.size == 0:
            continue
        index, depth = _visible(
            block if subset is None else _take(block, subset),
            cameras,
            j,
            buffers[j],
            side,
            near,
            tolerance,
        )
        if index.size == 0:
            continue
        if subset is not None:
            index = subset[index]
        seen[j, index] = True
        toward = _unit_towards(centres[j], _take(block, index))
        for axis in range(3):
            direction_sum[axis, index] += toward[axis]
        focal = float(min(cameras.fx[j], cameras.fy[j]))
        # `index` is unique within one camera, so a plain gather-min-scatter is exact.
        block_gsd[index] = np.minimum(block_gsd[index], depth / np.float32(focal))
    views = seen.sum(axis=0).astype(np.int32)
    views[~finite] = 0
    gsd = np.where(views > 0, block_gsd, _UNSEEN_GSD).astype(np.float32)
    spread = _spread(block, centres, seen, direction_sum, views)
    return views, spread, gsd


class _ZBuffers:
    """The nearest opaque depth per z-buffer cell, per camera, filled a block at a time.

    Opaque rows arrive in row order in chunks of any size; they are regrouped into blocks
    of exactly `block` rows (the last one shorter), which is how `measure_support` always
    cut them, because each camera's depth sample is taken per block. Each block's depths
    are sampled per camera at a stride that keeps ~20k of them; samples beyond
    `SAMPLE_BYTES` are spilled to `spill` (a temporary directory by default), and the
    median of each camera's samples is its depth.
    """

    def __init__(
        self,
        cameras: Cameras,
        side: int,
        near: float,
        max_cells: int,
        block: int,
        spill: Path | None,
    ) -> None:
        self.cameras = cameras
        self.side = side
        self.near = near
        self.max_cells = max_cells
        self.block = block
        self.buffers: list[F32] = []
        for j in range(cameras.count):
            gw, gh = _grid(cameras, j, side)
            self.buffers.append(np.full(gw * gh, np.inf, dtype=np.float32))
        self._pending: list[tuple[Columns, F32]] = []
        self._pending_rows = 0
        self._samples: list[list[F32]] = [[] for _ in range(cameras.count)]
        self._sampled = np.zeros(cameras.count, dtype=np.bool_)
        self._held = 0
        self._spill_root = spill
        self._spill: Path | None = None
        self._temporary: tempfile.TemporaryDirectory[str] | None = None

    def add(self, columns: Columns, radius: F32) -> None:
        size = int(columns[0].shape[0])
        if size == 0:
            return
        self._pending.append((columns, radius))
        self._pending_rows += size
        while self._pending_rows >= self.block:
            self._fill(*self._take(self.block))

    def _take(self, count: int) -> tuple[Columns, F32]:
        xs, ys, zs, rs = zip(*((c[0], c[1], c[2], r) for c, r in self._pending), strict=True)
        x, y, z, r = (np.concatenate(parts) for parts in (xs, ys, zs, rs))
        self._pending = (
            [] if count >= x.shape[0] else [((x[count:], y[count:], z[count:]), r[count:])]
        )
        self._pending_rows = max(0, int(x.shape[0]) - count)
        return (x[:count], y[:count], z[:count]), r[:count]

    def _fill(self, occluders: Columns, radius: F32) -> None:
        """`_zbuffers`' inner loop for one block, for every camera."""
        cameras, side, max_cells = self.cameras, self.side, self.max_cells
        cull = _Cull(occluders)
        for j in range(cameras.count):
            subset = cull.rows(cameras, j, self.near)
            if subset is not None and subset.size == 0:
                continue
            index, u, v, z = _project(
                occluders if subset is None else _take(occluders, subset), cameras, j, self.near
            )
            if z.size == 0:
                continue
            if subset is not None:
                index = subset[index]
            gw, gh = _grid(cameras, j, side)
            buffer = self.buffers[j]
            cell_world = np.float32(
                float(cameras.width[j]) / gw / float(min(cameras.fx[j], cameras.fy[j]))
            )
            row, col = _rows_cols(u, v, cameras, j, side)
            np.minimum.at(buffer, row * gw + col, z)
            # The footprint, in cells: world radius over what one cell spans at that depth.
            reach = np.minimum((radius[index] / (z * cell_world)).astype(np.intp), max_cells)
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
            sample = z[:: max(1, z.size // 20_000)]
            self._samples[j].append(sample)
            self._sampled[j] = True
            self._held += int(sample.nbytes)
        if self._held > SAMPLE_BYTES:
            self._spill_samples()

    def _spill_dir(self) -> Path:
        if self._spill is None:
            if self._spill_root is None:
                self._temporary = tempfile.TemporaryDirectory(prefix="zbuffer-samples-")
                self._spill = Path(self._temporary.name)
            else:
                self._spill = self._spill_root
                self._spill.mkdir(parents=True, exist_ok=True)
        return self._spill

    def _spill_samples(self) -> None:
        directory = self._spill_dir()
        for j, held in enumerate(self._samples):
            if held:
                with (directory / f"{j}.f32").open("ab") as handle:
                    for sample in held:
                        np.ascontiguousarray(sample, dtype=np.float32).tofile(handle)
                held.clear()
        self._held = 0

    def finish(self) -> tuple[list[F32], F64]:
        if self._pending_rows:
            self._fill(*self._take(self._pending_rows))
        depths = np.full(self.cameras.count, np.nan, dtype=np.float64)
        try:
            for j in range(self.cameras.count):
                if not self._sampled[j]:
                    continue
                parts = list(self._samples[j])
                spilled = None if self._spill is None else self._spill / f"{j}.f32"
                if spilled is not None and spilled.is_file():
                    parts.append(np.fromfile(spilled, dtype=np.float32))
                    spilled.unlink()
                # Order does not matter to a median: only which values were sampled.
                depths[j] = float(np.median(np.concatenate(parts)))
        finally:
            if self._temporary is not None:
                self._temporary.cleanup()
        return self.buffers, depths


class _Cull:
    """A chunk's rows grouped into spatial cells, so a camera is tested against cells.

    `rows(cameras, j, near)` is None when camera `j` may see every cell (project them
    all), an empty array when it sees none, and otherwise the rows of the cells it may
    see, ascending -- so a camera's projections come out in the order, and with the
    values, they would have had over the whole chunk, and nothing it culls could have
    been in its frame.
    """

    def __init__(self, columns: Columns) -> None:
        size = int(columns[0].shape[0])
        self.cell: npt.NDArray[np.intp] | None = None
        if size < CULL_MIN_ROWS:
            return
        index = np.zeros(size, dtype=np.intp)
        for values in columns:
            least, most = float(values.min()), float(values.max())
            span = most - least
            if span > 0 and math.isfinite(span):
                bins = ((values.astype(np.float64) - least) * (CULL_CELLS / span)).astype(np.intp)
                index = index * CULL_CELLS + np.clip(bins, 0, CULL_CELLS - 1)
            else:
                index = index * CULL_CELLS
        order = np.argsort(index, kind="stable")
        ordered = index[order]
        starts = np.flatnonzero(np.concatenate([[True], ordered[1:] != ordered[:-1]]))
        self.cell = index
        self.occupied = ordered[starts]
        low = np.stack([np.minimum.reduceat(c[order], starts) for c in columns], axis=1)
        high = np.stack([np.maximum.reduceat(c[order], starts) for c in columns], axis=1)
        low, high = low.astype(np.float64), high.astype(np.float64)
        # The eight corners of each occupied cell's own bounding box.
        pick = np.array([[(k >> a) & 1 for a in range(3)] for k in range(8)], dtype=bool)
        self.corners = np.where(pick[None, :, :], high[:, None, :], low[:, None, :])
        self.magnitude = np.abs(self.corners)

    def rows(self, cameras: Cameras, j: int, near: float) -> npt.NDArray[np.intp] | None:
        if self.cell is None:
            return None
        visible = _frustum(cameras, j, self.corners, self.magnitude, near)
        if bool(visible.all()):
            return None
        if not bool(visible.any()):
            return np.zeros(0, dtype=np.intp)
        wanted = np.zeros(CULL_CELLS**3, dtype=np.bool_)
        wanted[self.occupied[visible]] = True
        return np.flatnonzero(wanted[self.cell])


def _frustum(cameras: Cameras, j: int, corners: F64, magnitude: F64, near: float) -> Bools:
    """Which boxes (given by their corners) camera `j` may see, conservatively.

    A box is culled only when all eight corners are beyond one side of the frustum --
    behind the near plane, or left, right, above or below the image -- by a margin, so
    that no point in it can be seen after `_project`'s float32 rounding. Each test is
    linear in the camera-frame point (`u < 0` is `fx x + cx z < 0` in front of the
    camera), so all corners beyond it means every point of the box is. The margin is 1e-4
    of the magnitudes involved -- float32 rounding is ~6e-8 of them -- and grows with how
    close to the near plane the box reaches, where dividing by depth amplifies rounding.
    """
    rotation = cameras.rotations[j]
    translation = cameras.translations[j]
    camera = corners @ rotation.T + translation
    size = magnitude @ np.abs(rotation).T + np.abs(translation)
    xc, yc, zc = camera[..., 0], camera[..., 1], camera[..., 2]
    mx, my, mz = size[..., 0], size[..., 1], size[..., 2]
    depth_scale = mz.max(axis=1)
    tau = 1e-4 * np.maximum(1.0, depth_scale / near)
    fx, fy = float(cameras.fx[j]), float(cameras.fy[j])
    cx, cy = float(cameras.cx[j]), float(cameras.cy[j])
    width, height = float(cameras.width[j]), float(cameras.height[j])
    margin_u = (tau * (fx * mx + (abs(cx) + width) * mz).max(axis=1))[:, None]
    margin_v = (tau * (fy * my + (abs(cy) + height) * mz).max(axis=1))[:, None]
    behind = (zc <= near - (tau * depth_scale)[:, None]).all(axis=1)
    left = (fx * xc + cx * zc < -margin_u).all(axis=1)
    right = (fx * xc + (cx - width) * zc > margin_u).all(axis=1)
    above = (fy * yc + cy * zc < -margin_v).all(axis=1)
    below = (fy * yc + (cy - height) * zc > margin_v).all(axis=1)
    visible: Bools = ~(behind | left | right | above | below)
    return visible


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


#: A stream of finite points, (m, 3) float64 per chunk. Called once per pass.
Points = Callable[[], Iterable[F64]]


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
    return extent_of(lambda: [kept], fallback, quantile)


def extent_of(kept: Points, fallback: Roi, quantile: float = 99.0) -> Roi:
    """`supported_extent` of streamed keep points: the same `np.median` (per axis) and
    `np.percentile`, exactly, in passes over them."""
    count = sum(int(points.shape[0]) for points in kept())
    if count < 32:
        return replace(fallback, method=f"{fallback.method} (too little keep for an extent)")

    def axes() -> Iterator[tuple[F64, npt.NDArray[np.int64]]]:
        for points in kept():
            yield points.T.reshape(-1), np.repeat(np.arange(3, dtype=np.int64), points.shape[0])

    selection = outofcore.order_statistics(axes, outofcore.median_ranks, groups=[0, 1, 2])
    middle = np.array(
        [[selection.at(axis, rank) for axis in range(3)] for rank in outofcore.median_ranks(count)],
        dtype=np.float64,
    )
    # `np.median(kept, axis=0)` is the mean of the one or two middle rows, over axis 0.
    centre = np.mean(middle, axis=0)

    def distances() -> Iterator[F64]:
        for points in kept():
            yield np.linalg.norm(points - centre, axis=1)

    found = outofcore.percentile(distances, [quantile])
    assert found is not None
    radius = float(found[0])
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
    positions = coverage_positions(int(candidates.size), limit, seed)
    return candidates if positions is None else candidates[positions]


def coverage_positions(candidates: int, limit: int, seed: int = 0) -> npt.NDArray[np.int64] | None:
    """Which of `candidates` visible gaussians `coverage.ply` samples, as their ranks
    among the candidates (None: all of them). Drawn from the count alone, so the stage can
    count the candidates in one pass and pick them out in the next: `rng.choice` over an
    array draws the same positions as over its length, and sorting them sorts the
    gaussians, since candidates are in row order."""
    if candidates <= limit:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, size=limit, replace=False)).astype(np.int64)


# ---------------------------------------------------------------------------------------
# What the capture could have done better
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoiStats:
    """What the tips need of the gaussians in the region of interest: how many there are,
    how many fall short of keep on each criterion, and their median view count."""

    count: int
    short_views: int
    short_spread: int
    short_gsd: int
    median_views: int | None

    @staticmethod
    def of(views: I32, spread: F32, gsd_ratio: F32, tiers: U8, thresholds: Thresholds) -> RoiStats:
        """From the ROI's gaussians' own arrays."""
        short = tiers != TIER_KEEP
        return RoiStats(
            count=int(views.size),
            short_views=int((short & (views < thresholds.keep_min_views)).sum()),
            short_spread=int((short & (spread < thresholds.keep_min_spread_deg)).sum()),
            short_gsd=int((short & (gsd_ratio > thresholds.keep_max_gsd_ratio)).sum()),
            median_views=int(np.median(views)) if views.size else None,
        )


def capture_tips(
    *,
    centres: F64,
    roi: Roi,
    up: F64 | None,
    roi_stats: RoiStats,
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
    if roi_stats.count:
        # Shares of the ROI's gaussians, as `(mask).mean()` over them gave: count / n.
        failing = {
            "views": roi_stats.short_views / roi_stats.count,
            "spread": roi_stats.short_spread / roi_stats.count,
            "gsd": roi_stats.short_gsd / roi_stats.count,
        }
        geometry["shortOfKeepBy"] = {key: round(value, 3) for key, value in failing.items()}
        median_views = roi_stats.median_views
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
# keepPct: its voxel and its share
# ---------------------------------------------------------------------------------------


#: keepPct's voxel is never coarser than this fraction of the scene sphere's diameter:
#: the fixed grid it used to be, now only the bound for a scene imaged so coarsely that
#: its footprint would make fewer voxels than that across it.
SHARE_MIN_CELLS = 24


def _share_voxel(
    mask: support_mask.SupportMask | None, near: support_mask.PointSet, scene: Roi
) -> tuple[float, str]:
    """keepPct's voxel side, in model units, and what decided it.

    The resolution the keep tier's pixel footprint supports -- the support mask's, before
    any memory or byte cap coarsened the mask itself -- so "X% of the scene met the bar"
    is measured at the data's own resolution, and the same capture at 25x the scale says
    the same number. A fixed count of cells across the scene made a building's voxels
    tens of times larger than a table's. With no mask (too little keep), the same rule
    over `near` -- every seen gaussian in the scene -- stands in.
    """
    ceiling = 2.0 * scene.radius / SHARE_MIN_CELLS
    sizing = mask.sizing if mask is not None else None
    basis = "keep-footprint"
    if sizing is None or sizing.rule != "footprint":
        sizing = support_mask.resolution_of(near)
        basis = "seen-footprint"
    if sizing is None:
        return ceiling, "scene-diameter"
    if sizing.resolution >= ceiling:
        return ceiling, f"{basis}, capped at 1/{SHARE_MIN_CELLS} of the scene's diameter"
    return sizing.resolution, basis


#: A stream of `(xyz (m, 3) float32, tiers, alpha)` chunks, for `occupied_share_of`.
TierRows = Callable[[], Iterable[tuple[F32, U8, F32]]]


def _occupied_share(
    xyz: F32, tiers: U8, alpha: F32, roi: Roi, floor: int, voxel: float
) -> float | None:
    """`occupied_share_of` arrays in memory."""
    return occupied_share_of(lambda: [(xyz, tiers, alpha)], roi, floor, voxel, int(xyz.shape[0]))


def occupied_share_of(
    rows: TierRows, roi: Roi, floor: int, voxel: float, count: int
) -> float | None:
    """The share of occupied voxels in the ROI whose gaussians mostly reach `floor`.

    "Volume that is keep" measured over the volume that has anything in it: an orbit's
    ROI sphere is mostly air, and a percentage of air would say nothing. `voxel` is the
    side in model units (`_share_voxel`). Only occupied voxels are ever held, a partition
    of them at a time (`outofcore.group_counts`, `count` an upper bound on the rows), so
    a fine voxel over a large scene costs neither the grid nor the gaussians.
    """
    cells = int(min(support_mask.MAX_DIM, max(1, math.ceil(2.0 * roi.radius / voxel))))
    size = max(voxel, 2.0 * roi.radius / cells)

    def keyed() -> Iterator[tuple[npt.NDArray[np.int64], F64]]:
        for xyz, tiers, alpha in rows():
            inside = roi.contains(xyz) & (np.nan_to_num(alpha, nan=0.0) >= 0.1)
            ijk = np.floor((xyz[inside].astype(np.float64) - (roi.centre - roi.radius)) / size)
            ijk = np.clip(ijk, 0, cells - 1).astype(np.int64)
            linear = (ijk[:, 0] * cells + ijk[:, 1]) * cells + ijk[:, 2]
            yield linear, (tiers[inside] >= floor).astype(np.float64)

    voxels = good = 0
    for keys, counts, sums in outofcore.group_counts(keyed, rows=count):
        voxels += int(keys.shape[0])
        good += int((sums >= 0.5 * counts).sum())
    if voxels == 0:
        return None
    return round(100.0 * float(good / voxels), 1)


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


# ---------------------------------------------------------------------------------------
# The stage, a chunk at a time
# ---------------------------------------------------------------------------------------

#: Bits of the per-gaussian `flags` column the stage keeps between passes.
FLAG_MEASURED = 1
FLAG_VERIFIED = 2
FLAG_DEMOTED = 4
FLAG_COVERAGE_KEEP = 8


class _Store:
    """The stage's per-gaussian columns, on disk between passes, read back in chunks.

    The stage never holds a whole column: positions, alpha and footprint are read from
    the splat once and kept here (16 bytes a gaussian, not the PLY's 56 -- or gsplat's
    248), and every later pass reads the columns it needs a chunk at a time.
    """

    def __init__(self, directory: Path, count: int, chunk: int) -> None:
        self.columns = splat_io.ColumnStore(directory)
        self.count = count
        self.chunk = chunk

    def rows(self, *names: str) -> Iterator[tuple[int, list[npt.NDArray[Any]]]]:
        for start, stop in splat_io.ranges(self.count, self.chunk):
            yield start, [self.columns.read(name, start, stop) for name in names]

    def xyz(self, start: int, stop: int) -> F32:
        return np.stack([self.columns.read(axis, start, stop) for axis in "xyz"], axis=1)


@dataclass(frozen=True)
class _Scale:
    """What turns a gaussian's GSD into the criteria's units: the ROI's median GSD, and
    metres per model unit when georeference measured one."""

    median_gsd: float
    metres: float | None

    def ratio(self, gsd: F32, seen: Bools) -> F32:
        if math.isfinite(self.median_gsd) and self.median_gsd > 0:
            return (gsd / np.float32(self.median_gsd)).astype(np.float32)
        return np.where(seen, np.float32(1.0), np.float32(np.inf)).astype(np.float32)

    def millimetres(self, gsd: F32) -> F32 | None:
        if self.metres is None:
            return None
        return (gsd * np.float32(self.metres * 1000.0)).astype(np.float32)


@dataclass
class _Tally:
    """Counts gathered over the final tiers, in one pass."""

    tiers: dict[int, int]
    candidates: int = 0
    roi: int = 0
    short_views: int = 0
    short_spread: int = 0
    short_gsd: int = 0
    roi_coverage_keep: int = 0
    roi_demoted: int = 0
    verified: int = 0
    demoted: int = 0
    measured: int = 0


@stage_impl(
    "support_gate",
    consumes=("trained.ply", "poses"),
    optional_consumes=("georef.json", "train_metrics.json", holdout.HOLDOUT.name),
    produces=(GATED_PLY, QUALITY_JSON, COVERAGE_PLY),
    summary="the quality bar: keep what enough frames saw well, and say what to capture next",
)
def support_gate(ctx: StageContext) -> StageOutcome:
    """Measure per-gaussian support, tier it, gate the splat by `bar`. See the module.

    A chunk at a time throughout (`chunk_gaussians` rows, 2^18 by default): the splat is
    read once into the stage's own columns under its work directory, and every result
    -- `gated.ply`, `quality.json`, `coverage.ply` -- is the one the whole-splat stage
    wrote (tests/test_chunked_equivalence.py holds it to a frozen copy of that stage).
    """
    started = time.perf_counter()
    mode = str(ctx.param("mode", "refine"))
    if mode not in MODES:
        raise ValueError(f"quality.mode must be one of {', '.join(MODES)}, not {mode!r}")
    bar = str(ctx.param("bar", "balanced"))
    if bar not in BARS:
        raise ValueError(f"quality.bar must be one of {', '.join(BARS)}, not {bar!r}")
    thresholds = Thresholds.from_params(ctx.params)
    chunk = int(ctx.param("chunk_gaussians", splat_io.CHUNK))
    source = splat_stream.open_splat(ctx.input(GATED_PLY_SOURCE), chunk=chunk)
    model = sfm.read_model(ctx.input("poses"))
    cameras = Cameras.from_model(model)
    scratch = ctx.work_dir / "quality-chunks"
    store = _Store(scratch / "columns", source.count, chunk)
    try:
        return _grade(ctx, source, model, cameras, store, scratch, thresholds, mode, bar, started)
    finally:
        store.columns.remove()
        shutil.rmtree(scratch, ignore_errors=True)


def _grade(
    ctx: StageContext,
    source: splat_stream.SplatSource,
    model: sfm.Model,
    cameras: Cameras,
    store: _Store,
    scratch: Path,
    thresholds: Thresholds,
    mode: str,
    bar: str,
    started: float,
) -> StageOutcome:
    count = source.count
    columns = store.columns
    # One read of the splat: what every later pass needs of it, 16 bytes a gaussian.
    writers = {
        name: columns.create(name, np.float32) for name in ("x", "y", "z", "alpha", "radius")
    }
    for _, read in source.chunks():
        for axis in "xyz":
            writers[axis].append(read[axis])
        writers["alpha"].append(np.ascontiguousarray(sigmoid(read["opacity"]), dtype=np.float32))
        writers["radius"].append(_footprint(read))
    columns.finish()

    def positions() -> Iterator[tuple[F32, F32, F32, F32, F32]]:
        for _, (x, y, z, alpha, radius) in store.rows("x", "y", "z", "alpha", "radius"):
            yield x, y, z, alpha, radius

    outputs = {
        "views": columns.create("views", np.int32),
        "spread": columns.create("spread", np.float32),
        "gsd": columns.create("gsd", np.float32),
    }

    def emit(_: int, views: I32, spread: F32, gsd: F32) -> None:
        outputs["views"].append(views)
        outputs["spread"].append(spread)
        outputs["gsd"].append(gsd)

    camera_depth = support_pass(
        positions,
        cameras,
        emit,
        zbuffer_side=int(ctx.param("zbuffer_side", 160)),
        occluder_opacity=float(ctx.param("occluder_opacity", 0.5)),
        depth_tolerance=float(ctx.param("depth_tolerance", 0.05)),
        spill=scratch / "zbuffer-samples",
    )
    columns.finish()
    measured_s = time.perf_counter() - started
    views_median = outofcore.grouped_median(
        lambda: (views for _, (views,) in store.rows("views")), None
    )
    ctx.log(
        f"support: {count} gaussians x {cameras.count} cameras in {measured_s:.1f} s; "
        f"median views {int(views_median[0]) if 0 in views_median else 'nan'}"
    )

    roi = estimate_roi(
        cameras.centres,
        cameras.axes,
        camera_depth,
        radius_factor=float(ctx.param("roi_radius_factor", 0.5)),
    )

    def in_roi_of(start: int, stop: int, views: I32) -> Bools:
        contained: Bools = roi.contains(store.xyz(start, stop)) & (views > 0)
        return contained

    # The GSD the criterion is relative to: the median over the ROI's seen gaussians, or
    # over every seen one when none of them is in the ROI.
    def reference_gsd() -> Iterator[tuple[F32, npt.NDArray[np.int64]]]:
        for start, (views, gsd) in store.rows("views", "gsd"):
            seen = views > 0
            inside = in_roi_of(start, start + views.shape[0], views)
            yield (
                np.concatenate([gsd[inside], gsd[seen]]),
                np.concatenate(
                    [np.zeros(int(inside.sum()), np.int64), np.ones(int(seen.sum()), np.int64)]
                ),
            )

    medians = outofcore.grouped_median(reference_gsd, [0, 1])
    median_gsd = float(medians[0]) if 0 in medians else float(medians.get(1, math.nan))
    georef = _read_json(ctx.input("georef.json")) if ctx.has_input("georef.json") else None
    scale = _Scale(median_gsd, _metres_per_unit(georef))

    def coverage_tiers(start: int, arrays: list[npt.NDArray[Any]]) -> U8:
        views, spread, gsd, alpha = arrays
        xyz = store.xyz(start, start + views.shape[0])
        roi_radii = (
            np.linalg.norm(xyz.astype(np.float64) - roi.centre, axis=1) / max(roi.radius, 1e-12)
        ).astype(np.float32)
        return assign_tiers(
            views,
            spread,
            scale.ratio(gsd, views > 0),
            alpha,
            thresholds,
            scale.millimetres(gsd),
            roi_radii,
        )

    first = columns.create("coverage_tiers", np.uint8)
    for start, arrays in store.rows("views", "spread", "gsd", "alpha"):
        first.append(coverage_tiers(start, arrays))
    columns.finish()

    # Measured accuracy, where held-out frames saw the splat: keep must also match them.
    held, held_status = (
        holdout.open_columns(ctx.input(holdout.HOLDOUT.name), count)
        if ctx.has_input(holdout.HOLDOUT.name)
        else (None, {"status": "missing", "reason": "the train stage wrote no holdout/"})
    )
    min_weight = thresholds.holdout_min_weight

    def measured_of(error: F32, weight: F32) -> Bools:
        measured: Bools = np.isfinite(error) & (weight >= min_weight)
        return measured

    reference: float | None = None
    limit: float | None = None
    if held is not None:
        columns_held = held

        def population() -> Iterator[tuple[F32, npt.NDArray[np.int64]]]:
            for start, (tiers,) in store.rows("coverage_tiers"):
                error, weight = columns_held.read(start, start + tiers.shape[0])
                measured = measured_of(error, weight)
                keep = measured & (tiers == TIER_KEEP)
                yield (
                    np.concatenate([error[keep], error[measured]]),
                    np.concatenate(
                        [
                            np.zeros(int(keep.sum()), np.int64),
                            np.ones(int(measured.sum()), np.int64),
                        ]
                    ),
                )

        errors = outofcore.grouped_median(population, [0, 1])
        # `holdout.judge`: the median of measured coverage-keep, else of everything measured.
        found = errors.get(0, errors.get(1))
        reference = None if found is None else float(found)
        limit = holdout.limit_of(
            reference,
            max_ratio=thresholds.keep_max_holdout_error_ratio,
            max_abs=thresholds.keep_max_holdout_error,
        )

    final = columns.create("tiers", np.uint8)
    flags = columns.create("flags", np.uint8)
    for start, (tiers,) in store.rows("coverage_tiers"):
        coverage_keep = tiers == TIER_KEEP
        bits = np.where(coverage_keep, FLAG_COVERAGE_KEEP, 0).astype(np.uint8)
        after = tiers
        if held is not None:
            error, weight = held.read(start, start + tiers.shape[0])
            measured = measured_of(error, weight)
            demoted = (
                coverage_keep & measured & (error > np.float32(limit))
                if limit is not None
                else np.zeros(tiers.shape, dtype=bool)
            )
            after = tiers.copy()
            after[demoted] = np.uint8(TIER_CONTEXT)
            verified = (after == TIER_KEEP) & measured
            bits |= np.where(measured, FLAG_MEASURED, 0).astype(np.uint8)
            bits |= np.where(verified, FLAG_VERIFIED, 0).astype(np.uint8)
            bits |= np.where(demoted, FLAG_DEMOTED, 0).astype(np.uint8)
        final.append(after)
        flags.append(bits)
    columns.finish()

    # Everything the rest of the stage counts, in one pass over the final tiers.
    tally = _Tally(tiers=dict.fromkeys(TIER_NAMES, 0))
    names = ("views", "spread", "gsd", "alpha", "tiers", "flags")
    for start, (views, spread, gsd, alpha, tiers, bits) in store.rows(*names):
        stop = start + views.shape[0]
        for tier in (TIER_DROP, TIER_CONTEXT, TIER_KEEP):
            tally.tiers[tier] += int((tiers == tier).sum())
        xyz = store.xyz(start, stop)
        tally.candidates += int((np.isfinite(xyz).all(axis=1) & (alpha >= 0.05)).sum())
        inside = in_roi_of(start, stop, views)
        short = inside & (tiers != TIER_KEEP)
        tally.roi += int(inside.sum())
        tally.short_views += int((short & (views < thresholds.keep_min_views)).sum())
        tally.short_spread += int((short & (spread < thresholds.keep_min_spread_deg)).sum())
        ratio = scale.ratio(gsd, views > 0)
        tally.short_gsd += int((short & (ratio > thresholds.keep_max_gsd_ratio)).sum())
        tally.roi_coverage_keep += int((inside & ((bits & FLAG_COVERAGE_KEEP) > 0)).sum())
        tally.roi_demoted += int((inside & ((bits & FLAG_DEMOTED) > 0)).sum())
        tally.verified += int(((bits & FLAG_VERIFIED) > 0).sum())
        tally.demoted += int(((bits & FLAG_DEMOTED) > 0).sum())
        tally.measured += int(((bits & FLAG_MEASURED) > 0).sum())
    coverage_keep_count = sum(
        int(((bits & FLAG_COVERAGE_KEEP) > 0).sum()) for _, (bits,) in store.rows("flags")
    )
    if held is not None:
        ctx.log(
            f"held-out: {tally.measured} of {count} gaussians measured; "
            f"{tally.demoted} of {coverage_keep_count} coverage-keep over "
            f"the limit {limit}; {tally.verified} keep verified"
        )
    else:
        ctx.log(
            f"held-out: not used ({held_status['status']}: {held_status.get('reason')}); "
            f"keep is judged by coverage alone"
        )

    # The bar, falling back while it would leave too little -- decided from the counts,
    # before anything is written, since `gated.ply`'s header states its row count.
    min_gaussians = int(ctx.param("min_gaussians", 1000))
    context_fade = float(ctx.param("context_fade", 0.5))
    passing = {
        "strict": tally.tiers[TIER_KEEP],
        "balanced": tally.tiers[TIER_KEEP] + tally.tiers[TIER_CONTEXT],
        "everything": count,
    }
    applied = bar
    while passing[applied] < min(min_gaussians, count) and applied != "everything":
        applied = BARS[BARS.index(applied) + 1]
        ctx.log(f"WARNING: bar {bar!r} leaves too little; falling back to {applied!r}")
    kept_count = passing[applied]
    with splat_io.PlyWriter(
        ctx.output(GATED_PLY.name), gaussians.CANONICAL_PROPERTIES, count=kept_count
    ) as writer:
        for start, read in source.chunks():
            tiers = columns.read("tiers", start, start + read["x"].shape[0])
            writer.append(gate(read, tiers, applied, context_fade=context_fade))
    written = ctx.output(GATED_PLY.name).stat().st_size

    # coverage.ply: a fixed-seed sample of the visible gaussians, then the camera path.
    limit_points = int(ctx.param("coverage_points", 150_000))
    positions_wanted = coverage_positions(tally.candidates, max(0, limit_points - cameras.count))
    sample_xyz: list[F32] = []
    sample_tiers: list[U8] = []
    seen_candidates = 0
    for start, (alpha, tiers) in store.rows("alpha", "tiers"):
        xyz = store.xyz(start, start + alpha.shape[0])
        candidates = np.flatnonzero(np.isfinite(xyz).all(axis=1) & (alpha >= 0.05))
        if positions_wanted is None:
            chosen = candidates
        else:
            low = np.searchsorted(positions_wanted, seen_candidates)
            high = np.searchsorted(positions_wanted, seen_candidates + candidates.shape[0])
            chosen = candidates[positions_wanted[low:high] - seen_candidates]
        seen_candidates += int(candidates.shape[0])
        sample_xyz.append(xyz[chosen])
        sample_tiers.append(tiers[chosen])
    path_order = np.argsort(np.asarray(cameras.names))
    coverage_xyz = np.concatenate(
        [*sample_xyz, cameras.centres[path_order].astype(np.float32)]
    ).reshape(-1, 3)
    coverage_tiers_out = np.concatenate(
        [*sample_tiers, np.full(cameras.count, TIER_CAMERA, dtype=np.uint8)]
    )
    write_coverage(ctx.output(COVERAGE_PLY.name), coverage_xyz, coverage_tiers_out)
    del sample_xyz, sample_tiers

    counts = {name: tally.tiers[tier] for tier, name in TIER_NAMES.items() if tier != TIER_CAMERA}

    def finite_rows(select: Callable[[int, list[npt.NDArray[Any]]], Bools], *names: str) -> Points:
        def stream() -> Iterator[F64]:
            for start, arrays in store.rows(*names):
                xyz = store.xyz(start, start + arrays[0].shape[0]).astype(np.float64)
                chosen = select(start, arrays) & np.isfinite(xyz).all(axis=1)
                yield xyz[chosen]

        return stream

    # What a Refine trains inside, and what the percentages are over: the extent the data
    # supports, not the region the cameras pointed at (`roi`, which stays the reference
    # for pixel size and the capture tips).
    extent = extent_of(finite_rows(lambda _, a: a[0] == TIER_KEEP, "tiers"), roi)
    # The region a Refine trains in: the voxels holding the keep tier, in whatever shape
    # they make -- the keep that held-out frames verified, when there is enough of it to
    # be a region (a capture whose held-out frames saw only part of the subject would
    # otherwise refine only that part). `extent` is only the frame the percentages below
    # are measured over.
    mask_source = "keep"
    mask_bit = 0
    if held is not None and tally.verified >= max(
        MIN_VERIFIED_FOR_MASK, VERIFIED_MASK_SHARE * tally.tiers[TIER_KEEP]
    ):
        mask_source, mask_bit = "verified-keep", FLAG_VERIFIED

    def mask_points() -> Iterator[tuple[F64, F64]]:
        for start, (tiers, bits, gsd) in store.rows("tiers", "flags", "gsd"):
            chosen = (bits & mask_bit) > 0 if mask_bit else tiers == TIER_KEEP
            xyz = store.xyz(start, start + tiers.shape[0])
            yield xyz[chosen].astype(np.float64), gsd[chosen].astype(np.float64)

    # Voxels sized from the keep tier's pixel footprint, not the scene's extent (see
    # `support_mask`'s module docstring), so a building's mask is as tight as a table's.
    mask = support_mask.build_from(support_mask.StreamedPoints(mask_points))
    if mask is not None and mask.sizing is not None:
        sizing = mask.sizing
        ctx.log(
            f"support mask: {mask.voxels} voxels of {mask.voxel:.4g} in {mask.runs} runs "
            f"(footprint {sizing.footprint:.4g} x {sizing.multiple:g}, "
            f"{sizing.spacing_steps} spacing step(s)"
            + ("" if sizing.bound is None else f"; bound by {sizing.bound}")
            + ")"
        )
    scene = replace(extent, radius=extent.radius * 1.5)

    def near_points() -> Iterator[tuple[F64, F64]]:
        for start, (views, gsd) in store.rows("views", "gsd"):
            xyz = store.xyz(start, start + views.shape[0])
            near = (views > 0) & scene.contains(xyz)
            yield xyz[near].astype(np.float64), gsd[near].astype(np.float64)

    share_voxel, share_basis = _share_voxel(mask, support_mask.StreamedPoints(near_points), scene)

    def tier_rows(verified_only: bool) -> TierRows:
        def stream() -> Iterator[tuple[F32, U8, F32]]:
            for start, (alpha, tiers, bits) in store.rows("alpha", "tiers", "flags"):
                xyz = store.xyz(start, start + alpha.shape[0])
                if verified_only:
                    tiers = np.where((bits & FLAG_VERIFIED) > 0, TIER_KEEP, TIER_DROP).astype(
                        np.uint8
                    )
                yield xyz, tiers, alpha

        return stream

    keep_pct = occupied_share_of(tier_rows(False), scene, TIER_KEEP, share_voxel, count)
    context_pct = occupied_share_of(tier_rows(False), scene, TIER_CONTEXT, share_voxel, count)
    # The same share, counting only keep that held-out frames confirmed.
    keep_verified_pct = (
        None
        if held is None
        else occupied_share_of(tier_rows(True), scene, TIER_KEEP, share_voxel, count)
    )
    metrics_doc = (
        _read_json(ctx.input("train_metrics.json")) if ctx.has_input("train_metrics.json") else {}
    )
    psnr = _round(_float_or_none(metrics_doc.get("psnr")), 2)
    up_estimate = sfm.camera_up(model)

    def roi_values() -> Iterator[I32]:
        for start, (views,) in store.rows("views"):
            inside = in_roi_of(start, start + views.shape[0], views)
            yield views[inside]

    def roi_spread() -> Iterator[F32]:
        for start, (views, spread) in store.rows("views", "spread"):
            yield spread[in_roi_of(start, start + views.shape[0], views)]

    roi_views_median = outofcore.median(roi_values)
    roi_spread_median = outofcore.median(roi_spread)
    roi_stats = RoiStats(
        count=tally.roi,
        short_views=tally.short_views,
        short_spread=tally.short_spread,
        short_gsd=tally.short_gsd,
        median_views=None if roi_views_median is None else int(roi_views_median),
    )
    tips, geometry = capture_tips(
        centres=cameras.centres,
        roi=roi,
        up=None if up_estimate is None else np.asarray(up_estimate.up, dtype=np.float64),
        roi_stats=roi_stats,
        thresholds=thresholds,
        keep_pct=keep_pct,
    )
    held_report: dict[str, object] = dict(held_status)
    if held is not None:
        held_report = _held_report(store, held, thresholds, reference, limit, tally)
        near_keep = tally.roi_coverage_keep
        demoted_share = float(tally.roi_demoted) / near_keep if near_keep else 0.0
        held_report["demotedShareNearSubject"] = round(demoted_share, 3)
        tip = holdout.accuracy_tip(held.summary, demoted_share)
        if tip is not None:
            # Accuracy is what failed, so it leads, and "well covered" is no longer true.
            tips = [tip, *(t for t in tips if t["id"] != "good")][:4]
    held_report["supportMaskFrom"] = mask_source
    metres = scale.metres
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
        "gaussians": {"in": count, "out": kept_count, **counts},
        "keepPct": keep_pct,
        "contextPct": context_pct,
        "keepPctNote": (
            "share of the occupied voxels (keepPctVoxel on a side, sized from the pixel "
            "footprint as the support mask's are) within 1.5x the supported extent whose "
            "gaussians are mostly keep (contextPct: keep or context)"
        ),
        "keepPctVoxel": {"size": share_voxel, "basis": share_basis},
        "heldOutPsnr": psnr,
        "heldOutPsnrNote": "gsplat's own held-out frames (every 8th), from train_metrics.json",
        # keepPct's measure, counting only keep that held-out frames confirmed; null when
        # accuracy was not measured (then every keep is on coverage alone).
        "keepVerifiedPct": keep_verified_pct,
        "heldOut": held_report,
        "views": {"medianRoi": roi_stats.median_views},
        "spreadDeg": {
            "medianRoi": _round(None if roi_spread_median is None else float(roi_spread_median), 1)
        },
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
        f"bar {applied}; {kept_count} gaussians out ({written} bytes); "
        f"keep {keep_pct}% of the ROI; {len(tips)} tip(s); {seconds:.1f} s"
    )
    metrics: dict[str, MetricValue] = {
        "mode": mode,
        "bar": applied,
        "gaussiansOut": kept_count,
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
    if held is not None:
        metrics["keepVerified"] = tally.verified
        metrics["demotedByHeldOut"] = tally.demoted
    if psnr is not None:
        metrics["heldOutPsnr"] = psnr
    summary = (
        f"{keep_pct:g}% of the scene met the high-quality bar"
        if keep_pct is not None
        else "nothing met the high-quality bar"
    )
    return StageOutcome(metrics=metrics, summary=summary)


def _held_report(
    store: _Store,
    held: holdout.HeldOutColumns,
    thresholds: Thresholds,
    reference: float | None,
    limit: float | None,
    tally: _Tally,
) -> dict[str, object]:
    """`holdout.tier_report`, from statistics gathered a chunk at a time: per tier, the
    median error of the measured exactly, and the evidence-weighted mean as float64 sums
    (a chunk's pairwise sum, then chunk by chunk -- which rounds differently from one
    pairwise sum over the whole tier only below the fifth decimal the report keeps)."""
    names = {tier: name for tier, name in TIER_NAMES.items() if tier != TIER_CAMERA}
    members = dict.fromkeys(names, 0)
    measured_counts = dict.fromkeys(names, 0)
    weighted = dict.fromkeys(names, 0.0)
    weights = dict.fromkeys(names, 0.0)

    def errors() -> Iterator[tuple[F32, npt.NDArray[np.int64]]]:
        for start, (tiers, bits) in store.rows("tiers", "flags"):
            error, _ = held.read(start, start + tiers.shape[0])
            measured = (bits & FLAG_MEASURED) > 0
            yield error[measured], tiers[measured].astype(np.int64)

    for start, (tiers, bits) in store.rows("tiers", "flags"):
        error, weight = held.read(start, start + tiers.shape[0])
        measured = (bits & FLAG_MEASURED) > 0
        for tier in names:
            mine = tiers == tier
            members[tier] += int(mine.sum())
            chosen = mine & measured
            measured_counts[tier] += int(chosen.sum())
            values = error[chosen]
            evidence = weight[chosen].astype(np.float64)
            weighted[tier] += float((values * evidence).sum())
            weights[tier] += float(evidence.sum())
    medians = outofcore.grouped_median(errors, list(names))
    stats = {
        name: holdout.TierStats(
            gaussians=members[tier],
            measured=measured_counts[tier],
            median_error=float(medians[tier]) if tier in medians else None,
            weighted_error=weighted[tier],
            weight=weights[tier],
        )
        for tier, name in names.items()
    }
    return holdout.report(
        stats,
        summary=held.summary,
        reference=reference,
        limit=limit,
        verified=tally.verified,
        kept=tally.tiers[TIER_KEEP],
        demoted=tally.demoted,
        max_ratio=thresholds.keep_max_holdout_error_ratio,
        max_abs=thresholds.keep_max_holdout_error,
        min_weight=thresholds.holdout_min_weight,
    )


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
