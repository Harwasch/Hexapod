"""Metres per model unit for a capture with no metric scale, from how high a phone is held.

A reconstruction from images alone is right up to scale: COLMAP normalises each model so
the camera path is about ten units across, whatever it was in metres, and a phone video
carries no per-frame GPS to align it to. Drawn at one unit to the metre, a cable spool
filmed from a couple of metres comes out two or three times its size and a field walked
across comes out several times its. What the footage *does* fix is a ratio: how high the
camera was above the ground, in the model's own units. People hold a phone at about
`HANDHELD_HEIGHT_M` -- standing, at chest to eye height -- so that ratio is a scale:

    metres per unit = 1.5 m / (median camera height above the ground, in units)

It is an estimate, and everything here is about saying how good a one it is and refusing
it when the capture does not support it:

* **the ground under each camera is the capture's own.** The reconstruction's points are
  binned into a horizontal grid in the levelled frame and each cell's ground is a low
  percentile of its points' heights -- the statistic `gaussians.ground_samples` rests a
  splat on the terrain with (5th percentile, at least 8 points), so one stray point under
  the floor is not the floor. A camera's height is its own up-coordinate minus the ground
  of the nearest populated cell within one camera height (two cells) of where it stood; a
  camera with no ground near it is skipped rather than guessed for. Per camera, not one
  plane, so a capture walked up a slope is measured against the ground it was over.
* **the cell is sized from the capture, not in metres**, because there are no metres yet:
  half the camera height -- first the rough one (the cameras' median up-coordinate over
  the points' 5th percentile), then the median the first pass measured. Fine enough to
  follow a slope, coarse enough that a cell of sparse points has floor in it.
* **the uncertainty is stated, not implied.** Handheld means anything from about 1.2 m to
  1.8 m (`HANDHELD_TOLERANCE_M` either side), which is ±20% on its own and does not average
  away however many frames there are: one person filmed the whole capture at one height.
  The spread of the per-camera heights -- interquartile range over median -- is added in
  quadrature as half of itself, because heights that disagree are a crouch, a slope the
  grid missed or a ground that was not ground, and the median inherits some of each.
* **None, rather than a number, when the capture cannot carry one:** fewer than
  `MIN_CAMERAS` cameras with ground under them, ground under fewer than
  `MIN_GROUND_SHARE` of them, more than `MAX_BELOW_SHARE` of them *under* the surface
  found (a canopy, a table, a ceiling -- not the ground), or a spread over `MAX_SPREAD`.
  The caller then keeps the scale unresolved, which is an honest answer; a confident
  wrong one is not.

What it cannot know: that the phone was handheld at all. A drone, a pole or a capture
filmed kneeling all break the prior, and nothing in the geometry says so. Nor that the low
surface within reach of the cameras is the floor: a table-top orbit whose reconstruction
has the table and none of the floor round it is refused only when the table is out of
reach, and measured from it when it is not. Hence an estimate wherever it travels, a
±% beside it, and an explicit scale that overrides it.

Pure numpy, no I/O: the `georeference` stage (`stages._frame_from_poses`) hands it COLMAP's
camera centres and sparse points, already levelled by the camera-up estimate.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]

#: How high a handheld phone is above the ground, in metres: a standing adult filming at
#: chest to eye height.
HANDHELD_HEIGHT_M = 1.5

#: Half the range people hold a phone at, in metres: 1.2 m to 1.8 m around the prior, so
#: ±20% before anything about the capture itself is measured.
HANDHELD_TOLERANCE_M = 0.3

#: Fewer cameras with ground under them than this is not a median worth quoting.
MIN_CAMERAS = 8

#: The share of the cameras offered that must have ground found near them. Below it, the
#: capture mostly did not see the ground it was standing on, and the cameras that did are
#: not evidence about the rest.
MIN_GROUND_SHARE = 0.5

#: The share of the cameras with ground that may be at or below it. A camera "under" the
#: surface found is under a canopy, a table top or a ceiling -- and when more than a quarter
#: are, the low surface the grid found is not the ground the prior is about.
MAX_BELOW_SHARE = 0.25

#: Interquartile range over median of the per-camera heights, above which the cameras or
#: the ground disagree too much for one height to describe them.
MAX_SPREAD = 0.5

#: Each cell's ground: this percentile of its points' heights, from at least
#: `MIN_CELL_POINTS` of them. `gaussians.ground_samples`'s statistic and defaults.
GROUND_PERCENTILE = 5.0
MIN_CELL_POINTS = 8


@dataclass(frozen=True)
class ScaleEstimate:
    """Metres per model unit from the camera height, and the evidence for it."""

    #: Metres per model unit: `prior_m / camera_height_units`.
    scale: float
    #: The median camera height above its ground, in model units.
    camera_height_units: float
    #: Interquartile range over median of the per-camera heights (0 when all agree).
    spread: float
    #: Cameras whose height went into the median.
    n_cameras: int
    #: The relative uncertainty of `scale`, in percent, either side.
    uncertainty_pct: float
    #: The handheld height assumed, in metres, and its tolerance either side.
    prior_m: float
    prior_tolerance_m: float
    #: Cameras offered, with or without ground found under them.
    cameras_offered: int
    #: The grid cell the ground was measured in, in model units.
    cell_units: float

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "camera-height",
            "metresPerUnit": round(self.scale, 6),
            "priorM": self.prior_m,
            "priorRangeM": [
                round(self.prior_m - self.prior_tolerance_m, 3),
                round(self.prior_m + self.prior_tolerance_m, 3),
            ],
            "cameraHeightUnits": round(self.camera_height_units, 6),
            "spread": round(self.spread, 4),
            "cameras": self.n_cameras,
            "camerasOffered": self.cameras_offered,
            "cellUnits": round(self.cell_units, 6),
            "uncertaintyPct": round(self.uncertainty_pct, 1),
            "note": (
                f"an estimate, not a measurement: assumes the phone was handheld about "
                f"{self.prior_m:g} m above the ground (a person holds one at "
                f"{self.prior_m - self.prior_tolerance_m:g}-"
                f"{self.prior_m + self.prior_tolerance_m:g} m), against the median height "
                f"of {self.n_cameras} cameras over the reconstruction's own ground"
            ),
        }


def estimate_metres_per_unit(
    camera_centres_up: npt.ArrayLike,
    ground_points: npt.ArrayLike,
    *,
    prior_m: float = HANDHELD_HEIGHT_M,
    prior_tolerance_m: float = HANDHELD_TOLERANCE_M,
    cell: float | None = None,
    percentile: float = GROUND_PERCENTILE,
    min_points: int = MIN_CELL_POINTS,
    min_cameras: int = MIN_CAMERAS,
    max_spread: float = MAX_SPREAD,
) -> ScaleEstimate | None:
    """Metres per model unit from how high the cameras were, or None when unreliable.

    `camera_centres_up` (N, 3) and `ground_points` (M, 3) are in the same levelled frame,
    +z up, in the reconstruction's own units. `cell` is the ground grid's side in those
    units; None sizes it at half the camera height (see the module docstring).
    """
    cameras = _finite(camera_centres_up)
    points = _finite(ground_points)
    offered = int(cameras.shape[0])
    if offered < min_cameras or points.shape[0] < min_points:
        return None
    measure = functools.partial(
        _heights_above_ground,
        cameras,
        points,
        percentile=percentile,
        min_points=min_points,
        min_cameras=min_cameras,
    )
    if cell is None:
        # Sized twice: from the rough height first, then from the height that measured.
        # The rough one is over the lowest ground anywhere in the model, so on a slope or
        # over a far background it is too tall, and a cell too wide reads more of a
        # slope's downhill side as the ground: on a 15% slope, 4.1% low sized once and
        # 2.5% low sized twice (tests/test_scale_estimate.py).
        rough = float(np.median(cameras[:, 2]) - np.percentile(points[:, 2], percentile))
        if not (math.isfinite(rough) and rough > 0.0):
            return None
        first = measure(cell=rough / 2.0)
        if first is None:
            return None
        cell = float(np.median(first)) / 2.0
    if not (math.isfinite(cell) and cell > 0.0):
        return None
    above = measure(cell=cell)
    if above is None:
        return None
    median = float(np.median(above))
    q1, q3 = (float(v) for v in np.percentile(above, [25.0, 75.0]))
    spread = (q3 - q1) / median
    if spread > max_spread:
        return None
    relative = math.hypot(prior_tolerance_m / prior_m, spread / 2.0)
    return ScaleEstimate(
        scale=prior_m / median,
        camera_height_units=median,
        spread=spread,
        n_cameras=int(above.size),
        uncertainty_pct=100.0 * relative,
        prior_m=prior_m,
        prior_tolerance_m=prior_tolerance_m,
        cameras_offered=offered,
        cell_units=cell,
    )


def _heights_above_ground(
    cameras: F64,
    points: F64,
    *,
    cell: float,
    percentile: float,
    min_points: int,
    min_cameras: int,
) -> F64 | None:
    """The positive camera heights, or None when too few cameras had ground under them
    or too many were under the surface found (see the module docstring)."""
    heights = camera_heights(
        cameras, points, cell=cell, radius=2.0 * cell, percentile=percentile, min_points=min_points
    )
    found = heights[np.isfinite(heights)]
    if found.size < max(min_cameras, MIN_GROUND_SHARE * cameras.shape[0]):
        return None
    below = found <= 0.0
    if float(below.mean()) > MAX_BELOW_SHARE:
        return None
    above: F64 = found[~below]
    return above if above.size >= min_cameras else None


def camera_heights(
    cameras: npt.ArrayLike,
    points: npt.ArrayLike,
    *,
    cell: float,
    radius: float,
    percentile: float = GROUND_PERCENTILE,
    min_points: int = MIN_CELL_POINTS,
) -> F64:
    """Each camera's height above the ground nearest it, NaN where there is none in reach.

    The ground is per grid cell -- `percentile` of the heights of a cell's points, for
    cells holding at least `min_points` -- and a camera is measured against the populated
    cell whose centre is nearest it horizontally, within `radius`.
    """
    centres = _finite(cameras)
    xyz = _finite(points)
    out = np.full(centres.shape[0], np.nan)
    if centres.shape[0] == 0 or xyz.shape[0] == 0:
        return out
    # Only the points that could be some camera's ground: the cameras' footprint, widened
    # by the reach and a cell. Keeps the grid small when the model has a far background.
    low = centres[:, :2].min(axis=0) - (radius + cell)
    high = centres[:, :2].max(axis=0) + (radius + cell)
    near = np.all((xyz[:, :2] >= low) & (xyz[:, :2] <= high), axis=1)
    xyz = xyz[near]
    if xyz.shape[0] == 0:
        return out
    ground = _cell_ground(xyz, low, cell, percentile, min_points)
    if not ground:
        return out
    # A cell `reach + 1` away has its centre at least `reach + 0.5` cells off: out of reach.
    reach = math.ceil(radius / cell)
    offsets = [(dx, dy) for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1)]
    for index, (x, y, z) in enumerate(centres):
        ix, iy = math.floor((x - low[0]) / cell), math.floor((y - low[1]) / cell)
        best = math.inf
        for dx, dy in offsets:
            surface = ground.get((ix + dx, iy + dy))
            if surface is None:
                continue
            cx = low[0] + (ix + dx + 0.5) * cell
            cy = low[1] + (iy + dy + 0.5) * cell
            distance = math.hypot(cx - x, cy - y)
            if distance <= radius and distance < best:
                best = distance
                out[index] = z - surface
    return out


def _cell_ground(
    xyz: F64, origin: F64, cell: float, percentile: float, min_points: int
) -> dict[tuple[int, int], float]:
    """`percentile` of each populated cell's heights, keyed by the cell's (ix, iy).

    numpy's default (linear) percentile, computed for every cell at once over the points
    sorted by cell and then height -- the same numbers as `np.percentile` cell by cell.
    """
    ix = np.floor((xyz[:, 0] - origin[0]) / cell).astype(np.int64)
    iy = np.floor((xyz[:, 1] - origin[1]) / cell).astype(np.int64)
    key = ix * (int(iy.max()) + 1) + iy
    order = np.lexsort((xyz[:, 2], key))
    _, starts, counts = np.unique(key[order], return_index=True, return_counts=True)
    keep = counts >= min_points
    if not bool(keep.any()):
        return {}
    starts, counts = starts[keep], counts[keep]
    heights = xyz[order, 2]
    virtual = (counts - 1) * (percentile / 100.0)
    below = np.floor(virtual).astype(np.int64)
    above = np.minimum(below + 1, counts - 1)
    fraction = virtual - below
    lower, upper = heights[starts + below], heights[starts + above]
    surface = lower + fraction * (upper - lower)
    rows = order[starts]
    return {
        (int(ix[row]), int(iy[row])): float(value) for row, value in zip(rows, surface, strict=True)
    }


def _finite(values: npt.ArrayLike) -> F64:
    array = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    finite: F64 = array[np.isfinite(array).all(axis=1)]
    return finite
