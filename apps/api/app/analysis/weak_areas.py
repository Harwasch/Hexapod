"""An epoch's weak cells as re-scan zones: smoothed patches on the globe, each with a way
to fly it (M4, ADR 0016 § API).

**Cells to patches.** The weak cells of the grid's finest level (weak-v1, the level its
summary is stated at) are merged into polygons in the tileset frame, where they are
squares of `cellM` metres east and north of the site frame's origin: closed by
`smoothing_m` with mitred joins (gaps and notches up to twice it bridged; the cells' right
angles stay right angles, so a block of cells keeps its exact area), simplified by half a
cell, parts under `min_area_m2` dropped, largest first. Then each vertex goes to the globe
through the frame's own east/north/up tangent plane (`site_projection`): PROJ's
`topocentric` conversion inverted, the exact inverse of the `enu_to_ecef` the pipeline
placed the tiles with, so a patch lies over the cells it was made of at any origin height.
(An azimuthal-equidistant projection at the origin is not that plane: it measures on the
ellipsoid, and at a 1,500 m origin it puts a point 1 km east 23 cm short.)

**Budget.** Over `budget_m2`, the weakest cells are kept first (unseen before scored, then
by score), then those nearest `focus`, and the patches are made again from what is kept
until they fit within one cell of it -- the same trimming as the Fog of Knowing's scout
(`knowledge.scout`).

**Patterns are UX heuristics, not tolerances.** How a patch is best flown again is a rule
of thumb from photogrammetry practice, stated here and labelled so in every hint; none of
its numbers was calibrated on a capture:

* mostly `views` / `spread` (seen from too few frames, or from too narrow a spread of
  directions): fly around it -- an `orbit` when it stands up (relief at least
  `ORBIT_MIN_RELIEF_M`) and is small enough to circle (at most `ORBIT_MAX_DIAMETER_M`
  across), else two crossing `oblique` passes;
* mostly `gsd` (seen too coarsely): `nadir`, lower, for at least the epoch's median GSD;
* mostly `unseen` / `accuracy` (not seen at all, or failing held-out frames): `nadir`;
* mostly `alignment` (the epoch's placement capped its scores): `nadir`, overlapping the
  reference area, so the next epoch can be aligned visually.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import shapely
from numpy.typing import ArrayLike, NDArray
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.validation import make_valid

from app.analysis.cell_grid import REASON_BITS, Plane

Pattern = Literal["nadir", "oblique", "orbit"]
Point = tuple[float, float]

#: UX heuristics (module docstring): an orbit suits what stands at least this tall ...
ORBIT_MIN_RELIEF_M = 3.0
#: ... and is at most this far across, so two rings at a stand-off close round it.
ORBIT_MAX_DIAMETER_M = 40.0
#: The camera pitch each pattern suggests, degrees below the horizon (also UX heuristics:
#: straight down; the common 45 degree oblique; 30 degrees for an orbit's side views).
GIMBAL_PITCH_DEG: Mapping[str, float] = {"nadir": -90.0, "oblique": -45.0, "orbit": -30.0}
#: Patches are closed by at least this much (a UX heuristic: a drone flies zones, not
#: quarter-metre cells), and parts under `MIN_AREA_M2` are not zones worth a flight.
MIN_SMOOTHING_M = 0.5
MIN_AREA_M2 = 1.0
HEURISTIC = "a rule of thumb, not a calibrated plan"

_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("alignment", ("alignment",)),
    ("views", ("views", "spread")),
    ("gsd", ("gsd",)),
    ("unseen", ("unseen", "accuracy")),
)


Coordinates = NDArray[np.float64]


@dataclass(frozen=True)
class SiteProjection:
    """The site frame's tangent plane: `to_geographic(x, y) -> (lon, lat)` for points east
    and north of the origin, and `to_metric(lon, lat) -> (x, y)` back; arrays or scalars."""

    to_geographic: Callable[[ArrayLike, ArrayLike], tuple[Coordinates, Coordinates]]
    to_metric: Callable[[ArrayLike, ArrayLike], tuple[Coordinates, Coordinates]]


def site_projection(lat: float, lon: float, height: float) -> SiteProjection:
    """The east/north/up plane at the frame origin (module docstring), on WGS84. The origin
    height is used as the pipeline placed the tiles with it (as ellipsoidal), whatever
    datum the frame records."""
    pipeline = (
        "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
        "+step +proj=cart +ellps=WGS84 "
        f"+step +proj=topocentric +ellps=WGS84 +lat_0={lat!r} +lon_0={lon!r} +h_0={height!r}"
    )
    forward = Transformer.from_pipeline(pipeline)

    def to_geographic(x: ArrayLike, y: ArrayLike) -> tuple[Coordinates, Coordinates]:
        east, north = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        lon_, lat_, _ = forward.transform(east, north, np.zeros_like(east), direction="INVERSE")
        return np.asarray(lon_, dtype=np.float64), np.asarray(lat_, dtype=np.float64)

    def to_metric(lon_: ArrayLike, lat_: ArrayLike) -> tuple[Coordinates, Coordinates]:
        # On the ellipsoid at the origin's height, where the plane's own up is 0.
        lons, lats = np.asarray(lon_, dtype=np.float64), np.asarray(lat_, dtype=np.float64)
        x, y, _ = forward.transform(lons, lats, np.full_like(lons, height))
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    return SiteProjection(to_geographic=to_geographic, to_metric=to_metric)


@dataclass(frozen=True)
class Patch:
    """One weak area: its outline on the globe and in the frame, and what made it weak."""

    footprint: Polygon
    metric: Polygon
    area_m2: float
    cells: int
    mean_score: float
    reasons: dict[str, int]
    relief_max_m: float
    diameter_m: float
    pattern: Pattern
    target_gsd_m: float | None
    gimbal_pitch_deg: float
    hint: str


@dataclass(frozen=True)
class Patches:
    patches: list[Patch]
    truncated_by_budget: bool
    weak_cells: int
    total_cells: int
    smoothing_m: float = 0.0


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if hasattr(geometry, "geoms"):
        return [polygon for part in geometry.geoms for polygon in _polygons(part)]
    return []


def _centres(
    plane: Plane, rows: NDArray[np.int64], columns: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    cell = plane.cell_m
    x = ((plane.i0 + columns + 0.5) * cell).astype(np.float64)
    y = ((plane.j0 + rows + 0.5) * cell).astype(np.float64)
    return x, y


def select_cells(
    plane: Plane,
    mask: NDArray[np.bool_],
    limit: int | None,
    focus_xy: Point | None,
) -> tuple[NDArray[np.bool_], bool]:
    """`mask`, or, over `limit` cells, the weakest of it first (unseen, then by score),
    then those nearest `focus_xy` (the mask's middle without one). Returns the cells and
    whether the limit cut them."""
    count = int(np.count_nonzero(mask))
    if limit is None or count <= limit:
        return mask, False
    rows, columns = np.nonzero(mask)
    score = plane.channel("score")[rows, columns]
    x, y = _centres(plane, rows, columns)
    if focus_xy is None:
        focus_xy = (float(x.mean()), float(y.mean()))
    distance = np.hypot(x - focus_xy[0], y - focus_xy[1])
    order = np.lexsort((distance, score))
    chosen = order[: max(0, limit)]
    out = np.zeros(mask.shape, dtype=bool)
    out[rows[chosen], columns[chosen]] = True
    return out, True


def mask_polygons(
    plane: Plane, mask: NDArray[np.bool_], *, smoothing_m: float, min_area_m2: float
) -> list[Polygon]:
    """The cells of `mask` as smoothed polygons in the tileset frame, largest first."""
    if not mask.any():
        return []
    cell = plane.cell_m
    # Row k of the plane is j = j0 + k, northward: a positive `e` keeps y increasing.
    affine = Affine(cell, 0.0, plane.i0 * cell, 0.0, cell, plane.j0 * cell)
    squares = [
        shape(geometry)
        for geometry, value in shapes(
            mask.astype(np.uint8), mask=mask, transform=affine, connectivity=4
        )
        if value == 1
    ]
    merged = unary_union(squares)
    smoothed = (
        merged.buffer(smoothing_m, join_style="mitre")
        .buffer(-smoothing_m, join_style="mitre")
        .simplify(cell / 2)
    )
    parts = [
        orient(part)
        for part in _polygons(make_valid(smoothed))
        if part.area >= max(min_area_m2, 2 * cell * cell)
    ]
    return sorted(parts, key=lambda part: part.area, reverse=True)


def group_counts(reasons: NDArray[np.int64]) -> dict[str, int]:
    """Per reason group (`alignment`, `views` = views or spread, `gsd`, `unseen` = unseen
    or accuracy), how many of these cells' `reasons` carry any of its bits."""
    out: dict[str, int] = {}
    for group, members in _GROUPS:
        bits = 0
        for name in members:
            bits |= REASON_BITS[name]
        out[group] = int(np.count_nonzero(reasons & bits))
    return out


def _dominant(groups: Mapping[str, int]) -> str | None:
    """The group carried by the most cells; ties go to the earlier group (the alignment
    cap first: it caps every score, so it is the one to fix first)."""
    best, best_count = None, 0
    for group, _ in _GROUPS:
        if groups.get(group, 0) > best_count:
            best, best_count = group, groups[group]
    return best


def pattern_for(
    groups: Mapping[str, int],
    relief_max_m: float,
    diameter_m: float,
    median_gsd_mm: float | None,
) -> tuple[Pattern, float | None, str]:
    """How a patch might be flown again (module docstring), its target GSD in metres
    (null unless it is the GSD that fell short, on a metric capture) and a hint. UX
    heuristics, labelled so."""
    group = _dominant(groups)
    if group == "views":
        standing = relief_max_m >= ORBIT_MIN_RELIEF_M and diameter_m <= ORBIT_MAX_DIAMETER_M
        if standing:
            return (
                "orbit",
                None,
                f"Seen from too few directions, and it stands {relief_max_m:.1f} m tall and "
                f"{diameter_m:.0f} m across: orbit it, two rings, camera tilted in "
                f"({HEURISTIC}).",
            )
        return (
            "oblique",
            None,
            "Seen from too few directions: fly two crossing passes with the camera tilted "
            f"({HEURISTIC}).",
        )
    if group == "gsd":
        target = median_gsd_mm / 1000.0 if median_gsd_mm is not None else None
        detail = (
            f" for {target * 100:.1f} cm a pixel or finer, the capture's median"
            if target is not None
            else " (the capture had no metric scale, so no target size)"
        )
        return (
            "nadir",
            target,
            f"Seen too coarsely: fly it lower, camera down{detail} ({HEURISTIC}).",
        )
    if group == "alignment":
        return (
            "nadir",
            None,
            "This epoch's placement capped its scores: fly it nadir, overlapping the "
            f"reference area, so the next epoch can be aligned visually ({HEURISTIC}).",
        )
    return "nadir", None, f"Not seen, or not seen well enough: fly it nadir ({HEURISTIC})."


def _tally(plane: Plane, cells: NDArray[np.bool_]) -> dict[str, int]:
    reasons = plane.channel("reasons")[cells]
    return {name: int(np.count_nonzero(reasons & bit)) for name, bit in REASON_BITS.items()}


def _diameter_m(polygon: Polygon) -> float:
    """How far across: the longer side of the polygon's smallest enclosing rectangle."""
    rectangle = polygon.minimum_rotated_rectangle
    if not isinstance(rectangle, Polygon):
        return 0.0
    xs, ys = rectangle.exterior.coords.xy
    sides = [math.hypot(xs[k + 1] - xs[k], ys[k + 1] - ys[k]) for k in range(len(xs) - 1)]
    return max(sides, default=0.0)


def _projected(
    projection: SiteProjection,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def apply(points: NDArray[np.float64]) -> NDArray[np.float64]:
        lon, lat = projection.to_geographic(points[:, 0], points[:, 1])
        return np.column_stack([lon, lat])

    return apply


def _to_globe(polygon: Polygon, projection: SiteProjection) -> Polygon:
    """The polygon's vertices through the site's tangent plane; the largest part should
    the projection make it more than one (it does not at site scale)."""
    globe = make_valid(shapely.transform(polygon, _projected(projection)))
    parts = _polygons(globe)
    if not parts:
        raise ValueError("a weak area did not survive projection to the globe")
    return orient(max(parts, key=lambda part: part.area))


def cell_patches(
    plane: Plane,
    mask: NDArray[np.bool_],
    projection: SiteProjection,
    *,
    max_areas: int = 3,
    budget_m2: float | None = None,
    focus: Point | None = None,
    median_gsd_mm: float | None = None,
    relief_unit_m: float = 0.25,
    total_cells: int | None = None,
) -> Patches:
    """The weak cells `mask` (over `plane`) as at most `max_areas` patches, together within
    `budget_m2` plus at most one cell (module docstring)."""
    cell = plane.cell_m
    cell_area = cell * cell
    smoothing = max(cell / 2, MIN_SMOOTHING_M)
    focus_xy: Point | None = None
    if focus is not None:
        fx, fy = projection.to_metric(focus[0], focus[1])
        focus_xy = (float(fx), float(fy))
    limit = None if budget_m2 is None else int(budget_m2 // cell_area)
    selected = mask
    polygons: list[Polygon] = []
    truncated = False
    for _ in range(8):
        selected, cut = select_cells(plane, mask, limit, focus_xy)
        truncated = truncated or cut
        polygons = mask_polygons(plane, selected, smoothing_m=smoothing, min_area_m2=MIN_AREA_M2)
        if len(polygons) > max_areas:
            polygons = polygons[:max_areas]
        total = sum(polygon.area for polygon in polygons)
        if budget_m2 is None or total <= budget_m2 + cell_area:
            break
        kept = int(np.count_nonzero(selected))
        limit = max(0, kept - math.ceil((total - budget_m2) / cell_area))
        truncated = True
    rows, columns = np.nonzero(selected)
    x, y = _centres(plane, rows, columns)
    score = plane.channel("score")
    relief = plane.channel("relief")
    out: list[Patch] = []
    for polygon in polygons:
        inside = shapely.contains_xy(polygon, x, y)
        cells = np.zeros(mask.shape, dtype=bool)
        cells[rows[inside], columns[inside]] = True
        count = int(np.count_nonzero(cells))
        scores = score[cells]
        scored = scores[scores > 0]
        reasons = _tally(plane, cells)
        relief_max = float(relief[cells].max(initial=0)) * relief_unit_m
        diameter = _diameter_m(polygon)
        groups = group_counts(plane.channel("reasons")[cells])
        pattern, target, hint = pattern_for(groups, relief_max, diameter, median_gsd_mm)
        out.append(
            Patch(
                footprint=_to_globe(polygon, projection),
                metric=polygon,
                area_m2=float(polygon.area),
                cells=count,
                mean_score=float(scored.mean()) if scored.size else 0.0,
                reasons=reasons,
                relief_max_m=relief_max,
                diameter_m=diameter,
                pattern=pattern,
                target_gsd_m=target,
                gimbal_pitch_deg=GIMBAL_PITCH_DEG[pattern],
                hint=hint,
            )
        )
    return Patches(
        patches=out,
        truncated_by_budget=truncated,
        weak_cells=int(np.count_nonzero(mask)),
        total_cells=int(total_cells if total_cells is not None else plane.cells),
        smoothing_m=smoothing,
    )
