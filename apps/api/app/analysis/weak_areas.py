"""An epoch's weak cells as re-scan zones: smoothed patches on the globe, each with a way
to fly it (M4, ADR 0016 § API).

**Cells to patches.** The weak cells of the grid's finest level (weak-v1, the level its
summary is stated at) are squares of `cellM` metres east and north of the site frame's
origin in the tileset frame. They are closed on the raster by a square of whole cells
(at least `MIN_SMOOTHING_M` across each way: exactly what a mitred buffer out and back in
does to cells, so gaps up to twice it are bridged and a block keeps its exact area),
outlined region by region, simplified by half a cell, and parts under `MIN_AREA_M2` are
dropped, largest first. Each patch's numbers are of the finest cells whose centres it
holds. A plane over `MAX_ZONE_CELLS` is drawn a level or more up, a coarse cell weak when
at least half of the children the epoch speaks for are (`zone_plane`): outlining every
region of a noisy 2^22-cell plane took minutes.

Each vertex then goes to the globe through the frame's own east/north/up tangent plane
(`site_projection`): PROJ's `topocentric` conversion inverted, the exact inverse of the
`enu_to_ecef` the pipeline placed the tiles with, so a patch lies over the cells it was
made of at any origin height. (An azimuthal-equidistant projection at the origin is not
that plane: it measures on the ellipsoid, and at a 1,500 m origin it puts a point 1 km
east 23 cm short.)

**Budget.** Over `budget_m2`, the weakest cells are kept first (unseen before scored, then
by score), then those nearest `focus`, and the patches are made again from what is kept
until they fit within one cell of it -- the same trimming as the Fog of Knowing's scout
(`knowledge.scout`); should closing still leave them over, only the patches that fit are
kept, largest first, so the budget holds.

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
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.validation import make_valid

from app.analysis.cell_grid import LATTICE_BASE_M, REASON_BITS, Plane, covered_mask

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
#: Zones are drawn on at most this many cells (1024 x 1024): outlining a noisy plane costs
#: more than linearly in its regions, and a grid at its 2^22-cell ceiling took minutes.
#: Past it they are drawn a level up, by majority (`zone_plane`).
MAX_ZONE_CELLS = 1 << 20
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
    #: The level the zones were drawn at (the grid's finest unless it was too large).
    zone_level: int = 0


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if hasattr(geometry, "geoms"):
        return [polygon for part in geometry.geoms for polygon in _polygons(part)]
    return []


def _affine(level_cell: float, i0: int, j0: int) -> Affine:
    # Row k of a plane is j = j0 + k, northward: a positive `e` keeps y increasing.
    return Affine(level_cell, 0.0, i0 * level_cell, 0.0, level_cell, j0 * level_cell)


@dataclass(frozen=True)
class ZonePlane:
    """The weak cells at the level zones are drawn at: the grid's finest when it holds at
    most `MAX_ZONE_CELLS`, else coarser (`zone_plane`). `order` ranks a cell for a budget:
    its weakest child's score (0 for unseen), large where nothing is weak."""

    level: int
    i0: int
    j0: int
    weak: NDArray[np.bool_]
    order: NDArray[np.int64]

    @property
    def cell_m(self) -> float:
        return math.ldexp(LATTICE_BASE_M, self.level)

    @property
    def shape(self) -> tuple[int, int]:
        nj, ni = self.weak.shape
        return nj, ni


_NOT_WEAK = np.int64(1) << 40


def _block(
    values: NDArray[np.int64], i0: int, j0: int, fill: int
) -> tuple[NDArray[np.int64], int, int]:
    """`values` padded to whole parents (`fill` outside) and shaped `(nj, 2, ni, 2)`."""
    nj, ni = values.shape
    pi0, pj0 = i0 >> 1, j0 >> 1
    pni, pnj = ((i0 + ni - 1) >> 1) - pi0 + 1, ((j0 + nj - 1) >> 1) - pj0 + 1
    padded = np.full((2 * pnj, 2 * pni), fill, dtype=np.int64)
    di, dj = i0 - 2 * pi0, j0 - 2 * pj0
    padded[dj : dj + nj, di : di + ni] = values
    return padded.reshape(pnj, 2, pni, 2), pi0, pj0


def zone_plane(plane: Plane, mask: NDArray[np.bool_]) -> ZonePlane:
    """`mask` at the level zones are drawn at. Past `MAX_ZONE_CELLS` it goes up a level at
    a time, a coarse cell weak when at least half of the children the epoch speaks for are
    weak -- weak-v1's own majority rule, one level up (a UX choice for drawing zones: the
    min-of-children pyramid would call a whole coarse cell weak for one weak child)."""
    weak = mask.astype(np.int64)
    covered = covered_mask(plane).astype(np.int64)
    order = np.where(mask, plane.channel("score"), _NOT_WEAK)
    level, i0, j0 = plane.level, plane.i0, plane.j0
    zone = mask
    while zone.size > MAX_ZONE_CELLS:
        w, pi0, pj0 = _block(weak, i0, j0, 0)
        c, _, _ = _block(covered, i0, j0, 0)
        o, _, _ = _block(order, i0, j0, int(_NOT_WEAK))
        weak, covered = w.sum(axis=(1, 3)), c.sum(axis=(1, 3))
        zone = (covered > 0) & (2 * weak >= covered)
        order = np.where(zone, o.min(axis=(1, 3)), _NOT_WEAK)
        level, i0, j0 = level + 1, pi0, pj0
    return ZonePlane(level, i0, j0, zone, order)


def select_cells(
    zone: ZonePlane, mask: NDArray[np.bool_], limit: int | None, focus_xy: Point | None
) -> tuple[NDArray[np.bool_], bool]:
    """`mask` (over `zone`), or, over `limit` cells, the weakest of it first (unseen, then
    by score), then those nearest `focus_xy` (the mask's middle without one). Returns the
    cells and whether the limit cut them."""
    count = int(np.count_nonzero(mask))
    if limit is None or count <= limit:
        return mask, False
    rows, columns = np.nonzero(mask)
    cell = zone.cell_m
    x = (zone.i0 + columns + 0.5) * cell
    y = (zone.j0 + rows + 0.5) * cell
    if focus_xy is None:
        focus_xy = (float(x.mean()), float(y.mean()))
    distance = np.hypot(x - focus_xy[0], y - focus_xy[1])
    order = np.lexsort((distance, zone.order[rows, columns]))
    chosen = order[: max(0, limit)]
    out = np.zeros(mask.shape, dtype=bool)
    out[rows[chosen], columns[chosen]] = True
    return out, True


def _box_any(mask: NDArray[np.bool_], k: int) -> NDArray[np.bool_]:
    """Square dilation by `k` cells (a `2k+1` square), by a summed-area table."""
    if k == 0:
        return mask
    nj, ni = mask.shape
    table = np.zeros((nj + 1, ni + 1), dtype=np.int64)
    table[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
    rows, columns = np.arange(nj), np.arange(ni)
    r0, r1 = np.clip(rows - k, 0, nj), np.clip(rows + k + 1, 0, nj)
    c0, c1 = np.clip(columns - k, 0, ni), np.clip(columns + k + 1, 0, ni)
    total = (
        table[np.ix_(r1, c1)]
        - table[np.ix_(r0, c1)]
        - table[np.ix_(r1, c0)]
        + table[np.ix_(r0, c0)]
    )
    out: NDArray[np.bool_] = total > 0
    return out


def close_mask(mask: NDArray[np.bool_], k: int) -> NDArray[np.bool_]:
    """Morphological closing by a `2k+1` square, in a window grown by `k` on every side
    and cropped back: on cells, exactly what a mitred buffer out and back in by `k` cells
    does to their outline (gaps and notches up to `2k` cells bridged, right angles kept)."""
    if k == 0:
        return mask
    grown = np.pad(mask, k)
    closed = ~_box_any(~_box_any(grown, k), k)
    out: NDArray[np.bool_] = closed[k:-k, k:-k]
    return out


def mask_polygons(
    zone: ZonePlane, mask: NDArray[np.bool_], *, smoothing_m: float, min_area_m2: float
) -> list[Polygon]:
    """The cells of `mask` (over `zone`) as smoothed polygons in the tileset frame,
    largest first: closed by `smoothing_m` in whole cells, outlined region by region,
    simplified by half a cell, parts under `min_area_m2` (and two cells) dropped."""
    if not mask.any():
        return []
    cell = zone.cell_m
    closed = close_mask(mask, math.floor(smoothing_m / cell + 1e-9))
    affine = _affine(cell, zone.i0, zone.j0)
    smallest = max(min_area_m2, 2 * cell * cell)
    parts: list[Polygon] = []
    for geometry, value in shapes(
        closed.astype(np.uint8), mask=closed, transform=affine, connectivity=4
    ):
        if value != 1:
            continue
        outline = make_valid(shape(geometry).simplify(cell / 2))
        parts.extend(orient(part) for part in _polygons(outline) if part.area >= smallest)
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
    `budget_m2` plus at most one zone cell (module docstring). Each patch's counts are of
    `plane`'s own weak cells whose centres it holds."""
    zone = zone_plane(plane, mask)
    cell = zone.cell_m
    cell_area = cell * cell
    smoothing = max(cell / 2, MIN_SMOOTHING_M)
    focus_xy: Point | None = None
    if focus is not None:
        fx, fy = projection.to_metric(focus[0], focus[1])
        focus_xy = (float(fx), float(fy))
    limit = None if budget_m2 is None else int(budget_m2 // cell_area)
    selected = zone.weak
    polygons: list[Polygon] = []
    truncated = False
    for _ in range(8):
        selected, cut = select_cells(zone, zone.weak, limit, focus_xy)
        truncated = truncated or cut
        polygons = mask_polygons(zone, selected, smoothing_m=smoothing, min_area_m2=MIN_AREA_M2)
        polygons = polygons[:max_areas]
        total = sum(polygon.area for polygon in polygons)
        if budget_m2 is None or total <= budget_m2 + cell_area:
            break
        kept = int(np.count_nonzero(selected))
        limit = max(0, kept - math.ceil((total - budget_m2) / cell_area))
        truncated = True
    if budget_m2 is not None and sum(p.area for p in polygons) > budget_m2 + cell_area:
        # Closing can grow a sparse selection faster than trimming shrinks it: keep, largest
        # first, the patches that still fit, so the budget is a promise.
        fitting: list[Polygon] = []
        for polygon in polygons:
            if sum(p.area for p in fitting) + polygon.area <= budget_m2 + cell_area:
                fitting.append(polygon)
        polygons = fitting
    score = plane.channel("score")
    relief = plane.channel("relief")
    reasons_plane = plane.channel("reasons")
    affine = _affine(plane.cell_m, plane.i0, plane.j0)
    out: list[Patch] = []
    for polygon in polygons:
        # The plane's weak cells whose centres are inside (rasterize's own rule).
        inside = rasterize(
            [(polygon, 1)], out_shape=(plane.nj, plane.ni), transform=affine, dtype=np.uint8
        ).astype(bool)
        cells = inside & mask
        count = int(np.count_nonzero(cells))
        scores = score[cells]
        scored = scores[scores > 0]
        relief_max = float(relief[cells].max(initial=0)) * relief_unit_m
        diameter = _diameter_m(polygon)
        groups = group_counts(reasons_plane[cells])
        pattern, target, hint = pattern_for(groups, relief_max, diameter, median_gsd_mm)
        out.append(
            Patch(
                footprint=_to_globe(polygon, projection),
                metric=polygon,
                area_m2=float(polygon.area),
                cells=count,
                mean_score=float(scored.mean()) if scored.size else 0.0,
                reasons=_tally(plane, cells),
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
        zone_level=zone.level,
    )
