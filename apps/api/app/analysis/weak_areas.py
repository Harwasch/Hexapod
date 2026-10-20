"""An epoch's weak cells as re-scan zones: smoothed patches on the globe, each with a way
to fly it (M4, ADR 0016 § API).

**Cells to patches.** The weak cells of the grid's finest level (weak-v1, the level its
summary is stated at) are squares of `cellM` metres east and north of the site frame's
origin in the tileset frame. They are closed on the raster by a square of whole cells
(at least `MIN_SMOOTHING_M` across each way: exactly what a mitred buffer out and back in
does to cells, so gaps up to twice it are bridged and a block keeps its exact area),
outlined region by region with holes under the smallest zone filled, simplified by half a
cell, and parts under `MIN_AREA_M2` are dropped, largest first. Each patch's numbers are
of the finest cells whose centres it holds. A plane over `MAX_ZONE_CELLS`, or whose closed
outline is over `MAX_ZONE_EDGES` cell edges, is drawn a level or more up, a coarse cell
weak when at least half of the children the epoch speaks for are (`zone_plane`):
outlining a noisy 2^22-cell plane cell by cell took minutes.

Each vertex then goes to the globe through the frame's own east/north/up tangent plane
(`site_projection`): PROJ's `topocentric` conversion inverted, the exact inverse of the
`enu_to_ecef` the pipeline placed the tiles with, so a patch lies over the cells it was
made of at any origin height. (An azimuthal-equidistant projection at the origin is not
that plane: it measures on the ellipsoid, and at a 1,500 m origin it puts a point 1 km
east 23 cm short.)

**The antimeridian.** A zone that crosses longitude 180 is cut there in the site plane,
into a zone each side (RFC 7946 3.1.9), and each part's longitudes are unwrapped about the
origin's and put back in [-180, 180] as a whole, so no zone is drawn the long way round.

**Budget.** Over `budget_m2`, whole zones are taken in rank order -- the one under
`focus` first, then the largest, then the nearest `focus` -- while they fit (within one
zone cell); the first that does not is clipped to a disc about the focus or its weakest
cell (unseen first, then by score), grown until it holds what is left. So a budget is a promise, it is
spent on the zones that rank first, and any budget that can hold a zone gets one.

**Every zone holds a weak cell.** The answer is the best non-empty set, never a refusal: a
zone that holds no weak cell's centre (clipped by the boundary down to ground closing
bridged) is skipped, and a disc about a focus on that strong ground is grown again about
the zone's weakest cell. A zone that still holds none is dropped, so `cells` is never 0.
(Trimming cells and outlining again, as the fog scout does, left nothing: closing grows
scattered cells into zones larger than themselves, faster than the trimming shrank them.)

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
import shapely.ops
from numpy.typing import ArrayLike, NDArray
from pyproj import Transformer
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.validation import make_valid

from app.analysis.cell_grid import LATTICE_BASE_M, MAX_LEVEL, REASON_BITS, Plane, covered_mask

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
#: ... and with at most this many cell edges in their outline (holes included): outlining
#: is linear in it, and a dense noisy plane is one region with a hole per strong cell.
MAX_ZONE_EDGES = 1 << 18
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
    #: The origin's longitude: where a zone's longitudes are unwrapped about (`_to_globe`).
    origin_lon: float = 0.0


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

    return SiteProjection(to_geographic=to_geographic, to_metric=to_metric, origin_lon=lon)


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
    #: `weak` closed by `smoothing_cells(level)`: what is outlined.
    closed: NDArray[np.bool_]

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


def smoothing_cells(level: int) -> int:
    """How many whole cells a zone is closed by at `level`: at least `MIN_SMOOTHING_M`."""
    return math.floor(MIN_SMOOTHING_M / math.ldexp(LATTICE_BASE_M, level) + 1e-9)


def outline_edges(mask: NDArray[np.bool_]) -> int:
    """Cell edges between a cell of `mask` and one outside it (the frame's border counts):
    what outlining it costs, holes included."""
    padded = np.pad(mask, 1)
    return int(
        np.count_nonzero(padded[:, 1:] != padded[:, :-1])
        + np.count_nonzero(padded[1:, :] != padded[:-1, :])
    )


def zone_plane(plane: Plane, mask: NDArray[np.bool_]) -> ZonePlane:
    """`mask` at the level zones are drawn at, closed (`ZonePlane.closed`). It goes up a
    level at a time while it is over `MAX_ZONE_CELLS` cells or its closed outline is over
    `MAX_ZONE_EDGES` cell edges -- a dense, noisy plane is one region with a hole per strong
    cell, which took minutes to outline -- a coarse cell weak when at least half of the
    children the epoch speaks for are weak: weak-v1's own majority rule, one level up (a UX
    choice for drawing zones: the min-of-children pyramid would call a whole coarse cell
    weak for one weak child)."""
    weak = mask.astype(np.int64)
    covered = covered_mask(plane).astype(np.int64)
    order = np.where(mask, plane.channel("score"), _NOT_WEAK)
    level, i0, j0 = plane.level, plane.i0, plane.j0
    zone = mask
    closed = close_mask(zone, smoothing_cells(level))
    while level < MAX_LEVEL and (
        zone.size > MAX_ZONE_CELLS or outline_edges(closed) > MAX_ZONE_EDGES
    ):
        w, pi0, pj0 = _block(weak, i0, j0, 0)
        c, _, _ = _block(covered, i0, j0, 0)
        o, _, _ = _block(order, i0, j0, int(_NOT_WEAK))
        weak, covered = w.sum(axis=(1, 3)), c.sum(axis=(1, 3))
        zone = (covered > 0) & (2 * weak >= covered)
        order = np.where(zone, o.min(axis=(1, 3)), _NOT_WEAK)
        level, i0, j0 = level + 1, pi0, pj0
        closed = close_mask(zone, smoothing_cells(level))
    return ZonePlane(level, i0, j0, zone, order, closed)


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


def zone_polygons(zone: ZonePlane, *, min_area_m2: float) -> list[Polygon]:
    """The zone's closed cells as polygons in the tileset frame, largest first: outlined
    region by region, holes under the smallest zone filled (a re-scan flies over a pinhole
    of strong cells anyway), simplified by half a cell, parts under `min_area_m2` (and two
    cells) dropped -- regions too small are dropped before any geometry is built."""
    if not zone.closed.any():
        return []
    cell = zone.cell_m
    affine = _affine(cell, zone.i0, zone.j0)
    smallest = max(min_area_m2, 2 * cell * cell)
    parts: list[Polygon] = []
    for geometry, value in shapes(
        zone.closed.astype(np.uint8), mask=zone.closed, transform=affine, connectivity=4
    ):
        if value != 1:
            continue
        rings = geometry["coordinates"]
        exterior = Polygon(rings[0])
        if exterior.area < smallest:
            continue
        holes = [ring for ring in rings[1:] if Polygon(ring).area >= smallest]
        outline = make_valid(Polygon(rings[0], holes).simplify(cell / 2))
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


def _unwrapped(lon: NDArray[np.float64], origin_lon: float) -> NDArray[np.float64]:
    """Longitudes made continuous about the origin's: within 180 degrees of it, so a zone
    beside the antimeridian is one run of numbers, past 180 or under -180 if need be."""
    out: NDArray[np.float64] = origin_lon + (np.mod(lon - origin_lon + 180.0, 360.0) - 180.0)
    return out


def _projected(
    projection: SiteProjection,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def apply(points: NDArray[np.float64]) -> NDArray[np.float64]:
        lon, lat = projection.to_geographic(points[:, 0], points[:, 1])
        return np.column_stack([_unwrapped(lon, projection.origin_lon), lat])

    return apply


#: How far past the antimeridian a cut vertex may land from rounding, degrees (~1 um).
_ON_THE_CUT = 1e-11


def split_at_antimeridian(polygon: Polygon, projection: SiteProjection) -> list[Polygon]:
    """`polygon` (site-frame metres) cut where it crosses longitude 180, so each part lies
    on one side: one part when it does not cross. The meridian is drawn into the plane
    through the projection itself, sampled over the polygon's latitudes and past them."""
    if polygon.is_empty:
        return []
    xs, ys = np.asarray(polygon.exterior.coords).T
    lon, lat = projection.to_geographic(xs, ys)
    unwrapped = _unwrapped(lon, projection.origin_lon)
    if unwrapped.max() <= 180.0 + _ON_THE_CUT and unwrapped.min() >= -180.0 - _ON_THE_CUT:
        return [polygon]
    margin = max(float(lat.max() - lat.min()), 1e-6)
    lats = np.linspace(float(lat.min()) - margin, float(lat.max()) + margin, 65)
    mx, my = projection.to_metric(np.full_like(lats, 180.0), lats)
    cut = shapely.LineString(np.column_stack([mx, my]))
    parts = _polygons(shapely.ops.split(polygon, cut))
    return [orient(part) for part in parts] if parts else [polygon]


def _to_globe(polygon: Polygon, projection: SiteProjection) -> Polygon:
    """The polygon's vertices through the site's tangent plane, longitudes unwrapped about
    the origin's and then put back in [-180, 180] as a whole (a part never straddles 180:
    `split_at_antimeridian` cut it there), so a zone is never drawn the long way round.
    The largest part should the projection make it more than one (it does not at site
    scale)."""
    globe = make_valid(shapely.transform(polygon, _projected(projection)))
    parts = _polygons(globe)
    if not parts:
        raise ValueError("a weak area did not survive projection to the globe")
    part = max(parts, key=lambda each: each.area)
    middle = part.centroid.x
    turn = -360.0 if middle > 180.0 else 360.0 if middle < -180.0 else 0.0

    def back(points: NDArray[np.float64]) -> NDArray[np.float64]:
        lon = np.clip(points[:, 0] + turn, -180.0, 180.0)
        return np.column_stack([lon, points[:, 1]])

    return orient(shapely.transform(part, back))


def _ranked(polygons: list[Polygon], focus_xy: Point | None) -> list[Polygon]:
    """Zones in the order a budget takes them: the one under the focus first, then the
    largest (the most weak cells), then the nearest the focus."""
    if focus_xy is None:
        return list(polygons)
    point = ShapelyPoint(focus_xy)
    return sorted(
        polygons,
        key=lambda polygon: (not polygon.covers(point), -polygon.area, polygon.distance(point)),
    )


def _weak_inside(zone: ZonePlane, polygon: Polygon) -> NDArray[np.bool_]:
    """The zone plane's weak cells whose centres `polygon` holds (rasterize's own rule)."""
    nj, ni = zone.weak.shape
    inside: NDArray[np.bool_] = (
        rasterize(
            [(polygon, 1)],
            out_shape=(nj, ni),
            transform=_affine(zone.cell_m, zone.i0, zone.j0),
            dtype=np.uint8,
        ).astype(bool)
        & zone.weak
    )
    return inside


def holds_weak(zone: ZonePlane, polygon: Polygon) -> bool:
    """Whether `polygon` holds the centre of at least one weak cell of the zone plane."""
    return bool(_weak_inside(zone, polygon).any())


def _anchor(zone: ZonePlane, polygon: Polygon, focus_xy: Point | None) -> Point:
    """Where a zone too large for the budget is clipped about: the focus when it is in the
    zone, else its weakest cell (unseen first, then by score) nearest its middle."""
    if focus_xy is not None and polygon.covers(ShapelyPoint(focus_xy)):
        return focus_xy
    cell = zone.cell_m
    inside = _weak_inside(zone, polygon)
    rows, columns = np.nonzero(inside)
    if rows.size == 0:
        middle = polygon.representative_point()
        return float(middle.x), float(middle.y)
    x = (zone.i0 + columns + 0.5) * cell
    y = (zone.j0 + rows + 0.5) * cell
    middle = polygon.centroid
    order = np.lexsort((np.hypot(x - middle.x, y - middle.y), zone.order[rows, columns]))
    return float(x[order[0]]), float(y[order[0]])


def _clip(
    zone: ZonePlane, polygon: Polygon, area_m2: float, focus_xy: Point | None
) -> Polygon | None:
    """The part of `polygon` within a disc about its anchor (`_anchor`) grown until it
    holds `area_m2` (a bisection on the radius, to a hundredth of the budget), or None when
    that is no zone (under the smallest, or holding no weak cell: a disc about a focus on
    the strong ground closing bridged is tried again about the weakest cell, and refused
    only when that holds none either). A disc, not a square: grown about a weakest cell
    at a zone's edge, or into a ring-shaped zone, it follows the zone instead of covering
    the strong ground beside it (a square about e2's edge cell held 36 m2 of 100)."""
    found = _clip_about(zone, polygon, area_m2, _anchor(zone, polygon, focus_xy))
    if found is not None and not holds_weak(zone, found):
        # The focus sat on strong ground the zone's closing bridged: the disc about it held
        # no weak cell. Clip about the zone's weakest cell instead, which always holds one.
        found = _clip_about(zone, polygon, area_m2, _anchor(zone, polygon, None))
    return found if found is not None and holds_weak(zone, found) else None


def _clip_about(zone: ZonePlane, polygon: Polygon, area_m2: float, anchor: Point) -> Polygon | None:
    """`_clip`'s disc about `anchor`, grown until it holds `area_m2`; None when that is
    under the smallest zone."""
    smallest = max(MIN_AREA_M2, 2 * zone.cell_m * zone.cell_m)
    x, y = anchor
    centre = ShapelyPoint(x, y)

    def piece(radius: float) -> Polygon | None:
        parts = _polygons(make_valid(polygon.intersection(centre.buffer(radius, quad_segs=16))))
        return orient(max(parts, key=lambda part: part.area)) if parts else None

    low, high = 0.0, float(centre.hausdorff_distance(polygon)) + zone.cell_m
    best: Polygon | None = None
    for _ in range(40):
        middle = (low + high) / 2
        found = piece(middle)
        if found is not None and found.area <= area_m2:
            best, low = found, middle
            if found.area >= 0.99 * area_m2:
                break
        else:
            high = middle
    return best if best is not None and best.area >= smallest else None


def choose_zones(
    zone: ZonePlane,
    polygons: list[Polygon],
    *,
    max_areas: int,
    budget_m2: float | None,
    focus_xy: Point | None,
) -> tuple[list[Polygon], bool]:
    """At most `max_areas` of `polygons`, together within `budget_m2` plus one zone cell,
    and whether the budget cut anything. In rank order (`_ranked`) each zone is taken whole
    while it fits; the first that does not is clipped to what is left of the budget, about
    its anchor (`_anchor`) -- so the budget goes on the zones that rank first, never on a
    speck behind a zone too large for it, and any budget that can hold a zone gets one."""
    ranked = [polygon for polygon in _ranked(polygons, focus_xy) if holds_weak(zone, polygon)]
    if budget_m2 is None:
        return ranked[:max_areas], False
    allowance = budget_m2 + zone.cell_m * zone.cell_m
    smallest = max(MIN_AREA_M2, 2 * zone.cell_m * zone.cell_m)
    chosen: list[Polygon] = []
    for polygon in ranked:
        left = allowance - sum(p.area for p in chosen)
        if len(chosen) == max_areas or left < smallest:
            break
        if polygon.area <= left:
            chosen.append(polygon)
            continue
        clipped = _clip(zone, polygon, min(left, budget_m2), focus_xy)
        if clipped is not None:
            chosen.append(clipped)
            break
    return chosen, chosen != ranked[:max_areas]


def site_plane(geometry: BaseGeometry, projection: SiteProjection) -> BaseGeometry:
    """`geometry` (lon/lat) in the site frame's metres, valid: what `allowed` takes."""

    def apply(points: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y = projection.to_metric(points[:, 0], points[:, 1])
        return np.column_stack([x, y])

    return make_valid(shapely.transform(geometry, apply))


def clip_to(polygons: list[Polygon], allowed: BaseGeometry) -> list[Polygon]:
    """Each zone's part inside `allowed`, parts under `MIN_AREA_M2` dropped, largest
    first. Clipped after closing, so a gap closing bridged across a hole is cut again."""
    parts = [
        orient(part)
        for polygon in polygons
        for part in _polygons(make_valid(polygon.intersection(allowed)))
        if part.area >= MIN_AREA_M2
    ]
    return sorted(parts, key=lambda part: part.area, reverse=True)


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
    allowed: BaseGeometry | None = None,
) -> Patches:
    """The weak cells `mask` (over `plane`) as at most `max_areas` patches, together within
    `budget_m2` plus at most one zone cell (module docstring). Each patch's counts are of
    `plane`'s own weak cells whose centres it holds. With `allowed` (site-frame metres:
    the ground a robot may fly, `site_plane`), every zone is clipped to it before the
    budget is spent, so no zone reaches outside it or into one of its holes."""
    zone = zone_plane(plane, mask)
    cell = zone.cell_m
    smoothing = smoothing_cells(zone.level) * cell
    focus_xy: Point | None = None
    if focus is not None:
        fx, fy = projection.to_metric(focus[0], focus[1])
        focus_xy = (float(fx), float(fy))
    drawn = zone_polygons(zone, min_area_m2=MIN_AREA_M2)
    if allowed is not None:
        drawn = clip_to(drawn, allowed)
    # A zone across the antimeridian is a zone each side of it (GeoJSON's rule, RFC 7946
    # 3.1.9): each is a polygon on the globe with its own cells and its own place in a budget.
    drawn = sorted(
        (
            part
            for polygon in drawn
            for part in split_at_antimeridian(polygon, projection)
            if part.area >= MIN_AREA_M2
        ),
        key=lambda part: part.area,
        reverse=True,
    )
    polygons, truncated = choose_zones(
        zone,
        drawn,
        max_areas=max_areas,
        budget_m2=budget_m2,
        focus_xy=focus_xy,
    )
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
        if count == 0:
            # Weak at the zone level drawn by majority, but none of the finest cells: no
            # re-scan zone (zones with no weak cell are never returned).
            continue
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
        smoothing_m=max(smoothing, cell / 2),
        zone_level=zone.level,
    )
