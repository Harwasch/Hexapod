"""The Fog of Knowing, policy fog-v1: how fine and how fresh the stored knowledge of each patch
of a Land is.

    density = 1 - max over records r covering the cell of  Q(class r) x 0.5^(age r / H(class r))

Every record is a stored, cited thing already in the database (a capture, a person's
inspection, a dated satellite scene, an aligned air photo, research evidence); nothing here
fetches, and nothing simulated or inferred counts (plans, the demo fleet, Living Survey
motion, inferred fill). The class table below is a policy, not a measurement: the qualities
and half-lives are judgement, stated beside every number they produce, and any change to them
is a new policy version.

Pure module: no database. `app.services.land_knowledge` collects the records; this module
decides what they are worth on a lon/lat grid (`app.analysis.lonlat_grid`), composes the fog,
and turns the foggiest cells into scout patches.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pyproj import CRS, Geod, Transformer
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from shapely.validation import make_valid

from app.analysis.lonlat_grid import LonLatGridSpec, grid_for, inside_mask, touched_mask

POLICY_VERSION = "fog-v1"
#: An undated record counts as if it were two half-lives old.
UNDATED_FACTOR = 0.25
#: How far a person's inspection of a mapped point or line is taken to reach, metres.
FIELD_REACH_M: dict[str, float] = {"Point": 15.0, "LineString": 5.0}
#: Cells at or above this density read as foggy (the legend's "Foggy" share, the default
#: scout threshold).
FOGGY = 0.5
#: Densities within this step of each other rank as equally foggy when choosing scout cells,
#: so a patch stays in one piece near the view instead of chasing a 0.004 difference.
SCOUT_TIE_STEP = 0.05
MIN_CELL_M = 10.0
MAX_CELL_M = 100.0
MAX_GRID_CELLS = 256
#: Density byte for a cell outside the boundary; inside cells use 0..250 (density x 250).
OUTSIDE = 255
#: Best-record index for a cell no record covers.
NO_RECORD = 65535

ClassId = Literal["centimetre", "field", "aerial", "satellite", "regional"]
SourceKind = Literal[
    "capture",
    "inspection",
    "raster-observation",
    "archive-alignment",
    "research-evidence",
    "survey",
]

GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class KnowledgeClass:
    id: ClassId
    label: str
    quality: float
    half_life_days: float
    #: The coarsest ground resolution the class admits, metres; None when resolution does not
    #: define it (a person's inspection; regional and unlocated records).
    max_resolution_m: float | None
    example: str


CLASSES: tuple[KnowledgeClass, ...] = (
    KnowledgeClass(
        "centimetre",
        "Centimetre capture",
        1.00,
        180,
        0.05,
        "A drone or phone capture at 5 cm per pixel or finer",
    ),
    KnowledgeClass(
        "field",
        "Field inspection",
        0.85,
        120,
        None,
        "A person's dated inspection of a mapped feature",
    ),
    KnowledgeClass(
        "aerial",
        "Aerial imagery",
        0.70,
        365,
        1.0,
        "An air photo or historical map aligned to the ground, 5 cm to 1 m",
    ),
    KnowledgeClass(
        "satellite",
        "Satellite, 1 to 20 m",
        0.45,
        90,
        20.0,
        "A cloud-masked Sentinel-2 scene at 10 m",
    ),
    KnowledgeClass(
        "regional",
        "Regional or unlocated",
        0.20,
        1825,
        None,
        "A soil map, a 30 m elevation model, a document about the land",
    ),
)
CLASS_BY_ID: dict[str, KnowledgeClass] = {item.id: item for item in CLASSES}
REGIONAL = CLASS_BY_ID["regional"]
#: One class coarser, for a record placed less precisely than a cell.
DEMOTION: dict[ClassId, ClassId] = {
    "centimetre": "aerial",
    "field": "aerial",
    "aerial": "satellite",
    "satellite": "regional",
    "regional": "regional",
}
FORMULA = (
    "Fog density = 1 - the best credit among the records that cover a cell. "
    "Credit = class quality Q x 0.5^(age in days / the class half-life). "
    "An undated record counts at Q x 0.25, as if two half-lives old. "
    "A record with no published location covers the whole land, with Q capped at the "
    "regional class's 0.20. A record placed less precisely than a cell drops one class."
)


def class_for_resolution(resolution_m: float | None) -> ClassId:
    """The class a ground resolution earns: 5 cm or finer, to 1 m, to 20 m, coarser."""
    if resolution_m is None or not math.isfinite(resolution_m) or resolution_m <= 0:
        return "regional"
    if resolution_m <= 0.05:
        return "centimetre"
    if resolution_m <= 1.0:
        return "aerial"
    if resolution_m <= 20.0:
        return "satellite"
    return "regional"


def demote(class_id: ClassId) -> ClassId:
    return DEMOTION[class_id]


def age_days(observed_at: datetime | None, as_of: datetime) -> float | None:
    if observed_at is None:
        return None
    return max(0.0, (as_of - observed_at).total_seconds() / 86_400)


def credit(
    class_id: ClassId, observed_at: datetime | None, as_of: datetime, *, located: bool = True
) -> float:
    """What one record is worth in a cell it covers: Q x 0.5^(age/H), Q x 0.25 undated, with
    Q capped at the regional class's when the record has no published location."""
    cls = CLASS_BY_ID[class_id]
    quality = cls.quality if located else min(cls.quality, REGIONAL.quality)
    age = age_days(observed_at, as_of)
    if age is None:
        return quality * UNDATED_FACTOR
    return float(quality * 0.5 ** (age / cls.half_life_days))


def _age_text(days: float) -> str:
    return f"{days:.1f}" if days < 1 else f"{days:.0f}"


def arithmetic(
    class_id: ClassId, observed_at: datetime | None, as_of: datetime, *, located: bool = True
) -> str:
    """The credit as the cell card prints it, e.g. '0.45 x 0.5^(7/90) = 0.42'."""
    cls = CLASS_BY_ID[class_id]
    value = credit(class_id, observed_at, as_of, located=located)
    quality = (
        f"min({cls.quality:.2f}, {REGIONAL.quality:.2f})"
        if not located and cls.quality > REGIONAL.quality
        else f"{cls.quality:.2f}"
    )
    age = age_days(observed_at, as_of)
    if age is None:
        return f"{quality} x {UNDATED_FACTOR:.2f} (undated) = {value:.2f}"
    return f"{quality} x 0.5^({_age_text(age)}/{cls.half_life_days:g}) = {value:.2f}"


@dataclass(frozen=True)
class FogGrid:
    spec: LonLatGridSpec
    boundary: BaseGeometry
    inside: NDArray[np.bool_]
    metric_crs: CRS
    to_metric: Transformer
    to_geographic: Transformer

    @property
    def shape(self) -> tuple[int, int]:
        return self.spec.shape

    @property
    def cell_m(self) -> float:
        return self.spec.cell_m

    @property
    def inside_cells(self) -> int:
        return int(np.count_nonzero(self.inside))


def make_fog_grid(boundary: BaseGeometry, cell_m: float = 20.0) -> FogGrid:
    """The fog's grid over a Land: lon/lat cells of `cell_m` (10 to 100 m), grown until it is
    at most 256 x 256; a cell is inside when its centre is (holes stay out)."""
    size = min(MAX_CELL_M, max(MIN_CELL_M, float(cell_m)))
    spec = grid_for(boundary, size, max_cells=MAX_GRID_CELLS)
    point = boundary.representative_point()
    metric = CRS.from_proj4(
        f"+proj=aeqd +lat_0={point.y} +lon_0={point.x} +datum=WGS84 +units=m +no_defs"
    )
    return FogGrid(
        spec=spec,
        boundary=boundary,
        inside=inside_mask(spec, boundary),
        metric_crs=metric,
        to_metric=Transformer.from_crs(4326, metric, always_xy=True),
        to_geographic=Transformer.from_crs(metric, 4326, always_xy=True),
    )


@dataclass
class KnowledgeSource:
    """One stored record as the fog weighs it."""

    key: str
    kind: SourceKind
    title: str
    class_id: ClassId
    resolution_m: float | None
    observed_at: datetime | None
    #: How the date reads: "2026-10-07 (scene acquisition)", "undated: catalog gives none".
    observed_label: str
    recorded_at: datetime
    located: bool
    positional_uncertainty_m: float | None
    provenance: dict[str, Any]
    link: dict[str, Any]
    geometry: BaseGeometry | None = None
    #: A precomputed per-cell coverage on the fog grid (a raster's valid cells).
    mask: NDArray[np.bool_] | None = None
    #: Said beside the class, e.g. "class assumed from the representation".
    class_note: str | None = None


@dataclass(frozen=True)
class Weighed:
    """A record's effective class on this grid and its credit."""

    class_id: ClassId
    #: The class it counts as: regional when it has no location and would earn more.
    counted_class_id: ClassId
    credit: float
    arithmetic: str
    note: str | None


def weigh(grid: FogGrid, source: KnowledgeSource, as_of: datetime) -> Weighed:
    class_id = source.class_id
    notes = [source.class_note] if source.class_note else []
    uncertainty = source.positional_uncertainty_m
    if source.located and uncertainty is not None and uncertainty > grid.cell_m:
        demoted = demote(class_id)
        if demoted != class_id:
            notes.append(
                f"One class down from {CLASS_BY_ID[class_id].label.lower()}: placed to within "
                f"{uncertainty:.0f} m, coarser than the {grid.cell_m:.0f} m cell."
            )
        class_id = demoted
    counted = class_id
    if not source.located and CLASS_BY_ID[class_id].quality > REGIONAL.quality:
        counted = "regional"
        notes.append("No published location: counts for the whole land, capped at regional.")
    return Weighed(
        class_id=class_id,
        counted_class_id=counted,
        credit=credit(class_id, source.observed_at, as_of, located=source.located),
        arithmetic=arithmetic(class_id, source.observed_at, as_of, located=source.located),
        note=" ".join(notes) or None,
    )


def _reach(geometry: BaseGeometry) -> float:
    kind = geometry.geom_type.removeprefix("Multi")
    return FIELD_REACH_M.get(kind, 0.0)


def footprint_mask(grid: FogGrid, source: KnowledgeSource) -> NDArray[np.bool_]:
    """The inside cells a record covers: its raster mask as given; a point or line buffered by
    the field reach; a footprint as drawn (any cell it touches); unlocated, every cell."""
    if source.mask is not None:
        return source.mask & grid.inside
    if not source.located or source.geometry is None or source.geometry.is_empty:
        return grid.inside.copy()
    geometry = source.geometry
    parts = list(getattr(geometry, "geoms", [geometry]))
    reached: list[BaseGeometry] = []
    for part in parts:
        reach = _reach(part)
        if reach > 0:
            metric = transform_geometry(grid.to_metric.transform, part).buffer(reach, quad_segs=8)
            reached.append(transform_geometry(grid.to_geographic.transform, metric))
        elif part.geom_type in {"Polygon", "MultiPolygon"}:
            reached.append(part)
    if not reached:
        return np.zeros(grid.shape, dtype=bool)
    return touched_mask(grid.spec, unary_union(reached)) & grid.inside


@dataclass
class Counted:
    source: KnowledgeSource
    weighed: Weighed
    mask: NDArray[np.bool_]


@dataclass
class Composed:
    #: 0..250 = density x 250 inside; 255 outside.
    density: NDArray[np.uint8]
    #: Index into `counted` of the record that sets each cell; 65535 where none does.
    best: NDArray[np.uint16]
    counted: list[Counted] = field(default_factory=list)


def counts_at(source: KnowledgeSource, as_of: datetime) -> bool:
    """A record counts at `as_of` only once it was both observed and stored by then."""
    if source.recorded_at > as_of:
        return False
    return source.observed_at is None or source.observed_at <= as_of


def compose(grid: FogGrid, sources: Sequence[KnowledgeSource], as_of: datetime) -> Composed:
    best_credit = np.zeros(grid.shape, dtype=np.float64)
    best = np.full(grid.shape, NO_RECORD, dtype=np.uint16)
    counted: list[Counted] = []
    for source in sources:
        if not counts_at(source, as_of):
            continue
        mask = footprint_mask(grid, source)
        if not mask.any():
            continue
        weighed = weigh(grid, source, as_of)
        index = len(counted)
        if index >= NO_RECORD:
            break
        counted.append(Counted(source, weighed, mask))
        better = mask & (weighed.credit > best_credit)
        best_credit[better] = weighed.credit
        best[better] = index
    density = np.where(grid.inside, np.rint((1.0 - best_credit) * 250.0), OUTSIDE).astype(np.uint8)
    best[~grid.inside] = NO_RECORD
    return Composed(density=density, best=best, counted=counted)


def summary(grid: FogGrid, composed: Composed) -> dict[str, Any]:
    inside = grid.inside
    total = int(np.count_nonzero(inside))
    by_class: dict[str, float] = {item.id: 0.0 for item in CLASSES}
    by_class["none"] = 0.0
    if total == 0:
        return {
            "inside_cells": 0,
            "by_best_class": by_class,
            "foggy_fraction": 0.0,
            "median_density": None,
        }
    best = composed.best[inside]
    none = int(np.count_nonzero(best == NO_RECORD))
    by_class["none"] = none / total
    for index, item in enumerate(composed.counted):
        cells = int(np.count_nonzero(best == index))
        if cells:
            by_class[item.weighed.counted_class_id] += cells / total
    density = composed.density[inside].astype(np.float64) / 250.0
    return {
        "inside_cells": total,
        "by_best_class": by_class,
        "foggy_fraction": float(np.count_nonzero(density >= FOGGY - 1e-9) / total),
        "median_density": float(np.median(density)),
    }


@dataclass(frozen=True)
class ScoutPatch:
    footprint: Polygon
    area_m2: float
    cells: int
    mean_density: float
    #: Cells by the class of the record that sets them ('none' for no record).
    best_class_counts: dict[str, int]
    #: The most common best record in the patch, by index into `counted`, if any.
    dominant_record: int | None


def _geodesic_area_m2(geometry: BaseGeometry) -> float:
    polygons = (
        list(geometry.geoms)
        if isinstance(geometry, MultiPolygon)
        else [geometry]
        if isinstance(geometry, Polygon)
        else []
    )
    return sum(abs(GEOD.geometry_area_perimeter(orient(item))[0]) for item in polygons)


def scout_mask(
    grid: FogGrid,
    density: NDArray[np.uint8],
    threshold: float,
    limit_cells: int | None,
    focus: tuple[float, float] | None,
) -> tuple[NDArray[np.bool_], bool]:
    """The cells to scout: inside and at least `threshold` foggy. Over `limit_cells`, the
    foggiest first (in steps of `SCOUT_TIE_STEP`), then those nearest `focus` (the land's
    centre without one). Returns the mask and whether the limit cut it."""
    value = density.astype(np.float64) / 250.0
    candidates = grid.inside & (density != OUTSIDE) & (value >= threshold - 1e-9)
    count = int(np.count_nonzero(candidates))
    if limit_cells is None or count <= limit_cells:
        return candidates, False
    rows, columns = np.nonzero(candidates)
    longitudes = grid.spec.west + (columns + 0.5) * grid.spec.dlon
    latitudes = grid.spec.north - (rows + 0.5) * grid.spec.dlat
    if focus is None:
        center = grid.boundary.representative_point()
        focus = (center.x, center.y)
    scale_x = math.cos(math.radians(focus[1])) * 111_320.0
    distance = np.hypot((longitudes - focus[0]) * scale_x, (latitudes - focus[1]) * 110_574.0)
    tier = np.floor(value[rows, columns] / SCOUT_TIE_STEP)
    order = np.lexsort((distance, -tier))
    chosen = order[: max(0, limit_cells)]
    mask = np.zeros(grid.shape, dtype=bool)
    mask[rows[chosen], columns[chosen]] = True
    return mask, True


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if hasattr(geometry, "geoms"):
        return [polygon for part in geometry.geoms for polygon in _polygons(part)]
    return []


def mask_polygons(grid: FogGrid, mask: NDArray[np.bool_]) -> list[Polygon]:
    """Cells as smooth patches: the cell squares merged, closed by half a cell (the staircase
    rounded, one-cell gaps bridged), simplified, clipped to the boundary; parts under two
    cells dropped; largest first."""
    if not mask.any():
        return []
    squares = [
        shape(geometry)
        for geometry, value in shapes(
            mask.astype(np.uint8), mask=mask, transform=grid.spec.transform, connectivity=4
        )
        if value == 1
    ]
    merged = unary_union(squares)
    half = grid.cell_m / 2
    metric = transform_geometry(grid.to_metric.transform, merged)
    smoothed = metric.buffer(half, quad_segs=4).buffer(-half, quad_segs=4).simplify(half)
    geographic = make_valid(transform_geometry(grid.to_geographic.transform, smoothed))
    clipped = geographic.intersection(grid.boundary)
    parts = [
        orient(part)
        for part in _polygons(make_valid(clipped))
        if _geodesic_area_m2(part) >= 2 * grid.spec.cell_area_m2
    ]
    return sorted(parts, key=_geodesic_area_m2, reverse=True)


def scout(
    grid: FogGrid,
    composed: Composed,
    *,
    threshold: float = FOGGY,
    budget_m2: float | None = None,
    focus: tuple[float, float] | None = None,
    max_areas: int = 3,
) -> tuple[list[ScoutPatch], bool]:
    """The foggiest patches to fly, at most `max_areas`, together within `budget_m2` plus at
    most one cell (smoothing a patch's outline can add a sliver; the cell count is trimmed
    until it fits). Returns the patches and whether the budget cut the fog short."""
    cell_area = grid.spec.cell_area_m2
    limit = None if budget_m2 is None else int(budget_m2 // cell_area)
    patches: list[Polygon] = []
    mask = np.zeros(grid.shape, dtype=bool)
    truncated = False
    for _ in range(8):
        mask, cut = scout_mask(grid, composed.density, threshold, limit, focus)
        truncated = truncated or cut
        patches = mask_polygons(grid, mask)[:max_areas]
        total = sum(_geodesic_area_m2(item) for item in patches)
        if budget_m2 is None or total <= budget_m2 + cell_area:
            break
        selected = int(np.count_nonzero(mask))
        limit = max(0, selected - math.ceil((total - budget_m2) / cell_area))
        truncated = True
    value = composed.density.astype(np.float64) / 250.0
    result: list[ScoutPatch] = []
    for polygon in patches:
        cells = inside_mask(grid.spec, polygon) & mask
        count = int(np.count_nonzero(cells))
        best = composed.best[cells]
        counts: dict[str, int] = {}
        for index in np.unique(best):
            label = (
                "none"
                if int(index) == NO_RECORD
                else composed.counted[int(index)].weighed.counted_class_id
            )
            counts[label] = counts.get(label, 0) + int(np.count_nonzero(best == index))
        records = best[best != NO_RECORD]
        dominant = int(np.bincount(records).argmax()) if records.size else None
        result.append(
            ScoutPatch(
                footprint=polygon,
                area_m2=_geodesic_area_m2(polygon),
                cells=count,
                mean_density=float(value[cells].mean()) if count else 0.0,
                best_class_counts=counts,
                dominant_record=dominant,
            )
        )
    return result, truncated
