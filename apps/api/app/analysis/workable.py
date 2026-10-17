"""Workable ground: the part of an outline a ground machine class can work, by slope and creeks.

The dress rehearsal (apps/web/src/missions/rehearsal.ts) raises a hand where a plan's drawn
lane goes past a limit its machine class is assumed to keep: slope over `maxSlopePct`, or
nearer than `creekBufferM` to a mapped creek (apps/api/app/services/machine_classes.json).
This answers the question that follows, "then where can it work?", on the same ground and by
the same rules, so a plan redrafted on the answer drives lanes the rehearsal can pass.

Slope is read per cell of a stored elevation raster on its local metric grid: the steepest
gradient, `hypot(dz/dx, dz/dy)`, from central differences one cell either side (`np.gradient`),
one-sided at the edge of the data as the rehearsal reads it. A cell within the limit can be
driven across or up and down without passing it, which is why the steepest gradient and not
the grade along one heading decides. Ground is split in order, so the parts add up to the
outline and nothing is counted twice:

1. **no data**: where the raster has no height (outside the Land's saved outline). A cell the
   outline cuts has no height either (cells count by their centre), so it takes the steepest
   slope of its neighbours: ground along the outline is not lost for want of a centre;
2. **steep**: cells over the limit, as whole cell squares;
3. **creek buffer**: the rest within the buffer of a counted creek line;
4. **slivers**: pieces of what is left too small to work (under 0.1 ac), dropped and counted;
5. **workable**: everything else.

Pure: arrays, shapely geometry in the grid's metric CRS, no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from affine import Affine
from numpy.typing import NDArray
from rasterio.features import shapes
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from app.analysis.similarity import centres_in, polygon_parts

ACRE_M2 = 4046.8564224
#: Workable pieces smaller than this are slivers: too small to work as a pass.
MIN_PART_M2 = 0.1 * ACRE_M2
#: The creek buffer is drawn this much wider than the class's limit, with fine arcs: a buffer
#: polygon's chords cut inside its true circle, and a lane run along the cut edge must not
#: come back from the rehearsal a hair inside the limit (it reads distance exactly).
BUFFER_EDGE_M = 0.1
BUFFER_QUAD_SEGS = 16
#: NHD kinds counted as creeks, as the rehearsal counts them (`countsAsCreek`): every natural
#: channel and the artificial paths that carry one through a pond; ephemeral drainages only
#: on request; canals, ditches, connectors and pipelines never.
CREEK_KINDS = frozenset({"perennial", "intermittent", "unclassified", "artificial-path"})


def counts_as_creek(kind: str, include_ephemeral: bool) -> bool:
    return kind in CREEK_KINDS or (include_ephemeral and kind == "ephemeral")


def _axis_gradient(values: NDArray[np.float64], step: float, axis: int) -> NDArray[np.float64]:
    """dz per metre along one axis: central where both neighbours have a height, one-sided
    where only one does, NaN where neither (or the cell itself) has one."""
    plus = np.full_like(values, np.nan)
    minus = np.full_like(values, np.nan)
    if axis == 0:
        plus[:-1] = values[1:]
        minus[1:] = values[:-1]
    else:
        plus[:, :-1] = values[:, 1:]
        minus[:, 1:] = values[:, :-1]
    central = (plus - minus) / (2 * step)
    forward = (plus - values) / step
    backward = (values - minus) / step
    return np.where(
        np.isfinite(central), central, np.where(np.isfinite(forward), forward, backward)
    )


def slope_percent(elevation: NDArray[np.floating], resolution_m: float) -> NDArray[np.float32]:
    """The steepest gradient of each cell in percent (100 = 45 degrees), NaN without a height.

    Inside the data this is `np.gradient`'s central difference (the stored slope band's
    method); at its edge, where a neighbour has no height, the one-sided difference."""
    values = np.asarray(elevation, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        dz_dy = _axis_gradient(values, resolution_m, 0)
        dz_dx = _axis_gradient(values, resolution_m, 1)
        slope = np.hypot(dz_dx, dz_dy) * 100
    slope[~np.isfinite(values)] = np.nan
    return slope.astype(np.float32)


def edge_filled(slope_pct: NDArray[np.floating]) -> NDArray[np.float32]:
    """`slope_pct` with each cell that has no slope but a neighbour (of eight) that does given
    the steepest neighbour's: one ring of cells around the data, where the Land's outline cuts
    cells whose centres fall outside it."""
    values = np.asarray(slope_pct, dtype=np.float32)
    padded = np.pad(values, 1, constant_values=np.nan)
    rows, columns = values.shape
    steepest = np.full_like(values, np.nan)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                neighbour = padded[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + columns]
                steepest = np.fmax(steepest, neighbour)
    missing = ~np.isfinite(values)
    filled = values.copy()
    filled[missing] = steepest[missing]
    return filled


def cell_region(mask: NDArray[np.bool_], transform: Affine) -> BaseGeometry:
    """The cells of a mask as one geometry of whole cell squares (in the grid's CRS)."""
    if not mask.any():
        return Polygon()
    squares = [
        shape(geometry)
        for geometry, value in shapes(
            mask.astype(np.uint8), mask=mask, transform=transform, connectivity=4
        )
        if value == 1
    ]
    return make_valid(unary_union(squares))


def _area(geometry: BaseGeometry) -> BaseGeometry:
    """Only the area of a geometry: an intersection's stray lines and points go."""
    parts = polygon_parts(make_valid(geometry)) if not geometry.is_empty else []
    if not parts:
        return Polygon()
    return parts[0] if len(parts) == 1 else make_valid(unary_union(parts))


@dataclass(frozen=True)
class GroundSplit:
    """An outline split by what a machine class can work, in the grid's metric CRS."""

    workable: BaseGeometry
    no_data: BaseGeometry
    steep: BaseGeometry
    creek_buffer: BaseGeometry
    slivers: BaseGeometry
    sliver_count: int
    #: How many of the outline's cells had a slope read, and how many were over the limit.
    cells_read: int
    cells_steep: int


def split_ground(
    footprint: BaseGeometry,
    slope_pct: NDArray[np.floating],
    transform: Affine,
    *,
    max_slope_pct: float,
    creeks: Sequence[BaseGeometry] = (),
    buffer_m: float = 0.0,
    min_part_m2: float = MIN_PART_M2,
) -> GroundSplit:
    """`footprint` (metric, the grid's CRS) split into no data, steep, creek buffer, slivers
    and workable ground, in that order. `slope_pct` is per cell on `transform`'s grid (NaN for
    no data; the ring of cells around the data takes its neighbours' slope, `edge_filled`)."""
    outline = _area(footprint)
    slope = edge_filled(slope_pct)
    known = np.isfinite(slope)
    over = known & (np.nan_to_num(slope, nan=0.0) > max_slope_pct)
    read = cell_region(known, transform)
    steep_cells = cell_region(over, transform)
    no_data = _area(outline.difference(read))
    ground = _area(outline.intersection(read))
    steep = _area(ground.intersection(steep_cells))
    rest = _area(ground.difference(steep_cells))
    buffers = (
        make_valid(
            unary_union(
                [
                    line.buffer(buffer_m + BUFFER_EDGE_M, quad_segs=BUFFER_QUAD_SEGS)
                    for line in creeks
                ]
            )
        )
        if creeks and buffer_m > 0
        else Polygon()
    )
    creek_buffer = _area(rest.intersection(buffers)) if not buffers.is_empty else Polygon()
    if not buffers.is_empty:
        rest = _area(rest.difference(buffers))
    kept = [part for part in polygon_parts(rest) if part.area >= min_part_m2]
    small = [part for part in polygon_parts(rest) if part.area < min_part_m2]
    # Which cells the outline covers, by centre, for the counts the answer cites.
    inside = centres_in(outline, (slope.shape[0], slope.shape[1]), transform)
    return GroundSplit(
        workable=MultiPolygon(kept) if kept else Polygon(),
        no_data=no_data,
        steep=steep,
        creek_buffer=creek_buffer,
        slivers=make_valid(unary_union(small)) if small else Polygon(),
        sliver_count=len(small),
        cells_read=int(np.count_nonzero(inside & known)),
        cells_steep=int(np.count_nonzero(inside & over)),
    )
