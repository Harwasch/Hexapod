"""Lon/lat-aligned grids over a Land, and the one helper that puts an array onto them.

The analysis rasters live on a local metric grid (`raster_grid.make_grid`); a grid the
console drapes on the globe is instead aligned to longitude and latitude, so its cells are
the texels of one `Rectangle.fromDegrees` texture and need no reprojection in the browser.
Cells are square on the ground at the grid's centre: the degree steps come from the
ellipsoid at that latitude (pyproj.Geod), so a 20 m cell stays close to 20 m across a ranch.

`reproject_band` is `land_rasters.tile()`'s idiom (rasterio.warp.reproject, nearest, NaN
for nodata) with the grid as its destination. `encode` writes the `LonLatGrid` wire format.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from affine import Affine
from numpy.typing import NDArray
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from app.schemas.grids import LonLatGrid

GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class LonLatGridSpec:
    west: float
    south: float
    east: float
    north: float
    columns: int
    rows: int
    dlon: float
    dlat: float
    #: Ground size of a cell at the grid's centre latitude, metres (after any growth to fit).
    cell_m: float

    @property
    def transform(self) -> Affine:
        return Affine(self.dlon, 0.0, self.west, 0.0, -self.dlat, self.north)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.columns)

    @property
    def cell_area_m2(self) -> float:
        return self.cell_m * self.cell_m

    def cell_index(self, longitude: float, latitude: float) -> tuple[int, int] | None:
        """(column, row) of the cell holding the point, row 0 at the north; None outside."""
        column = math.floor((longitude - self.west) / self.dlon)
        row = math.floor((self.north - latitude) / self.dlat)
        if 0 <= column < self.columns and 0 <= row < self.rows:
            return column, row
        return None

    def cell_center(self, column: int, row: int) -> tuple[float, float]:
        return (
            self.west + (column + 0.5) * self.dlon,
            self.north - (row + 0.5) * self.dlat,
        )


def degree_steps(longitude: float, latitude: float, cell_m: float) -> tuple[float, float]:
    """Degrees of longitude and latitude that span `cell_m` metres at a point."""
    east, _, _ = GEOD.fwd(longitude, latitude, 90.0, cell_m)
    _, north, _ = GEOD.fwd(longitude, latitude, 0.0, cell_m)
    return abs(east - longitude), abs(north - latitude)


def grid_for(
    geometry: BaseGeometry,
    cell_m: float,
    *,
    max_cells: int = 256,
) -> LonLatGridSpec:
    """The grid covering `geometry` at `cell_m`, grown until it is at most `max_cells` cells
    on each side. The origin snaps to whole steps, so a small edit of an outline keeps the
    cells it had."""
    west, south, east, north = geometry.bounds
    if east - west > 10 or north - south > 10:
        raise ValueError("A lon/lat grid covers at most 10 degrees; split the area first.")
    center_lon, center_lat = (west + east) / 2, (south + north) / 2
    size = float(cell_m)
    for _ in range(32):
        dlon, dlat = degree_steps(center_lon, center_lat, size)
        origin_west = math.floor(west / dlon) * dlon
        origin_north = math.ceil(north / dlat) * dlat
        columns = max(1, math.ceil((east - origin_west) / dlon - 1e-9))
        rows = max(1, math.ceil((origin_north - south) / dlat - 1e-9))
        if columns <= max_cells and rows <= max_cells:
            return LonLatGridSpec(
                west=origin_west,
                south=origin_north - rows * dlat,
                east=origin_west + columns * dlon,
                north=origin_north,
                columns=columns,
                rows=rows,
                dlon=dlon,
                dlat=dlat,
                cell_m=size,
            )
        # Grow by the overshoot, plus a little so snapping cannot keep it one cell over.
        size *= max(columns, rows) / max_cells * 1.001
    raise ValueError("The area is too large for a lon/lat grid.")


def inside_mask(grid: LonLatGridSpec, geometry: BaseGeometry) -> NDArray[np.bool_]:
    """Cells whose centre lies in the geometry (holes stay out)."""
    if geometry.is_empty:
        return np.zeros(grid.shape, dtype=bool)
    mask: NDArray[np.bool_] = geometry_mask(
        [mapping(geometry)],
        out_shape=grid.shape,
        transform=grid.transform,
        invert=True,
        all_touched=False,
    )
    return mask


def touched_mask(grid: LonLatGridSpec, geometry: BaseGeometry) -> NDArray[np.bool_]:
    """Cells the geometry touches at all: what a footprint, a buffered line or point covers."""
    if geometry.is_empty:
        return np.zeros(grid.shape, dtype=bool)
    mask: NDArray[np.bool_] = geometry_mask(
        [mapping(geometry)],
        out_shape=grid.shape,
        transform=grid.transform,
        invert=True,
        all_touched=True,
    )
    return mask


def reproject_band(
    grid: LonLatGridSpec,
    source: NDArray[Any],
    src_transform: Affine,
    src_crs: Any,
    src_nodata: float | None,
    resampling: Resampling = Resampling.nearest,
) -> NDArray[np.float32]:
    """One band of a stored raster on the grid; NaN where the source has no value."""
    destination = np.full(grid.shape, np.nan, dtype=np.float32)
    reproject(
        source=source.astype(np.float32, copy=False),
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=grid.transform,
        dst_crs="EPSG:4326",
        dst_nodata=float("nan"),
        resampling=resampling,
        num_threads=1,
        warp_mem_limit=16,
    )
    return destination


def encode(
    grid: LonLatGridSpec,
    values: NDArray[Any],
    encoding: Literal["u8", "u16", "f32"],
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    nodata: float | None = None,
) -> LonLatGrid:
    """The `LonLatGrid` wire form of raw values already in the encoding's range."""
    if values.shape != grid.shape:
        raise ValueError("The values do not match the grid's shape.")
    dtype = {"u8": "<u1", "u16": "<u2", "f32": "<f4"}[encoding]
    raw = np.ascontiguousarray(values.astype(dtype, copy=False)).tobytes()
    return LonLatGrid(
        west=grid.west,
        south=grid.south,
        east=grid.east,
        north=grid.north,
        columns=grid.columns,
        rows=grid.rows,
        cell_m=grid.cell_m,
        encoding=encoding,
        scale=scale,
        offset=offset,
        nodata=nodata,
        data=base64.b64encode(raw).decode("ascii"),
    )


def decode(grid: LonLatGrid) -> NDArray[Any]:
    """The raw values of a `LonLatGrid` as a (rows, columns) array (tests and tools)."""
    dtype = {"u8": "<u1", "u16": "<u2", "f32": "<f4"}[grid.encoding]
    return np.frombuffer(base64.b64decode(grid.data), dtype=dtype).reshape(grid.rows, grid.columns)
