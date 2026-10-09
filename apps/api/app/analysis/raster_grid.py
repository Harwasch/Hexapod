"""Shared bounded metric grid and polygon/hole mask for registered land analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from affine import Affine
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry

from app.schemas.geojson import Footprint
from app.schemas.land_rasters import RasterRequest


@dataclass(frozen=True)
class AnalysisGrid:
    geometry: BaseGeometry
    bounds: tuple[float, float, float, float]
    latitude: float
    crs: CRS
    affine: Affine
    resolution: float
    width: int
    height: int
    inside: NDArray[np.bool_]


def make_grid(boundary: Footprint, request: RasterRequest) -> AnalysisGrid:
    geometry = shape(boundary.model_dump())
    west, south, east, north = geometry.bounds
    if east - west > 10 or north - south > 10:
        raise ValueError(
            "Analyze a regional area at a time. Split distant or antimeridian-spanning parts before raster analysis."
        )
    point = geometry.representative_point()
    metric_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={point.y} +lon_0={point.x} +datum=WGS84 +units=m +no_defs"
    )
    projected = transform_geometry(
        Transformer.from_crs(4326, metric_crs, always_xy=True).transform, geometry
    )
    left, bottom, right, top = projected.bounds
    span = max(right - left, top - bottom)
    if span > 250_000:
        raise ValueError(
            "This regional analysis supports extents up to 250 km. Analyze a smaller portion of the land."
        )
    resolution = max(request.resolution_m, span / (request.max_dimension - 6))
    origin_x, origin_y = (
        math.floor(left / resolution) * resolution - resolution,
        math.ceil(top / resolution) * resolution + resolution,
    )
    width, height = (
        max(3, math.ceil((right - origin_x) / resolution) + 1),
        max(3, math.ceil((origin_y - bottom) / resolution) + 1),
    )
    affine = from_origin(origin_x, origin_y, resolution, resolution)
    inside = geometry_mask(
        [mapping(projected)],
        out_shape=(height, width),
        transform=affine,
        invert=True,
        all_touched=False,
    )
    return AnalysisGrid(
        geometry,
        (west, south, east, north),
        point.y,
        metric_crs,
        affine,
        resolution,
        width,
        height,
        inside,
    )
