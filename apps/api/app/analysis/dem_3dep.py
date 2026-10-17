"""USGS 3DEP 1/3 arc-second bare-earth elevation and slope over a land boundary (US only).

The 1/3 arc-second (about 10 m) seamless DEM is a bare-earth terrain model compiled from lidar
and older sources, published as 1-degree tiles. Unlike the Copernicus surface model it is
meant to exclude canopy and buildings, which is what machine slope limits need.
"""

from __future__ import annotations

import math

import httpx
import numpy as np
import rasterio
from pydantic import HttpUrl
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from app.analysis.raster_grid import make_grid
from app.analysis.raster_io import RasterOpener, RasterStatusError, ReadBudget
from app.analysis.terrain import (
    TerrainResult,
    band_statistics,
    derive,
    encode_cog,
    padded_bounds,
)
from app.research.providers.base import SourceSpec
from app.research.providers.us_land import covers_us
from app.schemas.geojson import Footprint
from app.schemas.land_rasters import RasterMetadata, RasterRequest, RasterSource

BASE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/"
LICENSE_URL = "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits"
SPEC = SourceSpec(
    id="usgs-3dep-13",
    name="USGS 3DEP bare-earth elevation",
    domain="physical",
    coverage="United States; source projects and their dates vary by area",
    resolution="1/3 arc-second (about 10 m) bare-earth DTM, NAVD88 metres; derived slope over "
    "3-cell central differences",
    license="Public domain (U.S. Government work)",
    attribution="U.S. Geological Survey, 3D Elevation Program (3DEP) 1/3 arc-second DEM",
    documentation_url="https://www.usgs.gov/3d-elevation-program",
    endpoint=BASE,
)
UNCOVERED = "USGS 3DEP covers the United States; use cop-dem-glo-30 here."
NODATA = -999999.0
MAX_TILES = 4
TAGS = {
    "algorithm": "usgs-3dep-13-v1",
    "source": "USGS 3DEP 1/3 arc-second DEM",
    "vertical_reference": "NAVD88 (elevation band)",
    "interpretation": "Bare-earth terrain model; not a ground survey or geotechnical assessment.",
}


def tile_names(bounds: tuple[float, float, float, float] | list[float]) -> list[str]:
    """The 1-degree 3DEP tiles a lon/lat box touches, north to south then west to east.

    Tiles are named by their north-west corner: n36w121 spans 35-36 N and 121-120 W."""
    west, south, east, north = bounds
    if south < 0 or east > 0:
        raise ValueError(UNCOVERED)
    rows = range(math.ceil(north), math.floor(south), -1)
    columns = range(math.ceil(-west), math.floor(-east), -1)
    names = [f"n{row}w{column:03d}" for row in rows for column in columns]
    if len(names) > MAX_TILES:
        raise ValueError(
            "This land needs more than four USGS 3DEP tiles. Analyze a smaller section."
        )
    return names


def tile_url(name: str) -> str:
    return f"{BASE}{name}/USGS_13_{name}.tif"


def analyze(boundary: Footprint, request: RasterRequest, client: httpx.Client) -> TerrainResult:
    grid = make_grid(boundary, request)
    point = grid.geometry.representative_point()
    if not covers_us(point.x, point.y):
        raise ValueError(UNCOVERED)
    west, south, east, north = grid.bounds
    names = tile_names(padded_bounds(grid))
    surface = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    sources: list[RasterSource] = []
    missing: list[str] = []
    budget = ReadBudget(max_bytes=64 * 1024 * 1024, max_requests=256, max_seconds=60)
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_CACHEMAX=64 * 1024 * 1024,
    ):
        for name in names:
            url = tile_url(name)
            opener = RasterOpener(client, url, budget)
            try:
                # rasterio hides the opener's error, so probe once: a missing tile answers 404
                # (ocean, outside the US), which is "not covered", not an outage.
                opener(url).close()
            except RasterStatusError as error:
                if error.status_code == 404:
                    missing.append(name)
                    continue
                raise
            with rasterio.open(url, opener=opener, driver="GTiff") as source:
                if source.count != 1 or source.crs is None or source.dtypes != ("float32",):
                    raise ValueError(
                        "The 3DEP tile does not have its expected band or coordinate system."
                    )
                with WarpedVRT(
                    source,
                    crs=grid.crs.to_wkt(),
                    transform=grid.affine,
                    width=grid.width,
                    height=grid.height,
                    dtype="float32",
                    src_nodata=NODATA if source.nodata is None else source.nodata,
                    nodata=float("nan"),
                    resampling=Resampling.bilinear,
                ) as warped:
                    values = warped.read(1, masked=True).filled(np.nan).astype(np.float32)
                # Tiles overlap by a few cells; the first tile to cover a cell keeps it.
                np.copyto(surface, values, where=~np.isfinite(surface) & np.isfinite(values))
            version = opener.files[-1]
            sources.append(
                RasterSource(
                    id=f"USGS_13_{name}",
                    url=HttpUrl(url),
                    catalog_url=HttpUrl(f"{BASE}{name}/USGS_13_{name}.xml"),
                    etag=version.etag,
                    last_modified=version.last_modified,
                    catalog_datetime=None,
                    observation_period="Compiled from source projects of varying dates (see the "
                    "tile's FGDC metadata); the file date is a publication date, not an "
                    "acquisition date.",
                    license=SPEC.license,
                    license_url=HttpUrl(LICENSE_URL),
                    attribution=SPEC.attribution,
                    purpose="bare-earth elevation tile",
                )
            )
    if not sources:
        raise ValueError(UNCOVERED)
    elevation, slopes = derive(surface, grid.inside, grid.resolution)
    count = int(np.count_nonzero(grid.inside))
    valid = int(np.count_nonzero(np.isfinite(elevation)))
    if count and not valid:
        raise ValueError(UNCOVERED)
    warnings = [
        "Elevation is NAVD88 metres (orthometric), not ellipsoid heights; compare it with other "
        "models only after a datum conversion.",
        "This is a bare-earth terrain model compiled from lidar and older sources of varying "
        "dates and accuracy. It is not a ground survey or a geotechnical assessment, and recent "
        "earthworks may be missing.",
        "Slope is the gradient between neighboring 10 m cells (3-cell central differences). It "
        "smooths short steep banks, ditches and terraces narrower than about 20 m.",
        "Values are sampled on a local metric grid using bilinear interpolation. Cell centers "
        "determine boundary inclusion; small features and narrow exclusions may be unresolved.",
    ]
    if missing:
        warnings.append(
            f"USGS publishes no 3DEP tile for {', '.join(missing)}; cells there have no data."
        )
    if grid.resolution > request.resolution_m * 1.001:
        warnings.append(
            f"The analysis grid was coarsened to {grid.resolution:.1f} m to fit the selected "
            f"extent within {request.max_dimension} pixels."
        )
    if not count:
        warnings.append(
            "No analysis cell centers fall inside this boundary. "
            "The source cannot resolve a reliable area summary at this scale."
        )
    elif valid < count:
        warnings.append(
            "Some boundary cells have no source data. Summary statistics describe valid sampled "
            "cells only."
        )
    metadata = RasterMetadata(
        algorithm=TAGS["algorithm"],
        method=(
            "USGS 3DEP 1/3 arc-second bare-earth DEM tiles, bilinear resampling onto a local "
            "azimuthal-equidistant metric grid with tiles merged in name order; cell-center "
            "polygon/hole mask; central-difference slope in degrees; unweighted valid-cell "
            "statistics. Approximate sampled area is cell count multiplied by squared grid "
            "spacing, not a cadastral area measurement."
        ),
        bounds=[west, south, east, north],
        crs=grid.crs.to_wkt(),
        resolution_m=grid.resolution,
        width=grid.width,
        height=grid.height,
        boundary_cells=count,
        valid_cells=valid,
        coverage_fraction=valid / count if count else None,
        sampled_area_m2=valid * grid.resolution**2,
        bands=[
            band_statistics(elevation, 1, "Bare-earth elevation", "m"),
            band_statistics(slopes, 2, "Bare-earth slope", "degrees"),
        ],
        sources=sources,
        warnings=warnings,
        downloaded_bytes=budget.downloaded_bytes,
    )
    data = encode_cog(
        [elevation, slopes],
        metadata.crs,
        grid.affine,
        names=("Bare-earth elevation", "Bare-earth slope"),
        tags=TAGS,
    )
    return TerrainResult(data, metadata)
