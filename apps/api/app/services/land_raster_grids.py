"""One band of a stored land raster on a lon/lat grid, for the console to compute with.

Map tiles (`land_rasters.tile`) are pictures; some console features need the numbers. The dress
rehearsal (apps/web/src/missions/rehearsal.ts) reads the slope along and across every lane of a
plan from a Land's elevation, so it asks for the elevation band here. The band is resampled from
the raster's metric analysis grid onto a lon/lat grid of about the same cell size
(`lonlat_grid.grid_for`), bilinearly for a measured quantity and nearest for a categorical band,
and sent as float32 with NaN where the raster has no value (outside the pinned boundary, or no
source data). The grid is never larger than MAX_CELLS a side; a larger raster gets coarser cells,
and `cellM` says how coarse.
"""

from __future__ import annotations

from rasterio.enums import Resampling
from shapely.geometry import box
from sqlalchemy.orm import Session

from app.analysis.lonlat_grid import encode, grid_for, reproject_band
from app.models.land_raster import LandRaster
from app.schemas.grids import LonLatGrid
from app.schemas.land_rasters import RasterMetadata
from app.services import land_rasters
from app.services.errors import NotFoundError

#: Cells a side at most: 512 x 512 float32 is 1 MiB before base64.
MAX_CELLS = 512


def band_grid(db: Session, row: LandRaster, band: int) -> LonLatGrid:
    """`band` of the raster (1-based, as in its metadata) on a lon/lat grid. Callers check
    access first (`land_rasters.scoped`)."""
    metadata = RasterMetadata.model_validate(row.metadata_json)
    definition = next((value for value in metadata.bands if value.index == band), None)
    if definition is None:
        raise NotFoundError("raster band", band)
    west, south, east, north = metadata.bounds
    spec = grid_for(box(west, south, east, north), metadata.resolution_m, max_cells=MAX_CELLS)
    source = land_rasters.band_array(db, row, band)
    resampling = Resampling.nearest if definition.palette == "categorical" else Resampling.bilinear
    values = reproject_band(
        spec, source.data, source.transform, source.crs, source.nodata, resampling
    )
    return encode(spec, values, "f32")
