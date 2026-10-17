from __future__ import annotations

import hashlib
import io
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass

import morecantile
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform
from rasterio.windows import Window
from rio_tiler.colormap import cmap
from shapely.geometry import Point, shape
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.analysis.terrain import TerrainResult
from app.models.land import LandArea, LandBoundaryRevision
from app.models.land_raster import LandRaster, LandRasterBlob
from app.models.research import Investigation, ResearchRun
from app.models.workspace import Workspace
from app.schemas.land_rasters import LandRasterRead, RasterMetadata, RasterRequest, RasterSample
from app.services.errors import InvalidInputError, NotFoundError

TMS = morecantile.tms.get("WorldCRS84Quad")
#: Decoded bands kept in memory between map tiles. One map view asks for 20 to 40 tiles of a
#: band, and each would otherwise read the whole blob from Postgres (a signature holds 20
#: bands, up to 16 MiB) and inflate that band again. Keyed by the blob's content hash, so a
#: hit is always the same bytes a caller was already allowed to read.
BAND_CACHE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class DecodedBand:
    data: np.ndarray
    transform: Affine
    crs: CRS
    nodata: float | None


_bands: OrderedDict[tuple[str, int], DecodedBand] = OrderedDict()
_loading: dict[tuple[str, int], threading.Lock] = {}
_bands_lock = threading.Lock()


def scoped(db: Session, workspace_id: uuid.UUID, raster_id: uuid.UUID) -> LandRaster:
    row = db.scalar(
        select(LandRaster)
        .join(LandArea, LandArea.id == LandRaster.land_id)
        .where(LandArea.workspace_id == workspace_id, LandRaster.id == raster_id)
    )
    if row is None:
        raise NotFoundError("land raster", raster_id)
    return row


def read(db: Session, row: LandRaster) -> LandRasterRead:
    revision = db.scalar(select(LandArea.revision).where(LandArea.id == row.land_id))
    return LandRasterRead(
        id=row.id,
        run_id=row.run_id,
        land_id=row.land_id,
        boundary_revision=row.boundary_revision,
        request=RasterRequest.model_validate(row.request),
        metadata=RasterMetadata.model_validate(row.metadata_json),
        sha256=row.sha256,
        byte_size=row.byte_size,
        created_at=row.created_at,
        stale=revision != row.boundary_revision,
    )


def save(
    db: Session,
    workspace_id: uuid.UUID,
    run: ResearchRun,
    raster_id: uuid.UUID,
    request: RasterRequest,
    result: TerrainResult,
    quota_bytes: int,
) -> LandRaster:
    """The worker holds the run's live lease lock; caller commits with its checkpoint."""
    investigation = db.get(Investigation, run.investigation_id)
    if investigation is None:
        raise NotFoundError("investigation", run.investigation_id)
    land = db.get(LandArea, investigation.land_id)
    if land is None or land.workspace_id != workspace_id:
        raise NotFoundError("land", investigation.land_id)
    db.execute(select(Workspace.id).where(Workspace.id == workspace_id).with_for_update())
    existing = db.get(LandRaster, raster_id)
    if existing is not None:
        return existing
    if not 0 < len(result.data) <= 16 * 1024 * 1024:
        raise InvalidInputError("The raster output exceeds the 16 MiB storage limit.")
    used = (
        db.scalar(
            select(func.coalesce(func.sum(LandRaster.byte_size), 0))
            .join(LandArea, LandArea.id == LandRaster.land_id)
            .where(LandArea.workspace_id == workspace_id)
        )
        or 0
    )
    if used + len(result.data) > quota_bytes:
        raise InvalidInputError("This workspace's raster storage allowance is full.")
    row = LandRaster(
        id=raster_id,
        run_id=run.id,
        land_id=land.id,
        boundary_revision=investigation.boundary_revision,
        request=request.model_dump(mode="json"),
        metadata_json=result.metadata.model_dump(mode="json"),
        sha256=hashlib.sha256(result.data).hexdigest(),
        byte_size=len(result.data),
    )
    db.add(row)
    db.flush()
    db.add(LandRasterBlob(raster_id=row.id, data=result.data))
    db.flush()
    return row


def blob(db: Session, row: LandRaster) -> bytes:
    original = db.get(LandRasterBlob, row.id)
    if original is None:
        raise NotFoundError("raster bytes", row.id)
    return original.data


def band_array(db: Session, row: LandRaster, band: int) -> DecodedBand:
    """One band of a stored raster on its analysis grid, from the in-memory cache when a
    recent tile (or a later similarity) already decoded it. Callers check access first."""
    key = (row.sha256, band)
    with _bands_lock:
        cached = _bands.get(key)
        if cached is not None:
            _bands.move_to_end(key)
            return cached
        loading = _loading.setdefault(key, threading.Lock())
    # A map view's tiles arrive together: the first decodes, the rest wait and reuse it.
    with loading:
        with _bands_lock:
            cached = _bands.get(key)
        if cached is not None:
            return cached
        try:
            with MemoryFile(blob(db, row)) as memory, memory.open(driver="GTiff") as dataset:
                decoded = DecodedBand(
                    dataset.read(band), dataset.transform, dataset.crs, dataset.nodata
                )
            with _bands_lock:
                _bands[key] = decoded
                size = sum(item.data.nbytes for item in _bands.values())
                while size > BAND_CACHE_BYTES and len(_bands) > 1:
                    size -= _bands.popitem(last=False)[1].data.nbytes
        finally:
            with _bands_lock:
                _loading.pop(key, None)
    return decoded


def transparent_tile() -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (256, 256), (0, 0, 0, 0)).save(output, format="PNG")
    return output.getvalue()


def tile(db: Session, row: LandRaster, band: int, z: int, x: int, y: int) -> bytes:
    metadata = RasterMetadata.model_validate(row.metadata_json)
    definition = next((value for value in metadata.bands if value.index == band), None)
    if definition is None or not 0 <= z <= 22 or not 0 <= x < 2 ** (z + 1) or not 0 <= y < 2**z:
        raise NotFoundError("raster tile", f"{band}/{z}/{x}/{y}")
    if not definition.valid_cells:
        return transparent_tile()
    bounds = TMS.bounds(morecantile.Tile(x, y, z))
    west, south, east, north = metadata.bounds
    if bounds.right <= west or bounds.left >= east or bounds.top <= south or bounds.bottom >= north:
        return transparent_tile()
    # Reproject directly into the fixed output grid. A virtual raster covering a
    # hemisphere in a local metric CRS can allocate/work at source resolution for
    # the entire requested extent, even though the final tile is only 256 pixels.
    destination = np.full((256, 256), np.nan, dtype=np.float32)
    source = band_array(db, row, band)
    with rasterio.Env(GDAL_CACHEMAX=16 * 1024 * 1024):
        reproject(
            source=source.data,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=from_bounds(*bounds, 256, 256),
            dst_crs="EPSG:4326",
            dst_nodata=float("nan"),
            resampling=Resampling.nearest,
            num_threads=1,
            warp_mem_limit=16,
        )
    valid = np.isfinite(destination)
    if definition.palette == "categorical":
        palette = np.zeros((256, 4), dtype=np.uint8)
        for item in definition.classes:
            palette[item.code] = (*bytes.fromhex(item.color[1:]), 255)
        indices = np.where(valid, destination, 0).astype(np.uint8)
        rgba = palette[indices]
    else:
        if definition.minimum is None or definition.maximum is None:
            return transparent_tile()
        low = (
            definition.display_minimum
            if definition.display_minimum is not None
            else definition.minimum
        )
        high = (
            definition.display_maximum
            if definition.display_maximum is not None
            else definition.maximum
        )
        high = high if high > low else low + 1
        normalized = np.where(valid, (destination - low) / (high - low), 0)
        indices = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        palette = np.array(
            [cmap.get(definition.palette)[index] for index in range(256)], dtype=np.uint8
        )
        rgba = palette[indices]
    rgba[~valid, 3] = 0
    output = io.BytesIO()
    Image.fromarray(rgba).save(output, format="PNG")
    return output.getvalue()


def sample(db: Session, row: LandRaster, longitude: float, latitude: float) -> RasterSample:
    metadata = RasterMetadata.model_validate(row.metadata_json)
    values: list[float | None] = [None] * len(metadata.bands)
    boundary = db.scalar(
        select(LandBoundaryRevision).where(
            LandBoundaryRevision.land_id == row.land_id,
            LandBoundaryRevision.revision == row.boundary_revision,
        )
    )
    if boundary is not None and shape(boundary.boundary).covers(Point(longitude, latitude)):
        with MemoryFile(blob(db, row)) as memory, memory.open(driver="GTiff") as dataset:
            xs, ys = transform("EPSG:4326", dataset.crs, [longitude], [latitude])
            pixel_y, pixel_x = dataset.index(xs[0], ys[0])
            if 0 <= pixel_x < dataset.width and 0 <= pixel_y < dataset.height:
                data = dataset.read(window=Window(pixel_x, pixel_y, 1, 1), masked=True)
                values = [
                    None if data.mask[index, 0, 0] else float(data[index, 0, 0])
                    for index in range(dataset.count)
                ]
    return RasterSample(
        longitude=longitude,
        latitude=latitude,
        values=values,
        units=[band.unit for band in metadata.bands],
        resolution_m=metadata.resolution_m,
        interpretation="Nearest analysis-grid cell; not an on-site measurement. "
        "Null means outside the pinned boundary or no valid data.",
    )
