from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, Response
from sqlalchemy import select

from app.api.deps import DbSession
from app.api.workspace_deps import WorkspaceDep
from app.models.land_raster import LandRaster
from app.schemas.land_rasters import LandRasterRead, RasterSample
from app.services import land_rasters
from app.services.land import get_land

router = APIRouter(tags=["land rasters"])


@router.get("/land/{land_id}/rasters", response_model=list[LandRasterRead])
def listing(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[LandRasterRead]:
    get_land(db, scope.id, land_id)
    rows = db.scalars(
        select(LandRaster)
        .where(LandRaster.land_id == land_id)
        .order_by(LandRaster.created_at.desc(), LandRaster.id)
        .limit(limit)
        .offset(offset)
    )
    return [land_rasters.read(db, row) for row in rows]


@router.get("/land/rasters/{raster_id}", response_model=LandRasterRead)
def get(raster_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> LandRasterRead:
    return land_rasters.read(db, land_rasters.scoped(db, scope.id, raster_id))


@router.get("/land/rasters/{raster_id}/sample", response_model=RasterSample)
def sample(
    raster_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    longitude: float = Query(ge=-180, le=180),
    latitude: float = Query(ge=-90, le=90),
) -> RasterSample:
    return land_rasters.sample(
        db, land_rasters.scoped(db, scope.id, raster_id), longitude, latitude
    )


@router.get(
    "/land/rasters/{raster_id}/tiles/{band}/{z}/{x}/{y}.png",
    responses={200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}}},
)
def tile(
    raster_id: uuid.UUID, band: int, z: int, x: int, y: int, db: DbSession, scope: WorkspaceDep
) -> Response:
    data = land_rasters.tile(db, land_rasters.scoped(db, scope.id, raster_id), band, z, x, y)
    return Response(
        content=data, media_type="image/png", headers={"Cache-Control": "private, no-store"}
    )


@router.get(
    "/land/rasters/{raster_id}/download",
    responses={
        200: {"content": {"image/tiff": {"schema": {"type": "string", "format": "binary"}}}}
    },
)
def download(raster_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> Response:
    row = land_rasters.scoped(db, scope.id, raster_id)
    return Response(
        content=land_rasters.blob(db, row),
        media_type="image/tiff",
        headers={
            "Content-Disposition": f'attachment; filename="land-{row.id}.tif"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
