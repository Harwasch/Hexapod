from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import DbSession, Storage
from app.schemas.base import CamelModel

router = APIRouter(tags=["system"])


class HealthStatus(CamelModel):
    status: str
    database: bool
    postgis_version: str | None = None
    object_storage: str
    object_storage_available: bool


@router.get("/health", response_model=HealthStatus, summary="Health check")
def health(db: DbSession, storage: Storage) -> HealthStatus:
    database = False
    postgis_version: str | None = None
    try:
        postgis_version = db.execute(text("SELECT PostGIS_Lib_Version()")).scalar_one()
        database = True
    except Exception:
        database = False
    return HealthStatus(
        status="ok" if database else "degraded",
        database=database,
        postgis_version=postgis_version,
        object_storage=storage.name,
        object_storage_available=storage.available,
    )
