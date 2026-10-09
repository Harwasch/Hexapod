from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, Response
from sqlalchemy import select

from app.analysis.solar import location
from app.api.deps import DbSession
from app.api.workspace_deps import WorkspaceDep
from app.models.land_solar import LandSolar
from app.schemas.land_solar import SolarAssessmentRead, SolarPreview, SolarRequest
from app.services import land_solar
from app.services.errors import InvalidInputError
from app.services.land import FOOTPRINT, get_land

router = APIRouter(tags=["land solar"])


@router.get("/land/{land_id}/solar-assessments", response_model=list[SolarAssessmentRead])
def listing(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[SolarAssessmentRead]:
    get_land(db, scope.id, land_id)
    rows = db.scalars(
        select(LandSolar)
        .where(LandSolar.land_id == land_id)
        .order_by(LandSolar.created_at.desc(), LandSolar.id)
        .limit(limit)
        .offset(offset)
    )
    return [land_solar.read(db, row) for row in rows]


@router.get("/land/solar-assessments/{assessment_id}", response_model=SolarAssessmentRead)
def get(assessment_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> SolarAssessmentRead:
    return land_solar.read(db, land_solar.scoped(db, scope.id, assessment_id))


@router.get(
    "/land/solar-assessments/{assessment_id}/download",
    responses={
        200: {"content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}}}
    },
)
def download(assessment_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> Response:
    row = land_solar.scoped(db, scope.id, assessment_id)
    return Response(
        content=land_solar.blob(db, row),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="land-{row.id}.zip"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/land/{land_id}/solar-assessments/preview", response_model=SolarPreview)
def preview(
    land_id: uuid.UUID, payload: SolarRequest, db: DbSession, scope: WorkspaceDep
) -> SolarPreview:
    land = get_land(db, scope.id, land_id)
    try:
        longitude, latitude, area = location(FOOTPRINT.validate_python(land.boundary), payload)
    except ValueError as error:
        raise InvalidInputError(str(error)) from error
    capacity = payload.module_area_m2 * payload.module_efficiency
    return SolarPreview(
        mapped_zone_area_m2=area,
        capacity_kw_dc=capacity,
        inverter_kw_ac=capacity / payload.dc_ac_ratio,
        longitude=longitude,
        latitude=latitude,
    )
