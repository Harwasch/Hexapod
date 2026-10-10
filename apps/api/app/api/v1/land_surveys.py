from __future__ import annotations

import uuid

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.deps import DbSession
from app.api.workspace_deps import WorkspaceDep
from app.models.land_survey import LandSurvey
from app.schemas.land_surveys import SurveyCreate, SurveyRead, SurveySummary
from app.services import land_surveys
from app.services.land import get_land

router = APIRouter(prefix="/land/{land_id}/surveys", tags=["land ecology"])


@router.get("", response_model=list[SurveyRead])
def listing(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[SurveyRead]:
    get_land(db, scope.id, land_id)
    rows = db.scalars(
        select(LandSurvey)
        .where(LandSurvey.land_id == land_id)
        .order_by(LandSurvey.created_at.desc(), LandSurvey.id)
        .limit(limit)
        .offset(offset)
    )
    return [land_surveys.read(db, row) for row in rows]


@router.post("/preview", response_model=SurveySummary)
def preview(
    land_id: uuid.UUID, payload: SurveyCreate, db: DbSession, scope: WorkspaceDep
) -> SurveySummary:
    return land_surveys.preview(db, scope.id, land_id, payload)


@router.post("", response_model=SurveyRead, status_code=201)
def create(
    land_id: uuid.UUID, payload: SurveyCreate, db: DbSession, scope: WorkspaceDep
) -> SurveyRead:
    scope.require("owner", "editor")
    return land_surveys.create(db, scope.id, land_id, scope.principal.id, payload)


@router.get("/{survey_id}", response_model=SurveyRead)
def get(land_id: uuid.UUID, survey_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> SurveyRead:
    return land_surveys.read(db, land_surveys.scoped(db, scope.id, land_id, survey_id))
