from __future__ import annotations

import uuid

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.deps import DbSession
from app.api.workspace_deps import WorkspaceDep
from app.models.scenario import LandScenario, LandScenarioRevision
from app.schemas.scenarios import (
    ScenarioCreate,
    ScenarioRead,
    ScenarioRequestRead,
    ScenarioResult,
    ScenarioRevise,
)
from app.services import scenarios
from app.services.errors import NotFoundError
from app.services.land import get_land

router = APIRouter(prefix="/land/{land_id}/scenarios", tags=["land scenarios"])


@router.post("/preview", response_model=ScenarioResult)
def preview(
    land_id: uuid.UUID, payload: ScenarioCreate, db: DbSession, scope: WorkspaceDep
) -> ScenarioResult:
    scope.require("owner", "editor")
    return scenarios.preview(db, scope.id, land_id, payload)


@router.post("", response_model=ScenarioRead, status_code=201)
def create(
    land_id: uuid.UUID, payload: ScenarioCreate, db: DbSession, scope: WorkspaceDep
) -> ScenarioRead:
    scope.require("owner", "editor")
    return scenarios.create(db, scope.id, land_id, scope.principal.id, payload)


@router.get("", response_model=list[ScenarioRead])
def listing(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[ScenarioRead]:
    get_land(db, scope.id, land_id)
    rows = db.scalars(
        select(LandScenario)
        .where(LandScenario.land_id == land_id)
        .order_by(LandScenario.updated_at.desc(), LandScenario.id)
        .limit(limit)
        .offset(offset)
    )
    return [scenarios.read(db, row) for row in rows]


@router.get("/requests/{request_key}", response_model=ScenarioRequestRead)
def request_read(
    land_id: uuid.UUID, request_key: uuid.UUID, db: DbSession, scope: WorkspaceDep
) -> ScenarioRequestRead:
    return scenarios.request_read(db, scope.id, land_id, request_key)


@router.get("/{scenario_id}", response_model=ScenarioRead)
def get(
    land_id: uuid.UUID,
    scenario_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    revision: int | None = Query(None, ge=1),
) -> ScenarioRead:
    row = scenarios.scoped(db, scope.id, land_id, scenario_id)
    if revision is None:
        return scenarios.read(db, row)
    snapshot = db.scalar(
        select(LandScenarioRevision).where(
            LandScenarioRevision.scenario_id == scenario_id,
            LandScenarioRevision.revision == revision,
        )
    )
    if snapshot is None:
        raise NotFoundError("scenario revision", revision)
    return scenarios.read(db, row, snapshot)


@router.put("/{scenario_id}", response_model=ScenarioRead)
def revise(
    land_id: uuid.UUID,
    scenario_id: uuid.UUID,
    payload: ScenarioRevise,
    db: DbSession,
    scope: WorkspaceDep,
) -> ScenarioRead:
    scope.require("owner", "editor")
    return scenarios.revise(db, scope.id, land_id, scenario_id, payload)


@router.get("/{scenario_id}/revisions", response_model=list[ScenarioRead])
def history(
    land_id: uuid.UUID,
    scenario_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[ScenarioRead]:
    row = scenarios.scoped(db, scope.id, land_id, scenario_id)
    snapshots = db.scalars(
        select(LandScenarioRevision)
        .where(LandScenarioRevision.scenario_id == scenario_id)
        .order_by(LandScenarioRevision.revision.desc())
        .limit(limit)
        .offset(offset)
    )
    return [scenarios.read(db, row, snapshot) for snapshot in snapshots]
