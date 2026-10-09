from __future__ import annotations

import uuid

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.deps import DbSession
from app.api.workspace_deps import WorkspaceDep
from app.models.land_action import LandAction, LandActionRevision
from app.schemas.land_actions import (
    ActionMissionCreate,
    LandActionCreate,
    LandActionRead,
    LandActionReview,
    LandActionRevise,
)
from app.schemas.plan import PlanRead
from app.services import land_actions, plans
from app.services.errors import NotFoundError
from app.services.land import get_land

router = APIRouter(prefix="/land/{land_id}/actions", tags=["land actions"])


@router.get("", response_model=list[LandActionRead])
def listing(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[LandActionRead]:
    get_land(db, scope.id, land_id)
    rows = db.scalars(
        select(LandAction)
        .where(LandAction.land_id == land_id)
        .order_by(LandAction.updated_at.desc(), LandAction.id)
        .limit(limit)
        .offset(offset)
    )
    return [land_actions.read(db, row) for row in rows]


@router.post("", response_model=LandActionRead, status_code=201)
def create(
    land_id: uuid.UUID, payload: LandActionCreate, db: DbSession, scope: WorkspaceDep
) -> LandActionRead:
    scope.require("owner", "editor")
    return land_actions.create(db, scope.id, land_id, scope.principal.id, payload)


@router.get("/{action_id}", response_model=LandActionRead)
def get(
    land_id: uuid.UUID, action_id: uuid.UUID, db: DbSession, scope: WorkspaceDep
) -> LandActionRead:
    return land_actions.read(db, land_actions.scoped(db, scope.id, land_id, action_id))


@router.put("/{action_id}", response_model=LandActionRead)
def revise(
    land_id: uuid.UUID,
    action_id: uuid.UUID,
    payload: LandActionRevise,
    db: DbSession,
    scope: WorkspaceDep,
) -> LandActionRead:
    scope.require("owner", "editor")
    return land_actions.revise(db, scope.id, land_id, action_id, payload)


@router.get("/{action_id}/revisions", response_model=list[LandActionRead])
def revisions(
    land_id: uuid.UUID,
    action_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[LandActionRead]:
    row = land_actions.scoped(db, scope.id, land_id, action_id)
    versions = db.scalars(
        select(LandActionRevision)
        .where(LandActionRevision.action_id == action_id)
        .order_by(LandActionRevision.revision.desc())
        .limit(limit)
        .offset(offset)
    )
    return [land_actions.read(db, row, version) for version in versions]


@router.post("/{action_id}/approve", response_model=LandActionRead)
def approve(
    land_id: uuid.UUID,
    action_id: uuid.UUID,
    payload: LandActionReview,
    db: DbSession,
    scope: WorkspaceDep,
) -> LandActionRead:
    scope.require("owner", "editor")
    return land_actions.approve(db, scope.id, land_id, action_id, scope.principal.id, payload)


@router.post("/{action_id}/mission", response_model=PlanRead, status_code=201)
def schedule(
    land_id: uuid.UUID,
    action_id: uuid.UUID,
    payload: ActionMissionCreate,
    db: DbSession,
    scope: WorkspaceDep,
) -> PlanRead:
    scope.require("owner", "editor")
    return land_actions.schedule(db, scope.id, land_id, action_id, payload)


@router.get("/{action_id}/mission", response_model=PlanRead)
def mission(
    land_id: uuid.UUID,
    action_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    revision: int | None = Query(None, ge=1),
) -> PlanRead:
    row = land_actions.scoped(db, scope.id, land_id, action_id)
    version = land_actions.snapshot(db, row, revision)
    if not version.mission_id:
        raise NotFoundError("scheduled mission", action_id)
    return plans.plan_to_read(plans.get_plan(db, version.mission_id, workspace_id=scope.id))
