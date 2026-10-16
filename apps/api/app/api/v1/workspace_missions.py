"""Missions held in a workspace: plans on private ground, listed and acted on privately.

The public `/plans` routes never list a workspace's plans, on purpose; these do, for the
workspace the caller belongs to. A plan made on a saved Land is created here, linked to the
land and the boundary revision it was planned on, so it is private from the first write.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, status
from sqlalchemy import select

from app.api.deps import DbSession
from app.api.workspace_deps import WorkspaceDep
from app.models.land import LandBoundaryRevision
from app.schemas.plan import (
    PlanRead,
    PlanRevise,
    PlanStatusUpdate,
    WorkspaceMissionCreate,
)
from app.services import plans as plan_service
from app.services.errors import InvalidInputError, NotFoundError
from app.services.land import get_land

router = APIRouter(prefix="/workspaces/missions", tags=["workspace missions"])

LAND_PROJECT_PREFIX = "land:"


@router.get("", response_model=list[PlanRead], summary="List the workspace's missions")
def list_missions(
    db: DbSession,
    scope: WorkspaceDep,
    project_id: Annotated[
        str | None, Query(alias="projectId", min_length=1, max_length=120)
    ] = None,
    land_id: Annotated[uuid.UUID | None, Query(alias="landId")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 200,
) -> list[PlanRead]:
    return [
        plan_service.plan_to_read(plan)
        for plan in plan_service.list_plans(
            db, project_id, workspace_id=scope.id, land_id=land_id, limit=limit
        )
    ]


@router.post(
    "",
    response_model=PlanRead,
    status_code=status.HTTP_201_CREATED,
    summary="Approve a mission in the workspace",
)
def create_mission(payload: WorkspaceMissionCreate, db: DbSession, scope: WorkspaceDep) -> PlanRead:
    scope.require("owner", "editor")
    if payload.land_id is not None:
        get_land(db, scope.id, payload.land_id)
        if payload.boundary_revision is not None and (
            db.scalar(
                select(LandBoundaryRevision.id).where(
                    LandBoundaryRevision.land_id == payload.land_id,
                    LandBoundaryRevision.revision == payload.boundary_revision,
                )
            )
            is None
        ):
            raise NotFoundError("boundary revision", payload.boundary_revision)
    if payload.project_id.startswith(LAND_PROJECT_PREFIX) and (
        payload.land_id is None or payload.project_id != f"{LAND_PROJECT_PREFIX}{payload.land_id}"
    ):
        raise InvalidInputError("A land project's plans name that land: give its landId.")
    return plan_service.plan_to_read(
        plan_service.create_plan(
            db,
            payload,
            workspace_id=scope.id,
            land_id=payload.land_id,
            boundary_revision=payload.boundary_revision,
        )
    )


@router.get("/{plan_id}", response_model=PlanRead, summary="Get a workspace mission")
def get_mission(plan_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> PlanRead:
    return plan_service.plan_to_read(plan_service.get_plan(db, plan_id, workspace_id=scope.id))


@router.put("/{plan_id}", response_model=PlanRead, summary="Approve a new mission revision")
def revise_mission(
    plan_id: uuid.UUID, payload: PlanRevise, db: DbSession, scope: WorkspaceDep
) -> PlanRead:
    scope.require("owner", "editor")
    return plan_service.plan_to_read(
        plan_service.revise_plan(db, plan_id, payload, workspace_id=scope.id)
    )


@router.patch(
    "/{plan_id}/status", response_model=PlanRead, summary="Change a mission's lifecycle state"
)
def set_mission_status(
    plan_id: uuid.UUID, payload: PlanStatusUpdate, db: DbSession, scope: WorkspaceDep
) -> PlanRead:
    scope.require("owner", "editor")
    return plan_service.plan_to_read(
        plan_service.set_status(db, plan_id, payload.status, workspace_id=scope.id)
    )
