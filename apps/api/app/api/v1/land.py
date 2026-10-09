from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, Response

from app.api.deps import DbSession
from app.api.workspace_deps import WorkspaceDep
from app.schemas.land import (
    BoundaryOperation,
    BoundaryResult,
    BoundaryRevisionRead,
    CorridorRequest,
    LandCreate,
    LandRead,
    LandRevise,
)
from app.services import land

router = APIRouter(prefix="/land", tags=["land"])


@router.get("", response_model=list[LandRead])
def list_areas(
    db: DbSession,
    workspace: WorkspaceDep,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[LandRead]:
    return land.list_land(db, workspace.id, limit, offset)


@router.post("", response_model=LandRead, status_code=201)
def create_area(payload: LandCreate, db: DbSession, workspace: WorkspaceDep) -> LandRead:
    workspace.require("owner", "editor")
    return land.create_land(db, workspace.id, payload)


@router.post("/corridor", response_model=BoundaryResult)
def make_corridor(
    payload: CorridorRequest, db: DbSession, workspace: WorkspaceDep
) -> BoundaryResult:
    workspace.require("owner", "editor")
    return land.corridor(db, payload)


@router.post("/operations", response_model=BoundaryResult)
def operate(payload: BoundaryOperation, db: DbSession, workspace: WorkspaceDep) -> BoundaryResult:
    workspace.require("owner", "editor")
    return land.operate(db, payload)


@router.get("/{land_id}", response_model=LandRead)
def get_area(land_id: uuid.UUID, db: DbSession, workspace: WorkspaceDep) -> LandRead:
    return land.get_land(db, workspace.id, land_id)


@router.put("/{land_id}", response_model=LandRead)
def revise_area(
    land_id: uuid.UUID, payload: LandRevise, db: DbSession, workspace: WorkspaceDep
) -> LandRead:
    workspace.require("owner", "editor")
    return land.revise_land(db, workspace.id, land_id, payload)


@router.get("/{land_id}/revisions", response_model=list[BoundaryRevisionRead])
def area_revisions(
    land_id: uuid.UUID, db: DbSession, workspace: WorkspaceDep
) -> list[BoundaryRevisionRead]:
    return land.revisions(db, workspace.id, land_id)


@router.delete("/{land_id}", status_code=204)
def delete_area(land_id: uuid.UUID, db: DbSession, workspace: WorkspaceDep) -> Response:
    workspace.require("owner", "editor")
    land.delete_land(db, workspace.id, land_id)
    return Response(status_code=204)
