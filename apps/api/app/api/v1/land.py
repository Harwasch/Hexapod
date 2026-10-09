from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, Response

from app.api.deps import DbSession, RequireWriteToken
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

# Land records are NOT part of the public catalog. During the single-operator pilot,
# reads and spatial computation require the existing token too. Workspace identities
# and membership authorization must replace this before multi-user availability.
router = APIRouter(prefix="/land", tags=["land"], dependencies=[RequireWriteToken])


@router.get("", response_model=list[LandRead])
def list_areas(
    db: DbSession,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[LandRead]:
    return land.list_land(db, limit, offset)


@router.post("", response_model=LandRead, status_code=201)
def create_area(payload: LandCreate, db: DbSession) -> LandRead:
    return land.create_land(db, payload)


@router.post("/corridor", response_model=BoundaryResult)
def make_corridor(payload: CorridorRequest, db: DbSession) -> BoundaryResult:
    return land.corridor(db, payload)


@router.post("/operations", response_model=BoundaryResult)
def operate(payload: BoundaryOperation, db: DbSession) -> BoundaryResult:
    return land.operate(db, payload)


@router.get("/{land_id}", response_model=LandRead)
def get_area(land_id: uuid.UUID, db: DbSession) -> LandRead:
    return land.get_land(db, land_id)


@router.put("/{land_id}", response_model=LandRead)
def revise_area(land_id: uuid.UUID, payload: LandRevise, db: DbSession) -> LandRead:
    return land.revise_land(db, land_id, payload)


@router.get("/{land_id}/revisions", response_model=list[BoundaryRevisionRead])
def area_revisions(land_id: uuid.UUID, db: DbSession) -> list[BoundaryRevisionRead]:
    return land.revisions(db, land_id)


@router.delete("/{land_id}", status_code=204)
def delete_area(land_id: uuid.UUID, db: DbSession) -> Response:
    land.delete_land(db, land_id)
    return Response(status_code=204)
