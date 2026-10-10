from __future__ import annotations

import uuid
from typing import Annotated

import httpx
from fastapi import APIRouter, Form, Query, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import DbSession, SettingsDep
from app.api.workspace_deps import WorkspaceDep
from app.schemas.land import (
    BoundaryOperation,
    BoundaryResult,
    BoundaryRevisionRead,
    BoundarySplit,
    BoundarySplitResult,
    CorridorRequest,
    LandCreate,
    LandRead,
    LandRevise,
)
from app.schemas.land_import import BoundaryImportRead
from app.schemas.land_selection import (
    CandidateRequest,
    CandidateResult,
    SelectionInstruction,
    SelectionInterpretation,
)
from app.services import land, land_import, land_selection

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


@router.post("/split", response_model=BoundarySplitResult)
def split_boundary(
    payload: BoundarySplit, db: DbSession, workspace: WorkspaceDep
) -> BoundarySplitResult:
    workspace.require("owner", "editor")
    return land.split_boundary(db, payload)


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


@router.post("/selection/candidates", response_model=CandidateResult)
def selection_candidates(
    payload: CandidateRequest, db: DbSession, workspace: WorkspaceDep
) -> CandidateResult:
    workspace.require("owner", "editor")
    with httpx.Client(headers={"User-Agent": "LivingWorld-LandSelection/1.0"}) as client:
        return land_selection.candidates(db, payload, client)


@router.post("/selection/interpret", response_model=SelectionInterpretation)
def selection_interpret(
    payload: SelectionInstruction, settings: SettingsDep, workspace: WorkspaceDep
) -> SelectionInterpretation:
    workspace.require("owner", "editor")
    return land_selection.interpret(payload, settings)


@router.post("/import", response_model=BoundaryImportRead)
async def import_area(
    workspace: WorkspaceDep,
    file: UploadFile,
    source_crs: Annotated[str | None, Form(pattern=r"^EPSG:\d+$")] = None,
    layer: Annotated[str | None, Form(max_length=200)] = None,
    repair: Annotated[bool, Form()] = False,
) -> BoundaryImportRead:
    workspace.require("owner", "editor")
    data = await file.read(land_import.MAX_UPLOAD + 1)
    # Parsing/GDAL/reprojection are blocking work; keep them off the API event loop.
    return await run_in_threadpool(
        land_import.import_boundary, data, file.filename or "boundary", source_crs, layer, repair
    )
