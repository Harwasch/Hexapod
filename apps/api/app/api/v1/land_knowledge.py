"""The Fog of Knowing over a saved Land: read-only computation, so viewers may ask too."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, Response

from app.api.deps import DbSession
from app.api.workspace_deps import WorkspaceDep
from app.schemas.land_knowledge import (
    KnowledgeCellRead,
    KnowledgeGridRead,
    KnowledgeScoutRead,
    KnowledgeScoutRequest,
)
from app.services import land_knowledge

router = APIRouter(prefix="/land/{land_id}/knowledge", tags=["land knowledge"])

CellM = Annotated[float, Query(alias="cellM", ge=10, le=100)]
AsOf = Annotated[datetime | None, Query(alias="asOf")]


@router.get("", response_model=KnowledgeGridRead)
def grid(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    response: Response,
    cell_m: CellM = 20,
    as_of: AsOf = None,
) -> KnowledgeGridRead:
    response.headers["Cache-Control"] = "private, no-store"
    return land_knowledge.knowledge_grid(db, scope.id, land_id, cell_m, as_of)


@router.get("/cell", response_model=KnowledgeCellRead)
def cell(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    response: Response,
    longitude: Annotated[float, Query(ge=-180, le=180)],
    latitude: Annotated[float, Query(ge=-90, le=90)],
    cell_m: CellM = 20,
    as_of: AsOf = None,
) -> KnowledgeCellRead:
    response.headers["Cache-Control"] = "private, no-store"
    return land_knowledge.knowledge_cell(db, scope.id, land_id, longitude, latitude, cell_m, as_of)


@router.post("/scout-areas", response_model=KnowledgeScoutRead)
def scout_areas(
    land_id: uuid.UUID,
    body: KnowledgeScoutRequest,
    db: DbSession,
    scope: WorkspaceDep,
    response: Response,
) -> KnowledgeScoutRead:
    response.headers["Cache-Control"] = "private, no-store"
    return land_knowledge.scout_areas(db, scope.id, land_id, body)
