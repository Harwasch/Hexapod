from __future__ import annotations

import hashlib
import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Response
from sqlalchemy import select

from app.api.deps import DbSession
from app.api.workspace_deps import WorkspaceDep
from app.models.land_view import LandView
from app.schemas.land_views import LandViewCreate, LandViewOpen, LandViewRead, LandViewRename
from app.services import land_views
from app.services.errors import ConflictError, NotFoundError

router = APIRouter(prefix="/land/{land_id}/views", tags=["land views"])


@router.get("", response_model=list[LandViewRead])
def listing(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[LandViewRead]:
    land_views.area(db, scope.id, land_id)
    rows = db.scalars(
        select(LandView)
        .where(LandView.land_id == land_id)
        .order_by(LandView.created_at.desc(), LandView.id)
        .offset(offset)
        .limit(limit)
    )
    return [LandViewRead.model_validate(row) for row in rows]


@router.post("", response_model=LandViewRead, status_code=201)
def create(
    land_id: uuid.UUID, payload: LandViewCreate, db: DbSession, scope: WorkspaceDep
) -> LandViewRead:
    scope.require("owner", "editor")
    return land_views.create(db, scope.id, land_id, payload)


@router.post("/recover", response_model=LandViewRead)
def recover(
    land_id: uuid.UUID, payload: LandViewCreate, db: DbSession, scope: WorkspaceDep
) -> LandViewRead:
    """Read-only reconciliation; a request body keeps captured context out of URL logs."""
    land_views.area(db, scope.id, land_id)
    row = db.scalar(
        select(LandView).where(
            LandView.land_id == land_id, LandView.request_key == payload.request_key
        )
    )
    if row is None:
        raise NotFoundError("saved view request", payload.request_key)
    if row.request_sha256 != hashlib.sha256(payload.model_dump_json().encode()).hexdigest():
        raise ConflictError(
            "This request saved a different capture. Download your local capture and review the saved views."
        )
    return LandViewRead.model_validate(row)


@router.get("/{view_id}", response_model=LandViewOpen)
def open_view(
    land_id: uuid.UUID, view_id: uuid.UUID, db: DbSession, scope: WorkspaceDep
) -> LandViewOpen:
    row = land_views.scoped(db, scope.id, land_id, view_id)
    return land_views.resolve(
        db, land_views.area(db, scope.id, land_id), LandViewRead.model_validate(row)
    )


@router.patch("/{view_id}", response_model=LandViewRead)
def rename(
    land_id: uuid.UUID,
    view_id: uuid.UUID,
    payload: LandViewRename,
    db: DbSession,
    scope: WorkspaceDep,
) -> LandViewRead:
    scope.require("owner", "editor")
    row = land_views.scoped(db, scope.id, land_id, view_id, lock=True)
    if row.revision != payload.expected_revision:
        raise ConflictError(
            "This view was renamed elsewhere. Refresh saved views before changing it."
        )
    row.name, row.revision = payload.name, row.revision + 1
    db.commit()
    return LandViewRead.model_validate(row)


@router.delete("/{view_id}", status_code=204)
def remove(
    land_id: uuid.UUID,
    view_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    expected_revision: Annotated[int, Query(ge=1)],
) -> Response:
    scope.require("owner", "editor")
    row = land_views.scoped(db, scope.id, land_id, view_id, lock=True)
    if row.revision != expected_revision:
        raise ConflictError("This view changed elsewhere. Refresh saved views before deleting it.")
    db.delete(row)
    db.commit()
    return Response(status_code=204)
