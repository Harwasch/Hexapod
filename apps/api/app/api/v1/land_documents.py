from __future__ import annotations

import uuid
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Request, Response
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.api.deps import DbSession, SettingsDep
from app.api.workspace_deps import WorkspaceDep
from app.models.land_document import LandDocument, LandDocumentBlob, LandDocumentLink
from app.schemas.land_documents import (
    DocumentLinkCreate,
    DocumentLinkRead,
    DocumentPageRead,
    DocumentSearchHit,
    LandDocumentCreate,
    LandDocumentRead,
)
from app.services import land_documents
from app.services.errors import NotFoundError
from app.services.land import get_land

router = APIRouter(prefix="/land/{land_id}/documents", tags=["land documents"])


@router.post("", response_model=LandDocumentRead, status_code=201)
def initiate(
    land_id: uuid.UUID,
    payload: LandDocumentCreate,
    db: DbSession,
    scope: WorkspaceDep,
    settings: SettingsDep,
) -> LandDocumentRead:
    scope.require("owner", "editor")
    return land_documents.create(
        db,
        scope.id,
        land_id,
        scope.principal.id,
        payload,
        settings.land_document_workspace_quota_bytes,
    )


@router.get("", response_model=list[LandDocumentRead])
def listing(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[LandDocumentRead]:
    get_land(db, scope.id, land_id)
    rows = db.scalars(
        select(LandDocument)
        .where(LandDocument.land_id == land_id)
        .order_by(LandDocument.created_at.desc(), LandDocument.id)
        .limit(limit)
        .offset(offset)
    )
    return [land_documents.read(row) for row in rows]


@router.get("/search", response_model=list[DocumentSearchHit])
def search(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    q: str = Query(min_length=3, max_length=200),
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
) -> list[DocumentSearchHit]:
    return land_documents.search(db, scope.id, land_id, q, limit, offset)


@router.post("/links", response_model=DocumentLinkRead, status_code=201)
def link(
    land_id: uuid.UUID, payload: DocumentLinkCreate, db: DbSession, scope: WorkspaceDep
) -> DocumentLinkRead:
    scope.require("owner", "editor")
    return land_documents.link(db, scope.id, land_id, scope.principal.id, payload)


@router.get("/links", response_model=list[DocumentLinkRead])
def links(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[DocumentLinkRead]:
    get_land(db, scope.id, land_id)
    rows = db.scalars(
        select(LandDocumentLink)
        .where(LandDocumentLink.land_id == land_id)
        .order_by(LandDocumentLink.created_at.desc(), LandDocumentLink.id)
        .limit(limit)
        .offset(offset)
    )
    return [land_documents.link_read(row) for row in rows]


@router.get("/{document_id}", response_model=LandDocumentRead)
def get(
    land_id: uuid.UUID, document_id: uuid.UUID, db: DbSession, scope: WorkspaceDep
) -> LandDocumentRead:
    return land_documents.read(land_documents.scoped(db, scope.id, land_id, document_id))


@router.put(
    "/{document_id}/content",
    response_model=LandDocumentRead,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            },
        }
    },
)
async def upload(
    land_id: uuid.UUID, document_id: uuid.UUID, request: Request, db: DbSession, scope: WorkspaceDep
) -> LandDocumentRead:
    scope.require("owner", "editor")
    row = land_documents.scoped(db, scope.id, land_id, document_id)
    declared_size = row.size_bytes
    db.rollback()
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > min(declared_size, land_documents.MAX_UPLOAD):
            raise HTTPException(413, "The document exceeds its declared size or 20 MiB limit.")
        data.extend(chunk)
    return await run_in_threadpool(
        land_documents.complete, db, scope.id, land_id, document_id, bytes(data)
    )


@router.get(
    "/{document_id}/content",
    responses={
        200: {
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            }
        }
    },
)
def download(
    land_id: uuid.UUID, document_id: uuid.UUID, db: DbSession, scope: WorkspaceDep
) -> Response:
    row = land_documents.scoped(db, scope.id, land_id, document_id)
    original = db.get(LandDocumentBlob, document_id)
    if original is None:
        raise NotFoundError("document original", document_id)
    filename = quote(row.metadata_json["filename"], safe="")
    return Response(
        content=original.data,
        media_type=row.metadata_json["media_type"],
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-store",
        },
    )


@router.get("/{document_id}/pages/{page}", response_model=DocumentPageRead)
def page(
    land_id: uuid.UUID, document_id: uuid.UUID, page: int, db: DbSession, scope: WorkspaceDep
) -> DocumentPageRead:
    return land_documents.page(db, land_documents.scoped(db, scope.id, land_id, document_id), page)
