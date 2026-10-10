from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import Uuid, delete, func, literal, select, union_all
from sqlalchemy.orm import Session

from app.models.land import LandArea
from app.models.land_document import (
    LandDocument,
    LandDocumentBlob,
    LandDocumentLink,
    LandDocumentOcr,
    LandDocumentPage,
)
from app.models.workspace import Workspace
from app.schemas.land_documents import (
    DocumentLinkCreate,
    DocumentLinkRead,
    DocumentPageRead,
    DocumentSearchHit,
    LandDocumentCreate,
    LandDocumentRead,
)
from app.services.errors import ConflictError, InvalidInputError, NotFoundError
from app.services.land import get_land

MAX_UPLOAD = 20 * 1024 * 1024


def scoped(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    identifier: uuid.UUID,
    *,
    lock: bool = False,
) -> LandDocument:
    query = (
        select(LandDocument)
        .join(LandArea, LandArea.id == LandDocument.land_id)
        .where(
            LandArea.workspace_id == workspace_id,
            LandDocument.land_id == land_id,
            LandDocument.id == identifier,
        )
    )
    if lock:
        query = query.with_for_update(of=LandDocument).execution_options(populate_existing=True)
    row = db.scalar(query)
    if row is None:
        raise NotFoundError("land document", identifier)
    return row


def read(row: LandDocument) -> LandDocumentRead:
    return LandDocumentRead(
        **row.metadata_json,
        id=row.id,
        land_id=row.land_id,
        status=row.status,
        sha256=row.sha256,
        page_count=row.page_count,
        extracted_characters=row.extracted_characters,
        warnings=row.warnings,
        created_at=row.created_at,
    )


def create(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    principal: str,
    payload: LandDocumentCreate,
    quota_bytes: int,
) -> LandDocumentRead:
    get_land(db, workspace_id, land_id)
    db.execute(select(Workspace.id).where(Workspace.id == workspace_id).with_for_update())
    # Unfinished uploads reserve quota for a day; no original or extracted evidence exists yet.
    stale = (
        select(LandDocument.id)
        .join(LandArea, LandArea.id == LandDocument.land_id)
        .where(
            LandArea.workspace_id == workspace_id,
            LandDocument.status == "awaiting-upload",
            LandDocument.created_at < func.now() - timedelta(days=1),
        )
    )
    db.execute(delete(LandDocument).where(LandDocument.id.in_(stale)))
    identifier = uuid.uuid5(land_id, f"document/{payload.request_key}")
    existing = db.get(LandDocument, identifier)
    content = payload.model_dump(mode="json")
    if existing:
        if existing.metadata_json != content:
            raise ConflictError("This document upload request already has different metadata.")
        return read(existing)
    used = (
        db.scalar(
            select(func.coalesce(func.sum(LandDocument.size_bytes), 0))
            .join(LandArea, LandArea.id == LandDocument.land_id)
            .where(LandArea.workspace_id == workspace_id)
        )
        or 0
    )
    if used + payload.size_bytes > quota_bytes:
        raise InvalidInputError("This workspace's document storage allowance is full.")
    row = LandDocument(
        id=identifier,
        land_id=land_id,
        metadata_json=content,
        size_bytes=payload.size_bytes,
        created_by=principal,
        status="awaiting-upload",
    )
    db.add(row)
    db.commit()
    return read(row)


def parse(data: bytes, media_type: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="land-record-") as temporary:
        source = Path(temporary) / "original"
        source.write_bytes(data)
        try:
            result = subprocess.run(  # noqa: S603 -- trusted executable/module, generated path, no shell
                [sys.executable, "-m", "app.analysis.document_parser", str(source), media_type],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
            if result.returncode != 0:
                raise ValueError("Parser resource limit reached")
            parsed: dict[str, Any] = json.loads(result.stdout)
            return parsed
        except (subprocess.TimeoutExpired, ValueError):
            return {
                "status": "unreadable",
                "pages": [],
                "page_count": 0,
                "characters": 0,
                "warnings": [
                    "Extraction exceeded its time or memory budget. The original is preserved; "
                    "split complex documents into smaller files."
                ],
            }


def complete(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, identifier: uuid.UUID, data: bytes
) -> LandDocumentRead:
    row = scoped(db, workspace_id, land_id, identifier, lock=True)
    if len(data) != row.size_bytes or len(data) > MAX_UPLOAD:
        raise InvalidInputError(
            "Uploaded bytes must match the declared file size and the 20 MiB limit."
        )
    digest = hashlib.sha256(data).hexdigest()
    if row.sha256:
        if row.sha256 != digest:
            raise ConflictError(
                "Document originals are immutable. Add a new record for a different file."
            )
        return read(row)
    extracted = parse(data, row.metadata_json["media_type"])
    db.add(LandDocumentBlob(document_id=row.id, data=data))
    for page in extracted["pages"]:
        db.add(
            LandDocumentPage(
                document_id=row.id,
                page=page["page"],
                text=page["text"],
                truncated=page["truncated"],
            )
        )
    row.status = extracted["status"]
    row.sha256 = digest
    row.page_count = extracted["page_count"]
    row.extracted_characters = extracted["characters"]
    row.warnings = extracted["warnings"]
    db.commit()
    return read(row)


def page(db: Session, row: LandDocument, number: int) -> DocumentPageRead:
    stored = db.get(LandDocumentPage, (row.id, number))
    if stored is None or row.sha256 is None:
        raise NotFoundError("extracted document page", number)
    return DocumentPageRead(
        document_id=row.id,
        page=stored.page,
        text=stored.text,
        truncated=stored.truncated,
        sha256=row.sha256,
    )


def search(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    query: str,
    limit: int = 20,
    offset: int = 0,
) -> list[DocumentSearchHit]:
    get_land(db, workspace_id, land_id)
    passages = union_all(
        select(
            LandDocumentPage.document_id,
            LandDocumentPage.page,
            LandDocumentPage.text,
            LandDocumentPage.truncated,
            literal(None, type_=Uuid).label("ocr_id"),
        ),
        select(
            LandDocumentOcr.document_id,
            LandDocumentOcr.page,
            LandDocumentOcr.text,
            LandDocumentOcr.content["truncated"].as_boolean(),
            LandDocumentOcr.id,
        ),
    ).subquery()
    rows = db.execute(
        select(
            LandDocument, passages.c.page, passages.c.text, passages.c.truncated, passages.c.ocr_id
        )
        .join(passages, passages.c.document_id == LandDocument.id)
        .where(LandDocument.land_id == land_id, passages.c.text.icontains(query, autoescape=True))
        .order_by(
            LandDocument.created_at.desc(), LandDocument.id, passages.c.page, passages.c.ocr_id
        )
        .limit(limit)
        .offset(offset)
    )
    hits = []
    for document, page_number, page_text, truncated, ocr_id in rows:
        index = page_text.casefold().find(query.casefold())
        start = max(0, index - 300)
        hits.append(
            DocumentSearchHit(
                document_id=document.id,
                title=document.metadata_json["title"],
                page=page_number,
                excerpt=page_text[start : start + 1500],
                truncated=truncated,
                ocr_id=ocr_id,
            )
        )
    return hits


def link_read(row: LandDocumentLink) -> DocumentLinkRead:
    return DocumentLinkRead(
        **row.content,
        from_title=row.from_document.metadata_json["title"],
        to_title=row.to_document.metadata_json["title"],
        id=row.id,
        land_id=row.land_id,
        created_by=row.created_by,
        created_at=row.created_at,
    )


def link(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    principal: str,
    payload: DocumentLinkCreate,
) -> DocumentLinkRead:
    source = scoped(db, workspace_id, land_id, payload.from_document_id)
    target = scoped(db, workspace_id, land_id, payload.to_document_id)
    page(db, source, payload.from_page)
    page(db, target, payload.to_page)
    if source.id == target.id and payload.from_page == payload.to_page:
        raise InvalidInputError("Choose two different documents or pages to relate.")
    db.execute(select(LandArea.id).where(LandArea.id == land_id).with_for_update())
    identifier = uuid.uuid5(land_id, f"document-link/{payload.request_key}")
    content = payload.model_dump(mode="json")
    existing = db.get(LandDocumentLink, identifier)
    if existing:
        if existing.content != content:
            raise ConflictError(
                "This relationship request was already used with different details."
            )
        return link_read(existing)
    row = LandDocumentLink(
        id=identifier,
        land_id=land_id,
        request_key=payload.request_key,
        from_document_id=source.id,
        to_document_id=target.id,
        content=content,
        created_by=principal,
    )
    db.add(row)
    db.commit()
    return link_read(row)
