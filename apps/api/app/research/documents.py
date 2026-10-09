from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.research.providers.base import SourceResult
from app.schemas.land_documents import DocumentLocator
from app.schemas.research import EvidenceContent
from app.services import land_documents


def retrieve(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    *,
    query: str | None = None,
    document_id: uuid.UUID | None = None,
    first_page: int = 1,
    count: int = 1,
) -> SourceResult:
    records = []
    if query is not None:
        records = [
            hit.model_dump() for hit in land_documents.search(db, workspace_id, land_id, query)
        ]
    elif document_id is not None:
        document = land_documents.scoped(db, workspace_id, land_id, document_id)
        for number in range(first_page, min(document.page_count + 1, first_page + count)):
            page = land_documents.page(db, document, number)
            records.append(
                {
                    "document_id": document_id,
                    "page": page.page,
                    "excerpt": page.text,
                    "truncated": page.truncated,
                }
            )
    result = SourceResult(
        "land-documents",
        "available" if any(record["excerpt"].strip() for record in records) else "empty",
        "Document passages with immutable file hashes and page references. "
        "Current rights and land applicability remain unresolved.",
        data={"pages": []},
    )
    for record in records:
        document = land_documents.scoped(db, workspace_id, land_id, record["document_id"])
        if not document.sha256:
            continue
        meta = document.metadata_json
        excerpt = record["excerpt"]
        key = f"{document.id}/page/{record['page']}/{hashlib.sha256(excerpt.encode()).hexdigest()[:16]}"
        result.evidence.append(
            (
                key,
                EvidenceContent(
                    provider="land-document",
                    title=f"{meta['title']} · page {record['page']}",
                    url=meta.get("source_url"),
                    document=DocumentLocator(
                        land_id=land_id,
                        document_id=document.id,
                        page=record["page"],
                        sha256=document.sha256,
                    ),
                    license=meta["license"],
                    attribution=meta["source_note"],
                    record_id=meta.get("recording_number") or str(document.id),
                    retrieved_at=datetime.now(UTC),
                    excerpt=excerpt,
                    snapshot_hash=document.sha256,
                    spatial_relevance="unresolved",
                    relevance_note=meta["relevance_note"],
                ),
            )
        )
        result.data["pages"].append(
            {
                "documentId": str(document.id),
                "page": record["page"],
                "text": excerpt,
                "truncated": record["truncated"],
                "documentDate": meta.get("document_date"),
                "recordedDate": meta.get("recorded_date"),
                "recordingNumber": meta.get("recording_number"),
            }
        )
    return result
