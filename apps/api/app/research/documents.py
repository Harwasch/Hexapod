from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.research.providers.base import SourceResult
from app.schemas.land_documents import DocumentLocator
from app.schemas.research import EvidenceContent
from app.services import document_ocr, land_documents


def retrieve(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    *,
    query: str | None = None,
    document_id: uuid.UUID | None = None,
    first_page: int = 1,
    count: int = 1,
    ocr_id: uuid.UUID | None = None,
) -> SourceResult:
    records: list[dict[str, Any]] = []
    if ocr_id is not None and document_id is not None:
        ocr = document_ocr.read(document_ocr.scoped(db, workspace_id, land_id, document_id, ocr_id))
        records = [
            {
                "document_id": document_id,
                "page": ocr.page,
                "excerpt": ocr.text,
                "truncated": ocr.truncated,
                "ocr_id": ocr.id,
            }
        ]
    elif query is not None:
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
        extraction_id = record.get("ocr_id")
        passage_hash = hashlib.sha256(excerpt.encode()).hexdigest()[:16]
        key = f"{document.id}/page/{record['page']}/{extraction_id or 'native'}/{passage_hash}"
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
                        ocr_id=extraction_id,
                    ),
                    license=meta["license"],
                    attribution=meta["source_note"],
                    record_id=meta.get("recording_number") or str(document.id),
                    retrieved_at=datetime.now(UTC),
                    excerpt=excerpt,
                    snapshot_hash=document.sha256,
                    spatial_relevance="unresolved",
                    relevance_note=meta["relevance_note"][:1800]
                    + (
                        " Machine OCR; verify names, numbers and rights language against the original."
                        if extraction_id
                        else ""
                    ),
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
                "extraction": "machine-ocr" if extraction_id else "native-text",
                "ocrId": str(extraction_id) if extraction_id else None,
            }
        )
    return result
