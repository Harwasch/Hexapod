from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import Field, HttpUrl

from app.schemas.base import CamelModel


class LandDocumentCreate(CamelModel):
    request_key: uuid.UUID = Field(default_factory=uuid.uuid4)
    title: str = Field(min_length=1, max_length=300)
    filename: str = Field(min_length=1, max_length=200)
    media_type: Literal["application/pdf", "text/plain"]
    size_bytes: int = Field(gt=0, le=20 * 1024 * 1024)
    kind: Literal[
        "deed", "easement", "mineral", "water", "survey", "historical", "report", "other"
    ] = "other"
    document_date: date | None = None
    recorded_date: date | None = None
    recording_number: str = Field(default="", max_length=200)
    jurisdiction: str = Field(default="", max_length=300)
    parcel_references: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=50
    )
    source_url: HttpUrl | None = None
    source_note: str = Field(min_length=1, max_length=2000)
    relevance_note: str = Field(
        default="Applicability to the selected land has not been established.",
        min_length=1,
        max_length=2000,
    )
    license: str = Field(
        default="User-provided record; workspace access only. Reuse rights unverified.",
        min_length=1,
        max_length=1000,
    )


class LandDocumentRead(LandDocumentCreate):
    id: uuid.UUID
    land_id: uuid.UUID
    status: Literal["awaiting-upload", "ready", "needs-ocr", "unreadable"]
    sha256: str | None
    page_count: int
    extracted_characters: int
    warnings: list[str]
    created_at: datetime


class DocumentPageRead(CamelModel):
    document_id: uuid.UUID
    page: int
    text: str
    truncated: bool
    sha256: str


class DocumentSearchHit(CamelModel):
    document_id: uuid.UUID
    title: str
    page: int
    excerpt: str
    truncated: bool


class DocumentLinkCreate(CamelModel):
    request_key: uuid.UUID = Field(default_factory=uuid.uuid4)
    from_document_id: uuid.UUID
    from_page: int = Field(ge=1, le=500)
    to_document_id: uuid.UUID
    to_page: int = Field(ge=1, le=500)
    relation: Literal[
        "amends", "supersedes", "conflicts-with", "mentions", "parcel-lineage", "related"
    ]
    basis: str = Field(min_length=1, max_length=3000)
    effective_date: date | None = None


class DocumentLinkRead(DocumentLinkCreate):
    from_title: str
    to_title: str
    id: uuid.UUID
    land_id: uuid.UUID
    created_by: str
    created_at: datetime


class DocumentLocator(CamelModel):
    land_id: uuid.UUID
    document_id: uuid.UUID
    page: int = Field(ge=1, le=500)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
