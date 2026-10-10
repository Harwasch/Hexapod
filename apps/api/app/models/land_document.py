from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class LandDocument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_documents"
    land_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_areas.id", ondelete="CASCADE"), index=True
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(30), default="awaiting-upload")
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str | None] = mapped_column(String(64))
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    extracted_characters: Mapped[int] = mapped_column(Integer, default=0)
    warnings: Mapped[list[str]] = mapped_column(JSONB, default=list)
    created_by: Mapped[str] = mapped_column(String(64))


class LandDocumentBlob(Base):
    """Small, bounded originals stay private behind workspace-authenticated routes.

    A separate relation keeps original bytes out of document lists and research contexts.
    Large raster/imagery artifacts use object storage; these originals are capped at 20 MiB.
    """

    __tablename__ = "land_document_blobs"
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_documents.id", ondelete="CASCADE"), primary_key=True
    )
    data: Mapped[bytes] = mapped_column(LargeBinary)


class LandDocumentPage(Base):
    __tablename__ = "land_document_pages"
    __table_args__ = (
        Index(
            "ix_land_document_pages_text_search",
            "text",
            postgresql_using="gin",
            postgresql_ops={"text": "gin_trgm_ops"},
        ),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_documents.id", ondelete="CASCADE"), primary_key=True
    )
    page: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    truncated: Mapped[bool] = mapped_column(Boolean, default=False)


class LandDocumentOcr(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_document_ocr"
    __table_args__ = (
        UniqueConstraint("document_id", "page", "language"),
        Index(
            "ix_land_document_ocr_text_search",
            "text",
            postgresql_using="gin",
            postgresql_ops={"text": "gin_trgm_ops"},
        ),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_documents.id", ondelete="CASCADE"), index=True
    )
    page: Mapped[int] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(20))
    text: Mapped[str] = mapped_column(Text)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB)


class LandDocumentLink(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_document_links"
    __table_args__ = (UniqueConstraint("land_id", "request_key"),)
    land_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_areas.id", ondelete="CASCADE"), index=True
    )
    request_key: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    from_document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_documents.id", ondelete="CASCADE")
    )
    to_document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_documents.id", ondelete="CASCADE")
    )
    from_document: Mapped[LandDocument] = relationship(
        foreign_keys=[from_document_id], lazy="joined"
    )
    to_document: Mapped[LandDocument] = relationship(foreign_keys=[to_document_id], lazy="joined")
    content: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_by: Mapped[str] = mapped_column(String(64))
