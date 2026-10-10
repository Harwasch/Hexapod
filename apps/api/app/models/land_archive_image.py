from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKey, Integer, LargeBinary
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class LandArchiveImage(TimestampMixin, Base):
    __tablename__ = "land_archive_images"
    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_evidence.id", ondelete="CASCADE"), primary_key=True
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    byte_size: Mapped[int] = mapped_column(Integer)


class LandArchiveImageBlob(Base):
    __tablename__ = "land_archive_image_blobs"
    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("land_archive_images.evidence_id", ondelete="CASCADE"),
        primary_key=True,
    )
    original: Mapped[bytes] = mapped_column(LargeBinary)
    preview: Mapped[bytes] = mapped_column(LargeBinary)
