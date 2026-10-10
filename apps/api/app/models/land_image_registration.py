from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKey, Integer, LargeBinary, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class LandImageRegistration(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_image_registrations"
    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("land_archive_images.evidence_id", ondelete="CASCADE"),
        index=True,
    )
    request: Mapped[dict[str, Any]] = mapped_column(JSONB)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB)
    sha256: Mapped[str] = mapped_column(String(64))
    byte_size: Mapped[int] = mapped_column(Integer)
    display_width: Mapped[int] = mapped_column(Integer)
    display_height: Mapped[int] = mapped_column(Integer)


class LandImageRegistrationBlob(Base):
    __tablename__ = "land_image_registration_blobs"
    registration_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("land_image_registrations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    data: Mapped[bytes] = mapped_column(LargeBinary)
