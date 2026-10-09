from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKeyConstraint, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class LandSurvey(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_surveys"
    __table_args__ = (
        ForeignKeyConstraint(
            ["land_id", "boundary_revision"],
            ["land_boundary_revisions.land_id", "land_boundary_revisions.revision"],
            ondelete="CASCADE",
        ),
    )
    land_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    boundary_revision: Mapped[int] = mapped_column(Integer)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB)
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB)
    sha256: Mapped[str] = mapped_column(String(64))
    recorded_by: Mapped[str] = mapped_column(String(64))
