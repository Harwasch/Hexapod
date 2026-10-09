from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class LandView(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_views"
    __table_args__ = (UniqueConstraint("land_id", "request_key"),)

    land_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_areas.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(160))
    revision: Mapped[int] = mapped_column(Integer, default=1)
    request_key: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    request_sha256: Mapped[str] = mapped_column(String(64))
    state: Mapped[dict[str, Any]] = mapped_column(JSONB)
