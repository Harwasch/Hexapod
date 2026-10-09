from __future__ import annotations

import uuid
from typing import Any

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class LandArea(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An enduring area of interest; not a mission zone or a claim of ownership."""

    __tablename__ = "land_areas"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    boundary: Mapped[WKBElement] = mapped_column(
        Geometry("MULTIPOLYGON", srid=4326, spatial_index=True), nullable=False
    )
    source: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class LandBoundaryRevision(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Append-only snapshots. Research will pin (land_id, revision)."""

    __tablename__ = "land_boundary_revisions"
    __table_args__ = (UniqueConstraint("land_id", "revision"),)

    land_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_areas.id", ondelete="CASCADE"), nullable=False
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    boundary: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    source: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    note: Mapped[str] = mapped_column(String(500), nullable=False)
