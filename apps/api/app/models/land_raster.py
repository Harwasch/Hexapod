from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKey, ForeignKeyConstraint, Integer, LargeBinary, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class LandRaster(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_rasters"
    __table_args__ = (
        ForeignKeyConstraint(
            ["land_id", "boundary_revision"],
            ["land_boundary_revisions.land_id", "land_boundary_revisions.revision"],
            ondelete="CASCADE",
        ),
    )
    land_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    boundary_revision: Mapped[int] = mapped_column(Integer)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_research_runs.id", ondelete="CASCADE"), index=True
    )
    request: Mapped[dict[str, Any]] = mapped_column(JSONB)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    sha256: Mapped[str] = mapped_column(String(64))
    byte_size: Mapped[int] = mapped_column(Integer)


class LandRasterBlob(Base):
    __tablename__ = "land_raster_blobs"
    raster_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_rasters.id", ondelete="CASCADE"), primary_key=True
    )
    data: Mapped[bytes] = mapped_column(LargeBinary)
