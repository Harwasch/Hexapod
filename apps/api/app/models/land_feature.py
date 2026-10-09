from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class LandFeature(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_features"
    land_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_areas.id", ondelete="CASCADE"), index=True
    )
    geometry: Mapped[WKBElement] = mapped_column(
        Geometry("GEOMETRY", srid=4326, spatial_index=True)
    )
    revision: Mapped[int] = mapped_column(Integer, default=1)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB)


class LandFeatureRevision(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_feature_revisions"
    __table_args__ = (UniqueConstraint("feature_id", "revision"),)
    feature_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_features.id", ondelete="CASCADE")
    )
    revision: Mapped[int] = mapped_column(Integer)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB)
    note: Mapped[str] = mapped_column(String(1000))


class FeatureInspection(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_feature_inspections"
    __table_args__ = (
        ForeignKeyConstraint(
            ["feature_id", "feature_revision"],
            ["land_feature_revisions.feature_id", "land_feature_revisions.revision"],
            ondelete="CASCADE",
        ),
    )
    feature_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    feature_revision: Mapped[int] = mapped_column(Integer)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    recorded_by: Mapped[str] = mapped_column(String(64))
    content: Mapped[dict[str, Any]] = mapped_column(JSONB)


class LandFeatureBatch(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_feature_batches"
    __table_args__ = (UniqueConstraint("land_id", "request_key"),)
    land_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_areas.id", ondelete="CASCADE"), index=True
    )
    request_key: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    request_sha256: Mapped[str] = mapped_column(String(64))
    boundary_revision: Mapped[int] = mapped_column(Integer)
    source_file_sha256: Mapped[str] = mapped_column(String(64))
    source_label: Mapped[str] = mapped_column(String(200))
    results: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
