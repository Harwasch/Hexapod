"""Scan-local observations and completed answer replay ledger."""

import uuid
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import CheckConstraint, ForeignKey, Index, String, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class AskRun(TimestampMixin, Base):
    __tablename__ = "ask_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    capture_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("captures.id", ondelete="CASCADE"), index=True
    )
    site_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sites.id", ondelete="SET NULL"), index=True
    )
    request: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    events: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    __table_args__ = (CheckConstraint("state IN ('running','complete','failed')"),)


class ScanEntity(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "scan_entities"
    capture_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("captures.id", ondelete="CASCADE"), nullable=False
    )
    site_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sites.id", ondelete="SET NULL"), index=True
    )
    ask_id: Mapped[str | None] = mapped_column(
        ForeignKey("ask_runs.id", ondelete="SET NULL"), index=True
    )
    world_anchor: Mapped[Any] = mapped_column(
        Geometry("POINTZ", dimension=3, spatial_index=False), nullable=False
    )
    geo_anchor: Mapped[Any | None] = mapped_column(
        Geometry("POINTZ", srid=4326, dimension=3, spatial_index=False)
    )
    # The unchanged transport object retains altitude, masks, extent and provenance.
    entity: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    __table_args__ = (
        Index("ix_scan_entities_capture_id", "capture_id"),
        Index("ix_scan_entities_geo_anchor", "geo_anchor", postgresql_using="gist"),
        # Property memory (0040): attribute filters, metre distances and per-site history.
        Index("ix_scan_entities_attrs", text("(entity -> 'attrs')"), postgresql_using="gin"),
        Index("ix_scan_entities_geography", text("geography(geo_anchor)"), postgresql_using="gist"),
        Index("ix_scan_entities_site_time", "site_id", "created_at", "id"),
    )
