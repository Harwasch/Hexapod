from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import AssetProvider, Representation

if TYPE_CHECKING:
    from app.models.site import Site


class Asset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A derived, renderable delivery asset (3D Tiles today) with provenance."""

    __tablename__ = "assets"

    site_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sites.id", ondelete="CASCADE"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    representation: Mapped[Representation] = mapped_column(
        Enum(Representation, name="representation", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    provider: Mapped[AssetProvider] = mapped_column(
        Enum(AssetProvider, name="asset_provider", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    # Provider-specific locator: {"type": "cesium-ion", "assetId": 123} or {"type": "3d-tiles-url", "url": "..."}
    source: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    footprint: Mapped[WKBElement | None] = mapped_column(
        Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=True), nullable=True
    )
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolution: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    crs: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    license: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    attribution: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    render_config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    default_visible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    site: Mapped[Site | None] = relationship(back_populates="assets")
