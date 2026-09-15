from __future__ import annotations

from datetime import datetime
from typing import Any

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import Boolean, DateTime, Enum, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import LayerCategory, LayerSourceType


class Layer(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A composable world layer from the open-data catalog or the user's data."""

    __tablename__ = "layers"

    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[LayerCategory] = mapped_column(
        Enum(LayerCategory, name="layer_category", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        index=True,
    )
    source_type: Mapped[LayerSourceType] = mapped_column(
        Enum(
            LayerSourceType,
            name="layer_source_type",
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    # Discriminated source definition; see app.schemas.layer.LayerSource.
    source: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    spatial_extent: Mapped[WKBElement | None] = mapped_column(
        Geometry(geometry_type="POLYGON", srid=4326, spatial_index=True), nullable=True
    )
    temporal_extent: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolution: Mapped[str | None] = mapped_column(String(120), nullable=True)
    coverage: Mapped[str | None] = mapped_column(String(120), nullable=True)
    render: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    legend: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    attribution: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    license: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    default_visible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    builtin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
