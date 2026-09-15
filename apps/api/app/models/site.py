from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.asset import Asset
    from app.models.bookmark import CameraBookmark


class Site(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A physical place with one or more reality captures."""

    __tablename__ = "sites"

    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    boundary: Mapped[WKBElement] = mapped_column(
        Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=True), nullable=False
    )
    centroid: Mapped[WKBElement] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=False), nullable=False
    )
    centroid_height: Mapped[float | None] = mapped_column(Float, nullable=True)
    thumbnail_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    # Free-form, versioned metadata (quality, capture platform, notes...).
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    attribution: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    license: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    assets: Mapped[list[Asset]] = relationship(
        back_populates="site", cascade="all, delete-orphan", order_by="Asset.created_at"
    )
    bookmarks: Mapped[list[CameraBookmark]] = relationship(
        back_populates="site", cascade="all, delete-orphan", order_by="CameraBookmark.created_at"
    )
