from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import CheckConstraint, Float, Index, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import SiteBoundarySource

if TYPE_CHECKING:
    from app.models.asset import Asset
    from app.models.bookmark import CameraBookmark
    from app.models.epoch import Epoch, SiteFrame


class Site(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A physical place with one or more reality captures."""

    __tablename__ = "sites"
    __table_args__ = (
        # A distance in metres on the ellipsoid (`services/epochs.nearest_framed_site`) is
        # on `geography(boundary)`, which the geometry column's gist index cannot answer.
        # Migration 0032 builds it CONCURRENTLY.
        Index("ix_sites_boundary_geography", text("geography(boundary)"), postgresql_using="gist"),
        CheckConstraint(
            f"boundary_source IN ({', '.join(repr(v.value) for v in SiteBoundarySource)})",
            name="ck_sites_boundary_source",
        ),
    )

    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    boundary: Mapped[WKBElement] = mapped_column(
        Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=True), nullable=False
    )
    #: Who drew the boundary (`SiteBoundarySource`, migration 0038): only `registration`'s may
    #: shrink when a re-run measures less; a `manual` one is only grown.
    boundary_source: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=SiteBoundarySource.MANUAL.value,
        server_default=SiteBoundarySource.MANUAL.value,
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
    # The site's fixed frame and its epochs (decision D3, migration 0032). A legacy site --
    # seeded, catalog, hand-made -- has neither.
    frame: Mapped[SiteFrame | None] = relationship(
        cascade="all, delete-orphan", uselist=False, passive_deletes=True
    )
    epochs: Mapped[list[Epoch]] = relationship(
        cascade="all, delete-orphan", order_by="Epoch.ordinal", passive_deletes=True
    )
    bookmarks: Mapped[list[CameraBookmark]] = relationship(
        back_populates="site", cascade="all, delete-orphan", order_by="CameraBookmark.created_at"
    )
