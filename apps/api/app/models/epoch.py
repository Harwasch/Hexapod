"""A site's fixed frame, and the epochs (dated captures) expressed in it -- decision D3.

A site is captured more than once. Its first registered capture defines one east/north/up
frame (`SiteFrame`, fixed once created) and every later capture of it is an `Epoch`
placed in that frame, so nothing a previous epoch published is ever rewritten. An asset
belongs to an epoch (`Asset.epoch_id`); a re-run of the same capture inserts a new asset
revision of that epoch and marks the old one superseded (`valid_to` set). The epoch's
current asset is its newest observed one with no `valid_to` (`services/epochs.py`).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, utcnow
from app.models.enums import (
    EpochJoinMethod,
    EpochPlatform,
    EpochStatus,
    HeightDatum,
    ObservedAtSource,
    SiteFrameSource,
)


def _in(column: str, values: type[Any]) -> str:
    return f"{column} IN ({', '.join(repr(v.value) for v in values)})"


class SiteFrame(Base):
    """A site's origin: the WGS84 point every epoch's tiles are placed about. 1:1 with the
    site, fixed once created (no `updated_at`)."""

    __tablename__ = "site_frames"
    __table_args__ = (
        CheckConstraint(
            "lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180", name="ck_site_frames_origin"
        ),
        CheckConstraint(_in("height_datum", HeightDatum), name="ck_site_frames_height_datum"),
        CheckConstraint(
            "(epsg BETWEEN 32601 AND 32660) OR (epsg BETWEEN 32701 AND 32760)",
            name="ck_site_frames_epsg",
        ),
        CheckConstraint(_in("source", SiteFrameSource), name="ck_site_frames_source"),
    )

    site_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sites.id", ondelete="CASCADE"), primary_key=True
    )
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    height_datum: Mapped[str] = mapped_column(String(20), nullable=False)
    # The UTM zone the origin is in, for anything that wants a projected CRS of the site.
    epsg: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )


class Epoch(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One dated capture of a site, expressed in the site's frame."""

    __tablename__ = "epochs"
    __table_args__ = (
        UniqueConstraint("capture_id", name="uq_epochs_capture_id"),
        UniqueConstraint("site_id", "ordinal", name="uq_epochs_site_ordinal"),
        CheckConstraint("ordinal >= 1", name="ck_epochs_ordinal"),
        CheckConstraint(_in("status", EpochStatus), name="ck_epochs_status"),
        CheckConstraint(_in("join_method", EpochJoinMethod), name="ck_epochs_join_method"),
        CheckConstraint(_in("platform", EpochPlatform), name="ck_epochs_platform"),
        CheckConstraint(
            _in("observed_at_source", ObservedAtSource), name="ck_epochs_observed_at_source"
        ),
    )

    site_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sites.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # One capture per epoch for now (D3); `captures.epoch_id` -- several captures of one
    # day as one epoch -- is future work.
    capture_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("captures.id", ondelete="SET NULL"), nullable=True
    )
    # The run that produced the epoch's current revision.
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    observed_at_source: Mapped[str] = mapped_column(String(20), nullable=False)
    footprint: Mapped[WKBElement] = mapped_column(
        Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=True), nullable=False
    )
    platform: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    join_method: Mapped[str] = mapped_column(String(20), nullable=False)
    # How this epoch was aligned to the ones before it (M3 fills it).
    alignment: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    # A `CaptureQuality` dump, the same shape as `captures.quality`.
    quality: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
