from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import CaptureKind, CaptureStatus, GeorefMethod, ScaleSource, UploadStatus

if TYPE_CHECKING:
    from app.models.job import Job
    from app.models.site import Site


class Capture(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One upload session: a source dataset as it arrived, before anything ran on it.

    A capture is not a site. It becomes one only once a pipeline run has processed and
    registered it, which is why ``site_id`` is nullable and is cleared rather than
    cascaded when a site is deleted.
    """

    __tablename__ = "captures"

    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[CaptureStatus] = mapped_column(
        Enum(CaptureStatus, name="capture_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=CaptureStatus.AWAITING_FILES,
    )
    kind: Mapped[CaptureKind] = mapped_column(
        Enum(CaptureKind, name="capture_kind", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )

    # Source: what recorded it, and when. Free-form sensor detail (lens, ARKit
    # confidence, drone flight log...) belongs in `metadata`, as it does on Site.
    device: Mapped[str | None] = mapped_column(String(200), nullable=True)
    sensor: Mapped[str | None] = mapped_column(String(120), nullable=True)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # app.schemas.common.TemporalExtent — a span, because a video or a 4D capture has one.
    temporal_extent: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Georeference, per the plan's B4: how it was placed, where metric scale came from,
    # and how wrong that could be. Null until a georeference stage has run.
    georef_method: Mapped[GeorefMethod | None] = mapped_column(
        Enum(GeorefMethod, name="georef_method", values_callable=lambda e: [m.value for m in e]),
        nullable=True,
    )
    scale_source: Mapped[ScaleSource | None] = mapped_column(
        Enum(ScaleSource, name="scale_source", values_callable=lambda e: [m.value for m in e]),
        nullable=True,
    )
    uncertainty_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    # app.schemas.common.Attribution / LicenseMetadata / Provenance — the same shapes
    # Site and Asset already store, so a capture's credit survives into the site it becomes.
    attribution: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    license: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    provenance: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    site_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sites.id", ondelete="SET NULL"), nullable=True, index=True
    )

    site: Mapped[Site | None] = relationship()
    files: Mapped[list[CaptureFile]] = relationship(
        back_populates="capture", cascade="all, delete-orphan", order_by="CaptureFile.created_at"
    )
    jobs: Mapped[list[Job]] = relationship(
        back_populates="capture", cascade="all, delete-orphan", order_by="Job.created_at"
    )


class CaptureFile(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One uploaded source object, with enough S3 multipart state to resume it.

    Parts are counted, not rowed: a 12 GB video at 8 MiB parts is 1536 parts (A0's
    sizing), and the counts are all a resume needs — A3 presigns in windows and the
    client replays a window rather than consulting per-part rows that would exist only
    to be counted.
    """

    __tablename__ = "capture_files"

    capture_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("captures.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(200), nullable=True)
    bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Unique: one object in the bucket is one row, which is what makes the console's
    # storage-against-database reconciliation a lookup rather than a guess.
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True, index=True)
    status: Mapped[UploadStatus] = mapped_column(
        Enum(UploadStatus, name="upload_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=UploadStatus.NOT_STARTED,
    )
    # The provider's opaque multipart handle, and 1024 rather than 255 because a real
    # provider is what settled the size. MinIO and S3 hand back something short, so every
    # test in this repository passed at 255; R2's are 343 characters, and the first
    # presigned upload against it failed on `value too long for type character
    # varying(255)` -- after the API had already created the row's capture, so the
    # failure arrived as a 500 halfway through a two-call sequence.
    #
    # Sized to match `storage_key` above. It is opaque and provider-defined, so there is
    # no length to be right about, only a length to be beyond: 1024 is three times the
    # longest anyone here has seen and costs nothing in Postgres, where varchar(n) and
    # text are the same storage.
    upload_id: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    parts_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parts_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    capture: Mapped[Capture] = relationship(back_populates="files")
