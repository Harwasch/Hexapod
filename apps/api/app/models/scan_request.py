from __future__ import annotations

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ScanRequest(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A site scan requested from the landing page's form, by a visitor who has not signed in.

    Read only with the write token (`GET /api/v1/scan-requests`, the admin console). The
    boundary is the land the visitor found (parcels dissolved into one outline, or drawn); none
    when they gave only its acres, which then open the notes.
    """

    __tablename__ = "scan_requests"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(254), nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    boundary: Mapped[WKBElement | None] = mapped_column(
        Geometry("MULTIPOLYGON", srid=4326, spatial_index=True), nullable=True
    )
    parcel_ids: Mapped[list[str]] = mapped_column(
        ARRAY(String(200)), nullable=False, default=list, server_default="{}"
    )
