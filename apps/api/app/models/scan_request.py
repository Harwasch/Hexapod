from __future__ import annotations

from typing import Any

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import Boolean, Index, String, Text, false
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ScanRequest(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A site scan requested from the landing page's form, by a visitor who has not signed in.

    Read only with the write token (`GET /api/v1/scan-requests`, the admin console). The
    boundary is the land the visitor found (parcels dissolved into one outline, or drawn); none
    when they gave only its acres, which then open the notes.

    Kept indefinitely, with no automatic deletion (Harrison's decision, 2026-10-20): a request is
    deleted only when the person who sent it asks.
    """

    __tablename__ = "scan_requests"
    # Read newest first, and counted by the day for the daily cap. Requests are kept
    # indefinitely; the index also keeps a later prune by age cheap, should that change.
    __table_args__ = (Index("ix_scan_requests_created_at", "created_at"),)

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(254), nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    boundary: Mapped[WKBElement | None] = mapped_column(
        Geometry("MULTIPOLYGON", srid=4326, spatial_index=True), nullable=True
    )
    parcel_ids: Mapped[list[str]] = mapped_column(
        ARRAY(String(200)), nullable=False, default=list, server_default="{}"
    )
    #: The outline could not be cleaned in time (app/services/scan_requests.py): it is kept as
    #: sent, in `raw_outline`, and `boundary` is empty, rather than the request being lost.
    outline_unvalidated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    raw_outline: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
