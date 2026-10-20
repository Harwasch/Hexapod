"""A site's surveyed ground marks (M3b): what an epoch's alignment is scored against.

Rows, not JSON on `site_frames`: a site's frame is fixed once created (D3), and targets
are re-surveyed and edited one at a time. A changed target is never updated in place: the
old row is retired (`retired_at`) and a new one inserted, so an epoch scored against the
old survey can still be read against what it was scored with. At most one active row per
label (`uq_site_check_targets_active_label`, partial on `retired_at IS NULL`).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Index, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import CheckTargetRole, HeightDatum

#: Spelled once: the model declares it and migration 0034 builds it.
ACTIVE_LABEL_INDEX = "uq_site_check_targets_active_label"


def _in(column: str, values: type[CheckTargetRole] | type[HeightDatum]) -> str:
    return f"{column} IN ({', '.join(repr(v.value) for v in values)})"


class SiteCheckTarget(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One surveyed mark: where it is (WGS84, with its height datum) and how well."""

    __tablename__ = "site_check_targets"
    __table_args__ = (
        CheckConstraint(_in("role", CheckTargetRole), name="ck_site_check_targets_role"),
        CheckConstraint(
            _in("height_datum", HeightDatum), name="ck_site_check_targets_height_datum"
        ),
        CheckConstraint(
            "lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180",
            name="ck_site_check_targets_position",
        ),
        CheckConstraint("sigma_h_m > 0 AND sigma_v_m > 0", name="ck_site_check_targets_sigma"),
        Index(
            ACTIVE_LABEL_INDEX,
            "site_id",
            "label",
            unique=True,
            postgresql_where=text("retired_at IS NULL"),
        ),
    )

    site_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sites.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    height_datum: Mapped[str] = mapped_column(String(20), nullable=False)
    #: The survey's own one-sigma, horizontal and vertical, in metres.
    sigma_h_m: Mapped[float] = mapped_column(Float, nullable=False)
    sigma_v_m: Mapped[float] = mapped_column(Float, nullable=False)
    #: How it was surveyed: `rtk-rover`, `total-station`, `synthetic`, ...
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    surveyed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: Set when a later survey replaced it or it was removed; never deleted.
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
