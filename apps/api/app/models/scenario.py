from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import (
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class LandScenario(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_scenarios"
    land_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_areas.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(30))
    revision: Mapped[int] = mapped_column(Integer, default=1)
    created_by: Mapped[str] = mapped_column(String(64))


class LandScenarioRevision(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_scenario_revisions"
    __table_args__ = (
        Index("ix_land_scenario_revisions_request", "land_id", text("(payload ->> 'request_key')")),
        UniqueConstraint("scenario_id", "revision"),
        ForeignKeyConstraint(
            ["land_id", "boundary_revision"],
            ["land_boundary_revisions.land_id", "land_boundary_revisions.revision"],
            ondelete="CASCADE",
        ),
    )
    scenario_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("land_scenarios.id", ondelete="CASCADE")
    )
    land_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    boundary_revision: Mapped[int] = mapped_column(Integer)
    revision: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB)
