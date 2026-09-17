from __future__ import annotations

import uuid
from datetime import date
from typing import TYPE_CHECKING, Any

from sqlalchemy import Date, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    pass


class Plan(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A mission for the fleet: an approved draft with its lifecycle state.

    Keyed by the mission project (the demo project or a fleet backend's id); optionally
    linked to a catalog site. The planner's output (steps, estimates, assumptions, risks,
    questions) is stored as JSON: it changes with the planner, not with migrations.
    """

    __tablename__ = "plans"

    project_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    site_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sites.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="scheduled")
    cadence: Mapped[str] = mapped_column(String(20), nullable=False, default="once")
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    zone_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    machine_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    steps: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    estimates: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    assumptions: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    risks: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    questions: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="rules")
    model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    revisions: Mapped[list[PlanRevision]] = relationship(
        back_populates="plan", cascade="all, delete-orphan", order_by="PlanRevision.revision"
    )


class PlanRevision(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A snapshot of a plan as approved; one row per approval (create or revise)."""

    __tablename__ = "plan_revisions"
    __table_args__ = (UniqueConstraint("plan_id", "revision", name="uq_plan_revision"),)

    plan_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("plans.id", ondelete="CASCADE"), nullable=False, index=True
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    note: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    plan: Mapped[Plan] = relationship(back_populates="revisions")
