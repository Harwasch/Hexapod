"""Persistent land research, separate from capture/reconstruction jobs."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Investigation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_investigations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["land_id", "boundary_revision"],
            ["land_boundary_revisions.land_id", "land_boundary_revisions.revision"],
            ondelete="CASCADE",
        ),
    )
    land_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    boundary_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)


class ResearchMessage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_research_messages"
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("land_investigations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)


class ResearchRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_research_runs"
    __table_args__ = (
        UniqueConstraint("investigation_id", "request_key"),
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'partial', 'failed', 'cancelled')"
        ),
    )
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("land_investigations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    request_key: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    budget: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    analysis: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    focus: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    focus_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    checkpoint: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    lease_token: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def focus_label(self) -> str | None:
        return str(self.focus_snapshot["feature"]["label"]) if self.focus_snapshot else None


class ResearchEvent(Base):
    __tablename__ = "land_research_events"
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("land_research_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Evidence(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_evidence"
    __table_args__ = (UniqueConstraint("run_id", "source_key"),)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("land_research_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_key: Mapped[str] = mapped_column(String(300), nullable=False)
    # Validated EvidenceContent includes URL, license, retrieval/observation dates,
    # excerpts, spatial relevance and the provider's original record identifier.
    content: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class Finding(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_findings"
    __table_args__ = (UniqueConstraint("run_id", "output_key"),)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("land_research_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    output_key: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    disposition: Mapped[str] = mapped_column(String(20), nullable=False, default="visible")


class ResearchArtifact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "land_research_artifacts"
    __table_args__ = (UniqueConstraint("run_id", "output_key"),)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("land_research_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    output_key: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
