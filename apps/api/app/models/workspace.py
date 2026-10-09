from __future__ import annotations

import uuid

from sqlalchemy import CheckConstraint, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

PILOT_WORKSPACE_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


class Workspace(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "workspaces"

    name: Mapped[str] = mapped_column(String(200), nullable=False)


class Membership(TimestampMixin, Base):
    __tablename__ = "workspace_memberships"
    __table_args__ = (CheckConstraint("role IN ('owner', 'editor', 'viewer')"),)

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    principal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    role: Mapped[str] = mapped_column(String(10), nullable=False)
