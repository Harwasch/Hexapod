"""Workspace display profiles and single-use invitations.

Revision ID: 0026
Revises: 0025
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
        for name in ("created_at", "updated_at")
    ]


def upgrade() -> None:
    op.create_table(
        "workspace_profiles",
        sa.Column("principal_id", sa.String(64), primary_key=True),
        sa.Column("display_name", sa.String(120), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "workspace_invitations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("label", sa.String(120), nullable=False),
        sa.Column("role", sa.String(10), nullable=False),
        sa.Column("created_by", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column("accepted_at", sa.DateTime(timezone=True)),
        sa.Column("accepted_by", sa.String(64)),
        *_timestamps(),
        sa.CheckConstraint("role IN ('editor', 'viewer')"),
    )
    op.create_index(
        "ix_workspace_invitations_workspace_id", "workspace_invitations", ["workspace_id"]
    )


def downgrade() -> None:
    op.drop_table("workspace_invitations")
    op.drop_table("workspace_profiles")
