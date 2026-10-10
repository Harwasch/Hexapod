"""Workspace ownership for private land records.

Revision ID: 0012
Revises: 0011
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
        for name in ("created_at", "updated_at")
    ]


def upgrade() -> None:
    op.create_table(
        "workspaces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        *_timestamps(),
    )
    op.execute(
        "INSERT INTO workspaces (id, name) VALUES ('00000000-0000-0000-0000-000000000001', 'Pilot workspace')"
    )
    op.create_table(
        "workspace_memberships",
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("principal_id", sa.String(64), primary_key=True),
        sa.Column("role", sa.String(10), nullable=False),
        *_timestamps(),
        sa.CheckConstraint("role IN ('owner', 'editor', 'viewer')"),
    )
    op.add_column(
        "land_areas",
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("'00000000-0000-0000-0000-000000000001'"),
        ),
    )
    op.create_foreign_key(
        "fk_land_workspace",
        "land_areas",
        "workspaces",
        ["workspace_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_land_areas_workspace_id", "land_areas", ["workspace_id"])
    op.alter_column("land_areas", "workspace_id", server_default=None)


def downgrade() -> None:
    op.drop_column("land_areas", "workspace_id")
    op.drop_table("workspace_memberships")
    op.drop_table("workspaces")
