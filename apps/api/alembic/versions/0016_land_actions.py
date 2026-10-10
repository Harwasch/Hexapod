"""Versioned land actions and workspace-scoped mission handoffs."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def timestamps():
    return [
        sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
        for name in ("created_at", "updated_at")
    ]


def upgrade():
    op.add_column("plans", sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_plans_workspace", "plans", "workspaces", ["workspace_id"], ["id"], ondelete="CASCADE"
    )
    op.create_index("ix_plans_workspace_id", "plans", ["workspace_id"])
    op.create_table(
        "land_actions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "land_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_areas.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.String(64), nullable=False),
        *timestamps(),
    )
    op.create_index("ix_land_actions_land_id", "land_actions", ["land_id"])
    op.create_table(
        "land_action_revisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "action_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_actions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("land_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("boundary_revision", sa.Integer(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("effective_boundary", postgresql.JSONB(), nullable=False),
        sa.Column("note", sa.String(1000), nullable=False),
        sa.Column("approved_by", sa.String(64), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approval_note", sa.Text(), nullable=True),
        sa.Column(
            "mission_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("plans.id", ondelete="SET NULL"),
            nullable=True,
            unique=True,
        ),
        sa.UniqueConstraint("action_id", "revision"),
        sa.ForeignKeyConstraint(
            ["land_id", "boundary_revision"],
            ["land_boundary_revisions.land_id", "land_boundary_revisions.revision"],
            ondelete="CASCADE",
        ),
        *timestamps(),
    )


def downgrade():
    # Do not silently make private plans public when removing workspace ownership.
    connection = op.get_bind()
    if connection.scalar(
        sa.text("SELECT EXISTS (SELECT 1 FROM plans WHERE workspace_id IS NOT NULL)")
    ):
        raise RuntimeError(
            "Retain or explicitly export private missions before downgrading their workspace ownership."
        )
    op.drop_table("land_action_revisions")
    op.drop_table("land_actions")
    op.drop_index("ix_plans_workspace_id", "plans")
    op.drop_constraint("fk_plans_workspace", "plans", type_="foreignkey")
    op.drop_column("plans", "workspace_id")
