"""Immutable land scenario assumptions and reproducible results."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def timestamps():
    return [
        sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
        for name in ("created_at", "updated_at")
    ]


def upgrade():
    op.create_table(
        "land_scenarios",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "land_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_areas.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.String(64), nullable=False),
        *timestamps(),
    )
    op.create_index("ix_land_scenarios_land_id", "land_scenarios", ["land_id"])
    op.create_table(
        "land_scenario_revisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "scenario_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_scenarios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("land_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("boundary_revision", sa.Integer(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("result", postgresql.JSONB(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("scenario_id", "revision"),
        sa.ForeignKeyConstraint(
            ["land_id", "boundary_revision"],
            ["land_boundary_revisions.land_id", "land_boundary_revisions.revision"],
            ondelete="CASCADE",
        ),
    )


def downgrade():
    op.drop_table("land_scenario_revisions")
    op.drop_table("land_scenarios")
