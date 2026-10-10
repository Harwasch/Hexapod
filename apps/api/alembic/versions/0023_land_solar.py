"""Immutable hourly solar assessments and private reproducibility archives."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "land_solar_assessments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("land_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("boundary_revision", sa.Integer(), nullable=False),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_research_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("request", postgresql.JSONB(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["land_id", "boundary_revision"],
            ["land_boundary_revisions.land_id", "land_boundary_revisions.revision"],
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_land_solar_assessments_land_id", "land_solar_assessments", ["land_id"])
    op.create_index("ix_land_solar_assessments_run_id", "land_solar_assessments", ["run_id"])
    op.create_table(
        "land_solar_blobs",
        sa.Column(
            "assessment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_solar_assessments.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("data", sa.LargeBinary(), nullable=False),
    )


def downgrade():
    op.drop_table("land_solar_blobs")
    op.drop_table("land_solar_assessments")
