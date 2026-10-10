"""Private bounded raster outputs and durable raster analysis requests."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("land_research_runs", sa.Column("analysis", postgresql.JSONB(), nullable=True))
    op.create_table(
        "land_rasters",
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
    op.create_index("ix_land_rasters_land_id", "land_rasters", ["land_id"])
    op.create_index("ix_land_rasters_run_id", "land_rasters", ["run_id"])
    op.create_table(
        "land_raster_blobs",
        sa.Column(
            "raster_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_rasters.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("data", sa.LargeBinary(), nullable=False),
    )


def downgrade():
    op.drop_table("land_raster_blobs")
    op.drop_table("land_rasters")
    op.drop_column("land_research_runs", "analysis")
