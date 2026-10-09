"""Immutable dated field species surveys and sampling summaries."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "land_surveys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("land_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("boundary_revision", sa.Integer(), nullable=False),
        sa.Column("content", postgresql.JSONB(), nullable=False),
        sa.Column("summary", postgresql.JSONB(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("recorded_by", sa.String(64), nullable=False),
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
    op.create_index("ix_land_surveys_land_id", "land_surveys", ["land_id"])


def downgrade():
    op.drop_table("land_surveys")
