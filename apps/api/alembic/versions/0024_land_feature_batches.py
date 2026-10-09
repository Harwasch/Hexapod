"""Atomic inventory import receipts."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "land_feature_batches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "land_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_areas.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("request_key", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("boundary_revision", sa.Integer(), nullable=False),
        sa.Column("source_file_sha256", sa.String(64), nullable=False),
        sa.Column("source_label", sa.String(200), nullable=False),
        sa.Column("results", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("land_id", "request_key"),
    )
    op.create_index("ix_land_feature_batches_land_id", "land_feature_batches", ["land_id"])


def downgrade():
    op.drop_table("land_feature_batches")
