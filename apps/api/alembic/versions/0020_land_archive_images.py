"""Immutable private archive-image snapshots and normalized display copies."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "land_archive_images",
        sa.Column(
            "evidence_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_evidence.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_table(
        "land_archive_image_blobs",
        sa.Column(
            "evidence_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_archive_images.evidence_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("original", sa.LargeBinary(), nullable=False),
        sa.Column("preview", sa.LargeBinary(), nullable=False),
    )


def downgrade():
    op.drop_table("land_archive_image_blobs")
    op.drop_table("land_archive_images")
