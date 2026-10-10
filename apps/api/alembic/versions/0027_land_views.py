"""Private land exploration views.

Revision ID: 0027
Revises: 0026
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "land_views",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "land_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_areas.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("request_key", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("state", postgresql.JSONB(), nullable=False),
        *[
            sa.Column(
                name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
            )
            for name in ("created_at", "updated_at")
        ],
        sa.UniqueConstraint("land_id", "request_key"),
    )
    op.create_index("ix_land_views_land_id", "land_views", ["land_id"])


def downgrade() -> None:
    op.drop_table("land_views")
