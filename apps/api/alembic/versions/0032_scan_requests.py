"""Site scans requested from the landing page: who asked, their notes, and the land.

Revision ID: 0032
Revises: 0031
"""

from __future__ import annotations

import sqlalchemy as sa
from geoalchemy2 import Geometry
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scan_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("email", sa.String(254), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("boundary", Geometry("MULTIPOLYGON", srid=4326), nullable=True),
        sa.Column(
            "parcel_ids", postgresql.ARRAY(sa.String(200)), nullable=False, server_default="{}"
        ),
    )
    # Newest first, counted by the day (the daily cap), and pruned by age once retention is set.
    op.create_index("ix_scan_requests_created_at", "scan_requests", ["created_at"])


def downgrade() -> None:
    op.drop_table("scan_requests")
