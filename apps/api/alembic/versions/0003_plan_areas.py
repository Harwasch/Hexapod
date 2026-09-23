"""areas drawn in the console, stored on the plan

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-17
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "plans",
        sa.Column(
            "areas", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
    )
    op.alter_column("plans", "areas", server_default=None)


def downgrade() -> None:
    op.drop_column("plans", "areas")
