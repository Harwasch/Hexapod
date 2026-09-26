"""captures.quality: the quality bar's verdict on the capture's latest finished run.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-26

The photo-reconstruct recipe's `quality` stage decides, per gaussian, whether the capture
supports it, and forecasts how much of the region the cameras were pointed at reached
high quality. The phone shows that forecast after a cheap preview and offers Refine,
which trains again inside the same region -- so the region, in the reconstruction's own
frame, has to be somewhere the API can read it back. A column rather than a key in the
free-form `metadata`, because `metadata` is what the person who made the capture said
about it and this is what a run measured; and nullable, because a capture that has not
finished a run with a quality stage has no verdict.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("captures", sa.Column("quality", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("captures", "quality")
