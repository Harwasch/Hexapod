"""Research worker presence: tell an idle worker from a missing one.

Run leases (land_research_runs.lease_expires_at) only exist while a run is held, so a queued
run could not say whether anything would ever pick it up. Each worker process now checks in
here every 15 s, idle or busy, and marks itself stopped on a clean exit.

Revision ID: 0030
Revises: 0029
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "land_research_workers",
        # The worker picks its own id at start-up and upserts it, so there is no default.
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("hostname", sa.String(255), nullable=False),
        sa.Column("pid", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        # No foreign key: a presence write must never wait on a run's row lock.
        sa.Column("current_run_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        "ix_land_research_workers_last_seen_at", "land_research_workers", ["last_seen_at"]
    )


def downgrade() -> None:
    op.drop_table("land_research_workers")
