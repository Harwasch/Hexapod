"""an epoch revision's per-cell maps (M4)

Revision ID: 0035
Revises: 0034
Create Date: 2026-10-20

`epoch_maps`: one row per asset (an epoch revision) and kind -- today its confidence grid
(ADR 0016) -- holding the grid's descriptor, its summary and the URL of the file beside
the tiles. Rows, not JSON on the epoch: a re-run is a new asset, and the old revision keeps
the map it was measured with, as it keeps its tiles. Deleting an epoch or an asset deletes
its maps; deleting the run that measured one keeps the map (`job_id` set null).

Downgrade drops the table; nothing else refers to it.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0035"
down_revision = "0034"
branch_labels = None
depends_on = None

KINDS = ("confidence",)


def upgrade() -> None:
    op.create_table(
        "epoch_maps",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "epoch_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("epochs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "asset_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("assets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("format", sa.String(40), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("cell_m", sa.Float(), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("descriptor", postgresql.JSONB(), nullable=False),
        sa.Column("summary", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("asset_id", "kind", name="uq_epoch_maps_asset_kind"),
        sa.CheckConstraint(
            f"kind IN ({', '.join(repr(k) for k in KINDS)})", name="ck_epoch_maps_kind"
        ),
        sa.CheckConstraint("cell_m > 0", name="ck_epoch_maps_cell_m"),
    )
    op.create_index("ix_epoch_maps_epoch_id", "epoch_maps", ["epoch_id"])


def downgrade() -> None:
    op.drop_index("ix_epoch_maps_epoch_id", table_name="epoch_maps")
    op.drop_table("epoch_maps")
