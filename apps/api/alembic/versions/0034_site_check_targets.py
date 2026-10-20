"""a site's surveyed check targets (M3b)

Revision ID: 0034
Revises: 0033
Create Date: 2026-10-20

`site_check_targets`: the ground marks an epoch's alignment is scored against, one row per
survey of a mark. A table rather than JSON on `site_frames`, which is fixed once created
(D3), while targets are re-surveyed and edited one at a time: a changed target retires its
row (`retired_at`) and gets a new one, and at most one row per label is active (a partial
unique index, declared in the model with `postgresql_where` so `alembic check` agrees).

Downgrade drops the table; nothing else refers to it.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0034"
down_revision = "0033"
branch_labels = None
depends_on = None

ROLES = ("check", "control")
HEIGHT_DATUMS = ("ellipsoid-wgs84", "msl-unverified")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def upgrade() -> None:
    op.create_table(
        "site_check_targets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "site_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sites.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("label", sa.String(64), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("height", sa.Float(), nullable=False),
        sa.Column("height_datum", sa.String(20), nullable=False),
        sa.Column("sigma_h_m", sa.Float(), nullable=False),
        sa.Column("sigma_v_m", sa.Float(), nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("surveyed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(_in("role", ROLES), name="ck_site_check_targets_role"),
        sa.CheckConstraint(
            _in("height_datum", HEIGHT_DATUMS), name="ck_site_check_targets_height_datum"
        ),
        sa.CheckConstraint(
            "lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180",
            name="ck_site_check_targets_position",
        ),
        sa.CheckConstraint("sigma_h_m > 0 AND sigma_v_m > 0", name="ck_site_check_targets_sigma"),
    )
    op.create_index("ix_site_check_targets_site_id", "site_check_targets", ["site_id"])
    op.create_index(
        "uq_site_check_targets_active_label",
        "site_check_targets",
        ["site_id", "label"],
        unique=True,
        postgresql_where=sa.text("retired_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_site_check_targets_active_label", table_name="site_check_targets")
    op.drop_index("ix_site_check_targets_site_id", table_name="site_check_targets")
    op.drop_table("site_check_targets")
