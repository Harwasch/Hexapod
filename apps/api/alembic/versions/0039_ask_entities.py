"""Scan-local entities with metric and geographic anchors, plus answer replay.

Revision ID: 0039
Revises: 0038
"""

import sqlalchemy as sa
from geoalchemy2 import Geometry
from sqlalchemy.dialects import postgresql as pg

from alembic import op

revision = "0039"
down_revision = "0038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    def timestamps():
        return [
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        ]

    op.create_table(
        "ask_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column(
            "capture_id", pg.UUID(as_uuid=True), sa.ForeignKey("captures.id", ondelete="CASCADE")
        ),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("events", pg.JSONB(), nullable=False),
        sa.CheckConstraint("state IN ('running','complete','failed')"),
        *timestamps(),
    )
    op.create_index("ix_ask_runs_capture_id", "ask_runs", ["capture_id"])
    op.create_table(
        "scan_entities",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "capture_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("captures.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("site_id", pg.UUID(as_uuid=True), sa.ForeignKey("sites.id", ondelete="SET NULL")),
        sa.Column("ask_id", sa.String(), sa.ForeignKey("ask_runs.id", ondelete="SET NULL")),
        sa.Column(
            "world_anchor",
            Geometry("POINTZ", dimension=3, spatial_index=False),
            nullable=False,
        ),
        sa.Column("geo_anchor", Geometry("POINTZ", srid=4326, dimension=3, spatial_index=False)),
        sa.Column("entity", pg.JSONB(), nullable=False),
        *timestamps(),
    )
    for column in ("capture_id", "site_id", "ask_id"):
        op.create_index(f"ix_scan_entities_{column}", "scan_entities", [column])
    op.create_index(
        "ix_scan_entities_geo_anchor", "scan_entities", ["geo_anchor"], postgresql_using="gist"
    )


def downgrade() -> None:
    op.drop_table("scan_entities")
    op.drop_table("ask_runs")
