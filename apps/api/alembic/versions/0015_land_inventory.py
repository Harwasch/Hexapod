"""Independent land inventory and dated inspection history."""

import sqlalchemy as sa
from geoalchemy2 import Geometry
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def timestamps():
    return [
        sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
        for name in ("created_at", "updated_at")
    ]


def upgrade():
    op.create_table(
        "land_features",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "land_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_areas.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("geometry", Geometry("GEOMETRY", srid=4326, spatial_index=False), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("content", postgresql.JSONB(), nullable=False),
        *timestamps(),
    )
    op.create_index("ix_land_features_land_id", "land_features", ["land_id"])
    op.create_index(
        "idx_land_features_geometry", "land_features", ["geometry"], postgresql_using="gist"
    )
    op.create_table(
        "land_feature_revisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "feature_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_features.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("content", postgresql.JSONB(), nullable=False),
        sa.Column("note", sa.String(1000), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("feature_id", "revision"),
    )
    op.create_table(
        "land_feature_inspections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("feature_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("feature_revision", sa.Integer(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_by", sa.String(64), nullable=False),
        sa.Column("content", postgresql.JSONB(), nullable=False),
        *timestamps(),
        sa.ForeignKeyConstraint(
            ["feature_id", "feature_revision"],
            ["land_feature_revisions.feature_id", "land_feature_revisions.revision"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_land_feature_inspections_feature_id", "land_feature_inspections", ["feature_id"]
    )


def downgrade():
    op.drop_table("land_feature_inspections")
    op.drop_table("land_feature_revisions")
    op.drop_table("land_features")
