"""initial schema: sites, assets, layers, camera bookmarks

Revision ID: 0001
Revises:
Create Date: 2026-09-15
"""

from __future__ import annotations

import geoalchemy2
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

REPRESENTATION = ("gaussian-splat", "mesh", "point-cloud", "terrain", "imagery")
ASSET_PROVIDER = ("cesium-ion", "3d-tiles-url")
LAYER_CATEGORY = (
    "reality",
    "terrain",
    "imagery",
    "hydrology",
    "land-cover",
    "ecology",
    "infrastructure",
    "my-data",
)
LAYER_SOURCE_TYPE = (
    "cesium-ion-terrain",
    "cesium-ion-imagery",
    "cesium-ion-3d-tiles",
    "google-photorealistic",
    "3d-tiles-url",
    "xyz",
    "wmts",
    "wms",
    "arcgis-mapserver",
    "geojson",
    "czml",
    "mvt",
    "stac",
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    representation = postgresql.ENUM(*REPRESENTATION, name="representation", create_type=False)
    asset_provider = postgresql.ENUM(*ASSET_PROVIDER, name="asset_provider", create_type=False)
    layer_category = postgresql.ENUM(*LAYER_CATEGORY, name="layer_category", create_type=False)
    layer_source_type = postgresql.ENUM(
        *LAYER_SOURCE_TYPE, name="layer_source_type", create_type=False
    )
    bind = op.get_bind()
    for enum in (representation, asset_provider, layer_category, layer_source_type):
        enum.create(bind, checkfirst=True)

    op.create_table(
        "sites",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("slug", sa.String(120), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "boundary",
            geoalchemy2.Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=False),
            nullable=False,
        ),
        sa.Column(
            "centroid",
            geoalchemy2.Geometry(geometry_type="POINT", srid=4326, spatial_index=False),
            nullable=False,
        ),
        sa.Column("centroid_height", sa.Float(), nullable=True),
        sa.Column("thumbnail_url", sa.String(2048), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("attribution", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("license", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_sites_slug", "sites", ["slug"], unique=True)
    op.create_index("idx_sites_boundary", "sites", ["boundary"], postgresql_using="gist")

    op.create_table(
        "assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "site_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sites.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("representation", representation, nullable=False),
        sa.Column("provider", asset_provider, nullable=False),
        sa.Column("source", postgresql.JSONB(), nullable=False),
        sa.Column(
            "footprint",
            geoalchemy2.Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=False),
            nullable=True,
        ),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution", postgresql.JSONB(), nullable=True),
        sa.Column("crs", postgresql.JSONB(), nullable=True),
        sa.Column("license", postgresql.JSONB(), nullable=True),
        sa.Column("attribution", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("render_config", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("default_visible", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_assets_site_id", "assets", ["site_id"])
    op.create_index("idx_assets_footprint", "assets", ["footprint"], postgresql_using="gist")

    op.create_table(
        "layers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("slug", sa.String(120), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", layer_category, nullable=False),
        sa.Column("source_type", layer_source_type, nullable=False),
        sa.Column("source", postgresql.JSONB(), nullable=False),
        sa.Column(
            "spatial_extent",
            geoalchemy2.Geometry(geometry_type="POLYGON", srid=4326, spatial_index=False),
            nullable=True,
        ),
        sa.Column("temporal_extent", postgresql.JSONB(), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution", sa.String(120), nullable=True),
        sa.Column("coverage", sa.String(120), nullable=True),
        sa.Column("render", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("legend", postgresql.JSONB(), nullable=True),
        sa.Column("attribution", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("license", postgresql.JSONB(), nullable=True),
        sa.Column("default_visible", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("builtin", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_layers_slug", "layers", ["slug"], unique=True)
    op.create_index("ix_layers_category", "layers", ["category"])
    op.create_index(
        "idx_layers_spatial_extent", "layers", ["spatial_extent"], postgresql_using="gist"
    )

    op.create_table(
        "camera_bookmarks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "site_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sites.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("height", sa.Float(), nullable=False),
        sa.Column("heading", sa.Float(), nullable=False, server_default="0"),
        sa.Column("pitch", sa.Float(), nullable=False, server_default="-30"),
        sa.Column("roll", sa.Float(), nullable=False, server_default="0"),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_camera_bookmarks_site_id", "camera_bookmarks", ["site_id"])


def downgrade() -> None:
    op.drop_table("camera_bookmarks")
    op.drop_index("idx_layers_spatial_extent", table_name="layers")
    op.drop_table("layers")
    op.drop_index("idx_assets_footprint", table_name="assets")
    op.drop_table("assets")
    op.drop_index("idx_sites_boundary", table_name="sites")
    op.drop_table("sites")
    for name in ("layer_source_type", "layer_category", "asset_provider", "representation"):
        op.execute(f"DROP TYPE IF EXISTS {name}")
