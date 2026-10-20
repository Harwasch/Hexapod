"""Site frames and epochs: a second capture of a site is a new epoch of it (decision D3).

`site_frames` is 1:1 with sites: the WGS84 origin every epoch of the site is placed about,
fixed once created. `epochs` is one dated capture of a site in that frame. `assets` gains
`epoch_id` (which epoch an asset is a revision of; NULL for seeded, catalog and hand-made
assets) and `source_kind` (`observed` | `inferred`).

Backfill. Every site the pipeline registered -- `metadata.captureId` naming a capture of
that site, and a numeric `metadata.registration.georef` lat/lon in range -- gets a frame
from that georef (`height_datum` `msl-unverified`: GPS altitude is above the geoid and no
geoid model is applied; `source` `backfill`) and one epoch: ordinal 1, `new-site`,
`published`, `observed_at` the capture's `captured_at` or else its upload time (flagged
`observed_at_source = 'created'`), the platform `phone-video` for a phone-key capture and
`unknown` otherwise, the capture's quality, the site boundary as its footprint, and the
run named by `metadata.jobId` (else the capture's newest complete run). The site's first
splat by creation -- the one the worker repointed until now -- becomes that epoch's asset
and takes its `observed_at` where it had none. Seeded and catalog sites have no
`captureId` and stay legacy.

Downgrade drops the tables and columns and nothing else: no asset row is deleted and no
`source` is touched. A site with several revisions of one epoch is then a site with
several splat assets; the newest stays `default_visible`, and a 0031 worker would repoint
the oldest. The backfilled `observed_at` stays, since it is a date 0031 already models.
"""

import geoalchemy2
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


HEIGHT_DATUMS = ("ellipsoid-wgs84", "msl-unverified")
FRAME_SOURCES = ("registration", "backfill", "echo")
STATUSES = ("published", "rejected", "superseded")
JOIN_METHODS = ("explicit", "spatial", "new-site")
PLATFORMS = ("rtk-drone", "drone", "phone-video", "phone-photo", "unknown")
OBSERVED_AT_SOURCES = ("captured-at", "manifest", "created")


def upgrade() -> None:
    op.create_table(
        "site_frames",
        sa.Column(
            "site_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sites.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("height", sa.Float(), nullable=False),
        sa.Column("height_datum", sa.String(20), nullable=False),
        sa.Column("epsg", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180", name="ck_site_frames_origin"
        ),
        sa.CheckConstraint(_in("height_datum", HEIGHT_DATUMS), name="ck_site_frames_height_datum"),
        sa.CheckConstraint(
            "(epsg BETWEEN 32601 AND 32660) OR (epsg BETWEEN 32701 AND 32760)",
            name="ck_site_frames_epsg",
        ),
        sa.CheckConstraint(_in("source", FRAME_SOURCES), name="ck_site_frames_source"),
    )
    op.create_table(
        "epochs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "site_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sites.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "capture_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("captures.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("observed_at_source", sa.String(20), nullable=False),
        sa.Column(
            "footprint",
            geoalchemy2.Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=False),
            nullable=False,
        ),
        sa.Column("platform", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("join_method", sa.String(20), nullable=False),
        sa.Column("alignment", postgresql.JSONB(), nullable=True),
        sa.Column("quality", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("capture_id", name="uq_epochs_capture_id"),
        sa.UniqueConstraint("site_id", "ordinal", name="uq_epochs_site_ordinal"),
        sa.CheckConstraint("ordinal >= 1", name="ck_epochs_ordinal"),
        sa.CheckConstraint(_in("status", STATUSES), name="ck_epochs_status"),
        sa.CheckConstraint(_in("join_method", JOIN_METHODS), name="ck_epochs_join_method"),
        sa.CheckConstraint(_in("platform", PLATFORMS), name="ck_epochs_platform"),
        sa.CheckConstraint(
            _in("observed_at_source", OBSERVED_AT_SOURCES), name="ck_epochs_observed_at_source"
        ),
    )
    op.create_index("ix_epochs_site_id", "epochs", ["site_id"])
    op.create_index("idx_epochs_footprint", "epochs", ["footprint"], postgresql_using="gist")

    op.add_column(
        "assets",
        sa.Column(
            "epoch_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("epochs.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "assets",
        sa.Column("source_kind", sa.String(10), nullable=False, server_default="observed"),
    )
    op.create_check_constraint(
        "ck_assets_source_kind", "assets", "source_kind IN ('observed', 'inferred')"
    )
    op.create_index("ix_assets_epoch_id", "assets", ["epoch_id"])

    # The backfill: one frame and one epoch for every site the pipeline registered.
    # `c.id::text = metadata->>'captureId'` rather than a uuid cast, so a malformed id
    # makes a site ineligible instead of failing the migration.
    op.execute(
        """
        CREATE TEMPORARY TABLE _m0032_registered ON COMMIT DROP AS
        SELECT
            s.id AS site_id,
            c.id AS capture_id,
            (s.metadata #>> '{registration,georef,lat}')::float8 AS lat,
            (s.metadata #>> '{registration,georef,lon}')::float8 AS lon,
            CASE
                WHEN jsonb_typeof(s.metadata #> '{registration,georef,height}') = 'number'
                THEN (s.metadata #>> '{registration,georef,height}')::float8
                ELSE 0
            END AS height,
            gen_random_uuid() AS epoch_id
        FROM sites s
        JOIN captures c ON c.site_id = s.id AND c.id::text = s.metadata ->> 'captureId'
        WHERE s.metadata ? 'captureId'
          AND jsonb_typeof(s.metadata #> '{registration,georef,lat}') = 'number'
          AND jsonb_typeof(s.metadata #> '{registration,georef,lon}') = 'number'
          AND (s.metadata #>> '{registration,georef,lat}')::float8 BETWEEN -90 AND 90
          AND (s.metadata #>> '{registration,georef,lon}')::float8 BETWEEN -180 AND 180
        """
    )
    op.execute(
        """
        INSERT INTO site_frames (site_id, lat, lon, height, height_datum, epsg, source, created_at)
        SELECT
            r.site_id, r.lat, r.lon, r.height, 'msl-unverified',
            CASE WHEN r.lat >= 0 THEN 32600 ELSE 32700 END
                + LEAST(60, floor((r.lon + 180) / 6)::int + 1),
            'backfill', now()
        FROM _m0032_registered r
        """
    )
    op.execute(
        """
        INSERT INTO epochs (
            id, site_id, capture_id, job_id, ordinal, observed_at, observed_at_source,
            footprint, platform, status, join_method, alignment, quality, created_at, updated_at
        )
        SELECT
            r.epoch_id,
            r.site_id,
            c.id,
            COALESCE(
                (SELECT j.id FROM jobs j
                 WHERE j.capture_id = c.id AND j.id::text = s.metadata ->> 'jobId'),
                (SELECT j.id FROM jobs j
                 WHERE j.capture_id = c.id AND j.status = 'complete'
                 ORDER BY j.finished_at DESC NULLS LAST, j.created_at DESC
                 LIMIT 1)
            ),
            1,
            COALESCE(c.captured_at, c.created_at),
            CASE WHEN c.captured_at IS NULL THEN 'created' ELSE 'captured-at' END,
            s.boundary,
            CASE WHEN c.metadata ->> 'origin' = 'phone-key' THEN 'phone-video' ELSE 'unknown' END,
            'published',
            'new-site',
            NULL,
            c.quality,
            now(),
            now()
        FROM _m0032_registered r
        JOIN sites s ON s.id = r.site_id
        JOIN captures c ON c.id = r.capture_id
        """
    )
    # The site's first splat by creation -- the one the worker repointed until now, the
    # rule of `registration.splat_asset` -- is its epoch's asset. Unless it has a
    # `valid_to`: before 0032 that was a validity window anyone could PATCH, and on an
    # epoch asset it means "superseded", which would hide the splat. Such a splat stays
    # legacy with its window, and the epoch's next run gives it its first revision.
    op.execute(
        """
        UPDATE assets a
        SET epoch_id = e.id, observed_at = COALESCE(a.observed_at, e.observed_at)
        FROM (
            SELECT DISTINCT ON (x.site_id) x.id, x.site_id
            FROM assets x
            JOIN _m0032_registered r ON r.site_id = x.site_id
            WHERE x.representation = 'gaussian-splat'
            ORDER BY x.site_id, x.created_at, x.id
        ) first_splat
        JOIN epochs e ON e.site_id = first_splat.site_id AND e.ordinal = 1
        WHERE a.id = first_splat.id AND a.valid_to IS NULL
        """
    )
    op.execute("DROP TABLE IF EXISTS _m0032_registered")


def downgrade() -> None:
    # Tables and columns only: every asset row, and every asset's `source`, survives.
    op.drop_index("ix_assets_epoch_id", table_name="assets")
    op.drop_constraint("ck_assets_source_kind", "assets", type_="check")
    op.drop_column("assets", "source_kind")
    op.drop_column("assets", "epoch_id")
    op.drop_index("idx_epochs_footprint", table_name="epochs")
    op.drop_index("ix_epochs_site_id", table_name="epochs")
    op.drop_table("epochs")
    op.drop_table("site_frames")
