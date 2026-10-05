"""the camera-height-estimate scale source

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-05

A phone video carries no per-frame GPS, so its reconstruction has no measured scale and
was registered `unresolved` -- and drawn at one model unit to the metre, two to eight
times too big. The pipeline now estimates the scale from how high a handheld phone is
(about 1.5 m above the ground the reconstruction shows) and says so: `scaleSource:
camera-height-estimate`, never one of the measured sources. This is the value the
`captures.scale_source` column needs to hold it. The asset's provenance (a JSON column)
carries it, and the estimate's ±%, without a migration.
"""

from __future__ import annotations

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None

# 0004's members, in 0004's order. `camera-height-estimate` goes in before `unresolved`,
# where `ScaleSource` declares it.
SCALE_SOURCE = ("arkit", "exif-gps", "manual", "unresolved")


def upgrade() -> None:
    # As 0005: PostgreSQL 12+ allows ADD VALUE inside a transaction as long as the new
    # value is not used in the same one, and nothing here inserts.
    op.execute(
        "ALTER TYPE scale_source ADD VALUE IF NOT EXISTS 'camera-height-estimate' "
        "BEFORE 'unresolved'"
    )


def downgrade() -> None:
    # An enum value cannot be dropped, so the type is rebuilt, as 0005's downgrade does.
    # An estimate folds into `unresolved` -- what it was registered as before this -- and
    # so does the copy of it in each asset's provenance, which the older schema would
    # refuse to read (its `scaleUncertaintyPct` too: that schema forbids extra keys).
    members = ", ".join(f"'{value}'" for value in SCALE_SOURCE)
    op.execute(
        "UPDATE captures SET scale_source = 'unresolved' "
        "WHERE scale_source = 'camera-height-estimate'"
    )
    op.execute(
        "UPDATE assets SET render_config = jsonb_set("
        "render_config #- '{provenance,scaleUncertaintyPct}', "
        "'{provenance,scaleSource}', '\"unresolved\"') "
        "WHERE render_config -> 'provenance' ->> 'scaleSource' = 'camera-height-estimate'"
    )
    op.execute("ALTER TYPE scale_source RENAME TO scale_source_old")
    op.execute(f"CREATE TYPE scale_source AS ENUM ({members})")
    op.execute(
        "ALTER TABLE captures ALTER COLUMN scale_source TYPE scale_source "
        "USING scale_source::text::scale_source"
    )
    op.execute("DROP TYPE scale_source_old")
