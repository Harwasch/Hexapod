"""the visual-epoch georeference method and scale source (M3b)

Revision ID: 0033
Revises: 0032
Create Date: 2026-10-20

An epoch whose frames were registered into an earlier epoch of its site is placed by that
alignment, not by its own GNSS (decision D4), and its capture and asset say so:
`georef_method = 'visual-epoch'`, and -- when the earlier epoch's scale was itself
measured -- `scale_source = 'visual-epoch'`. Both go in before the type's "nothing"
member (`none`, `unresolved`), where `GeorefMethod` and `ScaleSource` declare them, so
the database's enum order stays the Python order (`test_0010_*` reads it back).

Downgrade folds the value into what 0032's schema can hold -- `none` and `unresolved`,
in the capture columns and in every asset's `render_config.provenance` (which 0032's
`Provenance` would refuse to read) -- then rebuilds both types, as 0010 does.
"""

from __future__ import annotations

from alembic import op

revision = "0033"
down_revision = "0032"
branch_labels = None
depends_on = None

#: 0032's members, in 0032's order (0004's, plus 0010's estimate before `unresolved`).
GEOREF_METHOD = ("exif-gps", "arkit", "manual", "none")
SCALE_SOURCE = ("arkit", "exif-gps", "manual", "camera-height-estimate", "unresolved")
VALUE = "visual-epoch"


def upgrade() -> None:
    # As 0010: PostgreSQL 12+ allows ADD VALUE inside a transaction as long as the new
    # value is not used in the same one, and nothing here inserts.
    op.execute(f"ALTER TYPE georef_method ADD VALUE IF NOT EXISTS '{VALUE}' BEFORE 'none'")
    op.execute(f"ALTER TYPE scale_source ADD VALUE IF NOT EXISTS '{VALUE}' BEFORE 'unresolved'")


def _rebuild(type_name: str, members: tuple[str, ...]) -> None:
    listed = ", ".join(f"'{value}'" for value in members)
    op.execute(f"ALTER TYPE {type_name} RENAME TO {type_name}_old")
    op.execute(f"CREATE TYPE {type_name} AS ENUM ({listed})")
    op.execute(
        f"ALTER TABLE captures ALTER COLUMN {type_name} TYPE {type_name} "
        f"USING {type_name}::text::{type_name}"
    )
    op.execute(f"DROP TYPE {type_name}_old")


def downgrade() -> None:
    # An enum value cannot be dropped, so each type is rebuilt. A visually aligned epoch
    # reads, to the older schema, as one with no georeference method of its own and no
    # resolved scale -- the honest subset of what it was; its uncertainty stays.
    op.execute("UPDATE captures SET georef_method = 'none' WHERE georef_method = 'visual-epoch'")
    op.execute(
        "UPDATE captures SET scale_source = 'unresolved' WHERE scale_source = 'visual-epoch'"
    )
    op.execute(
        "UPDATE assets SET render_config = jsonb_set(render_config, "
        "'{provenance,georefMethod}', '\"none\"') "
        "WHERE render_config -> 'provenance' ->> 'georefMethod' = 'visual-epoch'"
    )
    op.execute(
        "UPDATE assets SET render_config = jsonb_set(render_config, "
        "'{provenance,scaleSource}', '\"unresolved\"') "
        "WHERE render_config -> 'provenance' ->> 'scaleSource' = 'visual-epoch'"
    )
    _rebuild("georef_method", GEOREF_METHOD)
    _rebuild("scale_source", SCALE_SOURCE)
