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
in the capture columns and in every JSON record 0032's `Provenance` or `ScaleEvidence`
would refuse to read (`JSON_FIELDS`: an asset's `render_config.provenance` and
`scaleEvidence.registeredScaleSource`, a capture's `provenance`, a layer's
`render.provenance`) -- then rebuilds both types, as 0010 does.
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
#: (table, JSONB column, path) of every stored field typed `GeorefMethod` or `ScaleSource`.
JSON_FIELDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("assets", "render_config", ("provenance", "georefMethod")),
    ("assets", "render_config", ("provenance", "scaleSource")),
    ("assets", "render_config", ("scaleEvidence", "registeredScaleSource")),
    ("captures", "provenance", ("georefMethod",)),
    ("captures", "provenance", ("scaleSource",)),
    ("layers", "render", ("provenance", "georefMethod")),
    ("layers", "render", ("provenance", "scaleSource")),
)


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
    # Every JSON record 0032's code reads these enums from: a `Provenance` in an asset's
    # render config, a capture's own column and a layer's render block; and the scale an
    # asset was registered with, kept in its `scaleEvidence` (the R2 review of #52).
    for table, column, path in JSON_FIELDS:
        folded = "none" if path[-1] == "georefMethod" else "unresolved"
        pointer = "{" + ",".join(path) + "}"
        reach = "".join(f" -> '{key}'" for key in path[:-1]) + f" ->> '{path[-1]}'"
        # Every name and value here is this module's own constant; nothing is input.
        op.execute(
            f"UPDATE {table} SET {column} = jsonb_set({column}, '{pointer}', '\"{folded}\"') "  # noqa: S608
            f"WHERE {column}{reach} = '{VALUE}'"
        )
    _rebuild("georef_method", GEOREF_METHOD)
    _rebuild("scale_source", SCALE_SOURCE)
