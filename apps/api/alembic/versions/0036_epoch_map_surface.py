"""an epoch revision's height surface is a map too (M5)

Revision ID: 0036
Revises: 0035
Create Date: 2026-10-20

`epoch_maps.kind` may now be `surface` as well as `confidence`: the per-cell heights a run
measured beside its tiles (`surface.bin`, ADR 0018), recorded per revision exactly as its
confidence grid is. Only the check constraint changes.

Downgrade deletes the surface rows first -- 0035's constraint cannot hold them -- and then
puts 0035's constraint back. Nothing else refers to a map row, so nothing else is lost;
the planes stay in the bucket beside the tiles.
"""

from alembic import op

revision = "0036"
down_revision = "0035"
branch_labels = None
depends_on = None

KINDS = ("confidence", "surface")
BEFORE = ("confidence",)


def _check(kinds: tuple[str, ...]) -> str:
    return f"kind IN ({', '.join(repr(k) for k in kinds)})"


def upgrade() -> None:
    op.drop_constraint("ck_epoch_maps_kind", "epoch_maps", type_="check")
    op.create_check_constraint("ck_epoch_maps_kind", "epoch_maps", _check(KINDS))


def downgrade() -> None:
    op.execute("DELETE FROM epoch_maps WHERE kind = 'surface'")
    op.drop_constraint("ck_epoch_maps_kind", "epoch_maps", type_="check")
    op.create_check_constraint("ck_epoch_maps_kind", "epoch_maps", _check(BEFORE))
