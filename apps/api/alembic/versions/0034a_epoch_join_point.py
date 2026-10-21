"""which point a spatial join was matched by: the video's GPS or the upload's

Revision ID: 0034a
Revises: 0034
Create Date: 2026-10-20

A capture is matched to an existing site by its video's own location when the video
carries one, else by the location sent with its upload (ADR 0013, "Joining a site by the
video's GPS"). The epoch keeps which: `join_point_source` (`video` | `upload`, null for an
`explicit` or `new-site` join) and, for the video's, the horizontal accuracy the phone
wrote (`join_point_accuracy_m`).

Backfill: before this, every `spatial` join was by the located point -- a place typed for
the run, or the fix sent with the upload -- so those epochs are `upload`, accuracy null.

Named `0034a`, not `0035`, so a branch that adds its own 0035 can be rebased onto it by
changing one `down_revision`.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0034a"
down_revision = "0034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("epochs", sa.Column("join_point_source", sa.String(length=10), nullable=True))
    op.add_column("epochs", sa.Column("join_point_accuracy_m", sa.Float(), nullable=True))
    op.create_check_constraint(
        "ck_epochs_join_point_source", "epochs", "join_point_source IN ('video', 'upload')"
    )
    op.execute("UPDATE epochs SET join_point_source = 'upload' WHERE join_method = 'spatial'")


def downgrade() -> None:
    op.drop_constraint("ck_epochs_join_point_source", "epochs", type_="check")
    op.drop_column("epochs", "join_point_accuracy_m")
    op.drop_column("epochs", "join_point_source")
