"""assets: sidecar_flags, what a republish could not carry into the new generation.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-03

Sidecars -- `instances.json`, a backfilled `collision.bin`, an inferred fill, the streamed
level of detail, a plant rig -- are published beside a scan's tiles and declared on its
root `extras`. Since the one-publisher change they are attached through the API, which
cuts a new generation (`app/services/attach.py`), and a worker republish carries each one
into the run's new generation only when it is still true of the new splats
(`app/worker/carry.py`). A kind that is not carried has gone from the live site, and that
must be visible rather than only logged: "Objects need re-segmenting" on the asset, in
the API's response and the console.

So one JSON column, a list of `{kind, action, reason, jobId, flaggedAt}`, empty for every
asset that exists today: nothing has been dropped yet. A column rather than a key inside
`render_config`, because the render config is the viewer's input and is replaced whole by
a PATCH, which would erase a flag nobody had acted on.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "assets",
        sa.Column(
            "sidecar_flags",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("assets", "sidecar_flags")
