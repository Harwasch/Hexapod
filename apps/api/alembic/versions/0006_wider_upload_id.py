"""Widen capture_files.upload_id: R2's multipart handles are 343 characters.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-23

255 was enough for every provider this had been run against -- MinIO in CI, moto in the
tests -- and the first presigned upload against Cloudflare R2 answered with a 343-character
upload id, which Postgres refused. The API had already created the `captures` row by then,
so it surfaced as a 500 in the middle of a two-call sequence rather than as a validation
error, and the test suite could not have caught it: the length is the provider's, not ours.

Widened to 1024 to match `storage_key`. The column is nullable and holds an opaque handle
that only matters between `create_multipart` and `complete_multipart`, so no data is
rewritten and nothing depends on the old bound.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "capture_files",
        "upload_id",
        existing_type=sa.String(255),
        type_=sa.String(1024),
        existing_nullable=True,
    )


def downgrade() -> None:
    # Narrowing again truncates any handle longer than 255, which is every R2 one. An
    # in-progress upload whose handle is cut cannot be completed or aborted, so this is
    # only safe with no uploads in flight -- which is the ordinary condition for a
    # downgrade and is why it is written rather than refused.
    op.alter_column(
        "capture_files",
        "upload_id",
        existing_type=sa.String(1024),
        type_=sa.String(255),
        existing_nullable=True,
    )
