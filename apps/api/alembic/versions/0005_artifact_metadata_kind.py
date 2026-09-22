"""the metadata artifact kind the worker needs to row what it uploads

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-22
"""

from __future__ import annotations

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

# 0004's members, in 0004's order. `metadata` is appended.
ARTIFACT_KIND = (
    "frames",
    "poses",
    "masks",
    "splat",
    "deformation-field",
    "mesh",
    "point-cloud",
    "3d-tiles",
    "thumbnail",
    "ground-samples",
    "manifest",
    "clip",
)


def upgrade() -> None:
    # PostgreSQL 12+ allows ADD VALUE inside a transaction as long as the new value is
    # not used in the same one. Nothing here inserts, so this is safe under alembic's
    # transaction-per-migration.
    op.execute("ALTER TYPE artifact_kind ADD VALUE IF NOT EXISTS 'metadata'")


def downgrade() -> None:
    # An enum value cannot be dropped, so the type is rebuilt. Rows that used the value
    # being removed are folded into `manifest`, which is where they would have gone if
    # this migration had never existed.
    members = ", ".join(f"'{value}'" for value in ARTIFACT_KIND)
    op.execute("UPDATE artifacts SET kind = 'manifest' WHERE kind = 'metadata'")
    op.execute("ALTER TYPE artifact_kind RENAME TO artifact_kind_old")
    op.execute(f"CREATE TYPE artifact_kind AS ENUM ({members})")
    op.execute(
        "ALTER TABLE artifacts ALTER COLUMN kind TYPE artifact_kind USING kind::text::artifact_kind"
    )
    op.execute("DROP TYPE artifact_kind_old")
