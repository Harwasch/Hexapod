"""Who drew a site's boundary: `registration` (its epochs' footprints) or `manual`.

A re-run fits a registration-made boundary to its epochs' current footprints, so it can
shrink (`registration.fit_site_boundary`); a boundary somebody drew is only grown. Nothing
recorded which was which, so the backfill reads it off the shapes, conservatively:
`registration` only where the boundary is exactly the union of every footprint the site's
published epochs have had -- their current ones and the ones in `history` that a re-run
replaced (Spool after its grow-only v15 re-run) -- and `manual` for everything else. It
reads database rows only, in Python, as `registration_union` computes the same union.

Revision ID: 0038
Revises: 0037
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import sqlalchemy as sa
from shapely import wkb
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from alembic import op

revision = "0038"
down_revision = "0037"
branch_labels = None
depends_on = None

#: `geometry.CONTAINS_TOLERANCE_DEG`, restated: a migration does not import app code.
TOLERANCE_DEG = 1e-9


def _same(a: BaseGeometry, b: BaseGeometry) -> bool:
    return bool(a.buffer(TOLERANCE_DEG).covers(b) and b.buffer(TOLERANCE_DEG).covers(a))


def _history_footprints(history: Any) -> list[BaseGeometry]:
    shapes: list[BaseGeometry] = []
    for entry in history if isinstance(history, list) else []:
        footprint = entry.get("footprint") if isinstance(entry, dict) else None
        if isinstance(footprint, dict):
            try:
                shapes.append(shape(footprint))
            except (ValueError, TypeError, AttributeError, KeyError):
                continue
    return shapes


def upgrade() -> None:
    op.add_column(
        "sites",
        sa.Column("boundary_source", sa.String(20), nullable=False, server_default="manual"),
    )
    op.create_check_constraint(
        "ck_sites_boundary_source", "sites", "boundary_source IN ('registration', 'manual')"
    )
    bind = op.get_bind()
    drawn: dict[str, list[BaseGeometry]] = defaultdict(list)
    for site_id, footprint, history in bind.execute(
        sa.text(
            "SELECT site_id::text, ST_AsBinary(footprint), history FROM epochs"
            " WHERE status = 'published'"
        )
    ):
        if footprint is not None:
            drawn[site_id].append(wkb.loads(bytes(footprint)))
        drawn[site_id].extend(_history_footprints(history))
    registration = [
        site_id
        for site_id, boundary in bind.execute(
            sa.text("SELECT id::text, ST_AsBinary(boundary) FROM sites")
        )
        if drawn.get(site_id) and _same(wkb.loads(bytes(boundary)), unary_union(drawn[site_id]))
    ]
    if registration:
        bind.execute(
            sa.text(
                "UPDATE sites SET boundary_source = 'registration'"
                " WHERE id = ANY(CAST(:ids AS uuid[]))"
            ),
            {"ids": registration},
        )


def downgrade() -> None:
    op.drop_constraint("ck_sites_boundary_source", "sites", type_="check")
    op.drop_column("sites", "boundary_source")
