"""A site's surveyed check targets (M3b): the active set, read and replaced.

Replacing never updates a row in place. A target sent again with the same values is the
same survey and keeps its row; one sent with different values is a re-survey, so its old
row is retired (`retired_at`) and a new one inserted; one not sent is retired. What an
epoch was scored with is therefore always still in the table.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Site, SiteCheckTarget
from app.schemas.check_target import CheckTargetCreate, CheckTargetRead, CheckTargetSet
from app.services.errors import NotFoundError


def _require_site(db: Session, site_id: uuid.UUID, *, lock: bool = False) -> Site:
    site = db.get(Site, site_id, with_for_update=lock) if lock else db.get(Site, site_id)
    if site is None:
        raise NotFoundError("site", site_id)
    return site


def list_active(db: Session, site_id: uuid.UUID) -> list[SiteCheckTarget]:
    """The site's active targets by label. No 404 here: the worker asks for a site it has
    already resolved, and an empty list is the answer for a site with none."""
    return list(
        db.scalars(
            select(SiteCheckTarget)
            .where(SiteCheckTarget.site_id == site_id, SiteCheckTarget.retired_at.is_(None))
            .order_by(SiteCheckTarget.label, SiteCheckTarget.id)
        )
    )


def list_for_site(db: Session, site_id: uuid.UUID) -> list[SiteCheckTarget]:
    """`list_active`, with a 404 for a site that does not exist (the GET route)."""
    _require_site(db, site_id)
    return list_active(db, site_id)


def _same(row: SiteCheckTarget, target: CheckTargetCreate) -> bool:
    """The same survey of the same mark: every surveyed value equal, exactly. The numbers
    come back through JSON exactly as they went in, so equality is the right test."""
    return (
        row.role == target.role.value
        and row.lat == target.lat
        and row.lon == target.lon
        and row.height == target.height
        and row.height_datum == target.height_datum.value
        and row.sigma_h_m == target.sigma_h_m
        and row.sigma_v_m == target.sigma_v_m
        and row.source == target.source
        and row.surveyed_at == target.surveyed_at
    )


def _row(site_id: uuid.UUID, target: CheckTargetCreate) -> SiteCheckTarget:
    return SiteCheckTarget(
        site_id=site_id,
        label=target.label,
        role=target.role.value,
        lat=target.lat,
        lon=target.lon,
        height=target.height,
        height_datum=target.height_datum.value,
        sigma_h_m=target.sigma_h_m,
        sigma_v_m=target.sigma_v_m,
        source=target.source,
        surveyed_at=target.surveyed_at,
    )


def replace_active(
    db: Session, site_id: uuid.UUID, payload: CheckTargetSet
) -> list[SiteCheckTarget]:
    """Make `payload` the site's active set (module docstring) and commit; the new set.

    The site row is locked first, so two replacements of one site happen one after the
    other rather than both retiring the same row and both inserting its successor.
    """
    _require_site(db, site_id, lock=True)
    now = datetime.now(tz=UTC)
    active = {row.label: row for row in list_active(db, site_id)}
    wanted = {target.label: target for target in payload.targets}
    for label, row in active.items():
        target = wanted.get(label)
        if target is None or not _same(row, target):
            row.retired_at = now
    # The retirements reach the database before their successors do: the active-label
    # index is checked row by row, not at commit.
    db.flush()
    for label, target in wanted.items():
        kept = active.get(label)
        if kept is None or kept.retired_at is not None:
            db.add(_row(site_id, target))
    db.commit()
    return list_active(db, site_id)


def to_read(row: SiteCheckTarget) -> CheckTargetRead:
    return CheckTargetRead.model_validate(row)
