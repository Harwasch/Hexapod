"""Shared phone origin, ownership and daily creation budget (all keys share a scope)."""

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import Request
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Capture, Site
from app.services import epochs
from app.services.errors import ConflictError, NotFoundError, UnauthorizedError

ORIGIN = "phone-key"


def require_explicit_refine(request: Request, settings: Settings) -> None:
    """Legacy clients auto-call process after upload; refuse those paid requests."""
    if not settings.phone_upload_auto_process and request.headers.get("X-Hex-Refine") != "1":
        raise ConflictError("Update the app to refine in the cloud")


def capture(db: Session, capture_id: uuid.UUID) -> Capture:
    row = db.get(Capture, capture_id)
    if row is None or (row.metadata_ or {}).get("origin") != ORIGIN:
        raise UnauthorizedError("That phone key is not right.")
    return row


def site(db: Session, site_id: uuid.UUID) -> None:
    # Reads may reach a site a phone capture belongs to. Creation/processing has the
    # stricter existing phone.py rule: only join a site originally made by a phone.
    owned = db.scalar(
        select(Capture.id)
        .where(Capture.site_id == site_id, Capture.metadata_["origin"].as_string() == ORIGIN)
        .limit(1)
    )
    if owned is None:
        raise UnauthorizedError("That phone key is not right.")


def creation_budget(db: Session, settings: Settings) -> None:
    # Serialize both phone entry points until create_capture commits, so concurrent
    # first requests cannot exceed the shared rolling daily limit.
    db.execute(text("SELECT pg_advisory_xact_lock(7241923401)"))
    since = datetime.now(tz=UTC) - timedelta(hours=24)
    recent = db.scalar(
        select(func.count(Capture.id)).where(
            Capture.metadata_["origin"].as_string() == ORIGIN, Capture.created_at >= since
        )
    )
    if (recent or 0) >= settings.api_phone_daily_captures:
        raise ConflictError(
            f"The phone key has started {settings.api_phone_daily_captures} captures in "
            "the last 24 hours, which is its limit. Try again later."
        )


def check_join_site(db: Session, site_id: uuid.UUID) -> None:
    """The phone key may name only a site a phone capture made (decision D3's `siteId`).

    The key is a weak shared credential that reaches only what it created. Naming a site
    makes the capture that site's next epoch: on a site with no frame yet (every seeded,
    catalog or hand-made one) its run would fix the frame where the phone happens to be,
    add a default-visible splat, grow the boundary and take the thumbnail. So a site the
    key did not make -- seeded, catalog, hand-made, or made by a desktop upload -- is
    refused here; the write-token route (`POST /captures`) can still name any site.
    """
    site = db.get(Site, site_id)
    if site is None:
        raise NotFoundError("site", site_id)
    made_by = (site.metadata_ or {}).get("captureId")
    maker = None
    if isinstance(made_by, str):
        try:
            maker = db.get(Capture, uuid.UUID(made_by))
        except ValueError:
            maker = None
    if maker is None or (maker.metadata_ or {}).get("origin") != ORIGIN:
        raise ConflictError(
            "A phone capture can join only a site a phone capture made. Start a new site, "
            "or add this capture to that site from the console.",
            code="site_not_phone",
        )


def check_run(db: Session, capture: Capture) -> None:
    """A run the phone key starts goes into a site the key may still join.

    A re-run of an epoch goes into that epoch's site, a capture the console put on a site
    into that site, and one that named a site at creation into the named one. Each is held
    to the rule its first join was (`check_join_site`) every time the key starts a run,
    since what made it allowed can change: a site's metadata is replaced by PATCH, and the
    console may move a phone capture onto any site. The write-token routes are not limited.
    """
    epoch = epochs.epoch_of_capture(db, capture.id)
    if epoch is not None:
        site_id: uuid.UUID | None = epoch.site_id
    elif capture.site_id is not None:
        site_id = capture.site_id
    else:
        named = epochs.requested_site(db, capture)
        site_id = named.id if named is not None else None
    if site_id is not None:
        check_join_site(db, site_id)
