"""A site's frame and epochs (decision D3): reading them, and which asset an epoch shows.

A site is captured more than once. Its first registered capture defines its frame (the
origin every epoch's tiles are placed about) and each capture of it is an epoch. A
re-run of a capture inserts a new asset revision of its epoch and marks the old one
superseded by setting its `valid_to`; nothing about the old row is rewritten. So:

* an asset is **superseded** when it belongs to an epoch and has a `valid_to`
  (`is_superseded`). A legacy asset's `valid_to` is just its validity window;
* an epoch's **current asset** is its newest observed asset with no `valid_to`
  (`current_asset_of`, `current_asset`);
* a site **shows** every asset that is not superseded (`live_assets`).
"""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from geoalchemy2 import WKBElement
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Asset, Capture, Epoch, Site, SiteFrame
from app.models.enums import (
    AssetSourceKind,
    EpochJoinMethod,
    EpochPlatform,
    EpochStatus,
    HeightDatum,
    ObservedAtSource,
    SiteFrameSource,
)
from app.schemas.epoch import EpochRead, SiteFrameRead
from app.services import geometry
from app.services.captures import quality_from
from app.services.errors import NotFoundError

log = logging.getLogger(__name__)


# ── Which asset an epoch shows ───────────────────────────────────────────────────────────


def is_superseded(asset: Asset) -> bool:
    """A revision of an epoch that a later run replaced: history, not what is shown."""
    return asset.epoch_id is not None and asset.valid_to is not None


def live_assets(site: Site) -> list[Asset]:
    """What the site shows: every asset but superseded revisions, in `Site.assets` order."""
    return [asset for asset in site.assets if not is_superseded(asset)]


def current_asset_of(epoch_id: uuid.UUID, assets: Iterable[Asset]) -> Asset | None:
    """The epoch's newest observed asset with no `valid_to`, from already-loaded rows."""
    candidates = [
        asset
        for asset in assets
        if asset.epoch_id == epoch_id
        and asset.valid_to is None
        and asset.source_kind == AssetSourceKind.OBSERVED.value
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda asset: (asset.created_at, str(asset.id)))


def current_asset(db: Session, epoch_id: uuid.UUID, *, lock: bool = False) -> Asset | None:
    """`current_asset_of` in SQL, optionally locked (`FOR UPDATE`): the row lock a sidecar
    attach takes, so a register and an attach on one asset happen one after the other."""
    statement = (
        select(Asset)
        .where(
            Asset.epoch_id == epoch_id,
            Asset.valid_to.is_(None),
            Asset.source_kind == AssetSourceKind.OBSERVED.value,
        )
        .order_by(Asset.created_at.desc(), Asset.id.desc())
        .limit(1)
    )
    if lock:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    return db.scalars(statement).first()


# ── Reading ───────────────────────────────────────────────────────────────────────────


def frame_to_read(frame: SiteFrame | None) -> SiteFrameRead | None:
    if frame is None:
        return None
    return SiteFrameRead(
        site_id=frame.site_id,
        lat=frame.lat,
        lon=frame.lon,
        height=frame.height,
        height_datum=HeightDatum(frame.height_datum),
        epsg=frame.epsg,
        source=SiteFrameSource(frame.source),
        created_at=frame.created_at,
    )


def epoch_to_read(epoch: Epoch, assets: Sequence[Asset]) -> EpochRead:
    footprint = geometry.wkb_to_footprint(epoch.footprint)
    assert footprint is not None
    current = current_asset_of(epoch.id, assets)
    return EpochRead(
        id=epoch.id,
        site_id=epoch.site_id,
        ordinal=epoch.ordinal,
        capture_id=epoch.capture_id,
        job_id=epoch.job_id,
        observed_at=epoch.observed_at,
        observed_at_source=ObservedAtSource(epoch.observed_at_source),
        footprint=footprint,
        platform=EpochPlatform(epoch.platform),
        status=EpochStatus(epoch.status),
        join_method=EpochJoinMethod(epoch.join_method),
        alignment=epoch.alignment,
        quality=quality_from(epoch.quality),
        current_asset_id=current.id if current is not None else None,
        created_at=epoch.created_at,
        updated_at=epoch.updated_at,
    )


def list_epochs(db: Session, site_id: uuid.UUID) -> list[Epoch]:
    """The site's epochs, oldest (ordinal 1) first; 404 when there is no such site."""
    if db.get(Site, site_id) is None:
        raise NotFoundError("site", site_id)
    return list(db.scalars(select(Epoch).where(Epoch.site_id == site_id).order_by(Epoch.ordinal)))


def prior_epochs(
    db: Session,
    *,
    site_id: uuid.UUID,
    footprint: WKBElement,
    before: datetime | None = None,
    exclude: uuid.UUID | None = None,
) -> list[Epoch]:
    """The site's published epochs whose footprint meets `footprint`, newest first.

    What a new epoch is compared with (alignment, change): the earlier captures of the
    same ground. `before` keeps only those observed earlier; `exclude` drops one (the
    epoch asking).
    """
    statement = select(Epoch).where(
        Epoch.site_id == site_id,
        Epoch.status == EpochStatus.PUBLISHED.value,
        func.ST_Intersects(Epoch.footprint, footprint),
    )
    if before is not None:
        statement = statement.where(Epoch.observed_at < before)
    if exclude is not None:
        statement = statement.where(Epoch.id != exclude)
    statement = statement.order_by(Epoch.observed_at.desc(), Epoch.ordinal.desc())
    return list(db.scalars(statement))


def utm_epsg(lat: float, lon: float) -> int:
    """The WGS84 UTM zone EPSG code of a point; longitude 180 is zone 60, not 61."""
    zone = min(60, math.floor((lon + 180.0) / 6.0) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def platform_of(capture: Capture) -> EpochPlatform:
    """What captured it, as far as is known before M3 reads XMP and RTK: a phone-key
    upload is a phone video, everything else is `unknown`."""
    metadata: dict[str, Any] = capture.metadata_ or {}
    if metadata.get("origin") == "phone-key":
        return EpochPlatform.PHONE_VIDEO
    return EpochPlatform.UNKNOWN
