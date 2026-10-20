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
from dataclasses import dataclass
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


# ── Which frame a run is placed in (decided before it starts) ─────────────────────────────

#: How far outside a framed site's boundary a capture may be and still join it (D3).
SPATIAL_JOIN_BUFFER_M = 25.0


@dataclass(frozen=True)
class RunFrame:
    """The site frame a run is told to place its capture in (`site_frame` param)."""

    site_id: uuid.UUID
    lat: float
    lon: float
    height: float

    def param(self) -> dict[str, Any]:
        return {
            "siteId": str(self.site_id),
            "lat": self.lat,
            "lon": self.lon,
            "height": self.height,
        }

    @staticmethod
    def of(frame: SiteFrame) -> RunFrame:
        return RunFrame(site_id=frame.site_id, lat=frame.lat, lon=frame.lon, height=frame.height)


def epoch_of_capture(db: Session, capture_id: uuid.UUID, *, lock: bool = False) -> Epoch | None:
    """The epoch a capture already is (a re-run's), optionally locked (`FOR UPDATE`)."""
    statement = select(Epoch).where(Epoch.capture_id == capture_id)
    if lock:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    return db.scalars(statement).first()


def requested_site(db: Session, capture: Capture) -> Site | None:
    """The site the capture named when it was created (`metadata.siteId`), if it exists."""
    raw = (capture.metadata_ or {}).get("siteId")
    if raw is None:
        return None
    try:
        site_id = uuid.UUID(str(raw))
    except ValueError:
        log.warning("capture %s: metadata.siteId %r is not a site id", capture.id, raw)
        return None
    site = db.get(Site, site_id)
    if site is None:
        log.warning("capture %s names site %s, which no longer exists", capture.id, site_id)
    return site


def _point(lat: float, lon: float) -> Any:
    return func.ST_SetSRID(func.ST_MakePoint(lon, lat), 4326)


def nearest_framed_site(
    db: Session, lat: float, lon: float, *, buffer_m: float = SPATIAL_JOIN_BUFFER_M
) -> Site | None:
    """The framed site whose boundary is within `buffer_m` of (lat, lon), nearest first.

    Only sites with a frame: a legacy site has no origin to place a new epoch about.
    """
    point = func.geography(_point(lat, lon))
    boundary = func.geography(Site.boundary)
    return db.scalars(
        select(Site)
        .join(SiteFrame, SiteFrame.site_id == Site.id)
        .where(func.ST_DWithin(boundary, point, buffer_m))
        .order_by(func.ST_Distance(boundary, point), Site.created_at, Site.id)
        .limit(1)
    ).first()


def _frame_of(db: Session, site_id: uuid.UUID) -> RunFrame | None:
    frame = db.get(SiteFrame, site_id)
    return RunFrame.of(frame) if frame is not None else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def frame_for_run(db: Session, capture: Capture) -> RunFrame | None:
    """The frame a run of this capture is placed in, or None to let it define its own.

    The first that applies:

    1. the capture is already an epoch (a re-run): that epoch's site's frame, always;
    2. the capture was registered into a site with no epoch for it (legacy): its frame;
    3. the capture named a site (`metadata.siteId`): its frame, or none yet -- the run
       is then that site's first epoch and defines it;
    4. the capture has a coordinate (`metadata.lat`/`lon`): the nearest framed site
       within `SPATIAL_JOIN_BUFFER_M`;
    5. none: the run makes a new site and frame.

    Decided before the run starts, because the pipeline places the capture in it. A
    capture located only by its EXIF cannot be matched here (nothing knows where it is
    until `normalize` runs); it joins a site only by naming it (M3 moves this later).
    """
    epoch = epoch_of_capture(db, capture.id)
    if epoch is not None:
        return _frame_of(db, epoch.site_id)
    if capture.site_id is not None:
        return _frame_of(db, capture.site_id)
    named = requested_site(db, capture)
    if named is not None:
        return _frame_of(db, named.id)
    metadata = capture.metadata_ or {}
    lat, lon = _number(metadata.get("lat")), _number(metadata.get("lon"))
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    site = nearest_framed_site(db, lat, lon)
    return _frame_of(db, site.id) if site is not None else None
