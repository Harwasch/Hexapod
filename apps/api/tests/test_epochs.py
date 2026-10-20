"""`services/epochs.py`: which epochs came before, and which frame a run is placed in."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from geoalchemy2 import WKBElement
from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, box
from sqlalchemy.orm import Session

from app.models import Asset, Epoch, Site, SiteFrame
from app.models.enums import AssetProvider, Representation
from app.services import epochs

ORIGIN = (47.6445, -122.138, 30.0)


def square(lon: float, lat: float, half_deg: float = 0.0005) -> WKBElement:
    """A small square footprint about (lon, lat), stored the way the columns are."""
    return from_shape(
        MultiPolygon([box(lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)]),
        srid=4326,
    )


def make_site(
    db: Session,
    slug: str = "field",
    *,
    at: tuple[float, float] = (ORIGIN[1], ORIGIN[0]),
    framed: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Site:
    lon, lat = at
    site = Site(
        slug=slug,
        name=slug,
        boundary=square(lon, lat),
        centroid=from_shape(box(lon, lat, lon, lat).centroid, srid=4326),
        metadata_=metadata or {},
        attribution=[],
    )
    db.add(site)
    db.flush()
    if framed:
        db.add(
            SiteFrame(
                site_id=site.id,
                lat=lat,
                lon=lon,
                height=ORIGIN[2],
                height_datum="msl-unverified",
                epsg=epochs.utm_epsg(lat, lon),
                source="registration",
            )
        )
        db.flush()
    return site


def make_epoch(
    db: Session,
    site: Site,
    ordinal: int,
    *,
    observed: datetime,
    footprint: WKBElement | None = None,
    status: str = "published",
    quality: dict[str, Any] | None = None,
) -> Epoch:
    epoch = Epoch(
        site_id=site.id,
        ordinal=ordinal,
        observed_at=observed,
        observed_at_source="captured-at",
        footprint=footprint if footprint is not None else site.boundary,
        platform="unknown",
        status=status,
        join_method="new-site" if ordinal == 1 else "explicit",
        quality=quality,
    )
    db.add(epoch)
    db.flush()
    return epoch


def make_asset(
    db: Session,
    site: Site,
    epoch: Epoch | None,
    url: str,
    *,
    created: datetime,
    valid_to: datetime | None = None,
) -> Asset:
    asset = Asset(
        site_id=site.id,
        name="splat",
        representation=Representation.GAUSSIAN_SPLAT,
        provider=AssetProvider.TILES_3D_URL,
        source={"type": "3d-tiles-url", "url": url},
        attribution=[],
        render_config={},
        default_visible=True,
        epoch_id=epoch.id if epoch is not None else None,
        observed_at=epoch.observed_at if epoch is not None else None,
        valid_to=valid_to,
        created_at=created,
        updated_at=created,
    )
    db.add(asset)
    db.flush()
    return asset


T0 = datetime(2026, 9, 12, tzinfo=UTC)


def test_prior_epochs_are_the_published_ones_whose_footprint_intersects(db: Session) -> None:
    site = make_site(db)
    lon, lat = ORIGIN[1], ORIGIN[0]
    first = make_epoch(db, site, 1, observed=T0)
    second = make_epoch(db, site, 2, observed=T0 + timedelta(days=30))
    make_epoch(db, site, 3, observed=T0 + timedelta(days=40), footprint=square(lon + 0.01, lat))
    make_epoch(db, site, 4, observed=T0 + timedelta(days=50), status="rejected")
    other = make_site(db, "other")
    make_epoch(db, other, 1, observed=T0)
    db.commit()

    found = epochs.prior_epochs(db, site_id=site.id, footprint=square(lon, lat, 0.0001))

    assert [epoch.id for epoch in found] == [second.id, first.id]


def test_prior_epochs_can_be_limited_to_before_a_date_and_exclude_one(db: Session) -> None:
    site = make_site(db)
    first = make_epoch(db, site, 1, observed=T0)
    second = make_epoch(db, site, 2, observed=T0 + timedelta(days=30))
    third = make_epoch(db, site, 3, observed=T0 + timedelta(days=60))
    db.commit()

    before = epochs.prior_epochs(
        db, site_id=site.id, footprint=site.boundary, before=third.observed_at
    )
    excluded = epochs.prior_epochs(db, site_id=site.id, footprint=site.boundary, exclude=second.id)

    assert [epoch.id for epoch in before] == [second.id, first.id]
    assert [epoch.id for epoch in excluded] == [third.id, first.id]


def test_utm_epsg_north_south_and_the_antimeridian() -> None:
    assert epochs.utm_epsg(47.6, -122.3) == 32610
    assert epochs.utm_epsg(-33.9, 151.2) == 32756
    assert epochs.utm_epsg(0.0, -180.0) == 32601
    assert epochs.utm_epsg(10.0, 180.0) == 32660
    assert epochs.utm_epsg(-10.0, 179.999) == 32760


def test_the_current_asset_is_the_newest_observed_one_with_no_valid_to(db: Session) -> None:
    site = make_site(db)
    epoch = make_epoch(db, site, 1, observed=T0)
    old = make_asset(
        db,
        site,
        epoch,
        "https://cdn.example.com/a/tileset.json",
        created=T0,
        valid_to=T0 + timedelta(days=1),
    )
    new = make_asset(
        db, site, epoch, "https://cdn.example.com/b/tileset.json", created=T0 + timedelta(days=1)
    )
    legacy = make_asset(
        db, site, None, "https://cdn.example.com/c/tileset.json", created=T0, valid_to=T0
    )
    db.commit()

    assert epochs.current_asset(db, epoch.id) == new
    assert epochs.current_asset_of(epoch.id, [old, new, legacy]) == new
    assert epochs.is_superseded(old) and not epochs.is_superseded(new)
    # A legacy asset's valid_to is a validity window, not a supersession.
    assert not epochs.is_superseded(legacy)
    assert epochs.current_asset(db, uuid.uuid4()) is None
