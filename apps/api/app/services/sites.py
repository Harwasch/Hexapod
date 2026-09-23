from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.models import Asset, CameraBookmark, Site
from app.schemas.asset import AssetBase, AssetRead
from app.schemas.bookmark import CameraBookmarkCreate, CameraBookmarkRead
from app.schemas.common import Attribution, GeoPosition, LicenseMetadata
from app.schemas.site import SiteCreate, SiteQuality, SiteRead, SiteSummary, SiteUpdate
from app.services import geometry
from app.services.assets import asset_to_read, build_asset
from app.services.errors import ConflictError, NotFoundError
from app.services.slugs import slugify
from app.services.urls import validate_dataset_url


def _unique_slug(db: Session, base: str) -> str:
    slug = base
    counter = 2
    while db.scalar(select(Site.id).where(Site.slug == slug)) is not None:
        slug = f"{base}-{counter}"
        counter += 1
    return slug


def _validate_asset_sources(assets: Sequence[AssetBase]) -> None:
    for asset in assets:
        if asset.source.type == "3d-tiles-url":
            validate_dataset_url(str(asset.source.url))


def create_site(db: Session, payload: SiteCreate) -> Site:
    _validate_asset_sources(payload.assets)
    if payload.slug and db.scalar(select(Site.id).where(Site.slug == payload.slug)) is not None:
        raise ConflictError(f"site slug '{payload.slug}' already exists")
    centroid = payload.centroid or geometry.centroid_of(payload.boundary)
    site = Site(
        slug=payload.slug or _unique_slug(db, slugify(payload.name)),
        name=payload.name,
        description=payload.description,
        boundary=geometry.footprint_to_wkb(payload.boundary),
        centroid=geometry.position_to_wkb(centroid),
        centroid_height=centroid.height,
        thumbnail_url=str(payload.thumbnail_url) if payload.thumbnail_url else None,
        metadata_=payload.metadata,
        attribution=[a.model_dump(mode="json", by_alias=True) for a in payload.attribution],
        license=payload.license.model_dump(mode="json", by_alias=True) if payload.license else None,
    )
    for asset_payload in payload.assets:
        site.assets.append(build_asset(asset_payload, site_id=None))
    for bookmark_payload in payload.camera_bookmarks:
        site.bookmarks.append(build_bookmark(bookmark_payload))
    db.add(site)
    db.commit()
    db.refresh(site)
    return get_site(db, site.id)


def build_bookmark(payload: CameraBookmarkCreate) -> CameraBookmark:
    return CameraBookmark(**payload.model_dump())


def list_sites(db: Session) -> list[Site]:
    stmt = (
        select(Site)
        .options(selectinload(Site.assets), selectinload(Site.bookmarks))
        .order_by(Site.created_at.asc())
    )
    return list(db.scalars(stmt).all())


def get_site(db: Session, site_id: uuid.UUID) -> Site:
    stmt = (
        select(Site)
        .options(selectinload(Site.assets), selectinload(Site.bookmarks))
        .where(Site.id == site_id)
    )
    site = db.scalar(stmt)
    if site is None:
        raise NotFoundError("site", site_id)
    return site


def get_site_by_slug(db: Session, slug: str) -> Site:
    stmt = (
        select(Site)
        .options(selectinload(Site.assets), selectinload(Site.bookmarks))
        .where(Site.slug == slug)
    )
    site = db.scalar(stmt)
    if site is None:
        raise NotFoundError("site", slug)
    return site


def update_site(db: Session, site_id: uuid.UUID, payload: SiteUpdate) -> Site:
    site = get_site(db, site_id)
    data = payload.model_dump(exclude_unset=True)
    if "boundary" in data and payload.boundary is not None:
        site.boundary = geometry.footprint_to_wkb(payload.boundary)
        if payload.centroid is None and "centroid" not in data:
            centroid = geometry.centroid_of(payload.boundary)
            site.centroid = geometry.position_to_wkb(centroid)
    if "centroid" in data and payload.centroid is not None:
        site.centroid = geometry.position_to_wkb(payload.centroid)
        site.centroid_height = payload.centroid.height
    for field in ("name", "description"):
        if field in data:
            setattr(site, field, data[field])
    if "thumbnail_url" in data:
        site.thumbnail_url = str(payload.thumbnail_url) if payload.thumbnail_url else None
    if "metadata" in data and payload.metadata is not None:
        site.metadata_ = payload.metadata
    if "attribution" in data and payload.attribution is not None:
        site.attribution = [a.model_dump(mode="json", by_alias=True) for a in payload.attribution]
    if "license" in data:
        site.license = (
            payload.license.model_dump(mode="json", by_alias=True) if payload.license else None
        )
    db.commit()
    db.refresh(site)
    return get_site(db, site.id)


def delete_site(db: Session, site_id: uuid.UUID) -> None:
    site = get_site(db, site_id)
    db.delete(site)
    db.commit()


def site_area_m2(db: Session, site: Site) -> float:
    value = db.scalar(select(func.ST_Area(func.geography(Site.boundary))).where(Site.id == site.id))
    return float(value or 0.0)


def _summary_fields(site: Site) -> tuple[list[str], datetime | None, SiteQuality | None]:
    representations = sorted({asset.representation.value for asset in site.assets})
    observed = [a.observed_at for a in site.assets if a.observed_at is not None]
    latest = max(observed) if observed else None
    quality_raw: Any = site.metadata_.get("quality") if site.metadata_ else None
    quality = SiteQuality.model_validate(quality_raw) if isinstance(quality_raw, dict) else None
    return representations, latest, quality


def site_to_summary(db: Session, site: Site) -> SiteSummary:
    representations, latest, quality = _summary_fields(site)
    return SiteSummary(
        id=site.id,
        slug=site.slug,
        name=site.name,
        description=site.description,
        centroid=geometry.wkb_to_position(site.centroid, site.centroid_height),
        area_m2=site_area_m2(db, site),
        thumbnail_url=site.thumbnail_url,
        representations=representations,
        latest_observed_at=latest,
        quality=quality,
        created_at=site.created_at,
        updated_at=site.updated_at,
    )


def site_to_read(db: Session, site: Site) -> SiteRead:
    boundary = geometry.wkb_to_footprint(site.boundary)
    assert boundary is not None
    return SiteRead(
        id=site.id,
        slug=site.slug,
        name=site.name,
        description=site.description,
        boundary=boundary,
        centroid=geometry.wkb_to_position(site.centroid, site.centroid_height),
        area_m2=site_area_m2(db, site),
        thumbnail_url=site.thumbnail_url,
        metadata=site.metadata_,
        attribution=[Attribution.model_validate(a) for a in site.attribution],
        license=LicenseMetadata.model_validate(site.license) if site.license else None,
        assets=[asset_to_read(a) for a in site.assets],
        camera_bookmarks=[CameraBookmarkRead.model_validate(b) for b in site.bookmarks],
        created_at=site.created_at,
        updated_at=site.updated_at,
    )


__all__ = [
    "Asset",
    "AssetRead",
    "GeoPosition",
    "create_site",
    "delete_site",
    "get_site",
    "get_site_by_slug",
    "list_sites",
    "site_to_read",
    "site_to_summary",
    "update_site",
]
