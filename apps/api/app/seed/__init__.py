"""Idempotent database seeding."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Layer, Site
from app.schemas.bookmark import CameraBookmarkCreate
from app.schemas.geojson import MultiPolygon, Polygon
from app.schemas.site import SiteCreate
from app.seed.captures import capture_sites
from app.seed.data import DEMO_SITE, LAYERS
from app.services import geometry
from app.services import layers as layer_service
from app.services import sites as site_service
from app.services.assets import build_asset

logger = logging.getLogger("twin.seed")

#: Seeded records this deployment cannot serve, withdrawn in 2026-10 and deleted from any
#: database seeded before then. The Cesium ion samples 404 for this account's ion token
#: (adding them to the account from the ion Asset Depot is what would bring them back), and
#: three pre-pipeline captures' tiles never reached the bucket. Their definitions are in git
#: history: to restore one, put it back and take it off these lists.
WITHDRAWN_SITES = (
    "agi-hq-drone-mesh",  # ion 40866
    "boathouse-gaussian-splat",  # ion 3667783
    "chappes-church-laser-scan",  # ion 16421
    "melbourne-mesh-vs-points",  # ion 69380, 43978
    "san-francisco-aerometrex-mesh",  # ion 1415196
    "brighton-beach",  # no sites/brighton-beach/ in the bucket
    "mygla",  # no sites/mygla/
    "sheffield-park",  # no sites/sheffield-park/
)
#: Assets withdrawn from a site that stays. The demo site keeps its place, because the
#: simulated fleet is drawn inside it, but not its splat (ion 4547222).
WITHDRAWN_ASSETS = {DEMO_SITE.slug: ("Gaussian splat (LOD)",)}
WITHDRAWN_LAYERS = ("sentinel-2",)  # ion 3954


def seed(db: Session) -> dict[str, int]:
    """Creates the seeded sites and layers once, keeps the seeded sites' camera bookmarks
    in step with the code (they are tuning, not user data), and removes what was withdrawn."""
    created = {"sites": 0, "layers": 0}
    wanted = [DEMO_SITE, *capture_sites(get_settings())]
    _withdraw(db, wanted)
    for site in wanted:
        existing = db.scalar(select(Site).where(Site.slug == site.slug))
        if existing is None:
            site_service.create_site(db, site)
            created["sites"] += 1
        else:
            _refresh(db, existing, site)
    for layer in LAYERS:
        if db.scalar(select(Layer.id).where(Layer.slug == layer.slug)) is None:
            layer_service.create_layer(db, layer, builtin=True)
            created["layers"] += 1
    logger.info("seeded %s", created)
    return created


def _withdraw(db: Session, wanted: list[SiteCreate]) -> None:
    """Deletes the withdrawn sites, assets and layers a database still has. Idempotent: on
    a database without them it changes nothing."""
    removed: list[str] = []
    slugs = {site.slug for site in wanted}
    for slug in WITHDRAWN_SITES:
        site = db.scalar(select(Site).where(Site.slug == slug))
        # A capture built again on this machine (`data/tiles/<slug>/site.json`) is wanted.
        if site is not None and slug not in slugs:
            db.delete(site)
            removed.append(slug)
    for code in wanted:
        names = WITHDRAWN_ASSETS.get(code.slug or "", ())
        existing = db.scalar(select(Site).where(Site.slug == code.slug)) if names else None
        gone = [asset for asset in existing.assets if asset.name in names] if existing else []
        if existing is None or not gone:
            continue
        for asset in gone:
            db.delete(asset)
            removed.append(f"{code.slug}: {asset.name}")
        # The site's words, credits and licence described what it showed; take the code's.
        existing.description = code.description
        existing.metadata_ = dict(code.metadata)
        existing.attribution = [a.model_dump(mode="json", by_alias=True) for a in code.attribution]
        existing.license = (
            code.license.model_dump(mode="json", by_alias=True) if code.license else None
        )
    for slug in WITHDRAWN_LAYERS:
        layer = db.scalar(select(Layer).where(Layer.slug == slug, Layer.builtin.is_(True)))
        if layer is not None:
            db.delete(layer)
            removed.append(slug)
    if removed:
        db.commit()
        logger.info("withdrew %s", ", ".join(removed))


def _bookmarks_differ(site: Site, wanted: list[CameraBookmarkCreate]) -> bool:
    have = sorted(
        (
            b.name,
            round(b.longitude, 7),
            round(b.latitude, 7),
            round(b.height, 3),
            b.heading,
            b.pitch,
        )
        for b in site.bookmarks
    )
    want = sorted(
        (
            b.name,
            round(b.longitude, 7),
            round(b.latitude, 7),
            round(b.height, 3),
            b.heading,
            b.pitch,
        )
        for b in wanted
    )
    return have != want


def _refresh(db: Session, existing: Site, wanted: SiteCreate) -> None:
    """Seeded sites' bookmarks, boundary and asset footprints are tuning that lives in the
    code; bring an older database row in step without touching anything a user added."""
    changed: list[str] = []
    if _bookmarks_differ(existing, wanted.camera_bookmarks):
        existing.bookmarks = [site_service.build_bookmark(b) for b in wanted.camera_bookmarks]
        changed.append("bookmarks")
    if _rings(geometry.wkb_to_footprint(existing.boundary)) != _rings(wanted.boundary):
        existing.boundary = geometry.footprint_to_wkb(wanted.boundary)
        centroid = wanted.centroid or geometry.centroid_of(wanted.boundary)
        existing.centroid = geometry.position_to_wkb(centroid)
        existing.centroid_height = centroid.height
        changed.append("boundary")
    # A capture site is rebuilt whole by tools/captures: its manifest is the source of truth
    # for assets, quality and provenance, so new representations and updated numbers land.
    rebuilt = wanted.metadata.get("origin") == "capture-pipeline"
    if rebuilt:
        if existing.metadata_ != wanted.metadata:
            existing.metadata_ = dict(wanted.metadata)
            changed.append("metadata")
        if existing.description != wanted.description:
            existing.description = wanted.description
            changed.append("description")
        if wanted.centroid is not None and existing.centroid_height != wanted.centroid.height:
            existing.centroid = geometry.position_to_wkb(wanted.centroid)
            existing.centroid_height = wanted.centroid.height
            changed.append("centroid")
    by_name = {asset.name: asset for asset in existing.assets}
    for payload in wanted.assets:
        asset = by_name.get(payload.name)
        if asset is None:
            if rebuilt:
                existing.assets.append(build_asset(payload, site_id=existing.id))
                changed.append(f"added {payload.name}")
            continue
        if rebuilt:
            resolution = (
                payload.resolution.model_dump(mode="json", by_alias=True)
                if payload.resolution
                else None
            )
            source = payload.source.model_dump(mode="json", by_alias=True)
            if asset.resolution != resolution or asset.source != source:
                asset.resolution = resolution
                asset.source = source
                changed.append(f"quality of {payload.name}")
        if payload.footprint is not None and _rings(
            geometry.wkb_to_footprint(asset.footprint)
        ) != _rings(payload.footprint):
            asset.footprint = geometry.footprint_to_wkb(payload.footprint)
            changed.append(f"footprint of {payload.name}")
        wanted_render = payload.render_config.model_dump(mode="json", by_alias=True)
        if asset.render_config != wanted_render:
            asset.render_config = wanted_render
            changed.append(f"render config of {payload.name}")
    if changed:
        db.commit()
        logger.info("refreshed seeded site %s: %s", wanted.slug, ", ".join(changed))


def _rings(footprint: Polygon | MultiPolygon | None) -> list[list[list[tuple[float, float]]]]:
    """Polygon rings rounded to ~1 cm, so a Polygon and its MultiPolygon storage compare equal."""
    if footprint is None:
        return []
    polygons = [footprint.coordinates] if footprint.type == "Polygon" else footprint.coordinates
    return [
        [[(round(x, 7), round(y, 7)) for x, y in ring] for ring in polygon] for polygon in polygons
    ]
