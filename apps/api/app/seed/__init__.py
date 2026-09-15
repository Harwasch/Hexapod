"""Idempotent database seeding."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Layer, Site
from app.schemas.bookmark import CameraBookmarkCreate
from app.seed.data import COMPARISON_SITES, DEMO_SITE, LAYERS
from app.services import layers as layer_service
from app.services import sites as site_service

logger = logging.getLogger("twin.seed")


def seed(db: Session) -> dict[str, int]:
    """Creates the seeded sites and layers once, and keeps the seeded sites' camera
    bookmarks in step with the code (they are tuning, not user data)."""
    created = {"sites": 0, "layers": 0}
    for site in [DEMO_SITE, *COMPARISON_SITES]:
        existing = db.scalar(select(Site).where(Site.slug == site.slug))
        if existing is None:
            site_service.create_site(db, site)
            created["sites"] += 1
        elif _bookmarks_differ(existing, site.camera_bookmarks):
            existing.bookmarks = [site_service.build_bookmark(b) for b in site.camera_bookmarks]
            db.commit()
            logger.info("refreshed bookmarks of seeded site %s", site.slug)
    for layer in LAYERS:
        if db.scalar(select(Layer.id).where(Layer.slug == layer.slug)) is None:
            layer_service.create_layer(db, layer, builtin=True)
            created["layers"] += 1
    logger.info("seeded %s", created)
    return created


def _bookmarks_differ(site: Site, wanted: list[CameraBookmarkCreate]) -> bool:
    have = sorted(
        (b.name, round(b.longitude, 7), round(b.latitude, 7), round(b.height, 3), b.heading, b.pitch)
        for b in site.bookmarks
    )
    want = sorted(
        (b.name, round(b.longitude, 7), round(b.latitude, 7), round(b.height, 3), b.heading, b.pitch)
        for b in wanted
    )
    return have != want
