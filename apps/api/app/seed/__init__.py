"""Idempotent database seeding."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Layer, Site
from app.seed.data import COMPARISON_SITES, DEMO_SITE, LAYERS
from app.services import layers as layer_service
from app.services import sites as site_service

logger = logging.getLogger("twin.seed")


def seed(db: Session) -> dict[str, int]:
    created = {"sites": 0, "layers": 0}
    for site in [DEMO_SITE, *COMPARISON_SITES]:
        if db.scalar(select(Site.id).where(Site.slug == site.slug)) is None:
            site_service.create_site(db, site)
            created["sites"] += 1
    for layer in LAYERS:
        if db.scalar(select(Layer.id).where(Layer.slug == layer.slug)) is None:
            layer_service.create_layer(db, layer, builtin=True)
            created["layers"] += 1
    logger.info("seeded %s", created)
    return created
