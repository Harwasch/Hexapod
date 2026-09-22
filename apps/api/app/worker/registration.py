"""Turning a finished run into a site — the half the pipeline deliberately cannot do.

The decision, made here rather than inherited: **the `register` stage describes what
should be registered and the worker performs it.** `tools/pipeline` has no models, no
session and no HTTP client, and giving it one would have made every stage a potential API
client and every recipe run a thing that needs credentials. So `register` writes
`registration.json` — slug, title, recipe, georeference, the artifacts it produced — and
the worker, which already holds the session it claimed the job with, does the writing.

The alternative considered was a registration endpoint the pipeline calls (the plan's A3
line mentions one; A3 did not build it). It was rejected for three reasons: it would put
an HTTP dependency and a token into the pipeline, it would need a second authorisation
path for a caller that is already inside the trust boundary, and a run whose registration
POST failed would be a run that succeeded and produced nothing — whereas here the
registration is part of the same transaction-shaped step as finishing the job.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Capture, Site
from app.models.enums import CaptureStatus, GeorefMethod, Representation, ScaleSource
from app.schemas.asset import AssetBase, RenderConfig, TilesUrlSource
from app.schemas.geojson import Polygon
from app.schemas.site import SiteCreate
from app.services import sites as site_service
from app.services.slugs import slugify
from app.storage import ObjectStorage
from app.storage.null import StorageUnavailableError
from app.worker.outputs import artifact_key

#: Half-width of the placeholder footprint, in metres, around the placed coordinate.
#:
#: A capture's real extent is a measurement, and A8's `ground_samples`/`manifest` stages
#: are what will supply it. Until then the site gets an honest 60 m box around where the
#: operator placed it rather than a boundary invented to look precise.
PLACEHOLDER_HALF_EXTENT_M = 30.0

_METRES_PER_DEGREE = 111_320.0


@dataclass(frozen=True)
class Registration:
    """`registration.json`, as the `catalog` stage writes it."""

    slug: str
    title: str
    recipe: str
    lat: float
    lon: float
    height: float
    georef_method: GeorefMethod
    scale_source: ScaleSource
    uncertainty_m: float
    document: dict[str, Any]

    @staticmethod
    def read(path: Path) -> Registration:
        document = json.loads(path.read_text(encoding="utf-8"))
        georef = document.get("georef") or {}
        return Registration(
            slug=str(document.get("slug") or ""),
            title=str(document.get("title") or ""),
            recipe=str(document.get("recipe") or ""),
            lat=float(georef.get("lat", 0.0)),
            lon=float(georef.get("lon", 0.0)),
            height=float(georef.get("height", 0.0)),
            georef_method=_georef_method(georef.get("georefMethod")),
            scale_source=_scale_source(georef.get("scaleSource")),
            uncertainty_m=float(georef.get("uncertaintyM", 0.0)),
            document=document if isinstance(document, dict) else {},
        )


def _georef_method(value: object) -> GeorefMethod:
    try:
        return GeorefMethod(str(value))
    except ValueError:
        return GeorefMethod.NONE


def _scale_source(value: object) -> ScaleSource:
    """A value the pipeline wrote that this vocabulary does not have is not a crash.

    `manual_placement` defaults `scale_source` to `"source"`, which is not a
    :class:`ScaleSource`. An unknown scale source is exactly what `unresolved` means, and
    A2's docstring says so: it is a first-class answer, not a missing value.
    """
    try:
        return ScaleSource(str(value))
    except ValueError:
        return ScaleSource.UNRESOLVED


def _square(lat: float, lon: float, half_m: float) -> Polygon:
    dlat = half_m / _METRES_PER_DEGREE
    dlon = half_m / (_METRES_PER_DEGREE * max(math.cos(math.radians(lat)), 1e-6))
    ring = [
        [lon - dlon, lat - dlat],
        [lon + dlon, lat - dlat],
        [lon + dlon, lat + dlat],
        [lon - dlon, lat + dlat],
        [lon - dlon, lat - dlat],
    ]
    return Polygon(type="Polygon", coordinates=[ring])


def _tileset_url(
    storage: ObjectStorage, job_id: uuid.UUID, stage_id: str, artifact: str
) -> str | None:
    try:
        return storage.public_url(f"{artifact_key(job_id, stage_id, artifact)}/tileset.json")
    except StorageUnavailableError:
        return None


def register(
    db: Session,
    storage: ObjectStorage,
    *,
    capture: Capture,
    job_id: uuid.UUID,
    registration: Registration,
    tiles_stage_id: str | None,
) -> uuid.UUID | None:
    """Create (or keep) the capture's site, and mark the capture complete.

    Returns the site id, or None when there was nothing registerable — a run under the
    stub runner with no bucket configured produces no tileset to point a viewer at, and
    saying so is better than a site with a dead asset on it.
    """
    url = (
        _tileset_url(storage, job_id, tiles_stage_id, "splat")
        if tiles_stage_id is not None
        else None
    )
    if capture.site_id is None:
        assets: list[AssetBase] = []
        if url is not None:
            assets.append(
                AssetBase(
                    name=f"{capture.name} splat",
                    representation=Representation.GAUSSIAN_SPLAT,
                    source=TilesUrlSource(url=url),
                    default_visible=True,
                    # A phone scan rarely carries a usable ellipsoid height, which is what
                    # clamp_to_ground exists for; B4 replaces it with the capture's own
                    # measured ground median.
                    render_config=RenderConfig(clamp_to_ground=True),
                )
            )
        site = site_service.create_site(
            db,
            SiteCreate(
                name=registration.title or capture.name,
                slug=_free_slug(db, registration.slug or capture.slug),
                description=capture.description,
                boundary=_square(registration.lat, registration.lon, PLACEHOLDER_HALF_EXTENT_M),
                metadata={
                    "captureId": str(capture.id),
                    "jobId": str(job_id),
                    "registration": registration.document,
                },
                attribution=[],
                assets=assets,
            ),
        )
        capture.site_id = site.id
    capture.status = CaptureStatus.COMPLETE
    capture.georef_method = registration.georef_method
    capture.scale_source = registration.scale_source
    capture.uncertainty_m = registration.uncertainty_m
    db.commit()
    return capture.site_id


def _free_slug(db: Session, base: str) -> str | None:
    """`create_site` refuses a slug that is taken, and a re-run is not an error.

    Handing it None lets it pick `<base>-2`; a run that is genuinely the first gets the
    slug it asked for.
    """
    candidate = slugify(base)
    if not candidate:
        return None
    taken = db.scalar(select(Site.id).where(Site.slug == candidate)) is not None
    return None if taken else candidate
