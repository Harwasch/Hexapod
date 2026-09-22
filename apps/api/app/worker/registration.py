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

#: Half-width of the fallback footprint, in metres, around the placed coordinate.
#:
#: A capture's real extent is a measurement, and A8's `manifest` stage supplies it:
#: `registration.json` now carries the packaged splat's own local bounding box, and the
#: boundary below is that box on the globe. This square is what a run with no manifest
#: still gets -- a recipe without the stage, or a run under the stub -- and it stays 60 m
#: rather than being invented smaller to look precise.
PLACEHOLDER_HALF_EXTENT_M = 30.0

#: No side of a site's boundary is allowed to be thinner than this, in metres.
#:
#: A capture with no extent along an axis (one gaussian, a flat wall scanned face-on) would
#: otherwise produce a degenerate polygon that PostGIS accepts and nothing can be clicked on.
MIN_HALF_EXTENT_M = 0.5

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
    #: The packaged splat's own bounding box in the capture's east/north/up frame, in
    #: metres, when a `manifest` stage measured one. None is not an error: it is a recipe
    #: with no manifest stage, or a run under the stub runner.
    bbox_local_m: tuple[list[float], list[float]] | None = None
    #: The thumbnail artifact's file name, when the run produced one.
    thumbnail: str | None = None

    @staticmethod
    def read(path: Path) -> Registration:
        document = json.loads(path.read_text(encoding="utf-8"))
        georef = document.get("georef") or {}
        thumbnail = document.get("thumbnail")
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
            bbox_local_m=_bbox(document.get("bboxLocalM")),
            thumbnail=str(thumbnail) if thumbnail else None,
        )


def _bbox(value: object) -> tuple[list[float], list[float]] | None:
    """The manifest's `{min: [e, n, u], max: [e, n, u]}`, or None if it is not that.

    Everything the pipeline writes crosses a file boundary before it gets here, so it is
    read defensively: a malformed box falls back to the placeholder square rather than
    failing a run that otherwise succeeded.
    """
    if not isinstance(value, dict):
        return None
    low, high = value.get("min"), value.get("max")
    if not isinstance(low, list) or not isinstance(high, list):
        return None
    if len(low) != 3 or len(high) != 3:
        return None
    try:
        return ([float(v) for v in low], [float(v) for v in high])
    except (TypeError, ValueError):
        return None


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


def _rectangle(
    lat: float, lon: float, east: tuple[float, float], north: tuple[float, float]
) -> Polygon:
    """A boundary in degrees from metre offsets east and north of the placed coordinate."""
    scale = max(math.cos(math.radians(lat)), 1e-6)
    west_deg, east_deg = (v / (_METRES_PER_DEGREE * scale) for v in east)
    south_deg, north_deg = (v / _METRES_PER_DEGREE for v in north)
    ring = [
        [lon + west_deg, lat + south_deg],
        [lon + east_deg, lat + south_deg],
        [lon + east_deg, lat + north_deg],
        [lon + west_deg, lat + north_deg],
        [lon + west_deg, lat + south_deg],
    ]
    return Polygon(type="Polygon", coordinates=[ring])


def _boundary(registration: Registration) -> Polygon:
    """The site's footprint: the capture's measured extent, or the placeholder square.

    A7 drew a 60 m square around the placed coordinate and said in a comment that it was
    waiting for A8's measured extent. This is that extent -- the packaged splat's own
    bounding box, which is the thing the console will draw a site outline around, offset
    from the placed origin exactly as the gaussians are.
    """
    lat, lon = registration.lat, registration.lon
    box = registration.bbox_local_m
    if box is None:
        half = PLACEHOLDER_HALF_EXTENT_M
        return _rectangle(lat, lon, (-half, half), (-half, half))
    (min_e, min_n, _), (max_e, max_n, _) = box
    return _rectangle(lat, lon, _widen(min_e, max_e), _widen(min_n, max_n))


def _widen(low: float, high: float) -> tuple[float, float]:
    if high - low >= 2 * MIN_HALF_EXTENT_M:
        return (low, high)
    middle = (low + high) / 2
    return (middle - MIN_HALF_EXTENT_M, middle + MIN_HALF_EXTENT_M)


def _public_url(
    storage: ObjectStorage, job_id: uuid.UUID, stage_id: str, key_suffix: str
) -> str | None:
    try:
        return storage.public_url(f"{artifact_key(job_id, stage_id, key_suffix)}")
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
    thumbnail_stage_id: str | None = None,
) -> uuid.UUID | None:
    """Create (or keep) the capture's site, and mark the capture complete.

    Returns the site id, or None when there was nothing registerable — a run under the
    stub runner with no bucket configured produces no tileset to point a viewer at, and
    saying so is better than a site with a dead asset on it.
    """
    url = (
        _public_url(storage, job_id, tiles_stage_id, "splat/tileset.json")
        if tiles_stage_id is not None
        else None
    )
    # The plan's note on `thumbnail.jpg` was "the sites list -- the endpoint exists
    # already and nothing calls it". A8's thumbnail stage is what calls it.
    thumbnail = (
        _public_url(storage, job_id, thumbnail_stage_id, registration.thumbnail)
        if thumbnail_stage_id is not None and registration.thumbnail
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
                boundary=_boundary(registration),
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
    registered = db.get(Site, capture.site_id) if capture.site_id else None
    if registered is not None and thumbnail is not None:
        # Set on a re-run too: the tileset is replaced in place at the same key, so the
        # picture of it should be replaced as well.
        registered.thumbnail_url = thumbnail
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
