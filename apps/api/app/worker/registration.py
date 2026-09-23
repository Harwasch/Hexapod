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

from app.models import Asset, Capture, Site
from app.models.enums import (
    AssetProvider,
    CaptureStatus,
    GeorefMethod,
    Representation,
    ScaleSource,
)
from app.schemas.asset import AssetBase, GroundSample, RenderConfig, TilesUrlSource
from app.schemas.common import Provenance
from app.schemas.geojson import Polygon
from app.schemas.site import SiteCreate
from app.services import sites as site_service
from app.services.slugs import slugify
from app.storage import ObjectStorage
from app.worker.outputs import artifact_key, stage_prefix
from app.worker.publish import Publisher, PublishError

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

#: How many ground cells travel with the asset. Same number as `splat_ground`'s own
#: `max_cells` and as `RenderConfig.ground_samples`' cap: a catalog response carries a
#: measurement of the ground, not a point cloud.
MAX_GROUND_SAMPLES = 64

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
    #: The capture's own measured ground, already on the globe. Empty is not an error: a
    #: recipe with no `ground_samples` stage, or a run under the stub runner.
    ground_samples: tuple[GroundSample, ...] = ()

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
            ground_samples=_ground_samples(document.get("ground")),
        )


def _ground_samples(value: object) -> tuple[GroundSample, ...]:
    """`ground_samples.json` turned into ellipsoid heights, or `()` if it is not that.

    One addition, done here rather than in the browser: the pipeline measures `z`, the
    capture's own up-coordinate relative to the placed origin, and `origin.height` is
    where that origin sits on the ellipsoid, so a cell's ellipsoid height is their sum.
    Doing it here means the viewer compares two heights in one datum and never has to
    know what frame the capture was reconstructed in.

    Read as defensively as `_bbox` above and for the same reason: everything the pipeline
    writes crosses a file boundary to get here, and a malformed cell drops out rather
    than failing a run that otherwise succeeded. A capture with no readable cells
    registers exactly as it did before this existed -- the viewer falls back to the
    bounding box -- which is why this never raises.
    """
    if not isinstance(value, dict):
        return ()
    origin = value.get("origin")
    base = _float(origin.get("height")) if isinstance(origin, dict) else 0.0
    rows = value.get("samples")
    if not isinstance(rows, list):
        return ()
    out: list[GroundSample] = []
    for row in rows[:MAX_GROUND_SAMPLES]:
        if not isinstance(row, dict):
            continue
        try:
            out.append(
                GroundSample(
                    lon=float(str(row["lon"])),
                    lat=float(str(row["lat"])),
                    height=base + float(str(row["z"])),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(out)


def _float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


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


def _publish_tileset(publish: Publisher, job_id: uuid.UUID, stage_id: str) -> str | None:
    """The packaged tileset, copied to where a browser can read it.

    The whole `splat/` directory goes, not just `tileset.json`: the root file names the
    tiles and a site whose tiles are missing renders as nothing at all.
    """
    prefix = f"{stage_prefix(job_id, stage_id)}/splat"
    return publish.publish_tree(prefix, f"{prefix}/tileset.json")


def _publish_object(
    publish: Publisher, job_id: uuid.UUID, stage_id: str, key_suffix: str
) -> str | None:
    return publish.publish_object(artifact_key(job_id, stage_id, key_suffix))


def register(
    db: Session,
    storage: ObjectStorage,
    *,
    publish: Publisher | None = None,
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
    # With no publisher the private bucket is also the public one, which is the
    # single-bucket behaviour every caller had before the split; see app/worker/publish.py.
    publisher = publish or Publisher(private=storage, public=storage)
    try:
        url = (
            _publish_tileset(publisher, job_id, tiles_stage_id)
            if tiles_stage_id is not None
            else None
        )
    except PublishError:
        # The run succeeded and its outputs are safely in the private bucket; only the
        # copy to the public one failed. A site pointed at half a tileset looks like a
        # working site until somebody opens it, so there is no site instead.
        url = None
    # The plan's note on `thumbnail.jpg` was "the sites list -- the endpoint exists
    # already and nothing calls it". A8's thumbnail stage is what calls it.
    try:
        thumbnail = (
            _publish_object(publisher, job_id, thumbnail_stage_id, registration.thumbnail)
            if thumbnail_stage_id is not None and registration.thumbnail
            else None
        )
    except PublishError:
        # A missing thumbnail is a cosmetic loss, not a reason to withhold the site.
        thumbnail = None
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
                    # clamp_to_ground exists for. What the clamp rests on is the capture's
                    # own measured ground when the run measured one, and its bounding box
                    # when it did not.
                    render_config=RenderConfig(
                        clamp_to_ground=True,
                        ground_samples=list(registration.ground_samples),
                    ),
                    provenance=_provenance(registration),
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
    if registered is not None and url is not None:
        # A re-run writes to `runs/<job id>/...`, which is a *different* key from the run
        # before it -- so the site does not follow the new reconstruction unless it is
        # repointed here. Without this the site keeps the first run's geometry, takes the
        # newest run's thumbnail, and the reconstruction you just paid for is invisible
        # and unreferenced. A10's reconciliation is what caught that: the second run's
        # tileset showed up in the unreferenced list while its thumbnail did not.
        #
        # The newest successful run wins. Which run that was is on the site's metadata,
        # and every run remains in the console; if a published-run pointer is ever wanted
        # it belongs on the site, not in the absence of this update.
        _repoint_splat(db, registered, url, job_id, registration)
    if registered is not None and thumbnail is not None:
        registered.thumbnail_url = thumbnail
    capture.status = CaptureStatus.COMPLETE
    capture.georef_method = registration.georef_method
    capture.scale_source = registration.scale_source
    capture.uncertainty_m = registration.uncertainty_m
    db.commit()
    return capture.site_id


def _provenance(registration: Registration) -> Provenance:
    """How the capture was placed and scaled, on the asset the console reads.

    The same three values the capture row already carries (`app/models/capture.py`), put
    where the inspector can reach them: the inspector is looking at a site's asset, not at
    the capture that produced it, and a capture placed by hand at plus or minus ten metres
    must not read like one aligned to EXIF GPS.
    """
    return Provenance(
        georef_method=registration.georef_method,
        scale_source=registration.scale_source,
        uncertainty_m=max(registration.uncertainty_m, 0.0),
    )


def _render_config_document(registration: Registration) -> dict[str, Any]:
    """The two placement keys a re-run replaces, in the render config's own camel case."""
    return {
        "groundSamples": [
            sample.model_dump(mode="json", by_alias=True) for sample in registration.ground_samples
        ],
        "provenance": _provenance(registration).model_dump(mode="json", by_alias=True),
    }


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


def _repoint_splat(
    db: Session, site: Site, url: str, job_id: uuid.UUID, registration: Registration
) -> None:
    """Point the site's splat asset at this run's tileset, or add one if it has none.

    A first run creates the asset; a re-run reaches here instead. A site whose first run
    produced no tileset (the stub runner with no bucket) has no asset to move, so one is
    created rather than the re-run silently registering nothing.

    The measured ground and the georeference provenance move with the tileset, because
    they are measurements *of that tileset*: leaving the first run's cells on an asset
    now pointing at the second run's geometry would be a placement derived from geometry
    nobody is looking at. A re-run that measured nothing clears them for the same reason.
    """
    splat = next(
        (a for a in site.assets if a.representation == Representation.GAUSSIAN_SPLAT), None
    )
    placement = _render_config_document(registration)
    if splat is None:
        db.add(
            Asset(
                site_id=site.id,
                name=f"{site.name} splat",
                representation=Representation.GAUSSIAN_SPLAT,
                provider=AssetProvider.TILES_3D_URL,
                source={"type": "3d-tiles-url", "url": url},
                default_visible=True,
                render_config={"clampToGround": True, **placement},
            )
        )
    else:
        splat.source = {**dict(splat.source), "url": url}
        splat.render_config = {**dict(splat.render_config), **placement}
    metadata = dict(site.metadata_ or {})
    metadata["jobId"] = str(job_id)
    site.metadata_ = metadata
