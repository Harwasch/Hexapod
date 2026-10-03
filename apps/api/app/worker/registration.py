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
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
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
from app.schemas.capture import QUALITY_BARS, QUALITY_MODES, CaptureQuality
from app.schemas.common import Provenance
from app.schemas.geojson import Polygon
from app.schemas.site import SiteCreate
from app.services import sidecars
from app.services import sites as site_service
from app.services.slugs import slugify
from app.storage import ObjectStorage
from app.worker.carry import CarryPlan, plan_carry
from app.worker.outputs import artifact_key, stage_prefix
from app.worker.publish import Publisher, PublishError

log = logging.getLogger("app.worker")

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
    #: The quality stage's summary (`catalog` copies it out of `quality.json`), or None
    #: for a recipe without one.
    quality: dict[str, Any] | None = None
    #: The placed coverage point cloud's file name, when the run made one with points.
    coverage: str | None = None

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
            quality=document.get("quality") if isinstance(document.get("quality"), dict) else None,
            coverage=str(document["coverage"]) if document.get("coverage") else None,
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


def capture_quality(
    summary: dict[str, Any] | None, job_id: uuid.UUID, coverage_url: str | None
) -> CaptureQuality | None:
    """`registration.json`'s quality summary as the API's typed verdict, or None.

    Read as defensively as everything else here: a summary that does not parse costs the
    capture its forecast and its Refine button, never the run that produced it.
    """
    if not summary:
        return None
    counts = _mapping(summary.get("gaussians"))
    roi = _mapping(summary.get("roi")) or None
    raw_tips = summary.get("tips")
    tips: list[Any] = raw_tips if isinstance(raw_tips, list) else []
    gsd = _mapping(summary.get("gsd"))
    views = _mapping(summary.get("views"))
    mode = str(summary.get("mode") or "refine")
    bar = str(summary.get("bar") or "everything")
    try:
        return CaptureQuality(
            job_id=job_id,
            mode=mode if mode in QUALITY_MODES else "refine",
            bar=bar if bar in QUALITY_BARS else "everything",
            bar_applied=str(summary.get("barApplied") or bar),
            keep_pct=_optional_number(summary.get("keepPct")),
            keep_verified_pct=_optional_number(summary.get("keepVerifiedPct")),
            context_pct=_optional_number(summary.get("contextPct")),
            held_out_psnr=_optional_number(summary.get("heldOutPsnr")),
            gaussians={
                "total": int(counts.get("in", 0)),
                "kept": int(counts.get("out", 0)),
                "keep": int(counts.get("keep", 0)),
                "context": int(counts.get("context", 0)),
                "drop": int(counts.get("drop", 0)),
            },
            roi=None if roi is None else {"center": roi.get("center"), "radius": roi.get("radius")},
            tips=[
                {"id": str(tip.get("id", "")), "text": str(tip.get("text", ""))}
                for tip in tips
                if isinstance(tip, dict) and tip.get("text")
            ],
            gsd_mm=_optional_number(gsd.get("medianRoiMm")),
            median_views=_optional_int(views.get("medianRoi")),
            coverage_url=coverage_url,
        )
    except (ValidationError, TypeError, ValueError):
        return None


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _optional_int(value: object) -> int | None:
    number = _optional_number(value)
    return None if number is None else int(number)


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


def _tileset_prefix(job_id: uuid.UUID, stage_id: str) -> str:
    return f"{stage_prefix(job_id, stage_id)}/splat"


def _publish_tileset(
    publish: Publisher,
    job_id: uuid.UUID,
    stage_id: str,
    generation: str | None,
    plan: CarryPlan | None = None,
) -> str | None:
    """The packaged tileset, copied to where a browser can read it, into `generation`.

    The whole `splat/` directory goes, not just `tileset.json`: the root file names the
    tiles and a site whose tiles are missing renders as nothing at all. Beside it, the
    sidecars `plan` carries from the live generation, and its `tileset.json` with their
    root extras.
    """
    prefix = _tileset_prefix(job_id, stage_id)
    return publish.publish_tree(
        prefix,
        f"{prefix}/tileset.json",
        generation=generation,
        carried=plan.objects if plan is not None else (),
        document=plan.document if plan is not None else None,
    )


@dataclass(frozen=True)
class Published:
    """What `publish_outputs` copied to the public bucket: each URL, or None.

    `withheld` says the run had a tileset that could not be published, so nothing of the
    run went on the site -- not its thumbnail, not its coverage overlay -- and `register`
    leaves a site that already exists exactly as it was.
    """

    tileset: str | None
    thumbnail: str | None
    coverage: str | None
    withheld: bool = False
    #: What the publish carried from the live generation, and what it dropped
    #: (`app/worker/carry.py`); its `based_on` is the URL `register` checks the asset
    #: still points at before repointing it.
    carry: CarryPlan = field(default_factory=lambda: CarryPlan(based_on=None))


class LiveMoved(Exception):  # noqa: N818 - an event, as `StopRequested` is
    """The asset moved while the run was publishing: an attach cut a generation from the
    one the publish carried from. Repointing now would drop what it attached, so nothing
    is written, and the caller publishes again on top of `url` (`runner`)."""

    def __init__(self, url: str | None) -> None:
        super().__init__(f"the asset now points at {url}")
        self.url = url


def splat_asset(db: Session, site_id: uuid.UUID, *, lock: bool = False) -> Asset | None:
    """The site's splat asset -- the one a run repoints -- optionally locked.

    The first by creation, as `Site.assets` orders them. Locked (`FOR UPDATE`) it is the
    same row lock the sidecar attach takes (`app/services/attach.py`), so a register and
    an attach on one asset happen one after the other, never interleaved.
    """
    statement = (
        select(Asset)
        .where(Asset.site_id == site_id, Asset.representation == Representation.GAUSSIAN_SPLAT)
        .order_by(Asset.created_at, Asset.id)
        .limit(1)
    )
    if lock:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    return db.scalars(statement).first()


def tileset_url(asset: Asset | None) -> str | None:
    source = asset.source if asset is not None and isinstance(asset.source, dict) else {}
    url = source.get("url")
    return url if source.get("type") == "3d-tiles-url" and isinstance(url, str) else None


def live_tileset_url(db: Session, capture: Capture | None) -> str | None:
    """The tileset the capture's site shows now: what a republish carries sidecars from."""
    if capture is None or capture.site_id is None:
        return None
    return tileset_url(splat_asset(db, capture.site_id))


def publish_outputs(
    storage: ObjectStorage,
    *,
    publish: Publisher | None = None,
    job_id: uuid.UUID,
    registration: Registration,
    tiles_stage_id: str | None,
    thumbnail_stage_id: str | None = None,
    coverage_stage_id: str | None = None,
    carry_from: str | None = None,
) -> Published:
    """Copy the run's browser-facing outputs to the public bucket. No database here.

    `carry_from` is the tileset the site shows now (`live_tileset_url`): the sidecars beside
    it that still hold for the new tiles go into the new generation with them
    (`app/worker/carry.py`), and the rest are reported as dropped, for `register` to flag.

    This is object-store I/O proportional to the tileset -- 514 tiles took ~8 min on the
    worker one copy at a time, and `Publisher.publish_tree` now copies eight at a time
    (about a fifteenth of the round trips' time against a fake store with a fixed
    latency) -- so the caller runs it with **no transaction open**: a session left idle in
    a transaction for minutes is killed by the database's idle-in-transaction timeout,
    and the registration after it then fails on a dead connection. That is exactly what a
    22.7M-gaussian upload hit, and a faster publish does not make the rule optional.
    """
    # With no publisher the private bucket is also the public one, which is the
    # single-bucket behaviour every caller had before the split; see app/worker/publish.py.
    publisher = publish or Publisher(private=storage, public=storage)
    tiles = _tileset_prefix(job_id, tiles_stage_id) if tiles_stage_id is not None else None
    # The plan's note on `thumbnail.jpg` was "the sites list -- the endpoint exists
    # already and nothing calls it". A8's thumbnail stage is what calls it. Beside it the
    # quality bar's coverage cloud, for the same reason: a browser fetches it.
    thumbnail_key = (
        artifact_key(job_id, thumbnail_stage_id, registration.thumbnail)
        if thumbnail_stage_id is not None and registration.thumbnail
        else None
    )
    coverage_key = (
        artifact_key(job_id, coverage_stage_id, registration.coverage)
        if coverage_stage_id is not None and registration.coverage
        else None
    )
    try:
        plan = (
            plan_carry(
                publisher,
                live_url=carry_from,
                tiles_prefix=tiles,
                entry=f"{tiles}/tileset.json",
            )
            if tiles is not None
            else CarryPlan(based_on=carry_from)
        )
    except PublishError:
        # The live generation could not be read, so what it carries is unknown. Publishing
        # anyway would drop every sidecar of a site that has nothing wrong with it.
        return Published(tileset=None, thumbnail=None, coverage=None, withheld=True)
    # One generation for everything this run puts on the globe, so the thumbnail and the
    # overlay sit beside the tiles they were made from, in keys no viewer has fetched
    # (app/services/published.py) -- the carried sidecars included, so a publish that
    # carries them never lands on one that did not. None with one bucket: nothing is copied.
    generation = publisher.generation(
        prefixes=[tiles] if tiles is not None else [],
        keys=[key for key in (thumbnail_key, coverage_key) if key is not None],
        also=plan.lines,
    )
    try:
        url = (
            _publish_tileset(publisher, job_id, tiles_stage_id, generation, plan)
            if tiles_stage_id is not None
            else None
        )
    except PublishError:
        # The run succeeded and its outputs are safely in the private bucket; only the
        # copy to the public one failed. A site pointed at half a tileset looks like a
        # working site until somebody opens it, so the site does not move to it -- and
        # nothing else of this run goes on it either: a new thumbnail over the old scan,
        # or a coverage overlay measured on geometry nobody is shown, is the same mix of
        # two runs by other means. The live generation was never touched.
        return Published(tileset=None, thumbnail=None, coverage=None, withheld=True)
    try:
        thumbnail = (
            publisher.publish_object(thumbnail_key, generation=generation)
            if thumbnail_key is not None
            else None
        )
    except PublishError:
        # A missing thumbnail is a cosmetic loss, not a reason to withhold the site.
        thumbnail = None
    # Also cosmetic, so a failed publish drops the overlay only.
    try:
        coverage = (
            publisher.publish_object(coverage_key, generation=generation)
            if coverage_key is not None
            else None
        )
    except PublishError:
        coverage = None
    return Published(tileset=url, thumbnail=thumbnail, coverage=coverage, carry=plan)


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
    coverage_stage_id: str | None = None,
    published: Published | None = None,
) -> uuid.UUID | None:
    """Create (or keep) the capture's site, and mark the capture complete.

    Returns the site id, or None when there was nothing registerable — a run under the
    stub runner with no bucket configured produces no tileset to point a viewer at, and
    saying so is better than a site with a dead asset on it.

    Pass `published` (from `publish_outputs`, run first with no transaction open) so that
    this does only database work; without it the copy happens here, inside whatever
    transaction the session holds, which is only safe for small outputs.

    Before anything is written, the site's splat asset is locked -- the row lock the
    sidecar attach takes -- and must still point at the tileset the publish carried
    sidecars from (`published.carry.based_on`). If an attach moved it in between,
    `LiveMoved` is raised with nothing written, and the caller publishes again on top of
    the attach's generation; repointing anyway would drop what it attached.
    """
    if published is None:
        published = publish_outputs(
            storage,
            publish=publish,
            job_id=job_id,
            registration=registration,
            tiles_stage_id=tiles_stage_id,
            thumbnail_stage_id=thumbnail_stage_id,
            coverage_stage_id=coverage_stage_id,
            carry_from=live_tileset_url(db, capture),
        )
    url, thumbnail, coverage = published.tileset, published.thumbnail, published.coverage
    splat = None
    if capture.site_id is not None and url is not None:
        splat = splat_asset(db, capture.site_id, lock=True)
        if tileset_url(splat) != published.carry.based_on:
            raise LiveMoved(tileset_url(splat))
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
        _repoint_splat(db, registered, url, job_id, registration, splat, published.carry)
    if registered is not None and thumbnail is not None:
        registered.thumbnail_url = thumbnail
    if registered is not None and not published.withheld:
        # The viewer reads the overlay off the site it is showing. Cleared by a run with
        # none, so an overlay never sits on a splat it was not measured on -- and left
        # alone by a run whose tileset was withheld, whose splat is not the one shown.
        metadata = dict(registered.metadata_ or {})
        if coverage:
            metadata["coverageUrl"] = coverage
        else:
            metadata.pop("coverageUrl", None)
        # Beside it, how much of the keep tier held-out frames verified, for the legend.
        verified = _optional_number((registration.quality or {}).get("keepVerifiedPct"))
        if coverage and verified is not None:
            metadata["keepVerifiedPct"] = verified
        else:
            metadata.pop("keepVerifiedPct", None)
        registered.metadata_ = metadata
    verdict = capture_quality(registration.quality, job_id, coverage)
    capture.quality = None if verdict is None else verdict.model_dump(mode="json", by_alias=True)
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
    db: Session,
    site: Site,
    url: str,
    job_id: uuid.UUID,
    registration: Registration,
    splat: Asset | None = None,
    carry: CarryPlan | None = None,
) -> None:
    """Point the site's splat asset at this run's tileset, or add one if it has none.

    A first run creates the asset; a re-run reaches here instead. A site whose first run
    produced no tileset (the stub runner with no bucket) has no asset to move, so one is
    created rather than the re-run silently registering nothing.

    The measured ground and the georeference provenance move with the tileset, because
    they are measurements *of that tileset*: leaving the first run's cells on an asset
    now pointing at the second run's geometry would be a placement derived from geometry
    nobody is looking at. A re-run that measured nothing clears them for the same reason.

    A motion rig (`rigUrl`) is dropped when the URL changes, for the same reason and one
    more. It is stamped with the checksums of the tiles it was built on
    (living-plants.yml), and it sits *beside* them -- the path is relative to the
    tileset's URL -- so after a re-run or a Refine, which publishes into a generation of
    its own, it names a file the new tileset's directory does not have and binds tiles
    the new tileset does not contain. Rig the new tiles again to animate them. The
    exception is a republish of the very same tiles, which carries the rig beside them
    (`app/worker/carry.py`).

    What the publish could not carry is flagged on the asset, one entry per sidecar kind
    (`assets.sidecar_flags`: "Objects need re-segmenting"), and what it carried or the run
    made itself clears that kind's flag.
    """
    if splat is None:
        splat = next(
            (a for a in site.assets if a.representation == Representation.GAUSSIAN_SPLAT), None
        )
    carry = carry or CarryPlan(based_on=None)
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
        render = {**dict(splat.render_config), **placement}
        now = datetime.now(tz=UTC)
        flags = [
            sidecars.flag(gone.kind, action=gone.action, reason=gone.reason, job_id=job_id, at=now)
            for gone in carry.dropped
        ]
        if dict(splat.source).get("url") != url and not carry.keep_rig:
            had_rig = render.pop("rigUrl", None) is not None
            if had_rig and all(entry["kind"] != "rig" for entry in flags):
                flags.append(
                    sidecars.flag(
                        "rig",
                        action=sidecars.RIG.action,
                        reason="the asset moved to new tiles, and its rig was stamped on the old",
                        job_id=job_id,
                        at=now,
                    )
                )
        splat.source = {**dict(splat.source), "url": url}
        splat.render_config = render
        splat.sidecar_flags = sidecars.updated_flags(
            splat.sidecar_flags, clear={*carry.carried, *carry.provided}, add=flags
        )
        for entry in flags:
            log.warning("register: asset %s: %s (%s)", splat.id, entry["action"], entry["reason"])
    metadata = dict(site.metadata_ or {})
    metadata["jobId"] = str(job_id)
    site.metadata_ = metadata
