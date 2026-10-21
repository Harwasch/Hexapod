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

import dataclasses
import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from geoalchemy2 import WKBElement
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Asset, Capture, Epoch, Job, Site, SiteFrame
from app.models.enums import (
    AssetSourceKind,
    CaptureKind,
    CaptureStatus,
    EpochJoinMethod,
    EpochPlatform,
    EpochStatus,
    GeorefMethod,
    HeightDatum,
    ObservedAtSource,
    Representation,
    ScaleSource,
    SiteFrameSource,
)
from app.schemas.asset import AssetBase, GroundSample, RenderConfig, TilesUrlSource
from app.schemas.capture import QUALITY_BARS, QUALITY_MODES, CaptureQuality
from app.schemas.common import Provenance
from app.schemas.geojson import Polygon
from app.schemas.site import SiteCreate
from app.services import epoch_maps, epochs, fill_gate, geometry, placement, sidecars
from app.services import sites as site_service
from app.services.assets import build_asset
from app.services.slugs import slugify
from app.storage import ObjectStorage
from app.worker.carry import CarryPlan, FillCheck, plan_carry
from app.worker.outputs import artifact_key, stage_prefix
from app.worker.pipeline_bridge import PLATFORM_WORDS
from app.worker.publish import Publisher, PublishError

log = logging.getLogger("app.worker")

#: How many ground cells travel with the asset. Same number as `splat_ground`'s own
#: `max_cells` and as `RenderConfig.ground_samples`' cap: a catalog response carries a
#: measurement of the ground, not a point cloud.
MAX_GROUND_SAMPLES = 64


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
    #: How far off an estimated scale may be, in percent either side: the ±% of the
    #: evidence `exif_gps` writes beside a `camera-height-estimate` (`frame.scaleEstimate`).
    #: None for every other scale source.
    scale_uncertainty_pct: float | None = None
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
    #: What captured it, as `normalize` detected from the frames (`capture_platform`,
    #: copied up by `catalog`); None when the lane detects nothing (`ingest_splat`) or
    #: the document names something outside the epoch vocabulary.
    platform: EpochPlatform | None = None
    #: What that platform rests on (`capture_platform.PLATFORM_EVIDENCE`): `metadata`
    #: when the frames' own metadata showed it, `claimed` when only a request or the
    #: upload route said so, `none`; None in a document from before the field.
    platform_evidence: str | None = None
    #: How the run was registered to its site (`align_epoch`'s `epoch_alignment.json`, less
    #: its per-image residuals, copied up by `catalog`), or None for a recipe with no align
    #: stage (`registration.alignment_of` then records why it is unaligned).
    alignment: dict[str, Any] | None = None
    #: The confidence grid the run shipped beside its tiles, as the tileset declares it
    #: (`extras.confidence`, copied up by `catalog`; ADR 0016), or None when the run
    #: measured none (`splat-ingest`). `register` records it as the revision's map.
    confidence: dict[str, Any] | None = None

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
            scale_uncertainty_pct=_scale_uncertainty_pct(georef),
            bbox_local_m=_bbox(document.get("bboxLocalM")),
            thumbnail=str(thumbnail) if thumbnail else None,
            ground_samples=_ground_samples(document.get("ground")),
            quality=document.get("quality") if isinstance(document.get("quality"), dict) else None,
            coverage=str(document["coverage"]) if document.get("coverage") else None,
            platform=_platform(document.get("platform")),
            platform_evidence=(
                str(document["platformEvidence"])
                if isinstance(document.get("platformEvidence"), str)
                else None
            ),
            alignment=document.get("alignment")
            if isinstance(document.get("alignment"), dict)
            else None,
            confidence=document.get("confidence")
            if isinstance(document.get("confidence"), dict)
            else None,
        )


def _platform(value: object) -> EpochPlatform | None:
    """A detected platform, or None for an absent or unknown one (logged, not raised: the
    tiles are good whatever the detector called them)."""
    if value is None:
        return None
    try:
        return EpochPlatform(str(value))
    except ValueError:
        log.warning("registration.json names platform %r, which is not an epoch platform", value)
        return None


def epoch_platform(capture: Capture, registration: Registration, current: str | None) -> str:
    """The platform an epoch records after this run (decision D4's tiers).

    The platform the run's alignment was judged for (`epoch_alignment.json`'s
    `platform`), when it names one: its verdict and label are that platform's -- "survey"
    is only ever an RTK drone's -- so the epoch must say the same. Else what the run
    detected from its own frames, when it detected something; else what the epoch
    already says (a re-run that detected nothing, or only `unknown`, does not erase a
    known platform); else, for a first run, what the capture row implies
    (`epochs.platform_of`: a phone-key upload is a phone video).

    `rtk-drone` -- the survey tier -- only when the frames showed it (`platformEvidence:
    metadata`, `capture_platform.rtk_evidence`): a platform label never makes a capture
    surveyed, and the epoch has no column for what its platform rests on. A claimed one
    (a write-token client's `metadata.platform`, or a document from before the field)
    is recorded as the `drone` it claims to be, without the survey part.
    """
    alignment = registration.alignment or {}
    aligned = _platform(alignment.get("platform"))
    if aligned is EpochPlatform.RTK_DRONE and alignment.get("platformEvidence") != "metadata":
        # The same rule for the alignment's platform: it is what the run was judged for,
        # and a `platform: rtk-drone` param makes it so without the frames showing it.
        log.warning(
            "register: the alignment's rtk-drone rests on %r, not the frames' metadata; "
            "recorded as drone",
            alignment.get("platformEvidence"),
        )
        aligned = EpochPlatform.DRONE
    if aligned is not None and aligned is not EpochPlatform.UNKNOWN:
        return aligned.value
    detected = registration.platform
    if detected is EpochPlatform.RTK_DRONE and registration.platform_evidence != "metadata":
        log.warning(
            "register: rtk-drone rests on %r, not the frames' metadata; recorded as drone",
            registration.platform_evidence,
        )
        detected = EpochPlatform.DRONE
    if detected is not None and detected is not EpochPlatform.UNKNOWN:
        return detected.value
    if current is not None:
        return current
    return epochs.platform_of(capture).value


#: The scale sources that are measurements. A visually aligned epoch takes its scale from
#: the epoch it was aligned to, so it is `visual-epoch` only when that one's was one of
#: these; an estimated or unresolved scale is inherited as what it is (`_aligned_scale`).
MEASURED_SCALES = frozenset({ScaleSource.ARKIT, ScaleSource.EXIF_GPS, ScaleSource.VISUAL_EPOCH})

#: How a run with no alignment of its own was placed, in a label's words.
_PLACED_BY: dict[GeorefMethod, str] = {
    GeorefMethod.MANUAL: "placed by hand",
    GeorefMethod.EXIF_GPS: "placed by its GPS",
    GeorefMethod.ARKIT: "placed by ARKit",
}


def alignment_of(capture: Capture, registration: Registration) -> dict[str, Any]:
    """What `epochs.alignment` records for this run (decision D4).

    The run's own `epoch_alignment.json` (via `registration.json`) when its recipe has an
    align stage. Otherwise the run was never aligned to its site, and says so in the
    shape the web reads (`label`, `verdict`, `uncertaintyM`): a splat upload has no frames
    to match (M3b leaves ICP of uploaded splats for later), and any other run without an
    align stage was placed by its georeference alone. Never a claim it did not make.
    """
    if registration.alignment is not None:
        return _survey_only_for_rtk(registration.alignment)
    splat = capture.kind in (CaptureKind.GAUSSIAN_SPLAT, CaptureKind.POINT_CLOUD)
    platform = epoch_platform(capture, registration, None)
    words = "Splat upload" if splat else PLATFORM_WORDS.get(platform, "Capture")
    placed = _PLACED_BY.get(registration.georef_method, "placed by its georeference")
    uncertainty = registration.uncertainty_m if registration.uncertainty_m > 0 else None
    told = f"±{uncertainty:.2g} m" if uncertainty is not None else "uncertainty not recorded"
    return {
        "version": 1,
        "method": "none",
        "referenceEpochId": None,
        "platform": platform,
        "verdict": "unaligned",
        "confidence": 0.0,
        "uncertaintyM": uncertainty,
        "label": f"{words}, not aligned: {placed}, {told}",
        "reasons": [
            "a splat upload has no frames to match"
            if splat
            else "the run's recipe has no align stage: placed by its georeference only"
        ],
        "frameSource": "georef",
    }


def _survey_only_for_rtk(alignment: dict[str, Any]) -> dict[str, Any]:
    """The alignment as recorded, unless it claims `survey` for anything but an RTK drone
    (D4; the D6 amendment: "survey" may only be claimed for platform `rtk-drone`), or for
    one with no RTK evidence in its data (`rtkEvidence`; the platform string alone never
    makes survey -- the R2 review of #51/#52). The
    pipeline never writes that; if a run ever did, the epoch is recorded `coarse`, its label
    rebuilt in the same words `align_epoch` uses, with the reason -- not shown as surveyed."""
    if alignment.get("verdict") != "survey" or (
        alignment.get("platform") == EpochPlatform.RTK_DRONE.value
        and epochs.survey_evidenced(alignment)
    ):
        return alignment
    platform = str(alignment.get("platform") or EpochPlatform.UNKNOWN.value)
    why = (
        "survey needs evidence in the data: an RTK fix in the frames (rtkEvidence) on a "
        "platform the frames' metadata showed (platformEvidence metadata); the alignment "
        "records no RTK evidence or no such platform; recorded coarse"
        if platform == EpochPlatform.RTK_DRONE.value
        else f"survey is only an RTK drone's; recorded coarse for platform {platform!r}"
    )
    how = "aligned visually" if alignment.get("method") == "visual" else "aligned to its GNSS"
    uncertainty = _optional_number(alignment.get("uncertaintyM"))
    told = f"±{uncertainty:.2g} m" if uncertainty is not None else "uncertainty not recorded"
    reasons = alignment.get("reasons")
    log.error("register: an alignment claimed survey without its evidence: %s", why)
    return {
        **alignment,
        "verdict": "coarse",
        "label": f"{PLATFORM_WORDS.get(platform, 'Capture')}, {how}, {told}",
        "reasons": [
            *(reasons if isinstance(reasons, list) else []),
            why,
        ],
    }


def _aligned_scale(db: Session, reference_id: object) -> tuple[ScaleSource, float | None]:
    """The scale a visually aligned epoch has: its reference epoch's, carried by the
    alignment's similarity. `visual-epoch` when that was measured; else that word itself
    (an estimate keeps its +/-% from the reference's asset); `unresolved` when the
    reference is gone or never said."""
    try:
        epoch = db.get(Epoch, uuid.UUID(str(reference_id)))
    except ValueError:
        epoch = None
    capture = db.get(Capture, epoch.capture_id) if epoch and epoch.capture_id else None
    source = capture.scale_source if capture is not None else None
    if source is None:
        return ScaleSource.UNRESOLVED, None
    if source in MEASURED_SCALES:
        return ScaleSource.VISUAL_EPOCH, None
    if source is ScaleSource.CAMERA_HEIGHT_ESTIMATE and epoch is not None:
        current = epochs.current_asset(db, epoch.id)
        provenance = (current.render_config or {}).get("provenance") if current else None
        pct = (
            _optional_number(provenance.get("scaleUncertaintyPct"))
            if isinstance(provenance, dict)
            else None
        )
        return source, pct
    return source, None


def as_aligned(db: Session, registration: Registration) -> Registration:
    """The registration with its provenance as the alignment has it (M3b.2).

    Its uncertainty is the alignment's `uncertaintyM` whenever the alignment states one
    (what the epoch is claimed to: an unaligned epoch's is its georeference's anyway). An
    epoch the alignment placed visually -- `method: visual`, aligned -- was placed by its
    reference, not by its own GNSS: `georef_method` is `visual-epoch`, and its scale the
    reference's (`_aligned_scale`). One that fell back to, or kept, its own GNSS frame
    keeps the georeference's provenance.
    """
    alignment = registration.alignment or {}
    changes: dict[str, Any] = {}
    uncertainty = _optional_number(alignment.get("uncertaintyM"))
    if uncertainty is not None and uncertainty > 0:
        changes["uncertainty_m"] = uncertainty
    if alignment.get("method") == "visual" and alignment.get("verdict") in ("survey", "coarse"):
        scale, pct = _aligned_scale(db, alignment.get("referenceEpochId"))
        changes.update(
            georef_method=GeorefMethod.VISUAL_EPOCH, scale_source=scale, scale_uncertainty_pct=pct
        )
    return dataclasses.replace(registration, **changes) if changes else registration


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
    A2's docstring says so: it is a first-class answer, not a missing value. A known one
    passes through as itself -- `camera-height-estimate` included, which is an estimate
    rather than nothing, and must not be folded into `unresolved` on the way in.
    """
    try:
        return ScaleSource(str(value))
    except ValueError:
        return ScaleSource.UNRESOLVED


def _scale_uncertainty_pct(georef: dict[str, Any]) -> float | None:
    """The ±% of a `camera-height-estimate` scale, out of `frame.scaleEstimate`; else None.

    Only for that source: a measured scale has no such figure, and an unresolved one has
    no scale for it to describe. Read defensively, as everything from `registration.json`.
    """
    if _scale_source(georef.get("scaleSource")) is not ScaleSource.CAMERA_HEIGHT_ESTIMATE:
        return None
    frame = georef.get("frame")
    evidence = frame.get("scaleEstimate") if isinstance(frame, dict) else None
    pct = _optional_number(evidence.get("uncertaintyPct")) if isinstance(evidence, dict) else None
    return pct if pct is not None and pct >= 0 else None


def _boundary(registration: Registration) -> Polygon:
    """The site's footprint: the capture's measured extent about its placed coordinate, or
    the placeholder square (`placement.boundary`, which `set_scale` resizes about the same
    origin). A run placed in a site frame (`georef.siteFrame`) has its box about the frame's
    origin, up to 2 km from the capture, so its corners are placed exactly."""
    georef = registration.document.get("georef")
    in_frame = isinstance(georef, dict) and isinstance(georef.get("siteFrame"), dict)
    return placement.boundary(
        registration.lat,
        registration.lon,
        registration.bbox_local_m,
        frame_height=registration.height if in_frame else None,
    )


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
    #: True when the tileset went into a content-hashed generation (two buckets,
    #: `Publisher.generation`), where the same URL is the same bytes. With one bucket a
    #: site points at the run's own keys, which a Refine rewrites in place, so a URL
    #: says nothing about what is behind it; False is the answer that is always safe.
    content_addressed: bool = False


class LiveMoved(Exception):  # noqa: N818 - an event, as `StopRequested` is
    """The asset moved while the run was publishing: an attach cut a generation from the
    one the publish carried from. Repointing now would drop what it attached, so nothing
    is written, and the caller publishes again on top of `url` (`runner`)."""

    def __init__(self, url: str | None) -> None:
        super().__init__(f"the asset now points at {url}")
        self.url = url


def splat_asset(db: Session, site_id: uuid.UUID, *, lock: bool = False) -> Asset | None:
    """The splat the site shows: what a test or a reader means by "the site's splat".

    The current observed asset of the site's highest-ordinal epoch that has one
    (`services/epochs.current_asset`); for a legacy site with no epochs, its first splat
    by creation, as `Site.assets` orders them. Locked (`FOR UPDATE`) it is the row lock the
    sidecar attach takes (`app/services/attach.py`).
    """
    for epoch in db.scalars(
        select(Epoch).where(Epoch.site_id == site_id).order_by(Epoch.ordinal.desc())
    ):
        current = epochs.current_asset(db, epoch.id, lock=lock)
        if current is not None:
            return current
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


def legacy_splat(db: Session, capture: Capture, *, lock: bool = False) -> Asset | None:
    """The splat a capture registered before epochs existed, when the 0032 backfill could
    not make an epoch of it: its site's first splat by creation that belongs to no epoch
    (the one the worker repointed until then). None for a capture of no site."""
    if capture.site_id is None:
        return None
    statement = (
        select(Asset)
        .where(
            Asset.site_id == capture.site_id,
            Asset.representation == Representation.GAUSSIAN_SPLAT,
            Asset.epoch_id.is_(None),
        )
        .order_by(Asset.created_at, Asset.id)
        .limit(1)
    )
    if lock:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    return db.scalars(statement).first()


def live_tileset_url(db: Session, capture: Capture | None) -> str | None:
    """The tileset the capture's epoch shows now: what a republish carries sidecars from.

    Only from the same epoch (D3): a capture that is not an epoch yet -- the first run of
    a new site, or the next epoch of an existing one -- carries nothing, because another
    epoch's sidecars were made on other tiles of another day.
    """
    if capture is None:
        return None
    epoch = epochs.epoch_of_capture(db, capture.id)
    if epoch is None:
        # A capture registered before epochs: what its site showed for it, as before.
        return tileset_url(legacy_splat(db, capture))
    return tileset_url(epochs.current_asset(db, epoch.id))


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
    tiles_dir: Path | None = None,
    fill_check: FillCheck | None = None,
) -> Published:
    """Copy the run's browser-facing outputs to the public bucket. No database here --
    except what `fill_check` (the fill gate each carried inferred layer must pass,
    `fill_gate.carry_check`) reads, which ends its own transaction where the caller asks.

    `carry_from` is the tileset the site shows now (`live_tileset_url`): the sidecars beside
    it that still hold for the new tiles go into the new generation with them
    (`app/worker/carry.py`), and the rest are reported as dropped, for `register` to flag.
    `tiles_dir` is the run's tileset directory in the workdir, where the worker still has
    it: what a kind bound to positions is checked against, read from disk rather than
    downloaded again.

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
                tiles_dir=tiles_dir,
                fill_check=fill_check,
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
    return Published(
        tileset=url,
        thumbnail=thumbnail,
        coverage=coverage,
        carry=plan,
        content_addressed=publisher.splits_buckets,
    )


class FrameMismatch(Exception):  # noqa: N818 - a refusal, as `LiveMoved` is an event
    """The run's tiles are placed about another origin than its site's frame.

    Exactly equal or refused: the frame's numbers reach the pipeline as JSON (Python's
    shortest round-trip `repr`) and come back the same way, so a run placed in the frame
    has the frame's origin to the last bit, and one that does not was placed elsewhere --
    a job whose params named another frame, or a resumed run whose frame changed. Nothing
    is written; the runner dead-letters the job (`runner._finish_complete`).
    """


@dataclass(frozen=True)
class FrameEcho:
    """The `siteFrame` block a placing stage wrote into `georef.json`: the frame it used."""

    site_id: uuid.UUID
    lat: float
    lon: float
    height: float


def _frame_echo(registration: Registration) -> FrameEcho | None:
    """`georef.siteFrame`, or None for a run that was given no frame.

    Refused when it is malformed or names another origin than the georeference itself
    (whose top-level lat/lon/height are the frame's origin whenever a frame was given).
    """
    georef = registration.document.get("georef")
    raw = georef.get("siteFrame") if isinstance(georef, dict) else None
    if raw is None:
        return None
    try:
        if not isinstance(raw, dict):
            raise TypeError(raw)
        echo = FrameEcho(
            site_id=uuid.UUID(str(raw["siteId"])),
            lat=float(raw["lat"]),
            lon=float(raw["lon"]),
            height=float(raw["height"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FrameMismatch(f"the run's georef.siteFrame is malformed: {raw!r}") from error
    if (echo.lat, echo.lon, echo.height) != (
        registration.lat,
        registration.lon,
        registration.height,
    ):
        raise FrameMismatch(
            f"the run says it was placed in site {echo.site_id}'s frame at "
            f"({echo.lat}, {echo.lon}, {echo.height}) but its georeference is about "
            f"({registration.lat}, {registration.lon}, {registration.height})"
        )
    return echo


def _check_frame(site: Site, registration: Registration, echo: FrameEcho | None) -> None:
    """A run of an epoch of `site` must be placed about the site's frame origin, exactly."""
    frame = site.frame
    if frame is None:
        return
    if echo is not None and echo.site_id != site.id:
        raise FrameMismatch(
            f"the run was placed in site {echo.site_id}'s frame, but its capture is an epoch "
            f"of site {site.id}"
        )
    if (frame.lat, frame.lon, frame.height) != (
        registration.lat,
        registration.lon,
        registration.height,
    ):
        raise FrameMismatch(
            f"site {site.id}'s frame is about ({frame.lat}, {frame.lon}, {frame.height}) but "
            f"the run's tiles are placed about ({registration.lat}, {registration.lon}, "
            f"{registration.height})"
        )


def _observed_at(capture: Capture, registration: Registration) -> tuple[datetime, ObservedAtSource]:
    """When the epoch was captured, and on whose say-so.

    The rule, stated once (ADR 0013, DATA_MODEL `epochs.observed_at_source`):

    1. the manifest's tz-aware `capture.capturedAtUtc` (M3a writes it from the files' own
       clocks), labelled `manifest`. It is **not** copied into `capture.captured_at`:
       that column is what a person or the console said, and a date that came from the
       files must not pass for one. First, because the files' own clock, with a zone, is
       the better evidence of when the capture was made;
    2. the capture row's `captured_at` (what a person or the console said), `captured-at`;
    3. the manifest's naive `capture.capturedAt` (a video's container date), `manifest`,
       taken as UTC;
    4. the upload's `created_at`, `created`.
    """
    manifest = registration.document.get("manifest")
    facts = manifest.get("capture") if isinstance(manifest, dict) else None
    facts = facts if isinstance(facts, dict) else {}
    utc = _parsed_time(facts.get("capturedAtUtc"), aware_only=True)
    if utc is not None:
        return utc, ObservedAtSource.MANIFEST
    if capture.captured_at is not None:
        return capture.captured_at, ObservedAtSource.CAPTURED_AT
    naive = _parsed_time(facts.get("capturedAt"), aware_only=False)
    if naive is not None:
        return naive, ObservedAtSource.MANIFEST
    return capture.created_at, ObservedAtSource.CREATED


def _parsed_time(raw: object, *, aware_only: bool) -> datetime | None:
    """An ISO 8601 time, or None. `aware_only` refuses one without an offset (a
    `capturedAtUtc` that is not is malformed); otherwise a naive time is taken as UTC."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None if aware_only else parsed.replace(tzinfo=UTC)
    return parsed


def _frame_from_registration(
    site_id: uuid.UUID, registration: Registration, source: SiteFrameSource
) -> SiteFrame | None:
    """A new site frame about the run's georeference; None for a run with none (the
    stub runner's), which leaves the site frameless until a run that has one."""
    origin = placement.registered_origin({"registration": registration.document})
    if origin is None:
        return None
    return SiteFrame(
        site_id=site_id,
        lat=origin.lat,
        lon=origin.lon,
        height=origin.height,
        height_datum=HeightDatum.MSL_UNVERIFIED.value,
        epsg=epochs.utm_epsg(origin.lat, origin.lon),
        source=source.value,
    )


def _new_site(db: Session, capture: Capture, job_id: uuid.UUID, registration: Registration) -> Site:
    """The site a capture's first epoch makes, in this transaction (no commit)."""
    return site_service.create_site(
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
            assets=[],
        ),
        commit=False,
    )


def _target(
    db: Session,
    capture: Capture,
    job_id: uuid.UUID,
    registration: Registration,
    echo: FrameEcho | None,
) -> tuple[Site, SiteFrame | None, EpochJoinMethod]:
    """The site a capture's first registered run makes it an epoch of, and how it got there.

    * The run was placed in a site's frame (`echo`): that site, locked, if it still has
      that frame -- `explicit` when the capture named it, else `spatial` (the worker
      matched it by location before the run). A site that has no frame or another one
      is a `FrameMismatch`; a site deleted since is re-made in the echoed frame
      (`SiteFrameSource.ECHO`), because the tiles are about that origin.
    * No frame, but the capture named a site that has none: this run is that site's first
      epoch and defines its frame (`explicit`). If that site gained a frame while the run
      was going, the run's tiles are not in it: a new site, with a warning.
    * Otherwise a new site, in a frame about this run's georeference. There is no spatial
      match here: a run given no frame has tiles about its own origin.
    """
    explicit = epochs.requested_site(db, capture)
    if echo is not None:
        site = db.get(Site, echo.site_id, with_for_update=True, populate_existing=True)
        if site is not None:
            frame = site.frame
            if frame is None or (frame.lat, frame.lon, frame.height) != (
                echo.lat,
                echo.lon,
                echo.height,
            ):
                held = (
                    "no frame" if frame is None else f"({frame.lat}, {frame.lon}, {frame.height})"
                )
                raise FrameMismatch(
                    f"the run was placed in site {site.id}'s frame at ({echo.lat}, {echo.lon}, "
                    f"{echo.height}), but that site now has {held}"
                )
            named = explicit is not None and explicit.id == site.id
            return site, frame, EpochJoinMethod.EXPLICIT if named else EpochJoinMethod.SPATIAL
        log.warning(
            "register: capture %s was placed in site %s's frame, which is gone; a new site "
            "is made in that frame",
            capture.id,
            echo.site_id,
        )
        site = _new_site(db, capture, job_id, registration)
        frame = SiteFrame(
            site_id=site.id,
            lat=echo.lat,
            lon=echo.lon,
            height=echo.height,
            height_datum=HeightDatum.MSL_UNVERIFIED.value,
            epsg=epochs.utm_epsg(echo.lat, echo.lon),
            source=SiteFrameSource.ECHO.value,
        )
        db.add(frame)
        return site, frame, EpochJoinMethod.NEW_SITE
    if explicit is not None:
        site = db.get(Site, explicit.id, with_for_update=True, populate_existing=True)
        away = _distance_from(db, site, registration)
        if site is not None and site.frame is None and away <= epochs.EXPLICIT_JOIN_MAX_M:
            frame = _frame_from_registration(site.id, registration, SiteFrameSource.REGISTRATION)
            if frame is not None:
                db.add(frame)
            return site, frame, EpochJoinMethod.EXPLICIT
        if site is not None and site.frame is None:
            # The site's frame would be fixed here for good, kilometres from what it shows.
            log.warning(
                "register: capture %s named site %s but lies %.0f m from it, past the "
                "%g m a site's frame can hold; it is a new site",
                capture.id,
                explicit.id,
                away,
                epochs.EXPLICIT_JOIN_MAX_M,
            )
        else:
            log.warning(
                "register: capture %s named site %s, which gained a frame while the run "
                "was going; its tiles are not in that frame, so it is a new site",
                capture.id,
                explicit.id,
            )
    elif _near_a_framed_site(db, registration) is not None:
        log.info(
            "register: capture %s lies within %g m of a framed site but its run was given no "
            "frame; it is a new site",
            capture.id,
            epochs.SPATIAL_JOIN_BUFFER_M,
        )
    site = _new_site(db, capture, job_id, registration)
    frame = _frame_from_registration(site.id, registration, SiteFrameSource.REGISTRATION)
    if frame is not None:
        db.add(frame)
    return site, frame, EpochJoinMethod.NEW_SITE


def _distance_from(db: Session, site: Site | None, registration: Registration) -> float:
    """Metres from the run's georeference to the site's boundary; 0 for a run with no
    georeference (the stub runner's), which fixes no frame."""
    origin = placement.registered_origin({"registration": registration.document})
    if site is None or origin is None:
        return 0.0
    return epochs.distance_to_site_m(db, site.id, origin.lat, origin.lon) or 0.0


def _near_a_framed_site(db: Session, registration: Registration) -> Site | None:
    origin = placement.registered_origin({"registration": registration.document})
    if origin is None:
        return None
    return epochs.nearest_framed_site(db, origin.lat, origin.lon)


def _new_asset(
    site: Site,
    epoch: Epoch,
    capture: Capture,
    url: str,
    registration: Registration,
    carry: CarryPlan,
    job_id: uuid.UUID,
    now: datetime,
) -> Asset:
    """An epoch's first asset: the splat today's first run made, as a revision of `epoch`."""
    asset = build_asset(
        AssetBase(
            name=f"{capture.name} splat",
            representation=Representation.GAUSSIAN_SPLAT,
            source=TilesUrlSource(url=url),
            default_visible=epoch.ordinal == 1,
            # A phone scan rarely carries a usable ellipsoid height, which is what
            # clamp_to_ground exists for. What the clamp rests on is the capture's own
            # measured ground when the run measured one, and its bounding box when not.
            render_config=RenderConfig(
                clamp_to_ground=True, ground_samples=list(registration.ground_samples)
            ),
            provenance=_provenance(registration),
        ),
        site_id=site.id,
    )
    asset.epoch_id = epoch.id
    asset.observed_at = epoch.observed_at
    # The ground this run covered, not the site's boundary (the union of every epoch's):
    # what the asset is drawn, clipped and counted as knowledge by (`land_knowledge`).
    asset.footprint = epoch.footprint
    asset.source_kind = AssetSourceKind.OBSERVED.value
    asset.sidecar_flags = [
        sidecars.flag(gone.kind, action=gone.action, reason=gone.reason, job_id=job_id, at=now)
        for gone in carry.dropped
    ]
    return asset


def _revision_render_config(
    current: Asset,
    registration: Registration,
    url: str,
    carry: CarryPlan,
    job_id: uuid.UUID,
    now: datetime,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The render config a new revision of `current` starts from, and the flags it raises.

    The user's keys of the old revision stay. The measured ground and the georeference
    provenance are replaced, because they are measurements *of the new tiles*. A runtime
    scale (`renderConfig.scale`, `PUT /assets/{id}/scale`) does not come along: it was a
    correction to the tiles it was set on, and the new tiles carry the run's own scale.

    A motion rig (`rigUrl`) is dropped when the URL changes: it is stamped with the
    checksums of the tiles it was built on and sits beside them, so on new tiles it names
    a file the new directory does not have (`app/worker/carry.py` keeps it for a
    republish of the very same tiles). What the publish could not carry is flagged, one
    entry per sidecar kind (`assets.sidecar_flags`).
    """
    render = {**dict(current.render_config), **_render_config_document(registration)}
    render.pop("scale", None)
    render.pop("scaleEvidence", None)
    flags = [
        sidecars.flag(gone.kind, action=gone.action, reason=gone.reason, job_id=job_id, at=now)
        for gone in carry.dropped
    ]
    if tileset_url(current) != url and not carry.keep_rig:
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
    return render, flags


def _supersede(
    db: Session,
    site: Site,
    epoch: Epoch,
    current: Asset,
    url: str,
    registration: Registration,
    carry: CarryPlan,
    job_id: uuid.UUID,
    now: datetime,
) -> Asset:
    """A re-run's tiles as a new revision of the epoch; the old revision becomes history.

    The old row keeps everything it had -- its URL, its render config, its footprint --
    and gets only `valid_to` (now) and `default_visible = false`, the two writes D3
    sanctions on it. A runtime scale it carried is undone on the *site's* boundary only,
    about the site's frame (or, for a frameless site, the origin it was registered at):
    the boundary drew the scaled model, and the new revision is at the run's own scale.

    Only while the epoch is the site's only one. A scale is set on a site of one epoch
    (`assets._placed_epoch`), but later epochs may have joined since and grown the
    boundary; resizing it then would resize their part of it too, about an origin that is
    not theirs. So with any other epoch on the site the boundary is left as it is -- what
    the scaled model drew stays inside it, which costs a margin, never coverage -- and
    `register` grows it to the new revision.
    """
    scale = _optional_number((current.render_config or {}).get("scale"))
    origin = (
        placement.frame_origin(site.frame)
        if site.frame is not None
        else placement.registered_origin(site.metadata_)
    )
    others = db.scalar(
        select(func.count())
        .select_from(Epoch)
        .where(Epoch.site_id == site.id, Epoch.id != epoch.id)
    )
    if scale is not None and scale > 0 and scale != 1.0 and origin is not None and not others:
        placement.rescale_site(site, origin, 1.0 / scale)
    render, flags = _revision_render_config(current, registration, url, carry, job_id, now)
    revision = Asset(
        site_id=site.id,
        name=current.name,
        representation=current.representation,
        provider=current.provider,
        source={"type": "3d-tiles-url", "url": url},
        resolution=current.resolution,
        crs=current.crs,
        license=current.license,
        attribution=current.attribution,
        render_config=render,
        default_visible=current.default_visible,
        sidecar_flags=sidecars.updated_flags(
            current.sidecar_flags, clear={*carry.carried, *carry.provided}, add=flags
        ),
        epoch_id=epoch.id,
        source_kind=AssetSourceKind.OBSERVED.value,
        observed_at=epoch.observed_at,
        # This run's ground (`_replace_run` has just put it on the epoch); the old
        # revision keeps the footprint its own run had.
        footprint=epoch.footprint,
    )
    db.add(revision)
    current.valid_to = now
    current.default_visible = False
    for entry in flags:
        log.warning("register: epoch %s: %s (%s)", epoch.id, entry["action"], entry["reason"])
    return revision


def _replace_run(
    epoch: Epoch,
    producing: uuid.UUID | None,
    footprint: WKBElement,
    quality: dict[str, Any] | None,
    now: datetime,
    *,
    published: bool,
) -> None:
    """A re-run's run, footprint and quality on its epoch, with what they replace kept.

    The three describe the epoch's current revision, so a re-run sets them; what they said
    of the revision before is appended to `epoch.history` (never rewritten), so the epoch
    still records how each earlier run saw the ground. So is its alignment, which the
    caller then replaces: a rejection a later run passed, or an earlier run's reference,
    verdict and check targets, is evidence, and stays (the R2 review of #52)."""
    previous = geometry.wkb_to_footprint(epoch.footprint)
    entry = {
        "jobId": str(epoch.job_id) if epoch.job_id is not None else None,
        "footprint": previous.model_dump(mode="json") if previous is not None else None,
        "quality": epoch.quality,
        "alignment": epoch.alignment,
        "replacedAt": now.isoformat(),
        "replacedByJobId": str(producing) if producing is not None else None,
        # The replacing run published tiles (new or the very same): its files are the
        # ones the epoch shows as of `replacedAt` (`epochs.reference_epoch` reads it).
        "tilesPublished": published,
    }
    epoch.history = [*(epoch.history or []), entry]
    epoch.job_id = producing
    epoch.footprint = footprint
    epoch.quality = quality


def _grow_boundary(site: Site, footprint: Any) -> None:
    """The site's boundary, grown to cover an epoch's footprint (never shrunk, so every
    epoch stays inside it). The centroid stays, which keeps fly-to where it was."""
    if not geometry.contains(site.boundary, footprint):
        site.boundary = geometry.union_footprints(site.boundary, footprint)


def _mirror_site_metadata(
    site: Site, registration: Registration, job_id: uuid.UUID, coverage: str | None
) -> None:
    """The deprecated mirror of the latest run on a pipeline-made site's metadata.

    `jobId`, `registration` and the coverage overlay (`coverageUrl`, `keepVerifiedPct`)
    are read off the site by `view.html`, the web's real-size tool and older tests; the
    epoch row is the record of truth now, and M2b moves those readers onto it. Only on a
    site a pipeline run made (`captureId`): a seeded or hand-made site's metadata is its
    author's. And only for the site's first epoch, the one those readers show (`register`).
    """
    metadata = dict(site.metadata_ or {})
    if not metadata.get("captureId"):
        return
    metadata["jobId"] = str(job_id)
    if isinstance(registration.document.get("georef"), dict):
        metadata["registration"] = registration.document
    # Cleared by a run with none, so an overlay never sits on a splat it was not measured on.
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
    site.metadata_ = metadata


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
    """Make the run an epoch of its site (decision D3), and mark the capture complete.

    A capture's first registered run makes it an epoch: of a new site (with a frame about
    the run's georeference), or -- `_target` -- of the site it named or whose frame its
    run was placed in. A re-run of the same capture (a Refine, a retry) is a new
    *revision* of that epoch: a new asset row, the old one superseded (`valid_to` set),
    never rewritten. A republish onto the URL the epoch already shows is no revision at
    all when it is content-addressed (two buckets: the same URL is the same bytes); with
    one bucket it is a revision all the same, since the URL cannot vouch for the bytes.

    Returns the site id, or None when there was nothing registerable.

    Pass `published` (from `publish_outputs`, run first with no transaction open) so that
    this does only database work; without it the copy happens here, inside whatever
    transaction the session holds, which is only safe for small outputs.

    Rows are locked epoch, then its current asset, then the site -- the order nothing else
    reverses (an attach locks only an asset; a scale, an asset then its site). The current
    asset must still point at the tileset the publish carried sidecars from
    (`published.carry.based_on`): if an attach moved it in between, `LiveMoved` is raised
    with nothing written. A run placed about another origin than its site's frame raises
    `FrameMismatch`, also with nothing written.
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
            fill_check=fill_gate.carry_check(
                db, (publish or Publisher(private=storage, public=storage)).public, capture.id
            ),
        )
    url, thumbnail, coverage = published.tileset, published.thumbnail, published.coverage
    registration = as_aligned(db, registration)
    echo = _frame_echo(registration)
    epoch = epochs.epoch_of_capture(db, capture.id, lock=True)
    legacy = capture.site_id is not None and epoch is None
    if epoch is not None:
        current = epochs.current_asset(db, epoch.id, lock=True)
    elif legacy:
        current = legacy_splat(db, capture, lock=True)
    else:
        current = None
    if url is not None and tileset_url(current) != published.carry.based_on:
        raise LiveMoved(tileset_url(current))
    verdict = capture_quality(registration.quality, job_id, coverage)
    quality = None if verdict is None else verdict.model_dump(mode="json", by_alias=True)
    now = datetime.now(tz=UTC)
    footprint = geometry.footprint_to_wkb(_boundary(registration))
    # The run that produced the epoch's current revision, when it is a row (a register
    # driven by hand, as some tests do, names a run nobody queued).
    producing = job_id if db.get(Job, job_id) is not None else None
    site: Site | None = None
    join = EpochJoinMethod.EXPLICIT
    if epoch is not None or legacy:
        # A re-run of an epoch, or the first run since 0032 of a capture registered into
        # a site the backfill could not make an epoch of: the same site, its frame checked.
        site_id = epoch.site_id if epoch is not None else capture.site_id
        site = db.get(Site, site_id, with_for_update=True, populate_existing=True)
    if site is not None and epoch is None:
        # A legacy capture's first epoch: `new-site` when its own earlier run made the site,
        # as the 0032 backfill labels the sites it could read; `explicit` when it was put
        # in somebody else's.
        made_it = (site.metadata_ or {}).get("captureId") == str(capture.id)
        join = EpochJoinMethod.NEW_SITE if made_it else EpochJoinMethod.EXPLICIT
    if site is not None:
        _check_frame(site, registration, echo)
        if site.frame is None:
            frame = _frame_from_registration(site.id, registration, SiteFrameSource.REGISTRATION)
            if frame is not None:
                db.add(frame)
    else:
        site, _frame, join = _target(db, capture, job_id, registration, echo)
    if epoch is None:
        observed_at, observed_source = _observed_at(capture, registration)
        source, accuracy = (
            epochs.spatial_join_point(capture) if join is EpochJoinMethod.SPATIAL else (None, None)
        )
        ordinal = (
            db.scalar(select(func.max(Epoch.ordinal)).where(Epoch.site_id == site.id)) or 0
        ) + 1
        epoch = Epoch(
            site_id=site.id,
            capture_id=capture.id,
            job_id=producing,
            ordinal=ordinal,
            observed_at=observed_at,
            observed_at_source=observed_source.value,
            footprint=footprint,
            platform=epoch_platform(capture, registration, None),
            status=EpochStatus.PUBLISHED.value,
            join_method=join.value,
            join_point_source=source,
            join_point_accuracy_m=accuracy,
            alignment=alignment_of(capture, registration),
            quality=quality,
        )
        db.add(epoch)
        db.flush()
        if current is not None:
            # The legacy splat this capture's earlier runs repointed becomes the epoch's
            # first revision, as the 0032 backfill does for the sites it could read.
            current.epoch_id = epoch.id
            current.observed_at = current.observed_at or epoch.observed_at
            current.footprint = (
                current.footprint if current.footprint is not None else epoch.footprint
            )
    else:
        # A re-run of the epoch: what it detected this time is the epoch's platform (a
        # later normalize reads more of the frames than an earlier one did), published
        # tiles or not -- the platform is a fact about the capture, not the tiles.
        epoch.platform = epoch_platform(capture, registration, epoch.platform)
        # A capture a quality gate rejected (`record_rejected`) whose run has now passed:
        # its epoch is published, dated by this run's evidence, and takes its footprint,
        # quality and alignment whether or not its tiles could be published. What they
        # replace is kept in the epoch's history (`_replace_run`).
        promoted = epoch.status == EpochStatus.REJECTED.value
        if promoted:
            epoch.status = EpochStatus.PUBLISHED.value
            observed_at, observed_source = _observed_at(capture, registration)
            epoch.observed_at = observed_at
            epoch.observed_at_source = observed_source.value
        if url is not None or promoted:
            _replace_run(epoch, producing, footprint, quality, now, published=url is not None)
            epoch.alignment = alignment_of(capture, registration)
    same_tiles = published.content_addressed and url == tileset_url(current)
    shown: Asset | None = None
    if url is not None and current is not None and not same_tiles:
        # New tiles -- or, with one bucket, a republish onto the run's own keys, which a
        # Refine rewrites in place: the URL cannot vouch for the bytes, so the new run's
        # render config gets a revision of its own rather than the old run's.
        shown = _supersede(
            db, site, epoch, current, url, registration, published.carry, job_id, now
        )
    elif url is not None and current is not None:
        # The very same tiles (a content-hashed generation it already shows): no new
        # revision, only what the publish carried or the run made clears its flags.
        current.sidecar_flags = sidecars.updated_flags(
            current.sidecar_flags,
            clear={*published.carry.carried, *published.carry.provided},
            add=[],
        )
        shown = current
    elif url is not None:
        # An epoch's first tiles: its first run, or one after a run that was withheld.
        shown = _new_asset(site, epoch, capture, url, registration, published.carry, job_id, now)
        db.add(shown)
    if shown is not None and url is not None:
        # What the run measured on the tiles it published is that revision's map; a
        # superseded revision keeps the one it was measured with (ADR 0016, `epoch_maps`).
        db.flush()
        carried_grid = (
            registration.confidence is None
            and current is not None
            and "confidence" in published.carry.carried
        )
        if carried_grid and current is not None and shown is not current:
            # The same tiles republished by a run that measured nothing: the grid the
            # publish carried into the new generation is still the old revision's map.
            epoch_maps.carry_map(db, epoch=epoch, source=current, target=shown, tileset_url=url)
        elif carried_grid:
            # That publish retried: its deterministic URL is the revision the first attempt
            # made, which already holds the carried map, and the retry carried the grid
            # again. Nothing was measured anew, so nothing about the map changes.
            pass
        else:
            epoch_maps.record_map(
                db,
                epoch=epoch,
                asset=shown,
                job_id=producing,
                descriptor=registration.confidence,
                tileset_url=url,
            )
    capture.site_id = site.id
    if url is not None:
        _grow_boundary(site, epoch.footprint)
    if thumbnail is not None:
        site.thumbnail_url = thumbnail
    if not published.withheld and epoch.ordinal == 1:
        # The mirror describes the splat `view.html` shows, which is the first epoch's
        # (`default_visible`): a later epoch's coverage laid over it would be an overlay
        # measured on other tiles. M2b reads coverage per epoch instead.
        _mirror_site_metadata(site, registration, job_id, coverage)
    capture.quality = quality
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
    must not read like one aligned to EXIF GPS. Plus, for an estimated scale, its ±%:
    "estimated" without a figure would read as nearly measured.
    """
    return Provenance(
        georef_method=registration.georef_method,
        scale_source=registration.scale_source,
        uncertainty_m=max(registration.uncertainty_m, 0.0),
        scale_uncertainty_pct=registration.scale_uncertainty_pct,
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
