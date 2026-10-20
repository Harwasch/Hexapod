"""A site's frame and epochs (decision D3): reading them, and which asset an epoch shows.

A site is captured more than once. Its first registered capture defines its frame (the
origin every epoch's tiles are placed about) and each capture of it is an epoch. A
re-run of a capture inserts a new asset revision of its epoch and marks the old one
superseded by setting its `valid_to`; nothing about the old row is rewritten. So:

* an asset is **superseded** when it belongs to an epoch and has a `valid_to`
  (`is_superseded`). A legacy asset's `valid_to` is just its validity window;
* an epoch's **current asset** is its newest observed asset with no `valid_to`
  (`current_asset_of`, `current_asset`);
* a site **shows** every asset that is not superseded (`live_assets`).
"""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from geoalchemy2 import WKBElement
from sqlalchemy import Select, String, cast, func, select
from sqlalchemy.orm import Session

from app.models import Artifact, Asset, Capture, Epoch, Job, JobStep, Site, SiteFrame
from app.models.enums import (
    AssetSourceKind,
    EpochJoinMethod,
    EpochPlatform,
    EpochStatus,
    HeightDatum,
    ObservedAtSource,
    RunStatus,
    SiteFrameSource,
)
from app.schemas.epoch import EpochRead, SiteFrameRead
from app.services import geometry
from app.services.captures import quality_from
from app.services.errors import NotFoundError

log = logging.getLogger(__name__)


# ── Which asset an epoch shows ───────────────────────────────────────────────────────────


def is_superseded(asset: Asset) -> bool:
    """A revision of an epoch that a later run replaced: history, not what is shown."""
    return asset.epoch_id is not None and asset.valid_to is not None


def live_assets(site: Site) -> list[Asset]:
    """What the site shows: every asset but superseded revisions, in `Site.assets` order."""
    return [asset for asset in site.assets if not is_superseded(asset)]


def current_asset_of(epoch_id: uuid.UUID, assets: Iterable[Asset]) -> Asset | None:
    """The epoch's newest observed asset with no `valid_to`, from already-loaded rows."""
    candidates = [
        asset
        for asset in assets
        if asset.epoch_id == epoch_id
        and asset.valid_to is None
        and asset.source_kind == AssetSourceKind.OBSERVED.value
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda asset: (asset.created_at, str(asset.id)))


def current_asset(db: Session, epoch_id: uuid.UUID, *, lock: bool = False) -> Asset | None:
    """`current_asset_of` in SQL, optionally locked (`FOR UPDATE`): the row lock a sidecar
    attach takes, so a register and an attach on one asset happen one after the other."""
    statement = (
        select(Asset)
        .where(
            Asset.epoch_id == epoch_id,
            Asset.valid_to.is_(None),
            Asset.source_kind == AssetSourceKind.OBSERVED.value,
        )
        .order_by(Asset.created_at.desc(), Asset.id.desc())
        .limit(1)
    )
    if lock:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    return db.scalars(statement).first()


# ── Reading ───────────────────────────────────────────────────────────────────────────


def frame_to_read(frame: SiteFrame | None) -> SiteFrameRead | None:
    if frame is None:
        return None
    return SiteFrameRead(
        site_id=frame.site_id,
        lat=frame.lat,
        lon=frame.lon,
        height=frame.height,
        height_datum=HeightDatum(frame.height_datum),
        epsg=frame.epsg,
        source=SiteFrameSource(frame.source),
        created_at=frame.created_at,
    )


def epoch_to_read(epoch: Epoch, assets: Sequence[Asset]) -> EpochRead:
    footprint = geometry.wkb_to_footprint(epoch.footprint)
    assert footprint is not None
    current = current_asset_of(epoch.id, assets)
    return EpochRead(
        id=epoch.id,
        site_id=epoch.site_id,
        ordinal=epoch.ordinal,
        capture_id=epoch.capture_id,
        job_id=epoch.job_id,
        observed_at=epoch.observed_at,
        observed_at_source=ObservedAtSource(epoch.observed_at_source),
        footprint=footprint,
        platform=EpochPlatform(epoch.platform),
        status=EpochStatus(epoch.status),
        join_method=EpochJoinMethod(epoch.join_method),
        alignment=epoch.alignment,
        quality=quality_from(epoch.quality),
        current_asset_id=current.id if current is not None else None,
        created_at=epoch.created_at,
        updated_at=epoch.updated_at,
    )


def list_epochs(db: Session, site_id: uuid.UUID) -> list[Epoch]:
    """The site's epochs, oldest (ordinal 1) first; 404 when there is no such site."""
    if db.get(Site, site_id) is None:
        raise NotFoundError("site", site_id)
    return list(db.scalars(select(Epoch).where(Epoch.site_id == site_id).order_by(Epoch.ordinal)))


def prior_epochs(
    db: Session,
    *,
    site_id: uuid.UUID,
    footprint: WKBElement,
    before: datetime | None = None,
    exclude: uuid.UUID | None = None,
) -> list[Epoch]:
    """The site's published epochs whose footprint meets `footprint`, newest first.

    What a new epoch is compared with (alignment, change): the earlier captures of the
    same ground. `before` keeps only those observed earlier; `exclude` drops one (the
    epoch asking).
    """
    statement = select(Epoch).where(
        Epoch.site_id == site_id,
        Epoch.status == EpochStatus.PUBLISHED.value,
        func.ST_Intersects(Epoch.footprint, footprint),
    )
    if before is not None:
        statement = statement.where(Epoch.observed_at < before)
    if exclude is not None:
        statement = statement.where(Epoch.id != exclude)
    statement = statement.order_by(Epoch.observed_at.desc(), Epoch.ordinal.desc())
    return list(db.scalars(statement))


def utm_epsg(lat: float, lon: float) -> int:
    """The WGS84 UTM zone EPSG code of a point; longitude 180 is zone 60, not 61."""
    zone = min(60, math.floor((lon + 180.0) / 6.0) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def platform_of(capture: Capture) -> EpochPlatform:
    """What captured it, as far as is known before M3 reads XMP and RTK: a phone-key
    upload is a phone video, everything else is `unknown`."""
    if is_phone_capture(capture):
        return EpochPlatform.PHONE_VIDEO
    return EpochPlatform.UNKNOWN


def _phone_origin() -> str:
    # Imported here: the phone routes import the services, not the other way round.
    from app.api.v1.phone import ORIGIN

    return ORIGIN


def is_phone_capture(capture: Capture) -> bool:
    """A capture the phone key started (`metadata.origin`, `phone.ORIGIN`)."""
    metadata: dict[str, Any] = capture.metadata_ or {}
    return bool(metadata.get("origin") == _phone_origin())


# ── Which frame a run is placed in (decided before it starts) ─────────────────────────────

#: How far outside a framed site's boundary a capture may be and still join it (D3).
SPATIAL_JOIN_BUFFER_M = 25.0


@dataclass(frozen=True)
class RunFrame:
    """The site frame a run is told to place its capture in (`site_frame` param).

    The param is the shared contract `{siteId, lat, lon, height, heightDatum, epsg}`; the
    pipeline places by the first four (`tools/pipeline/site_frame.py`) and carries the
    datum and zone for the stages that need them.
    """

    site_id: uuid.UUID
    lat: float
    lon: float
    height: float
    height_datum: str
    epsg: int

    def param(self) -> dict[str, Any]:
        return {
            "siteId": str(self.site_id),
            "lat": self.lat,
            "lon": self.lon,
            "height": self.height,
            "heightDatum": self.height_datum,
            "epsg": self.epsg,
        }

    @staticmethod
    def of(frame: SiteFrame) -> RunFrame:
        return RunFrame(
            site_id=frame.site_id,
            lat=frame.lat,
            lon=frame.lon,
            height=frame.height,
            height_datum=frame.height_datum,
            epsg=frame.epsg,
        )


def epoch_of_capture(db: Session, capture_id: uuid.UUID, *, lock: bool = False) -> Epoch | None:
    """The epoch a capture already is (a re-run's), optionally locked (`FOR UPDATE`)."""
    statement = select(Epoch).where(Epoch.capture_id == capture_id)
    if lock:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    return db.scalars(statement).first()


def requested_site(db: Session, capture: Capture) -> Site | None:
    """The site the capture named when it was created (`metadata.siteId`), if it exists."""
    raw = (capture.metadata_ or {}).get("siteId")
    if raw is None:
        return None
    try:
        site_id = uuid.UUID(str(raw))
    except ValueError:
        log.warning("capture %s: metadata.siteId %r is not a site id", capture.id, raw)
        return None
    site = db.get(Site, site_id)
    if site is None:
        log.warning("capture %s names site %s, which no longer exists", capture.id, site_id)
    return site


def _point(lat: float, lon: float) -> Any:
    return func.ST_SetSRID(func.ST_MakePoint(lon, lat), 4326)


def nearest_framed_site(
    db: Session,
    lat: float,
    lon: float,
    *,
    buffer_m: float = SPATIAL_JOIN_BUFFER_M,
    phone_made_only: bool = False,
) -> Site | None:
    """The framed site whose boundary is within `buffer_m` of (lat, lon), nearest first.

    Only sites with a frame: a legacy site has no origin to place a new epoch about. With
    `phone_made_only`, only sites a phone-key capture made (`metadata.captureId` naming
    one): the rule `phone._check_phone_site` holds a phone capture that names its site to,
    so the key cannot reach by standing in a site what it may not reach by naming it.
    """
    statement = nearest_framed_site_statement(
        lat, lon, buffer_m=buffer_m, phone_made_only=phone_made_only
    )
    return db.scalars(statement).first()


def nearest_framed_site_statement(
    lat: float,
    lon: float,
    *,
    buffer_m: float = SPATIAL_JOIN_BUFFER_M,
    phone_made_only: bool = False,
) -> Select[tuple[Site]]:
    """`nearest_framed_site`'s query. `ST_DWithin` on `geography(boundary)` is answered from
    the gist index on that expression (`ix_sites_boundary_geography`, migration 0032); the
    one on the geometry column cannot answer a distance in metres on the ellipsoid."""
    point = func.geography(_point(lat, lon))
    boundary = func.geography(Site.boundary)
    statement = (
        select(Site)
        .join(SiteFrame, SiteFrame.site_id == Site.id)
        .where(func.ST_DWithin(boundary, point, buffer_m))
    )
    if phone_made_only:
        maker = select(Capture.id).where(
            cast(Capture.id, String) == Site.metadata_["captureId"].as_string(),
            Capture.metadata_["origin"].as_string() == _phone_origin(),
        )
        statement = statement.where(maker.exists())
    return statement.order_by(func.ST_Distance(boundary, point), Site.created_at, Site.id).limit(1)


#: How far from a frameless site's boundary a capture that names it may be and still
#: define its frame: the SPZ range a tile holds about its origin (D1, `tools/pipeline/
#: site_frame.MAX_OFFSET_M`). Farther, no later epoch of what the site shows could share
#: that frame, so the capture is a site of its own (`registration._target`).
EXPLICIT_JOIN_MAX_M = 2047.0


def distance_to_site_m(db: Session, site_id: uuid.UUID, lat: float, lon: float) -> float | None:
    """Metres from (lat, lon) to the site's boundary on the ellipsoid; 0 inside it, None for
    a site that does not exist."""
    distance = db.scalar(
        select(
            func.ST_Distance(func.geography(Site.boundary), func.geography(_point(lat, lon)))
        ).where(Site.id == site_id)
    )
    return None if distance is None else float(distance)


def _frame_of(db: Session, site_id: uuid.UUID) -> RunFrame | None:
    frame = db.get(SiteFrame, site_id)
    return RunFrame.of(frame) if frame is not None else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def frame_for_run(
    db: Session, capture: Capture, located: tuple[float, float] | None = None
) -> RunFrame | None:
    """The frame a run of this capture is placed in, or None to let it define its own:
    the frame of `site_for_run`'s site, when it has one."""
    site_id = site_for_run(db, capture, located)
    return _frame_of(db, site_id) if site_id is not None else None


def site_for_run(
    db: Session, capture: Capture, located: tuple[float, float] | None = None
) -> uuid.UUID | None:
    """The site a run of this capture belongs to before it starts, or None for a new one.

    The first that applies:

    1. the capture is already an epoch (a re-run): that epoch's site, always;
    2. the capture was registered into a site with no epoch for it (legacy): that site;
    3. the capture named a site (`metadata.siteId`): that site, framed or not -- with no
       frame yet the run is its first epoch and defines it;
    4. the run has a located point (`located`, `worker/params.located_point`: a typed
       place in the job's params, else the capture's `metadata.lat`/`lon`): the nearest
       framed site within `SPATIAL_JOIN_BUFFER_M` -- for a phone-key capture, the nearest
       such site a phone capture made, as when it names one;
    5. none: the run makes a new site and frame.

    Decided before the run starts, because the pipeline places the capture in it. A
    capture located only by its EXIF cannot be matched here (nothing knows where it is
    until `normalize` runs); it joins a site only by naming it (M3 moves this later).

    The frame (`frame_for_run`), the reference epoch and the check targets a run is given
    (`worker/site_context.py`) and the site a rejected run is recorded on
    (`record_rejected`) all come from this one answer.
    """
    epoch = epoch_of_capture(db, capture.id)
    if epoch is not None:
        return epoch.site_id
    if capture.site_id is not None:
        return capture.site_id
    named = requested_site(db, capture)
    if named is not None:
        return named.id
    point = located_or_metadata(capture, located)
    if point is None:
        return None
    site = nearest_framed_site(db, *point, phone_made_only=is_phone_capture(capture))
    return site.id if site is not None else None


def located_or_metadata(
    capture: Capture, located: tuple[float, float] | None
) -> tuple[float, float] | None:
    """`located` (`worker/params.located_point`), else the capture's own `metadata.lat` /
    `lon`, when it is a point on the globe; else None."""
    if located is None:
        metadata = capture.metadata_ or {}
        lat, lon = _number(metadata.get("lat")), _number(metadata.get("lon"))
    else:
        lat, lon = _number(located[0]), _number(located[1])
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


# ── Which earlier epoch a new one is aligned to (M3b) ─────────────────────────────────────

#: The reference epoch's artifacts a run is seeded with (`worker/site_context.py`), by the
#: name its recipe gave them: normalize's kept frames, pose's model (`cameras.bin`,
#: `images.bin`, `points3D.bin`, `poses.json`), place's similarity from that model to its
#: east/north/up (`placement.json`) and its georeference. Only a Lane 2 run has all four.
REFERENCE_ARTIFACTS: dict[str, str] = {
    "frames": "frames",
    "poses": "poses",
    "placement": "placement.json",
    "georef": "georef.json",
}

#: Verdicts best first; an epoch with no alignment recorded (an M2-era epoch) ranks last.
VERDICT_RANK: dict[str | None, int] = {"survey": 0, "coarse": 1, "unaligned": 2, None: 3}

#: A run in these states may be writing its artifacts' keys right now (`jobs.ACTIVE_STATUSES`).
RUNS_IN_FLIGHT = (RunStatus.NOT_STARTED, RunStatus.IN_PROGRESS)

#: The uncertainty a reference is taken at when nothing recorded one: the figure a capture
#: placed by its GPS point or by hand is given (`manual_placement`'s ten metres).
UNRECORDED_UNCERTAINTY_M = 10.0


@dataclass(frozen=True)
class ReferenceChoice:
    """The earlier epoch a run is aligned to, as plain values read in one transaction.

    `keys` maps each of `REFERENCE_ARTIFACTS` to its storage key: a prefix for the two
    directories (`frames`, `poses`), a key for the two files. `revision` names the
    artifact rows those keys were read from: a retry that rewrites the keys changes it.
    """

    epoch_id: uuid.UUID
    job_id: uuid.UUID
    observed_at: datetime
    created_at: datetime
    platform: str
    verdict: str | None
    uncertainty_m: float
    keys: dict[str, str]
    revision: str = ""
    #: The height datum its origin is on, as its own alignment recorded it (`origin.
    #: heightDatum`), else None: `align_epoch` never reframes across two datums.
    height_datum: str | None = None


def claimed_verdict(alignment: object, platform: str) -> str | None:
    """The verdict an epoch's alignment claims, as far as it may: `survey` is only ever
    an RTK drone's (D4, D6 amendment), so anything else that says so reads as `coarse`;
    a verdict outside the vocabulary, or no alignment, is None."""
    verdict = alignment.get("verdict") if isinstance(alignment, dict) else None
    if verdict not in ("survey", "coarse", "unaligned"):
        return None
    if verdict == "survey" and platform != EpochPlatform.RTK_DRONE.value:
        return "coarse"
    return str(verdict)


def _reference_uncertainty(epoch: Epoch, capture: Capture | None) -> float:
    """What the reference epoch is known to: its alignment's `uncertaintyM`, else what its
    capture was registered with, else `UNRECORDED_UNCERTAINTY_M`. Always positive."""
    alignment = epoch.alignment if isinstance(epoch.alignment, dict) else {}
    for value in (alignment.get("uncertaintyM"), capture.uncertainty_m if capture else None):
        number = _number(value)
        if number is not None and number > 0:
            return number
    return UNRECORDED_UNCERTAINTY_M


def _reference_datum(epoch: Epoch) -> str | None:
    """The height datum the reference's origin is on, from its alignment's `origin`, when
    that names one of the site frames' datums; else None (never a guess)."""
    alignment = epoch.alignment if isinstance(epoch.alignment, dict) else {}
    origin = alignment.get("origin")
    datum = origin.get("heightDatum") if isinstance(origin, dict) else None
    known = {member.value for member in HeightDatum}
    return datum if isinstance(datum, str) and datum in known else None


def _confirmed_at(epoch: Epoch, shown_at: datetime) -> datetime:
    """When the epoch's run last published the tiles it shows: the shown asset's creation,
    or later, a successful re-run by the same run that republished the very same
    content-hashed tiles -- which keeps that asset and its timestamp, while its artifact
    rows are new (the R2 review of #52). `register` records each publication in the
    epoch's history (`tilesPublished`, `replacedAt`); a run that failed records none."""
    confirmed = shown_at
    for entry in epoch.history or []:
        if not isinstance(entry, dict) or entry.get("tilesPublished") is not True:
            continue
        if entry.get("replacedByJobId") != str(epoch.job_id):
            continue
        try:
            at = datetime.fromisoformat(str(entry.get("replacedAt")))
        except ValueError:
            continue
        confirmed = max(confirmed, at)
    return confirmed


def _reference_keys(db: Session, job_id: uuid.UUID) -> tuple[dict[str, str], datetime | None, str]:
    """Each reference artifact's storage key for the run, from its newest row, when the
    newest of those rows was written, and the revision they make; only the names it has.
    A directory artifact's key is its prefix (`worker/outputs`). A retried stage writes
    the same key again, so the newest row is the one whose bytes the bucket holds -- and
    a new row, so the revision (a digest of the rows chosen) changes with every rewrite
    while the keys, the epoch and the job do not."""
    rows = db.execute(
        select(Artifact.storage_key, Artifact.created_at, Artifact.id)
        .join(JobStep, Artifact.job_step_id == JobStep.id)
        .where(JobStep.job_id == job_id)
        .order_by(Artifact.created_at.desc(), Artifact.id.desc())
    ).all()
    keys: dict[str, str] = {}
    ids: dict[str, str] = {}
    written: datetime | None = None
    for key, created, artifact_id in rows:
        name = str(key).rsplit("/", 1)[-1]
        for role, wanted in REFERENCE_ARTIFACTS.items():
            if name == wanted and role not in keys:
                keys[role] = str(key)
                ids[role] = str(artifact_id)
                written = created if written is None else max(written, created)
    listed = ";".join(f"{role}={ids[role]}" for role in sorted(ids))
    revision = hashlib.sha256(listed.encode("utf-8")).hexdigest()[:16]
    return keys, written, revision


def reference_epoch(
    db: Session, site_id: uuid.UUID, *, capture_id: uuid.UUID | None = None
) -> ReferenceChoice | None:
    """The earlier epoch of the site a new run of `capture_id` is aligned to, or None.

    Candidates: the site's `published` epochs whose producing run kept every one of
    `REFERENCE_ARTIFACTS` (a Lane 2 run), other than the capture's own, when those files
    are the ones the epoch shows. Not by the run row's current status: a phone Refine is a
    retry of the same row (`jobs.retry_job`), and a Refine that failed leaves the epoch
    showing its old revision, whose files are still in the bucket -- it stays the scaffold.
    Two guards instead. A run in flight (queued or running) is skipped: it writes the same
    keys, and must not be read mid-write. And every reference artifact's newest row must be
    no newer than when the epoch's current revision was last published (`_confirmed_at`: a
    run that republished the very same tiles confirms its new rows): a Refine that
    re-uploaded its placement and then failed left a placement the shown tiles were not
    made from. For
    a re-run of an epoch, only epochs that joined the site before it (a lower ordinal):
    the epoch that made the frame is never aligned onto a later one that was aligned to
    it, so errors do not go round in a circle.

    Best verdict first (`VERDICT_RANK`, after `claimed_verdict`), then oldest observed,
    then first recorded: every epoch aligns to one fixed scaffold, and errors do not chain
    epoch to epoch (the design's open question 2, its default). An RTK epoch is still not
    moved onto a reference coarser than itself -- the pipeline keeps its own RTK frame and
    records the visual fit as a check (`tools/pipeline/alignment.py`), which reads the
    verdict passed here.
    """
    own = epoch_of_capture(db, capture_id) if capture_id is not None else None
    statement = (
        select(Epoch)
        .join(Job, Job.id == Epoch.job_id)
        .where(
            Epoch.site_id == site_id,
            Epoch.status == EpochStatus.PUBLISHED.value,
            Job.status.not_in(RUNS_IN_FLIGHT),
        )
    )
    if capture_id is not None:
        statement = statement.where((Epoch.capture_id.is_(None)) | (Epoch.capture_id != capture_id))
    if own is not None:
        statement = statement.where(Epoch.ordinal < own.ordinal)
    candidates: list[ReferenceChoice] = []
    for epoch in db.scalars(statement):
        assert epoch.job_id is not None
        keys, written, revision = _reference_keys(db, epoch.job_id)
        if set(keys) != set(REFERENCE_ARTIFACTS):
            continue
        shown = current_asset(db, epoch.id)
        if shown is None or written is None or written > _confirmed_at(epoch, shown.created_at):
            log.info(
                "reference: epoch %s is not a reference: %s",
                epoch.id,
                "it shows no tiles"
                if shown is None
                else "its run's files are newer than the tiles it shows (a Refine that failed)",
            )
            continue
        capture = db.get(Capture, epoch.capture_id) if epoch.capture_id is not None else None
        candidates.append(
            ReferenceChoice(
                epoch_id=epoch.id,
                job_id=epoch.job_id,
                observed_at=epoch.observed_at,
                created_at=epoch.created_at,
                platform=epoch.platform,
                verdict=claimed_verdict(epoch.alignment, epoch.platform),
                uncertainty_m=_reference_uncertainty(epoch, capture),
                keys=keys,
                revision=revision,
                height_datum=_reference_datum(epoch),
            )
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (VERDICT_RANK[c.verdict], c.observed_at, c.created_at, str(c.epoch_id)),
    )


# ── A capture a quality gate rejected (M3b) ───────────────────────────────────────────────

#: How far about a rejected capture's located point its epoch's footprint reaches: the
#: same 25 m a capture may lie outside a site's boundary and still join it (D3).
REJECTED_FOOTPRINT_M = SPATIAL_JOIN_BUFFER_M


@dataclass(frozen=True)
class Rejection:
    """A run a quality gate the recipe turned on failed: which stage, which gate
    (`{gate, measured, limit, unit, reason}`, `errors.parse_quality_gate`; empty when its
    text could not be read back), and the run's located point, if it had one."""

    stage_id: str
    gate: dict[str, object]
    located: tuple[float, float] | None = None


def _rejected_alignment(platform: str, job_id: uuid.UUID, rejection: Rejection) -> dict[str, Any]:
    """What a rejected epoch's `alignment` says, in the shape the web reads (`label`,
    `verdict`, `uncertaintyM`): unaligned, with the gate as its reason. The API's
    `CaptureQuality` has no field for a gate, and `quality` stays null rather than holding
    something that would not parse (the M3b review's fix 8)."""
    from app.worker.pipeline_bridge import PLATFORM_WORDS

    gate = rejection.gate
    name = str(gate.get("gate") or "unknown")
    unit = f" {gate['unit']}" if gate.get("unit") else ""
    measured = (
        f"measured {gate['measured']:.4g}{unit} against a limit of {gate['limit']:.4g}{unit}"
        if isinstance(gate.get("measured"), float) and isinstance(gate.get("limit"), float)
        else "its measurement could not be read back"
    )
    why = f": {gate['reason']}" if gate.get("reason") else ""
    words = PLATFORM_WORDS.get(platform, "Capture")
    return {
        "version": 1,
        "method": "none",
        "referenceEpochId": None,
        "platform": platform,
        "verdict": "unaligned",
        "confidence": 0.0,
        "uncertaintyM": None,
        "label": f"{words}, rejected: quality gate {name} failed",
        "reasons": [f"stage {rejection.stage_id!r} failed quality gate {name}, {measured}{why}"],
        "rejected": {"stage": rejection.stage_id, "jobId": str(job_id), **gate},
        "frameSource": None,
    }


def record_rejected(
    db: Session, job_id: uuid.UUID, capture: Capture, rejection: Rejection
) -> Epoch | None:
    """Keep a capture a quality gate rejected as a `rejected` epoch of its site (D3, D4).

    Its site is `site_for_run`'s -- the one its run was placed in -- and only a site with
    a frame: a frameless site's first epoch is the one that defines its frame, and a
    rejected run defines nothing. No site, no epoch (none is made for it); the job's error
    still says which gate failed. No asset. Its footprint is the run's located point
    buffered by `REJECTED_FOOTPRINT_M`, else the site's boundary (the epochs footprint is
    never null). Dated by the capture row, or its upload flagged `created`.

    The capture already an epoch: a rejected one takes this run's rejection; a published
    one is left as it is -- its current revision passed, and the gate failed a later run of
    it, which the job's error records. A later run that passes publishes a rejected epoch
    (`worker/registration.register`). Not committed here: the caller's transaction.
    """
    existing = epoch_of_capture(db, capture.id, lock=True)
    platform = existing.platform if existing is not None else platform_of(capture).value
    alignment = _rejected_alignment(platform, job_id, rejection)
    if existing is not None:
        if existing.status == EpochStatus.REJECTED.value:
            # The rejection it replaces stays in the epoch's history (as `register`'s
            # `_replace_run` keeps a re-run's): each failed gate is evidence.
            kept = geometry.wkb_to_footprint(existing.footprint)
            entry = {
                "jobId": str(existing.job_id) if existing.job_id is not None else None,
                "footprint": kept.model_dump(mode="json") if kept is not None else None,
                "quality": existing.quality,
                "alignment": existing.alignment,
                "replacedAt": datetime.now(UTC).isoformat(),
                "replacedByJobId": str(job_id),
                "tilesPublished": False,
            }
            existing.history = [*(existing.history or []), entry]
            existing.alignment = alignment
            existing.job_id = job_id if db.get(Job, job_id) is not None else None
        return existing if existing.status == EpochStatus.REJECTED.value else None
    site_id = site_for_run(db, capture, rejection.located)
    site = db.get(Site, site_id, with_for_update=True) if site_id is not None else None
    if site is None or db.get(SiteFrame, site.id) is None:
        return None
    point = located_or_metadata(capture, rejection.located)
    footprint = (
        db.scalar(
            select(
                func.ST_Multi(
                    func.geometry(
                        func.ST_Buffer(func.geography(_point(*point)), REJECTED_FOOTPRINT_M)
                    )
                )
            )
        )
        if point is not None
        else site.boundary
    )
    named = requested_site(db, capture)
    explicit = capture.site_id == site.id or (named is not None and named.id == site.id)
    ordinal = (db.scalar(select(func.max(Epoch.ordinal)).where(Epoch.site_id == site.id)) or 0) + 1
    observed = capture.captured_at or capture.created_at
    epoch = Epoch(
        site_id=site.id,
        capture_id=capture.id,
        job_id=job_id if db.get(Job, job_id) is not None else None,
        ordinal=ordinal,
        observed_at=observed,
        observed_at_source=(
            ObservedAtSource.CAPTURED_AT if capture.captured_at else ObservedAtSource.CREATED
        ).value,
        footprint=footprint,
        platform=platform,
        status=EpochStatus.REJECTED.value,
        join_method=(EpochJoinMethod.EXPLICIT if explicit else EpochJoinMethod.SPATIAL).value,
        alignment=alignment,
        quality=None,
    )
    db.add(epoch)
    db.flush()
    return epoch
