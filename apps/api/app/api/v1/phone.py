"""Capture from a phone with the phone key alone. See app/services/phone_key.py."""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Final, Literal

from fastapi import APIRouter, Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import Field
from sqlalchemy import func, select

from app.api.deps import DbSession, SettingsDep, Storage
from app.models.capture import Capture
from app.models.enums import CaptureKind
from app.models.job import Job
from app.schemas.base import CamelModel
from app.schemas.capture import QUALITY_BARS, QUALITY_MODES, CaptureCreate, CaptureRead
from app.schemas.job import JobCreate, JobRead
from app.services import captures as capture_service
from app.services import handoff, phone_key
from app.services import jobs as job_service
from app.services import recipes as recipe_service
from app.services.errors import ConflictError, UnauthorizedError
from app.storage import ObjectStorage
from app.worker.outputs import artifact_key

router = APIRouter(prefix="/phone", tags=["phone"])

#: What a phone capture is marked with, and how the key's routes recognise their own.
ORIGIN = "phone-key"
#: The only recipes a phone can start: the two lanes a capture can go down.
PHONE_RECIPES = frozenset({"splat-ingest", "photo-reconstruct"})

#: A rule that is not a range or a set of values: `FLAG` is a boolean, `ROI` is a region
#: of interest, `{"center": [x, y, z], "radius": r}` with every number finite and r > 0.
FLAG: Final = "flag"
ROI: Final = "roi"


@dataclass(frozen=True)
class RangeOr:
    """A number in [low, high], or one of a few words: `max_side` is 800..4000 or `auto`."""

    low: float
    high: float
    words: frozenset[str]


Rule = tuple[float, float] | frozenset[str] | RangeOr | Literal["flag", "roi"]

#: The options a phone may set, per recipe and stage: each parameter's allowed range, or
#: its allowed values. Anything else is refused by name rather than passed through, so
#: the key cannot reach a stage parameter that was never meant to be a phone's choice
#: (a trainer path, a Python interpreter, a GPU tier).
#:
#: Nothing for `package`. It was `max_gaussians` -- how many of a scan's gaussians the map
#: and the viewer were sent, the rest discarded -- and it is gone because nothing is
#: discarded any more: every scan is packed whole as a level-of-detail tileset
#: (tools/captures/splat_tiles.py), and the phone's "Detail" is how much *that phone* draws,
#: kept on the phone (apps/web src/lib/detail.ts). A stale page still sending it is refused
#: by name, like any other option a phone may not set.
PHONE_OPTIONS: dict[str, dict[str, dict[str, Rule]]] = {
    "photo-reconstruct": {
        # The long side frames are kept at: a number, or `auto` (the recipe's default:
        # 1600 unless the capture measurably holds detail above it); the most candidate
        # frames a second taken from a video (keyframes are then chosen by camera
        # motion); and how many photos are kept (the recipe keeps 100; pose matches a
        # photo set exhaustively, quadratic in it). `select` is for comparing the motion
        # rule with the time-windowed one it replaced (experiments/run_variants.py).
        "normalize": {
            "max_side": RangeOr(800, 4000, frozenset({"auto"})),
            "fps": (1, 30),
            "keep": (20, 200),
            "select": frozenset({"viewpoint", "sharpness-windowed"}),
        },
        # The training schedule (`training.schedule_scale`) and the gaussian cap; then the
        # preview's and Refine's knobs: a forced schedule scale, the training image size,
        # the region to train inside (COLMAP frame, from a preview's quality stage), and
        # gsplat's quality switches -- the last three being the phone-capture ones: pose
        # refinement, per-image appearance, per-image colour (bilateral grid).
        #
        # The recipe sizes the cap to the capture (`cap_max: auto`, tools/pipeline/
        # gaussian_budget.py); a phone's quality tier scales that measured budget by
        # `density_scale` (Quick 0.5, Best 2) rather than naming a count, and the
        # pipeline's own floor and ceilings still apply. An explicit `cap_max` remains an
        # override: the preview's fixed 200k is one.
        "train": {
            "schedule_full_at": (0, 400),
            "schedule_floor": (0.1, 1.0),
            "cap_max": (100_000, 1_500_000),
            "density_scale": (0.25, 4.0),
            "schedule_scale": (0.05, 1.0),
            "train_max_side": (400, 4000),
            "roi": ROI,
            "antialiased": FLAG,
            "depth_loss": FLAG,
            "opacity_reg": (0.0, 0.05),
            "pose_opt": FLAG,
            "app_opt": FLAG,
            "bilateral_grid": FLAG,
            # Where a Refine starts: the preview's splat (the default a Refine sets) or
            # COLMAP's points, and the schedule it runs when it starts from the preview.
            "init_from": frozenset({"sfm", "preview"}),
            "init_schedule_scale": (0.05, 1.0),
            # Block training (tools/pipeline/blocks.py): `auto` trains in blocks only when
            # the budget is more than one GPU holds; a number forces that many, to compare
            # a capture in blocks with the same capture whole (the spool at 2 against 1).
            # The camera test's 1 - SSIM threshold and the frozen ring are the knobs that
            # comparison calibrates; exposure is the existing `bilateral_grid` above.
            "blocks": RangeOr(1, 16, frozenset({"auto"})),
            "block_epsilon": (0.0, 0.5),
            "block_ring": FLAG,
        },
        # The quality bar: what is kept, and whether this run is a preview or a refine.
        "quality": {"bar": frozenset(QUALITY_BARS), "mode": frozenset(QUALITY_MODES)},
    },
    "splat-ingest": {
        "normalize": {
            "up_axis": frozenset({"", "z", "-z", "y", "-y", "x", "-x"}),
            "heading_deg": (-360, 360),
        },
    },
}


def _checked_options(recipe: str, params: dict[str, object]) -> dict[str, dict[str, object]]:
    allowed = PHONE_OPTIONS.get(recipe, {})
    checked: dict[str, dict[str, object]] = {}
    for stage, values in params.items():
        stage_allowed = allowed.get(stage)
        if stage_allowed is None or not isinstance(values, dict):
            raise ConflictError(f"A phone cannot set options on {stage!r} for {recipe}.")
        for name, value in values.items():
            rule = stage_allowed.get(name)
            if rule is None:
                raise ConflictError(f"A phone cannot set {stage}.{name}.")
            if rule == FLAG:
                if not isinstance(value, bool):
                    raise ConflictError(f"{stage}.{name} must be true or false.")
            elif rule == ROI:
                value = _checked_roi(stage, value)
            elif isinstance(rule, RangeOr):
                if isinstance(value, str):
                    if value not in rule.words:
                        raise ConflictError(
                            f"{stage}.{name} must be a number from {rule.low:g} to "
                            f"{rule.high:g}, or one of {sorted(rule.words)}."
                        )
                elif (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not rule.low <= value <= rule.high
                ):
                    raise ConflictError(
                        f"{stage}.{name} must be a number from {rule.low:g} to {rule.high:g}, "
                        f"or one of {sorted(rule.words)}."
                    )
            elif isinstance(rule, frozenset):
                if value not in rule:
                    raise ConflictError(f"{stage}.{name} must be one of {sorted(rule)}.")
            elif isinstance(rule, tuple) and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not rule[0] <= value <= rule[1]
            ):
                raise ConflictError(f"{stage}.{name} must be a number from {rule[0]} to {rule[1]}.")
            checked.setdefault(stage, {})[name] = value
    return checked


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _checked_roi(stage: str, value: object) -> dict[str, object]:
    """`{"center": [x, y, z], "radius": r}`, exactly, or a refusal that shows the shape."""
    shape = f'{stage}.roi must be {{"center": [x, y, z], "radius": r}} with r > 0.'
    if not isinstance(value, dict) or set(value) != {"center", "radius"}:
        raise ConflictError(shape)
    center, radius = value["center"], value["radius"]
    if not isinstance(center, list) or len(center) != 3 or not all(_finite(v) for v in center):
        raise ConflictError(shape)
    if not _finite(radius) or float(radius) <= 0:
        raise ConflictError(shape)
    return {"center": [float(v) for v in center], "radius": float(radius)}


phone_key_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="phoneKey",
    description="The phone key, as `Authorization: Bearer <key>`. Opens only /phone routes.",
)


def require_phone_key(
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(phone_key_scheme)],
) -> None:
    phone_key.check(settings, credentials.credentials if credentials is not None else "")


RequirePhoneKey = Depends(require_phone_key)


class PhoneCaptureCreate(CamelModel):
    """Where the phone was, if it said. Both or neither."""

    name: str | None = Field(default=None, max_length=200)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    accuracy_m: float | None = Field(default=None, ge=0)


class PhoneCapture(CamelModel):
    capture: CaptureRead
    #: A handoff token for this capture's upload routes, renewed on every response.
    upload_token: str


@router.post("/check", status_code=status.HTTP_204_NO_CONTENT, dependencies=[RequirePhoneKey])
def check_key() -> None:
    """204 when the key is right, so the page can say so before anything is picked."""


@router.post(
    "/captures",
    response_model=PhoneCapture,
    status_code=status.HTTP_201_CREATED,
    dependencies=[RequirePhoneKey],
    summary="Start a capture from a phone",
)
def create_phone_capture(
    payload: PhoneCaptureCreate, db: DbSession, settings: SettingsDep
) -> PhoneCapture:
    since = datetime.now(tz=UTC) - timedelta(hours=24)
    recent = db.scalar(
        select(func.count(Capture.id)).where(
            Capture.metadata_["origin"].as_string() == ORIGIN, Capture.created_at >= since
        )
    )
    if (recent or 0) >= settings.api_phone_daily_captures:
        raise ConflictError(
            f"The phone key has started {settings.api_phone_daily_captures} captures in "
            "the last 24 hours, which is its limit. Try again later."
        )
    metadata: dict[str, object] = {"origin": ORIGIN}
    if payload.lat is not None and payload.lon is not None:
        # The phone's own fix, which is a far better placement guess than a desktop
        # camera's: it is where the capture was actually made.
        metadata.update(lat=round(payload.lat, 6), lon=round(payload.lon, 6))
        if payload.accuracy_m is not None:
            metadata["locationAccuracyM"] = round(payload.accuracy_m, 1)
    name = payload.name or f"Phone capture {datetime.now(tz=UTC):%Y-%m-%d %H:%M} UTC"
    capture = capture_service.create_capture(
        db, CaptureCreate(name=name, kind=CaptureKind.VIDEO, metadata=metadata)
    )
    token = handoff.issue(handoff.key_for(settings), capture.id, now=int(time.time()))
    return PhoneCapture(capture=capture_service.capture_to_read(capture), upload_token=token)


@router.post(
    "/captures/{capture_id}/process",
    response_model=JobRead,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[RequirePhoneKey],
    summary="Queue a run over a capture this phone key started",
)
def process_phone_capture(capture_id: uuid.UUID, payload: JobCreate, db: DbSession) -> JobRead:
    capture = capture_service.get_capture(db, capture_id)
    if (capture.metadata_ or {}).get("origin") != ORIGIN:
        # Same answer as a wrong key: the phone key does not reach other captures.
        raise UnauthorizedError("That phone key is not right.")
    if payload.recipe not in PHONE_RECIPES:
        raise ConflictError(f"A phone can start {', '.join(sorted(PHONE_RECIPES))}, not that.")
    params = _checked_options(payload.recipe, payload.params)
    return job_service.job_to_read(
        job_service.create_job(db, capture_id, JobCreate(recipe=payload.recipe, params=params))
    )


#: Where a Refine resumes a finished run: training, keeping the frames and poses.
REFINE_FROM = "train"


class PhoneRefine(CamelModel):
    """The phone's options for the full-quality pass (the same whitelist as `process`).

    `normalize` options are accepted and ignored: a Refine keeps the preview's frames and
    poses, which is the whole point -- the region of interest is only meaningful in them.
    """

    params: dict[str, Any] = Field(default_factory=dict)


@router.post(
    "/captures/{capture_id}/refine",
    response_model=JobRead,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[RequirePhoneKey],
    summary="Refine a finished preview: train it again at full quality, inside its region",
    description=(
        "Resumes the capture's latest finished photo-reconstruct run at `train` with new "
        "parameters: the phone's quality options, `train.support_mask` set to the voxels the "
        "preview's well-supported splats occupy (any shape), `train.init_from` = `preview` "
        "(start from the preview's splat on a shorter schedule; the phone may send `sfm`), "
        "and `quality.mode` = `refine`. "
        "The frames and poses are kept, so the mask is in the frame it was measured in. If the "
        "worker no longer has them, the run starts over and trains uncropped rather than "
        "applying the region to a different reconstruction."
    ),
)
def refine_phone_capture(
    capture_id: uuid.UUID, payload: PhoneRefine, db: DbSession, storage: Storage
) -> JobRead:
    capture = capture_service.get_capture(db, capture_id)
    if (capture.metadata_ or {}).get("origin") != ORIGIN:
        raise UnauthorizedError("That phone key is not right.")
    job = db.scalars(
        select(Job).where(Job.capture_id == capture_id).order_by(Job.created_at.desc())
    ).first()
    if job is None:
        raise ConflictError("Nothing to refine yet: process it first.")
    if job.status in job_service.ACTIVE_STATUSES:
        raise ConflictError("It is still running. Refine it when it has finished.")
    if job.recipe != "photo-reconstruct":
        raise ConflictError("Only a photo or video capture can be refined.")
    # The verdict is what makes a run refinable: it is written when a run *finishes*, so
    # one that belongs to this job means its frames and poses were made -- including when
    # the job has since failed or been stopped as a Refine, which is then refined again.
    verdict = capture_service.quality_of(capture)
    if verdict is None or verdict.job_id != job.id:
        raise ConflictError("That run has no finished quality check to refine from. Try again.")
    requested = _checked_options(job.recipe, payload.params)
    params: dict[str, dict[str, object]] = {
        stage: dict(values) for stage, values in requested.items() if stage != "normalize"
    }
    previous = job.params if isinstance(job.params, dict) else {}
    if isinstance(previous.get("normalize"), dict):
        # What the kept frames were made with, recorded rather than silently changed.
        params["normalize"] = dict(previous["normalize"])
    train = params.setdefault("train", {})
    # The region to train in is the preview's support mask: the voxels its well-supported
    # splats occupy, in whatever shape they make. It is too large for the capture's summary,
    # so it is read from the preview's own quality.json. The sphere is only a fallback for
    # a verdict written before masks existed.
    mask = _support_mask(storage, job.id)
    if mask is not None:
        train["support_mask"] = mask
    elif verdict.roi is not None:
        train["roi"] = {"center": list(verdict.roi.center), "radius": verdict.roi.radius}
    # Start from the preview's own splat on a shorter schedule rather than from COLMAP's
    # sparse points on the whole one (tools/pipeline/init_seed.py). The train stage checks
    # that the seed is in these poses, and trains from scratch, saying so, if it is not.
    train.setdefault("init_from", "preview")
    quality = params.setdefault("quality", {})
    quality["mode"] = "refine"
    quality.setdefault("bar", "strict")
    recipe_service.check_overrides(job.recipe, params)
    job.params = params
    try:
        refined = job_service.retry_job(db, job.id, from_stage=REFINE_FROM)
    except ValueError as error:
        db.rollback()
        raise ConflictError(f"That run cannot be refined: {error}") from error
    return job_service.job_to_read(refined)


@router.post(
    "/captures/{capture_id}/stop",
    response_model=JobRead,
    dependencies=[RequirePhoneKey],
    summary="Stop the run in progress over a capture this phone key started",
)
def stop_phone_capture(capture_id: uuid.UUID, db: DbSession) -> JobRead:
    """The phone's Stop button. The same cancel as `POST /jobs/{id}/cancel`, reached with
    the phone key and only for this phone's own captures, so a run that is taking far
    too long can be stopped from the phone that started it."""
    capture = capture_service.get_capture(db, capture_id)
    if (capture.metadata_ or {}).get("origin") != ORIGIN:
        raise UnauthorizedError("That phone key is not right.")
    active = db.scalars(
        select(Job)
        .where(Job.capture_id == capture_id, Job.status.in_(job_service.ACTIVE_STATUSES))
        .order_by(Job.created_at.desc())
    ).first()
    if active is None:
        raise ConflictError("Nothing is running for that capture.")
    return job_service.job_to_read(job_service.cancel_job(db, active.id))


def _support_mask(storage: ObjectStorage, job_id: uuid.UUID) -> dict[str, object] | None:
    """The support mask in a run's quality.json, or None if it has none or can't be read.

    None is safe: the Refine then crops to the sphere, or not at all, and still applies
    the quality bar to what it trains.
    """
    try:
        document = json.loads(storage.get_object(artifact_key(job_id, "quality", "quality.json")))
    except Exception:
        return None
    mask = document.get("supportMask") if isinstance(document, dict) else None
    return mask if isinstance(mask, dict) else None
