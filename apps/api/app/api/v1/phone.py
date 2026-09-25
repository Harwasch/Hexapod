"""Capture from a phone with the phone key alone. See app/services/phone_key.py."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import Field
from sqlalchemy import func, select

from app.api.deps import DbSession, SettingsDep
from app.models.capture import Capture
from app.models.enums import CaptureKind
from app.models.job import Job
from app.schemas.base import CamelModel
from app.schemas.capture import CaptureCreate, CaptureRead
from app.schemas.job import JobCreate, JobRead
from app.services import captures as capture_service
from app.services import handoff, phone_key
from app.services import jobs as job_service
from app.services.errors import ConflictError, UnauthorizedError

router = APIRouter(prefix="/phone", tags=["phone"])

#: What a phone capture is marked with, and how the key's routes recognise their own.
ORIGIN = "phone-key"
#: The only recipes a phone can start: the two lanes a capture can go down.
PHONE_RECIPES = frozenset({"splat-ingest", "photo-reconstruct"})

#: The options a phone may set, per recipe and stage: each parameter's allowed range, or
#: its allowed values. Anything else is refused by name rather than passed through, so
#: the key cannot reach a stage parameter that was never meant to be a phone's choice
#: (a trainer path, a Python interpreter, a GPU tier).
PHONE_OPTIONS: dict[str, dict[str, dict[str, tuple[float, float] | frozenset[str]]]] = {
    "photo-reconstruct": {
        # The long side frames are shrunk to before pose and training; frames per second
        # taken from a video.
        "normalize": {"max_side": (800, 4000), "fps": (1, 10)},
        # The training schedule (`training.schedule_scale`) and the gaussian cap.
        "train": {
            "schedule_full_at": (0, 400),
            "schedule_floor": (0.1, 1.0),
            "cap_max": (100_000, 1_500_000),
        },
        # How many gaussians the map and the viewer are sent.
        "package": {"max_gaussians": (100_000, 1_000_000)},
    },
    "splat-ingest": {
        "normalize": {
            "up_axis": frozenset({"", "z", "-z", "y", "-y", "x", "-x"}),
            "heading_deg": (-360, 360),
        },
        "package": {"max_gaussians": (100_000, 1_000_000)},
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
            if isinstance(rule, frozenset):
                if value not in rule:
                    raise ConflictError(f"{stage}.{name} must be one of {sorted(rule)}.")
            elif (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not rule[0] <= value <= rule[1]
            ):
                raise ConflictError(f"{stage}.{name} must be a number from {rule[0]} to {rule[1]}.")
            checked.setdefault(stage, {})[name] = value
    return checked


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
