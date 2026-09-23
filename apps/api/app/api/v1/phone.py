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
    return job_service.job_to_read(
        job_service.create_job(db, capture_id, JobCreate(recipe=payload.recipe))
    )
