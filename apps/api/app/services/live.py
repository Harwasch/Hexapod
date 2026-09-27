"""The live viewer's read: where a run has got to, in things a person can look at.

The worker's heartbeat copies the newest `live-cameras` and `live-splat` lines out of a
running stage's log onto its step (`metrics.live`, `app/worker/steps.report_live`). This
assembles them across the run's steps -- the pose stage's cameras stay on its finished
step, so training's intermediate splats are drawn among them -- and signs a short-lived
GET for the newest splat snapshot.

**Why a signed URL and not a proxy.** The snapshot sits in the *private* bucket under
`runs/<run>/<stage>/checkpoint/live/`, where the remote stage's checkpoint syncer puts
it. The viewer is served from the web origin (Cloudflare Pages), and the private bucket's
CORS rule (`infra/cors/upload.json`, applied for that same origin by `provision.yml`)
already allows `GET` from it -- the phone page's multipart upload from the same origin
depends on that rule. So the browser can fetch the signed URL directly, and a snapshot of
a megabyte or two never passes through the API. The URL is only handed out once the
object exists (a HEAD), because the syncer uploads on an interval and the log line that
names the snapshot usually arrives first.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Capture, Job, JobStep
from app.models.enums import RunStatus
from app.schemas.live import LiveCameras, LiveProgress, LiveSplat, LiveStage, LiveState
from app.services.errors import NotFoundError
from app.services.jobs import get_job
from app.storage import ObjectStorage
from app.storage.null import StorageUnavailableError

#: Long enough for a phone on a slow link to fetch a few megabytes; the viewer asks
#: again every few seconds anyway.
SIGNED_FOR_S = 10 * 60

#: The only keys this signs: the live snapshots the pipeline writes, nothing else in the
#: private bucket, whatever a log line claimed.
_LIVE_SEGMENT = "/checkpoint/live/"


def live_for_capture(db: Session, storage: ObjectStorage, capture_id: uuid.UUID) -> LiveState:
    """The live state of the capture's newest run."""
    capture = db.get(Capture, capture_id)
    if capture is None:
        raise NotFoundError("capture", capture_id)
    job = db.scalars(
        select(Job)
        .where(Job.capture_id == capture_id)
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(1)
    ).first()
    if job is None:
        raise NotFoundError("run for capture", capture_id)
    return _state(db, storage, job, capture)


def live_for_job(db: Session, storage: ObjectStorage, job_id: uuid.UUID) -> LiveState:
    job = get_job(db, job_id)
    capture = db.get(Capture, job.capture_id)
    if capture is None:  # pragma: no cover - a job's capture is a foreign key
        raise NotFoundError("capture", job.capture_id)
    return _state(db, storage, job, capture)


def _state(db: Session, storage: ObjectStorage, job: Job, capture: Capture) -> LiveState:
    steps = sorted(
        db.scalars(select(JobStep).where(JobStep.job_id == job.id)).all(),
        key=lambda step: step.ordinal,
    )
    # A step row exists from the moment its stage starts.
    running = [step for step in steps if step.status == RunStatus.IN_PROGRESS]
    current = running[-1] if running else (steps[-1] if steps else None)
    cameras: LiveCameras | None = None
    splat: LiveSplat | None = None
    # The newest of each, by position in the run: a re-run pose stage's cameras replace
    # the old ones, and a Refine's training replaces the preview's splat.
    for step in steps:
        live = (step.metrics or {}).get("live")
        if not isinstance(live, dict):
            continue
        found_cameras = _cameras(step, live.get("cameras"))
        cameras = found_cameras or cameras
        found_splat = _splat(step, live.get("splat"), storage)
        splat = found_splat or splat
    return LiveState(
        job_id=job.id,
        capture_id=capture.id,
        capture_name=capture.name,
        recipe=job.recipe,
        status=job.status,
        error=job.error,
        site_id=capture.site_id if job.status == RunStatus.COMPLETE else None,
        stage=None if current is None else _stage(current),
        steps_done=sum(1 for step in steps if step.status == RunStatus.COMPLETE),
        steps_started=len(steps),
        cameras=cameras,
        splat=splat,
    )


def _stage(step: JobStep) -> LiveStage:
    progress = (
        (step.metrics or {}).get("progress") if step.status == RunStatus.IN_PROGRESS else None
    )
    parsed: LiveProgress | None = None
    if isinstance(progress, dict):
        try:
            parsed = LiveProgress.model_validate(progress)
        except ValidationError:
            parsed = None
    return LiveStage(
        stage_id=step.stage_id,
        impl=step.impl,
        status=step.status,
        ordinal=step.ordinal,
        started_at=step.started_at,
        progress=parsed,
    )


def _cameras(step: JobStep, value: Any) -> LiveCameras | None:
    """The stored payload, as the schema, or None when it is not one (never an error)."""
    if not isinstance(value, dict):
        return None
    fields = {
        key: value.get(key)
        for key in (
            "seq",
            "final",
            "registered",
            "frames",
            "origin",
            "scale",
            "up",
            "aspect",
            "cameraCount",
            "cameras",
            "pointCount",
            "pointsTotal",
            "points",
        )
    }
    try:
        return LiveCameras.model_validate({**fields, "stageId": step.stage_id})
    except ValidationError:
        return None


def _splat(step: JobStep, value: Any, storage: ObjectStorage) -> LiveSplat | None:
    if not isinstance(value, dict):
        return None
    key = value.get("key")
    if (
        not isinstance(key, str)
        or not key.startswith("runs/")
        or _LIVE_SEGMENT not in key
        or ".." in key
    ):
        return None
    try:
        return LiveSplat.model_validate(
            {
                "stageId": step.stage_id,
                "step": value.get("step"),
                "total": value.get("total"),
                "count": value.get("count"),
                "of": value.get("of"),
                "bytes": value.get("bytes"),
                "up": value.get("up"),
                "name": key.rsplit("/", 1)[-1],
                "url": _signed(storage, key),
            }
        )
    except ValidationError:
        return None


def _signed(storage: ObjectStorage, key: str) -> str | None:
    """A signed GET once the object is there; None before the syncer has sent it."""
    try:
        if storage.head_object(key) is None:
            return None
        return storage.presign_get(key, expires_in=SIGNED_FOR_S)
    except StorageUnavailableError:
        return None
