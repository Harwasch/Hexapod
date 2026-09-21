"""Pipeline runs over a capture.

This service queues work and reads it back. It never executes a stage: a job is
inserted `not-started` and A7's worker claims it with a committed lease (see the
comment on app.models.job.Job).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Select, select
from sqlalchemy.orm import Session, selectinload

from app.models import Job, JobStep
from app.models.enums import RunStatus, UploadStatus
from app.schemas.job import JobCreate, JobRead
from app.services.captures import get_capture
from app.services.errors import ConflictError, NotFoundError

#: Recipes this API will queue, and the version it stamps on a run.
#:
#: A6 builds the real registry in `tools/pipeline`, with each recipe's stages and its
#: own version; this map is the API's half of that contract until it exists, and is
#: what makes `recipeVersion` a resolved fact rather than something a caller can claim.
#: A6 replaces the literals here with a read of the registry -- the endpoint's shape
#: does not change.
RECIPE_VERSIONS: dict[str, str] = {
    "splat-ingest": "0.1.0",
    "photo-reconstruct": "0.1.0",
}

#: A run that has not finished. A capture may be run many times -- comparing two runs
#: is the point of the console -- but not twice at once.
ACTIVE_STATUSES = (RunStatus.NOT_STARTED, RunStatus.IN_PROGRESS)


def create_job(db: Session, capture_id: uuid.UUID, payload: JobCreate) -> Job:
    """Queue a run. Nothing is executed here, and no step rows are written: the recipe
    decides the steps, and the worker that resolves the recipe writes them."""
    capture = get_capture(db, capture_id)
    version = RECIPE_VERSIONS.get(payload.recipe)
    if version is None:
        known = ", ".join(sorted(RECIPE_VERSIONS))
        raise ValueError(f"unknown recipe '{payload.recipe}'; known recipes are {known}")
    if not any(file.status is UploadStatus.COMPLETE for file in capture.files):
        raise ConflictError(
            f"capture {capture.id} has no uploaded files; finish an upload before processing"
        )
    running = db.scalar(
        select(Job.id).where(Job.capture_id == capture.id, Job.status.in_(ACTIVE_STATUSES))
    )
    if running is not None:
        raise ConflictError(f"capture {capture.id} already has job {running} queued or running")
    job = Job(
        capture_id=capture.id,
        recipe=payload.recipe,
        recipe_version=version,
        params=payload.params,
        status=RunStatus.NOT_STARTED,
        provider=payload.provider,
        tier=payload.tier,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _newest_first() -> Select[tuple[Job]]:
    # The steps and their artifacts are what a caller reads a job for, so they are
    # loaded in two more queries rather than one per row.
    return (
        select(Job)
        .options(selectinload(Job.steps).selectinload(JobStep.artifacts))
        .order_by(Job.created_at.desc(), Job.id.desc())
    )


def list_jobs(
    db: Session,
    *,
    capture_id: uuid.UUID | None = None,
    status: RunStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Job]:
    stmt = _newest_first().limit(limit).offset(offset)
    if capture_id is not None:
        stmt = stmt.where(Job.capture_id == capture_id)
    if status is not None:
        stmt = stmt.where(Job.status == status)
    return list(db.scalars(stmt).all())


def jobs_for_capture(db: Session, capture_id: uuid.UUID) -> list[Job]:
    """Every run over one capture — unpaged, because a capture's own history is short
    and the console compares runs against each other."""
    return list(db.scalars(_newest_first().where(Job.capture_id == capture_id)).all())


def get_job(db: Session, job_id: uuid.UUID) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise NotFoundError("job", job_id)
    return job


def cancel_job(db: Session, job_id: uuid.UUID) -> Job:
    """Mark a run cancelled. A worker holding it sees the status and stops; nothing is
    killed from here, because the process doing the work is not this one."""
    job = get_job(db, job_id)
    if job.status not in ACTIVE_STATUSES:
        raise ConflictError(f"job {job.id} is {job.status.value} and cannot be cancelled")
    now = datetime.now(tz=UTC)
    job.status = RunStatus.CANCELLED
    job.finished_at = now
    if job.claimed_at is not None:
        # How long it actually ran, not how long it sat in the queue.
        job.duration_s = (now - job.claimed_at).total_seconds()
    for step in job.steps:
        if step.status in ACTIVE_STATUSES:
            step.status = RunStatus.CANCELLED
            step.finished_at = now
    db.commit()
    db.refresh(job)
    return job


def job_to_read(job: Job) -> JobRead:
    return JobRead.model_validate(job)
