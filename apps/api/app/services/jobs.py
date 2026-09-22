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
from app.schemas.job import JobCreate, JobRead, JobStepLog
from app.services.captures import get_capture
from app.services.errors import ConflictError, NotFoundError
from app.storage import ObjectStorage

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


def retry_job(db: Session, job_id: uuid.UUID, from_stage: str | None = None) -> Job:
    """Re-queue a finished run, resuming at one of its stages.

    This is the *human* retry, and it is deliberately not the same thing as the worker's
    automatic one. The worker retries a stage that failed or was interrupted, up to
    `worker_max_attempts`, and then dead-letters the job so a poison capture cannot be
    picked up forever. Asking for it again here is a new decision by a person, so the
    attempt budget of the stages being re-run is reset with it — otherwise "Retry" on a
    dead-lettered job would fail instantly and say nothing new.

    Every step before `from_stage` keeps its `complete` row and its artifacts; the worker
    skips those stages and resumes from this one, using the intermediate artifacts still
    in the run's workdir.
    """
    job = get_job(db, job_id)
    if job.status in ACTIVE_STATUSES:
        raise ConflictError(f"job {job.id} is {job.status.value}; cancel it before retrying")
    if not job.steps:
        raise ConflictError(f"job {job.id} has no steps to retry from; queue a new run instead")
    target = _retry_target(job, from_stage)
    for step in job.steps:
        if step.ordinal < target.ordinal:
            continue
        step.status = RunStatus.NOT_STARTED
        step.started_at = None
        step.finished_at = None
        step.preempted_at = None
        # Back to zero, not to `attempt`: the worker adds one when it starts the stage,
        # so this run begins at attempt 1 with the full budget again.
        step.attempt = 0
    job.status = RunStatus.NOT_STARTED
    job.error = None
    job.finished_at = None
    job.duration_s = None
    # Nobody holds it: the claim loop wants an unstarted row with no lease on it.
    job.claimed_by = None
    job.claimed_at = None
    job.lease_expires_at = None
    db.commit()
    db.refresh(job)
    return job


def _retry_target(job: Job, from_stage: str | None) -> JobStep:
    ordered = sorted(job.steps, key=lambda step: step.ordinal)
    if from_stage is None:
        failed = [s for s in ordered if s.status in (RunStatus.ERROR, RunStatus.CANCELLED)]
        incomplete = [s for s in ordered if s.status is not RunStatus.COMPLETE]
        return (failed or incomplete or ordered)[0]
    for step in ordered:
        if step.stage_id == from_stage:
            return step
    known = ", ".join(step.stage_id for step in ordered)
    raise ValueError(f"job {job.id} has no stage '{from_stage}'; its stages are {known}")


def read_step_log(
    db: Session, storage: ObjectStorage, job_id: uuid.UUID, step_id: uuid.UUID
) -> JobStepLog:
    """Fetch one step's log out of object storage.

    Logs are not in the database on purpose — a run's logs are unbounded — so this reads
    the object `job_steps.log_key` names. A step that has not written one yet is a 404,
    not an empty string, because "no log" and "an empty log" are different answers.
    """
    step = db.get(JobStep, step_id)
    if step is None or step.job_id != job_id:
        raise NotFoundError("job step", step_id)
    if not step.log_key:
        raise NotFoundError("log for job step", step_id)
    return JobStepLog(
        step_id=step.id,
        stage_id=step.stage_id,
        log_key=step.log_key,
        text=storage.get_object(step.log_key).decode("utf-8", errors="replace"),
    )


def job_to_read(job: Job) -> JobRead:
    return JobRead.model_validate(job)
