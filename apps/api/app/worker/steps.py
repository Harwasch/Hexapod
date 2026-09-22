"""Writing `job_step` and `artifacts` as the run goes on.

Every one of these commits. That is the point: the panel polls `GET /jobs` and has to see
a stage appear when it starts and turn green when it finishes, not five rows at once when
the run is over. It is also what stops the worker holding a transaction open across a
stage — A0 measured what an open transaction does to the vacuum horizon, and a stage can
run for hours.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Artifact, JobStep
from app.models.enums import RunStatus
from app.worker.outputs import UploadedArtifact


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def steps_of(db: Session, job_id: uuid.UUID) -> dict[str, JobStep]:
    """This job's steps, by stage id. The (job_id, ordinal) constraint makes one row per
    position, so a retried stage updates its row rather than adding a second one."""
    rows = db.scalars(select(JobStep).where(JobStep.job_id == job_id)).all()
    return {row.stage_id: row for row in rows}


def start_step(
    db: Session,
    job_id: uuid.UUID,
    *,
    stage_id: str,
    ordinal: int,
    impl: str,
    attempt: int,
) -> JobStep:
    """The row exists while the stage is still running, which is what makes it live."""
    step = db.scalar(select(JobStep).where(JobStep.job_id == job_id, JobStep.ordinal == ordinal))
    if step is None:
        step = JobStep(job_id=job_id, stage_id=stage_id, ordinal=ordinal, impl=impl)
        db.add(step)
    step.stage_id = stage_id
    step.impl = impl
    step.attempt = attempt
    step.status = RunStatus.IN_PROGRESS
    step.started_at = utcnow()
    step.finished_at = None
    if attempt > 1:
        # Not a status: A2 deliberately has no `preempted` member, because a step that is
        # being tried again is resting in `in-progress`. The column records that it
        # happened.
        step.preempted_at = utcnow()
    db.commit()
    return step


def finish_step(
    db: Session,
    step: JobStep,
    *,
    metrics: dict[str, Any],
    log_key: str | None,
    checkpoint_key: str | None,
    artifacts: list[UploadedArtifact],
) -> None:
    step.status = RunStatus.COMPLETE
    step.finished_at = utcnow()
    step.metrics = metrics
    step.log_key = log_key
    step.checkpoint_key = checkpoint_key
    # A retried stage replaces what it produced last time rather than accumulating it:
    # the keys are the same, so two rows would both claim the same object.
    step.artifacts.clear()
    for uploaded in artifacts:
        step.artifacts.append(
            Artifact(
                kind=uploaded.kind,
                storage_key=uploaded.storage_key,
                bytes=uploaded.bytes,
                checksum=uploaded.checksum,
                content_type=uploaded.content_type,
            )
        )
    db.commit()


def fail_step(db: Session, step: JobStep, *, log_key: str | None) -> None:
    step.status = RunStatus.ERROR
    step.finished_at = utcnow()
    if log_key is not None:
        step.log_key = log_key
    db.commit()


def stop_active_steps(db: Session, job_id: uuid.UUID, status: RunStatus) -> None:
    """Close out whatever was still running — used when a job is cancelled or dead-lettered."""
    for step in db.scalars(select(JobStep).where(JobStep.job_id == job_id)).all():
        if step.status in (RunStatus.NOT_STARTED, RunStatus.IN_PROGRESS):
            step.status = status
            step.finished_at = utcnow()
    db.commit()
