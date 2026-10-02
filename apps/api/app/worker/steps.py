"""Writing `job_step` and `artifacts` as the run goes on.

Every one of these commits. That is the point: the panel polls `GET /jobs` and has to see
a stage appear when it starts and turn green when it finishes, not five rows at once when
the run is over. It is also what stops the worker holding a transaction open across a
stage — A0 measured what an open transaction does to the vacuum horizon, and a stage can
run for hours.
"""

from __future__ import annotations

import json
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
    started_at: datetime | None = None,
) -> JobStep:
    """The row exists while the stage is still running, which is what makes it live.

    `started_at` is when the recipe process started the stage, when it said; now, when
    it did not. The supervisor can read a quick stage's start after the stage is over."""
    step = db.scalar(select(JobStep).where(JobStep.job_id == job_id, JobStep.ordinal == ordinal))
    if step is None:
        step = JobStep(job_id=job_id, stage_id=stage_id, ordinal=ordinal, impl=impl)
        db.add(step)
    step.stage_id = stage_id
    step.impl = impl
    step.attempt = attempt
    step.status = RunStatus.IN_PROGRESS
    step.started_at = started_at or utcnow()
    # A restarted step (a retry, the phone's Refine) starts with nothing measured: the
    # previous attempt's metrics, its progress bar included, describe a run that is over.
    step.metrics = {}
    step.finished_at = None
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
    # The live viewer's last cameras outlive the stage that solved them: training, which
    # runs next, draws its intermediate splats among them (see `report_live`).
    live = (step.metrics or {}).get("live")
    step.metrics = {**metrics, "live": live} if live else metrics
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


def report_progress(db: Session, step: JobStep, progress: dict[str, Any]) -> bool:
    """Record how far a running stage has got, read from its log. True when it changed.

    Kept under `metrics.progress` because `finish_step` replaces the metrics wholesale,
    so the progress of a finished stage disappears with the stage rather than going stale.
    """
    if step.status != RunStatus.IN_PROGRESS or (step.metrics or {}).get("progress") == progress:
        return False
    step.metrics = {**(step.metrics or {}), "progress": progress}
    db.commit()
    return True


#: The most a live-cameras payload may hold, as JSON, before it is refused: the pipeline
#: caps it well below this (400 cameras, 1,500 points), so a bigger one is not its own.
MAX_LIVE_BYTES = 96_000


def report_live(db: Session, step: JobStep, live: dict[str, Any]) -> bool:
    """Record the newest live-viewer state a running stage has logged. True when it changed.

    `live` holds `cameras` (registered cameras and a point sample, from the pose stage)
    and/or `splat` (the newest intermediate splat's object key, from training), exactly
    as `tools/pipeline/live.py` parsed them. Merged into `metrics.live`, so a stage that
    logs only one kind keeps the other; `finish_step` keeps it too, and `start_step`
    clears it with everything else when the stage runs again.
    """
    if step.status != RunStatus.IN_PROGRESS:
        return False
    cameras = live.get("cameras")
    if cameras is not None and len(json.dumps(cameras)) > MAX_LIVE_BYTES:
        live = {key: value for key, value in live.items() if key != "cameras"}
    current = (step.metrics or {}).get("live") or {}
    merged = {**current, **live}
    if not live or merged == current:
        return False
    step.metrics = {**(step.metrics or {}), "live": merged}
    db.commit()
    return True


def fail_step(
    db: Session,
    step: JobStep,
    *,
    log_key: str | None,
    preempted: bool = False,
    checkpoint_key: str | None = None,
) -> None:
    """Close out an attempt that did not finish.

    `preempted` is the real signal, not an inference: B1b's `CloudRunner` raises
    `PreemptedError` when a provider takes the machine back, and only that sets
    `preempted_at`. Until B1b the column was set for *any* second attempt, which made a
    stage that merely fails twice indistinguishable from one that was interrupted — and
    those are the two cases the whole cloud path exists to keep apart.

    The status stays `error` while the stage is between attempts; the supervisor's loop
    is what decides whether there is another one. `checkpoint_key` is recorded here
    because a preempted attempt has a checkpoint and no StepResult to name it in.
    """
    step.status = RunStatus.ERROR
    step.finished_at = utcnow()
    if log_key is not None:
        step.log_key = log_key
    if preempted:
        step.preempted_at = utcnow()
    if checkpoint_key is not None:
        step.checkpoint_key = checkpoint_key
    db.commit()


def stop_active_steps(db: Session, job_id: uuid.UUID, status: RunStatus) -> None:
    """Close out whatever was still running — used when a job is cancelled or dead-lettered."""
    for step in db.scalars(select(JobStep).where(JobStep.job_id == job_id)).all():
        if step.status in (RunStatus.NOT_STARTED, RunStatus.IN_PROGRESS):
            step.status = status
            step.finished_at = utcnow()
    db.commit()
