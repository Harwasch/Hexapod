from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Query

from app.api.deps import DbSession, RequireWriteToken, Storage
from app.models.enums import RunStatus
from app.schemas.common import Problem
from app.schemas.job import JobRead, JobRetry, JobStepLog
from app.services import jobs as job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])

#: Reading a log needs a bucket, and says so in the contract.
LOG_RESPONSES: dict[int | str, dict[str, Any]] = {
    503: {"model": Problem, "description": "Object storage is not configured"}
}


@router.get("", response_model=list[JobRead], summary="List jobs, newest first")
def list_jobs(
    db: DbSession,
    capture_id: Annotated[uuid.UUID | None, Query(alias="captureId")] = None,
    status: Annotated[RunStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[JobRead]:
    jobs = job_service.list_jobs(
        db, capture_id=capture_id, status=status, limit=limit, offset=offset
    )
    return [job_service.job_to_read(job) for job in jobs]


@router.get("/{job_id}", response_model=JobRead, summary="Get a job with its steps")
def get_job(job_id: uuid.UUID, db: DbSession) -> JobRead:
    return job_service.job_to_read(job_service.get_job(db, job_id))


@router.post(
    "/{job_id}/cancel",
    response_model=JobRead,
    dependencies=[RequireWriteToken],
    summary="Cancel a job",
)
def cancel_job(job_id: uuid.UUID, db: DbSession) -> JobRead:
    return job_service.job_to_read(job_service.cancel_job(db, job_id))


@router.post(
    "/{job_id}/retry",
    response_model=JobRead,
    dependencies=[RequireWriteToken],
    summary="Retry a finished job, from a stage",
    description=(
        "Re-queues a job that failed or was cancelled, resuming at `fromStage` — or at "
        "the stage that failed, when it is omitted. Earlier stages keep their completed "
        "steps and their artifacts; the worker skips them and picks up from this one. "
        "The attempt budget of the stages being re-run is reset, so a dead-lettered job "
        "can be retried by a person after the worker has stopped retrying it by itself."
    ),
)
def retry_job(job_id: uuid.UUID, payload: JobRetry, db: DbSession) -> JobRead:
    return job_service.job_to_read(job_service.retry_job(db, job_id, payload.from_stage))


@router.get(
    "/{job_id}/steps/{step_id}/log",
    response_model=JobStepLog,
    responses=LOG_RESPONSES,
    summary="Read one step's log",
    description=(
        "Logs live in object storage, not in the database: `logKey` on a step is a key, "
        "and this is what turns it into text. 404 when the step has not written one."
    ),
)
def read_step_log(
    job_id: uuid.UUID, step_id: uuid.UUID, db: DbSession, storage: Storage
) -> JobStepLog:
    return job_service.read_step_log(db, storage, job_id, step_id)
