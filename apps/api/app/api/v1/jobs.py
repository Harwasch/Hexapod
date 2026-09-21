from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import DbSession, RequireWriteToken
from app.models.enums import RunStatus
from app.schemas.job import JobRead
from app.services import jobs as job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


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
