from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Query, status

from app.api.deps import (
    DbSession,
    RequireUploadToken,
    RequireWriteToken,
    SettingsDep,
    Storage,
)
from app.schemas.capture import (
    CaptureCreate,
    CaptureDetail,
    CaptureFileComplete,
    CaptureFileCreate,
    CaptureFilePartsRequest,
    CaptureFileRead,
    CaptureFileUpload,
    CaptureHandoff,
    CaptureRead,
    UploadWindow,
)
from app.schemas.common import Problem
from app.schemas.job import JobCreate, JobRead
from app.services import captures as capture_service
from app.services import jobs as job_service

router = APIRouter(prefix="/captures", tags=["captures"])

#: The upload routes are the only ones that need a bucket, and say so in the contract.
STORAGE_RESPONSES: dict[int | str, dict[str, Any]] = {
    503: {"model": Problem, "description": "Object storage is not configured"}
}

Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0)]


@router.get("", response_model=list[CaptureRead], summary="List captures, newest first")
def list_captures(db: DbSession, limit: Limit = 50, offset: Offset = 0) -> list[CaptureRead]:
    return [
        capture_service.capture_to_read(c)
        for c in capture_service.list_captures(db, limit=limit, offset=offset)
    ]


@router.post(
    "",
    response_model=CaptureRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[RequireWriteToken],
    summary="Create a capture",
)
def create_capture(payload: CaptureCreate, db: DbSession) -> CaptureRead:
    return capture_service.capture_to_read(capture_service.create_capture(db, payload))


@router.get(
    "/{capture_id}", response_model=CaptureDetail, summary="Get a capture, its files and its jobs"
)
def get_capture(capture_id: uuid.UUID, db: DbSession) -> CaptureDetail:
    capture = capture_service.get_capture(db, capture_id)
    jobs = [job_service.job_to_read(j) for j in job_service.jobs_for_capture(db, capture.id)]
    return capture_service.capture_to_detail(capture, jobs)


@router.post(
    "/{capture_id}/files",
    response_model=CaptureFileUpload,
    status_code=status.HTTP_201_CREATED,
    dependencies=[RequireUploadToken],
    responses=STORAGE_RESPONSES,
    summary="Register a file and begin its upload",
)
def register_file(
    capture_id: uuid.UUID, payload: CaptureFileCreate, db: DbSession, storage: Storage
) -> CaptureFileUpload:
    return capture_service.register_file(db, storage, capture_id, payload)


@router.post(
    "/{capture_id}/files/{file_id}/parts",
    response_model=UploadWindow,
    dependencies=[RequireUploadToken],
    responses=STORAGE_RESPONSES,
    summary="Presign the next window of parts",
)
def presign_parts(
    capture_id: uuid.UUID,
    file_id: uuid.UUID,
    payload: CaptureFilePartsRequest,
    db: DbSession,
    storage: Storage,
) -> UploadWindow:
    return capture_service.presign_parts(db, storage, capture_id, file_id, payload)


@router.post(
    "/{capture_id}/files/{file_id}/complete",
    response_model=CaptureFileRead,
    dependencies=[RequireUploadToken],
    responses=STORAGE_RESPONSES,
    summary="Complete a file's multipart upload",
)
def complete_file(
    capture_id: uuid.UUID,
    file_id: uuid.UUID,
    payload: CaptureFileComplete,
    db: DbSession,
    storage: Storage,
) -> CaptureFileRead:
    file = capture_service.complete_file(db, storage, capture_id, file_id, payload)
    return capture_service.file_to_read(file)


@router.post(
    "/{capture_id}/files/{file_id}/abort",
    response_model=CaptureFileRead,
    dependencies=[RequireUploadToken],
    responses=STORAGE_RESPONSES,
    summary="Abort a file's multipart upload",
)
def abort_file(
    capture_id: uuid.UUID, file_id: uuid.UUID, db: DbSession, storage: Storage
) -> CaptureFileRead:
    return capture_service.file_to_read(
        capture_service.abort_file(db, storage, capture_id, file_id)
    )


@router.post(
    "/{capture_id}/process",
    response_model=JobRead,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[RequireWriteToken],
    summary="Queue a pipeline run over a capture",
    description=(
        "Inserts a `not-started` job and returns it. Nothing runs here: the worker "
        "claims the row with a lease and executes the recipe."
    ),
)
def process_capture(capture_id: uuid.UUID, payload: JobCreate, db: DbSession) -> JobRead:
    return job_service.job_to_read(job_service.create_job(db, capture_id, payload))


@router.post(
    "/{capture_id}/handoff",
    response_model=CaptureHandoff,
    status_code=status.HTTP_201_CREATED,
    dependencies=[RequireWriteToken],
    summary="Mint a phone-upload link for this capture, and draw its QR code",
    description=(
        "Returns a short-lived token scoped to this capture's upload endpoints, the URL "
        "that carries it in its fragment, and that URL as an inline SVG QR code. The "
        "token is **not** the write token: it cannot create a capture, queue a job, or "
        "touch any other capture, and it expires. Minting one needs the write token."
    ),
)
def create_handoff(capture_id: uuid.UUID, db: DbSession, settings: SettingsDep) -> CaptureHandoff:
    return capture_service.create_handoff(db, settings, capture_id)
