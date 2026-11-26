"""Idempotent bundle entry points over the existing capture/multipart/job ledger."""

from __future__ import annotations

import uuid

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Capture, Job
from app.models.enums import CaptureKind, UploadStatus
from app.schemas.capture import CaptureCreate, CaptureFileCreate, CaptureFileRead
from app.schemas.job import JobCreate
from app.services import captures, jobs, phone_access
from app.services.errors import ConflictError, InvalidInputError
from app.storage import ObjectStorage

FORMAT = "hex-capture-v1"
CONTENT_TYPE = "application/x-hex-capture"
PARAMS = {"normalize": {"capture_bundle": True}, "pose": {"capture_bundle": True}}


def begin(
    db: Session,
    bundle_id: uuid.UUID,
    payload: CaptureCreate,
    *,
    phone_settings: Settings | None = None,
) -> Capture:
    # The client persists this UUID BEFORE its first request. A lost response must not
    # create a second capture. Transaction lock also covers simultaneous first requests.
    key = int.from_bytes(bundle_id.bytes[:8], "big", signed=True)
    db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})
    slug = f"hex-{bundle_id}"
    existing = db.scalar(select(Capture).where(Capture.slug == slug))
    if existing is not None:
        if phone_settings is not None:
            phone_access.capture(db, existing.id)
        if existing.metadata_.get("bundleFormat") != FORMAT:
            raise ConflictError("bundle identifier belongs to another capture")
        return existing
    if payload.kind != CaptureKind.VIDEO:
        raise InvalidInputError("a capture bundle must have kind video")
    if phone_settings is not None:
        phone_access.creation_budget(db, phone_settings)
        if payload.site_id is not None:
            phone_access.check_join_site(db, payload.site_id)
        payload = payload.model_copy(
            update={"metadata": {**payload.metadata, "origin": phone_access.ORIGIN}}
        )
    value = payload.model_copy(
        update={"slug": slug, "metadata": {**payload.metadata, "bundleFormat": FORMAT}}
    )
    return captures.create_capture(db, value)


def locked_capture(db: Session, capture_id: uuid.UUID) -> Capture:
    capture = captures.get_capture(db, capture_id)
    db.execute(select(Capture.id).where(Capture.id == capture_id).with_for_update())
    db.refresh(capture)
    if capture.kind != CaptureKind.VIDEO or capture.metadata_.get("bundleFormat") != FORMAT:
        raise InvalidInputError("capture is not a hex-capture-v1 bundle")
    return capture


def register(
    db: Session, storage: ObjectStorage, capture_id: uuid.UUID, payload: CaptureFileCreate
) -> CaptureFileRead:
    capture = locked_capture(db, capture_id)
    if (
        payload.filename != "capture.hexcap"
        or payload.content_type != CONTENT_TYPE
        or payload.bytes is None
        or payload.bytes <= 0
    ):
        raise InvalidInputError(
            "bundle requires capture.hexcap, its content type and positive bytes"
        )
    if capture.files:
        if len(capture.files) != 1:
            raise ConflictError("bundle requires exactly one archive")
        file = capture.files[0]
        if (
            file.filename != payload.filename
            or file.bytes != payload.bytes
            or file.content_type != CONTENT_TYPE
            or file.status not in (UploadStatus.IN_PROGRESS, UploadStatus.COMPLETE)
        ):
            raise ConflictError("bundle upload differs from the existing upload")
        return captures.file_to_read(file)
    return captures.register_file(db, storage, capture_id, payload).file


def completed_capture(db: Session, capture_id: uuid.UUID) -> Capture:
    """Lock and validate the single completed archive for processing or correction."""
    capture = locked_capture(db, capture_id)
    if (
        len(capture.files) != 1
        or capture.files[0].status != UploadStatus.COMPLETE
        or capture.files[0].filename != "capture.hexcap"
        or capture.files[0].content_type != CONTENT_TYPE
    ):
        raise ConflictError("complete the single bundle archive before processing")
    return capture


def process(db: Session, capture_id: uuid.UUID, *, phone: bool = False) -> Job:
    completed_capture(db, capture_id)
    if phone:
        return jobs.create_on_demand(
            db, capture_id, JobCreate(recipe="photo-reconstruct", params=PARAMS)
        )
    existing = db.scalar(
        select(Job)
        .where(Job.capture_id == capture_id, Job.recipe == "photo-reconstruct")
        .order_by(Job.created_at)
    )
    if existing is not None:
        if existing.params != PARAMS:
            raise ConflictError("capture already has a different reconstruction")
        return existing
    # Validation of untrusted archive contents runs in normalize, before any paid pose
    # or training stage. The API never downloads a multi-gigabyte capture into memory.
    return jobs.create_job(db, capture_id, JobCreate(recipe="photo-reconstruct", params=PARAMS))
