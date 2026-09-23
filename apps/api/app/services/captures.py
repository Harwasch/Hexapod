"""Captures and their uploads.

Bytes never pass through this service. It creates the multipart upload, hands out
presigned part URLs a window at a time, and records what the client reports back; the
object itself goes browser -> object storage.
"""

from __future__ import annotations

import math
import re
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import segno
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.config import Settings
from app.models import Capture, CaptureFile
from app.models.enums import CaptureStatus, UploadStatus
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
    PresignedPart,
    UploadWindow,
)
from app.schemas.common import Attribution, LicenseMetadata, Provenance, TemporalExtent
from app.schemas.job import JobRead
from app.services import handoff
from app.services.errors import ConflictError, NotFoundError
from app.services.slugs import slugify
from app.storage import DEFAULT_EXPIRES_IN, MultipartPart, ObjectStorage

_MIB = 1024 * 1024

#: Part size for a multipart upload. 8 MiB is A0's figure: comfortably over S3's 5 MiB
#: floor for a non-final part, small enough that a dropped part on a phone costs one
#: retry of 8 MiB rather than of 100 MiB.
PART_SIZE_BYTES = 8 * _MIB

#: How many parts one presign call hands out.
#:
#: Presigning a whole upload does not work: A0 measured a 12 GB video at 8 MiB parts as
#: 1536 parts, about 590 KB of URL JSON in a single response -- and the last of those
#: URLs would expire (DEFAULT_EXPIRES_IN, one hour) hours before a phone on cellular
#: reached it. 32 is chosen against both limits at once: it is 256 MiB of upload per
#: window, which is ~7 minutes at 5 Mbps and ~36 minutes even at 1 Mbps, so a window
#: is used well inside the hour its URLs are signed for; and it is ~13 KB of JSON,
#: three orders of magnitude off the 590 KB. The cost is 48 presign calls across that
#: 12 GB upload, which is nothing next to the transfer itself.
PRESIGN_WINDOW_PARTS = 32

#: S3's hard ceiling on parts in one multipart upload. Past 78 GiB (10 000 x 8 MiB)
#: the part size grows instead of the part count.
MAX_MULTIPART_PARTS = 10_000

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_DEFAULT_CONTENT_TYPE = "application/octet-stream"


def choose_part_size(total_bytes: int | None) -> int:
    """Part size for a declared upload size, rounded up to whole MiB.

    Pure in `total_bytes`, and `total_bytes` is stored on the row, so the same file
    yields the same part size on every later window -- which is why no part_size column
    is needed to resume an upload.
    """
    if total_bytes is None or total_bytes <= PART_SIZE_BYTES * MAX_MULTIPART_PARTS:
        return PART_SIZE_BYTES
    return math.ceil(total_bytes / MAX_MULTIPART_PARTS / _MIB) * _MIB


def count_parts(total_bytes: int | None, part_size: int) -> int | None:
    """Parts for a declared size, or None when the client did not declare one."""
    if total_bytes is None:
        return None
    return max(1, math.ceil(total_bytes / part_size))


def object_name(filename: str) -> str:
    """The last path segment of `filename`, reduced to characters safe in a key."""
    stem = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    safe = _UNSAFE_NAME.sub("-", stem).strip(".-")
    return safe[:200] or "file.bin"


def storage_key_for(capture_id: uuid.UUID, file_id: uuid.UUID, filename: str) -> str:
    """Keys are per-file, not per-name: two files called IMG_0001.MOV do not collide."""
    return f"captures/{capture_id}/source/{file_id}/{object_name(filename)}"


# --- captures ---------------------------------------------------------------


def _unique_slug(db: Session, base: str) -> str:
    slug = base
    counter = 2
    while db.scalar(select(Capture.id).where(Capture.slug == slug)) is not None:
        slug = f"{base}-{counter}"
        counter += 1
    return slug


def create_capture(db: Session, payload: CaptureCreate) -> Capture:
    if payload.slug and db.scalar(select(Capture.id).where(Capture.slug == payload.slug)):
        raise ConflictError(f"capture slug '{payload.slug}' already exists")
    capture = Capture(
        slug=payload.slug or _unique_slug(db, slugify(payload.name)),
        name=payload.name,
        description=payload.description,
        kind=payload.kind,
        status=CaptureStatus.AWAITING_FILES,
        device=payload.device,
        sensor=payload.sensor,
        captured_at=payload.captured_at,
        temporal_extent=payload.temporal_extent.model_dump(mode="json", by_alias=True)
        if payload.temporal_extent
        else None,
        metadata_=payload.metadata,
        attribution=[a.model_dump(mode="json", by_alias=True) for a in payload.attribution],
        license=payload.license.model_dump(mode="json", by_alias=True) if payload.license else None,
        provenance=payload.provenance.model_dump(mode="json", by_alias=True)
        if payload.provenance
        else None,
    )
    db.add(capture)
    db.commit()
    db.refresh(capture)
    return capture


def list_captures(db: Session, *, limit: int = 50, offset: int = 0) -> list[Capture]:
    """Newest first: a capture panel is about what was just uploaded."""
    stmt = (
        select(Capture)
        .options(selectinload(Capture.files))
        .order_by(Capture.created_at.desc(), Capture.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(stmt).all())


def get_capture(db: Session, capture_id: uuid.UUID) -> Capture:
    capture = db.get(Capture, capture_id)
    if capture is None:
        raise NotFoundError("capture", capture_id)
    return capture


# --- phone handoff ----------------------------------------------------------

#: QR error correction. "M" recovers ~15% of the symbol, which is what makes a code
#: readable off a glossy monitor at an angle; "L" is smaller but marginal in exactly
#: those conditions, and "Q"/"H" grow the symbol for a link that is on screen for seconds.
_QR_ERROR = "m"
#: 4 px per module, so the whole symbol is 212 px square for a token-length URL -- big
#: enough for a phone camera at arm's length. segno's SVG carries no viewBox, so this is
#: the rendered size, not a hint.
_QR_SCALE = 4
#: Quiet zone, in modules. The spec asks for 4; 2 is the practical minimum and survives a
#: light card on a dark panel, which is what this sits on.
_QR_BORDER = 2
#: Black on white, not `currentColor`: segno rejects it, and a scanner wants the real
#: thing. The white quiet zone is what makes the code readable on the console's dark glass.
_QR_DARK = "#000000"
_QR_LIGHT = "#ffffff"


def create_handoff(db: Session, settings: Settings, capture_id: uuid.UUID) -> CaptureHandoff:
    """Mint an upload link for one existing capture, and draw it.

    The capture is looked up first, so a handoff for a capture that does not exist is a
    404 rather than a token for nothing. This is the only place a handoff token is
    created, and reaching it needs the write token (see the router).
    """
    capture = get_capture(db, capture_id)
    now = int(time.time())
    key = handoff.key_for(settings)
    token = handoff.issue(key, capture.id, now=now)
    url = f"{settings.public_web_base.rstrip('/')}/upload.html#{token}"
    return CaptureHandoff(
        capture_id=capture.id,
        url=url,
        token=token,
        expires_at=datetime.fromtimestamp(now + handoff.TOKEN_TTL_SECONDS, tz=UTC),
        expires_in=handoff.TOKEN_TTL_SECONDS,
        renewable_until=datetime.fromtimestamp(now + handoff.RENEWABLE_FOR_SECONDS, tz=UTC),
        qr_svg=segno.make(url, error=_QR_ERROR).svg_inline(
            scale=_QR_SCALE,
            border=_QR_BORDER,
            dark=_QR_DARK,
            light=_QR_LIGHT,
            svgclass="qr-code",
            title="Scan to upload from a phone",
        ),
    )


# --- files ------------------------------------------------------------------


def get_file(db: Session, capture_id: uuid.UUID, file_id: uuid.UUID) -> CaptureFile:
    file = db.get(CaptureFile, file_id)
    if file is None or file.capture_id != capture_id:
        raise NotFoundError("capture file", file_id)
    return file


def _presign_window(
    storage: ObjectStorage, file: CaptureFile, *, first_part_number: int, count: int | None
) -> UploadWindow:
    if file.upload_id is None:  # pragma: no cover - guarded by the callers
        raise ConflictError(f"capture file {file.id} has no upload in progress")
    part_size = choose_part_size(file.bytes)
    total = file.parts_total
    if total is not None and first_part_number > total:
        raise ValueError(f"firstPartNumber {first_part_number} is past the last part ({total})")
    window = min(count or PRESIGN_WINDOW_PARTS, PRESIGN_WINDOW_PARTS)
    last = first_part_number + window - 1
    if total is not None:
        last = min(last, total)
    parts = [
        PresignedPart(
            part_number=number,
            url=storage.presign_part(
                file.storage_key, file.upload_id, number, expires_in=DEFAULT_EXPIRES_IN
            ),
        )
        for number in range(first_part_number, last + 1)
    ]
    return UploadWindow(
        upload_id=file.upload_id,
        storage_key=file.storage_key,
        part_size=part_size,
        parts_total=total,
        parts=parts,
        expires_in=DEFAULT_EXPIRES_IN,
        next_part_number=None if total is not None and last >= total else last + 1,
    )


def register_file(
    db: Session, storage: ObjectStorage, capture_id: uuid.UUID, payload: CaptureFileCreate
) -> CaptureFileUpload:
    """Create the row and the multipart upload, and presign the first window."""
    capture = get_capture(db, capture_id)
    if capture.status not in {CaptureStatus.AWAITING_FILES, CaptureStatus.NOT_STARTED}:
        raise ConflictError(
            f"capture {capture.id} is {capture.status.value}; files can only be added "
            "before processing starts"
        )
    file_id = uuid.uuid4()
    key = storage_key_for(capture.id, file_id, payload.filename)
    content_type = payload.content_type or _DEFAULT_CONTENT_TYPE
    part_size = choose_part_size(payload.bytes)
    upload_id = storage.create_multipart(key, content_type)
    file = CaptureFile(
        id=file_id,
        capture_id=capture.id,
        filename=payload.filename,
        content_type=content_type,
        bytes=payload.bytes,
        storage_key=key,
        status=UploadStatus.IN_PROGRESS,
        upload_id=upload_id,
        parts_total=count_parts(payload.bytes, part_size),
        parts_completed=0,
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return CaptureFileUpload(
        file=file_to_read(file),
        upload=_presign_window(storage, file, first_part_number=1, count=None),
    )


def presign_parts(
    db: Session,
    storage: ObjectStorage,
    capture_id: uuid.UUID,
    file_id: uuid.UUID,
    payload: CaptureFilePartsRequest,
) -> UploadWindow:
    """The next window, or the same one again — this is also how a resume works."""
    file = get_file(db, capture_id, file_id)
    if file.status is not UploadStatus.IN_PROGRESS or file.upload_id is None:
        raise ConflictError(
            f"capture file {file.id} is {file.status.value}; only an upload in progress "
            "can be presigned"
        )
    window = _presign_window(
        storage, file, first_part_number=payload.first_part_number, count=payload.count
    )
    # Asking for the window at part N says the client has N-1 parts behind it. That is
    # the only progress signal the API gets -- the parts themselves go to storage -- and
    # it never moves backwards, so a client re-presigning an earlier window (a retry)
    # does not undo recorded progress.
    file.parts_completed = max(file.parts_completed, payload.first_part_number - 1)
    db.commit()
    return window


def complete_file(
    db: Session,
    storage: ObjectStorage,
    capture_id: uuid.UUID,
    file_id: uuid.UUID,
    payload: CaptureFileComplete,
) -> CaptureFile:
    file = get_file(db, capture_id, file_id)
    if file.status is not UploadStatus.IN_PROGRESS or file.upload_id is None:
        raise ConflictError(
            f"capture file {file.id} is {file.status.value}; only an upload in progress "
            "can be completed"
        )
    parts: Sequence[MultipartPart] = [
        MultipartPart(part_number=part.part_number, etag=part.etag) for part in payload.parts
    ]
    stored = storage.complete_multipart(file.storage_key, file.upload_id, parts)
    file.status = UploadStatus.COMPLETE
    file.bytes = stored.size
    # S3's ETag unless the client computed something better: for a multipart object it
    # is a digest of the part digests with a `-N` suffix, not an MD5 of the content.
    file.checksum = payload.checksum or stored.etag
    file.parts_completed = len(payload.parts)
    file.parts_total = len(payload.parts)
    file.upload_id = None
    _refresh_capture_status(file.capture)
    db.commit()
    db.refresh(file)
    return file


def abort_file(
    db: Session, storage: ObjectStorage, capture_id: uuid.UUID, file_id: uuid.UUID
) -> CaptureFile:
    file = get_file(db, capture_id, file_id)
    if file.status is UploadStatus.COMPLETE:
        raise ConflictError(f"capture file {file.id} is already complete")
    if file.upload_id is not None:
        storage.abort_multipart(file.storage_key, file.upload_id)
    file.status = UploadStatus.ABORTED
    # The id is dead once S3 has aborted it; keeping it would invite a retry that
    # cannot work.
    file.upload_id = None
    _refresh_capture_status(file.capture)
    db.commit()
    db.refresh(file)
    return file


def _refresh_capture_status(capture: Capture) -> None:
    """A capture is ready to process once every file it is still waiting on has landed.

    Only ever moves between `awaiting-files` and `not-started`: `in-progress`,
    `complete` and `error` belong to a pipeline run, and an upload must not overwrite
    them.
    """
    if capture.status not in {CaptureStatus.AWAITING_FILES, CaptureStatus.NOT_STARTED}:
        return
    pending = {UploadStatus.NOT_STARTED, UploadStatus.IN_PROGRESS}
    complete = any(file.status is UploadStatus.COMPLETE for file in capture.files)
    waiting = any(file.status in pending for file in capture.files)
    capture.status = (
        CaptureStatus.NOT_STARTED if complete and not waiting else CaptureStatus.AWAITING_FILES
    )


# --- read models ------------------------------------------------------------


def file_to_read(file: CaptureFile) -> CaptureFileRead:
    return CaptureFileRead.model_validate(file)


def capture_to_read(capture: Capture) -> CaptureRead:
    return CaptureRead(**_capture_fields(capture))


def capture_to_detail(capture: Capture, jobs: Sequence[JobRead]) -> CaptureDetail:
    return CaptureDetail(**_capture_fields(capture), jobs=list(jobs))


def _capture_fields(capture: Capture) -> dict[str, object]:
    # Spelled out rather than model_validate(capture): the ORM attribute is
    # `metadata_`, because `metadata` on a SQLAlchemy model is the MetaData object.
    return {
        "id": capture.id,
        "slug": capture.slug,
        "name": capture.name,
        "description": capture.description,
        "status": capture.status,
        "kind": capture.kind,
        "device": capture.device,
        "sensor": capture.sensor,
        "captured_at": capture.captured_at,
        "temporal_extent": TemporalExtent.model_validate(capture.temporal_extent)
        if capture.temporal_extent
        else None,
        "georef_method": capture.georef_method,
        "scale_source": capture.scale_source,
        "uncertainty_m": capture.uncertainty_m,
        "metadata": capture.metadata_,
        "attribution": [Attribution.model_validate(a) for a in capture.attribution],
        "license": LicenseMetadata.model_validate(capture.license) if capture.license else None,
        "provenance": Provenance.model_validate(capture.provenance) if capture.provenance else None,
        "site_id": capture.site_id,
        "files": [file_to_read(file) for file in capture.files],
        "created_at": capture.created_at,
        "updated_at": capture.updated_at,
    }
