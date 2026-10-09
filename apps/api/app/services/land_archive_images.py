from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.analysis.archive_image import MAX_PREVIEW_BYTES, MAX_SOURCE_BYTES
from app.models.land import LandArea
from app.models.land_archive_image import LandArchiveImage, LandArchiveImageBlob
from app.models.land_image_registration import LandImageRegistration
from app.models.research import Evidence, Investigation, ResearchRun
from app.models.workspace import Workspace
from app.schemas.land_archives import ArchiveImageMetadata, ArchiveImageRead, ArchiveMedia
from app.services.errors import InvalidInputError, NotFoundError


@dataclass(frozen=True)
class ImageSnapshot:
    original: bytes
    preview: bytes
    metadata: ArchiveImageMetadata


def scoped_evidence(db: Session, workspace_id: uuid.UUID, evidence_id: uuid.UUID) -> Evidence:
    row = db.scalar(
        select(Evidence)
        .join(ResearchRun, ResearchRun.id == Evidence.run_id)
        .join(Investigation, Investigation.id == ResearchRun.investigation_id)
        .join(LandArea, LandArea.id == Investigation.land_id)
        .where(LandArea.workspace_id == workspace_id, Evidence.id == evidence_id)
    )
    if row is None:
        raise NotFoundError("archive evidence", evidence_id)
    return row


def media(row: Evidence) -> ArchiveMedia:
    if not row.content.get("media") or row.content.get("provider") not in {
        "commons-place-images",
        "usgs-historical-maps",
    }:
        raise InvalidInputError("This evidence has no registered openly licensed archive preview.")
    return ArchiveMedia.model_validate(row.content["media"])


def retrieve(source: ArchiveMedia, client: httpx.Client) -> ImageSnapshot:
    # The URL is selected from saved registered-source evidence, never from request input.
    source = ArchiveMedia.model_validate(source.model_dump())
    started = time.monotonic()
    data = bytearray()
    try:
        with client.stream(
            "GET",
            str(source.preview_url),
            follow_redirects=False,
            headers={"Accept-Encoding": "identity"},
            timeout=httpx.Timeout(10, read=5),
        ) as response:
            response.raise_for_status()
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise ValueError("Archive image downloads require an unencoded response.")
            if int(response.headers.get("content-length", "0")) > MAX_SOURCE_BYTES:
                raise ValueError("This archive preview exceeds the 5 MiB limit.")
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > MAX_SOURCE_BYTES or time.monotonic() - started > 25:
                    raise ValueError(
                        "The archive preview exceeded its download size or time limit."
                    )
            etag = response.headers.get("etag", "")[:500] or None
            modified = response.headers.get("last-modified", "")[:500] or None
    except (httpx.HTTPError, ValueError) as error:
        raise InvalidInputError(
            "The registered archive preview could not be saved within its download limits."
        ) from error
    with tempfile.TemporaryDirectory(prefix="land-archive-") as temporary:
        directory = Path(temporary)
        (directory / "source").write_bytes(data)
        try:
            result = subprocess.run(  # noqa: S603 -- fixed decoder; paths and bytes are data, never commands
                [sys.executable, "-m", "app.analysis.archive_image", str(directory)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=12,
                check=False,
                env={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                },
            )
        except subprocess.TimeoutExpired as error:
            raise InvalidInputError(
                "The archive preview exceeded its decoding time limit."
            ) from error
        metadata_file, preview_file = directory / "result.json", directory / "preview.png"
        if result.returncode or not metadata_file.is_file() or metadata_file.stat().st_size > 4096:
            raise InvalidInputError("The archive preview could not be safely decoded.")
        decoded = json.loads(metadata_file.read_text())
        if (
            "error" in decoded
            or not preview_file.is_file()
            or preview_file.stat().st_size > MAX_PREVIEW_BYTES
        ):
            raise InvalidInputError(
                "The archive preview could not be safely decoded within its image limits."
            )
        preview = preview_file.read_bytes()
    metadata = ArchiveImageMetadata(
        **decoded,
        sha256=hashlib.sha256(preview).hexdigest(),
        source_sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(preview),
        source_byte_size=len(data),
        source_url=source.preview_url,
        source_etag=etag,
        source_last_modified=modified,
    )
    return ImageSnapshot(bytes(data), preview, metadata)


def save(
    db: Session,
    workspace_id: uuid.UUID,
    evidence_id: uuid.UUID,
    snapshot: ImageSnapshot,
    quota_bytes: int,
) -> LandArchiveImage:
    scoped_evidence(db, workspace_id, evidence_id)
    # Serialize the quota and duplicate check together. Simultaneous retries retain one image.
    db.execute(select(Workspace.id).where(Workspace.id == workspace_id).with_for_update())
    existing = db.get(LandArchiveImage, evidence_id)
    if existing is not None:
        return existing
    used = workspace_used_bytes(db, workspace_id)
    size = len(snapshot.original) + len(snapshot.preview)
    if used + size > quota_bytes:
        raise InvalidInputError("This workspace's archive image storage allowance is full.")
    row = LandArchiveImage(
        evidence_id=evidence_id,
        metadata_json=snapshot.metadata.model_dump(mode="json"),
        byte_size=size,
    )
    db.add(row)
    db.flush()
    db.add(
        LandArchiveImageBlob(
            evidence_id=evidence_id, original=snapshot.original, preview=snapshot.preview
        )
    )
    db.flush()
    return row


def read(row: LandArchiveImage) -> ArchiveImageRead:
    return ArchiveImageRead(
        evidence_id=row.evidence_id, created_at=row.created_at, **row.metadata_json
    )


def vision_input(preview: bytes) -> tuple[bytes, dict[str, str | int]]:
    """Reproducible bounded derivative. Callers pin and verify its checksum on resume."""
    with tempfile.TemporaryDirectory(prefix="land-archive-vision-") as temporary:
        directory = Path(temporary)
        (directory / "source").write_bytes(preview)
        try:
            result = subprocess.run(  # noqa: S603 -- fixed image processor; no executable user input
                [sys.executable, "-m", "app.analysis.archive_image", str(directory), "vision"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=12,
                check=False,
                env={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                },
            )
        except subprocess.TimeoutExpired as error:
            raise InvalidInputError(
                "The image input exceeded its processing time limit."
            ) from error
        metadata_file, output = directory / "result.json", directory / "vision.jpg"
        if result.returncode or not metadata_file.is_file() or metadata_file.stat().st_size > 4096:
            raise InvalidInputError("The saved image could not be prepared for visual inspection.")
        metadata = json.loads(metadata_file.read_text())
        if "error" in metadata or not output.is_file() or output.stat().st_size > 2 * 1024 * 1024:
            raise InvalidInputError(
                "The saved image could not be prepared within its vision limits."
            )
        data = output.read_bytes()
        metadata["sha256"] = hashlib.sha256(data).hexdigest()
        return data, metadata


def workspace_used_bytes(db: Session, workspace_id: uuid.UUID) -> int:
    total = 0
    for model in (LandArchiveImage, LandImageRegistration):
        total += int(
            db.scalar(
                select(func.coalesce(func.sum(model.byte_size), 0))
                .join(Evidence, Evidence.id == model.evidence_id)
                .join(ResearchRun, ResearchRun.id == Evidence.run_id)
                .join(Investigation, Investigation.id == ResearchRun.investigation_id)
                .join(LandArea, LandArea.id == Investigation.land_id)
                .where(LandArea.workspace_id == workspace_id)
            )
            or 0
        )
    return total
