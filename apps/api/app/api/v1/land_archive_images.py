from __future__ import annotations

import uuid

import httpx
from fastapi import APIRouter, Response

from app.api.deps import DbSession, SettingsDep
from app.api.workspace_deps import WorkspaceDep, workspace_access
from app.models.land_archive_image import LandArchiveImage, LandArchiveImageBlob
from app.schemas.land_archives import ArchiveImageRead
from app.services import land_archive_images as images
from app.services.errors import NotFoundError

router = APIRouter(prefix="/research/evidence/{evidence_id}/image", tags=["land archive images"])


@router.get("", response_model=ArchiveImageRead | None)
def get(evidence_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> ArchiveImageRead | None:
    images.scoped_evidence(db, scope.id, evidence_id)
    row = db.get(LandArchiveImage, evidence_id)
    return images.read(row) if row else None


@router.post("", response_model=ArchiveImageRead)
def capture(
    evidence_id: uuid.UUID, db: DbSession, scope: WorkspaceDep, settings: SettingsDep
) -> ArchiveImageRead:
    scope.require("owner", "editor")
    evidence = images.scoped_evidence(db, scope.id, evidence_id)
    existing = db.get(LandArchiveImage, evidence_id)
    if existing is not None:
        return images.read(existing)
    source = images.media(evidence)
    db.rollback()  # Never hold a database transaction while downloading or decoding.
    with httpx.Client(headers={"User-Agent": "LivingWorld-LandResearch/1.0"}) as client:
        snapshot = images.retrieve(source, client)
    workspace_access(scope.principal, db, scope.id).require("owner", "editor")
    row = images.save(
        db, scope.id, evidence_id, snapshot, settings.land_archive_workspace_quota_bytes
    )
    db.commit()
    return images.read(row)


@router.get(
    "/preview",
    responses={200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}}},
)
def preview(evidence_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> Response:
    images.scoped_evidence(db, scope.id, evidence_id)
    blob = db.get(LandArchiveImageBlob, evidence_id)
    if blob is None:
        raise NotFoundError("saved archive image", evidence_id)
    return Response(
        blob.preview,
        media_type="image/png",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.get(
    "/original",
    responses={
        200: {
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            }
        }
    },
)
def original(evidence_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> Response:
    images.scoped_evidence(db, scope.id, evidence_id)
    blob = db.get(LandArchiveImageBlob, evidence_id)
    row = db.get(LandArchiveImage, evidence_id)
    if blob is None or row is None:
        raise NotFoundError("saved archive image", evidence_id)
    media_type = images.read(row).source_media_type
    extension = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}[media_type]
    return Response(
        blob.original,
        media_type=media_type,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'attachment; filename="archive-{evidence_id}.{extension}"',
        },
    )
