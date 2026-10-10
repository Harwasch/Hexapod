from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, Response
from sqlalchemy import select

from app.api.deps import DbSession, SettingsDep
from app.api.workspace_deps import WorkspaceDep, workspace_access
from app.models.land_archive_image import LandArchiveImageBlob
from app.models.land_image_registration import LandImageRegistration
from app.schemas.land_image_registrations import (
    ImageRegistrationCreate,
    ImageRegistrationRead,
    ImageRegistrationResult,
)
from app.services import land_archive_images as images
from app.services import land_image_registrations as registrations
from app.services.errors import NotFoundError

router = APIRouter(tags=["land image alignment"])


@router.get(
    "/research/evidence/{evidence_id}/image/registrations",
    response_model=list[ImageRegistrationRead],
)
def listing(
    evidence_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[ImageRegistrationRead]:
    images.scoped_evidence(db, scope.id, evidence_id)
    rows = db.scalars(
        select(LandImageRegistration)
        .where(LandImageRegistration.evidence_id == evidence_id)
        .order_by(LandImageRegistration.created_at.desc(), LandImageRegistration.id)
        .limit(limit)
        .offset(offset)
    )
    return [registrations.read(row) for row in rows]


@router.post(
    "/research/evidence/{evidence_id}/image/registrations/preview",
    response_model=ImageRegistrationResult,
)
def preview(
    evidence_id: uuid.UUID, payload: ImageRegistrationCreate, db: DbSession, scope: WorkspaceDep
) -> ImageRegistrationResult:
    return registrations.preview(db, scope.id, evidence_id, payload)


@router.post(
    "/research/evidence/{evidence_id}/image/registrations",
    response_model=ImageRegistrationRead,
    status_code=201,
)
def create(
    evidence_id: uuid.UUID,
    payload: ImageRegistrationCreate,
    db: DbSession,
    scope: WorkspaceDep,
    settings: SettingsDep,
) -> ImageRegistrationRead:
    scope.require("owner", "editor")
    result = registrations.preview(db, scope.id, evidence_id, payload)
    found = registrations.existing(db, evidence_id, payload)
    if found is not None:
        return registrations.read(found)
    image = db.get(LandArchiveImageBlob, evidence_id)
    if image is None:
        raise NotFoundError("saved image", evidence_id)
    source = image.preview
    db.rollback()
    rendered = registrations.render_image(source, result)
    workspace_access(scope.principal, db, scope.id).require("owner", "editor")
    row = registrations.save(
        db,
        scope.id,
        evidence_id,
        payload,
        result,
        rendered,
        settings.land_archive_workspace_quota_bytes,
    )
    db.commit()
    return registrations.read(row)


@router.get("/research/image-registrations/{registration_id}", response_model=ImageRegistrationRead)
def get(registration_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> ImageRegistrationRead:
    return registrations.read(registrations.scoped(db, scope.id, registration_id))


@router.get(
    "/research/image-registrations/{registration_id}/tiles/{z}/{x}/{y}.png",
    responses={200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}}},
)
def tile(
    registration_id: uuid.UUID, z: int, x: int, y: int, db: DbSession, scope: WorkspaceDep
) -> Response:
    row = registrations.scoped(db, scope.id, registration_id)
    return Response(
        registrations.tile(db, row, z, x, y),
        media_type="image/png",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.get(
    "/research/image-registrations/{registration_id}/download",
    responses={
        200: {"content": {"image/tiff": {"schema": {"type": "string", "format": "binary"}}}}
    },
)
def download(registration_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> Response:
    row = registrations.scoped(db, scope.id, registration_id)
    return Response(
        registrations.blob(db, row),
        media_type="image/tiff",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'attachment; filename="aligned-map-{row.id}.tif"',
        },
    )
