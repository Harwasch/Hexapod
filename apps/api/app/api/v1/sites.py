from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.deps import DbSession, Storage
from app.schemas.asset import AssetRead
from app.schemas.bookmark import CameraBookmarkCreate, CameraBookmarkRead
from app.schemas.site import SiteCreate, SiteRead, SiteSummary, SiteUpdate
from app.services import assets as asset_service
from app.services import bookmarks as bookmark_service
from app.services import sites as site_service
from app.storage import StorageUnavailableError

router = APIRouter(prefix="/sites", tags=["sites"])

THUMBNAIL_TYPES = {"image/png", "image/jpeg", "image/webp"}
THUMBNAIL_MAX_BYTES = 2 * 1024 * 1024


@router.get("", response_model=list[SiteSummary], summary="List sites")
def list_sites(db: DbSession) -> list[SiteSummary]:
    return [site_service.site_to_summary(db, site) for site in site_service.list_sites(db)]


@router.post(
    "", response_model=SiteRead, status_code=status.HTTP_201_CREATED, summary="Create a site"
)
def create_site(payload: SiteCreate, db: DbSession) -> SiteRead:
    site = site_service.create_site(db, payload)
    return site_service.site_to_read(db, site)


@router.get("/by-slug/{slug}", response_model=SiteRead, summary="Get a site by slug")
def get_site_by_slug(slug: str, db: DbSession) -> SiteRead:
    return site_service.site_to_read(db, site_service.get_site_by_slug(db, slug))


@router.get("/{site_id}", response_model=SiteRead, summary="Get a site")
def get_site(site_id: uuid.UUID, db: DbSession) -> SiteRead:
    return site_service.site_to_read(db, site_service.get_site(db, site_id))


@router.patch("/{site_id}", response_model=SiteRead, summary="Update a site")
def update_site(site_id: uuid.UUID, payload: SiteUpdate, db: DbSession) -> SiteRead:
    return site_service.site_to_read(db, site_service.update_site(db, site_id, payload))


@router.delete("/{site_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a site")
def delete_site(site_id: uuid.UUID, db: DbSession) -> None:
    site_service.delete_site(db, site_id)


@router.get("/{site_id}/assets", response_model=list[AssetRead], summary="List a site's assets")
def list_site_assets(site_id: uuid.UUID, db: DbSession) -> list[AssetRead]:
    site_service.get_site(db, site_id)
    return [asset_service.asset_to_read(a) for a in asset_service.list_assets(db, site_id=site_id)]


@router.get(
    "/{site_id}/bookmarks", response_model=list[CameraBookmarkRead], summary="List camera bookmarks"
)
def list_bookmarks(site_id: uuid.UUID, db: DbSession) -> list[CameraBookmarkRead]:
    return [
        CameraBookmarkRead.model_validate(b) for b in bookmark_service.list_bookmarks(db, site_id)
    ]


@router.post(
    "/{site_id}/bookmarks",
    response_model=CameraBookmarkRead,
    status_code=status.HTTP_201_CREATED,
    summary="Save a camera bookmark",
)
def create_bookmark(
    site_id: uuid.UUID, payload: CameraBookmarkCreate, db: DbSession
) -> CameraBookmarkRead:
    return CameraBookmarkRead.model_validate(bookmark_service.create_bookmark(db, site_id, payload))


@router.delete(
    "/{site_id}/bookmarks/{bookmark_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a camera bookmark",
)
def delete_bookmark(site_id: uuid.UUID, bookmark_id: uuid.UUID, db: DbSession) -> None:
    bookmark_service.delete_bookmark(db, site_id, bookmark_id)


@router.post("/{site_id}/thumbnail", response_model=SiteRead, summary="Upload a site thumbnail")
async def upload_thumbnail(
    site_id: uuid.UUID, db: DbSession, storage: Storage, file: Annotated[UploadFile, File()]
) -> SiteRead:
    site = site_service.get_site(db, site_id)
    if file.content_type not in THUMBNAIL_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "thumbnail must be png, jpeg or webp"
        )
    data = await file.read(THUMBNAIL_MAX_BYTES + 1)
    if len(data) > THUMBNAIL_MAX_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "thumbnail must be 2 MB or smaller")
    extension = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}[file.content_type]
    try:
        stored = storage.put_object(
            f"sites/{site.id}/thumbnail.{extension}", data, content_type=file.content_type
        )
    except StorageUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    site.thumbnail_url = stored.url
    db.commit()
    return site_service.site_to_read(db, site_service.get_site(db, site.id))
