from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models import CameraBookmark, Site
from app.schemas.bookmark import CameraBookmarkCreate
from app.services.errors import NotFoundError


def list_bookmarks(db: Session, site_id: uuid.UUID) -> list[CameraBookmark]:
    if db.get(Site, site_id) is None:
        raise NotFoundError("site", site_id)
    stmt = (
        select(CameraBookmark)
        .where(CameraBookmark.site_id == site_id)
        .order_by(CameraBookmark.created_at)
    )
    return list(db.scalars(stmt).all())


def create_bookmark(
    db: Session, site_id: uuid.UUID, payload: CameraBookmarkCreate
) -> CameraBookmark:
    if db.get(Site, site_id) is None:
        raise NotFoundError("site", site_id)
    if payload.is_default:
        db.execute(
            update(CameraBookmark).where(CameraBookmark.site_id == site_id).values(is_default=False)
        )
    bookmark = CameraBookmark(site_id=site_id, **payload.model_dump())
    db.add(bookmark)
    db.commit()
    db.refresh(bookmark)
    return bookmark


def delete_bookmark(db: Session, site_id: uuid.UUID, bookmark_id: uuid.UUID) -> None:
    bookmark = db.get(CameraBookmark, bookmark_id)
    if bookmark is None or bookmark.site_id != site_id:
        raise NotFoundError("bookmark", bookmark_id)
    db.delete(bookmark)
    db.commit()
