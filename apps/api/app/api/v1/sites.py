from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.api.deps import DbSession, LimitCellMap, PublicStorage, RequireWriteToken
from app.schemas.asset import AssetRead
from app.schemas.bookmark import CameraBookmarkCreate, CameraBookmarkRead
from app.schemas.change import ChangeMapRead
from app.schemas.check_target import CheckTargetRead, CheckTargetSet
from app.schemas.common import Problem
from app.schemas.confidence import FillTargetsRead, WeakAreasRead, WeakAreasRequest
from app.schemas.epoch import EpochRead
from app.schemas.site import SiteCreate, SiteRead, SiteSummary, SiteUpdate
from app.services import assets as asset_service
from app.services import bookmarks as bookmark_service
from app.services import change_maps, confidence_maps
from app.services import check_targets as check_target_service
from app.services import epochs as epoch_service
from app.services import sites as site_service
from app.storage import StorageUnavailableError

router = APIRouter(prefix="/sites", tags=["sites"])

THUMBNAIL_TYPES = {"image/png", "image/jpeg", "image/webp"}
THUMBNAIL_MAX_BYTES = 2 * 1024 * 1024


@router.get("", response_model=list[SiteSummary], summary="List sites")
def list_sites(db: DbSession) -> list[SiteSummary]:
    return [
        site_service.site_to_summary(db, site, area)
        for site, area in site_service.list_sites_with_area(db)
    ]


@router.post(
    "",
    response_model=SiteRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[RequireWriteToken],
    summary="Create a site",
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


@router.patch(
    "/{site_id}",
    response_model=SiteRead,
    dependencies=[RequireWriteToken],
    summary="Update a site",
)
def update_site(site_id: uuid.UUID, payload: SiteUpdate, db: DbSession) -> SiteRead:
    return site_service.site_to_read(db, site_service.update_site(db, site_id, payload))


@router.delete(
    "/{site_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[RequireWriteToken],
    summary="Delete a site",
)
def delete_site(site_id: uuid.UUID, db: DbSession) -> None:
    site_service.delete_site(db, site_id)


@router.get("/{site_id}/assets", response_model=list[AssetRead], summary="List a site's assets")
def list_site_assets(site_id: uuid.UUID, db: DbSession) -> list[AssetRead]:
    site_service.get_site(db, site_id)
    return [asset_service.asset_to_read(a) for a in asset_service.list_assets(db, site_id=site_id)]


@router.get("/{site_id}/epochs", response_model=list[EpochRead], summary="List a site's epochs")
def list_site_epochs(site_id: uuid.UUID, db: DbSession) -> list[EpochRead]:
    """The site's dated captures in its one frame, oldest first, each with the asset it
    currently shows. Empty for a legacy (seeded, catalog, hand-made) site."""
    site = site_service.get_site(db, site_id)
    return [epoch_service.epoch_to_read(e, site.assets) for e in site.epochs]


@router.post(
    "/{site_id}/epochs/{epoch_id}/weak-areas",
    response_model=WeakAreasRead,
    summary="An epoch's weak areas as re-scan zones",
    dependencies=[RequireWriteToken, LimitCellMap],
    responses={
        404: {"model": Problem, "description": "No such site, or no such epoch of it."},
        409: {
            "model": Problem,
            "description": "`confidence-not-measured`: the epoch's current revision has no "
            "confidence map (a splat upload is not measured); `confidence-unreadable`: its "
            "grid cannot be read; `no-site-frame`: the site has no frame to place cells by.",
        },
    },
)
def epoch_weak_areas(
    site_id: uuid.UUID,
    epoch_id: uuid.UUID,
    payload: WeakAreasRequest,
    db: DbSession,
    storage: PublicStorage,
) -> WeakAreasRead:
    """The epoch's weak cells (weak-v1, ADR 0016) merged into at most `maxAreas` zones on
    the globe, largest first, within `budgetM2`, each with a pattern to fly it again and a
    hint. It stores nothing, but it is the first step of planning a flight -- the console's
    re-scan flow, which then asks for a plan draft -- so it takes the write token like every
    POST (and a per-client bucket: it outlines up to 2^22 cells). The patterns are UX
    heuristics, labelled so (`heuristic`)."""
    return confidence_maps.weak_areas(db, storage, site_id, epoch_id, payload)


@router.get(
    "/{site_id}/fill-targets",
    response_model=FillTargetsRead,
    summary="The cells a re-scan left weak",
    dependencies=[RequireWriteToken, LimitCellMap],
    responses={
        404: {"model": Problem, "description": "No such site, or no such epoch of it."},
        409: {
            "model": Problem,
            "description": "`confidence-not-measured`: the epoch has no confidence map (or the "
            "site no published epoch); `no-rescan-yet`: no earlier epoch of the site was "
            "measured; `confidence-unreadable`: a grid cannot be read.",
        },
    },
)
def site_fill_targets(
    site_id: uuid.UUID,
    db: DbSession,
    storage: PublicStorage,
    epoch_id: Annotated[uuid.UUID | None, Query(alias="epochId")] = None,
) -> FillTargetsRead:
    """`rescan-gate-v1`: the finest cells of epoch `epochId` (default: the site's latest
    published) that are weak in it, where the newest earlier measured epoch that covers the
    cell was weak too -- a re-scan that did not fix them; `satisfiedBy` names that epoch per
    target. An inferred fill fills only these (F4). A bit mask in the tileset frame of the site lattice. Behind the
    write token, which its one consumer (the fill workflow) holds: a call may decode a grid
    per earlier epoch, at most the eight most recent (`capped` says when there were more)."""
    return confidence_maps.fill_targets(db, storage, site_id, epoch_id)


@router.get(
    "/{site_id}/change",
    response_model=ChangeMapRead,
    summary="The change between two dates, from height surfaces",
    dependencies=[RequireWriteToken, LimitCellMap],
    responses={
        404: {"model": Problem, "description": "No such site, or no such epoch of it."},
        409: {
            "model": Problem,
            "description": "`epoch-not-published`: an epoch is rejected or superseded; "
            "`surface-not-measured`: an epoch's current revision has no height surface; "
            "`confidence-not-measured`: or no confidence map (its uncertainty is unknown); "
            "`frames-differ`: a surface is above another origin than the site frame's; "
            "`no-site-frame`; `surface-unreadable` / `confidence-unreadable`.",
        },
        422: {"model": Problem, "description": "`from` and `to` are the same epoch."},
    },
)
def site_change(
    site_id: uuid.UUID,
    db: DbSession,
    storage: PublicStorage,
    from_epoch: Annotated[uuid.UUID, Query(alias="from")],
    to_epoch: Annotated[uuid.UUID, Query(alias="to")],
) -> ChangeMapRead:
    """`change-v1` (ADR 0018): from epoch `from` to epoch `to`, both published, per cell of
    the coarser of their height surfaces' levels over the children both measured,
    `dsm_to - dsm_from` against three times the combined spread and placement uncertainty
    -- raised, lowered, unchanged, or insufficient where they share no measured child --
    with the cells either epoch's confidence does not vouch for flagged, not hidden, and the
    regions outlined on the globe. Derived, not observed; uncalibrated. It stores nothing, but a call may
    decode four grids (more than the grid cache holds at their largest), so it takes the
    write token like weak areas and fill targets and spends the same per-client bucket."""
    return change_maps.change(db, storage, site_id, from_epoch, to_epoch)


@router.get(
    "/{site_id}/check-targets",
    response_model=list[CheckTargetRead],
    summary="List a site's active check targets",
    dependencies=[RequireWriteToken],
)
def list_check_targets(site_id: uuid.UUID, db: DbSession) -> list[CheckTargetRead]:
    """The surveyed ground marks the site's epochs are scored against (M3b), by label.

    Behind the write token, unlike the site's other reads: these are survey marks on
    private land, and their coordinates are not public (the R2 review of #52)."""
    return [
        check_target_service.to_read(t) for t in check_target_service.list_for_site(db, site_id)
    ]


@router.put(
    "/{site_id}/check-targets",
    response_model=list[CheckTargetRead],
    summary="Replace a site's active check targets",
    dependencies=[RequireWriteToken],
)
def replace_check_targets(
    site_id: uuid.UUID, payload: CheckTargetSet, db: DbSession
) -> list[CheckTargetRead]:
    """The whole active set. A target sent unchanged keeps its row; a changed one retires
    its row for a new one; one left out is retired. Nothing is deleted. The next run of a
    capture of this site is scored against the set as it is when that run starts."""
    rows = check_target_service.replace_active(db, site_id, payload)
    return [check_target_service.to_read(t) for t in rows]


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
    dependencies=[RequireWriteToken],
    summary="Save a camera bookmark",
)
def create_bookmark(
    site_id: uuid.UUID, payload: CameraBookmarkCreate, db: DbSession
) -> CameraBookmarkRead:
    return CameraBookmarkRead.model_validate(bookmark_service.create_bookmark(db, site_id, payload))


@router.delete(
    "/{site_id}/bookmarks/{bookmark_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[RequireWriteToken],
    summary="Delete a camera bookmark",
)
def delete_bookmark(site_id: uuid.UUID, bookmark_id: uuid.UUID, db: DbSession) -> None:
    bookmark_service.delete_bookmark(db, site_id, bookmark_id)


# Into the **public** bucket, because the URL saved is the one a browser loads. It was
# written to the private one (`Storage`) while the URL saved was built on the public
# host, so in production -- two buckets -- every uploaded thumbnail was a 404.
#
# And a plain `def`: every call in here blocks (the database, boto3, the spooled upload's
# file), and as an `async def` they ran on the event loop, stalling every other request
# for the length of an upload. FastAPI runs a `def` in its threadpool.
@router.post(
    "/{site_id}/thumbnail",
    response_model=SiteRead,
    dependencies=[RequireWriteToken],
    summary="Upload a site thumbnail",
)
def upload_thumbnail(
    site_id: uuid.UUID,
    db: DbSession,
    storage: PublicStorage,
    file: Annotated[UploadFile, File()],
) -> SiteRead:
    site = site_service.get_site(db, site_id)
    if file.content_type not in THUMBNAIL_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "thumbnail must be png, jpeg or webp"
        )
    data = file.file.read(THUMBNAIL_MAX_BYTES + 1)
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
