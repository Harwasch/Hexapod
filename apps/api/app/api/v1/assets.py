from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Query, status

from app.api.deps import DbSession, PublicStorage, RequireWriteToken, Storage
from app.schemas.asset import AssetCreate, AssetRead, AssetUpdate
from app.schemas.common import Problem
from app.schemas.sidecar import SidecarAttach, SidecarAttachment
from app.services import assets as asset_service
from app.services import attach as attach_service

router = APIRouter(prefix="/assets", tags=["assets"])


@router.get("", response_model=list[AssetRead], summary="List assets")
def list_assets(
    db: DbSession, site_id: Annotated[uuid.UUID | None, Query(alias="siteId")] = None
) -> list[AssetRead]:
    return [asset_service.asset_to_read(a) for a in asset_service.list_assets(db, site_id=site_id)]


@router.post(
    "",
    response_model=AssetRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[RequireWriteToken],
    summary="Register an asset",
)
def create_asset(payload: AssetCreate, db: DbSession) -> AssetRead:
    return asset_service.asset_to_read(asset_service.create_asset(db, payload))


@router.get("/{asset_id}", response_model=AssetRead, summary="Get an asset")
def get_asset(asset_id: uuid.UUID, db: DbSession) -> AssetRead:
    return asset_service.asset_to_read(asset_service.get_asset(db, asset_id))


@router.patch(
    "/{asset_id}",
    response_model=AssetRead,
    dependencies=[RequireWriteToken],
    summary="Update an asset",
)
def update_asset(asset_id: uuid.UUID, payload: AssetUpdate, db: DbSession) -> AssetRead:
    return asset_service.asset_to_read(asset_service.update_asset(db, asset_id, payload))


@router.delete(
    "/{asset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[RequireWriteToken],
    summary="Delete an asset",
)
def delete_asset(asset_id: uuid.UUID, db: DbSession) -> None:
    asset_service.delete_asset(db, asset_id)


SIDECAR_RESPONSES: dict[int | str, dict[str, Any]] = {
    503: {"model": Problem, "description": "Object storage is not configured"},
}


@router.post(
    "/{asset_id}/sidecars",
    response_model=SidecarAttachment,
    responses=SIDECAR_RESPONSES,
    dependencies=[RequireWriteToken],
    summary="Attach sidecars to an asset by cutting a new generation",
    description=(
        "The one way to publish files beside a scan's tiles (`instances.json`, "
        "`collision.bin`, an inferred fill under `inferred/<name>/`, `sog/`, a plant rig). "
        "Stage them in the private bucket under `staging/assets/<asset id>/<token>/`, laid "
        "out as they should sit beside `tileset.json`, then call this. The asset's current "
        "tiles and every sidecar it already has are copied server side into a **new** "
        "generation with the staged files beside them, `tileset.json` with the merged root "
        "`extras` is written last, all of it immutable, and the asset's URL moves to it. "
        "What the request replaces is not copied: a staged kind's old files (its whole file "
        "set, or the directory unit), and the kinds keyed by a replaced kind's ids "
        "(materials and telemetry by `instances`), which are dropped and flagged on the "
        "asset unless the request sends them too. "
        "Attaches to one asset are serialized: a second waits and builds on the first's "
        "generation. 409 when `basedOn` no longer holds the asset's tiles (a republish "
        "replaced them), when the asset is not a run's tileset in the public bucket, or "
        "when another attach held the asset too long; 422 for a staged file outside the "
        "rules (path, extension, size, a tile or `tileset.json`). See docs/DEPLOYMENT.md."
    ),
)
def attach_sidecars(
    asset_id: uuid.UUID,
    payload: SidecarAttach,
    db: DbSession,
    storage: Storage,
    public: PublicStorage,
) -> SidecarAttachment:
    return attach_service.attach_sidecars(db, asset_id, payload, staging=storage, public=public)
