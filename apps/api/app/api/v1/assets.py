from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import DbSession, RequireWriteToken
from app.schemas.asset import AssetCreate, AssetRead, AssetUpdate
from app.services import assets as asset_service

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
