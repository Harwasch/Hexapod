from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Query, status

from app.api.deps import DbSession, PublicStorage, RequireWriteToken, Storage
from app.schemas.asset import AssetCreate, AssetRead, AssetScaleUpdate, AssetUpdate
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


SCALE_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": Problem, "description": "No such asset"},
    409: {
        "model": Problem,
        "description": (
            "`not_scalable`: not the splat a pipeline run registered (a seeded or hand-made "
            "asset, a mesh, a Cesium ion asset), so nothing records the origin its tiles are "
            "placed at"
        ),
    },
}


@router.put(
    "/{asset_id}/scale",
    response_model=AssetRead,
    responses=SCALE_RESPONSES,
    dependencies=[RequireWriteToken],
    summary="Set an asset's real-world size, or reset it",
    description=(
        "Sets `renderConfig.scale`, the factor the viewer draws a pipeline-placed splat at "
        "relative to the model as registered, and `renderConfig.scaleEvidence`, how it was "
        "found: `{scale, evidence?}`, or `{reset: true}` for 1 as registered. The scale is "
        "absolute, never relative to the one before (0.8 then 0.5 is 0.5). The site's "
        "boundary and centroid, the asset's footprint and its `groundSamples` are resized "
        "about the tiles' placed origin with it, so every position the catalog gives already "
        "fits the scaled model. The provenance's `scaleSource` becomes `manual` "
        "(`measured-length`, `direct`, or no evidence) or `camera-height-estimate` with its "
        "`scaleUncertaintyPct`; a reset restores the one the asset was registered with. A "
        "re-run that registers new tiles resets it. 422 for a scale outside 0.01-100, "
        "evidence missing what its method needs, or a measured length that says another "
        "scale. See docs/DATA_MODEL.md."
    ),
)
def set_asset_scale(asset_id: uuid.UUID, payload: AssetScaleUpdate, db: DbSession) -> AssetRead:
    return asset_service.asset_to_read(asset_service.set_scale(db, asset_id, payload))


@router.delete(
    "/{asset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[RequireWriteToken],
    summary="Delete an asset",
)
def delete_asset(asset_id: uuid.UUID, db: DbSession) -> None:
    asset_service.delete_asset(db, asset_id)


SIDECAR_RESPONSES: dict[int | str, dict[str, Any]] = {
    409: {
        "model": Problem,
        "description": (
            "Refused, and `code` says why: `tiles_changed` -- `basedOn` no longer holds the "
            "asset's tiles, so compute the sidecars again on its current tileset; `busy` -- "
            "another attach or a publish held the asset, so retry; `not_attachable` -- the "
            "asset or its directory cannot take an attach (not a run's 3D Tiles in the "
            "public bucket, not a tileset, too many objects, an object too large to copy)"
        ),
    },
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
        "generation. 409, with the Problem's `code`, when `basedOn` no longer holds the "
        "asset's tiles (`tiles_changed`: a republish replaced them), when another attach "
        "held the asset too long (`busy`), or when the asset cannot take an attach "
        "(`not_attachable`: not a run's tileset in the public bucket, say); 422 for a staged "
        "file outside the rules (path, extension, size, a tile or `tileset.json`). See "
        "docs/DEPLOYMENT.md."
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
