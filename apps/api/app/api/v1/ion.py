from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.deps import Ion
from app.schemas.ion import IonAssetMetadata, IonStatus
from app.services.ion import IonError

router = APIRouter(prefix="/ion", tags=["cesium-ion"])


@router.get("/status", response_model=IonStatus, summary="Cesium ion integration status")
def ion_status(ion: Ion) -> IonStatus:
    return ion.status()


@router.get(
    "/assets/{asset_id}",
    response_model=IonAssetMetadata,
    summary="Read ion asset metadata / tiling status",
    description=(
        "Narrow, documented read of `GET https://api.cesium.com/v1/assets/{id}` using the "
        "server-side ion token. Used to monitor reconstruction jobs. This is not a general proxy."
    ),
)
def ion_asset(asset_id: int, ion: Ion) -> IonAssetMetadata:
    try:
        return ion.get_asset(asset_id)
    except IonError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
