from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from app.api.deps import DbSession, Storage
from app.schemas.common import Problem
from app.schemas.storage import StorageReconciliation
from app.services import reconcile as reconcile_service

router = APIRouter(prefix="/storage", tags=["storage"])

STORAGE_RESPONSES: dict[int | str, dict[str, Any]] = {
    503: {"model": Problem, "description": "Object storage is not configured"}
}


@router.get(
    "/reconciliation",
    response_model=StorageReconciliation,
    responses=STORAGE_RESPONSES,
    summary="Reconcile object storage against the database, in both directions",
    description=(
        "Walks `captures/` and `runs/` and reports **orphans** — objects no row claims, "
        "which is what a run that died after uploading leaves behind — then takes every "
        "row whose object should exist and reports the **missing** ones. The walk stops "
        "at `maxObjects` and says so; the row check asks storage directly, so it is "
        "exact either way."
    ),
)
def get_reconciliation(
    db: DbSession,
    storage: Storage,
    max_objects: Annotated[int, Query(alias="maxObjects", ge=1, le=50_000)] = (
        reconcile_service.DEFAULT_MAX_OBJECTS
    ),
) -> StorageReconciliation:
    return reconcile_service.reconcile(db, storage, max_objects=max_objects)
