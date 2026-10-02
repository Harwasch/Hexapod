from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from app.api.deps import DbSession, LimitReconcile, RequireWriteToken, Storage
from app.schemas.common import Problem
from app.schemas.storage import StorageReconciliation
from app.services import reconcile as reconcile_service

router = APIRouter(prefix="/storage", tags=["storage"])

STORAGE_RESPONSES: dict[int | str, dict[str, Any]] = {
    429: {"model": Problem, "description": "Asked too often; see `Retry-After`"},
    503: {"model": Problem, "description": "Object storage is not configured"},
}


# A POST behind the write token, though it changes nothing. It was an open GET, and it is
# the most expensive thing this API does -- up to 50,000 objects listed, and a HEAD per
# row -- and its answer is every key in the private bucket, which is the half of the
# storage layout the bucket split exists to keep private. A GET would not carry the token
# from the console (apps/web/src/api/client.ts attaches it to writes only, and asks for it
# on a write's 401); a POST does, with no special case on either side.
@router.post(
    "/reconciliation",
    response_model=StorageReconciliation,
    responses=STORAGE_RESPONSES,
    # In this order: a request without the token is refused before it spends a token.
    dependencies=[RequireWriteToken, LimitReconcile],
    summary="Reconcile object storage against the database, in both directions",
    description=(
        "Walks `captures/` and `runs/` and reports **orphans** — objects no row claims, "
        "which is what a run that died after uploading leaves behind — then takes every "
        "row whose object should exist and reports the **missing** ones. The walk stops "
        "at `maxObjects` and says so; the row check asks storage directly, so it is "
        "exact either way. Changes nothing, but needs the write token: the answer lists "
        "the private bucket, and the walk is costly enough to be rate-limited."
    ),
)
def reconcile_storage(
    db: DbSession,
    storage: Storage,
    max_objects: Annotated[int, Query(alias="maxObjects", ge=1, le=50_000)] = (
        reconcile_service.DEFAULT_MAX_OBJECTS
    ),
) -> StorageReconciliation:
    return reconcile_service.reconcile(db, storage, max_objects=max_objects)
