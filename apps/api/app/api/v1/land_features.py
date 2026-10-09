from __future__ import annotations

import uuid

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.deps import DbSession
from app.api.workspace_deps import WorkspaceDep
from app.models.land_feature import (
    FeatureInspection,
    LandFeature,
    LandFeatureBatch,
    LandFeatureRevision,
)
from app.schemas.land_feature_batches import (
    FeatureBatchPreview,
    FeatureBatchRead,
    FeatureBatchRequest,
)
from app.schemas.land_features import (
    FeatureGeometryRead,
    FeatureGeometryRequest,
    FeatureInspectionCreate,
    FeatureInspectionRead,
    FeatureRequestRead,
    LandFeatureCreate,
    LandFeatureRead,
    LandFeatureRevise,
    LandFeatureRevisionRead,
)
from app.services import land_feature_batches, land_features
from app.services.land import get_land

router = APIRouter(prefix="/land/{land_id}/features", tags=["land inventory"])


@router.get("", response_model=list[LandFeatureRead])
def listing(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[LandFeatureRead]:
    get_land(db, scope.id, land_id)
    rows = db.scalars(
        select(LandFeature)
        .where(LandFeature.land_id == land_id)
        .order_by(LandFeature.updated_at.desc(), LandFeature.id)
        .limit(limit)
        .offset(offset)
    )
    return [land_features.read(db, row) for row in rows]


@router.post("", response_model=LandFeatureRead, status_code=201)
def create(
    land_id: uuid.UUID, payload: LandFeatureCreate, db: DbSession, scope: WorkspaceDep
) -> LandFeatureRead:
    scope.require("owner", "editor")
    return land_features.create(db, scope.id, land_id, payload)


@router.post("/geometry/preview", response_model=FeatureGeometryRead)
def geometry_preview(
    land_id: uuid.UUID, payload: FeatureGeometryRequest, db: DbSession, scope: WorkspaceDep
) -> FeatureGeometryRead:
    return land_features.preview_geometry(db, scope.id, land_id, payload)


@router.post("/imports/preview", response_model=FeatureBatchPreview)
def import_preview(
    land_id: uuid.UUID, payload: FeatureBatchRequest, db: DbSession, scope: WorkspaceDep
) -> FeatureBatchPreview:
    return land_feature_batches.preview(db, scope.id, land_id, payload)


@router.post("/imports", response_model=FeatureBatchRead, status_code=201)
def import_features(
    land_id: uuid.UUID, payload: FeatureBatchRequest, db: DbSession, scope: WorkspaceDep
) -> FeatureBatchRead:
    scope.require("owner", "editor")
    return land_feature_batches.create(db, scope.id, land_id, payload)


@router.get("/imports", response_model=list[FeatureBatchRead])
def imports(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[FeatureBatchRead]:
    get_land(db, scope.id, land_id)
    rows = db.scalars(
        select(LandFeatureBatch)
        .where(LandFeatureBatch.land_id == land_id)
        .order_by(LandFeatureBatch.created_at.desc(), LandFeatureBatch.id)
        .limit(limit)
        .offset(offset)
    )
    return [land_feature_batches.read(row) for row in rows]


@router.get("/imports/{import_id}", response_model=FeatureBatchRead)
def imported_batch(
    land_id: uuid.UUID, import_id: uuid.UUID, db: DbSession, scope: WorkspaceDep
) -> FeatureBatchRead:
    return land_feature_batches.read(land_feature_batches.scoped(db, scope.id, land_id, import_id))


@router.get("/requests/{request_key}", response_model=FeatureRequestRead)
def created_request(
    land_id: uuid.UUID, request_key: uuid.UUID, db: DbSession, scope: WorkspaceDep
) -> FeatureRequestRead:
    identifier = uuid.uuid5(land_id, f"feature/{request_key}")
    row = land_features.scoped(db, scope.id, land_id, identifier)
    original = db.scalars(
        select(LandFeatureRevision).where(
            LandFeatureRevision.feature_id == identifier, LandFeatureRevision.revision == 1
        )
    ).one()
    return FeatureRequestRead(
        current=land_features.read(db, row),
        original=LandFeatureCreate.model_validate(original.content),
    )


@router.get("/{feature_id}", response_model=LandFeatureRead)
def get(
    land_id: uuid.UUID, feature_id: uuid.UUID, db: DbSession, scope: WorkspaceDep
) -> LandFeatureRead:
    return land_features.read(db, land_features.scoped(db, scope.id, land_id, feature_id))


@router.put("/{feature_id}", response_model=LandFeatureRead)
def revise(
    land_id: uuid.UUID,
    feature_id: uuid.UUID,
    payload: LandFeatureRevise,
    db: DbSession,
    scope: WorkspaceDep,
) -> LandFeatureRead:
    scope.require("owner", "editor")
    return land_features.revise(db, scope.id, land_id, feature_id, payload)


@router.get("/{feature_id}/revisions", response_model=list[LandFeatureRevisionRead])
def history(
    land_id: uuid.UUID,
    feature_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[LandFeatureRevisionRead]:
    return land_features.history(
        db, land_features.scoped(db, scope.id, land_id, feature_id), limit, offset
    )


@router.post("/{feature_id}/inspections", response_model=FeatureInspectionRead, status_code=201)
def inspect(
    land_id: uuid.UUID,
    feature_id: uuid.UUID,
    payload: FeatureInspectionCreate,
    db: DbSession,
    scope: WorkspaceDep,
) -> FeatureInspectionRead:
    scope.require("owner", "editor")
    return land_features.inspect(db, scope.id, land_id, feature_id, scope.principal.id, payload)


@router.get("/{feature_id}/inspections", response_model=list[FeatureInspectionRead])
def inspections(
    land_id: uuid.UUID,
    feature_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[FeatureInspectionRead]:
    land_features.scoped(db, scope.id, land_id, feature_id)
    rows = db.scalars(
        select(FeatureInspection)
        .where(FeatureInspection.feature_id == feature_id)
        .order_by(FeatureInspection.observed_at.desc(), FeatureInspection.id)
        .limit(limit)
        .offset(offset)
    )
    return [land_features.inspection_read(row) for row in rows]
