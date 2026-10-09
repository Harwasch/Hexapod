from __future__ import annotations

import uuid

from geoalchemy2.shape import from_shape
from shapely.geometry import shape
from sqlalchemy import func, null, select
from sqlalchemy.orm import Session

from app.models.land import LandArea
from app.models.land_feature import FeatureInspection, LandFeature, LandFeatureRevision
from app.models.research import Evidence, Investigation, ResearchRun
from app.schemas.land_features import (
    FeatureGeometryRead,
    FeatureGeometryRequest,
    FeatureInspectionCreate,
    FeatureInspectionRead,
    LandFeatureCreate,
    LandFeatureRead,
    LandFeatureRevise,
    LandFeatureRevisionRead,
)
from app.services.errors import ConflictError, InvalidInputError, NotFoundError
from app.services.land import get_land


def preview_geometry(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, payload: FeatureGeometryRequest
) -> FeatureGeometryRead:
    get_land(db, workspace_id, land_id)
    geometry = func.ST_SetSRID(func.ST_GeomFromGeoJSON(payload.geometry.model_dump_json()), 4326)
    geographic = func.geography(geometry)
    polygon = payload.geometry.type in {"Polygon", "MultiPolygon"}
    intersects, distance, area, length, perimeter = db.execute(
        select(
            func.ST_Intersects(geometry, LandArea.boundary),
            func.ST_Distance(geographic, func.geography(LandArea.boundary)),
            func.ST_Area(geographic) if polygon else null(),
            func.ST_Length(geographic) if payload.geometry.type == "LineString" else null(),
            func.ST_Perimeter(geographic) if polygon else null(),
        ).where(LandArea.id == land_id)
    ).one()
    return FeatureGeometryRead(
        geometry=payload.geometry,
        intersects_land=intersects,
        distance_m=distance,
        area_m2=area,
        length_m=length,
        perimeter_m=perimeter,
    )


def validate_evidence(db: Session, land_id: uuid.UUID, ids: list[uuid.UUID]) -> None:
    if not ids:
        return
    found = set(
        db.scalars(
            select(Evidence.id)
            .join(ResearchRun, ResearchRun.id == Evidence.run_id)
            .join(Investigation, Investigation.id == ResearchRun.investigation_id)
            .where(Investigation.land_id == land_id, Evidence.id.in_(ids))
        )
    )
    if found != set(ids):
        raise InvalidInputError("Inventory evidence must belong to this land's investigations.")


def scoped(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    identifier: uuid.UUID,
    *,
    lock: bool = False,
) -> LandFeature:
    query = (
        select(LandFeature)
        .join(LandArea, LandArea.id == LandFeature.land_id)
        .where(
            LandArea.workspace_id == workspace_id,
            LandFeature.land_id == land_id,
            LandFeature.id == identifier,
        )
    )
    if lock:
        query = query.with_for_update(of=LandFeature).execution_options(populate_existing=True)
    row = db.scalar(query)
    if row is None:
        raise NotFoundError("inventory feature", identifier)
    return row


def read(db: Session, row: LandFeature) -> LandFeatureRead:
    revision, intersects, distance = db.execute(
        select(
            LandArea.revision,
            func.ST_Intersects(LandFeature.geometry, LandArea.boundary),
            func.ST_Distance(
                func.geography(LandFeature.geometry), func.geography(LandArea.boundary)
            ),
        )
        .join(LandFeature, LandFeature.land_id == LandArea.id)
        .where(LandFeature.id == row.id)
    ).one()
    return LandFeatureRead(
        **row.content,
        id=row.id,
        land_id=row.land_id,
        revision=row.revision,
        boundary_revision=revision,
        intersects_land=intersects,
        distance_m=distance,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def reject_duplicate_source(
    db: Session, land_id: uuid.UUID, payload: LandFeatureCreate, excluding: uuid.UUID
) -> None:
    # Exact source record duplicates become a reviewable conflict, never a second asset.
    if payload.source.record_id and payload.source.url:
        duplicate = db.scalar(
            select(LandFeature.id).where(
                LandFeature.land_id == land_id,
                LandFeature.id != excluding,
                LandFeature.content["source"]["record_id"].astext == payload.source.record_id,
                LandFeature.content["source"]["url"].astext == str(payload.source.url),
            )
        )
        if duplicate:
            raise ConflictError(f"This source record is already in the inventory ({duplicate}).")


def create(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, payload: LandFeatureCreate
) -> LandFeatureRead:
    get_land(db, workspace_id, land_id)
    db.execute(select(LandArea.id).where(LandArea.id == land_id).with_for_update())
    identifier = uuid.uuid5(land_id, f"feature/{payload.request_key}")
    existing = db.get(LandFeature, identifier)
    content = payload.model_dump(mode="json")
    if existing:
        original = db.scalar(
            select(LandFeatureRevision).where(
                LandFeatureRevision.feature_id == identifier, LandFeatureRevision.revision == 1
            )
        )
        if original is None or original.content != content:
            raise ConflictError("This inventory request was already used with different details.")
        return read(db, existing)
    validate_evidence(db, land_id, payload.evidence_ids)
    reject_duplicate_source(db, land_id, payload, identifier)
    row = LandFeature(
        id=identifier,
        land_id=land_id,
        content=content,
        revision=1,
        geometry=from_shape(shape(payload.geometry.model_dump()), srid=4326),
    )
    db.add(row)
    db.flush()
    db.add(
        LandFeatureRevision(feature_id=row.id, revision=1, content=content, note="Feature recorded")
    )
    db.commit()
    return read(db, row)


def revise(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    identifier: uuid.UUID,
    payload: LandFeatureRevise,
) -> LandFeatureRead:
    get_land(db, workspace_id, land_id)
    db.execute(select(LandArea.id).where(LandArea.id == land_id).with_for_update())
    row = scoped(db, workspace_id, land_id, identifier, lock=True)
    reject_duplicate_source(db, land_id, payload, identifier)
    if row.revision != payload.expected_revision:
        raise ConflictError("This feature changed. Reload before saving your edit.")
    validate_evidence(db, land_id, payload.evidence_ids)
    row.revision += 1
    row.content = payload.model_dump(mode="json", exclude={"expected_revision", "note"})
    row.geometry = from_shape(shape(payload.geometry.model_dump()), srid=4326)
    db.add(
        LandFeatureRevision(
            feature_id=row.id, revision=row.revision, content=row.content, note=payload.note
        )
    )
    db.commit()
    return read(db, row)


def history(
    db: Session, row: LandFeature, limit: int, offset: int
) -> list[LandFeatureRevisionRead]:
    revisions = db.scalars(
        select(LandFeatureRevision)
        .where(LandFeatureRevision.feature_id == row.id)
        .order_by(LandFeatureRevision.revision.desc())
        .limit(limit)
        .offset(offset)
    )
    return [
        LandFeatureRevisionRead(
            revision=item.revision,
            content=LandFeatureCreate.model_validate(item.content),
            note=item.note,
            created_at=item.created_at,
        )
        for item in revisions
    ]


def inspection_read(row: FeatureInspection) -> FeatureInspectionRead:
    return FeatureInspectionRead(
        **row.content,
        id=row.id,
        feature_id=row.feature_id,
        feature_revision=row.feature_revision,
        recorded_by=row.recorded_by,
        created_at=row.created_at,
    )


def inspect(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    identifier: uuid.UUID,
    principal: str,
    payload: FeatureInspectionCreate,
) -> FeatureInspectionRead:
    feature = scoped(db, workspace_id, land_id, identifier, lock=True)
    validate_evidence(db, land_id, payload.evidence_ids)
    inspection_id = uuid.uuid5(identifier, f"inspection/{payload.request_key}")
    content = payload.model_dump(mode="json")
    existing = db.get(FeatureInspection, inspection_id)
    if existing:
        if existing.content != content:
            raise ConflictError(
                "This inspection request was already used with different observations."
            )
        return inspection_read(existing)
    row = FeatureInspection(
        id=inspection_id,
        feature_id=identifier,
        feature_revision=feature.revision,
        recorded_by=principal,
        observed_at=payload.observed_at,
        content=content,
    )
    db.add(row)
    db.commit()
    return inspection_read(row)
