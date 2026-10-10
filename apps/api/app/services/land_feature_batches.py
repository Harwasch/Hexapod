from __future__ import annotations

import hashlib
import json
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.land import LandArea
from app.models.land_feature import LandFeatureBatch
from app.schemas.land_feature_batches import (
    FeatureBatchPreview,
    FeatureBatchRead,
    FeatureBatchRequest,
    FeatureBatchRow,
)
from app.schemas.land_features import FeatureGeometryRequest, LandFeatureCreate
from app.services import land_features
from app.services.errors import ConflictError, NotFoundError
from app.services.land import get_land


def read(row: LandFeatureBatch) -> FeatureBatchRead:
    return FeatureBatchRead(
        id=row.id,
        land_id=row.land_id,
        boundary_revision=row.boundary_revision,
        request_key=row.request_key,
        source_file_sha256=row.source_file_sha256,
        source_label=row.source_label,
        rows=[FeatureBatchRow.model_validate(value) for value in row.results],
        created_at=row.created_at,
    )


def scoped(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, identifier: uuid.UUID
) -> LandFeatureBatch:
    get_land(db, workspace_id, land_id)
    row = db.scalar(
        select(LandFeatureBatch).where(
            LandFeatureBatch.id == identifier, LandFeatureBatch.land_id == land_id
        )
    )
    if row is None:
        raise NotFoundError("inventory import", identifier)
    return row


def identities(feature: LandFeatureCreate) -> list[tuple[str, str, str]]:
    values = []
    if feature.external_ref:
        values.append(("external", feature.external_ref.namespace, feature.external_ref.record_id))
    if feature.source.url and feature.source.record_id:
        values.append(("source", str(feature.source.url), feature.source.record_id))
    return values


def preview(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, payload: FeatureBatchRequest
) -> FeatureBatchPreview:
    land = get_land(db, workspace_id, land_id)
    if land.revision != payload.boundary_revision:
        raise ConflictError(
            "The land boundary changed. Reopen the land and review this import again."
        )
    batch_id = uuid.uuid5(land_id, f"feature-batch/{payload.request_key}")
    seen: dict[tuple[str, str, str], uuid.UUID] = {}
    rows = []
    for item in payload.rows:
        feature = item.feature
        land_features.validate_evidence(db, land_id, feature.evidence_ids)
        identifier = uuid.uuid5(land_id, f"feature/{uuid.uuid5(batch_id, item.row_id)}")
        existing = land_features.duplicate_source(db, land_id, feature, identifier)
        keys = identities(feature)
        local = {seen[key] for key in keys if key in seen}
        if len(local | ({existing} if existing else set())) > 1:
            raise ConflictError(f"Row {item.row_id} has conflicting source identities.")
        duplicate = next(iter(local), None)
        target = existing or duplicate or identifier
        if (existing or duplicate) and not payload.skip_duplicates:
            raise ConflictError(
                f"Row {item.row_id} duplicates an asset. Choose skip duplicates or review its identity."
            )
        for key in keys:
            seen[key] = target
        rows.append(
            FeatureBatchRow(
                row_id=item.row_id,
                feature_id=target,
                name=feature.name,
                disposition="existing"
                if existing
                else "duplicate-in-file"
                if duplicate
                else "created",
                geometry_preview=land_features.preview_geometry(
                    db, workspace_id, land_id, FeatureGeometryRequest(geometry=feature.geometry)
                ),
            )
        )
    return FeatureBatchPreview(boundary_revision=land.revision, rows=rows)


def create(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, payload: FeatureBatchRequest
) -> FeatureBatchRead:
    get_land(db, workspace_id, land_id)
    db.execute(select(LandArea.id).where(LandArea.id == land_id).with_for_update())
    db.expire_all()
    identifier = uuid.uuid5(land_id, f"feature-batch/{payload.request_key}")
    digest = hashlib.sha256(
        json.dumps(payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    existing = db.get(LandFeatureBatch, identifier)
    if existing:
        if existing.request_sha256 != digest:
            raise ConflictError(
                "This import request was already used with different rows. Review a new import."
            )
        return read(existing)
    checked = preview(db, workspace_id, land_id, payload)
    try:
        for item, result in zip(payload.rows, checked.rows, strict=True):
            if result.disposition == "created":
                request = item.feature.model_copy(
                    update={"request_key": uuid.uuid5(identifier, item.row_id)}
                )
                feature = land_features.create(db, workspace_id, land_id, request, commit=False)
                if feature.id != result.feature_id:
                    raise RuntimeError("inventory import identity mismatch")
        row = LandFeatureBatch(
            id=identifier,
            land_id=land_id,
            request_key=payload.request_key,
            request_sha256=digest,
            boundary_revision=payload.boundary_revision,
            source_file_sha256=payload.source_file_sha256,
            source_label=payload.source_label,
            results=[
                value.model_dump(mode="json", exclude={"geometry_preview"})
                for value in checked.rows
            ],
        )
        db.add(row)
        db.commit()
        return read(row)
    except Exception:
        db.rollback()
        raise
