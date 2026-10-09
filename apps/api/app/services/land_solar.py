from __future__ import annotations

import hashlib
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.analysis.solar import SolarResult
from app.models.land import LandArea
from app.models.land_solar import LandSolar, LandSolarBlob
from app.models.research import Investigation, ResearchRun
from app.models.workspace import Workspace
from app.schemas.land_solar import SolarAssessmentRead, SolarMetadata, SolarRequest
from app.services.errors import InvalidInputError, NotFoundError


def scoped(db: Session, workspace_id: uuid.UUID, assessment_id: uuid.UUID) -> LandSolar:
    row = db.scalar(
        select(LandSolar)
        .join(LandArea, LandArea.id == LandSolar.land_id)
        .where(LandArea.workspace_id == workspace_id, LandSolar.id == assessment_id)
    )
    if row is None:
        raise NotFoundError("land solar assessment", assessment_id)
    return row


def read(db: Session, row: LandSolar) -> SolarAssessmentRead:
    revision = db.scalar(select(LandArea.revision).where(LandArea.id == row.land_id))
    return SolarAssessmentRead(
        id=row.id,
        run_id=row.run_id,
        land_id=row.land_id,
        boundary_revision=row.boundary_revision,
        request=SolarRequest.model_validate(row.request),
        metadata=SolarMetadata.model_validate(row.metadata_json),
        sha256=row.sha256,
        byte_size=row.byte_size,
        created_at=row.created_at,
        stale=revision != row.boundary_revision,
    )


def save(
    db: Session,
    workspace_id: uuid.UUID,
    run: ResearchRun,
    assessment_id: uuid.UUID,
    request: SolarRequest,
    result: SolarResult,
    quota_bytes: int,
) -> LandSolar:
    """The worker holds the run's live lease lock; caller commits with its checkpoint."""
    investigation = db.get(Investigation, run.investigation_id)
    if investigation is None:
        raise NotFoundError("investigation", run.investigation_id)
    land = db.get(LandArea, investigation.land_id)
    if land is None or land.workspace_id != workspace_id:
        raise NotFoundError("land", investigation.land_id)
    db.execute(select(Workspace.id).where(Workspace.id == workspace_id).with_for_update())
    existing = db.get(LandSolar, assessment_id)
    if existing is not None:
        return existing
    if not 0 < len(result.data) <= 16 * 1024 * 1024:
        raise InvalidInputError("The solar assessment output exceeds the 16 MiB storage limit.")
    used = (
        db.scalar(
            select(func.coalesce(func.sum(LandSolar.byte_size), 0))
            .join(LandArea, LandArea.id == LandSolar.land_id)
            .where(LandArea.workspace_id == workspace_id)
        )
        or 0
    )
    if used + len(result.data) > quota_bytes:
        raise InvalidInputError("This workspace's solar assessment storage allowance is full.")
    row = LandSolar(
        id=assessment_id,
        run_id=run.id,
        land_id=land.id,
        boundary_revision=investigation.boundary_revision,
        request=request.model_dump(mode="json"),
        metadata_json=result.metadata.model_dump(mode="json"),
        sha256=hashlib.sha256(result.data).hexdigest(),
        byte_size=len(result.data),
    )
    db.add(row)
    db.flush()
    db.add(LandSolarBlob(assessment_id=row.id, data=result.data))
    db.flush()
    return row


def blob(db: Session, row: LandSolar) -> bytes:
    original = db.get(LandSolarBlob, row.id)
    if original is None:
        raise NotFoundError("solar assessment bytes", row.id)
    return original.data
