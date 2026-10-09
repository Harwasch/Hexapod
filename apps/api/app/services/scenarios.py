from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.analysis.scenarios import analyze
from app.models.land import LandArea, LandBoundaryRevision
from app.models.research import Evidence, Investigation, ResearchRun
from app.models.scenario import LandScenario, LandScenarioRevision
from app.schemas.scenarios import ScenarioCreate, ScenarioRead, ScenarioResult, ScenarioRevise
from app.services.errors import ConflictError, InvalidInputError, NotFoundError
from app.services.land import FOOTPRINT, get_land


def preview(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, payload: ScenarioCreate
) -> ScenarioResult:
    get_land(db, workspace_id, land_id)
    snapshot = db.scalar(
        select(LandBoundaryRevision).where(
            LandBoundaryRevision.land_id == land_id,
            LandBoundaryRevision.revision == payload.boundary_revision,
        )
    )
    if snapshot is None:
        raise NotFoundError("land boundary revision", payload.boundary_revision)
    if payload.evidence_ids:
        ids = set(
            db.scalars(
                select(Evidence.id)
                .join(ResearchRun, ResearchRun.id == Evidence.run_id)
                .join(Investigation, Investigation.id == ResearchRun.investigation_id)
                .where(Investigation.land_id == land_id, Evidence.id.in_(payload.evidence_ids))
            )
        )
        if ids != set(payload.evidence_ids):
            raise InvalidInputError("Scenario evidence must come from this land's investigations.")
    geometry = func.ST_SetSRID(
        func.ST_GeomFromGeoJSON(FOOTPRINT.validate_python(snapshot.boundary).model_dump_json()),
        4326,
    )
    area = db.execute(select(func.ST_Area(func.geography(geometry)))).scalar_one()
    return analyze(payload.inputs, float(area))


def scoped(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    scenario_id: uuid.UUID,
    *,
    lock: bool = False,
) -> LandScenario:
    query = (
        select(LandScenario)
        .join(LandArea, LandArea.id == LandScenario.land_id)
        .where(
            LandArea.workspace_id == workspace_id,
            LandScenario.land_id == land_id,
            LandScenario.id == scenario_id,
        )
    )
    if lock:
        query = query.with_for_update(of=LandScenario).execution_options(populate_existing=True)
    row = db.scalar(query)
    if row is None:
        raise NotFoundError("land scenario", scenario_id)
    return row


def read(
    db: Session, row: LandScenario, snapshot: LandScenarioRevision | None = None
) -> ScenarioRead:
    if snapshot is None:
        snapshot = db.scalar(
            select(LandScenarioRevision).where(
                LandScenarioRevision.scenario_id == row.id,
                LandScenarioRevision.revision == row.revision,
            )
        )
    if snapshot is None:
        raise NotFoundError("scenario revision", row.revision)
    current = db.scalar(select(LandArea.revision).where(LandArea.id == row.land_id))
    return ScenarioRead(
        **snapshot.payload,
        id=row.id,
        land_id=row.land_id,
        revision=snapshot.revision,
        stale=current != snapshot.boundary_revision,
        result=ScenarioResult.model_validate(snapshot.result),
        created_at=row.created_at,
        updated_at=snapshot.created_at,
    )


def create(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    principal: str,
    payload: ScenarioCreate,
    *,
    identifier: uuid.UUID | None = None,
    commit: bool = True,
) -> ScenarioRead:
    identifier = identifier or uuid.uuid5(land_id, str(payload.request_key))
    # Serialize duplicate creates for this land before checking their stable identifier.
    db.execute(
        select(LandArea.id)
        .where(LandArea.id == land_id, LandArea.workspace_id == workspace_id)
        .with_for_update()
    )
    if db.get(LandScenario, identifier):
        row = scoped(db, workspace_id, land_id, identifier)
        original = db.scalar(
            select(LandScenarioRevision).where(
                LandScenarioRevision.scenario_id == identifier, LandScenarioRevision.revision == 1
            )
        )
        if original is None or original.payload != payload.model_dump(mode="json"):
            raise ConflictError("The scenario request was already used for different assumptions.")
        return read(db, row, original)
    result = preview(db, workspace_id, land_id, payload)
    row = LandScenario(
        id=identifier or uuid.uuid4(),
        land_id=land_id,
        name=payload.name,
        kind=payload.inputs.kind,
        revision=1,
        created_by=principal,
    )
    db.add(row)
    db.flush()
    db.add(
        LandScenarioRevision(
            scenario_id=row.id,
            land_id=land_id,
            revision=1,
            boundary_revision=payload.boundary_revision,
            payload=payload.model_dump(mode="json"),
            result=result.model_dump(mode="json"),
        )
    )
    if commit:
        db.commit()
    else:
        db.flush()
    return read(db, row)


def revise(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    scenario_id: uuid.UUID,
    payload: ScenarioRevise,
) -> ScenarioRead:
    row = scoped(db, workspace_id, land_id, scenario_id, lock=True)
    if row.revision != payload.expected_revision:
        raise ConflictError("This scenario changed. Reload it before saving your assumptions.")
    if row.kind != payload.inputs.kind:
        raise InvalidInputError("Create a separate scenario to change the analysis type.")
    result = preview(db, workspace_id, land_id, payload)
    row.revision += 1
    row.name = payload.name
    db.add(
        LandScenarioRevision(
            scenario_id=row.id,
            land_id=land_id,
            revision=row.revision,
            boundary_revision=payload.boundary_revision,
            payload=payload.model_dump(mode="json", exclude={"expected_revision"}),
            result=result.model_dump(mode="json"),
        )
    )
    db.commit()
    return read(db, row)
