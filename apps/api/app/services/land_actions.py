from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from pyproj import Geod
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.land import LandArea, LandBoundaryRevision
from app.models.land_action import LandAction, LandActionRevision
from app.models.land_feature import LandFeature, LandFeatureRevision
from app.models.scenario import LandScenario, LandScenarioRevision
from app.schemas.agent import PlanEstimates, PlanStep
from app.schemas.land_actions import (
    ActionMissionCreate,
    LandActionCreate,
    LandActionRead,
    LandActionReview,
    LandActionRevise,
)
from app.schemas.plan import PlanArea, PlanCreate, PlanRead
from app.services import plans
from app.services.errors import ConflictError, InvalidInputError, NotFoundError
from app.services.land import FOOTPRINT, get_land
from app.services.land_features import validate_evidence


def scoped(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    identifier: uuid.UUID,
    *,
    lock: bool = False,
) -> LandAction:
    query = (
        select(LandAction)
        .join(LandArea, LandArea.id == LandAction.land_id)
        .where(
            LandArea.workspace_id == workspace_id,
            LandAction.land_id == land_id,
            LandAction.id == identifier,
        )
    )
    if lock:
        query = query.with_for_update(of=LandAction).execution_options(populate_existing=True)
    row = db.scalar(query)
    if row is None:
        raise NotFoundError("land action", identifier)
    return row


def snapshot(db: Session, row: LandAction, revision: int | None = None) -> LandActionRevision:
    result = db.scalar(
        select(LandActionRevision)
        .where(
            LandActionRevision.action_id == row.id,
            LandActionRevision.revision == (revision or row.revision),
        )
        .execution_options(populate_existing=True)
    )
    if result is None:
        raise NotFoundError("action revision", revision)
    return result


def validate(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, payload: LandActionCreate
) -> dict[str, Any]:
    get_land(db, workspace_id, land_id)
    boundary = db.scalar(
        select(LandBoundaryRevision).where(
            LandBoundaryRevision.land_id == land_id,
            LandBoundaryRevision.revision == payload.boundary_revision,
        )
    )
    if boundary is None:
        raise NotFoundError("boundary revision", payload.boundary_revision)
    validate_evidence(
        db,
        land_id,
        payload.evidence_ids
        + [
            identifier
            for constraint in payload.constraints
            for identifier in constraint.evidence_ids
        ],
    )
    if payload.scenario:
        scenario = db.scalar(
            select(LandScenarioRevision).where(
                LandScenarioRevision.scenario_id == payload.scenario.id,
                LandScenarioRevision.land_id == land_id,
                LandScenarioRevision.revision == payload.scenario.revision,
            )
        )
        if scenario is None or scenario.boundary_revision != payload.boundary_revision:
            raise InvalidInputError(
                "Choose a scenario revision for this land and boundary revision."
            )
    for feature in payload.features:
        exists = db.scalar(
            select(LandFeatureRevision.id)
            .join(LandFeature, LandFeature.id == LandFeatureRevision.feature_id)
            .where(
                LandFeature.land_id == land_id,
                LandFeature.id == feature.id,
                LandFeatureRevision.revision == feature.revision,
            )
        )
        if exists is None:
            raise InvalidInputError("Every feature revision must belong to this land.")
    region = shape(boundary.boundary)
    if payload.exclusions:
        region = region.difference(
            unary_union([shape(item.model_dump()) for item in payload.exclusions])
        )
    if region.is_empty or region.geom_type not in {"Polygon", "MultiPolygon"}:
        raise InvalidInputError("Exclusions must leave a nonempty work area.")
    for step in payload.steps:
        if step.footprint and not region.covers(shape(step.footprint.model_dump())):
            raise InvalidInputError(
                f"The footprint for '{step.title}' must lie within the land and outside exclusions."
            )
    return FOOTPRINT.validate_python(mapping(region)).model_dump(mode="json")


def stale_reasons(db: Session, row: LandAction, payload: LandActionCreate) -> list[str]:
    reasons = []
    if (
        db.scalar(select(LandArea.revision).where(LandArea.id == row.land_id))
        != payload.boundary_revision
    ):
        reasons.append(
            "The land boundary has changed; review the action against the current boundary."
        )
    if (
        payload.scenario
        and db.scalar(select(LandScenario.revision).where(LandScenario.id == payload.scenario.id))
        != payload.scenario.revision
    ):
        reasons.append(
            "The supporting scenario has changed; review the selected scenario revision."
        )
    for feature in payload.features:
        if (
            db.scalar(select(LandFeature.revision).where(LandFeature.id == feature.id))
            != feature.revision
        ):
            reasons.append(
                f"Inventory feature {feature.id} has changed since this action was drafted."
            )
    return reasons


def read(db: Session, row: LandAction, version: LandActionRevision | None = None) -> LandActionRead:
    version = version or snapshot(db, row)
    payload = LandActionCreate.model_validate(version.payload)
    return LandActionRead(
        **payload.model_dump(),
        id=row.id,
        land_id=row.land_id,
        revision=version.revision,
        status="scheduled"
        if version.mission_id
        else "approved"
        if version.approved_at
        else "draft",
        stale_reasons=stale_reasons(db, row, payload),
        total_known_cost=sum(step.estimated_cost or 0 for step in payload.steps),
        uncosted_steps=sum(step.estimated_cost is None for step in payload.steps),
        effective_boundary=FOOTPRINT.validate_python(version.effective_boundary),
        approved_by=version.approved_by,
        approved_at=version.approved_at,
        approval_note=version.approval_note,
        mission_id=version.mission_id,
        created_at=row.created_at,
        updated_at=version.updated_at,
    )


def create(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    principal: str,
    payload: LandActionCreate,
    *,
    commit: bool = True,
) -> LandActionRead:
    get_land(db, workspace_id, land_id)
    db.execute(select(LandArea.id).where(LandArea.id == land_id).with_for_update())
    identifier = uuid.uuid5(land_id, f"action/{payload.request_key}")
    existing = db.get(LandAction, identifier)
    content = payload.model_dump(mode="json")
    if existing:
        if snapshot(db, existing, 1).payload != content:
            raise ConflictError("This action request was already used with different details.")
        return read(db, existing)
    region = validate(db, workspace_id, land_id, payload)
    row = LandAction(id=identifier, land_id=land_id, revision=1, created_by=principal)
    db.add(row)
    db.flush()
    db.add(
        LandActionRevision(
            action_id=row.id,
            land_id=land_id,
            revision=1,
            boundary_revision=payload.boundary_revision,
            payload=content,
            effective_boundary=region,
            note="Action drafted",
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
    identifier: uuid.UUID,
    payload: LandActionRevise,
) -> LandActionRead:
    row = scoped(db, workspace_id, land_id, identifier, lock=True)
    if row.revision != payload.expected_revision:
        raise ConflictError("This action changed. Reload before saving your revision.")
    region = validate(db, workspace_id, land_id, payload)
    row.revision += 1
    db.add(
        LandActionRevision(
            action_id=row.id,
            land_id=land_id,
            revision=row.revision,
            boundary_revision=payload.boundary_revision,
            payload=payload.model_dump(mode="json", exclude={"expected_revision", "note"}),
            effective_boundary=region,
            note=payload.note,
        )
    )
    db.commit()
    return read(db, row)


def reviewable(
    db: Session, row: LandAction, version: LandActionRevision, expected: int
) -> LandActionCreate:
    if row.revision != expected:
        raise ConflictError("Review the current action revision before continuing.")
    payload = LandActionCreate.model_validate(version.payload)
    # Hold referenced objects stable through review/approval/scheduling.
    if payload.scenario:
        db.execute(
            select(LandScenario.id).where(LandScenario.id == payload.scenario.id).with_for_update()
        )
    if payload.features:
        db.execute(
            select(LandFeature.id)
            .where(LandFeature.id.in_([item.id for item in payload.features]))
            .order_by(LandFeature.id)
            .with_for_update()
        )
    reasons = stale_reasons(db, row, payload)
    if reasons:
        raise ConflictError(" ".join(reasons))
    if any(not constraint.resolved for constraint in payload.constraints):
        raise InvalidInputError(
            "Resolve all recorded constraints before approving or scheduling this action."
        )
    return payload


def approve(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    identifier: uuid.UUID,
    principal: str,
    payload: LandActionReview,
) -> LandActionRead:
    # All boundary edits lock the land row. Keep the approval check and write in that transaction.
    get_land(db, workspace_id, land_id)
    db.execute(select(LandArea.id).where(LandArea.id == land_id).with_for_update())
    row = scoped(db, workspace_id, land_id, identifier, lock=True)
    version = snapshot(db, row)
    reviewable(db, row, version, payload.expected_revision)
    if version.approved_at is None:
        version.approved_by = principal
        version.approved_at = datetime.now(UTC)
        version.approval_note = payload.note
    db.commit()
    return read(db, row)


def schedule(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    identifier: uuid.UUID,
    payload: ActionMissionCreate,
) -> PlanRead:
    get_land(db, workspace_id, land_id)
    db.execute(select(LandArea.id).where(LandArea.id == land_id).with_for_update())
    row = scoped(db, workspace_id, land_id, identifier, lock=True)
    version = snapshot(db, row)
    if row.revision != payload.expected_revision:
        raise ConflictError("Review the current action revision before continuing.")
    if version.mission_id:
        existing = plans.get_plan(db, version.mission_id, workspace_id=workspace_id)
        if existing.project_id != payload.project_id:
            raise ConflictError("This action revision is already scheduled for another project.")
        return plans.plan_to_read(existing)
    content = reviewable(db, row, version, payload.expected_revision)
    if not version.approved_at:
        raise InvalidInputError("Approve this action revision before scheduling it.")
    region = FOOTPRINT.validate_python(version.effective_boundary)
    areas = [PlanArea(id="land", name="Approved work area", footprint=region)]
    steps = []
    for step in content.steps:
        zone = "land"
        if step.footprint:
            zone = f"step-{step.id}"
            areas.append(PlanArea(id=zone, name=step.title[:120], footprint=step.footprint))
        steps.append(
            PlanStep(
                title=step.title,
                detail=step.detail,
                machine_ids=step.machine_ids,
                zone_id=zone,
                when=f"Day {step.start_day + 1}",
                start_day=step.start_day,
                days=step.days,
            )
        )
    duration = max(step.start_day + step.days for step in content.steps)
    end = content.start_date + timedelta(days=duration - 1)
    # Signed geodesic polygon areas need consistent orientation before adding holes.
    from shapely.geometry.polygon import orient

    geometry = shape(version.effective_boundary)
    assert isinstance(geometry, (Polygon, MultiPolygon))
    polygons = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    area_m2 = sum(
        abs(Geod(ellps="WGS84").geometry_area_perimeter(orient(item))[0]) for item in polygons
    )
    mission = PlanCreate(
        project_id=payload.project_id,
        title=content.title,
        objective=content.objective,
        goal=content.objective,
        start_date=content.start_date,
        end_date=end,
        zone_ids=[area.id for area in areas],
        machine_ids=sorted({machine for step in content.steps for machine in step.machine_ids}),
        areas=areas,
        steps=steps,
        estimates=PlanEstimates(
            acres=area_m2 / 4046.8564224, machine_hours=0, calendar_days=duration
        ),
        assumptions=[
            *content.assumptions,
            "Machine hours are not estimated in this land action.",
            f"Source action {identifier}, revision {row.revision}; boundary revision {content.boundary_revision}.",
        ],
        risks=[
            constraint.text + ": " + constraint.resolution for constraint in content.constraints
        ],
        questions=[],
        source="rules",
    )
    plan = plans.create_plan(db, mission, workspace_id=workspace_id, commit=False)
    plan.revisions[0].note = payload.note[:300]
    plan.revisions[0].snapshot = {
        **plan.revisions[0].snapshot,
        "landAction": {
            "id": str(identifier),
            "revision": row.revision,
            "schedulingNote": payload.note,
        },
    }
    version.mission_id = plan.id
    db.commit()
    return plans.plan_to_read(plan)
