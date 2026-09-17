from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import Plan, PlanRevision, Site
from app.schemas.plan import (
    PlanBody,
    PlanCreate,
    PlanRead,
    PlanRevise,
    PlanRevisionRead,
    PlanStatus,
)
from app.services.errors import NotFoundError

ACTIVE_STATUSES: tuple[PlanStatus, ...] = ("scheduled", "dispatched", "paused")


def _body_columns(body: PlanBody) -> dict[str, object]:
    """Column values: dates as dates, JSON columns in the wire (camelCase) shape."""
    data: dict[str, object] = body.model_dump(mode="json", include=set(PlanBody.model_fields))
    data["steps"] = [step.model_dump(mode="json", by_alias=True) for step in body.steps]
    data["areas"] = [area.model_dump(mode="json", by_alias=True) for area in body.areas]
    data["estimates"] = body.estimates.model_dump(mode="json", by_alias=True)
    data["start_date"] = body.start_date
    data["end_date"] = body.end_date
    return data


def list_plans(db: Session, project_id: str) -> list[Plan]:
    stmt = (
        select(Plan)
        .where(Plan.project_id == project_id)
        .options(selectinload(Plan.revisions))
        .order_by(Plan.created_at)
    )
    return list(db.scalars(stmt).all())


def get_plan(db: Session, plan_id: uuid.UUID) -> Plan:
    stmt = select(Plan).where(Plan.id == plan_id).options(selectinload(Plan.revisions))
    plan = db.scalars(stmt).first()
    if plan is None:
        raise NotFoundError("plan", plan_id)
    return plan


def _snapshot(plan: Plan) -> dict[str, object]:
    return {
        "title": plan.title,
        "objective": plan.objective,
        "goal": plan.goal,
        "cadence": plan.cadence,
        "startDate": plan.start_date.isoformat(),
        "endDate": plan.end_date.isoformat() if plan.end_date else None,
        "zoneIds": plan.zone_ids,
        "machineIds": plan.machine_ids,
        "areas": plan.areas,
        "steps": plan.steps,
        "estimates": plan.estimates,
        "assumptions": plan.assumptions,
        "risks": plan.risks,
        "questions": plan.questions,
        "source": plan.source,
        "model": plan.model,
    }


def create_plan(db: Session, payload: PlanCreate) -> Plan:
    if payload.site_id is not None and db.get(Site, payload.site_id) is None:
        raise NotFoundError("site", payload.site_id)
    plan = Plan(
        project_id=payload.project_id,
        site_id=payload.site_id,
        status="scheduled",
        revision=1,
        **_body_columns(payload),
    )
    db.add(plan)
    db.flush()
    plan.revisions.append(PlanRevision(revision=1, note="Approved", snapshot=_snapshot(plan)))
    db.commit()
    return get_plan(db, plan.id)


def revise_plan(db: Session, plan_id: uuid.UUID, payload: PlanRevise) -> Plan:
    plan = get_plan(db, plan_id)
    for key, value in _body_columns(payload).items():
        setattr(plan, key, value)
    plan.revision += 1
    plan.revisions.append(
        PlanRevision(
            revision=plan.revision,
            note=payload.note or "Revised",
            snapshot=_snapshot(plan),
        )
    )
    db.commit()
    return get_plan(db, plan.id)


def set_status(db: Session, plan_id: uuid.UUID, status: PlanStatus) -> Plan:
    plan = get_plan(db, plan_id)
    plan.status = status
    db.commit()
    return get_plan(db, plan.id)


def delete_plan(db: Session, plan_id: uuid.UUID) -> None:
    plan = get_plan(db, plan_id)
    db.delete(plan)
    db.commit()


def plan_to_read(plan: Plan) -> PlanRead:
    return PlanRead(
        id=plan.id,
        project_id=plan.project_id,
        site_id=plan.site_id,
        title=plan.title,
        objective=plan.objective,
        goal=plan.goal,
        status=plan.status,
        cadence=plan.cadence,
        start_date=plan.start_date,
        end_date=plan.end_date,
        zone_ids=plan.zone_ids,
        machine_ids=plan.machine_ids,
        areas=plan.areas,
        steps=plan.steps,
        estimates=plan.estimates,
        assumptions=plan.assumptions,
        risks=plan.risks,
        questions=plan.questions,
        source=plan.source,
        model=plan.model,
        revision=plan.revision,
        revisions=[
            PlanRevisionRead(
                revision=r.revision,
                note=r.note,
                created_at=r.created_at,
                title=str(r.snapshot.get("title", "")),
                machine_ids=list(r.snapshot.get("machineIds", [])),
                zone_ids=list(r.snapshot.get("zoneIds", [])),
                estimates=r.snapshot.get(
                    "estimates", {"acres": 0, "machineHours": 0, "calendarDays": 0}
                ),
            )
            for r in plan.revisions
        ],
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )
