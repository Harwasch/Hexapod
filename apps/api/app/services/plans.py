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
from app.services.errors import ConflictError, NotFoundError

ACTIVE_STATUSES: tuple[PlanStatus, ...] = ("scheduled", "dispatched", "paused")
#: Plans a list answers with at most: the newest, so a plan just approved is always listed.
LIST_LIMIT = 200
#: Set on a list response that left older plans out.
TRUNCATED_HEADER = "X-Truncated"


def _body_columns(body: PlanBody) -> dict[str, object]:
    """Column values: dates as dates, JSON columns in the wire (camelCase) shape."""
    data: dict[str, object] = body.model_dump(mode="json", include=set(PlanBody.model_fields))
    data["steps"] = [step.model_dump(mode="json", by_alias=True) for step in body.steps]
    data["areas"] = [area.model_dump(mode="json", by_alias=True) for area in body.areas]
    data["estimates"] = body.estimates.model_dump(mode="json", by_alias=True)
    data["start_date"] = body.start_date
    data["end_date"] = body.end_date
    return data


def list_plans(
    db: Session,
    project_id: str | None,
    *,
    workspace_id: uuid.UUID | None = None,
    land_id: uuid.UUID | None = None,
    limit: int | None = None,
) -> tuple[list[Plan], bool]:
    """A project's plans in one scope: the public ones (no workspace) unless a workspace is
    given, and then only that workspace's. Never both, so a public list cannot carry a
    private plan.

    The newest `limit` (LIST_LIMIT by default), oldest first, and whether older ones were
    left out: past the limit a list drops the oldest, never a plan just approved."""
    limit = LIST_LIMIT if limit is None else limit
    scope = (
        Plan.workspace_id.is_(None) if workspace_id is None else Plan.workspace_id == workspace_id
    )
    stmt = select(Plan).where(scope)
    if project_id is not None:
        stmt = stmt.where(Plan.project_id == project_id)
    if land_id is not None:
        stmt = stmt.where(Plan.land_id == land_id)
    stmt = (
        stmt.options(selectinload(Plan.revisions))
        .order_by(Plan.created_at.desc(), Plan.id.desc())
        .limit(limit + 1)
    )
    newest = list(db.scalars(stmt).all())
    return newest[:limit][::-1], len(newest) > limit


def get_plan(db: Session, plan_id: uuid.UUID, *, workspace_id: uuid.UUID | None = None) -> Plan:
    stmt = (
        select(Plan)
        .where(Plan.id == plan_id, Plan.workspace_id == workspace_id)
        .options(selectinload(Plan.revisions))
    )
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


def create_plan(
    db: Session,
    payload: PlanCreate,
    *,
    workspace_id: uuid.UUID | None = None,
    land_id: uuid.UUID | None = None,
    boundary_revision: int | None = None,
    commit: bool = True,
) -> Plan:
    if payload.site_id is not None and db.get(Site, payload.site_id) is None:
        raise NotFoundError("site", payload.site_id)
    plan = Plan(
        workspace_id=workspace_id,
        land_id=land_id,
        boundary_revision=boundary_revision,
        project_id=payload.project_id,
        site_id=payload.site_id,
        status="scheduled",
        revision=1,
        **_body_columns(payload),
    )
    db.add(plan)
    db.flush()
    plan.revisions.append(PlanRevision(revision=1, note="Approved", snapshot=_snapshot(plan)))
    if commit:
        db.commit()
    else:
        db.flush()
    return get_plan(db, plan.id, workspace_id=workspace_id)


def land_action_id(plan: Plan) -> uuid.UUID | None:
    """The land action a mission was scheduled from (kept on its first revision), if any."""
    first = plan.revisions[0] if plan.revisions else None
    landed = first.snapshot.get("landAction") if first else None
    raw = landed.get("id") if isinstance(landed, dict) else None
    try:
        return uuid.UUID(str(raw)) if raw else None
    except ValueError:
        return None


def revise_plan(
    db: Session,
    plan_id: uuid.UUID,
    payload: PlanRevise,
    *,
    workspace_id: uuid.UUID | None = None,
) -> Plan:
    plan = get_plan(db, plan_id, workspace_id=workspace_id)
    if land_action_id(plan) is not None:
        # The action holds the reviewed, approved version; a second editor here would fork it.
        raise ConflictError(
            "This mission was scheduled from a land action; revise the action in Land."
        )
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
    return get_plan(db, plan.id, workspace_id=workspace_id)


def set_status(
    db: Session,
    plan_id: uuid.UUID,
    status: PlanStatus,
    *,
    workspace_id: uuid.UUID | None = None,
) -> Plan:
    plan = get_plan(db, plan_id, workspace_id=workspace_id)
    plan.status = status
    db.commit()
    return get_plan(db, plan.id, workspace_id=workspace_id)


def delete_plan(db: Session, plan_id: uuid.UUID, *, workspace_id: uuid.UUID | None = None) -> None:
    plan = get_plan(db, plan_id, workspace_id=workspace_id)
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
        land_id=plan.land_id,
        boundary_revision=plan.boundary_revision,
        private=plan.workspace_id is not None,
        land_action_id=land_action_id(plan),
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )
