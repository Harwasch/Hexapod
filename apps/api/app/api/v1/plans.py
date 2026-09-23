from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, Response, status

from app.api.deps import DbSession, RequireWriteToken
from app.schemas.plan import PlanCreate, PlanRead, PlanRevise, PlanStatusUpdate
from app.services import plans as plan_service

router = APIRouter(prefix="/plans", tags=["plans"])


@router.get("", response_model=list[PlanRead], summary="List a project's plans")
def list_plans(
    db: DbSession, project_id: str = Query(alias="projectId", min_length=1, max_length=120)
) -> list[PlanRead]:
    return [plan_service.plan_to_read(p) for p in plan_service.list_plans(db, project_id)]


@router.post(
    "",
    response_model=PlanRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[RequireWriteToken],
    summary="Approve a plan",
)
def create_plan(payload: PlanCreate, db: DbSession) -> PlanRead:
    return plan_service.plan_to_read(plan_service.create_plan(db, payload))


@router.get("/{plan_id}", response_model=PlanRead, summary="Get a plan with its history")
def get_plan(plan_id: uuid.UUID, db: DbSession) -> PlanRead:
    return plan_service.plan_to_read(plan_service.get_plan(db, plan_id))


@router.put(
    "/{plan_id}",
    response_model=PlanRead,
    dependencies=[RequireWriteToken],
    summary="Approve a new revision",
)
def revise_plan(plan_id: uuid.UUID, payload: PlanRevise, db: DbSession) -> PlanRead:
    return plan_service.plan_to_read(plan_service.revise_plan(db, plan_id, payload))


@router.patch(
    "/{plan_id}/status",
    response_model=PlanRead,
    dependencies=[RequireWriteToken],
    summary="Change lifecycle state",
)
def set_status(plan_id: uuid.UUID, payload: PlanStatusUpdate, db: DbSession) -> PlanRead:
    return plan_service.plan_to_read(plan_service.set_status(db, plan_id, payload.status))


@router.delete(
    "/{plan_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[RequireWriteToken],
    summary="Delete a plan",
)
def delete_plan(plan_id: uuid.UUID, db: DbSession) -> Response:
    plan_service.delete_plan(db, plan_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
