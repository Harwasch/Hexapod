from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.deps import OutlinerDep, PlannerDep, RequireWriteToken
from app.schemas.agent import Outline, OutlineRequest, PlanDraft, PlanDraftRequest, PlannerStatus
from app.services.planner import PlannerError

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/status", response_model=PlannerStatus, summary="Mission planner status")
def planner_status(planner: PlannerDep) -> PlannerStatus:
    return planner.status()


# Gated even though it mutates nothing: an unauthenticated caller can spend the
# account's Anthropic credits, and the write token guards anything that costs money
# or changes state, not just rows.
@router.post(
    "/plan-draft",
    dependencies=[RequireWriteToken],
    response_model=PlanDraft,
    summary="Draft a mission plan from a goal",
    description=(
        "Turns an operator goal plus the project's zones, machines and existing plans into a "
        "structured plan draft for review. Drafted by Claude when the API is configured with "
        "a key, by a rule-based planner otherwise; the `source` field says which."
    ),
)
def plan_draft(body: PlanDraftRequest, planner: PlannerDep) -> PlanDraft:
    try:
        return planner.draft(body)
    except PlannerError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


# Gated for the same reason as /plan-draft: it calls a vision model on every request.
@router.post(
    "/outline",
    dependencies=[RequireWriteToken],
    response_model=Outline,
    summary="Outline the ground feature under a clicked point",
    description=(
        "Takes one image of the map view with the clicked point marked and returns the outline "
        "of the field, pond or lot around it in normalized image coordinates. Needs the API to "
        "be configured with a key; there is no rule-based fallback for looking at imagery."
    ),
    responses={503: {"description": "No planning model is configured"}},
)
def outline(body: OutlineRequest, outliner: OutlinerDep) -> Outline:
    if outliner is None:
        raise HTTPException(
            503, "Outlining from imagery needs the API configured with ANTHROPIC_API_KEY."
        )
    try:
        return outliner.outline(body)
    except PlannerError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
