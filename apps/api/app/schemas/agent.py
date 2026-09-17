"""Mission-planning agent schemas: the context the operator's console sends and the draft
that comes back. Every draft names its source so the UI never presents a rule-based draft as
a model's work."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from app.schemas.base import CamelModel

Cadence = Literal["once", "daily", "weekly", "monthly", "seasonal"]
PlannerSource = Literal["claude", "rules"]


class PlannerMachine(CamelModel):
    id: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=120)
    model: str = Field(default="", max_length=120)
    status: str = Field(default="idle", max_length=40)
    task: str = Field(default="", max_length=200)
    battery_pct: int = Field(default=100, ge=0, le=100)


class PlannerZone(CamelModel):
    id: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=120)
    acres: float = Field(default=0, ge=0)
    task: str = Field(default="", max_length=200)
    progress_pct: int = Field(default=0, ge=0, le=100)
    treated: bool = False
    note: str = Field(default="", max_length=500)


class PlannerExistingPlan(CamelModel):
    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=200)
    zone_ids: list[str] = Field(default_factory=list)


class PlanStep(CamelModel):
    """Every field explicit (no defaults) so the contract marks them required on read."""

    title: str = Field(min_length=1, max_length=200)
    detail: str = Field(max_length=1000)
    machine_ids: list[str]
    zone_id: str | None
    when: str = Field(max_length=120)


class PlanEstimates(CamelModel):
    acres: float = Field(ge=0)
    machine_hours: float = Field(ge=0)
    calendar_days: int = Field(ge=0)


class PlanDraftBody(CamelModel):
    """The plan itself, as drafted: what the operator reviews, edits and approves."""

    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=2000)
    zone_ids: list[str] = Field(default_factory=list)
    machine_ids: list[str] = Field(default_factory=list)
    cadence: Cadence = "once"
    start_date: date
    end_date: date | None = None
    estimates: PlanEstimates
    steps: list[PlanStep] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)


class PlanDraftRequest(CamelModel):
    goal: str = Field(min_length=3, max_length=2000)
    project_name: str = Field(min_length=1, max_length=200)
    site_name: str | None = Field(default=None, max_length=200)
    machines: list[PlannerMachine] = Field(default_factory=list, max_length=200)
    zones: list[PlannerZone] = Field(default_factory=list, max_length=500)
    existing_plans: list[PlannerExistingPlan] = Field(default_factory=list, max_length=200)
    # Operator pre-selections from the console (a selected zone, machines ticked in Fleet).
    preferred_zone_ids: list[str] = Field(default_factory=list)
    preferred_machine_ids: list[str] = Field(default_factory=list)
    # A follow-up instruction on a previous draft ("use three machines", "skip Z-21").
    refinement: str | None = Field(default=None, max_length=2000)
    previous_draft: PlanDraftBody | None = None
    today: date | None = None


class PlanDraft(CamelModel):
    """Read model: every field is explicit (no defaults) so the contract marks it required."""

    title: str
    objective: str
    zone_ids: list[str]
    machine_ids: list[str]
    cadence: Cadence
    start_date: date
    end_date: date | None
    estimates: PlanEstimates
    steps: list[PlanStep]
    risks: list[str]
    questions: list[str]
    source: PlannerSource
    model: str | None
    # Shown next to the draft: who drafted it and what it assumed.
    note: str


class PlannerStatus(CamelModel):
    configured: bool
    provider: PlannerSource
    model: str | None
