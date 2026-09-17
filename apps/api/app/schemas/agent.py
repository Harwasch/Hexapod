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


class BusyWindow(CamelModel):
    """Days a machine is already booked by another plan (inclusive start, exclusive end)."""

    machine_id: str = Field(min_length=1, max_length=40)
    start_date: date
    end_date: date


class PlannerExistingPlan(CamelModel):
    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=200)
    zone_ids: list[str] = Field(default_factory=list)
    status: str = Field(default="scheduled", max_length=20)
    busy: list[BusyWindow] = Field(default_factory=list, max_length=500)


class PlannerRate(CamelModel):
    """A treatment rate learned from the work log, per task family."""

    task: str = Field(min_length=1, max_length=80)
    acres_per_machine_hour: float = Field(gt=0)
    samples: int = Field(ge=1)
    machine_ids: list[str] = Field(default_factory=list)


ClarificationKind = Literal["choice", "range", "area"]
AnswerValue = str | float | bool


class ClarificationOption(CamelModel):
    value: str = Field(min_length=1, max_length=40)
    label: str = Field(min_length=1, max_length=80)


class Clarification(CamelModel):
    """A question the planner asks with a structured answer: a choice (chips), a range
    (slider) or an area choice (how the ground is picked). Never free text."""

    id: str = Field(min_length=1, max_length=40)
    question: str = Field(min_length=1, max_length=200)
    kind: ClarificationKind
    options: list[ClarificationOption] = Field(default_factory=list, max_length=8)
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str | None = Field(default=None, max_length=20)
    default: AnswerValue | None = None
    # Why the answer matters, shown under the question.
    why: str = Field(default="", max_length=200)


class PlanStep(CamelModel):
    """Every field explicit (no defaults) so the contract marks them required on read."""

    title: str = Field(min_length=1, max_length=200)
    detail: str = Field(max_length=1000)
    machine_ids: list[str]
    zone_id: str | None
    when: str = Field(max_length=120)
    # Schedule: offset from the plan start in days and duration in days (>= 1 for a real step).
    start_day: int = Field(ge=0)
    days: int = Field(ge=0)


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
    # What the estimate rests on (rates, hours per day, weather windows); shown with the numbers.
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)


class PlanDraftRequest(CamelModel):
    goal: str = Field(min_length=3, max_length=2000)
    project_name: str = Field(min_length=1, max_length=200)
    site_name: str | None = Field(default=None, max_length=200)
    machines: list[PlannerMachine] = Field(default_factory=list, max_length=200)
    zones: list[PlannerZone] = Field(default_factory=list, max_length=500)
    existing_plans: list[PlannerExistingPlan] = Field(default_factory=list, max_length=200)
    # Rates learned from the work log; the planner prefers them over its defaults.
    rates: list[PlannerRate] = Field(default_factory=list, max_length=100)
    # Operator pre-selections from the console (a selected zone, machines ticked in Fleet).
    preferred_zone_ids: list[str] = Field(default_factory=list)
    preferred_machine_ids: list[str] = Field(default_factory=list)
    # A follow-up instruction on a previous draft ("use three machines", "skip Z-21").
    refinement: str | None = Field(default=None, max_length=2000)
    # Answers to the planner's clarifications, by clarification id.
    answers: dict[str, AnswerValue] = Field(default_factory=dict)
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
    assumptions: list[str]
    risks: list[str]
    questions: list[str]
    clarifications: list[Clarification]
    source: PlannerSource
    model: str | None
    # Shown next to the draft: who drafted it and what it assumed.
    note: str


class PlannerStatus(CamelModel):
    configured: bool
    provider: PlannerSource
    model: str | None


class OutlinePoint(CamelModel):
    """A polygon vertex in normalized image coordinates: x right, y down, both 0..1."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class OutlineRequest(CamelModel):
    """One view of the map with a marked point; the model outlines the feature under it."""

    image: str = Field(min_length=64, max_length=4_000_000, description="JPEG or PNG, base64")
    width: int = Field(ge=64, le=4096)
    height: int = Field(ge=64, le=4096)
    point: OutlinePoint
    hint: str = Field(default="", max_length=300, description="The operator's goal, for context")


class Outline(CamelModel):
    points: list[OutlinePoint]
    label: str = Field(max_length=80)
    confidence: float = Field(ge=0, le=1)
    note: str = Field(default="", max_length=300)
    source: Literal["claude"] = "claude"
