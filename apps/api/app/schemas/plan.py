"""Persisted plans: an approved draft with its lifecycle state and revision history."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import Field

from app.schemas.agent import Cadence, PlanEstimates, PlannerSource, PlanStep
from app.schemas.base import CamelModel

PlanStatus = Literal["scheduled", "dispatched", "paused", "done", "cancelled"]


class PlanBody(CamelModel):
    """What an approval writes: the reviewed draft plus the goal it came from."""

    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=2000)
    goal: str = Field(min_length=1, max_length=2000)
    cadence: Cadence = "once"
    start_date: date
    end_date: date | None = None
    zone_ids: list[str] = Field(default_factory=list)
    machine_ids: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    estimates: PlanEstimates
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    source: PlannerSource = "rules"
    model: str | None = None


class PlanCreate(PlanBody):
    project_id: str = Field(min_length=1, max_length=120)
    site_id: uuid.UUID | None = None


class PlanRevise(PlanBody):
    """A new approved revision of an existing plan; the previous one is kept in history."""

    note: str = Field(default="", max_length=300)


class PlanStatusUpdate(CamelModel):
    status: PlanStatus


class PlanRevisionRead(CamelModel):
    revision: int
    note: str
    created_at: datetime
    title: str
    machine_ids: list[str]
    zone_ids: list[str]
    estimates: PlanEstimates


class PlanRead(CamelModel):
    """Read model: every field explicit so the contract marks it required."""

    id: uuid.UUID
    project_id: str
    site_id: uuid.UUID | None
    title: str
    objective: str
    goal: str
    status: PlanStatus
    cadence: Cadence
    start_date: date
    end_date: date | None
    zone_ids: list[str]
    machine_ids: list[str]
    steps: list[PlanStep]
    estimates: PlanEstimates
    assumptions: list[str]
    risks: list[str]
    questions: list[str]
    source: PlannerSource
    model: str | None
    revision: int
    revisions: list[PlanRevisionRead]
    created_at: datetime
    updated_at: datetime
