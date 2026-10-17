"""Persisted plans: an approved draft with its lifecycle state and revision history."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.agent import Cadence, PlanEstimates, PlannerSource, PlanStep
from app.schemas.base import CamelModel
from app.schemas.geojson import Footprint
from app.schemas.scout import ScoutCapture

PlanStatus = Literal["scheduled", "dispatched", "paused", "done", "cancelled"]


class PlanArea(CamelModel):
    """Ground the console found for this plan: an area drawn or outlined there, a saved
    Land's boundary or feature (the fleet backend owns the other zones)."""

    id: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=120)
    footprint: Footprint
    # Where the outline came from ("© OpenStreetMap contributors", "Saved boundary, revision
    # 2"), shown wherever the area is drawn. The one provenance field areas carry.
    attribution: str | None = Field(default=None, max_length=300)
    # A scout area's capture parameters (a "Lift the fog" patch): kept so an approved plan can
    # still be exported as a flight route. In the plans' areas JSONB, so no migration.
    capture: ScoutCapture | None = None


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
    # Areas drawn in the console that this plan covers; zone_ids may reference them.
    areas: list[PlanArea] = Field(default_factory=list, max_length=50)
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


class WorkspaceMissionCreate(PlanCreate):
    """A plan approved inside a workspace: private, and optionally on one of its saved Lands."""

    land_id: uuid.UUID | None = None
    boundary_revision: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _revision_needs_land(self) -> WorkspaceMissionCreate:
        if self.boundary_revision is not None and self.land_id is None:
            raise ValueError("boundaryRevision pins a land's outline; give landId with it")
        return self


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
    areas: list[PlanArea]
    steps: list[PlanStep]
    estimates: PlanEstimates
    assumptions: list[str]
    risks: list[str]
    questions: list[str]
    source: PlannerSource
    model: str | None
    revision: int
    revisions: list[PlanRevisionRead]
    # The saved Land this plan is on and the boundary revision it was planned against.
    land_id: uuid.UUID | None
    boundary_revision: int | None
    # Held in a workspace (never listed by the public /plans routes).
    private: bool
    # The land action this mission was scheduled from; revise it there, not here.
    land_action_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime
