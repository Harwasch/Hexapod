from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Annotated, Literal

from pydantic import Field, model_validator
from shapely.geometry import shape

from app.schemas.base import CamelModel
from app.schemas.geojson import Footprint


class VersionReference(CamelModel):
    id: uuid.UUID
    revision: int = Field(ge=1)


class ActionConstraint(CamelModel):
    text: str = Field(min_length=1, max_length=2000)
    resolved: bool = False
    resolution: str = Field(default="", max_length=2000)
    evidence_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def explain_resolution(self) -> ActionConstraint:
        if self.resolved and not self.resolution.strip():
            raise ValueError("a resolved constraint needs its resolution recorded")
        return self


class LandActionStep(CamelModel):
    id: str = Field(min_length=1, max_length=32, pattern=r"^[a-zA-Z0-9_-]+$")
    title: str = Field(min_length=1, max_length=200)
    detail: str = Field(default="", max_length=1000)
    start_day: int = Field(ge=0, le=36500)
    days: int = Field(ge=1, le=3650)
    resources: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=50
    )
    machine_ids: list[Annotated[str, Field(min_length=1, max_length=40)]] = Field(
        default_factory=list, max_length=100
    )
    estimated_cost: float | None = Field(default=None, ge=0, le=1e12, allow_inf_nan=False)
    cost_basis: str = Field(default="", max_length=2000)
    depends_on: list[str] = Field(default_factory=list, max_length=100)
    footprint: Footprint | None = None
    success_measure: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def measured_estimate(self) -> LandActionStep:
        if self.estimated_cost is not None and not self.cost_basis.strip():
            raise ValueError("a cost estimate needs its basis")
        return self


class LandActionCreate(CamelModel):
    request_key: uuid.UUID = Field(default_factory=uuid.uuid4)
    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=2000)
    boundary_revision: int = Field(ge=1)
    scenario: VersionReference | None = None
    features: list[VersionReference] = Field(default_factory=list, max_length=200)
    evidence_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)
    start_date: date
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    steps: list[LandActionStep] = Field(min_length=1, max_length=250)
    constraints: list[ActionConstraint] = Field(default_factory=list, max_length=100)
    exclusions: list[Footprint] = Field(default_factory=list, max_length=50)
    assumptions: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def schedule_graph(self) -> LandActionCreate:
        by_id = {step.id: step for step in self.steps}
        if len(by_id) != len(self.steps):
            raise ValueError("action step identifiers must be unique")
        for step in self.steps:
            for dependency in step.depends_on:
                before = by_id.get(dependency)
                if before is None or before.id == step.id:
                    raise ValueError("every dependency must reference another action step")
                if before.start_day + before.days > step.start_day:
                    raise ValueError("dependent work must begin after its prerequisites finish")
        if sum(step.footprint is not None for step in self.steps) > 49:
            raise ValueError("an action supports up to 49 distinct step footprints")
        # Positive durations and increasing start times also rule out dependency cycles.
        from shapely import get_num_coordinates

        geometries = self.exclusions + [step.footprint for step in self.steps if step.footprint]
        if sum(get_num_coordinates(shape(item.model_dump())) for item in geometries) > 20000:
            raise ValueError("action footprints are limited to 20,000 total vertices")
        if any(not shape(item.model_dump()).is_valid for item in geometries):
            raise ValueError("action footprints must be valid polygons")
        if any(len(text) > 2000 for text in self.assumptions):
            raise ValueError("each assumption must be at most 2,000 characters")
        try:
            self.start_date + timedelta(days=max(step.start_day + step.days for step in self.steps))
        except OverflowError as error:
            raise ValueError("action schedule exceeds the supported date range") from error
        return self


class LandActionRevise(LandActionCreate):
    expected_revision: int = Field(ge=1)
    note: str = Field(min_length=1, max_length=1000)


class LandActionReview(CamelModel):
    expected_revision: int = Field(ge=1)
    note: str = Field(min_length=1, max_length=2000)


class LandActionRead(LandActionCreate):
    id: uuid.UUID
    land_id: uuid.UUID
    revision: int
    status: Literal["draft", "approved", "scheduled"]
    stale_reasons: list[str]
    total_known_cost: float
    uncosted_steps: int
    effective_boundary: Footprint
    approved_by: str | None
    approved_at: datetime | None
    approval_note: str | None
    mission_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


class ActionMissionCreate(LandActionReview):
    project_id: str = Field(min_length=1, max_length=120)
