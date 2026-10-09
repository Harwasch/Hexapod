from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas.base import CamelModel
from app.schemas.geojson import MapGeometry, Point
from app.schemas.land import BoundarySource


class CandidateRequest(CamelModel):
    point: Point
    kind: Literal["parcel", "line", "building", "point"]
    radius_m: float = Field(default=250, gt=0, le=2000, allow_inf_nan=False)


class LandCandidate(CamelModel):
    id: str = Field(min_length=1, max_length=200)
    label: str = Field(min_length=1, max_length=300)
    geometry: MapGeometry
    source: BoundarySource
    distance_m: float = Field(ge=0)
    properties: dict[str, str | float | None] = Field(default_factory=dict)


class CandidateResult(CamelModel):
    status: Literal["available", "empty", "uncovered", "unavailable"]
    message: str
    candidates: list[LandCandidate]
    truncated: bool = False


class SelectionInstruction(CamelModel):
    instruction: str = Field(min_length=1, max_length=5000)
    candidates: list[LandCandidate] = Field(max_length=50)
    selected_ids: list[str] = Field(default_factory=list, max_length=50)


class SelectionInterpretation(CamelModel):
    candidate_ids: list[str] = Field(default_factory=list, max_length=50)
    operation: Literal["select", "union", "corridor", "clarify"]
    width_m: float | None = Field(default=None, gt=0, le=100_000, allow_inf_nan=False)
    cap: Literal["round", "flat", "square"] = "round"
    explanation: str = Field(min_length=1, max_length=2000)
    question: str | None = Field(default=None, max_length=1000)
    provider: Literal["model", "local"] = "local"
