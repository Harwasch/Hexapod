"""Land belongs to an investigation, independently of any mission or capture."""

from __future__ import annotations

import uuid
from datetime import datetime
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, field_validator

from app.schemas.base import CamelModel
from app.schemas.geojson import Footprint, Position, _check_position


class BoundarySource(CamelModel):
    method: Literal["drawn", "imported", "parcel", "mapped-feature", "imagery", "corridor"]
    label: str = Field(min_length=1, max_length=300)
    url: HttpUrl | None = None
    record_id: str | None = Field(default=None, max_length=300)
    observed_at: datetime | None = None
    attribution: str | None = Field(default=None, max_length=2000)
    # This describes what the boundary represents, never a claim of verified title.
    meaning: Literal["study-area", "recorded-parcel", "physical-feature"] = "study-area"


class LandCreate(CamelModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=5000)
    boundary: Footprint
    source: BoundarySource

    @field_validator("boundary")
    @classmethod
    def bounded_boundary(cls, boundary: Footprint) -> Footprint:
        polygons = [boundary.coordinates] if boundary.type == "Polygon" else boundary.coordinates
        if sum(len(ring) for polygon in polygons for ring in polygon) > 20_000:
            raise ValueError("a land boundary supports at most 20,000 vertices")
        for polygon in polygons:
            for ring in polygon:
                if any(abs(a[0] - b[0]) > 180 for a, b in pairwise(ring)):
                    raise ValueError(
                        "split boundaries crossing the antimeridian into a MultiPolygon"
                    )
        return boundary


class LandRevise(LandCreate):
    expected_revision: int = Field(ge=1)
    note: str = Field(default="Boundary revised", min_length=1, max_length=500)


class BoundaryRevisionRead(CamelModel):
    revision: int
    boundary: Footprint
    source: BoundarySource
    note: str
    created_at: datetime


class LandRead(CamelModel):
    id: uuid.UUID
    name: str
    description: str
    boundary: Footprint
    source: BoundarySource
    revision: int
    area_m2: float
    perimeter_m: float
    created_at: datetime
    updated_at: datetime


class CorridorRequest(CamelModel):
    """Total corridor width, not distance on each side of the centreline."""

    coordinates: Annotated[list[Position], Field(min_length=2, max_length=5000)]
    width_m: float = Field(gt=0, le=100_000, allow_inf_nan=False)
    cap: Literal["round", "flat", "square"] = "round"

    @field_validator("coordinates")
    @classmethod
    def valid_line(cls, points: list[list[float]]) -> list[list[float]]:
        for point in points:
            _check_position(point)
        if len({tuple(point[:2]) for point in points}) < 2:
            raise ValueError("a corridor needs at least two distinct points")
        return points


class BoundaryOperation(CamelModel):
    operation: Literal["union", "difference", "intersection"]
    left: Footprint
    right: Footprint


class BoundaryResult(CamelModel):
    boundary: Footprint
    area_m2: float
    perimeter_m: float


class BoundarySplit(CamelModel):
    boundary: Footprint
    coordinates: Annotated[list[Position], Field(min_length=2, max_length=2000)]
    keep_parts: list[Annotated[int, Field(ge=0, lt=100)]] | None = Field(
        default=None, min_length=1, max_length=100
    )

    @field_validator("boundary")
    @classmethod
    def bounded_boundary(cls, boundary: Footprint) -> Footprint:
        return LandCreate.bounded_boundary(boundary)

    @field_validator("coordinates")
    @classmethod
    def valid_line(cls, points: list[list[float]]) -> list[list[float]]:
        CorridorRequest.valid_line(points)
        if any(abs(a[0] - b[0]) > 180 for a, b in pairwise(points)):
            raise ValueError("split antimeridian-crossing cuts into separate operations")
        return points


class BoundarySplitResult(CamelModel):
    parts: list[BoundaryResult]
    selection: BoundaryResult | None = None
