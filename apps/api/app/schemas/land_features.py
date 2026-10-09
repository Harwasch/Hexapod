from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import Literal

from pydantic import Field, TypeAdapter, model_validator
from shapely import force_2d, get_num_coordinates
from shapely.geometry import mapping, shape

from app.schemas.base import CamelModel
from app.schemas.geojson import MapGeometry
from app.schemas.land import BoundarySource


class LandFeatureCreate(CamelModel):
    request_key: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str = Field(min_length=1, max_length=200)
    category: Literal["building", "power", "water", "transport", "equipment", "vegetation", "other"]
    geometry: MapGeometry
    source: BoundarySource
    status: Literal["candidate", "confirmed", "retired"] = "candidate"
    description: str = Field(default="", max_length=5000)
    attributes: dict[str, str | float | bool | None] = Field(default_factory=dict, max_length=100)
    evidence_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def bounded_feature(self) -> LandFeatureCreate:
        geometry = shape(self.geometry.model_dump())
        if geometry.is_empty or not geometry.is_valid or get_num_coordinates(geometry) > 20_000:
            raise ValueError(
                "an inventory feature needs valid geometry with at most 20,000 vertices"
            )
        if any(
            len(key) > 100 or (isinstance(value, str) and len(value) > 2000)
            for key, value in self.attributes.items()
        ):
            raise ValueError("attribute names and values exceed the inventory limits")
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in self.attributes.values()
        ):
            raise ValueError("numeric attributes must be finite")
        if geometry.has_z:
            self.geometry = TypeAdapter(MapGeometry).validate_python(mapping(force_2d(geometry)))
        return self


class LandFeatureRevise(LandFeatureCreate):
    expected_revision: int = Field(ge=1)
    note: str = Field(min_length=1, max_length=1000)


class LandFeatureRead(LandFeatureCreate):
    id: uuid.UUID
    land_id: uuid.UUID
    revision: int
    boundary_revision: int
    intersects_land: bool
    distance_m: float
    created_at: datetime
    updated_at: datetime


class LandFeatureRevisionRead(CamelModel):
    revision: int
    content: LandFeatureCreate
    note: str
    created_at: datetime


class FeatureInspectionCreate(CamelModel):
    request_key: uuid.UUID = Field(default_factory=uuid.uuid4)
    observed_at: datetime
    condition: Literal["unknown", "good", "fair", "poor", "critical"]
    notes: str = Field(min_length=1, max_length=10000)
    measurements: dict[str, float] = Field(default_factory=dict, max_length=100)
    measurement_units: dict[str, str] = Field(default_factory=dict, max_length=100)
    evidence_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def measured_units(self) -> FeatureInspectionCreate:
        if self.observed_at.tzinfo is None:
            raise ValueError("inspection time must include its timezone")
        if set(self.measurements) != set(self.measurement_units):
            raise ValueError("provide a unit for every measurement")
        if any(not math.isfinite(value) for value in self.measurements.values()):
            raise ValueError("inspection measurements must be finite")
        if any(not unit.strip() or len(unit) > 100 for unit in self.measurement_units.values()):
            raise ValueError("measurement units must be nonempty and at most 100 characters")
        return self


class FeatureInspectionRead(FeatureInspectionCreate):
    id: uuid.UUID
    feature_id: uuid.UUID
    feature_revision: int
    recorded_by: str
    created_at: datetime
