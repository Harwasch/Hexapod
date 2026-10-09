from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator

from app.schemas.base import CamelModel
from app.schemas.research import MapFeature

Finite = Annotated[float, Field(allow_inf_nan=False)]
ViewName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]


class LandViewCamera(CamelModel):
    longitude: Finite = Field(ge=-180, le=180)
    latitude: Finite = Field(ge=-90, le=90)
    height: Finite = Field(ge=-12000, le=100_000_000)
    heading: Finite = Field(ge=-360, le=360)
    pitch: Finite = Field(ge=-90, le=90)
    roll: Finite = Field(ge=-360, le=360)


class LandViewRaster(CamelModel):
    id: uuid.UUID
    kind: Literal["raster", "archive-alignment"] = "raster"
    band: int = Field(default=1, ge=1, le=16)
    opacity: Finite = Field(default=0.8, ge=0, le=1)


class LandViewState(CamelModel):
    camera: LandViewCamera
    boundary_revision: int = Field(ge=1)
    section: Literal["discover", "records", "inventory", "ecology", "scenarios", "actions"] = (
        "discover"
    )
    investigation_id: uuid.UUID | None = None
    inventory_id: uuid.UUID | None = None
    survey_id: uuid.UUID | None = None
    solar_id: uuid.UUID | None = None
    inventory_visible: bool = True
    artifact_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)
    rasters: list[LandViewRaster] = Field(default_factory=list, max_length=2)

    @field_validator("rasters")
    @classmethod
    def unique_rasters(cls, value: list[LandViewRaster]) -> list[LandViewRaster]:
        if len({item.id for item in value}) != len(value):
            raise ValueError("A view supports one band per raster; use distinct raster layers.")
        return value


class LandViewCreate(CamelModel):
    name: ViewName
    request_key: uuid.UUID
    state: LandViewState


class LandViewRename(CamelModel):
    name: ViewName
    expected_revision: int = Field(ge=1)


class LandViewRead(CamelModel):
    id: uuid.UUID
    land_id: uuid.UUID
    name: str
    revision: int
    state: LandViewState
    created_at: datetime
    updated_at: datetime


class LandViewMap(CamelModel):
    id: uuid.UUID
    title: str
    features: list[MapFeature]


class LandViewRasterRead(LandViewRaster):
    categorical: bool = False
    bounds: list[float]
    attribution: str


class LandViewOpen(CamelModel):
    view: LandViewRead
    state: LandViewState
    maps: list[LandViewMap]
    rasters: list[LandViewRasterRead]
    warnings: list[str]
