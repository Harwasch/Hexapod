from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.schemas.base import CamelModel
from app.schemas.geojson import Polygon

Finite = Annotated[float, Field(allow_inf_nan=False)]


class ImageControlPoint(CamelModel):
    label: str = Field(min_length=1, max_length=100)
    image_x: Finite = Field(ge=0, le=8192)
    image_y: Finite = Field(ge=0, le=8192)
    longitude: Finite = Field(ge=-180, le=180)
    latitude: Finite = Field(ge=-90, le=90)


class ImageRegistrationCreate(CamelModel):
    request_key: uuid.UUID
    name: str = Field(min_length=1, max_length=200)
    image_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    points: list[ImageControlPoint] = Field(min_length=4, max_length=30)
    notes: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def unique_points(self) -> ImageRegistrationCreate:
        for values in (
            [(point.image_x, point.image_y) for point in self.points],
            [(point.longitude, point.latitude) for point in self.points],
            [point.label for point in self.points],
        ):
            if len(set(values)) != len(values):
                raise ValueError(
                    "Each control point needs a distinct image position, map position and label."
                )
        return self


class ControlPointFit(CamelModel):
    label: str
    predicted_longitude: Finite
    predicted_latitude: Finite
    error_m: Finite = Field(ge=0)
    leave_one_out_error_m: Finite | None = Field(ge=0)


class ImageRegistrationResult(CamelModel):
    algorithm: Literal["affine-aeqd-v1"] = "affine-aeqd-v1"
    crs: str
    transform: Annotated[list[Finite], Field(min_length=6, max_length=6)]
    image_width: int
    image_height: int
    footprint: Polygon
    bounds: Annotated[list[Finite], Field(min_length=4, max_length=4)]
    rms_error_m: Finite = Field(ge=0)
    maximum_error_m: Finite = Field(ge=0)
    leave_one_out_rms_m: Finite | None = Field(ge=0)
    control_point_coverage: Finite = Field(ge=0, le=1)
    approximate_m_per_pixel: Finite = Field(gt=0)
    points: list[ControlPointFit]
    warnings: list[str]


class ImageRegistrationRead(CamelModel):
    id: uuid.UUID
    evidence_id: uuid.UUID
    request: ImageRegistrationCreate
    result: ImageRegistrationResult
    sha256: str
    byte_size: int
    display_width: int
    display_height: int
    created_at: datetime
