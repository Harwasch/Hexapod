from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, HttpUrl

from app.schemas.base import CamelModel

Finite = Annotated[float, Field(allow_inf_nan=False)]


class RasterRequest(CamelModel):
    dataset: Literal["cop-dem-glo-30"] = "cop-dem-glo-30"
    resolution_m: Finite = Field(default=30, ge=30, le=1000)
    max_dimension: Literal[256, 512, 1024] = 512


class RasterSource(CamelModel):
    id: str
    url: HttpUrl
    catalog_url: HttpUrl
    etag: str | None
    last_modified: str | None
    catalog_datetime: str | None
    observation_period: str
    license: str
    license_url: HttpUrl
    attribution: str


class RasterBand(CamelModel):
    index: int = Field(ge=1, le=16)
    name: str
    unit: str
    valid_cells: int = Field(ge=0)
    minimum: Finite | None
    maximum: Finite | None
    mean: Finite | None
    standard_deviation: Finite | None
    percentiles: dict[str, Finite]
    histogram_edges: list[Finite]
    histogram_counts: list[int]
    palette: Literal["viridis", "magma"]


class RasterMetadata(CamelModel):
    algorithm: str
    method: str
    bounds: Annotated[list[Finite], Field(min_length=4, max_length=4)]
    crs: str
    resolution_m: Finite = Field(gt=0)
    width: int = Field(gt=0, le=1024)
    height: int = Field(gt=0, le=1024)
    boundary_cells: int = Field(ge=0)
    valid_cells: int = Field(ge=0)
    coverage_fraction: Finite | None = Field(ge=0, le=1)
    sampled_area_m2: Finite = Field(ge=0)
    bands: list[RasterBand] = Field(min_length=1, max_length=16)
    sources: list[RasterSource] = Field(max_length=8)
    warnings: list[str]
    downloaded_bytes: int = Field(ge=0)


class LandRasterRead(CamelModel):
    id: uuid.UUID
    run_id: uuid.UUID
    land_id: uuid.UUID
    boundary_revision: int
    request: RasterRequest
    metadata: RasterMetadata
    sha256: str
    byte_size: int
    created_at: datetime
    stale: bool


class RasterSample(CamelModel):
    longitude: Finite
    latitude: Finite
    values: list[Finite | None]
    units: list[str]
    resolution_m: Finite
    interpretation: str
