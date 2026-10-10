from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, model_validator

from app.schemas.base import CamelModel
from app.schemas.geojson import Footprint
from app.schemas.land import LandCreate

Finite = Annotated[float, Field(allow_inf_nan=False)]
Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class SolarHorizonPoint(CamelModel):
    azimuth_degrees: Finite = Field(ge=0, lt=360)
    elevation_degrees: Finite = Field(ge=0, le=90)


class SolarRequest(CamelModel):
    dataset: Literal["nasa-power-hourly-solar"] = "nasa-power-hourly-solar"
    array_zone: Footprint
    zone_basis: str = Field(min_length=1, max_length=2000)
    year: int = Field(ge=2001)
    module_area_m2: Finite = Field(gt=0, le=1e7)
    module_efficiency: Finite = Field(gt=0, le=0.5)
    tilt_degrees: Finite = Field(ge=0, le=85)
    azimuth_degrees: Finite = Field(ge=0, lt=360)
    mounting: Literal[
        "open_rack_glass_glass", "close_mount_glass_glass", "insulated_back_glass_polymer"
    ]
    temperature_coefficient: Finite = Field(ge=-0.01, le=0)
    dc_ac_ratio: Finite = Field(ge=0.5, le=2)
    inverter_efficiency: Finite = Field(ge=0.8, le=1)
    system_loss: Fraction
    additional_shade_loss: Fraction
    albedo: Fraction
    iam_b: Finite = Field(ge=0, le=0.2)
    horizon: list[SolarHorizonPoint] = Field(default_factory=list, max_length=360)
    horizon_basis: str = Field(min_length=1, max_length=2000)
    assumptions: str = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def bounded_geometry_and_period(self) -> SolarRequest:
        LandCreate.bounded_boundary(self.array_zone)
        if self.year >= datetime.now(UTC).year:
            raise ValueError("Choose a completed calendar year from 2001 onward.")
        if self.horizon:
            azimuths = sorted(point.azimuth_degrees for point in self.horizon)
            if len(azimuths) < 4 or len(set(azimuths)) != len(azimuths):
                raise ValueError("A horizon needs at least four unique azimuths around the site.")
            if (
                max(
                    b - a for a, b in zip(azimuths, [*azimuths[1:], azimuths[0] + 360], strict=True)
                )
                > 90
            ):
                raise ValueError(
                    "Horizon azimuths must cover the full circle with gaps no greater than 90 degrees."
                )
        return self


class SolarMonth(CamelModel):
    month: int = Field(ge=1, le=12)
    expected_hours: int
    valid_hours: int
    generation_kwh: Finite | None
    unshaded_generation_kwh: Finite | None
    plane_irradiation_kwh_m2: Finite | None
    unshaded_plane_irradiation_kwh_m2: Finite | None


class SolarMetadata(CamelModel):
    algorithm: str
    model_version: str
    source_url: HttpUrl
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_bytes: int
    source_header: dict[str, object]
    source_units: dict[str, str]
    source_etag: str | None
    source_last_modified: str | None
    retrieved_at: datetime
    attribution: str
    license: str
    documentation_url: HttpUrl
    longitude: Finite
    latitude: Finite
    source_elevation_m: Finite
    mapped_zone_area_m2: Finite
    capacity_kw_dc: Finite
    inverter_kw_ac: Finite
    expected_hours: int
    valid_hours: int
    complete_year: bool
    invalid_parameter_hours: dict[str, int]
    modeled_generation_kwh: Finite
    annual_generation_kwh: Finite | None
    unshaded_generation_kwh: Finite
    plane_irradiation_kwh_m2: Finite
    unshaded_plane_irradiation_kwh_m2: Finite
    horizon_sky_fraction: Fraction
    peak_ac_kw: Finite
    monthly: list[SolarMonth] = Field(min_length=12, max_length=12)
    limitations: list[str]


class SolarAssessmentRead(CamelModel):
    id: uuid.UUID
    land_id: uuid.UUID
    run_id: uuid.UUID
    boundary_revision: int
    request: SolarRequest
    metadata: SolarMetadata
    sha256: str
    byte_size: int
    created_at: datetime
    stale: bool


class SolarPreview(CamelModel):
    mapped_zone_area_m2: Finite
    capacity_kw_dc: Finite
    inverter_kw_ac: Finite
    longitude: Finite
    latitude: Finite
