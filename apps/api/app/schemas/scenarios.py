from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.schemas.base import CamelModel

Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Money = Annotated[float, Field(ge=0, le=1e12, allow_inf_nan=False)]


class SolarInputs(CamelModel):
    kind: Literal["solar"]
    usable_roof_area_m2: float = Field(gt=0, le=1e8, allow_inf_nan=False)
    module_efficiency: float = Field(gt=0, le=0.5, allow_inf_nan=False)
    annual_plane_irradiation_kwh_m2: float = Field(ge=0, le=5000, allow_inf_nan=False)
    irradiation_basis: str = Field(min_length=1, max_length=2000)
    # Plane-of-array resource already incorporates tilt/orientation, separately
    # recorded so a regional horizontal resource is not mislabeled roof-specific.
    tilt_degrees: float = Field(ge=0, le=90, allow_inf_nan=False)
    azimuth_degrees: float = Field(ge=0, lt=360, allow_inf_nan=False)
    shade_loss: Fraction
    system_loss: Fraction
    degradation_per_year: float = Field(ge=0, le=0.1, allow_inf_nan=False)
    self_consumption_fraction: Fraction
    purchase_rate_per_kwh: Money
    export_rate_per_kwh: Money
    tariff_escalation: float = Field(ge=-0.1, le=0.2, allow_inf_nan=False)
    installed_cost: Money
    upfront_incentive: Money
    annual_maintenance_cost: Money
    maintenance_escalation: float = Field(ge=-0.1, le=0.2, allow_inf_nan=False)
    discount_rate: float = Field(ge=0, le=0.5, allow_inf_nan=False)
    financed_fraction: Fraction
    loan_interest_rate: float = Field(ge=0, le=0.5, allow_inf_nan=False)
    loan_years: int = Field(ge=1, le=40)
    years: int = Field(ge=1, le=40)
    replacement_year: int | None = Field(default=None, ge=1, le=40)
    replacement_cost: Money = 0
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    assumptions: str = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def sensible_financing(self) -> SolarInputs:
        if self.upfront_incentive > self.installed_cost:
            raise ValueError("upfront incentive cannot exceed installed cost")
        if self.financed_fraction and self.loan_years > self.years:
            raise ValueError("model at least the full loan term")
        if self.replacement_year and self.replacement_year > self.years:
            raise ValueError("replacement year must be within the modeled period")
        return self


class CoverClass(CamelModel):
    name: str = Field(min_length=1, max_length=200)
    baseline_percent: float = Field(ge=0, le=100, allow_inf_nan=False)
    target_percent: float = Field(ge=0, le=100, allow_inf_nan=False)
    evidence_basis: str = Field(min_length=1, max_length=2000)


class RestorationTreatment(CamelModel):
    name: str = Field(min_length=1, max_length=200)
    area_ha: float = Field(gt=0, le=1e8, allow_inf_nan=False)
    cost_per_ha: Money
    year: int = Field(ge=0, le=50)
    objective: str = Field(min_length=1, max_length=2000)


class RestorationInputs(CamelModel):
    kind: Literal["restoration"]
    reference_ecosystem: str = Field(min_length=1, max_length=2000)
    survey_date: date
    survey_method: str = Field(min_length=1, max_length=2000)
    confidence: Literal["field-survey", "remote-estimate", "user-estimate"]
    cover: list[CoverClass] = Field(min_length=1, max_length=100)
    treatments: list[RestorationTreatment] = Field(max_length=200)
    monitoring_years: list[int] = Field(min_length=1, max_length=30)
    monitoring_cost_per_visit: Money
    contingency_fraction: Fraction
    discount_rate: float = Field(ge=0, le=0.5, allow_inf_nan=False)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    assumptions: str = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def cover_and_schedule(self) -> RestorationInputs:
        if sum(row.baseline_percent for row in self.cover) > 100.000001:
            raise ValueError(
                "baseline cover classes must be mutually exclusive and total at most 100%"
            )
        if abs(sum(row.target_percent for row in self.cover) - 100) > 0.000001:
            raise ValueError("target cover classes must total 100%")
        if len({row.name.casefold() for row in self.cover}) != len(self.cover):
            raise ValueError("cover class names must be unique")
        if len(set(self.monitoring_years)) != len(self.monitoring_years) or any(
            year < 0 or year > 50 for year in self.monitoring_years
        ):
            raise ValueError("monitoring years must be unique values between 0 and 50")
        return self


ScenarioInputs = Annotated[SolarInputs | RestorationInputs, Field(discriminator="kind")]


class ScenarioCreate(CamelModel):
    request_key: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str = Field(min_length=1, max_length=200)
    boundary_revision: int = Field(ge=1)
    inputs: ScenarioInputs
    evidence_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)
    solar_assessment_id: uuid.UUID | None = None
    field_survey_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)


class ScenarioRevise(ScenarioCreate):
    expected_revision: int = Field(ge=1)


class ScenarioResult(CamelModel):
    algorithm: str
    summary: dict[str, float | int | str | None]
    rows: list[dict[str, float | int | str | None]]
    sensitivity: list[dict[str, float | str]] = Field(default_factory=list)
    limitations: list[str]


class ScenarioRead(ScenarioCreate):
    id: uuid.UUID
    land_id: uuid.UUID
    revision: int
    stale: bool
    result: ScenarioResult
    created_at: datetime
    updated_at: datetime
