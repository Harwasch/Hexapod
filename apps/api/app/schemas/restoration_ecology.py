from __future__ import annotations

import uuid
from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.base import CamelModel
from app.schemas.land_surveys import Percent, Stratum


class CoverResponse(CamelModel):
    """An explicit what-if response curve, not an automatically fitted ecological model."""

    start_year: int = Field(ge=0, le=50)
    asymptote_low: Percent
    asymptote_high: Percent
    annual_rate_low: float = Field(ge=0, le=5, allow_inf_nan=False)
    annual_rate_high: float = Field(ge=0, le=5, allow_inf_nan=False)
    basis: str = Field(min_length=1, max_length=3000)
    evidence_ids: list[uuid.UUID] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def ordered_ranges(self) -> CoverResponse:
        if self.asymptote_low > self.asymptote_high or self.annual_rate_low > self.annual_rate_high:
            raise ValueError("Response parameter lower bounds cannot exceed upper bounds.")
        return self


class SpeciesTarget(CamelModel):
    taxon: str = Field(min_length=1, max_length=200)
    stratum: Stratum
    baseline_survey_id: uuid.UUID | None = None
    baseline_percent: Percent | None = None
    baseline_basis: str = Field(min_length=1, max_length=2000)
    target_low: Percent
    target_high: Percent
    target_year: int = Field(ge=1, le=50)
    target_basis: Literal["hypothetical", "reference-evidence"] = "hypothetical"
    rationale: str = Field(min_length=1, max_length=3000)
    evidence_ids: list[uuid.UUID] = Field(default_factory=list, max_length=10)
    treatment_names: list[str] = Field(default_factory=list, max_length=20)
    monitoring_method: str = Field(min_length=1, max_length=2000)
    monitoring_season: str = Field(min_length=1, max_length=500)
    response_if_off_track: str = Field(min_length=1, max_length=2000)
    response: CoverResponse | None = None

    @model_validator(mode="after")
    def consistent_target(self) -> SpeciesTarget:
        if self.target_low > self.target_high:
            raise ValueError("Target lower cover cannot exceed upper cover.")
        if self.baseline_survey_id is not None and self.baseline_percent is not None:
            raise ValueError(
                "A survey baseline is read from its saved observations; omit manual cover."
            )
        if self.target_basis == "reference-evidence" and not self.evidence_ids:
            raise ValueError("Reference-backed targets need at least one evidence citation.")
        if self.response and self.response.start_year >= self.target_year:
            raise ValueError("The response start must precede the target assessment year.")
        if len(set(self.treatment_names)) != len(self.treatment_names):
            raise ValueError("A target cannot list the same treatment twice.")
        return self


class RestorationEcology(CamelModel):
    reference_basis: str = Field(min_length=1, max_length=4000)
    reference_evidence_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)
    site_constraints: str = Field(min_length=1, max_length=4000)
    species_targets: list[SpeciesTarget] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def bounded_references(self) -> RestorationEcology:
        keys = [(target.taxon.casefold(), target.stratum) for target in self.species_targets]
        if len(keys) != len(set(keys)):
            raise ValueError("Species targets must have distinct taxon and stratum combinations.")
        if len(self.evidence_ids()) > 100:
            raise ValueError("An ecological plan supports up to 100 distinct source citations.")
        if len({t.baseline_survey_id for t in self.species_targets if t.baseline_survey_id}) > 20:
            raise ValueError("An ecological plan supports up to 20 baseline surveys.")
        return self

    def evidence_ids(self) -> set[uuid.UUID]:
        return set(self.reference_evidence_ids) | {
            identifier
            for target in self.species_targets
            for identifier in [
                *target.evidence_ids,
                *(target.response.evidence_ids if target.response else []),
            ]
        }


class CoverProjection(CamelModel):
    year: int
    low: Percent
    high: Percent


class SpeciesTargetResult(CamelModel):
    target: SpeciesTarget
    baseline_percent: Percent | None
    baseline_scope: Literal["surveyed-plots", "entered-assumption", "unknown"]
    survey_sha256: str | None = None
    observed_on: date | None = None
    assessed_area_m2: float | None = None
    assessed_land_fraction: float | None = None
    identification_status: list[str] = Field(default_factory=list)
    projection: list[CoverProjection] = Field(default_factory=list, max_length=51)
    target_relation: Literal["not-modeled", "inside-target", "overlaps-target", "outside-target"]
    limitations: list[str]


class RestorationEcologyResult(CamelModel):
    algorithm: str = "restoration-species-targets-and-conditional-response/1"
    reference_basis: str
    reference_evidence_ids: list[uuid.UUID]
    site_constraints: str
    species: list[SpeciesTargetResult]
    limitations: list[str]
