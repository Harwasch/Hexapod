"""Field observations preserve sampling effort and do not infer unrecorded species absences."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.schemas.base import CamelModel
from app.schemas.geojson import Footprint
from app.schemas.land import LandCreate

Percent = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
Stratum = Literal["canopy", "shrub", "herb", "ground", "aquatic"]


class SpeciesObservation(CamelModel):
    taxon: str = Field(min_length=1, max_length=200)
    stratum: Stratum
    identification: Literal["verified", "tentative", "unidentified"]
    percent_cover: Percent | None = None
    hits: int | None = Field(default=None, ge=0, le=100000)
    notes: str = Field(default="", max_length=1000)


class SurveyPlot(CamelModel):
    label: str = Field(min_length=1, max_length=100)
    boundary: Footprint
    complete_inventory: bool = False
    sample_points: int | None = Field(default=None, ge=1, le=100000)
    observations: list[SpeciesObservation] = Field(max_length=200)

    @model_validator(mode="after")
    def validate_plot(self) -> SurveyPlot:
        LandCreate.bounded_boundary(self.boundary)
        keys = [(o.taxon.strip().casefold(), o.stratum) for o in self.observations]
        if len(set(keys)) != len(keys) or any(not name for name, _ in keys):
            raise ValueError(
                "Each taxon and stratum must occur once per plot with a nonblank name."
            )
        return self


class SurveyCreate(CamelModel):
    request_key: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str = Field(min_length=1, max_length=200)
    boundary_revision: int = Field(ge=1)
    observed_on: date
    observer: str = Field(min_length=1, max_length=200)
    method: Literal["visual-cover", "point-intercept"]
    design: Literal["census", "random", "systematic", "purposive"]
    assessed_strata: list[Stratum] = Field(min_length=1, max_length=5)
    method_notes: str = Field(min_length=1, max_length=5000)
    plots: list[SurveyPlot] = Field(min_length=1, max_length=100)
    supersedes_id: uuid.UUID | None = None

    @model_validator(mode="after")
    def consistent_effort(self) -> SurveyCreate:
        if self.observed_on > datetime.now(UTC).date():
            raise ValueError("A field observation cannot be dated in the future.")
        if len(set(self.assessed_strata)) != len(self.assessed_strata):
            raise ValueError("Assessed strata must be unique.")
        if len({p.label.strip().casefold() for p in self.plots}) != len(self.plots):
            raise ValueError("Plot labels must be unique.")
        if sum(len(p.observations) for p in self.plots) > 2000:
            raise ValueError("A survey supports at most 2,000 species observations.")
        vertices = 0
        for plot in self.plots:
            polygons = (
                [plot.boundary.coordinates]
                if plot.boundary.type == "Polygon"
                else plot.boundary.coordinates
            )
            vertices += sum(len(r) for p in polygons for r in p)
            if not plot.label.strip():
                raise ValueError("Plot labels must not be blank.")
            if self.method == "point-intercept" and plot.sample_points is None:
                raise ValueError("Point-intercept plots require the number of sampled points.")
            if self.method == "visual-cover" and plot.sample_points is not None:
                raise ValueError("Visual cover does not use point-intercept denominators.")
            for observation in plot.observations:
                if observation.stratum not in self.assessed_strata:
                    raise ValueError("Every observation must belong to an assessed stratum.")
                if self.method == "visual-cover":
                    if observation.percent_cover is None or observation.hits is not None:
                        raise ValueError(
                            "Visual observations require percent cover and no hit count."
                        )
                elif (
                    observation.percent_cover is not None
                    or observation.hits is None
                    or observation.hits > (plot.sample_points or 0)
                ):
                    raise ValueError(
                        "Point-intercept hits must be between zero and the sampled point count."
                    )
        if vertices > 20000:
            raise ValueError("A survey supports at most 20,000 total plot vertices.")
        return self


class SurveySpeciesSummary(CamelModel):
    taxon: str
    stratum: Stratum
    identifications: list[str]
    mean_percent: Percent
    minimum_percent: Percent
    maximum_percent: Percent
    measured_plots: int
    assessed_area_m2: float
    assessed_sample_fraction: float


class SurveySummary(CamelModel):
    algorithm: str = "field-cover-area-weighted-v1"
    land_area_m2: float
    sampled_area_m2: float
    sampled_fraction: float
    plot_areas_m2: list[float]
    species: list[SurveySpeciesSummary]
    limitations: list[str]


class SurveyRead(SurveyCreate):
    id: uuid.UUID
    land_id: uuid.UUID
    summary: SurveySummary
    sha256: str
    recorded_by: str
    created_at: datetime
    stale: bool


class SurveyLocator(CamelModel):
    land_id: uuid.UUID
    survey_id: uuid.UUID
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
