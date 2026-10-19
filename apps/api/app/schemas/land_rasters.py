from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, model_validator

from app.schemas.base import CamelModel

Finite = Annotated[float, Field(allow_inf_nan=False)]


class VegetationPeriod(CamelModel):
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def bounded_window(self) -> VegetationPeriod:
        if not 0 <= (self.end_date - self.start_date).days <= 30:
            raise ValueError("Each vegetation observation window must contain 1 to 31 days.")
        if self.start_date < date(2015, 6, 27) or self.end_date > datetime.now(UTC).date():
            raise ValueError("Use observation dates from the Sentinel-2 era through today.")
        return self


class RasterRequest(CamelModel):
    dataset: Literal[
        "cop-dem-glo-30",
        "esa-worldcover-2021",
        "sentinel-2-ndvi",
        "sentinel-2-signature",
        "usgs-3dep-13",
    ] = "cop-dem-glo-30"
    resolution_m: Finite = Field(default=30, ge=10, le=1000)
    max_dimension: Literal[256, 512, 1024] = 512
    periods: list[VegetationPeriod] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def source_resolution(self) -> RasterRequest:
        if self.dataset == "cop-dem-glo-30" and self.resolution_m < 30:
            raise ValueError("Copernicus GLO-30 analysis requires at least 30 m grid spacing.")
        if self.dataset in ("sentinel-2-ndvi", "sentinel-2-signature"):
            if self.dataset == "sentinel-2-ndvi" and (not self.periods or self.resolution_m < 20):
                raise ValueError(
                    "Vegetation analysis needs dated windows and at least 20 m grid spacing."
                )
            # A signature compares cells across seasons: one window is not a signature, and
            # every window adds five bands (six windows is the 30-band output ceiling).
            if self.dataset == "sentinel-2-signature" and not 2 <= len(self.periods) <= 6:
                raise ValueError("Seasonal signatures need two to six dated windows.")
            for previous, following in pairwise(self.periods):
                if previous.end_date >= following.start_date:
                    raise ValueError(
                        "Vegetation windows must be chronological and must not overlap."
                    )
        elif self.periods:
            raise ValueError("Dated windows are only supported for Sentinel-2 vegetation analysis.")
        return self


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
    band: str | None = None
    scale: Finite | None = None
    offset: Finite | None = None
    catalog_sha256: str | None = None
    purpose: str | None = None


class RasterClass(CamelModel):
    code: int = Field(ge=1, le=255)
    label: str
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    cells: int = Field(ge=0)
    fraction: Finite | None = Field(ge=0, le=1)
    sampled_area_m2: Finite = Field(ge=0)


class RasterBand(CamelModel):
    # Up to 30: a six-window Sentinel-2 signature holds five bands per window.
    index: int = Field(ge=1, le=30)
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
    palette: Literal["viridis", "magma", "rdylgn", "categorical"]
    classes: list[RasterClass] = Field(default_factory=list, max_length=255)
    display_minimum: Finite | None = None
    display_maximum: Finite | None = None


class VegetationObservation(CamelModel):
    period: VegetationPeriod
    # The NDVI band of this window (a signature interleaves five bands per window).
    band: int = Field(ge=1, le=30)
    scene_id: str | None
    acquired_at: datetime | None
    candidate_count: int = Field(ge=0)
    catalog_truncated: bool
    quality_candidates_examined: int = Field(ge=0, le=3)
    valid_cells: int = Field(ge=0)
    coverage_fraction: Finite | None = Field(ge=0, le=1)
    mean_ndvi: Finite | None
    common_mean_ndvi: Finite | None
    quality_counts: dict[str, int]
    explanation: str


class VegetationTimeline(CamelModel):
    observations: list[VegetationObservation] = Field(min_length=1, max_length=6)
    common_cells: int = Field(ge=0)
    common_coverage_fraction: Finite | None = Field(ge=0, le=1)
    change_band: int | None = Field(default=None, ge=1, le=7)
    change_cells: int = Field(ge=0)
    mean_change: Finite | None


class SignatureFeature(CamelModel):
    """One band of a seasonal Sentinel-2 signature: which window and which measurement."""

    band: int = Field(ge=1, le=30)
    # Index of the dated window in request.periods and vegetation.observations.
    window: int = Field(ge=0, le=5)
    key: Literal["green", "red", "nir", "swir16", "ndvi"]
    name: str
    # Mean and spread over the common cells: the z-score each similarity uses, so a
    # comparison can be reproduced from the stored bands alone. None without common cells.
    common_mean: Finite | None
    common_standard_deviation: Finite | None


class SignatureInfo(CamelModel):
    features: list[SignatureFeature] = Field(min_length=10, max_length=30)
    # Cells valid in every feature of every window inside the boundary: the only cells
    # that have a full signature and can be compared.
    common_cells: int = Field(ge=0)
    common_coverage_fraction: Finite | None = Field(ge=0, le=1)
    standardization: str
    similarity: str


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
    bands: list[RasterBand] = Field(min_length=1, max_length=30)
    # A six-window signature cites six catalog searches, up to twelve quality masks and
    # twenty-four reflectance files.
    sources: list[RasterSource] = Field(max_length=64)
    warnings: list[str]
    downloaded_bytes: int = Field(ge=0)
    vegetation: VegetationTimeline | None = None
    signature: SignatureInfo | None = None


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
