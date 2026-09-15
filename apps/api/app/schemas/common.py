from __future__ import annotations

from datetime import datetime

from pydantic import Field, HttpUrl, model_validator

from app.schemas.base import CamelModel


class Attribution(CamelModel):
    """Credit that must remain visible while a dataset is shown."""

    text: str = Field(min_length=1, max_length=500)
    organization: str | None = Field(default=None, max_length=200)
    url: HttpUrl | None = None


class LicenseMetadata(CamelModel):
    name: str = Field(min_length=1, max_length=200)
    spdx_id: str | None = Field(default=None, max_length=64)
    url: HttpUrl | None = None
    requires_attribution: bool = True
    notes: str | None = Field(default=None, max_length=2000)


class Provenance(CamelModel):
    """Where a dataset came from and when."""

    source_organization: str | None = Field(default=None, max_length=200)
    source_url: HttpUrl | None = None
    published_at: datetime | None = None
    notes: str | None = Field(default=None, max_length=2000)


class TemporalExtent(CamelModel):
    start: datetime | None = None
    end: datetime | None = None

    @model_validator(mode="after")
    def _ordered(self) -> TemporalExtent:
        if self.start and self.end and self.end < self.start:
            raise ValueError("temporal extent end must not precede start")
        return self


class GeoPosition(CamelModel):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    height: float | None = None


class BoundingBox(CamelModel):
    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def _ordered(self) -> BoundingBox:
        if self.south > self.north:
            raise ValueError("south must be <= north")
        return self


class Problem(CamelModel):
    """RFC 9457-style error payload."""

    title: str
    status: int
    detail: str | None = None
    errors: list[dict[str, object]] | None = None
