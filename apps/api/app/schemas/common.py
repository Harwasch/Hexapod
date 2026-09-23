from __future__ import annotations

from datetime import datetime

from pydantic import Field, HttpUrl, model_validator

from app.models.enums import GeorefMethod, ScaleSource
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
    """Where a dataset came from, when, and how well it is known to be where it is.

    The last three fields are the capture's georeference provenance, and they are here
    rather than on a new schema of their own because this is already the "where did this
    come from" block, it is already carried through `render_config["provenance"]` for
    every asset and layer, and it is already the one thing the inspector reads. A capture
    placed by hand at plus or minus ten metres and one aligned to EXIF GPS differ only in
    these three values, and the console has to be able to say so.

    All three are optional and default to None, which means *not recorded* -- not
    `none`, not `unresolved`, and not zero. Everything that predates the pipeline
    (`legacy_captures.json`, the seeded reference layers) has no georeference provenance
    at all, and inventing `GeorefMethod.NONE` for it would be a claim nobody made.
    """

    source_organization: str | None = Field(default=None, max_length=200)
    source_url: HttpUrl | None = None
    published_at: datetime | None = None
    notes: str | None = Field(default=None, max_length=2000)
    georef_method: GeorefMethod | None = None
    scale_source: ScaleSource | None = None
    #: Horizontal placement uncertainty, in metres. Never zero for a hand placement:
    #: `manual_placement` defaults it to ten metres precisely so that the inspector
    #: cannot read a dropped pin as a survey.
    uncertainty_m: float | None = Field(default=None, ge=0)


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
