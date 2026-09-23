from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, HttpUrl, field_validator

from app.models.enums import Representation
from app.schemas.asset import AssetBase, AssetRead
from app.schemas.base import CamelModel
from app.schemas.bookmark import CameraBookmarkCreate, CameraBookmarkRead
from app.schemas.common import Attribution, GeoPosition, LicenseMetadata
from app.schemas.geojson import Footprint

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SiteQuality(CamelModel):
    resolution_description: str | None = Field(default=None, max_length=300)
    ground_sample_distance_m: float | None = Field(default=None, gt=0)


class SiteBase(CamelModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    boundary: Footprint
    centroid: GeoPosition | None = Field(
        default=None, description="Defaults to the boundary centroid when omitted."
    )
    thumbnail_url: HttpUrl | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    attribution: list[Attribution] = Field(default_factory=list)
    license: LicenseMetadata | None = None


class SiteCreate(SiteBase):
    slug: str | None = Field(default=None, max_length=120)
    assets: list[AssetBase] = Field(default_factory=list)
    camera_bookmarks: list[CameraBookmarkCreate] = Field(default_factory=list)

    @field_validator("slug")
    @classmethod
    def _slug(cls, value: str | None) -> str | None:
        if value is not None and not SLUG_RE.match(value):
            raise ValueError("slug must be lowercase letters, digits and single hyphens")
        return value


class SiteUpdate(CamelModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    boundary: Footprint | None = None
    centroid: GeoPosition | None = None
    thumbnail_url: HttpUrl | None = None
    metadata: dict[str, Any] | None = None
    attribution: list[Attribution] | None = None
    license: LicenseMetadata | None = None


class SiteSummary(CamelModel):
    id: uuid.UUID
    slug: str
    name: str
    description: str | None
    centroid: GeoPosition
    area_m2: float
    thumbnail_url: HttpUrl | None
    representations: list[Representation]
    latest_observed_at: datetime | None
    quality: SiteQuality | None
    created_at: datetime
    updated_at: datetime


class SiteRead(CamelModel):
    """Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required."""

    id: uuid.UUID
    slug: str
    name: str
    description: str | None
    boundary: Footprint
    centroid: GeoPosition
    area_m2: float
    thumbnail_url: HttpUrl | None
    metadata: dict[str, Any]
    attribution: list[Attribution]
    license: LicenseMetadata | None
    assets: list[AssetRead]
    camera_bookmarks: list[CameraBookmarkRead]
    created_at: datetime
    updated_at: datetime
