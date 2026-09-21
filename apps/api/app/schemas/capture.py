from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from app.models.enums import CaptureKind, CaptureStatus, GeorefMethod, ScaleSource, UploadStatus
from app.schemas.base import CamelModel
from app.schemas.common import Attribution, LicenseMetadata, Provenance, TemporalExtent

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class CaptureBase(CamelModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    kind: CaptureKind
    device: str | None = Field(default=None, max_length=200)
    sensor: str | None = Field(default=None, max_length=120)
    captured_at: datetime | None = None
    temporal_extent: TemporalExtent | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    attribution: list[Attribution] = Field(default_factory=list)
    license: LicenseMetadata | None = None
    provenance: Provenance | None = None


class CaptureCreate(CaptureBase):
    slug: str | None = Field(default=None, max_length=120)

    @field_validator("slug")
    @classmethod
    def _slug(cls, value: str | None) -> str | None:
        if value is not None and not SLUG_RE.match(value):
            raise ValueError("slug must be lowercase letters, digits and single hyphens")
        return value


class CaptureFileCreate(CamelModel):
    filename: str = Field(min_length=1, max_length=500)
    content_type: str | None = Field(default=None, max_length=200)
    bytes: int | None = Field(default=None, ge=0)


class CaptureFileRead(CamelModel):
    """Read model: every field is explicit (no defaults) so the OpenAPI contract marks it
    required."""

    id: uuid.UUID
    capture_id: uuid.UUID
    filename: str
    content_type: str | None
    bytes: int | None
    checksum: str | None
    storage_key: str
    status: UploadStatus
    upload_id: str | None
    parts_total: int | None
    parts_completed: int
    created_at: datetime
    updated_at: datetime


class CaptureRead(CamelModel):
    id: uuid.UUID
    slug: str
    name: str
    description: str | None
    status: CaptureStatus
    kind: CaptureKind
    device: str | None
    sensor: str | None
    captured_at: datetime | None
    temporal_extent: TemporalExtent | None
    # How it was placed on the globe and how well, per the plan's B4. Null until a
    # georeference stage has run.
    georef_method: GeorefMethod | None
    scale_source: ScaleSource | None
    uncertainty_m: float | None
    metadata: dict[str, Any]
    attribution: list[Attribution]
    license: LicenseMetadata | None
    provenance: Provenance | None
    site_id: uuid.UUID | None
    files: list[CaptureFileRead]
    created_at: datetime
    updated_at: datetime
