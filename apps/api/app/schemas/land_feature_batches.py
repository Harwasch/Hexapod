from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator
from shapely import get_num_coordinates
from shapely.geometry import shape

from app.schemas.base import CamelModel
from app.schemas.land_features import FeatureGeometryRead, LandFeatureCreate


class FeatureBatchItem(CamelModel):
    row_id: str = Field(min_length=1, max_length=200)
    feature: LandFeatureCreate


class FeatureBatchRequest(CamelModel):
    request_key: uuid.UUID
    boundary_revision: int = Field(ge=1)
    source_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_label: str = Field(min_length=1, max_length=200)
    skip_duplicates: bool = True
    rows: list[FeatureBatchItem] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def bounded_import(self) -> FeatureBatchRequest:
        if len({row.row_id for row in self.rows}) != len(self.rows):
            raise ValueError("Import row IDs must be unique")
        if any(row.feature.status != "candidate" for row in self.rows):
            raise ValueError("Imported assets start as candidates for identity review")
        if any(row.feature.external_ref is None for row in self.rows):
            raise ValueError("Each imported row needs a dataset namespace and external record ID")
        if (
            sum(get_num_coordinates(shape(row.feature.geometry.model_dump())) for row in self.rows)
            > 100_000
        ):
            raise ValueError("Import at most 100,000 total vertices per batch")
        if len(self.model_dump_json().encode()) > 8 * 1024 * 1024:
            raise ValueError("Import payload exceeds 8 MiB")
        return self


class FeatureBatchRow(CamelModel):
    row_id: str
    feature_id: uuid.UUID
    name: str
    disposition: Literal["created", "existing", "duplicate-in-file"]
    geometry_preview: FeatureGeometryRead | None = None


class FeatureBatchPreview(CamelModel):
    boundary_revision: int
    rows: list[FeatureBatchRow]


class FeatureBatchRead(CamelModel):
    id: uuid.UUID
    land_id: uuid.UUID
    boundary_revision: int
    request_key: uuid.UUID
    source_file_sha256: str
    source_label: str
    rows: list[FeatureBatchRow]
    created_at: datetime
