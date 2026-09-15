from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, model_validator

from app.models.enums import AssetProvider, Representation
from app.schemas.base import CamelModel
from app.schemas.common import Attribution, LicenseMetadata, Provenance
from app.schemas.geojson import Footprint


class CesiumIonSource(CamelModel):
    type: Literal["cesium-ion"] = "cesium-ion"
    asset_id: int = Field(gt=0)


class TilesUrlSource(CamelModel):
    type: Literal["3d-tiles-url"] = "3d-tiles-url"
    url: HttpUrl


AssetSource = Annotated[CesiumIonSource | TilesUrlSource, Field(discriminator="type")]


def provider_for_source(source: CesiumIonSource | TilesUrlSource) -> AssetProvider:
    return AssetProvider.CESIUM_ION if source.type == "cesium-ion" else AssetProvider.TILES_3D_URL


class ResolutionMetadata(CamelModel):
    ground_sample_distance_m: float | None = Field(default=None, gt=0)
    point_spacing_m: float | None = Field(default=None, gt=0)
    description: str | None = Field(default=None, max_length=300)


class CrsMetadata(CamelModel):
    horizontal: str | None = Field(default=None, max_length=120)
    vertical: str | None = Field(default=None, max_length=120)
    notes: str | None = Field(default=None, max_length=500)


class PointCloudShading(CamelModel):
    attenuation: bool = True
    eye_dome_lighting: bool = True
    maximum_attenuation: float | None = Field(default=None, gt=0)


class RenderConfig(CamelModel):
    maximum_screen_space_error: float | None = Field(default=None, gt=0, le=512)
    point_cloud_shading: PointCloudShading | None = None
    # When true, the asset's footprint clips the globe/terrain and the global 3D world.
    clips_world: bool = True
    height_offset_m: float = 0.0


class AssetBase(CamelModel):
    name: str = Field(min_length=1, max_length=200)
    representation: Representation
    source: AssetSource
    footprint: Footprint | None = None
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    resolution: ResolutionMetadata | None = None
    crs: CrsMetadata | None = None
    license: LicenseMetadata | None = None
    attribution: list[Attribution] = Field(default_factory=list)
    provenance: Provenance | None = None
    render_config: RenderConfig = Field(default_factory=RenderConfig)
    default_visible: bool = False

    @model_validator(mode="after")
    def _validity_window(self) -> AssetBase:
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("validTo must not precede validFrom")
        return self


class AssetCreate(AssetBase):
    site_id: uuid.UUID | None = None


class AssetUpdate(CamelModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    resolution: ResolutionMetadata | None = None
    license: LicenseMetadata | None = None
    attribution: list[Attribution] | None = None
    render_config: RenderConfig | None = None
    default_visible: bool | None = None
    footprint: Footprint | None = None


class AssetRead(CamelModel):
    """Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required."""

    id: uuid.UUID
    site_id: uuid.UUID | None
    provider: AssetProvider
    name: str
    representation: Representation
    source: AssetSource
    footprint: Footprint | None
    observed_at: datetime | None
    valid_from: datetime | None
    valid_to: datetime | None
    resolution: ResolutionMetadata | None
    crs: CrsMetadata | None
    license: LicenseMetadata | None
    attribution: list[Attribution]
    provenance: Provenance | None
    render_config: RenderConfig
    default_visible: bool
    created_at: datetime
    updated_at: datetime
