from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, field_validator

from app.models.enums import LayerCategory, LayerSourceType
from app.schemas.base import CamelModel
from app.schemas.common import (
    Attribution,
    BoundingBox,
    LicenseMetadata,
    Provenance,
    TemporalExtent,
)

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TEMPLATE_RE = re.compile(r"\{(x|y|z|s|reverseY)\}")


class CesiumIonTerrainSource(CamelModel):
    type: Literal["cesium-ion-terrain"] = "cesium-ion-terrain"
    asset_id: int = Field(gt=0)
    request_vertex_normals: bool = True
    request_water_mask: bool = False


class CesiumIonImagerySource(CamelModel):
    type: Literal["cesium-ion-imagery"] = "cesium-ion-imagery"
    asset_id: int = Field(gt=0)


class CesiumIon3DTilesSource(CamelModel):
    type: Literal["cesium-ion-3d-tiles"] = "cesium-ion-3d-tiles"
    asset_id: int = Field(gt=0)


class GooglePhotorealisticSource(CamelModel):
    type: Literal["google-photorealistic"] = "google-photorealistic"


class TilesUrlLayerSource(CamelModel):
    type: Literal["3d-tiles-url"] = "3d-tiles-url"
    url: HttpUrl


class XyzSource(CamelModel):
    type: Literal["xyz"] = "xyz"
    url_template: str = Field(min_length=8, max_length=2048)
    minimum_level: int = Field(default=0, ge=0, le=30)
    maximum_level: int | None = Field(default=None, ge=0, le=30)
    subdomains: list[str] | None = None
    tile_width: int = Field(default=256, ge=64, le=1024)

    @field_validator("url_template")
    @classmethod
    def _template(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("urlTemplate must be an http(s) URL")
        if not TEMPLATE_RE.search(value):
            raise ValueError("urlTemplate must contain {x}, {y} and {z} placeholders")
        for placeholder in ("{x}", "{y}", "{z}"):
            has_y_alias = placeholder == "{y}" and "{reverseY}" in value
            if placeholder not in value and not has_y_alias:
                raise ValueError(f"urlTemplate is missing {placeholder}")
        return value


class WmtsSource(CamelModel):
    type: Literal["wmts"] = "wmts"
    url: HttpUrl
    layer: str = Field(min_length=1, max_length=200)
    style: str = Field(default="default", max_length=200)
    format: str = Field(default="image/png", max_length=100)
    tile_matrix_set_id: str = Field(min_length=1, max_length=200)
    maximum_level: int | None = Field(default=None, ge=0, le=30)


class WmsSource(CamelModel):
    type: Literal["wms"] = "wms"
    url: HttpUrl
    layers: str = Field(min_length=1, max_length=500)
    parameters: dict[str, str] = Field(default_factory=dict)


class ArcGisMapServerSource(CamelModel):
    type: Literal["arcgis-mapserver"] = "arcgis-mapserver"
    url: HttpUrl
    layers: str | None = Field(default=None, max_length=500)


class GeoJsonSource(CamelModel):
    type: Literal["geojson"] = "geojson"
    url: HttpUrl
    clamp_to_ground: bool = True
    stroke: str | None = Field(default=None, max_length=32)
    fill: str | None = Field(default=None, max_length=32)


class CzmlSource(CamelModel):
    type: Literal["czml"] = "czml"
    url: HttpUrl


class MvtSource(CamelModel):
    """Adapter point for Mapbox Vector Tiles (CesiumJS MVTDataProvider, experimental)."""

    type: Literal["mvt"] = "mvt"
    url_template: str = Field(min_length=8, max_length=2048)
    min_zoom: int = Field(default=0, ge=0, le=24)
    max_zoom: int = Field(default=14, ge=0, le=24)


class StacSource(CamelModel):
    """Reference to a STAC item/collection. Resolution to a renderable asset happens client-side."""

    type: Literal["stac"] = "stac"
    url: HttpUrl
    kind: Literal["item", "collection", "catalog"] = "item"
    asset_key: str | None = Field(default=None, max_length=120)


LayerSource = Annotated[
    CesiumIonTerrainSource
    | CesiumIonImagerySource
    | CesiumIon3DTilesSource
    | GooglePhotorealisticSource
    | TilesUrlLayerSource
    | XyzSource
    | WmtsSource
    | WmsSource
    | ArcGisMapServerSource
    | GeoJsonSource
    | CzmlSource
    | MvtSource
    | StacSource,
    Field(discriminator="type"),
]


def source_type_for(source: object) -> LayerSourceType:
    type_value = getattr(source, "type", None)
    if not isinstance(type_value, str):
        raise ValueError("layer source must have a type")
    return LayerSourceType(type_value)


class LegendEntry(CamelModel):
    label: str = Field(max_length=120)
    color: str = Field(max_length=32)


class LegendMetadata(CamelModel):
    title: str | None = Field(default=None, max_length=120)
    entries: list[LegendEntry] = Field(default_factory=list)
    image_url: HttpUrl | None = None


class RenderMetadata(CamelModel):
    opacity: float = Field(default=1.0, ge=0, le=1)
    minimum_altitude_m: float | None = Field(default=None, ge=0)
    maximum_altitude_m: float | None = Field(default=None, ge=0)
    maximum_screen_space_error: float | None = Field(default=None, gt=0, le=512)
    exclusive_group: str | None = Field(
        default=None,
        max_length=64,
        description="Layers sharing a group are mutually exclusive (e.g. 'terrain', 'world').",
    )


class LayerBase(CamelModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    category: LayerCategory
    source: LayerSource
    spatial_extent: BoundingBox | None = None
    temporal_extent: TemporalExtent | None = None
    observed_at: datetime | None = None
    resolution: str | None = Field(default=None, max_length=120)
    coverage: str | None = Field(default=None, max_length=120)
    render: RenderMetadata = Field(default_factory=RenderMetadata)
    legend: LegendMetadata | None = None
    attribution: list[Attribution] = Field(default_factory=list)
    license: LicenseMetadata | None = None
    provenance: Provenance | None = None
    default_visible: bool = False


class LayerCreate(LayerBase):
    slug: str | None = Field(default=None, max_length=120)

    @field_validator("slug")
    @classmethod
    def _slug(cls, value: str | None) -> str | None:
        if value is not None and not SLUG_RE.match(value):
            raise ValueError("slug must be lowercase letters, digits and single hyphens")
        return value


class LayerUpdate(CamelModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    category: LayerCategory | None = None
    render: RenderMetadata | None = None
    attribution: list[Attribution] | None = None
    license: LicenseMetadata | None = None
    default_visible: bool | None = None
    observed_at: datetime | None = None
    temporal_extent: TemporalExtent | None = None


class LayerRead(CamelModel):
    """Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required."""

    id: uuid.UUID
    slug: str
    name: str
    description: str | None
    category: LayerCategory
    source_type: LayerSourceType
    source: LayerSource
    spatial_extent: BoundingBox | None
    temporal_extent: TemporalExtent | None
    observed_at: datetime | None
    resolution: str | None
    coverage: str | None
    render: RenderMetadata
    legend: LegendMetadata | None
    attribution: list[Attribution]
    license: LicenseMetadata | None
    provenance: Provenance | None
    default_visible: bool
    builtin: bool
    created_at: datetime
    updated_at: datetime
