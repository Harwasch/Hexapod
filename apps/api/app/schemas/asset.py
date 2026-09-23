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


class GroundSample(CamelModel):
    """One cell of the capture's own measured ground, as a point on the globe.

    This is `ground_samples.json` after one addition and one subtraction: the pipeline
    writes `z`, the capture's own up-coordinate relative to the placed origin, and the
    worker adds the origin height so that what reaches the browser is an ellipsoid height
    in the same datum the terrain is sampled in. The viewer then compares like with like
    and never has to know what frame the capture was reconstructed in.
    """

    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)
    #: Ellipsoid height of the capture's own ground at this point, in metres.
    height: float


ClipFootprint = Literal["catalog", "tileset"]


class RenderConfig(CamelModel):
    maximum_screen_space_error: float | None = Field(default=None, gt=0, le=512)
    # Calibration of this asset's geometric errors against what its textures can show. Tilers
    # assign geometric error from geometry alone; a survey mesh with 2 cm textures but
    # conservative errors looks soft next to the Google world at the same screen-space error.
    # The viewer multiplies its screen-space error by this (0.25 = four times finer).
    screen_space_error_scale: float = Field(default=1.0, gt=0, le=4)
    point_cloud_shading: PointCloudShading | None = None
    # When true, the asset's footprint clips the globe/terrain and the global 3D world.
    clips_world: bool = True
    # "catalog": clip with the stored footprint/boundary; "tileset": derive the clip
    # polygon from the loaded tileset's root bounding volume (robust for assets whose
    # exact extent is only known once streamed).
    clip_footprint: ClipFootprint = "catalog"
    # A manual correction added on top of whatever the clamp works out. It used to be the
    # only way to fix a sloping capture, because the clamp rested the *lowest corner* of
    # the bounding box on the terrain *under the centre* -- two different places, so the
    # slope's own drop came out as float. `ground_samples` is what removes the need for
    # it: with samples present this is a deliberate nudge on top of a measured clamp, not
    # a correction for the clamp's own error.
    height_offset_m: float = 0.0
    # When true, the viewer rests the model on the ground once it is loaded. Phone scans
    # and downloaded objects rarely carry a usable ellipsoid height; this makes their
    # placement trivial.
    clamp_to_ground: bool = False
    # The capture's own measured ground, cell by cell, as ellipsoid heights. When this is
    # non-empty the clamp samples the ground at *these* points and takes the median
    # difference, which is the subtraction a person used to do by hand between
    # `tools/captures/ground_samples.py` and a browser console. Empty means nobody
    # measured it, and the clamp falls back to the bounding box exactly as before -- so
    # no asset that has never had samples can move because of this field.
    #
    # It rides in the render config rather than in a field of its own because it is an
    # input to placement and nothing else, it sits beside the two knobs it replaces, and
    # `render_config` is the one asset column that already round-trips free-form JSON.
    # Capped at 64 because that is `splat_ground`'s own `max_cells`, and a catalog
    # response is not a place to stream a point cloud.
    ground_samples: list[GroundSample] = Field(default_factory=list, max_length=64)
    # Where this asset's Living Survey motion rig sits, relative to its tileset URL
    # (`../source/rig.json` for everything tools/captures writes today). None means this
    # asset does not move, which is the answer for every asset that is not a single
    # object someone measured.
    #
    # This is a claim the catalog makes, not a file the viewer goes looking for. It
    # replaces the hand-maintained LIVING_RIGS table that used to be compiled into the web
    # bundle: same claim, same absence of any probe, but written where a capture is
    # registered rather than in a frontend source file that had to be edited and rebuilt
    # for every new capture. Relative on purpose -- the rig travels with the tiles, so the
    # same value is correct whether they are served from the dev static mount or a bucket.
    rig_url: str | None = Field(default=None, max_length=500)


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
