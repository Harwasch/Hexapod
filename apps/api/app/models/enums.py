"""Domain enumerations shared by the ORM and the API schemas.

Values are stable string identifiers that appear in the OpenAPI contract.
"""

from __future__ import annotations

from enum import StrEnum


class Representation(StrEnum):
    GAUSSIAN_SPLAT = "gaussian-splat"
    MESH = "mesh"
    POINT_CLOUD = "point-cloud"
    TERRAIN = "terrain"
    IMAGERY = "imagery"


class AssetProvider(StrEnum):
    """Delivery provider for a derived, renderable asset.

    Cesium ion is the current delivery provider; it is deliberately *not* the
    canonical data model (see docs/ARCHITECTURE.md).
    """

    CESIUM_ION = "cesium-ion"
    TILES_3D_URL = "3d-tiles-url"


class LayerCategory(StrEnum):
    REALITY = "reality"
    TERRAIN = "terrain"
    IMAGERY = "imagery"
    HYDROLOGY = "hydrology"
    LAND_COVER = "land-cover"
    ECOLOGY = "ecology"
    INFRASTRUCTURE = "infrastructure"
    MY_DATA = "my-data"


class LayerSourceType(StrEnum):
    CESIUM_ION_TERRAIN = "cesium-ion-terrain"
    CESIUM_ION_IMAGERY = "cesium-ion-imagery"
    CESIUM_ION_3D_TILES = "cesium-ion-3d-tiles"
    GOOGLE_PHOTOREALISTIC = "google-photorealistic"
    TILES_3D_URL = "3d-tiles-url"
    XYZ = "xyz"
    WMTS = "wmts"
    WMS = "wms"
    ARCGIS_MAPSERVER = "arcgis-mapserver"
    GEOJSON = "geojson"
    CZML = "czml"
    MVT = "mvt"
    STAC = "stac"
