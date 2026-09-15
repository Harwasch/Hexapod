"""Seed catalog: the public Cesium Gaussian-splat demo site and open-data layers.

Everything here is real, publicly documented data with its license and
attribution recorded. Observation dates are only set when the provider states
them; nothing is fabricated.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from app.models.enums import LayerCategory, Representation
from app.schemas.asset import AssetBase, CesiumIonSource, RenderConfig
from app.schemas.bookmark import CameraBookmarkCreate
from app.schemas.common import (
    Attribution,
    BoundingBox,
    GeoPosition,
    LicenseMetadata,
    Provenance,
    TemporalExtent,
)
from app.schemas.geojson import Polygon
from app.schemas.layer import (
    ArcGisMapServerSource,
    CesiumIon3DTilesSource,
    CesiumIonImagerySource,
    CesiumIonTerrainSource,
    GooglePhotorealisticSource,
    LayerCreate,
    LegendEntry,
    LegendMetadata,
    RenderMetadata,
    WmsSource,
    XyzSource,
)
from app.schemas.site import SiteCreate

# Cesium's public "3D Tiles Gaussian splats with LOD" Sandcastle asset.
DEMO_SPLAT_ASSET_ID = 4547222
DEMO_CENTER_LON = -122.13810992689156
DEMO_CENTER_LAT = 47.644519699638366
DEMO_CENTER_HEIGHT = 120.0
DEMO_RADIUS_M = 60.0
# Horizontal extent of the tileset's root oriented bounding box (read from the streamed
# tileset in CesiumJS 1.145): ~1.7 km x 2.8 km around Redmond, WA.
DEMO_BOUNDARY = Polygon(
    coordinates=[
        [
            [-122.143272, 47.635222],
            [-122.120532, 47.635222],
            [-122.120526, 47.660476],
            [-122.143272, 47.660476],
            [-122.143272, 47.635222],
        ]
    ]
)

WORLD = BoundingBox(west=-180, south=-90, east=180, north=90)
CONUS = BoundingBox(west=-125, south=24, east=-66, north=50)
US_ALL = BoundingBox(west=-180, south=17, east=-64, north=72)


def circle_polygon(lon: float, lat: float, radius_m: float, segments: int = 24) -> Polygon:
    """Geodesic-ish circle as a closed GeoJSON polygon (counter-clockwise)."""
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(lat))
    ring: list[list[float]] = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        ring.append(
            [
                lon + radius_m * math.cos(angle) / meters_per_deg_lon,
                lat + radius_m * math.sin(angle) / meters_per_deg_lat,
            ]
        )
    ring.append(list(ring[0]))
    return Polygon(coordinates=[ring])


def default_bookmark_for_demo() -> CameraBookmarkCreate:
    """Equivalent of the Sandcastle's viewBoundingSphere(heading 100°, pitch -25°, range 500 m)."""
    heading = 100.0
    pitch = -25.0
    range_m = 500.0
    horizontal = range_m * math.cos(math.radians(-pitch))
    vertical = range_m * math.sin(math.radians(-pitch))
    back_azimuth = math.radians(heading + 180.0)
    north = horizontal * math.cos(back_azimuth)
    east = horizontal * math.sin(back_azimuth)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(DEMO_CENTER_LAT))
    return CameraBookmarkCreate(
        name="Overview",
        longitude=DEMO_CENTER_LON + east / meters_per_deg_lon,
        latitude=DEMO_CENTER_LAT + north / meters_per_deg_lat,
        height=DEMO_CENTER_HEIGHT + vertical,
        heading=heading,
        pitch=pitch,
        roll=0.0,
        is_default=True,
    )


CESIUM_ATTRIBUTION = Attribution(
    text="Cesium sample data", organization="Cesium GS, Inc.", url="https://cesium.com/"
)

DEMO_SITE = SiteCreate(
    slug="cesium-splat-demo",
    name="Cesium Gaussian splat demo",
    description=(
        "Public 3D Gaussian splat tileset with hierarchical level of detail, published by "
        "Cesium for the '3D Tiles Gaussian splats with LOD' Sandcastle. It demonstrates a "
        "reality model embedded in the global world."
    ),
    boundary=DEMO_BOUNDARY,
    centroid=GeoPosition(
        longitude=DEMO_CENTER_LON, latitude=DEMO_CENTER_LAT, height=DEMO_CENTER_HEIGHT
    ),
    metadata={
        "quality": {"resolutionDescription": "Sub-decimetre splat detail (visual)"},
        "origin": "cesium-sandcastle",
    },
    attribution=[CESIUM_ATTRIBUTION],
    license=LicenseMetadata(
        name="Cesium ion sample asset",
        url="https://cesium.com/legal/terms-of-service/",
        requires_attribution=True,
        notes="Provided by Cesium for evaluation. Access depends on the ion token in use.",
    ),
    assets=[
        AssetBase(
            name="Gaussian splat (LOD)",
            representation=Representation.GAUSSIAN_SPLAT,
            source=CesiumIonSource(asset_id=DEMO_SPLAT_ASSET_ID),
            footprint=DEMO_BOUNDARY,
            attribution=[CESIUM_ATTRIBUTION],
            provenance=Provenance(
                source_organization="Cesium GS, Inc.",
                source_url="https://sandcastle.cesium.com/?id=3d-tiles-gaussian-splats-with-lod",
                notes="Sample asset referenced by the official CesiumJS Sandcastle.",
            ),
            render_config=RenderConfig(
                maximum_screen_space_error=16, clips_world=True, clip_footprint="tileset"
            ),
            default_visible=True,
        )
    ],
    camera_bookmarks=[default_bookmark_for_demo()],
)


def _cesium_ion_license(notes: str) -> LicenseMetadata:
    return LicenseMetadata(
        name="Cesium ion Terms of Service",
        url="https://cesium.com/legal/terms-of-service/",
        requires_attribution=True,
        notes=notes,
    )


PUBLIC_DOMAIN_US = LicenseMetadata(
    name="Public domain (U.S. Government work)",
    spdx_id="US-PD",
    url="https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits",
    requires_attribution=False,
    notes="USGS asks that the data be credited.",
)

LAYERS: list[LayerCreate] = [
    LayerCreate(
        slug="cesium-world-terrain",
        name="Cesium World Terrain",
        description="Global quantized-mesh terrain curated from open elevation sources.",
        category=LayerCategory.TERRAIN,
        source=CesiumIonTerrainSource(asset_id=1),
        spatial_extent=WORLD,
        resolution="~30 m global, ~1 m where sources allow",
        coverage="Global",
        render=RenderMetadata(exclusive_group="terrain"),
        attribution=[
            Attribution(
                text=(
                    "Cesium World Terrain: data available from the U.S. Geological Survey, "
                    "© CGIAR-CSI, produced using Copernicus data and information funded by the "
                    "European Union - EU-DEM layers, Land Information New Zealand, data.gov.uk, "
                    "Geoscience Australia"
                ),
                organization="Cesium GS, Inc.",
                url="https://cesium.com/platform/cesium-ion/content/cesium-world-terrain/",
            )
        ],
        license=_cesium_ion_license("Streamed from Cesium ion (asset 1)."),
        provenance=Provenance(
            source_organization="Cesium GS, Inc.",
            source_url="https://cesium.com/platform/cesium-ion/content/cesium-world-terrain/",
        ),
        default_visible=True,
    ),
    LayerCreate(
        slug="bing-maps-aerial",
        name="Bing Maps Aerial",
        description="Global aerial and satellite imagery basemap streamed through Cesium ion.",
        category=LayerCategory.IMAGERY,
        source=CesiumIonImagerySource(asset_id=2),
        spatial_extent=WORLD,
        resolution="~0.3 m urban, ~15 m global",
        coverage="Global",
        render=RenderMetadata(exclusive_group="basemap"),
        attribution=[
            Attribution(
                text="© Microsoft, Bing Maps",
                organization="Microsoft",
                url="https://www.bing.com/maps",
            ),
        ],
        license=_cesium_ion_license("Bing Maps imagery via Cesium ion (asset 2)."),
        provenance=Provenance(
            source_organization="Microsoft / Cesium ion",
            source_url="https://cesium.com/platform/cesium-ion/content/",
        ),
        default_visible=True,
    ),
    LayerCreate(
        slug="sentinel-2",
        name="Sentinel-2 cloudless",
        description="10 m global satellite mosaic from Copernicus Sentinel-2 (EOX cloudless).",
        category=LayerCategory.IMAGERY,
        source=CesiumIonImagerySource(asset_id=3954),
        spatial_extent=WORLD,
        resolution="10 m",
        coverage="Global",
        temporal_extent=TemporalExtent(
            start=datetime(2016, 1, 1, tzinfo=UTC), end=datetime(2017, 12, 31, tzinfo=UTC)
        ),
        render=RenderMetadata(exclusive_group="basemap"),
        attribution=[
            Attribution(
                text=(
                    "Sentinel-2 cloudless by EOX IT Services GmbH "
                    "(contains modified Copernicus Sentinel data 2016 & 2017)"
                ),
                organization="EOX IT Services GmbH",
                url="https://s2maps.eu",
            )
        ],
        license=LicenseMetadata(
            name="CC BY-NC-SA 4.0",
            spdx_id="CC-BY-NC-SA-4.0",
            url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
            requires_attribution=True,
            notes="Served via Cesium ion asset 3954. Check EOX terms for commercial use.",
        ),
        provenance=Provenance(
            source_organization="ESA Copernicus / EOX", source_url="https://s2maps.eu"
        ),
    ),
    LayerCreate(
        slug="openstreetmap",
        name="OpenStreetMap",
        description="Community-maintained street map raster tiles.",
        category=LayerCategory.IMAGERY,
        source=XyzSource(
            url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png", maximum_level=19
        ),
        spatial_extent=WORLD,
        resolution="Zoom 19 (~0.3 m/px)",
        coverage="Global",
        render=RenderMetadata(exclusive_group="basemap"),
        attribution=[
            Attribution(
                text="© OpenStreetMap contributors",
                organization="OpenStreetMap Foundation",
                url="https://www.openstreetmap.org/copyright",
            )
        ],
        license=LicenseMetadata(
            name="ODbL 1.0 (data); tile usage policy applies",
            spdx_id="ODbL-1.0",
            url="https://operations.osmfoundation.org/policies/tiles/",
            requires_attribution=True,
        ),
        provenance=Provenance(
            source_organization="OpenStreetMap Foundation",
            source_url="https://www.openstreetmap.org",
        ),
    ),
    LayerCreate(
        slug="cesium-osm-buildings",
        name="OSM Buildings",
        description="Global 3D building footprints extruded from OpenStreetMap, as 3D Tiles.",
        category=LayerCategory.INFRASTRUCTURE,
        source=CesiumIon3DTilesSource(asset_id=96188),
        spatial_extent=WORLD,
        coverage="Global",
        render=RenderMetadata(maximum_altitude_m=250_000),
        attribution=[
            Attribution(
                text="Cesium OSM Buildings · © OpenStreetMap contributors",
                organization="Cesium GS, Inc.",
                url="https://cesium.com/platform/cesium-ion/content/cesium-osm-buildings/",
            )
        ],
        license=LicenseMetadata(
            name="ODbL 1.0",
            spdx_id="ODbL-1.0",
            url="https://opendatacommons.org/licenses/odbl/",
            requires_attribution=True,
        ),
        provenance=Provenance(
            source_organization="Cesium GS, Inc. / OpenStreetMap",
            source_url="https://cesium.com/platform/cesium-ion/content/cesium-osm-buildings/",
        ),
    ),
    LayerCreate(
        slug="esa-worldcover-2021",
        name="ESA WorldCover 2021",
        description="Global 10 m land-cover map for 2021 from Sentinel-1 and Sentinel-2.",
        category=LayerCategory.LAND_COVER,
        source=WmsSource(
            url="https://services.terrascope.be/wms/v2",
            layers="WORLDCOVER_2021_MAP",
            parameters={"transparent": "true", "format": "image/png"},
        ),
        spatial_extent=WORLD,
        resolution="10 m",
        coverage="Global",
        observed_at=datetime(2021, 1, 1, tzinfo=UTC),
        temporal_extent=TemporalExtent(
            start=datetime(2021, 1, 1, tzinfo=UTC), end=datetime(2021, 12, 31, tzinfo=UTC)
        ),
        render=RenderMetadata(opacity=0.75),
        legend=LegendMetadata(
            title="Land cover class",
            entries=[
                LegendEntry(label="Tree cover", color="#006400"),
                LegendEntry(label="Shrubland", color="#ffbb22"),
                LegendEntry(label="Grassland", color="#ffff4c"),
                LegendEntry(label="Cropland", color="#f096ff"),
                LegendEntry(label="Built-up", color="#fa0000"),
                LegendEntry(label="Bare / sparse", color="#b4b4b4"),
                LegendEntry(label="Snow and ice", color="#f0f0f0"),
                LegendEntry(label="Permanent water", color="#0064c8"),
                LegendEntry(label="Herbaceous wetland", color="#0096a0"),
                LegendEntry(label="Mangroves", color="#00cf75"),
                LegendEntry(label="Moss and lichen", color="#fae6a0"),
            ],
        ),
        attribution=[
            Attribution(
                text=(
                    "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel "
                    "data (2021) processed by ESA WorldCover consortium"
                ),
                organization="ESA",
                url="https://esa-worldcover.org",
            )
        ],
        license=LicenseMetadata(
            name="CC BY 4.0",
            spdx_id="CC-BY-4.0",
            url="https://creativecommons.org/licenses/by/4.0/",
            requires_attribution=True,
        ),
        provenance=Provenance(
            source_organization="ESA / VITO Terrascope",
            source_url="https://esa-worldcover.org/en/data-access",
            published_at=datetime(2022, 10, 28, tzinfo=UTC),
        ),
    ),
    LayerCreate(
        slug="usgs-nhd-hydrography",
        name="USGS National Hydrography",
        description="Rivers, streams, lakes and watersheds from the USGS National Hydrography Dataset.",
        category=LayerCategory.HYDROLOGY,
        source=ArcGisMapServerSource(
            url="https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"
        ),
        spatial_extent=US_ALL,
        coverage="United States",
        resolution="1:24,000 (high-resolution NHD)",
        render=RenderMetadata(opacity=0.9),
        attribution=[
            Attribution(
                text="USGS National Hydrography Dataset",
                organization="U.S. Geological Survey",
                url="https://www.usgs.gov/national-hydrography",
            )
        ],
        license=PUBLIC_DOMAIN_US,
        provenance=Provenance(
            source_organization="U.S. Geological Survey",
            source_url="https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer",
        ),
    ),
    LayerCreate(
        slug="usgs-imagery",
        name="USGS Orthoimagery (NAIP)",
        description="High-resolution U.S. orthoimagery from The National Map, primarily NAIP.",
        category=LayerCategory.IMAGERY,
        source=ArcGisMapServerSource(
            url="https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer"
        ),
        spatial_extent=US_ALL,
        coverage="United States",
        resolution="~1 m (NAIP), 0.3 m in places",
        render=RenderMetadata(exclusive_group="basemap"),
        attribution=[
            Attribution(
                text="USGS The National Map: Orthoimagery. Data refreshed April 2023.",
                organization="U.S. Geological Survey",
                url="https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer",
            )
        ],
        license=PUBLIC_DOMAIN_US,
        provenance=Provenance(
            source_organization="U.S. Geological Survey / USDA NAIP",
            source_url="https://www.usgs.gov/programs/national-geospatial-program/national-map",
        ),
    ),
    LayerCreate(
        slug="usgs-3dep-shaded-relief",
        name="USGS 3DEP shaded relief",
        description="Hillshade derived from the 3D Elevation Program (lidar-based DEMs where available).",
        category=LayerCategory.TERRAIN,
        source=ArcGisMapServerSource(
            url="https://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer"
        ),
        spatial_extent=US_ALL,
        coverage="United States",
        resolution="1/3 arc-second (~10 m); 1 m lidar where available",
        render=RenderMetadata(opacity=0.6),
        attribution=[
            Attribution(
                text="USGS 3D Elevation Program (3DEP)",
                organization="U.S. Geological Survey",
                url="https://www.usgs.gov/3d-elevation-program",
            )
        ],
        license=PUBLIC_DOMAIN_US,
        provenance=Provenance(
            source_organization="U.S. Geological Survey",
            source_url="https://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer",
        ),
    ),
    LayerCreate(
        slug="google-photorealistic-3d-tiles",
        name="Google Photorealistic 3D Tiles",
        description=(
            "Photorealistic global 3D mesh from Google Maps Platform, streamed via Cesium ion. "
            "Visual context only; not analytical source data."
        ),
        category=LayerCategory.REALITY,
        source=GooglePhotorealisticSource(),
        spatial_extent=WORLD,
        coverage="Global (major regions)",
        render=RenderMetadata(exclusive_group="world"),
        attribution=[
            Attribution(
                text="Google · data providers credited on screen",
                organization="Google",
                url="https://developers.google.com/maps/documentation/tile/policies",
            )
        ],
        license=LicenseMetadata(
            name="Google Maps Platform Terms of Service",
            url="https://cloud.google.com/maps-platform/terms",
            requires_attribution=True,
            notes="Requires the feature flag and an ion token with access to asset 2275207.",
        ),
        provenance=Provenance(
            source_organization="Google",
            source_url="https://developers.google.com/maps/documentation/tile/3d-tiles",
        ),
    ),
]
