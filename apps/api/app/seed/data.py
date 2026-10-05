"""Seed catalog: the demo site the simulated fleet runs on, and open-data layers.

Everything here is real, publicly documented data with its license and
attribution recorded. Observation dates are only set when the provider states
them; nothing is fabricated.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from app.models.enums import LayerCategory
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

# The demo site is where Cesium's public "3D Tiles Gaussian splats with LOD" Sandcastle asset
# (ion 4547222) sits, in Redmond, WA. The splat itself is withdrawn (`app.seed.WITHDRAWN`):
# this deployment's ion account cannot read it. The place stays, because the simulated fleet
# (apps/web/src/missions/demo.ts, "Blackrock Mesa") is drawn inside it.
DEMO_CENTER_LON = -122.13810992689156
DEMO_CENTER_LAT = 47.644519699638366
DEMO_CENTER_HEIGHT = 120.0
DEMO_RADIUS_M = 60.0
# Horizontal extent of the tileset's root oriented bounding box (read from the streamed
# tileset in CesiumJS 1.145) around Redmond, WA. Outline of the splat's real coverage, traced
# from a top-down render: the tileset's root box is 1.7 km x 2.8 km and would cut the
# photorealistic world for whole blocks around the campus.
DEMO_BOUNDARY = Polygon(
    coordinates=[
        [
            [-122.142885, 47.658004],
            [-122.142885, 47.656463],
            [-122.140455, 47.656463],
            [-122.140455, 47.654922],
            [-122.139511, 47.654922],
            [-122.139511, 47.653381],
            [-122.13871, 47.653381],
            [-122.13871, 47.651069],
            [-122.13791, 47.651069],
            [-122.13791, 47.650299],
            [-122.143342, 47.650299],
            [-122.143342, 47.648758],
            [-122.140598, 47.648758],
            [-122.140598, 47.646446],
            [-122.142999, 47.646446],
            [-122.142999, 47.641823],
            [-122.143114, 47.641823],
            [-122.143114, 47.640282],
            [-122.143342, 47.640282],
            [-122.143342, 47.63797],
            [-122.143543, 47.63797],
            [-122.143543, 47.637199],
            [-122.143657, 47.637199],
            [-122.143657, 47.636429],
            [-122.132163, 47.636429],
            [-122.132163, 47.635658],
            [-122.132105, 47.635658],
            [-122.132105, 47.634888],
            [-122.127159, 47.634888],
            [-122.127159, 47.635658],
            [-122.126187, 47.635658],
            [-122.126187, 47.636429],
            [-122.126072, 47.636429],
            [-122.126072, 47.637199],
            [-122.124071, 47.637199],
            [-122.124071, 47.63797],
            [-122.12307, 47.63797],
            [-122.12307, 47.640282],
            [-122.12247, 47.640282],
            [-122.12247, 47.641823],
            [-122.120011, 47.641823],
            [-122.120011, 47.646446],
            [-122.131848, 47.646446],
            [-122.131848, 47.648758],
            [-122.131905, 47.648758],
            [-122.131905, 47.650299],
            [-122.131905, 47.651069],
            [-122.131934, 47.651069],
            [-122.131934, 47.653381],
            [-122.132077, 47.653381],
            [-122.132077, 47.654922],
            [-122.137309, 47.654922],
            [-122.137309, 47.656463],
            [-122.136938, 47.656463],
            [-122.136938, 47.658004],
            [-122.142885, 47.658004],
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


def overview_bookmark(
    name: str,
    lon: float,
    lat: float,
    height: float,
    heading: float,
    pitch: float,
    range_m: float,
) -> CameraBookmarkCreate:
    """Camera placed `range_m` from a target at (lon, lat, height), looking at it with the
    given heading and pitch (the equivalent of Cesium's viewBoundingSphere offset)."""
    horizontal = range_m * math.cos(math.radians(-pitch))
    vertical = range_m * math.sin(math.radians(-pitch))
    back_azimuth = math.radians(heading + 180.0)
    north = horizontal * math.cos(back_azimuth)
    east = horizontal * math.sin(back_azimuth)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(lat))
    return CameraBookmarkCreate(
        name=name,
        longitude=lon + east / meters_per_deg_lon,
        latitude=lat + north / meters_per_deg_lat,
        height=height + vertical,
        heading=heading,
        pitch=pitch,
        roll=0.0,
        is_default=True,
    )


def default_bookmark_for_demo() -> CameraBookmarkCreate:
    """Like the Sandcastle's viewBoundingSphere (heading 100°), but steeper (pitch -45°) so the
    arrival looks at the ground rather than the horizon."""
    return overview_bookmark(
        "Overview", DEMO_CENTER_LON, DEMO_CENTER_LAT, DEMO_CENTER_HEIGHT, 100.0, -45.0, 420.0
    )


def rect_polygon(west: float, south: float, east: float, north: float) -> Polygon:
    return Polygon(
        coordinates=[[[west, south], [east, south], [east, north], [west, north], [west, south]]]
    )


DEMO_SITE = SiteCreate(
    slug="cesium-splat-demo",
    name="Cesium Gaussian splat demo",
    description=(
        "Where the simulated fleet demo runs, in the photorealistic world around Cesium's "
        "'3D Tiles Gaussian splats with LOD' Sandcastle site. It has no reality model of its own."
    ),
    boundary=DEMO_BOUNDARY,
    centroid=GeoPosition(
        longitude=DEMO_CENTER_LON, latitude=DEMO_CENTER_LAT, height=DEMO_CENTER_HEIGHT
    ),
    metadata={"origin": "cesium-sandcastle"},
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
