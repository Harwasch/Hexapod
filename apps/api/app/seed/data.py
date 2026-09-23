"""Seed catalog: the public Cesium Gaussian-splat demo site and open-data layers.

Everything here is real, publicly documented data with its license and
attribution recorded. Observation dates are only set when the provider states
them; nothing is fabricated.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from app.models.enums import LayerCategory, Representation
from app.schemas.asset import (
    AssetBase,
    CesiumIonSource,
    PointCloudShading,
    RenderConfig,
    ResolutionMetadata,
)
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


# --- Comparison sites -------------------------------------------------------------------
# Public Cesium ion sample assets, each a different kind of high-resolution reality model, so
# look, feel and performance can be compared in one app. Extents were read from each
# tileset's root bounding volume; attribution and terms are copied from the ion asset
# metadata. Access depends on the ion token in use (the CesiumJS evaluation token reaches
# all of them).

CC_BY_4 = LicenseMetadata(
    name="CC BY 4.0",
    spdx_id="CC-BY-4.0",
    url="https://creativecommons.org/licenses/by/4.0/legalcode",
    requires_attribution=True,
)

MELBOURNE_ATTRIBUTION = Attribution(
    text="Modified from City of Melbourne 3D data (CC BY 4.0)",
    organization="City of Melbourne",
    url="https://data.melbourne.vic.gov.au/",
)
MELBOURNE_BOUNDARY = rect_polygon(144.89221, -37.85357, 144.99544, -37.77166)
MELBOURNE_SITE = SiteCreate(
    slug="melbourne-mesh-vs-points",
    name="Melbourne: mesh vs point cloud",
    description=(
        "The same city captured two ways in 2018: a 7.5 cm textured photogrammetry mesh and a "
        "~300 million point LiDAR cloud at 25 cm. Switch representations to compare an opaque "
        "mesh (cheapest to render, Google-Earth-like) with a point cloud of the same scene."
    ),
    boundary=MELBOURNE_BOUNDARY,
    centroid=GeoPosition(longitude=144.9634, latitude=-37.8150, height=30.0),
    metadata={
        "comparison": True,
        "origin": "cesium-ion-asset-depot",
        "quality": {"resolutionDescription": "7.5 cm mesh · 25 cm points"},
    },
    attribution=[MELBOURNE_ATTRIBUTION],
    license=CC_BY_4,
    assets=[
        AssetBase(
            name="Photogrammetry mesh (7.5 cm)",
            representation=Representation.MESH,
            source=CesiumIonSource(asset_id=69380),
            footprint=MELBOURNE_BOUNDARY,
            resolution=ResolutionMetadata(
                ground_sample_distance_m=0.075,
                description="Textured mesh from 7.5 cm GSD aerial photogrammetry, May 2018",
            ),
            observed_at=datetime(2018, 5, 1, tzinfo=UTC),
            license=CC_BY_4,
            attribution=[
                Attribution(
                    text="Modified from City of Melbourne 3D Textured Mesh (CC BY 4.0)",
                    organization="City of Melbourne",
                    url="https://data.melbourne.vic.gov.au/explore/dataset/city-of-melbourne-3d-textured-mesh-photomesh-2018/",
                )
            ],
            provenance=Provenance(
                source_organization="City of Melbourne via Cesium ion asset depot",
                source_url="https://data.melbourne.vic.gov.au/explore/dataset/city-of-melbourne-3d-textured-mesh-photomesh-2018/",
                notes="Cesium ion asset 69380 (sample asset).",
            ),
            render_config=RenderConfig(clips_world=True, clip_footprint="tileset"),
            default_visible=True,
        ),
        AssetBase(
            name="LiDAR point cloud (25 cm, ~300 M points)",
            representation=Representation.POINT_CLOUD,
            source=CesiumIonSource(asset_id=43978),
            footprint=MELBOURNE_BOUNDARY,
            resolution=ResolutionMetadata(
                point_spacing_m=0.25, description="~300 million RGB points, 2018 capture"
            ),
            observed_at=datetime(2018, 5, 1, tzinfo=UTC),
            license=CC_BY_4,
            attribution=[
                Attribution(
                    text="Modified from City of Melbourne 3D Point Cloud (CC BY 4.0)",
                    organization="City of Melbourne",
                    url="https://data.melbourne.vic.gov.au/explore/dataset/city-of-melbourne-3d-point-cloud-2018/",
                )
            ],
            provenance=Provenance(
                source_organization="City of Melbourne via Cesium ion asset depot",
                source_url="https://data.melbourne.vic.gov.au/explore/dataset/city-of-melbourne-3d-point-cloud-2018/",
                notes="Cesium ion asset 43978 (sample asset).",
            ),
            render_config=RenderConfig(
                clips_world=False,
                point_cloud_shading=PointCloudShading(attenuation=True, eye_dome_lighting=True),
            ),
        ),
    ],
    camera_bookmarks=[overview_bookmark("CBD", 144.9634, -37.8150, 30.0, 35.0, -45.0, 1700.0)],
)

AEROMETREX_ATTRIBUTION = Attribution(
    text="Aerometrex San Francisco 3D Model, non-commercial trial and evaluation use only",
    organization="Aerometrex",
    url="https://aerometrex.com/models/",
)
# Outline of the Aerometrex mesh itself, traced from a top-down render in 60 m bands and
# inset about 20 m inside the content edge (the tiles use bounding spheres, so the viewer
# cannot derive an outline from them). The world is cut along this line, so the Google mesh
# overlaps the survey's ragged edge instead of leaving a band of terrain between them.
SAN_FRANCISCO_BOUNDARY = Polygon(
    coordinates=[
        [
            [-122.420936, 37.811771],
            [-122.420936, 37.811371],
            [-122.422877, 37.811238],
            [-122.422877, 37.810838],
            [-122.425239, 37.810705],
            [-122.425239, 37.810304],
            [-122.426674, 37.810171],
            [-122.426674, 37.809771],
            [-122.427011, 37.809638],
            [-122.427011, 37.809238],
            [-122.43199, 37.809104],
            [-122.43199, 37.808704],
            [-122.43199, 37.808571],
            [-122.43199, 37.808171],
            [-122.431905, 37.808037],
            [-122.431905, 37.807637],
            [-122.431737, 37.807504],
            [-122.431737, 37.807104],
            [-122.431568, 37.80697],
            [-122.431568, 37.80657],
            [-122.431483, 37.806437],
            [-122.431483, 37.806037],
            [-122.431399, 37.805903],
            [-122.431399, 37.805503],
            [-122.43123, 37.80537],
            [-122.43123, 37.80497],
            [-122.431146, 37.804837],
            [-122.431146, 37.804436],
            [-122.430977, 37.804303],
            [-122.430977, 37.803903],
            [-122.430893, 37.80377],
            [-122.430893, 37.803369],
            [-122.430808, 37.803236],
            [-122.430808, 37.802836],
            [-122.43064, 37.802703],
            [-122.43064, 37.802303],
            [-122.430555, 37.802169],
            [-122.430555, 37.801769],
            [-122.430386, 37.801636],
            [-122.430386, 37.801236],
            [-122.430218, 37.801102],
            [-122.430218, 37.800169],
            [-122.430049, 37.800035],
            [-122.430049, 37.799635],
            [-122.429965, 37.799502],
            [-122.429965, 37.799102],
            [-122.42988, 37.798968],
            [-122.42988, 37.798568],
            [-122.429796, 37.798435],
            [-122.429796, 37.798035],
            [-122.429711, 37.797902],
            [-122.429711, 37.797501],
            [-122.429543, 37.797368],
            [-122.429543, 37.796968],
            [-122.429458, 37.796835],
            [-122.429458, 37.796435],
            [-122.429374, 37.796301],
            [-122.429374, 37.795901],
            [-122.429205, 37.795768],
            [-122.429205, 37.795368],
            [-122.429121, 37.795234],
            [-122.429121, 37.794834],
            [-122.429121, 37.794701],
            [-122.429121, 37.794301],
            [-122.429121, 37.794167],
            [-122.429121, 37.793767],
            [-122.428952, 37.793634],
            [-122.428952, 37.7927],
            [-122.428868, 37.792567],
            [-122.428868, 37.792167],
            [-122.428614, 37.792034],
            [-122.428614, 37.791633],
            [-122.428446, 37.7915],
            [-122.428446, 37.7911],
            [-122.428193, 37.790967],
            [-122.428193, 37.7895],
            [-122.431652, 37.789366],
            [-122.431652, 37.788966],
            [-122.434268, 37.788833],
            [-122.434268, 37.787366],
            [-122.434099, 37.787232],
            [-122.434099, 37.785765],
            [-122.434015, 37.785632],
            [-122.434015, 37.785232],
            [-122.433593, 37.785099],
            [-122.433593, 37.783098],
            [-122.433424, 37.782965],
            [-122.433424, 37.782565],
            [-122.432496, 37.782431],
            [-122.432496, 37.778297],
            [-122.432412, 37.778164],
            [-122.432412, 37.777763],
            [-122.432327, 37.77763],
            [-122.432327, 37.77723],
            [-122.432158, 37.777097],
            [-122.432158, 37.776697],
            [-122.432074, 37.776563],
            [-122.432074, 37.776163],
            [-122.431821, 37.77603],
            [-122.431821, 37.775096],
            [-122.431568, 37.774963],
            [-122.431568, 37.774029],
            [-122.431315, 37.773896],
            [-122.431315, 37.772962],
            [-122.430977, 37.772829],
            [-122.430977, 37.771895],
            [-122.430808, 37.771762],
            [-122.430808, 37.771362],
            [-122.43064, 37.771229],
            [-122.43064, 37.770829],
            [-122.430386, 37.770695],
            [-122.430386, 37.770295],
            [-122.429965, 37.770162],
            [-122.429965, 37.769762],
            [-122.429289, 37.769628],
            [-122.429289, 37.769228],
            [-122.421273, 37.769095],
            [-122.421273, 37.768695],
            [-122.421358, 37.768561],
            [-122.421358, 37.768161],
            [-122.421358, 37.768028],
            [-122.421358, 37.767628],
            [-122.421273, 37.767494],
            [-122.421273, 37.767094],
            [-122.421189, 37.766961],
            [-122.421189, 37.766561],
            [-122.421358, 37.766427],
            [-122.421358, 37.766027],
            [-122.421442, 37.765894],
            [-122.421442, 37.765494],
            [-122.421526, 37.765361],
            [-122.421526, 37.76496],
            [-122.414016, 37.76496],
            [-122.414016, 37.765361],
            [-122.404734, 37.765494],
            [-122.404734, 37.765894],
            [-122.397562, 37.766027],
            [-122.397562, 37.766427],
            [-122.387436, 37.766561],
            [-122.387436, 37.766961],
            [-122.386171, 37.767094],
            [-122.386171, 37.767494],
            [-122.385833, 37.767628],
            [-122.385833, 37.768028],
            [-122.385411, 37.768161],
            [-122.385411, 37.768561],
            [-122.385074, 37.768695],
            [-122.385074, 37.769095],
            [-122.384652, 37.769228],
            [-122.384652, 37.769628],
            [-122.384314, 37.769762],
            [-122.384314, 37.770162],
            [-122.383808, 37.770295],
            [-122.383808, 37.770695],
            [-122.383386, 37.770829],
            [-122.383386, 37.771229],
            [-122.383048, 37.771362],
            [-122.383048, 37.771762],
            [-122.382627, 37.771895],
            [-122.382627, 37.772829],
            [-122.382795, 37.772962],
            [-122.382795, 37.773896],
            [-122.383302, 37.774029],
            [-122.383302, 37.774963],
            [-122.383723, 37.775096],
            [-122.383723, 37.77603],
            [-122.384061, 37.776163],
            [-122.384061, 37.776563],
            [-122.38482, 37.776697],
            [-122.38482, 37.777097],
            [-122.385749, 37.77723],
            [-122.385749, 37.77763],
            [-122.388111, 37.777763],
            [-122.388111, 37.778164],
            [-122.385749, 37.778297],
            [-122.385749, 37.782431],
            [-122.386255, 37.782565],
            [-122.386255, 37.782965],
            [-122.388449, 37.783098],
            [-122.388449, 37.785099],
            [-122.387858, 37.785232],
            [-122.387858, 37.785632],
            [-122.385917, 37.785765],
            [-122.385917, 37.787232],
            [-122.386255, 37.787366],
            [-122.386255, 37.788833],
            [-122.386255, 37.788966],
            [-122.386255, 37.789366],
            [-122.386508, 37.7895],
            [-122.386508, 37.790967],
            [-122.388027, 37.7911],
            [-122.388027, 37.7915],
            [-122.390136, 37.791633],
            [-122.390136, 37.792034],
            [-122.390727, 37.792167],
            [-122.390727, 37.792567],
            [-122.391655, 37.7927],
            [-122.391655, 37.793634],
            [-122.39098, 37.793767],
            [-122.39098, 37.794167],
            [-122.390474, 37.794301],
            [-122.390474, 37.794701],
            [-122.390896, 37.794834],
            [-122.390896, 37.795234],
            [-122.391402, 37.795368],
            [-122.391402, 37.795768],
            [-122.391908, 37.795901],
            [-122.391908, 37.796301],
            [-122.392415, 37.796435],
            [-122.392415, 37.796835],
            [-122.392921, 37.796968],
            [-122.392921, 37.797368],
            [-122.393343, 37.797501],
            [-122.393343, 37.797902],
            [-122.393849, 37.798035],
            [-122.393849, 37.798435],
            [-122.394356, 37.798568],
            [-122.394356, 37.798968],
            [-122.394862, 37.799102],
            [-122.394862, 37.799502],
            [-122.395368, 37.799635],
            [-122.395368, 37.800035],
            [-122.396043, 37.800169],
            [-122.396043, 37.801102],
            [-122.396465, 37.801236],
            [-122.396465, 37.801636],
            [-122.396971, 37.801769],
            [-122.396971, 37.802169],
            [-122.397393, 37.802303],
            [-122.397393, 37.802703],
            [-122.397815, 37.802836],
            [-122.397815, 37.803236],
            [-122.398237, 37.803369],
            [-122.398237, 37.80377],
            [-122.398743, 37.803903],
            [-122.398743, 37.804303],
            [-122.399165, 37.804436],
            [-122.399165, 37.804837],
            [-122.399587, 37.80497],
            [-122.399587, 37.80537],
            [-122.400009, 37.805503],
            [-122.400009, 37.805903],
            [-122.400431, 37.806037],
            [-122.400431, 37.806437],
            [-122.400937, 37.80657],
            [-122.400937, 37.80697],
            [-122.401697, 37.807104],
            [-122.401697, 37.807504],
            [-122.402709, 37.807637],
            [-122.402709, 37.808037],
            [-122.403638, 37.808171],
            [-122.403638, 37.808571],
            [-122.404566, 37.808704],
            [-122.404566, 37.809104],
            [-122.405494, 37.809238],
            [-122.405494, 37.809638],
            [-122.406422, 37.809771],
            [-122.406422, 37.810171],
            [-122.407519, 37.810304],
            [-122.407519, 37.810705],
            [-122.408447, 37.810838],
            [-122.408447, 37.811238],
            [-122.410219, 37.811371],
            [-122.410219, 37.811771],
            [-122.420936, 37.811771],
        ]
    ]
)
SAN_FRANCISCO_SITE = SiteCreate(
    slug="san-francisco-aerometrex-mesh",
    name="San Francisco: high-resolution mesh",
    description=(
        "Aerometrex helicopter photogrammetry at 2 to 5 cm with street-level enhanced sections: "
        "the closest public dataset to a Google-Earth-style city mesh. Opaque triangles, no "
        "per-frame sorting, so this is the performance reference for the other formats."
    ),
    boundary=SAN_FRANCISCO_BOUNDARY,
    centroid=GeoPosition(longitude=-122.4033, latitude=37.7913, height=20.0),
    metadata={
        "comparison": True,
        "origin": "cesium-ion-asset-depot",
        "quality": {"resolutionDescription": "2 to 5 cm mesh, street-level enhanced"},
    },
    attribution=[AEROMETREX_ATTRIBUTION],
    license=LicenseMetadata(
        name="Aerometrex non-commercial trial",
        url="https://aerometrex.com/models/",
        requires_attribution=True,
        notes="Provided through the Cesium ion asset depot for non-commercial trial and "
        "evaluation use only.",
    ),
    assets=[
        AssetBase(
            name="Photogrammetry mesh (2 to 5 cm)",
            representation=Representation.MESH,
            source=CesiumIonSource(asset_id=1415196),
            footprint=SAN_FRANCISCO_BOUNDARY,
            resolution=ResolutionMetadata(
                ground_sample_distance_m=0.05,
                description="2 cm and 5 cm helicopter photogrammetry with street-level mesh",
            ),
            attribution=[AEROMETREX_ATTRIBUTION],
            provenance=Provenance(
                source_organization="Aerometrex via Cesium ion asset depot",
                source_url="https://aerometrex.com/models/",
                notes="Cesium ion asset 1415196 (sample asset).",
            ),
            render_config=RenderConfig(
                clips_world=True,
                clip_footprint="tileset",
                # Measured 7 km up: at 2 px this mesh draws 17k triangles and looks like a
                # grey blob next to Google; at 0.5 px it draws 119k and shows its detail.
                screen_space_error_scale=0.25,
            ),
            default_visible=True,
        )
    ],
    camera_bookmarks=[
        overview_bookmark("Financial District", -122.4033, 37.7913, 20.0, 60.0, -45.0, 1300.0)
    ],
)

BOATHOUSE_BOUNDARY = rect_polygon(-75.29282, 40.07167, -75.29061, 40.07371)
BOATHOUSE_SITE = SiteCreate(
    slug="boathouse-gaussian-splat",
    name="Boathouse: close-range Gaussian splat",
    description=(
        "Cesium's small Gaussian splat sample (14 MB): a single building captured at close "
        "range, so it shows what a splat looks like at its intended viewing distance, unlike "
        "the aerial Redmond campus splat."
    ),
    boundary=BOATHOUSE_BOUNDARY,
    centroid=GeoPosition(longitude=-75.29172, latitude=40.07269, height=8.0),
    metadata={
        "comparison": True,
        "origin": "cesium-sandcastle",
        "quality": {"resolutionDescription": "Close-range splat, centimetre detail"},
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
            name="Gaussian splat (Boathouse v2)",
            representation=Representation.GAUSSIAN_SPLAT,
            source=CesiumIonSource(asset_id=3667783),
            footprint=BOATHOUSE_BOUNDARY,
            attribution=[CESIUM_ATTRIBUTION],
            provenance=Provenance(
                source_organization="Cesium GS, Inc.",
                source_url="https://sandcastle.cesium.com/?id=3d-tiles-gaussian-splats",
                notes="Sample asset referenced by the official CesiumJS Sandcastle (ion 3667783).",
            ),
            render_config=RenderConfig(clips_world=True, clip_footprint="tileset"),
            default_visible=True,
        )
    ],
    camera_bookmarks=[overview_bookmark("Boathouse", -75.29172, 40.07269, 8.0, 210.0, -40.0, 85.0)],
)

CESIUM_SAMPLE_LICENSE = LicenseMetadata(
    name="Cesium ion sample asset",
    url="https://cesium.com/legal/terms-of-service/",
    requires_attribution=True,
    notes="Provided by Cesium for evaluation. Access depends on the ion token in use.",
)

AGI_HQ_BOUNDARY = rect_polygon(-75.60002, 40.0364, -75.59339, 40.04119)
AGI_HQ_SITE = SiteCreate(
    slug="agi-hq-drone-mesh",
    name="AGI HQ: close-range drone mesh",
    description=(
        "A single office building and its grounds as a drone photogrammetry mesh (40 MB). "
        "Small enough to load in seconds; the closest public mesh to the scale of a work site."
    ),
    boundary=AGI_HQ_BOUNDARY,
    centroid=GeoPosition(longitude=-75.59671, latitude=40.0388, height=95.0),
    metadata={
        "comparison": True,
        "origin": "cesium-ion-asset-depot",
        "quality": {"resolutionDescription": "Centimetre-class drone photogrammetry mesh"},
    },
    attribution=[CESIUM_ATTRIBUTION],
    license=CESIUM_SAMPLE_LICENSE,
    assets=[
        AssetBase(
            name="Drone photogrammetry mesh",
            representation=Representation.MESH,
            source=CesiumIonSource(asset_id=40866),
            footprint=AGI_HQ_BOUNDARY,
            resolution=ResolutionMetadata(
                description="Drone photogrammetry of the AGI headquarters, tiled by Cesium"
            ),
            attribution=[CESIUM_ATTRIBUTION],
            provenance=Provenance(
                source_organization="Cesium GS, Inc.",
                notes="Cesium ion sample asset 40866 (photogrammetry OBJ tiled to 3D Tiles).",
            ),
            render_config=RenderConfig(clips_world=True, clip_footprint="tileset"),
            default_visible=True,
        )
    ],
    camera_bookmarks=[overview_bookmark("Campus", -75.59671, 40.0388, 95.0, 30.0, -45.0, 240.0)],
)

CHAPPES_ATTRIBUTION = Attribution(
    text=(
        "Chappes Church point cloud by Prof. Peter Allen, Columbia University Robotics Lab; "
        "scanning by Alejandro Troccoli and Matei Ciocarlie"
    ),
    organization="Columbia University Robotics Lab",
    url="https://www.cs.columbia.edu/robotics/",
)
CHAPPES_BOUNDARY = rect_polygon(2.92599, 46.38833, 2.92659, 46.38862)
CHAPPES_SITE = SiteCreate(
    slug="chappes-church-laser-scan",
    name="Chappes church: terrestrial laser scan",
    description=(
        "A ground-based laser scan of a village church, stored at 1 cm precision (37 MB). "
        "The closest public sample to survey-grade close-range data: compare its point "
        "density with the aerial point cloud in Melbourne."
    ),
    boundary=CHAPPES_BOUNDARY,
    centroid=GeoPosition(longitude=2.92629, latitude=46.38847, height=413.0),
    metadata={
        "comparison": True,
        "origin": "cesium-ion-asset-depot",
        "quality": {"resolutionDescription": "Terrestrial LiDAR, 1 cm coordinate precision"},
    },
    attribution=[CHAPPES_ATTRIBUTION, CESIUM_ATTRIBUTION],
    license=CESIUM_SAMPLE_LICENSE,
    assets=[
        AssetBase(
            name="Terrestrial laser scan (1 cm precision)",
            representation=Representation.POINT_CLOUD,
            source=CesiumIonSource(asset_id=16421),
            footprint=CHAPPES_BOUNDARY,
            resolution=ResolutionMetadata(
                point_spacing_m=0.01,
                description="Draco-compressed at 0.01 m precision, outliers removed, on terrain",
            ),
            attribution=[CHAPPES_ATTRIBUTION],
            provenance=Provenance(
                source_organization="Columbia University Robotics Lab via Cesium ion",
                notes="Cesium ion sample asset 16421.",
            ),
            render_config=RenderConfig(
                clips_world=False,
                point_cloud_shading=PointCloudShading(attenuation=True, eye_dome_lighting=True),
            ),
            default_visible=True,
        )
    ],
    camera_bookmarks=[overview_bookmark("Church", 2.92629, 46.38847, 420.0, 200.0, -40.0, 65.0)],
)

COMPARISON_SITES: list[SiteCreate] = [
    SAN_FRANCISCO_SITE,
    MELBOURNE_SITE,
    AGI_HQ_SITE,
    CHAPPES_SITE,
    BOATHOUSE_SITE,
]


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
