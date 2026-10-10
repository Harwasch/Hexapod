from __future__ import annotations

import math

import httpx
import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from app.analysis.terrain import analyze, band_statistics, derive
from app.schemas.geojson import Polygon
from app.schemas.land_rasters import RasterRequest
from tests.test_raster_io import response as range_response

BOUNDARY = Polygon(
    coordinates=[
        [[-122.14, 47.64], [-122.13, 47.64], [-122.13, 47.65], [-122.14, 47.65], [-122.14, 47.64]],
        [
            [-122.138, 47.642],
            [-122.136, 47.642],
            [-122.136, 47.644],
            [-122.138, 47.644],
            [-122.138, 47.642],
        ],
    ]
)


def test_surface_slope_uses_metric_spacing_and_preserves_holes_and_nodata() -> None:
    y, x = np.mgrid[0:7, 0:7].astype(np.float32) * 30
    surface = (2 * x + 3 * y).astype(np.float32)
    inside = np.full((7, 7), True)
    inside[3, 3] = False
    elevations, slopes = derive(surface, inside, 30)
    assert np.isnan(elevations[3, 3]) and np.isnan(slopes[3, 3])
    assert float(slopes[2, 3]) == pytest.approx(math.degrees(math.atan(math.sqrt(13))), rel=1e-6)
    surface[2, 2] = np.nan
    _, missing = derive(surface, inside, 30)
    assert np.isnan(missing[2, 3])
    stats = band_statistics(elevations, 1, "Elevation", "m")
    assert stats.valid_cells == 48 and sum(stats.histogram_counts) == 48
    empty = band_statistics(np.full((3, 3), np.nan, dtype=np.float32), 1, "Elevation", "m")
    assert empty.mean is None and empty.histogram_counts == []


def fixture_client() -> httpx.Client:
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=128,
            width=128,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(-122.15, 47.66, 0.00025, 0.00025),
            nodata=-9999,
        ) as source:
            source.write(np.zeros((128, 128), dtype=np.float32), 1)
        data = memory.read()

    def serve(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search"):
            return httpx.Response(
                200,
                json={
                    "numberMatched": 1,
                    "links": [{"rel": "next", "href": "https://example.test/unused"}],
                    "features": [
                        {
                            "id": "fixture",
                            "assets": {"data": {"href": "s3://copernicus-dem-30m/tile/tile.tif"}},
                            "properties": {"datetime": "2021-04-22T00:00:00Z"},
                        }
                    ],
                },
            )
        return range_response(request, data=data)

    return httpx.Client(transport=httpx.MockTransport(serve))


def test_terrain_warps_masks_holes_and_emits_a_reproducible_private_cog() -> None:
    with fixture_client() as client:
        result = analyze(BOUNDARY, RasterRequest(), client)
    assert result.metadata.valid_cells == result.metadata.boundary_cells > 0
    assert result.metadata.coverage_fraction == 1
    assert result.metadata.bands[0].mean == 0  # Sea-level zero is valid data.
    assert result.metadata.bands[1].mean == 0
    assert result.metadata.sources[0].etag == '"version-1"'
    with MemoryFile(result.data) as memory, memory.open() as raster:
        assert raster.descriptions == ("Surface elevation", "Surface slope")
        assert raster.units == ("m", "degrees")
        assert raster.count == 2 and raster.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        data = raster.read(1, masked=True)
        assert data.count() == result.metadata.valid_cells
        assert data.mask.any()  # Outside boundary and its exclusion remain transparent/no-data.
    tiny = Polygon(
        coordinates=[
            [[-122.14, 47.64], [-122.13999, 47.64], [-122.13999, 47.64001], [-122.14, 47.64]]
        ]
    )
    with fixture_client() as client:
        small = analyze(tiny, RasterRequest(), client)
    assert small.metadata.boundary_cells == 0 and small.metadata.coverage_fraction is None
    assert small.metadata.bands[0].mean is None
    assert any("cannot resolve" in message for message in small.metadata.warnings)
