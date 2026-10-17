from __future__ import annotations

import math
import re
import uuid
from datetime import date

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from sqlalchemy.orm import Session, sessionmaker

from app.analysis.dem_3dep import NODATA, UNCOVERED, analyze, tile_names, tile_url
from app.analysis.raster_io import RasterHttpFile, ReadBudget
from app.schemas.geojson import Polygon
from app.schemas.land_rasters import RasterRequest, VegetationPeriod
from tests.test_raster_io import response as range_response

# A small synthetic patch near Templeton, CA, inside tile n36w121; no private source data.
BOUNDARY = Polygon(
    coordinates=[
        [
            [-120.605, 35.620],
            [-120.600, 35.620],
            [-120.600, 35.625],
            [-120.605, 35.625],
            [-120.605, 35.620],
        ],
        [
            [-120.6035, 35.6215],
            [-120.6030, 35.6215],
            [-120.6030, 35.6220],
            [-120.6035, 35.6220],
            [-120.6035, 35.6215],
        ],
    ]
)
STEP = 1 / 10800  # 1/3 arc-second
REQUEST = RasterRequest(dataset="usgs-3dep-13", resolution_m=10)


def tile() -> bytes:
    # A plane rising 1 m per source row northward, with a no-data hole inside the boundary.
    rows = np.arange(128, dtype=np.float32)[:, None]
    surface = np.repeat(400 - rows, 128, axis=1).astype(np.float32)
    surface[60:66, 60:66] = NODATA
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=128,
            height=128,
            count=1,
            dtype="float32",
            crs="EPSG:4269",
            transform=from_origin(-120.6085, 35.6285, STEP, STEP),
            nodata=NODATA,
        ) as dataset:
            dataset.write(surface, 1)
        data: bytes = memory.read()
    return data


def fixture_client(*, present: tuple[str, ...] = ("n36w121",)) -> httpx.Client:
    data = tile()
    served = {tile_url(name) for name in present}

    def serve(request: httpx.Request) -> httpx.Response:
        if str(request.url) in served:
            return range_response(request, data=data)
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(serve))


def test_tile_names_follow_the_north_west_corner_naming() -> None:
    assert tile_names((-120.60, 35.62, -120.60, 35.62)) == ["n36w121"]
    assert tile_names((-120.5, 35.9, -120.4, 36.1)) == ["n37w121", "n36w121"]
    assert tile_names((-121.2, 35.9, -120.8, 36.1)) == [
        "n37w122",
        "n37w121",
        "n36w122",
        "n36w121",
    ]
    # A box ending exactly on a tile edge does not need the tile beyond it.
    assert tile_names((-121.0, 35.0, -120.5, 35.5)) == ["n36w121"]
    with pytest.raises(ValueError, match="more than four"):
        tile_names((-122.5, 35.5, -120.5, 36.5))
    for outside in [(2.3, 48.8, 2.4, 48.9), (-60.1, -10.2, -60.0, -10.1)]:
        with pytest.raises(ValueError, match="cop-dem-glo-30"):
            tile_names(outside)


def test_bare_earth_elevation_and_slope_with_holes_and_source_no_data() -> None:
    with fixture_client() as client:
        result = analyze(BOUNDARY, REQUEST, client)
    metadata = result.metadata
    elevation, slope = metadata.bands
    assert (elevation.name, elevation.unit) == ("Bare-earth elevation", "m")
    assert (slope.name, slope.unit) == ("Bare-earth slope", "degrees")
    assert metadata.algorithm == "usgs-3dep-13-v1" and metadata.resolution_m == 10
    # 1 m per 1/3-arc-second row: atan(1 / row height in metres).
    row_m = STEP * 110_940
    expected = math.degrees(math.atan(1 / row_m))
    assert slope.percentiles["50"] == pytest.approx(expected, rel=0.03)
    assert elevation.minimum is not None and 300 < elevation.minimum < 400
    # The source hole drops elevation there and slope next to it as well.
    assert 0 < elevation.valid_cells < metadata.boundary_cells
    assert slope.valid_cells < elevation.valid_cells
    assert any("no source data" in warning for warning in metadata.warnings)
    assert any("NAVD88" in warning for warning in metadata.warnings)
    source = metadata.sources[0]
    assert source.id == "USGS_13_n36w121" and source.etag == '"version-1"'
    assert str(source.catalog_url).endswith("/n36w121/USGS_13_n36w121.xml")
    assert (
        source.catalog_datetime is None and "not an acquisition date" in source.observation_period
    )
    assert "Public domain" in source.license
    with MemoryFile(result.data) as memory, memory.open() as raster:
        assert raster.descriptions == ("Bare-earth elevation", "Bare-earth slope")
        assert raster.tags()["vertical_reference"] == "NAVD88 (elevation band)"
        assert raster.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        data = raster.read(1, masked=True)
        assert data.count() == elevation.valid_cells and data.mask.any()


def test_outside_the_united_states_is_refused_with_the_alternative() -> None:
    def never(request: httpx.Request) -> httpx.Response:
        raise AssertionError("No 3DEP request outside the United States")

    paris = Polygon(
        coordinates=[[[2.35, 48.85], [2.36, 48.85], [2.36, 48.86], [2.35, 48.86], [2.35, 48.85]]]
    )
    with (
        httpx.Client(transport=httpx.MockTransport(never)) as client,
        pytest.raises(ValueError, match=re.escape(UNCOVERED)),
    ):
        analyze(paris, REQUEST, client)
    with (
        fixture_client(present=()) as client,
        pytest.raises(ValueError, match="use cop-dem-glo-30 here"),
    ):
        analyze(BOUNDARY, REQUEST, client)


def test_only_approved_hosts_and_tif_paths_are_read() -> None:
    with fixture_client() as client:
        RasterHttpFile(client, tile_url("n36w121"), ReadBudget()).close()
        for url in [
            "https://prd-tnm.s3.amazonaws.com.example.test/USGS_13_n36w121.tif",
            tile_url("n36w121").replace(".tif", ".xml"),
            "http://prd-tnm.s3.amazonaws.com/StagedProducts/x.tif",
        ]:
            with pytest.raises(OSError, match="approved"):
                RasterHttpFile(client, url, ReadBudget())
    with pytest.raises(ValueError, match="only supported"):
        RasterRequest(
            dataset="usgs-3dep-13",
            periods=[VegetationPeriod(start_date=date(2025, 1, 1), end_date=date(2025, 1, 2))],
        )


def test_3dep_run_saves_a_cited_bare_earth_raster(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import Settings
    from app.research.worker import ResearchWorker
    from tests.test_land import BODY

    with fixture_client() as http:
        result = analyze(BOUNDARY, REQUEST, http)
    monkeypatch.setattr("app.research.worker.run_raster", lambda *args: result)
    land = client.post("/api/v1/land", json={**BODY, "boundary": BOUNDARY.model_dump()}).json()
    investigation = client.post(
        f"/api/v1/land/{land['id']}/investigations",
        json={"title": "Bare-earth elevation", "boundaryRevision": 1},
    ).json()["id"]
    created = client.post(
        f"/api/v1/research/investigations/{investigation}/runs",
        json={
            "kind": "raster",
            "requestKey": str(uuid.uuid4()),
            "question": "Analyze bare-earth elevation and slope",
            "analysis": REQUEST.model_dump(mode="json", by_alias=True),
        },
    )
    assert created.status_code == 201, created.text
    db.rollback()
    assert ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{investigation}").json()
    assert detail["runs"][0]["status"] == "succeeded", detail
    finding = detail["findings"][0]
    assert (
        finding["category"] == "physical" and finding["title"] == "USGS 3DEP bare-earth elevation"
    )
    assert "bare-earth elevation" in finding["summary"] and "(NAVD88)" in finding["summary"]
    assert detail["artifacts"][0]["title"] == "Bare-earth elevation and slope (USGS 3DEP)"
    assert len(detail["evidence"]) == 1 and detail["evidence"][0]["observedAt"] is None
    assert detail["evidence"][0]["provider"] == "usgs-3dep-13"
    raster_id = detail["artifacts"][0]["output"]["rasterId"]
    saved = client.get(f"/api/v1/land/rasters/{raster_id}").json()
    assert saved["request"] == {
        "dataset": "usgs-3dep-13",
        "resolutionM": 10.0,
        "maxDimension": 512,
        "periods": [],
    }
