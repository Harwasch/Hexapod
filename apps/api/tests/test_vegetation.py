from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from sqlalchemy.orm import Session, sessionmaker

from app.analysis.vegetation import BANDS, COLLECTION, PREFIX, analyze, ndvi
from app.schemas.geojson import Polygon
from app.schemas.land_rasters import RasterMetadata, RasterRequest, VegetationPeriod
from tests.test_raster_io import response as range_response

# Synthetic imagery over the public National Mall fixture; no private source data.
BOUNDARY = Polygon(
    coordinates=[
        [
            [-77.055, 38.886],
            [-77.045, 38.886],
            [-77.045, 38.891],
            [-77.055, 38.891],
            [-77.055, 38.886],
        ],
        [
            [-77.053, 38.887],
            [-77.052, 38.887],
            [-77.052, 38.888],
            [-77.053, 38.888],
            [-77.053, 38.887],
        ],
    ]
)


def request() -> RasterRequest:
    return RasterRequest(
        dataset="sentinel-2-ndvi",
        resolution_m=20,
        periods=[
            VegetationPeriod(start_date=date(2024, 7, 1), end_date=date(2024, 7, 31)),
            VegetationPeriod(start_date=date(2025, 7, 1), end_date=date(2025, 7, 31)),
        ],
    )


def fixture_client(*, empty_year: str = "", bad_scale: bool = False) -> httpx.Client:
    rasters = {}
    scenes = {}
    for year in ("2024", "2025"):
        candidates = []
        for cloudy in (True, False):
            day = "15" if cloudy else "16"
            identifier = f"S2B_T18SUJ_{year}07{day}T160531_L2A"
            assets = {}
            for key, band in BANDS.items():
                url = f"{PREFIX}18/S/UJ/{year}/7/{identifier}/{band}.tif"
                dtype = "uint8" if key == "scl" else "uint16"
                if key == "scl":
                    values = np.full((128, 128), 9 if cloudy else 4, dtype=np.uint8)
                    if not cloudy:
                        values[:, : 35 if year == "2024" else 25] = 9
                else:
                    values = np.full(
                        (128, 128),
                        3000 if key == "red" else (7000 if year == "2024" else 9000),
                        dtype=np.uint16,
                    )
                with MemoryFile() as memory:
                    with memory.open(
                        driver="GTiff",
                        width=128,
                        height=128,
                        count=1,
                        dtype=dtype,
                        crs="EPSG:4326",
                        transform=from_origin(-77.06, 38.90, 0.00025, 0.00025),
                        nodata=0,
                    ) as dataset:
                        dataset.write(values, 1)
                    rasters[url] = memory.read()
                info: dict[str, Any] = {
                    "nodata": 0,
                    "data_type": dtype,
                    "spatial_resolution": 20 if key == "scl" else 10,
                }
                if key != "scl":
                    info.update(scale=0.0001, offset=-0.1)
                    if bad_scale:
                        del info["offset"]
                assets[key] = {"href": url, "raster:bands": [info]}
            candidates.append(
                {
                    "id": identifier,
                    "collection": COLLECTION,
                    "geometry": BOUNDARY.model_dump(),
                    "properties": {
                        "datetime": f"{year}-07-{day}T16:05:31Z",
                        "eo:cloud_cover": 0 if cloudy else 70,
                    },
                    "assets": assets,
                }
            )
        scenes[year] = candidates

    def serve(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/search"):
            year = req.url.params["datetime"][:4]
            items = [] if year == empty_year else scenes[year]
            return httpx.Response(
                200, json={"features": items, "numberMatched": len(items), "links": []}
            )
        return range_response(req, data=rasters[str(req.url)])

    return httpx.Client(transport=httpx.MockTransport(serve))


def test_uses_scaled_reflectance_local_quality_and_common_cells() -> None:
    with fixture_client() as client:
        result = analyze(BOUNDARY, request(), client)
    metadata = result.metadata
    series = metadata.vegetation
    assert series is not None
    first, last = series.observations
    assert (
        first.scene_id is not None and "0716" in first.scene_id
    )  # local SCL beats whole-scene cloud metadata
    assert first.mean_ndvi == pytest.approx(0.5, abs=1e-6)
    assert last.mean_ndvi == pytest.approx(0.6, abs=1e-6)
    assert 0 < first.valid_cells < last.valid_cells < metadata.boundary_cells
    assert series.common_cells == series.change_cells == first.valid_cells
    assert series.mean_change == pytest.approx(0.1, abs=1e-6)
    assert first.common_mean_ndvi == pytest.approx(0.5, abs=1e-6)
    assert len(metadata.sources) == 10
    assert all(s.catalog_sha256 for s in metadata.sources)
    assert all(s.etag == '"version-1"' for s in metadata.sources if s.band)
    with MemoryFile(result.data) as memory, memory.open() as dataset:
        assert dataset.count == 3 and dataset.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        data = dataset.read(masked=True)
        assert data[0].count() == first.valid_cells and data[2].count() == series.change_cells
        assert np.all(data[2].mask | (~data[0].mask & ~data[1].mask))
        assert data[0].mask.any()
        assert dataset.units == ("NDVI", "NDVI", "NDVI difference")
        exported = RasterMetadata.model_validate_json(dataset.tags()["analysis_metadata"])
        assert exported == metadata
        assert dataset.tags()["license_url"].startswith("https://sentinels.copernicus.eu/")


def test_missing_observation_remains_missing_and_provides_catalog_evidence() -> None:
    with fixture_client(empty_year="2025") as client:
        result = analyze(BOUNDARY, request(), client)
    series = result.metadata.vegetation
    assert series is not None and series.common_cells == series.change_cells == 0
    assert series.mean_change is None
    assert series.observations[1].scene_id is None and series.observations[1].mean_ndvi is None
    assert all(o.common_mean_ndvi is None for o in series.observations)
    assert result.metadata.sources[-1].purpose == "bounded scene selection"
    assert result.metadata.bands[1].valid_cells == result.metadata.bands[2].valid_cells == 0


def test_ndvi_excludes_invalid_reflectance_and_nonland_quality_without_false_zero() -> None:
    red = np.array([[0.2, 0.2, 0.2, -0.1], [0, 0.2, np.nan, 0.2]], dtype=np.float32)
    nir = np.array([[0.6, 0.6, 0.6, 0.6], [0, 0.2, 0.6, 0.6]], dtype=np.float32)
    scl = np.array([[4, 9, 6, 4], [4, 5, 4, 4]], dtype=np.float32)
    inside = np.full(red.shape, True)
    inside[1, 3] = False
    result = ndvi(red, nir, scl, inside)
    assert result[0, 0] == pytest.approx(0.5) and result[1, 1] == 0
    assert np.count_nonzero(np.isfinite(result)) == 2


def test_reflectance_offsets_and_bounded_date_windows_are_required() -> None:
    with (
        fixture_client(bad_scale=True) as client,
        pytest.raises(ValueError, match="scale and offset"),
    ):
        analyze(BOUNDARY, request(), client)
    with pytest.raises(ValueError, match="dated windows"):
        RasterRequest(dataset="sentinel-2-ndvi")
    with pytest.raises(ValueError, match="chronological"):
        RasterRequest(dataset="sentinel-2-ndvi", periods=list(reversed(request().periods)))
    with pytest.raises(ValueError, match="31 days"):
        VegetationPeriod(start_date=date(2024, 1, 1), end_date=date(2024, 3, 1))
    with pytest.raises(ValueError, match="only supported"):
        RasterRequest(periods=request().periods)


def test_vegetation_run_persists_dates_evidence_and_durable_results(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import Settings
    from app.research.worker import ResearchWorker
    from app.services.land_rasters import TMS
    from tests.test_land import BODY

    with fixture_client() as http:
        result = analyze(BOUNDARY, request(), http)
    monkeypatch.setattr("app.research.worker.run_raster", lambda *args: result)
    land = client.post("/api/v1/land", json={**BODY, "boundary": BOUNDARY.model_dump()}).json()
    investigation = client.post(
        f"/api/v1/land/{land['id']}/investigations",
        json={"title": "Vegetation", "boundaryRevision": 1},
    ).json()["id"]
    endpoint = f"/api/v1/research/investigations/{investigation}/runs"
    body = {
        "kind": "raster",
        "requestKey": str(uuid.uuid4()),
        "question": "Compare dated vegetation",
        "analysis": request().model_dump(mode="json", by_alias=True),
    }
    created = client.post(endpoint, json=body)
    assert created.status_code == 201, created.text
    assert client.post(endpoint, json=body).json()["id"] == created.json()["id"]
    db.rollback()
    assert ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{investigation}").json()
    assert detail["runs"][0]["status"] == "succeeded", detail
    assert "NDVI" in detail["findings"][0]["summary"]
    assert detail["findings"][0]["category"] == "ecology"
    assert len(detail["evidence"]) == 10
    raster_id = detail["artifacts"][0]["output"]["rasterId"]
    base = f"/api/v1/land/rasters/{raster_id}"
    saved = client.get(base).json()
    assert saved["request"]["periods"][0]["startDate"] == "2024-07-01"
    assert saved["metadata"]["vegetation"]["changeCells"] > 0
    tile = TMS.tile(-77.047, 38.889, 14)
    response = client.get(f"{base}/tiles/3/{tile.z}/{tile.x}/{tile.y}.png")
    assert response.status_code == 200
    assert client.get(base + "/download").content == result.data
