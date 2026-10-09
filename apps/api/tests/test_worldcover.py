from __future__ import annotations

import io
import uuid

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from sqlalchemy.orm import Session, sessionmaker

from app.analysis.terrain import TerrainResult
from app.analysis.worldcover import analyze
from app.config import Settings
from app.models.research import ResearchRun
from app.research.worker import ResearchWorker
from app.schemas.land_rasters import RasterRequest
from app.services.land_rasters import TMS
from tests.test_land_rasters import start
from tests.test_raster_io import response as range_response
from tests.test_terrain import BOUNDARY


def fixture_client() -> httpx.Client:
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=128,
            height=128,
            count=1,
            dtype="uint8",
            nodata=0,
            crs="EPSG:4326",
            transform=from_origin(-122.15, 47.66, 0.00025, 0.00025),
        ) as source:
            values = np.full((128, 128), 10, dtype=np.uint8)
            values[:, 55:65] = 50
            values[:, 65:75] = 80
            values[:, 75:80] = 255  # Unrecognized values must not become an invented class.
            source.write(values, 1)
        data = memory.read()

    def serve(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".geojson"):
            return httpx.Response(
                200,
                json={
                    "features": [
                        {
                            "properties": {"ll_tile": "N45W123"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [[-123, 45], [-120, 45], [-120, 48], [-123, 48], [-123, 45]]
                                ],
                            },
                        }
                    ]
                },
            )
        return range_response(request, data=data)

    return httpx.Client(transport=httpx.MockTransport(serve))


@pytest.fixture
def cover(monkeypatch: pytest.MonkeyPatch) -> TerrainResult:
    with fixture_client() as client:
        result = analyze(
            BOUNDARY, RasterRequest(dataset="esa-worldcover-2021", resolution_m=10), client
        )
    monkeypatch.setattr("app.research.worker.run_raster", lambda *args: result)
    return result


def test_cover_preserves_class_codes_masks_unknowns_and_exports_a_categorical_cog(
    cover: TerrainResult,
) -> None:
    metadata = cover.metadata
    assert 0 < metadata.valid_cells < metadata.boundary_cells
    band = metadata.bands[0]
    assert band.mean is None and band.minimum is None and band.palette == "categorical"
    assert sum(item.cells for item in band.classes) == band.valid_cells
    assert sum(item.fraction or 0 for item in band.classes) == pytest.approx(1)
    assert sum(item.sampled_area_m2 for item in band.classes) == metadata.sampled_area_m2
    assert {item.code for item in band.classes if item.cells} == {10, 50, 80}
    assert any("unrecognized" in message for message in metadata.warnings)
    with MemoryFile(cover.data) as memory, memory.open() as raster:
        assert raster.dtypes == ("uint8",)
        assert set(np.unique(raster.read(1))) == {0, 10, 50, 80}
        assert raster.colormap(1)[10][:3] == (0, 100, 0)
        assert raster.tags()["reference_year"] == "2021"
        assert raster.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
    with pytest.raises(ValueError, match="30 m"):
        RasterRequest(dataset="cop-dem-glo-30", resolution_m=10)


def test_cover_publishes_ecological_evidence_and_exact_class_colors(
    client: TestClient, db: Session, sessions: sessionmaker[Session], cover: TerrainResult
) -> None:
    _, inv, run, _ = start(client)
    row = db.get(ResearchRun, uuid.UUID(run["id"]))
    assert row is not None
    row.analysis = RasterRequest(dataset="esa-worldcover-2021", resolution_m=10).model_dump()
    db.commit()
    assert ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv}").json()
    assert detail["runs"][0]["status"] == "succeeded", detail
    finding = detail["findings"][0]
    assert finding["category"] == "ecology" and "2021" in finding["summary"]
    assert "not a species survey" in finding["summary"]
    raster_id = detail["artifacts"][0]["output"]["rasterId"]
    tile = TMS.tile(-122.132, 47.648, 14)
    result = client.get(f"/api/v1/land/rasters/{raster_id}/tiles/1/{tile.z}/{tile.x}/{tile.y}.png")
    assert result.status_code == 200
    pixels = np.array(Image.open(io.BytesIO(result.content)))
    colors = {tuple(pixel[:3]) for pixel in pixels.reshape(-1, 4) if pixel[3]}
    assert colors and colors.issubset({(0, 100, 0), (250, 0, 0), (0, 100, 200)})
