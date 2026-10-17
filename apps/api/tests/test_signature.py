from __future__ import annotations

import uuid
from datetime import date
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyproj import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import transform
from sqlalchemy.orm import Session, sessionmaker

from app.analysis import vegetation
from app.analysis.signature import analyze
from app.analysis.vegetation import BANDS, COLLECTION, PREFIX
from app.schemas.land_rasters import RasterMetadata, RasterRequest, VegetationPeriod
from tests.test_raster_io import response as range_response
from tests.test_vegetation import BOUNDARY

# Synthetic scenes over the public National Mall fixture; no private source data.
WINDOWS = {"2025-04": "0416", "2025-07": "0716"}
# Digital numbers; reflectance = DN * 1e-4 - 0.1.
DN = {"green": 1500, "red": 3000, "nir": 7000, "swir16": 4000}
# A clear cell (east of every cloud column) and cells the quality mask excludes.
CLEAR = (-77.0460, 38.8875)
CLOUDY = (-77.0540, 38.8890)  # column 24: cloudy in both windows
WATER = (-77.0472, 38.8895)  # SCL 6 in the April scene only


def request(*months: str) -> RasterRequest:
    return RasterRequest(
        dataset="sentinel-2-signature",
        resolution_m=10,
        periods=[
            VegetationPeriod(
                start_date=date.fromisoformat(f"{month}-01"),
                end_date=date.fromisoformat(f"{month}-30"),
            )
            for month in (months or tuple(WINDOWS))
        ],
    )


def fixture_client(
    *, empty: str = "", resolution: dict[str, int] | None = None, counter: list[int] | None = None
) -> httpx.Client:
    rasters: dict[str, bytes] = {}
    scenes: dict[str, list[dict[str, Any]]] = {}
    for month, day in WINDOWS.items():
        candidates = []
        for cloudy in (True, False):
            stamp = f"{month.replace('-', '')}{day[2:] if not cloudy else '01'}"
            identifier = f"S2B_T18SUJ_{stamp}T160531_L2A"
            assets = {}
            for key, band in BANDS.items():
                url = f"{PREFIX}18/S/UJ/{month[:4]}/{int(month[5:])}/{identifier}/{band}.tif"
                dtype = "uint8" if key == "scl" else "uint16"
                if key == "scl":
                    values = np.full((128, 128), 9 if cloudy else 4, dtype=np.uint8)
                    if not cloudy:
                        values[:, : 25 if month == "2025-04" else 35] = 9
                        values[52:, :] = 5  # Not vegetated is still a clear land sample.
                        if month == "2025-04":
                            values[41:44, 50:54] = 6
                else:
                    values = np.full((128, 128), DN[key], dtype=np.uint16)
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
                    "spatial_resolution": (resolution or {}).get(key, vegetation.RESOLUTION_M[key]),
                }
                if key != "scl":
                    info.update(scale=0.0001, offset=-0.1)
                assets[key] = {"href": url, "raster:bands": [info]}
            candidates.append(
                {
                    "id": identifier,
                    "collection": COLLECTION,
                    "geometry": BOUNDARY.model_dump(),
                    "properties": {
                        "datetime": f"{month}-{stamp[-2:]}T16:05:31Z",
                        "eo:cloud_cover": 0 if cloudy else 70,
                    },
                    "assets": assets,
                }
            )
        scenes[month] = candidates

    def serve(req: httpx.Request) -> httpx.Response:
        if counter is not None:
            counter.append(1)
        if req.url.path.endswith("/search"):
            month = req.url.params["datetime"][:7]
            items = [] if month == empty else scenes[month]
            return httpx.Response(
                200, json={"features": items, "numberMatched": len(items), "links": []}
            )
        return range_response(req, data=rasters[str(req.url)])

    return httpx.Client(transport=httpx.MockTransport(serve))


def masks_at(data: Any, dataset: Any, lon: float, lat: float) -> list[bool]:
    xs, ys = transform("EPSG:4326", dataset.crs, [lon], [lat])
    row, col = dataset.index(xs[0], ys[0])
    return [bool(data.mask[band, row, col]) for band in range(data.shape[0])]


def test_signature_stacks_masked_bands_per_window_with_dated_scenes_and_sources() -> None:
    with fixture_client() as client:
        result = analyze(BOUNDARY, request(), client)
    metadata = result.metadata
    info, series = metadata.signature, metadata.vegetation
    assert info is not None and series is not None
    assert [band.name for band in metadata.bands[:5]] == [
        "2025-04-16 Green (B03)",
        "2025-04-16 Red (B04)",
        "2025-04-16 NIR (B08)",
        "2025-04-16 SWIR (B11)",
        "2025-04-16 NDVI",
    ]
    assert [band.unit for band in metadata.bands] == (["reflectance"] * 4 + ["NDVI"]) * 2
    assert [(f.band, f.window, f.key) for f in info.features] == [
        (index + 1, index // 5, key)
        for index, key in enumerate(["green", "red", "nir", "swir16", "ndvi"] * 2)
    ]
    assert metadata.bands[4].display_minimum == -1 and metadata.bands[0].display_maximum == 0.6
    # Dated scenes, VegetationObservation-shaped, keyed to each window's NDVI band.
    assert [o.band for o in series.observations] == [5, 10]
    assert [o.acquired_at.date().isoformat() for o in series.observations if o.acquired_at] == [
        "2025-04-16",
        "2025-07-16",
    ]
    assert all(o.scene_id and "T18SUJ" in o.scene_id for o in series.observations)
    assert series.observations[0].quality_counts["Water"] > 0
    assert series.observations[0].quality_counts["Not vegetated"] > 0
    assert series.change_band is None and series.common_cells == info.common_cells
    assert 0 < info.common_cells < series.observations[0].valid_cells < metadata.boundary_cells
    nir = next(f for f in info.features if f.key == "nir")
    assert nir.common_mean == pytest.approx(0.6, abs=1e-6)
    assert nir.common_standard_deviation == pytest.approx(0, abs=1e-6)
    assert series.observations[0].common_mean_ndvi == pytest.approx(0.5, abs=1e-6)
    # 2 catalog searches, 2 x 2 quality masks, 2 x 4 reflectance files.
    assert len(metadata.sources) == 14
    assert all(source.etag == '"version-1"' for source in metadata.sources if source.band)
    assert {source.band for source in metadata.sources} >= {"green", "red", "nir", "swir16"}
    assert any("not a species identification" in warning for warning in metadata.warnings)
    assert any("B11 is 20 m" in warning for warning in metadata.warnings)
    assert any("relative to the variety on this land" in w for w in metadata.warnings)
    assert len(result.data) <= 16 * 1024 * 1024
    with MemoryFile(result.data) as memory, memory.open() as dataset:
        assert dataset.count == 10
        structure = dataset.tags(ns="IMAGE_STRUCTURE")
        assert structure["LAYOUT"] == "COG" and structure["INTERLEAVE"] == "BAND"
        assert RasterMetadata.model_validate_json(dataset.tags()["analysis_metadata"]) == metadata
        data = dataset.read(masked=True)
        for window in (0, 1):
            first = data[window * 5].mask
            assert all(
                np.array_equal(first, data[band].mask) for band in range(window * 5, window * 5 + 5)
            )
            assert data[window * 5].count() == series.observations[window].valid_cells
        assert int(np.count_nonzero(~np.any(data.mask, axis=0))) == info.common_cells
        assert np.allclose(data[2].compressed(), 0.6) and np.allclose(data[4].compressed(), 0.5)
        # Cloud and water cells are missing in every band of their window, never zero.
        assert all(masks_at(data, dataset, *CLOUDY))
        assert masks_at(data, dataset, *WATER)[:5] == [True] * 5
        assert masks_at(data, dataset, *WATER)[5:] == [False] * 5
        assert not any(masks_at(data, dataset, *CLEAR))


def test_an_empty_window_leaves_no_common_signature_and_says_so() -> None:
    with fixture_client(empty="2025-07") as client:
        result = analyze(BOUNDARY, request(), client)
    info = result.metadata.signature
    assert info is not None and info.common_cells == 0
    assert all(feature.common_mean is None for feature in info.features)
    assert result.metadata.vegetation is not None
    assert result.metadata.vegetation.observations[1].scene_id is None
    assert any("never filled by interpolation" in w for w in result.metadata.warnings)
    assert any("No cell is clear in every window" in w for w in result.metadata.warnings)


def test_source_resolutions_are_checked_per_band_and_two_masks_per_window() -> None:
    # B11 is 20 m at the source and is accepted; green declared at 20 m is not B03.
    with (
        fixture_client(resolution={"green": 20}) as client,
        pytest.raises(ValueError, match="unexpected type or source resolution"),
    ):
        analyze(BOUNDARY, request(), client)
    with (
        fixture_client(resolution={"swir16": 10}) as client,
        pytest.raises(ValueError, match="unexpected type or source resolution"),
    ):
        analyze(BOUNDARY, request(), client)
    calls: list[int] = []
    with fixture_client(counter=calls) as client:
        result = analyze(BOUNDARY, request(), client)
    assert result.metadata.vegetation is not None
    assert {o.quality_candidates_examined for o in result.metadata.vegetation.observations} == {2}


def test_signature_requests_need_two_to_six_chronological_windows() -> None:
    assert len(request().periods) == 2
    with pytest.raises(ValueError, match="two to six dated windows"):
        request("2025-04")
    with pytest.raises(ValueError, match="two to six dated windows"):
        RasterRequest(dataset="sentinel-2-signature", resolution_m=10)
    with pytest.raises(ValueError, match="chronological"):
        request("2025-07", "2025-04")
    with pytest.raises(ValueError, match="only supported"):
        RasterRequest(dataset="usgs-3dep-13", resolution_m=10, periods=request().periods)
    with pytest.raises(ValueError):
        RasterRequest(dataset="sentinel-2-signature", resolution_m=5, periods=request().periods)
    six = RasterRequest(
        dataset="sentinel-2-signature",
        resolution_m=10,
        periods=[
            VegetationPeriod(start_date=date(2025, m, 1), end_date=date(2025, m, 28))
            for m in range(1, 12, 2)
        ],
    )
    assert len(six.periods) == 6


def test_output_limit_names_the_signature_and_how_to_fix_it() -> None:
    with fixture_client() as client:
        metadata = analyze(BOUNDARY, request(), client).metadata
    size = 1024
    noise = np.random.default_rng(0).random((size, size), dtype=np.float32)
    grid = SimpleNamespace(
        width=size,
        height=size,
        crs=CRS.from_epsg(3857),
        affine=from_origin(0, 0, 10, 10),
    )
    with pytest.raises(ValueError, match=r"signature raster exceeds its output limit\. Use fewer"):
        vegetation.encode(
            [noise + index for index in range(5)],
            metadata.bands[:5],
            grid,  # type: ignore[arg-type]
            metadata,
            name="signature",
            interleave="BAND",
        )


def test_signature_run_persists_dated_evidence_and_keeps_its_windows(
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
        json={"title": "Seasonal 10 m signatures", "boundaryRevision": 1},
    ).json()["id"]
    created = client.post(
        f"/api/v1/research/investigations/{investigation}/runs",
        json={
            "kind": "raster",
            "requestKey": str(uuid.uuid4()),
            "question": "Prepare seasonal signatures",
            "analysis": request().model_dump(mode="json", by_alias=True),
        },
    )
    assert created.status_code == 201, created.text
    db.rollback()
    assert ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{investigation}").json()
    assert detail["runs"][0]["status"] == "succeeded", detail
    finding = detail["findings"][0]
    assert finding["category"] == "ecology"
    assert "2 of 2 seasonal windows" in finding["summary"]
    assert "acquired 2025-04-16, 2025-07-16" in finding["summary"]
    assert "not the same species" in finding["summary"]
    assert detail["artifacts"][0]["title"] == "Seasonal 10 m signatures"
    assert len(detail["evidence"]) == 14
    # Every scene file is dated by its own acquisition, quality-mask candidates included.
    dated = {e["observedAt"][:10] for e in detail["evidence"] if e["observedAt"]}
    assert dated == {"2025-04-01", "2025-04-16", "2025-07-01", "2025-07-16"}
    raster_id = detail["artifacts"][0]["output"]["rasterId"]
    base = f"/api/v1/land/rasters/{raster_id}"
    saved = client.get(base).json()
    assert saved["request"]["dataset"] == "sentinel-2-signature"
    assert [p["startDate"] for p in saved["request"]["periods"]] == ["2025-04-01", "2025-07-01"]
    assert saved["metadata"]["signature"]["commonCells"] > 0
    assert saved["metadata"]["vegetation"]["observations"][1]["band"] == 10
    sample = client.get(base + "/sample", params={"longitude": CLEAR[0], "latitude": CLEAR[1]})
    assert sample.json()["values"][4] == pytest.approx(0.5, abs=1e-6)
    tile = TMS.tile(*CLEAR, 14)
    assert client.get(f"{base}/tiles/10/{tile.z}/{tile.x}/{tile.y}.png").status_code == 200


def test_signature_runs_get_the_longest_processing_time() -> None:
    from app.analysis.raster_runner import wall_seconds

    assert wall_seconds(request()) == 200
    assert wall_seconds(RasterRequest(dataset="sentinel-2-ndvi", periods=request().periods)) == 165
    assert wall_seconds(RasterRequest(dataset="usgs-3dep-13", resolution_m=10)) == 90
    assert wall_seconds(RasterRequest()) == 90
