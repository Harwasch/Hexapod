"""Show once, everywhere: one tapped cell of a stored signature raster against every other,
and the alike cells as areas for a plan (`app.analysis.similarity`, `land_similarity`)."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pytest
from affine import Affine
from fastapi.testclient import TestClient
from pyproj import Transformer
from rasterio.io import MemoryFile
from shapely.geometry import Point, shape
from shapely.geometry import Polygon as ShapelyPolygon
from sqlalchemy.orm import Session

from app.analysis.lonlat_grid import decode
from app.analysis.raster_grid import AnalysisGrid, make_grid
from app.analysis.signature import FeatureKey
from app.analysis.similarity import (
    Patch,
    build_stack,
    quantize,
    raw_threshold,
    resolution_label,
    shares,
)
from app.analysis.terrain import band_statistics
from app.models.land_raster import LandRaster, LandRasterBlob
from app.schemas.geojson import Polygon
from app.schemas.grids import LonLatGrid
from app.schemas.land_rasters import (
    RasterMetadata,
    RasterRequest,
    SignatureFeature,
    SignatureInfo,
    VegetationObservation,
    VegetationPeriod,
    VegetationTimeline,
)
from app.services.land_similarity import _zone
from tests.test_land import BODY, polygon
from tests.test_land_knowledge import research_run
from tests.test_workspaces import IdentityClient, headers
from tests.test_workspaces import identity_client as _identity_client

identity_client = _identity_client
KEYS: tuple[FeatureKey, ...] = ("green", "red", "nir", "swir16", "ndvi")
# Two seasons of two kinds of ground: A (green in spring, dry in summer) and B (bare).
A = [(0.05, 0.04, 0.30, 0.20, 0.76), (0.09, 0.12, 0.22, 0.31, 0.29)]
B = [(0.09, 0.12, 0.20, 0.30, 0.25), (0.08, 0.10, 0.19, 0.28, 0.31)]
PERIODS = [
    VegetationPeriod(start_date=date(2025, 4, 1), end_date=date(2025, 4, 30)),
    VegetationPeriod(start_date=date(2025, 7, 1), end_date=date(2025, 7, 30)),
]
REQUEST = RasterRequest(dataset="sentinel-2-signature", resolution_m=10, periods=PERIODS)
# Rows and columns of the analysis grid: three A patches (the first around a B island), an A
# speck, and a block with no full signature (cloud in the first window).
A_BIG = (slice(10, 50), slice(5, 35))
ISLAND = (slice(18, 23), slice(12, 17))
A_SOUTH = (slice(70, 90), slice(45, 65))
A_WEST = (slice(70, 80), slice(5, 15))
SPECK = (slice(100, 101), slice(10, 12))
CLOUD = (slice(58, 64), slice(5, 15))


def request_at(resolution_m: float) -> RasterRequest:
    return REQUEST.model_copy(update={"resolution_m": resolution_m})


def grid(resolution_m: float = 10) -> AnalysisGrid:
    return make_grid(Polygon.model_validate(polygon()), request_at(resolution_m))


def lonlat(analysis: AnalysisGrid, row: float, column: float) -> tuple[float, float]:
    x, y = analysis.affine @ (column + 0.5, row + 0.5)
    to_geographic = Transformer.from_crs(analysis.crs, 4326, always_xy=True)
    longitude, latitude = to_geographic.transform(x, y)
    return float(longitude), float(latitude)


def classes(analysis: AnalysisGrid) -> np.ndarray:
    """1 where the ground is A, 0 where B (inside), -1 outside or cloudy."""
    kind = np.where(analysis.inside, 0, -1)
    for block in (A_BIG, A_SOUTH, A_WEST, SPECK):
        kind[block] = 1
    kind[ISLAND] = 0
    kind[CLOUD] = -1
    kind[~analysis.inside] = -1
    return kind


def signature_bytes(analysis: AnalysisGrid) -> tuple[bytes, list[np.ndarray]]:
    kind = classes(analysis)
    rng = np.random.default_rng(7)
    arrays: list[np.ndarray] = []
    for window in range(2):
        for index in range(5):
            values = np.where(kind == 1, A[window][index], B[window][index]).astype(np.float32)
            values += rng.normal(0, 0.002, values.shape).astype(np.float32)
            values[~analysis.inside] = np.nan
            if window == 0:
                values[CLOUD] = np.nan
            arrays.append(values)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=analysis.width,
            height=analysis.height,
            count=len(arrays),
            dtype="float32",
            crs=analysis.crs.to_wkt(),
            transform=analysis.affine,
            nodata=-9999.0,
        ) as dataset:
            for band, values in enumerate(arrays, start=1):
                dataset.write(np.where(np.isfinite(values), values, -9999.0), band)
        return memory.read(), arrays


def signature_raster(
    db: Session, land_id: str, *, terrain: bool = False, resolution_m: float = 10
) -> LandRaster:
    """A stored signature raster of the grid's two kinds of ground (or, with `terrain`, the
    same bytes filed as a terrain raster), on a 10 m grid or a coarser one."""
    analysis = grid(resolution_m)
    data, arrays = signature_bytes(analysis)
    common = analysis.inside & np.logical_and.reduce([np.isfinite(a) for a in arrays])
    bands = []
    features = []
    for band, values in enumerate(arrays, start=1):
        window, key = divmod(band - 1, 5)
        name = f"2025-0{4 + 3 * window}-15 {KEYS[key]}"
        bands.append(band_statistics(values, band, name, "NDVI" if key == 4 else "reflectance"))
        features.append(
            SignatureFeature(
                band=band,
                window=window,
                key=KEYS[key],
                name=name,
                common_mean=float(values[common].astype(np.float64).mean()),
                common_standard_deviation=float(values[common].astype(np.float64).std()),
            )
        )
    total = int(analysis.inside.sum())
    metadata = RasterMetadata(
        algorithm="sentinel-2-signature-v1",
        method="Synthetic two-class signature for tests.",
        bounds=list(analysis.bounds),
        crs=analysis.crs.to_wkt(),
        resolution_m=analysis.resolution,
        width=analysis.width,
        height=analysis.height,
        boundary_cells=total,
        valid_cells=total,
        coverage_fraction=1.0,
        sampled_area_m2=total * analysis.resolution**2,
        bands=bands,
        sources=[],
        warnings=[
            "Similar spectral signature at 10 m across 2 dates; not a species identification."
        ],
        downloaded_bytes=0,
        vegetation=VegetationTimeline(
            observations=[
                VegetationObservation(
                    period=period,
                    band=5 * (window + 1),
                    scene_id=f"S2B_TEST_{window}",
                    acquired_at=datetime(2025, 4 + 3 * window, 15, 18, 55, tzinfo=UTC),
                    candidate_count=1,
                    catalog_truncated=False,
                    quality_candidates_examined=1,
                    valid_cells=total,
                    coverage_fraction=1.0,
                    mean_ndvi=0.4,
                    common_mean_ndvi=0.4,
                    quality_counts={},
                    explanation="Synthetic.",
                )
                for window, period in enumerate(PERIODS)
            ],
            common_cells=int(common.sum()),
            common_coverage_fraction=float(common.sum() / total),
            change_band=None,
            change_cells=0,
            mean_change=None,
        ),
        signature=SignatureInfo(
            features=features,
            common_cells=int(common.sum()),
            common_coverage_fraction=float(common.sum() / total),
            standardization="z-score per feature",
            similarity="cosine of standardized vectors",
        ),
    )
    request = (
        RasterRequest(dataset="cop-dem-glo-30") if terrain else request_at(resolution_m)
    ).model_dump(mode="json")
    run = research_run(db, land_id, "raster")
    row = LandRaster(
        land_id=uuid.UUID(land_id),
        boundary_revision=1,
        run_id=run.id,
        request=request,
        metadata_json=metadata.model_dump(mode="json"),
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
    )
    db.add(row)
    db.flush()
    db.add(LandRasterBlob(raster_id=row.id, data=data))
    db.commit()
    return row


def make_land(client: TestClient, auth: dict[str, str] | None = None) -> dict[str, Any]:
    response = client.post("/api/v1/land", json=BODY, headers=auth)
    assert response.status_code == 201, response.text
    land: dict[str, Any] = response.json()
    return land


def display_value(read: dict[str, Any], longitude: float, latitude: float) -> int:
    values = decode(LonLatGrid.model_validate(read["grid"]))
    g = read["grid"]
    column = int((longitude - g["west"]) / (g["east"] - g["west"]) * g["columns"])
    row = int((g["north"] - latitude) / (g["north"] - g["south"]) * g["rows"])
    return int(values[row, column])


def center(block: tuple[slice, slice]) -> tuple[int, int]:
    rows, columns = block
    return (rows.start + rows.stop) // 2, (columns.start + columns.stop) // 2


def test_quantized_thresholds_match_the_console() -> None:
    assert raw_threshold(0.8) == 200
    assert raw_threshold(0.81) == 203  # 202.5 rounds up: the cells shown are at least 81%
    assert raw_threshold(0.3) == 75 and raw_threshold(0.99) == 248
    scores = np.array([[1.0, 0.8, -0.4, np.nan]], dtype=np.float32)
    assert quantize(scores).tolist() == [[250, 200, 0, 255]]


def test_constant_bands_and_mean_cells_are_not_compared() -> None:
    features = [
        SignatureFeature(
            band=band,
            window=(band - 1) // 5,
            key=KEYS[(band - 1) % 5],
            name=str(band),
            common_mean=None,
            common_standard_deviation=None,
        )
        for band in range(1, 11)
    ]
    data = np.ones((10, 1, 3), dtype=np.float32)
    data[0] = [[0.0, 1.0, 2.0]]  # one band varies; the cell at its mean has no direction
    stack = build_stack(data, features, Affine.identity(), None, None)
    assert int(stack.used.sum()) == 1
    assert stack.cells.tolist() == [0, 2]
    assert np.allclose(stack.values(1), [2.0, *[1.0] * 9])


def test_similar_lights_the_alike_ground(client: TestClient, db: Session) -> None:
    land = make_land(client)
    raster = signature_raster(db, land["id"])
    analysis = grid()
    seed = lonlat(analysis, *center(A_BIG))
    path = f"/api/v1/land/rasters/{raster.id}/similar"
    response = client.get(path, params={"longitude": seed[0], "latitude": seed[1]})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    read = response.json()
    assert read["stale"] is False and read["boundaryRevision"] == 1
    assert read["grid"]["encoding"] == "u8" and read["grid"]["nodata"] == 255
    assert read["cellAreaM2"] == pytest.approx(100, rel=0.01)
    assert read["comparedCells"] == raster.metadata_json["signature"]["common_cells"]
    for block in (A_BIG, A_SOUTH, A_WEST):
        assert display_value(read, *lonlat(analysis, *center(block))) >= 245
    assert display_value(read, *lonlat(analysis, *center(ISLAND))) <= 10
    assert display_value(read, *lonlat(analysis, 40, 60)) <= 10  # B ground
    assert display_value(read, *lonlat(analysis, *center(CLOUD))) == 255
    # The seed is the cell it fell in, with its values per dated window.
    assert (
        shape({"type": "Point", "coordinates": seed}).distance(
            Point(read["seed"]["cellLongitude"], read["seed"]["cellLatitude"])
        )
        < 0.0001
    )
    windows = read["seed"]["windows"]
    assert [window["acquiredAt"][:10] for window in windows] == ["2025-04-15", "2025-07-15"]
    assert windows[0]["ndvi"] == pytest.approx(0.76, abs=0.01)
    assert windows[1]["swir16"] == pytest.approx(0.31, abs=0.01)
    assert [scene["sceneId"] for scene in read["scenes"]] == ["S2B_TEST_0", "S2B_TEST_1"]
    assert "Not a species ID" in read["interpretation"]
    assert read["interpretation"].startswith("Similar seasonal satellite signature at 10 m.")
    assert read["nativeResolutionM"] == pytest.approx(10)

    outside = client.get(path, params={"longitude": -122.15, "latitude": 47.645})
    assert outside.status_code == 422 and "outside" in outside.json()["detail"]
    cloud = lonlat(analysis, *center(CLOUD))
    cloudy = client.get(path, params={"longitude": cloud[0], "latitude": cloud[1]})
    assert cloudy.status_code == 422
    assert cloudy.json()["detail"].startswith("That 10 m cell has no full signature")

    revised = client.put(
        f"/api/v1/land/{land['id']}",
        json={**BODY, "boundary": polygon(-122.141, -122.13), "expectedRevision": 1},
    )
    assert revised.status_code == 200, revised.text
    stale = client.get(path, params={"longitude": seed[0], "latitude": seed[1]}).json()
    assert stale["stale"] is True and "boundary changed" in stale["warnings"][0]


def test_a_coarsened_grid_is_named_at_its_spacing(client: TestClient, db: Session) -> None:
    """A land too wide for 506 cells of 10 m is analysed on a coarser grid (`make_grid`):
    the interpretation, the errors and the areas say that spacing, never 10 m."""
    land = make_land(client)
    raster = signature_raster(db, land["id"], resolution_m=12)
    analysis = grid(12)
    seed = lonlat(analysis, *center(A_BIG))
    path = f"/api/v1/land/rasters/{raster.id}/similar"
    read = client.get(path, params={"longitude": seed[0], "latitude": seed[1]}).json()
    assert read["nativeResolutionM"] == pytest.approx(12)
    assert read["interpretation"].startswith("Similar seasonal satellite signature at 12 m.")
    cloud = lonlat(analysis, *center(CLOUD))
    cloudy = client.get(path, params={"longitude": cloud[0], "latitude": cloud[1]})
    assert cloudy.status_code == 422
    assert cloudy.json()["detail"].startswith("That 12 m cell has no full signature")
    zones = client.post(
        f"{path}/zones", json={"longitude": seed[0], "latitude": seed[1], "threshold": 0.9}
    ).json()
    assert zones["nativeResolutionM"] == pytest.approx(12)
    assert zones["cellAreaM2"] == pytest.approx(144)
    assert "12 m" in zones["interpretation"] and "10 m" not in zones["interpretation"]
    assert resolution_label(11.857) == "11.9 m" and resolution_label(10.0) == "10 m"


def test_only_signature_rasters_compare(client: TestClient, db: Session) -> None:
    land = make_land(client)
    raster = signature_raster(db, land["id"], terrain=True)
    response = client.get(
        f"/api/v1/land/rasters/{raster.id}/similar",
        params={"longitude": -122.135, "latitude": 47.645},
    )
    assert response.status_code == 409 and "signature" in response.json()["detail"]


def test_zones_union_alike_cells_with_holes_protection_and_a_remainder(
    client: TestClient, db: Session
) -> None:
    land = make_land(client)
    raster = signature_raster(db, land["id"])
    analysis = grid()
    seed = lonlat(analysis, *center(A_BIG))
    path = f"/api/v1/land/rasters/{raster.id}/similar/zones"
    body = {"longitude": seed[0], "latitude": seed[1], "threshold": 0.9}
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    read = response.json()
    big, south, west = read["zones"]
    island = 25
    expected = {
        "big": (40 * 30 - island) * 100,
        "south": 20 * 20 * 100,
        "west": 10 * 10 * 100,
    }
    assert big["areaM2"] == pytest.approx(expected["big"], rel=0.05)
    assert south["areaM2"] == pytest.approx(expected["south"], rel=0.05)
    assert west["areaM2"] == pytest.approx(expected["west"], rel=0.05)
    assert big["cells"] == 40 * 30 - island and big["meanSimilarity"] > 0.95
    # The B island is a hole in the big patch.
    outline = shape(big["footprint"])
    assert len(big["footprint"]["coordinates"]) == 2
    assert not outline.contains(Point(*lonlat(analysis, *center(ISLAND))))
    assert outline.contains(Point(*seed))
    # The two-cell speck is under half an acre: left out, and counted.
    assert read["dropped"] == {"patches": 1, "areaM2": pytest.approx(200, rel=0.3)}
    assert read["remainder"] is None
    assert read["matchedCells"] == 40 * 30 - island + 400 + 100 + 2
    assert read["protectedCells"] == 0

    # Protecting the south patch (its cells, as the console sends them) takes it out.
    rows, columns = A_SOUTH
    corners = [
        lonlat(analysis, rows.start - 0.5, columns.start - 0.5),
        lonlat(analysis, rows.stop - 0.5, columns.stop - 0.5),
    ]
    rectangle = [
        min(c[0] for c in corners),
        min(c[1] for c in corners),
        max(c[0] for c in corners),
        max(c[1] for c in corners),
    ]
    guarded = client.post(path, json={**body, "protected": [rectangle]}).json()
    assert len(guarded["zones"]) == 2
    assert guarded["protectedCells"] == 400
    assert not any(
        shape(zone["footprint"]).intersects(Point(*lonlat(analysis, *center(A_SOUTH))))
        for zone in guarded["zones"]
    )
    # Protecting ground inside a patch leaves a hole where it was.
    inner = lonlat(analysis, 40, 12)
    near = [inner[0] - 0.0001, inner[1] - 0.0001, inner[0] + 0.0001, inner[1] + 0.0001]
    holed = client.post(path, json={**body, "protected": [near]}).json()
    assert len(holed["zones"][0]["footprint"]["coordinates"]) == 3
    assert not shape(holed["zones"][0]["footprint"]).contains(Point(*inner))
    assert holed["zones"][0]["areaM2"] < big["areaM2"]
    assert holed["protectedCells"] > 0

    # One zone of its own: the rest join one remainder area.
    capped = client.post(path, json={**body, "maxZones": 1}).json()
    assert len(capped["zones"]) == 1
    remainder = capped["remainder"]
    assert remainder["footprint"]["type"] == "MultiPolygon" and remainder["patches"] == 2
    assert remainder["areaM2"] == pytest.approx(expected["south"] + expected["west"], rel=0.05)
    assert remainder["cells"] == 500

    # Without a minimum patch the speck is kept, and stricter alike is still the same ground.
    everything = client.post(path, json={**body, "minPatchM2": 0, "threshold": 0.3}).json()
    assert everything["dropped"]["patches"] == 0 and len(everything["zones"]) == 4
    assert client.post(path, json={**body, "threshold": 0.2}).status_code == 422


def test_similarity_is_scoped_to_the_workspace(
    identity_client: IdentityClient, db: Session
) -> None:
    client, token = identity_client
    alice = headers(token())
    workspace = client.post("/api/v1/workspaces", headers=alice, json={"name": "Ranch"}).json()[
        "id"
    ]
    alice = headers(token(), workspace)
    land = make_land(client, alice)
    raster = signature_raster(db, land["id"])
    seed = lonlat(grid(), *center(A_BIG))
    query = {"longitude": seed[0], "latitude": seed[1]}
    path = f"/api/v1/land/rasters/{raster.id}/similar"
    assert client.get(path, params=query, headers=alice).status_code == 200
    bob = headers(token("bob"))
    other = client.post("/api/v1/workspaces", headers=bob, json={"name": "Other"}).json()["id"]
    bob = headers(token("bob"), other)
    assert client.get(path, params=query, headers=bob).status_code == 404
    assert client.post(path + "/zones", json=query, headers=bob).status_code == 404


def _patch(footprint: ShapelyPolygon, cells: int = 10) -> Patch:
    return Patch(footprint=footprint, area_m2=footprint.area, cells=cells, mean_similarity=0.9)


def test_patches_that_rounding_breaks_are_repaired_or_counted() -> None:
    """Footprints are stored to 8 decimals. A patch whose rounding pinches its ring is snapped
    to that precision and kept; one narrower than the rounding cannot be an area, and is
    counted as left out rather than dropped without a word."""
    # A notch from the north edge reaching 3e-9 degrees from the south edge's middle vertex:
    # valid as computed, a self-touching ring once rounded to 8 decimals.
    pinched = ShapelyPolygon(
        [
            (0, 0),
            (5e-6, 0),
            (1e-5, 0),
            (1e-5, 1e-5),
            (6e-6, 1e-5),
            (5e-6, 3e-9),
            (4e-6, 1e-5),
            (0, 1e-5),
        ]
    )
    assert pinched.is_valid
    zone, left = _zone([_patch(pinched)])
    assert zone is not None and left == []
    assert shape(zone.footprint.model_dump()).is_valid
    assert zone.cells == 10 and zone.patches == 1

    sliver = ShapelyPolygon([(0, 0), (3e-9, 0), (3e-9, 3e-9), (0, 3e-9)])
    assert sliver.is_valid and sliver.area > 0
    lost = _patch(sliver, cells=1)
    zone, left = _zone([lost])
    assert zone is None and left == [lost]

    # In a remainder, the sliver is left out and counted; the rest is still one area.
    square = ShapelyPolygon([(1e-4, 0), (2e-4, 0), (2e-4, 1e-4), (1e-4, 1e-4)])
    kept = _patch(square, cells=40)
    remainder, left = _zone([kept, lost])
    assert remainder is not None and left == [lost]
    assert remainder.patches == 1 and remainder.cells == 40
    assert remainder.area_m2 == pytest.approx(square.area)


def test_a_split_part_shares_its_cells() -> None:
    assert shares(10, [3.0, 1.0]) == [8, 2]
    assert shares(7, [1.0, 1.0, 1.0]) == [3, 2, 2]
    assert sum(shares(101, [0.4, 0.35, 0.25])) == 101
    assert shares(5, [0.0, 0.0]) == [5, 0]
    assert shares(0, []) == []
