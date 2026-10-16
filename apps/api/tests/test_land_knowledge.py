"""The Fog of Knowing (policy fog-v1): what counts, what it is worth, and where it lifts."""

from __future__ import annotations

import hashlib
import math
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from shapely.geometry import shape
from sqlalchemy.orm import Session

from app.analysis import knowledge
from app.analysis.lonlat_grid import decode, grid_for
from app.analysis.raster_grid import make_grid
from app.models.land_raster import LandRaster, LandRasterBlob
from app.models.research import Evidence, Investigation, ResearchRun
from app.schemas.geojson import Polygon
from app.schemas.grids import LonLatGrid
from app.schemas.land_rasters import RasterRequest, VegetationPeriod
from tests.test_land import BODY, polygon
from tests.test_land_features import FEATURE
from tests.test_workspaces import IdentityClient, headers
from tests.test_workspaces import identity_client as _identity_client

identity_client = _identity_client
POINT = (-122.135, 47.645)


def make_land(client: TestClient, body: dict[str, Any] | None = None) -> dict[str, Any]:
    response = client.post("/api/v1/land", json=body or BODY)
    assert response.status_code == 201, response.text
    land: dict[str, Any] = response.json()
    return land


def fog(client: TestClient, land_id: str, **params: Any) -> dict[str, Any]:
    response = client.get(f"/api/v1/land/{land_id}/knowledge", params=params)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    read: dict[str, Any] = response.json()
    return read


def grid_values(read: dict[str, Any], key: str) -> np.ndarray:
    return decode(LonLatGrid.model_validate(read[key]))


def cell(read: dict[str, Any], longitude: float, latitude: float) -> tuple[int, int]:
    grid = read["density"]
    column = math.floor(
        (longitude - grid["west"]) / (grid["east"] - grid["west"]) * grid["columns"]
    )
    row = math.floor((grid["north"] - latitude) / (grid["north"] - grid["south"]) * grid["rows"])
    return row, column


def east_of(longitude: float, latitude: float, metres: float) -> float:
    return longitude + metres / (111_320 * math.cos(math.radians(latitude)))


def inspect(client: TestClient, land_id: str, observed: datetime, **feature: Any) -> str:
    path = f"/api/v1/land/{land_id}/features"
    created = client.post(path, json={**FEATURE, "requestKey": str(uuid.uuid4()), **feature})
    assert created.status_code == 201, created.text
    feature_id: str = created.json()["id"]
    response = client.post(
        f"{path}/{feature_id}/inspections",
        json={
            "requestKey": str(uuid.uuid4()),
            "observedAt": observed.isoformat(),
            "condition": "good",
            "notes": "Checked on foot",
        },
    )
    assert response.status_code == 201, response.text
    return feature_id


def research_run(db: Session, land_id: str, kind: str = "overview") -> ResearchRun:
    investigation = Investigation(
        land_id=uuid.UUID(land_id),
        boundary_revision=1,
        title="Overview",
        question="What is here?",
        created_by="test",
    )
    db.add(investigation)
    db.flush()
    run = ResearchRun(
        investigation_id=investigation.id,
        request_key=uuid.uuid4(),
        question="What is here?",
        status="succeeded",
        kind=kind,
        budget={},
    )
    db.add(run)
    db.flush()
    return run


def add_evidence(db: Session, run: ResearchRun, **content: Any) -> Evidence:
    base = {
        "provider": "usda-soils",
        "title": "Soil map unit",
        "url": "https://example.test/soils",
        "license": "Public domain",
        "attribution": "USDA NRCS",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "excerpt": "Map unit 101",
        "spatial_relevance": "within",
        "relevance_note": "The map unit covers the land.",
    }
    row = Evidence(run_id=run.id, source_key=str(uuid.uuid4()), content={**base, **content})
    db.add(row)
    db.commit()
    return row


def test_policy_credit() -> None:
    now = datetime(2026, 10, 16, tzinfo=UTC)
    week = now - timedelta(days=7)
    assert knowledge.credit("satellite", week, now) == pytest.approx(0.45 * 0.5 ** (7 / 90))
    assert knowledge.arithmetic("satellite", week, now) == "0.45 x 0.5^(7/90) = 0.43"
    assert knowledge.credit("regional", None, now) == pytest.approx(0.20 * 0.25)
    assert knowledge.credit("centimetre", None, now) == pytest.approx(0.25)
    assert "undated" in knowledge.arithmetic("centimetre", None, now)
    # No published location: Q is capped at the regional class's, the half-life stays.
    assert knowledge.credit("field", week, now, located=False) == pytest.approx(
        0.20 * 0.5 ** (7 / 120)
    )
    assert knowledge.arithmetic("field", week, now, located=False).startswith("min(0.85, 0.20)")
    assert knowledge.class_for_resolution(0.0015) == "centimetre"
    assert knowledge.class_for_resolution(0.6) == "aerial"
    assert knowledge.class_for_resolution(10) == "satellite"
    assert knowledge.class_for_resolution(30) == "regional"
    grid = knowledge.make_fog_grid(shape(polygon()), 20)
    placed = knowledge.KnowledgeSource(
        key="capture",
        kind="capture",
        title="A scan placed to within 32 m",
        class_id="centimetre",
        resolution_m=0.01,
        observed_at=week,
        observed_label="",
        recorded_at=week,
        located=True,
        positional_uncertainty_m=32,
        provenance={},
        link={},
    )
    weighed = knowledge.weigh(grid, placed, now)
    assert weighed.class_id == "aerial" and "32 m" in (weighed.note or "")
    placed.positional_uncertainty_m = 3
    assert knowledge.weigh(grid, placed, now).class_id == "centimetre"


def test_the_grid_is_lon_lat_aligned_and_bounded() -> None:
    boundary = shape(polygon())
    grid = grid_for(boundary, 20)
    assert grid.columns <= 256 and grid.rows <= 256
    assert grid.west <= -122.14 and grid.east >= -122.13
    assert grid.south <= 47.64 and grid.north >= 47.65
    # Square on the ground: the degree steps differ by cos(latitude).
    assert grid.dlon / grid.dlat == pytest.approx(1 / math.cos(math.radians(47.645)), rel=0.01)
    huge = shape(
        {
            "type": "Polygon",
            "coordinates": [[[-120, 34], [-119.8, 34], [-119.8, 34.2], [-120, 34.2], [-120, 34]]],
        }
    )
    grown = knowledge.make_fog_grid(huge, 20)
    assert grown.spec.columns <= 256 and grown.spec.rows <= 256 and grown.cell_m > 20
    assert grid.cell_index(-122.135, 47.645) is not None
    assert grid.cell_index(-121, 47.645) is None


def test_empty_land_is_full_fog(client: TestClient) -> None:
    land = make_land(client)
    read = fog(client, land["id"])
    density = grid_values(read, "density")
    best = grid_values(read, "best")
    inside = density != 255
    assert inside.any() and (~inside).any()
    assert (density[inside] == 250).all()
    assert (best == 65535).all()
    assert read["records"] == []
    assert read["density"]["encoding"] == "u8" and read["density"]["nodata"] == 255
    assert read["best"]["encoding"] == "u16" and read["best"]["nodata"] == 65535
    assert read["summary"]["byBestClass"]["none"] == 1
    assert read["summary"]["foggyFraction"] == 1
    assert read["policy"]["version"] == "fog-v1"
    assert [c["id"] for c in read["policy"]["classes"]] == [
        "centimetre",
        "field",
        "aerial",
        "satellite",
        "regional",
    ]
    titles = " ".join(item["title"] for item in read["notCounted"])
    assert "Google Photorealistic 3D Tiles" in titles
    assert "NAIP" in titles and "Sentinel-2" in titles
    assert "simulated fleet" in titles


def test_inspection_lifts_fog_around_point_feature(client: TestClient) -> None:
    land = make_land(client)
    inspect(client, land["id"], datetime.now(UTC) - timedelta(days=1))
    read = fog(client, land["id"])
    density = grid_values(read, "density")
    best = grid_values(read, "best")
    age = (
        datetime.fromisoformat(read["asOf"]) - (datetime.now(UTC) - timedelta(days=1))
    ).total_seconds() / 86_400
    expected = round((1 - 0.85 * 0.5 ** (1 / 120)) * 250)
    row, column = cell(read, *POINT)
    assert density[row, column] == expected
    assert abs(age - 1) < 0.01
    [record] = read["records"]
    assert best[row, column] == 0
    assert record["kind"] == "inspection" and record["classId"] == "field"
    assert record["title"].startswith("Inspection: Transmission pole")
    assert record["link"]["kind"] == "inventory"
    # The 15 m reach covers a few cells, never the whole land: 100 m away is still full fog.
    assert 1 <= record["cells"] <= 9
    far = cell(read, east_of(*POINT, 100), POINT[1])
    assert density[far] == 250 and best[far] == 65535


def test_as_of_excludes_later_evidence(client: TestClient) -> None:
    land = make_land(client)
    empty = fog(client, land["id"])
    inspect(client, land["id"], datetime.now(UTC) - timedelta(days=1))
    before = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    then = fog(client, land["id"], asOf=before)
    assert then["density"]["data"] == empty["density"]["data"]
    assert then["records"] == []
    # Observed before the asOf but stored after it: still not known then.
    recent = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    assert fog(client, land["id"], asOf=recent)["records"] == []
    later = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    assert (
        client.get(f"/api/v1/land/{land['id']}/knowledge", params={"asOf": later}).status_code
        == 422
    )


def ndvi_raster(
    db: Session, land_id: str, acquired: datetime, *, dataset: str = "sentinel-2-ndvi"
) -> LandRaster:
    """A stored one-scene NDVI raster whose western half came back cloudy (NaN)."""
    request = RasterRequest(
        dataset="sentinel-2-ndvi",
        resolution_m=20,
        periods=[VegetationPeriod(start_date=date(2026, 9, 20), end_date=date(2026, 10, 10))],
    )
    grid = make_grid(Polygon.model_validate(polygon()), request)
    values = np.full((grid.height, grid.width), 0.5, dtype=np.float32)
    values[:, : grid.width // 2] = -9999
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=grid.width,
            height=grid.height,
            count=1,
            dtype="float32",
            crs=grid.crs.to_wkt(),
            transform=grid.affine,
            nodata=-9999.0,
        ) as dataset_file:
            dataset_file.write(values, 1)
        data = memory.read()
    run = research_run(db, land_id, "raster")
    scene = "S2B_10SGD_20261007_0_L2A"
    row = LandRaster(
        land_id=uuid.UUID(land_id),
        boundary_revision=1,
        run_id=run.id,
        request={**request.model_dump(mode="json"), "dataset": dataset},
        metadata_json={
            "algorithm": "sentinel-2-ndvi-v1",
            "resolution_m": 20,
            "sources": [
                {
                    "id": scene,
                    "license": "Copernicus Sentinel data terms",
                    "attribution": "Contains modified Copernicus Sentinel data",
                    "catalog_url": "https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a",
                }
            ],
            "vegetation": {
                "observations": [
                    {
                        "band": 1,
                        "scene_id": scene,
                        "acquired_at": acquired.isoformat(),
                        "valid_cells": 100,
                    }
                ]
            },
        },
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
    )
    db.add(row)
    db.flush()
    db.add(LandRasterBlob(raster_id=row.id, data=data))
    db.commit()
    return row


def test_sentinel_cloud_cells_stay_foggy(client: TestClient, db: Session) -> None:
    land = make_land(client)
    acquired = datetime.now(UTC) - timedelta(days=7)
    ndvi_raster(db, land["id"], acquired)
    read = fog(client, land["id"])
    density = grid_values(read, "density")
    [record] = read["records"]
    assert record["kind"] == "raster-observation" and record["classId"] == "satellite"
    assert record["observedAt"] is not None and "acquired" in record["observedLabel"]
    assert record["link"] == {
        "kind": "raster",
        "id": record["link"]["id"],
        "detail": {"band": 1, "dataset": "sentinel-2-ndvi"},
    }
    age = (datetime.fromisoformat(read["asOf"]) - acquired).total_seconds() / 86_400
    clear = round((1 - 0.45 * 0.5 ** (age / 90)) * 250)
    west = cell(read, -122.1385, 47.645)
    east = cell(read, -122.1315, 47.645)
    assert density[west] == 250
    assert density[east] == clear
    # The same raster counted as the signature dataset (show-once) is satellite evidence too.
    other = make_land(client)
    ndvi_raster(db, other["id"], acquired, dataset="sentinel-2-signature")
    [signature] = fog(client, other["id"])["records"]
    assert signature["classId"] == "satellite" and "sentinel-2-signature" in signature["title"]


def test_site_asset_capture(client: TestClient) -> None:
    land = make_land(client)
    footprint = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.136, 47.644],
                [-122.134, 47.644],
                [-122.134, 47.646],
                [-122.136, 47.646],
                [-122.136, 47.644],
            ]
        ],
    }
    observed = datetime.now(UTC) - timedelta(days=30)
    site = {
        "name": "Pole scan",
        "description": "A scan of the pole",
        "boundary": footprint,
        "attribution": [{"text": "Field crew"}],
        "license": {"name": "CC BY 4.0"},
        "assets": [
            {
                "name": "Splat",
                "representation": "gaussian-splat",
                "source": {
                    "type": "3d-tiles-url",
                    "url": "https://tiles.example.com/a/tileset.json",
                },
                "footprint": footprint,
                "observedAt": observed.isoformat(),
                "resolution": {"groundSampleDistanceM": 0.0015},
            }
        ],
    }
    assert client.post("/api/v1/sites", json=site).status_code == 201
    far = {
        **site,
        "name": "Elsewhere",
        "boundary": {
            "type": "Polygon",
            "coordinates": [[[10, 10], [10.01, 10], [10.01, 10.01], [10, 10.01], [10, 10]]],
        },
        "assets": [{**site["assets"][0], "footprint": None}],  # type: ignore[index]
    }
    assert client.post("/api/v1/sites", json=far).status_code == 201
    read = fog(client, land["id"])
    [record] = read["records"]
    assert record["kind"] == "capture" and record["classId"] == "centimetre"
    assert record["resolutionM"] == 0.0015 and record["link"]["kind"] == "site"
    age = (datetime.fromisoformat(read["asOf"]) - observed).total_seconds() / 86_400
    density = grid_values(read, "density")
    assert density[cell(read, *POINT)] == round((1 - 0.5 ** (age / 180)) * 250)
    assert density[cell(read, -122.1385, 47.6485)] == 250

    undated = {
        **site,
        "name": "Undated scan",
        "assets": [{**site["assets"][0], "observedAt": None}],  # type: ignore[index]
    }
    other = make_land(client, {**BODY, "name": "Second"})
    assert client.post("/api/v1/sites", json=undated).status_code == 201
    records = {r["title"]: r for r in fog(client, other["id"])["records"]}
    plain = records["Undated scan: Splat (gaussian-splat)"]
    assert plain["observedAt"] is None and plain["observedLabel"].startswith("Undated")
    assert plain["peakCredit"] == pytest.approx(0.25)
    assert "undated" in plain["arithmetic"]


def test_research_evidence_land_wide(client: TestClient, db: Session) -> None:
    land = make_land(client)
    run = research_run(db, land["id"])
    add_evidence(db, run)
    observed = datetime.now(UTC) - timedelta(days=7)
    add_evidence(
        db,
        run,
        provider="gbif-occurrences",
        title="Quercus agrifolia occurrence",
        observed_at=observed.isoformat(),
        spatial_relevance="nearby",
    )
    add_evidence(db, run, provider="col-taxonomy", title="Quercus agrifolia (accepted name)")
    add_evidence(
        db, run, provider="public-web-search", title="A page", spatial_relevance="unresolved"
    )
    # Counted elsewhere: raster source evidence is counted as the raster's own masks.
    add_evidence(db, run, provider="cop-dem-glo-30", title="DEM tile")
    read = fog(client, land["id"])
    density = grid_values(read, "density")
    inside = density != 255
    records = {r["provenance"]["provider"]: r for r in read["records"]}
    assert set(records) == {"usda-soils", "gbif-occurrences"}
    soils, gbif = records["usda-soils"], records["gbif-occurrences"]
    assert soils["classId"] == "regional" and not soils["located"]
    assert soils["peakCredit"] == pytest.approx(0.05)
    assert gbif["classId"] == "field" and gbif["countedClassId"] == "regional"
    assert gbif["peakCredit"] <= 0.2 and gbif["arithmetic"].startswith("min(0.85, 0.20)")
    # Land-wide: every inside cell gets the same credit; GBIF is never drawn as a point.
    assert len(np.unique(density[inside])) == 1
    assert density[inside][0] == round((1 - gbif["peakCredit"]) * 250)
    assert read["summary"]["byBestClass"]["regional"] == 1
    reasons = " ".join(f"{item['title']} {item['reason']}" for item in read["notCounted"])
    assert "Quercus agrifolia (accepted name)" in reasons and "A page" in reasons


def test_simulated_things_never_count(client: TestClient) -> None:
    land = make_land(client)
    before = fog(client, land["id"])
    plan = {
        "projectId": "demo-ranch",
        "title": "Mow the land",
        "objective": "Mow it.",
        "goal": "Mow the land",
        "startDate": "2026-10-16",
        "zoneIds": ["A-01"],
        "areas": [{"id": "A-01", "name": "All of it", "footprint": polygon()}],
        "estimates": {"acres": 200, "machineHours": 100, "calendarDays": 5},
    }
    created = client.post("/api/v1/plans", json=plan)
    assert created.status_code == 201, created.text
    client.patch(f"/api/v1/plans/{created.json()['id']}/status", json={"status": "done"})
    after = fog(client, land["id"])
    assert after["density"]["data"] == before["density"]["data"]
    assert after["records"] == []


def test_cell_endpoint(client: TestClient, db: Session) -> None:
    land = make_land(client)
    inspect(client, land["id"], datetime.now(UTC) - timedelta(days=1))
    run = research_run(db, land["id"])
    add_evidence(db, run)
    path = f"/api/v1/land/{land['id']}/knowledge/cell"
    read = client.get(path, params={"longitude": POINT[0], "latitude": POINT[1]}).json()
    assert read["inside"] and read["density"] == pytest.approx(39 / 250)
    assert [r["record"]["kind"] for r in read["records"]] == ["inspection", "research-evidence"]
    credits = [r["credit"] for r in read["records"]]
    assert credits == sorted(credits, reverse=True)
    assert [r["best"] for r in read["records"]] == [True, False]
    assert all(" x " in r["arithmetic"] and "=" in r["arithmetic"] for r in read["records"])
    assert read["bounds"][0] <= POINT[0] <= read["bounds"][2]
    outside = client.get(path, params={"longitude": -122.2, "latitude": 47.645}).json()
    assert outside["inside"] is False and outside["records"] == []
    assert outside["density"] is None


def test_scout_areas_budget_and_holes(client: TestClient) -> None:
    land = make_land(client)
    inspect(client, land["id"], datetime.now(UTC) - timedelta(hours=2))
    path = f"/api/v1/land/{land['id']}/knowledge/scout-areas"
    whole = client.post(path, json={})
    assert whole.status_code == 200, whole.text
    read = whole.json()
    assert read["forecast"].startswith("Forecast, not evidence")
    assert read["truncatedByBudget"] is False and read["areas"]
    boundary = shape(polygon())
    hole = shape({"type": "Point", "coordinates": list(POINT)})
    areas = [shape(area["footprint"]) for area in read["areas"]]
    for area in areas:
        assert boundary.buffer(1e-9).covers(area)
    # The inspected point is cleared: no patch covers it.
    assert not any(area.contains(hole) for area in areas)
    sizes = [area["areaM2"] for area in read["areas"]]
    assert sizes == sorted(sizes, reverse=True)

    budget = 100_000.0
    capped = client.post(
        path, json={"budgetM2": budget, "focus": [-122.137, 47.648], "maxAreas": 3}
    ).json()
    assert capped["truncatedByBudget"] is True and capped["areas"]
    total = sum(area["areaM2"] for area in capped["areas"])
    assert total <= budget + 20 * 20 * 1.01
    assert "nearest your view" in capped["areas"][0]["reason"]
    near = shape(capped["areas"][0]["footprint"]).representative_point()
    assert near.distance(shape({"type": "Point", "coordinates": [-122.137, 47.648]})) < 0.004
    bad = client.post(path, json={"threshold": 0})
    assert bad.status_code == 422


def test_knowledge_scoping(identity_client: IdentityClient) -> None:
    client, token = identity_client
    alice = headers(token())
    workspace = client.post("/api/v1/workspaces", headers=alice, json={"name": "Ranch"}).json()[
        "id"
    ]
    alice = headers(token(), workspace)
    land = client.post("/api/v1/land", json=BODY, headers=alice).json()
    assert client.get(f"/api/v1/land/{land['id']}/knowledge", headers=alice).status_code == 200
    bob = headers(token("bob"))
    other = client.post("/api/v1/workspaces", headers=bob, json={"name": "Other"}).json()["id"]
    bob = headers(token("bob"), other)
    for method, tail, body in (
        ("get", "", None),
        ("get", "/cell?longitude=-122.135&latitude=47.645", None),
        ("post", "/scout-areas", {}),
    ):
        response = client.request(
            method.upper(), f"/api/v1/land/{land['id']}/knowledge{tail}", headers=bob, json=body
        )
        assert response.status_code == 404, response.text
