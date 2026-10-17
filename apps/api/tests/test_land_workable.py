"""Workable ground: an outline on a saved Land less what its machine class cannot work."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pytest
from affine import Affine
from fastapi.testclient import TestClient
from pydantic import HttpUrl
from pyproj import Transformer
from shapely.affinity import translate
from shapely.geometry import LineString, MultiPolygon, box, mapping, shape
from sqlalchemy.orm import Session, sessionmaker

from app.analysis.raster_grid import make_grid
from app.analysis.terrain import TerrainResult, band_statistics, derive, encode_cog
from app.analysis.workable import (
    MIN_PART_M2,
    counts_as_creek,
    edge_filled,
    slope_percent,
    split_ground,
)
from app.config import Settings
from app.research.worker import ResearchWorker
from app.schemas.geojson import Polygon as PolygonModel
from app.schemas.land_hydrography import HydrographyFeature, LandHydrography
from app.schemas.land_rasters import RasterMetadata, RasterRequest, RasterSource
from app.schemas.place import PlaceFactSource
from tests.test_land import BODY
from tests.test_workspaces import IdentityClient, headers
from tests.test_workspaces import identity_client as _identity_client

identity_client = _identity_client

# --- the pure split ---------------------------------------------------------------------------


def test_slope_is_the_steepest_gradient_in_percent_with_one_sided_edges() -> None:
    # A plane rising 10 m per 100 m east and 20 m per 100 m south: 22.4 % everywhere.
    rows, columns = np.mgrid[0:6, 0:8].astype(np.float64)
    plane = 0.1 * columns * 10 + 0.2 * rows * 10
    slope = slope_percent(plane, 10)
    assert np.allclose(slope, math.hypot(10, 20), atol=1e-4)
    # A cell without a height has no slope; its neighbours read one-sided, and still 22.4 %.
    holed = plane.copy()
    holed[2, 3] = np.nan
    slope = slope_percent(holed, 10)
    assert np.isnan(slope[2, 3])
    assert np.allclose(slope[np.isfinite(slope)], math.hypot(10, 20), atol=1e-4)
    # The ring around the data takes its steepest neighbour's slope, and no further.
    ring = np.full((5, 5), np.nan, dtype=np.float32)
    ring[2, 2] = 30.0
    ring[2, 3] = 10.0
    filled = edge_filled(ring)
    assert filled[1, 1] == 30.0 and filled[1, 4] == 10.0 and filled[2, 4] == 10.0
    assert np.isnan(filled[0, 0]) and np.isnan(filled[4, 4])


def test_ephemeral_drainages_count_only_on_request() -> None:
    for kind in ("perennial", "intermittent", "unclassified", "artificial-path"):
        assert counts_as_creek(kind, False)
    assert not counts_as_creek("ephemeral", False) and counts_as_creek("ephemeral", True)
    assert not counts_as_creek("other", True)


def grid(columns: int, rows: int) -> Affine:
    """A 10 m grid with its north-west corner at (0, rows * 10)."""
    return Affine(10, 0, 0, 0, -10, rows * 10)


def test_ground_is_split_in_order_and_the_parts_add_up_to_the_outline() -> None:
    columns, rows = 40, 20
    transform = grid(columns, rows)
    slope = np.full((rows, columns), 10.0, dtype=np.float32)
    # The east half is too steep for 25 %; one 2 x 2 cell pocket inside it is gentle, a sliver
    # of 400 m2 once the steep cells around it are cut away.
    slope[:, 20:] = 40.0
    slope[8:10, 30:32] = 5.0
    # No height at all in the west two columns (outside a Land), beyond the one-cell ring.
    slope[:, :2] = np.nan
    outline = box(0, 0, columns * 10, rows * 10)
    creek = LineString([(100, 0), (100, 200)])
    split = split_ground(outline, slope, transform, max_slope_pct=25, creeks=[creek], buffer_m=15)
    # Column 0 has no height and no neighbour with one; column 1 takes column 2's slope.
    assert split.no_data.area == pytest.approx(10 * 200)
    assert split.steep.area == pytest.approx(200 * 200 - 400)
    assert split.creek_buffer.area == pytest.approx(30 * 200, rel=0.01)
    assert split.sliver_count == 1 and split.slivers.area == pytest.approx(400)
    assert MIN_PART_M2 > 400
    parts = (
        split.no_data.area
        + split.steep.area
        + split.creek_buffer.area
        + split.slivers.area
        + split.workable.area
    )
    assert parts == pytest.approx(outline.area)
    # Two workable pieces, either side of the creek's buffer, and none of it steep.
    assert isinstance(split.workable, MultiPolygon) and len(split.workable.geoms) == 2
    assert split.workable.intersection(box(200, 0, 400, 200)).area == pytest.approx(0)
    assert split.cells_read == 20 * 39 and split.cells_steep == 20 * 20 - 4


def test_an_outline_off_the_data_is_all_no_data() -> None:
    slope = np.full((4, 4), np.nan, dtype=np.float32)
    split = split_ground(box(0, 0, 40, 40), slope, grid(4, 4), max_slope_pct=25)
    assert split.cells_read == 0 and split.workable.is_empty
    assert split.no_data.area == pytest.approx(1600)


# --- the route, on a saved Land with a synthetic bare-earth DEM -------------------------------

# A 600 m x 400 m study area at Oak Bench Ranch's latitude. Its ground rises 10 m per 100 m
# eastward in the west half and 32 m per 100 m in the east half.
LONGITUDE, LATITUDE = -120.04, 34.698
HALF_LON = 300 / (111_320 * math.cos(math.radians(LATITUDE)))
HALF_LAT = 200 / 110_950
RECTANGLE = PolygonModel(
    coordinates=[
        [
            [LONGITUDE - HALF_LON, LATITUDE - HALF_LAT],
            [LONGITUDE + HALF_LON, LATITUDE - HALF_LAT],
            [LONGITUDE + HALF_LON, LATITUDE + HALF_LAT],
            [LONGITUDE - HALF_LON, LATITUDE + HALF_LAT],
            [LONGITUDE - HALF_LON, LATITUDE - HALF_LAT],
        ]
    ]
)


def steep_east(x: np.ndarray, _y: np.ndarray) -> np.ndarray:
    return np.where(x < 0, 0.10 * x, 0.32 * x) + 300


def synthetic_dem(
    request: RasterRequest, surface: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> TerrainResult:
    """A DEM over RECTANGLE on the analysis grid a real run makes, from `surface(x, y)` in
    metres from the grid's centre, saved the way the 3DEP analysis saves one."""
    analysis = make_grid(RECTANGLE, request)
    columns = np.arange(analysis.width) + 0.5
    rows = np.arange(analysis.height) + 0.5
    xs = analysis.affine.c + columns * analysis.resolution
    ys = analysis.affine.f - rows * analysis.resolution
    x, y = np.meshgrid(xs, ys)
    elevation, slopes = derive(
        surface(x, y).astype(np.float32), analysis.inside, analysis.resolution
    )
    names = ("Bare-earth elevation", "Bare-earth slope")
    count = int(analysis.inside.sum())
    metadata = RasterMetadata(
        algorithm="synthetic-dem",
        method="Synthetic plane for tests",
        bounds=list(analysis.bounds),
        crs=analysis.crs.to_wkt(),
        resolution_m=analysis.resolution,
        width=analysis.width,
        height=analysis.height,
        boundary_cells=count,
        valid_cells=count,
        coverage_fraction=1,
        sampled_area_m2=count * analysis.resolution**2,
        bands=[
            band_statistics(elevation, 1, names[0], "m"),
            band_statistics(slopes, 2, names[1], "degrees"),
        ],
        sources=[
            RasterSource(
                id="synthetic",
                url=HttpUrl("https://example.test/dem.tif"),
                catalog_url=HttpUrl("https://example.test/dem.xml"),
                etag=None,
                last_modified="Wed, 27 Aug 2025 16:22:02 GMT",
                catalog_datetime=None,
                observation_period="Synthetic",
                license="Public domain",
                license_url=HttpUrl("https://example.test/license"),
                attribution="Test fixture",
            )
        ],
        warnings=[],
        downloaded_bytes=0,
    )
    data = encode_cog([elevation, slopes], metadata.crs, analysis.affine, names=names)
    return TerrainResult(data, metadata)


def saved_land(client: TestClient, auth: dict[str, str] | None = None) -> str:
    created = client.post(
        "/api/v1/land",
        json={**BODY, "name": "Bench pasture", "boundary": RECTANGLE.model_dump()},
        headers=auth,
    )
    assert created.status_code == 201, created.text
    return str(created.json()["id"])


def saved_dem(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    land_id: str,
    request: RasterRequest,
    auth: dict[str, str] | None = None,
) -> str:
    """A synthetic elevation raster for the land, saved through the worker as a run saves it."""
    result = synthetic_dem(request, steep_east)
    monkeypatch.setattr("app.research.worker.run_raster", lambda *args: result)
    investigation = client.post(
        f"/api/v1/land/{land_id}/investigations",
        json={"title": "Elevation", "boundaryRevision": 1},
        headers=auth,
    ).json()["id"]
    created = client.post(
        f"/api/v1/research/investigations/{investigation}/runs",
        json={
            "kind": "raster",
            "requestKey": str(uuid.uuid4()),
            "question": "Analyze elevation and slope",
            "analysis": request.model_dump(mode="json", by_alias=True),
        },
        headers=auth,
    )
    assert created.status_code == 201, created.text
    db.rollback()
    assert ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    db.expire_all()
    rasters = client.get(f"/api/v1/land/{land_id}/rasters", headers=auth).json()
    return str(rasters[0]["id"])


def flowline(identifier: str, kind: str, longitude: float) -> HydrographyFeature:
    """A north-south flowline across RECTANGLE at `longitude`."""
    return HydrographyFeature.model_validate(
        {
            "id": identifier,
            "permanentIdentifier": identifier,
            "name": "Bench Creek" if kind != "ephemeral" else None,
            "fcode": {"intermittent": 46003, "ephemeral": 46007}[kind],
            "kind": kind,
            "description": f"Stream/river, {kind}",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [longitude, LATITUDE - 2 * HALF_LAT],
                    [longitude, LATITUDE + 2 * HALF_LAT],
                ],
            },
            "lengthM": 800,
        }
    )


def hydrography(
    status: Literal["available", "unavailable"] = "available",
) -> Callable[..., LandHydrography]:
    """NHD as the land's hydrography answers it: an intermittent creek 150 m west of the
    centre (in the gentle half) and an ephemeral drainage 250 m west."""

    def answer(_db: Session, _workspace: uuid.UUID, land_id: uuid.UUID) -> LandHydrography:
        metre = 1 / (111_320 * math.cos(math.radians(LATITUDE)))
        return LandHydrography(
            land_id=land_id,
            boundary_revision=1,
            status=status,
            message="2 mapped flowlines within 100 m of this land."
            if status == "available"
            else "The USGS hydrography service did not answer. Try again in a minute.",
            buffer_m=100,
            features=[
                flowline("creek-1", "intermittent", LONGITUDE - 150 * metre),
                flowline("wash-1", "ephemeral", LONGITUDE - 250 * metre),
            ]
            if status == "available"
            else [],
            truncated=False,
            source=PlaceFactSource.model_validate(
                {
                    "provider": "usgs-nhd",
                    "name": "USGS National Hydrography Dataset (NHD) flowlines",
                    "url": "https://example.test/nhd",
                    "linkUrl": "https://example.test/nhd",
                    "documentationUrl": "https://example.test/docs",
                    "license": "Public domain (U.S. Government work)",
                    "attribution": "U.S. Geological Survey",
                    "resolution": "1:24,000 (Flowline - Large Scale); mapped channels",
                }
            ),
            retrieved_at=datetime(2026, 10, 17, tzinfo=UTC),
        )

    return answer


THREE_DEP = RasterRequest(dataset="usgs-3dep-13", resolution_m=10)


def ask(client: TestClient, land_id: str, auth: dict[str, str] | None = None, **body: Any) -> Any:
    return client.post(
        f"/api/v1/land/{land_id}/workable-ground",
        json={"footprint": RECTANGLE.model_dump(), "machineClass": "mower", **body},
        headers=auth,
    )


def test_workable_ground_leaves_out_the_steep_half_and_the_creek_buffer(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    land_id = saved_land(client)
    raster_id = saved_dem(client, db, sessions, monkeypatch, land_id, THREE_DEP)
    monkeypatch.setattr("app.services.land_hydrography.hydrography", hydrography())
    response = ask(client, land_id)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    answer = response.json()
    footprint = answer["footprintM2"]
    assert footprint == pytest.approx(600 * 400, rel=0.01)
    removed = answer["removed"]
    # The east half rises 32 m per 100 m, over the mower's assumed 25 %.
    assert removed["steepM2"] == pytest.approx(footprint / 2, rel=0.03)
    # 15 m either side of the intermittent creek, across the whole 400 m; not the ephemeral.
    assert removed["creekBufferM2"] == pytest.approx(30 * 400, rel=0.03)
    assert removed["noDataM2"] == pytest.approx(0, abs=1)
    total = (
        answer["workableM2"]
        + removed["steepM2"]
        + removed["creekBufferM2"]
        + removed["sliversM2"]
        + removed["noDataM2"]
    )
    assert total == pytest.approx(footprint, rel=1e-6)
    # The workable ground is the gentle half either side of the creek, inside the outline.
    workable = shape(answer["workable"])
    assert isinstance(workable, MultiPolygon) and len(workable.geoms) == 2
    assert workable.within(shape(RECTANGLE.model_dump()).buffer(1e-6))
    assert workable.bounds[2] < LONGITUDE + 5 / 91_000
    reasons = {item["reason"]: item for item in answer["leftOut"]}
    assert set(reasons) == {"steep", "creek-buffer"}
    assert reasons["steep"]["label"] == "Left out: too steep"
    assert reasons["creek-buffer"]["label"] == "Left out: creek buffer"
    assert shape(reasons["steep"]["geometry"]).area > 0
    assert answer["limits"] == {
        "machineClass": "mower",
        "label": "Robotic brush mower (assumed)",
        "maxSlopePct": 25,
        "creekBufferM": 15,
    }
    assert answer["elevation"] == {
        "rasterId": raster_id,
        "dataset": "usgs-3dep-13",
        "label": "USGS 3DEP 1/3″ bare-earth DEM, 10 m",
        "published": "2025-08-27",
        "resolutionM": 10,
        "stale": False,
        "fallback": False,
    }
    assert answer["creeks"]["counted"] == 1 and not answer["creeks"]["ephemeralIncluded"]
    assert answer["sources"] == [
        f"Slope: USGS 3DEP 1/3″ bare-earth DEM, 10 m, raster {raster_id[:8]}, source "
        "published 2025-08-27",
        "Creeks: USGS National Hydrography Dataset (NHD) flowlines, 1:24,000, read 2026-10-17",
        "Limits: assumed per machine class (machine_classes.json)",
    ]
    assumptions = " ".join(answer["assumptions"])
    assert "works slopes up to 25 % and keeps 15 m from creeks" in assumptions
    assert "not ephemeral drainages" in assumptions
    assert "counts once, as too steep" in assumptions
    assert answer["warnings"] == []

    # Ephemeral drainages count on request; a stricter class leaves out more.
    wider = ask(client, land_id, includeEphemeralCreeks=True).json()
    assert wider["removed"]["creekBufferM2"] == pytest.approx(2 * 30 * 400, rel=0.03)
    assert wider["creeks"]["counted"] == 2 and wider["creeks"]["ephemeralIncluded"]
    survey = ask(client, land_id, machineClass="survey").json()
    # The survey rover's assumed 35 % takes the east half too; only its 5 m buffer is cut.
    assert survey["removed"]["steepM2"] == pytest.approx(0, abs=1)
    assert survey["removed"]["creekBufferM2"] == pytest.approx(10 * 400, rel=0.03)


def test_a_part_off_the_land_is_left_out_as_no_data_and_creeks_unread_are_said(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    land_id = saved_land(client)
    saved_dem(client, db, sessions, monkeypatch, land_id, THREE_DEP)
    monkeypatch.setattr("app.services.land_hydrography.hydrography", hydrography("unavailable"))
    # Most of the gentle west half of the land, and 100 m beyond its west edge.
    west = LONGITUDE - HALF_LON
    east = LONGITUDE - HALF_LON / 6
    footprint = {
        "type": "Polygon",
        "coordinates": [
            [
                [west - HALF_LON / 3, LATITUDE - HALF_LAT],
                [east, LATITUDE - HALF_LAT],
                [east, LATITUDE + HALF_LAT],
                [west - HALF_LON / 3, LATITUDE + HALF_LAT],
                [west - HALF_LON / 3, LATITUDE - HALF_LAT],
            ]
        ],
    }
    answer = ask(client, land_id, footprint=footprint).json()
    # 100 m x 400 m off the land, less the ring of cells along its outline that take their
    # neighbours' slope (up to one 10 m cell).
    assert 89 * 400 < answer["removed"]["noDataM2"] < 100 * 400
    assert answer["removed"]["creekBufferM2"] == 0
    assert answer["creeks"]["status"] == "unavailable"
    assert answer["warnings"] == [
        "Creek buffer not applied: The USGS hydrography service did not answer. Try again in "
        "a minute."
    ]
    assert not any(source.startswith("Creeks:") for source in answer["sources"])
    assert "slope is not known" in " ".join(answer["assumptions"])
    assert {item["reason"] for item in answer["leftOut"]} == {"no-data"}


def test_without_bare_earth_the_surface_model_stands_in_and_says_so(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    land_id = saved_land(client)
    monkeypatch.setattr("app.services.land_hydrography.hydrography", hydrography())
    # No elevation at all: say what to do.
    missing = ask(client, land_id)
    assert missing.status_code == 409
    assert missing.json()["code"] == "no_elevation"
    assert "Analyze bare-earth terrain (USGS 3DEP)" in missing.json()["detail"]
    glo30 = saved_dem(
        client,
        db,
        sessions,
        monkeypatch,
        land_id,
        RasterRequest(dataset="cop-dem-glo-30", resolution_m=30),
    )
    answer = ask(client, land_id).json()
    assert answer["elevation"]["rasterId"] == glo30
    assert answer["elevation"]["fallback"] is True
    assert answer["elevation"]["label"].startswith("Copernicus GLO-30, a ")
    assert "surface model incl. canopy" in answer["elevation"]["label"]
    assert answer["warnings"][0].startswith("Slope is from Copernicus GLO-30")
    # Bare earth, once there, is preferred over the newer surface model.
    bare = saved_dem(client, db, sessions, monkeypatch, land_id, THREE_DEP)
    assert ask(client, land_id).json()["elevation"]["rasterId"] == bare
    assert ask(client, land_id, demRasterId=glo30).json()["elevation"]["rasterId"] == glo30


def test_a_revised_outline_needs_its_own_elevation(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    land_id = saved_land(client)
    raster_id = saved_dem(client, db, sessions, monkeypatch, land_id, THREE_DEP)
    monkeypatch.setattr("app.services.land_hydrography.hydrography", hydrography())
    revised = client.put(
        f"/api/v1/land/{land_id}",
        json={
            **BODY,
            "name": "Bench pasture",
            "boundary": RECTANGLE.model_dump(),
            "expectedRevision": 1,
            "note": "Again",
        },
    )
    assert revised.status_code == 200, revised.text
    assert ask(client, land_id).json()["code"] == "no_elevation"
    # Asked for by id, the earlier outline's raster is read and called stale.
    answer = ask(client, land_id, demRasterId=raster_id).json()
    assert answer["elevation"]["stale"] is True
    assert "earlier outline" in answer["warnings"][0]


def test_requests_that_cannot_be_answered_say_why(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    land_id = saved_land(client)
    saved_dem(client, db, sessions, monkeypatch, land_id, THREE_DEP)
    monkeypatch.setattr("app.services.land_hydrography.hydrography", hydrography())
    drone = ask(client, land_id, machineClass="drone")
    assert drone.status_code == 422
    assert "workable ground is for ground machines" in drone.json()["detail"]
    far = translate(shape(RECTANGLE.model_dump()), xoff=1.0)
    outside = ask(client, land_id, footprint=mapping(far))
    assert outside.status_code == 422
    assert "outside Bench pasture" in outside.json()["detail"]
    assert ask(client, str(uuid.uuid4())).status_code == 404
    assert ask(client, land_id, demRasterId=str(uuid.uuid4())).status_code == 404


def test_workable_ground_is_workspace_scoped(
    identity_client: IdentityClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, token = identity_client
    alice_space = client.post(
        "/api/v1/workspaces", json={"name": "Alice"}, headers=headers(token())
    ).json()["id"]
    alice = headers(token(), alice_space)
    land_id = saved_land(client, alice)
    saved_dem(client, db, sessions, monkeypatch, land_id, THREE_DEP, alice)
    monkeypatch.setattr("app.services.land_hydrography.hydrography", hydrography())
    assert ask(client, land_id, alice).status_code == 200
    bob_space = client.post(
        "/api/v1/workspaces", json={"name": "Bob"}, headers=headers(token("bob"))
    ).json()["id"]
    assert ask(client, land_id, headers(token("bob"), bob_space)).status_code == 404
    assert ask(client, land_id).status_code == 401


def test_the_metric_grid_and_the_outline_agree() -> None:
    """The fixture's centre is the grid's: the steep half starts at x = 0 (a cell edge)."""
    analysis = make_grid(RECTANGLE, THREE_DEP)
    forward = Transformer.from_crs(4326, analysis.crs, always_xy=True)
    x, y = forward.transform(LONGITUDE, LATITUDE)
    assert abs(x) < 1 and abs(y) < 1
