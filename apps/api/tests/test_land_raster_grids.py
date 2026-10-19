from __future__ import annotations

import math
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Point, shape
from sqlalchemy.orm import Session, sessionmaker

from app.analysis.dem_3dep import analyze
from app.analysis.lonlat_grid import decode
from app.config import Settings
from app.research.worker import ResearchWorker
from app.schemas.grids import LonLatGrid
from tests.test_dem_3dep import BOUNDARY, REQUEST, STEP, fixture_client
from tests.test_land import BODY
from tests.test_workspaces import IdentityClient, headers
from tests.test_workspaces import identity_client as _identity_client

identity_client = _identity_client


def saved_bare_earth(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    auth: dict[str, str] | None = None,
) -> str:
    """A 3DEP raster of the fixture plane (rising 1 m per 1/3-arc-second row northward),
    saved through the worker as a real run would; returns its id."""
    with fixture_client() as http:
        result = analyze(BOUNDARY, REQUEST, http)
    monkeypatch.setattr("app.research.worker.run_raster", lambda *args: result)
    land = client.post(
        "/api/v1/land", json={**BODY, "boundary": BOUNDARY.model_dump()}, headers=auth
    ).json()
    investigation = client.post(
        f"/api/v1/land/{land['id']}/investigations",
        json={"title": "Bare-earth elevation", "boundaryRevision": 1},
        headers=auth,
    ).json()["id"]
    created = client.post(
        f"/api/v1/research/investigations/{investigation}/runs",
        json={
            "kind": "raster",
            "requestKey": str(uuid.uuid4()),
            "question": "Analyze bare-earth elevation and slope",
            "analysis": REQUEST.model_dump(mode="json", by_alias=True),
        },
        headers=auth,
    )
    assert created.status_code == 201, created.text
    db.rollback()
    assert ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    db.expire_all()
    rasters = client.get(f"/api/v1/land/{land['id']}/rasters", headers=auth).json()
    return str(rasters[0]["id"])


def test_elevation_band_comes_as_a_lonlat_grid_with_its_slope_and_no_data(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster_id = saved_bare_earth(client, db, sessions, monkeypatch)
    response = client.get(f"/api/v1/land/rasters/{raster_id}/grid", params={"band": 1})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    grid = LonLatGrid.model_validate(response.json())
    assert grid.encoding == "f32" and grid.cell_m == pytest.approx(10, rel=0.01)
    values = decode(grid).astype(np.float64)
    assert grid.west <= -120.605 and grid.east >= -120.600
    assert grid.south <= 35.620 and grid.north >= 35.625
    # The plane rises 1 m per source row (1/3 arc-second of latitude) to the north: one cell
    # south is about cell_m / row_m metres lower, and nothing changes east to west.
    row_m = STEP * 110_940
    dlat_m = (grid.north - grid.south) / grid.rows * 110_940
    southward = values[1:, :] - values[:-1, :]
    eastward = values[:, 1:] - values[:, :-1]
    assert np.nanmedian(southward) == pytest.approx(-dlat_m / row_m, rel=0.05)
    assert abs(np.nanmedian(eastward)) < 0.05
    assert np.nanmin(values) > 300 and np.nanmax(values) < 400
    # Inside the boundary there is a value; in its hole there is none.
    dlon = (grid.east - grid.west) / grid.columns
    dlat = (grid.north - grid.south) / grid.rows
    outline = shape(BOUNDARY.model_dump())

    def at(longitude: float, latitude: float) -> float:
        column = math.floor((longitude - grid.west) / dlon)
        row = math.floor((grid.north - latitude) / dlat)
        return float(values[row, column])

    assert outline.contains(Point(-120.6045, 35.6245)) and math.isfinite(at(-120.6045, 35.6245))
    assert math.isnan(at(-120.60325, 35.62175))
    # The slope band is there too; a band the raster does not have is not.
    assert (
        client.get(f"/api/v1/land/rasters/{raster_id}/grid", params={"band": 2}).status_code == 200
    )
    assert (
        client.get(f"/api/v1/land/rasters/{raster_id}/grid", params={"band": 3}).status_code == 404
    )
    assert (
        client.get(f"/api/v1/land/rasters/{raster_id}/grid", params={"band": 0}).status_code == 422
    )


def test_band_grids_are_workspace_scoped(
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
    raster_id = saved_bare_earth(client, db, sessions, monkeypatch, alice)
    url = f"/api/v1/land/rasters/{raster_id}/grid"
    assert client.get(url, headers=alice).status_code == 200
    bob_space = client.post(
        "/api/v1/workspaces", json={"name": "Bob"}, headers=headers(token("bob"))
    ).json()["id"]
    assert client.get(url, headers=headers(token("bob"), bob_space)).status_code == 404
    assert client.get(url).status_code == 401
