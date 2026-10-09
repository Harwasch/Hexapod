from __future__ import annotations

import math
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.main import create_app
from app.models.land import LandBoundaryRevision


def polygon(west: float = -122.14, east: float = -122.13) -> dict[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [
            [[west, 47.64], [east, 47.64], [east, 47.65], [west, 47.65], [west, 47.64]]
        ],
    }


BODY: dict[str, Any] = {
    "name": "My land",
    "description": "An independent study area",
    "boundary": polygon(),
    "source": {"method": "drawn", "label": "Drawn on map"},
}


def test_land_lives_without_a_site_or_plan(client: TestClient, db: Session) -> None:
    response = client.post("/api/v1/land", json=BODY)
    assert response.status_code == 201, response.text
    area = response.json()
    assert area["areaM2"] > 800_000 and area["perimeterM"] > 3000
    assert area["source"]["meaning"] == "study-area"
    area_id = area["id"]
    assert client.get("/api/v1/land").json()[0]["id"] == area_id
    assert client.get("/api/v1/sites").json() == []
    assert client.get(f"/api/v1/land/{area_id}").json() == area
    revised = client.put(
        f"/api/v1/land/{area_id}",
        json={
            **BODY,
            "expectedRevision": 1,
            "boundary": polygon(east=-122.135),
            "note": "Exclude east half",
        },
    )
    assert revised.status_code == 200, revised.text
    assert revised.json()["revision"] == 2
    assert revised.json()["areaM2"] == pytest.approx(area["areaM2"] / 2, rel=0.001)
    history = client.get(f"/api/v1/land/{area_id}/revisions").json()
    assert [r["revision"] for r in history] == [2, 1]
    assert history[1]["boundary"] == BODY["boundary"]
    assert history[0]["note"] == "Exclude east half"
    stale = client.put(f"/api/v1/land/{area_id}", json={**BODY, "expectedRevision": 1})
    assert stale.status_code == 409
    assert len(client.get(f"/api/v1/land/{area_id}/revisions").json()) == 2
    assert client.delete(f"/api/v1/land/{area_id}").status_code == 204
    assert client.get(f"/api/v1/land/{area_id}").status_code == 404
    assert (
        db.scalars(
            select(LandBoundaryRevision).where(LandBoundaryRevision.land_id == uuid.UUID(area_id))
        ).all()
        == []
    )


def test_polygon_holes_remain_excluded(client: TestClient) -> None:
    shape = polygon()
    shape["coordinates"].append(polygon(-122.138, -122.132)["coordinates"][0])
    # A hole touching the outer boundary is invalid and must not be silently repaired.
    assert client.post("/api/v1/land", json={**BODY, "boundary": shape}).status_code == 422
    outer = client.post("/api/v1/land", json=BODY).json()
    shape["coordinates"][1] = [
        [-122.138, 47.642],
        [-122.132, 47.642],
        [-122.132, 47.648],
        [-122.138, 47.648],
        [-122.138, 47.642],
    ]
    response = client.post("/api/v1/land", json={**BODY, "boundary": shape})
    assert response.status_code == 201, response.text
    assert response.json()["areaM2"] == pytest.approx(outer["areaM2"] * 0.64, rel=0.001)
    assert len(response.json()["boundary"]["coordinates"][0]) == 2


def test_corridor_width_is_total_width_in_metres(client: TestClient) -> None:
    # About 1111 metres north/south at the equator. A 100-ft total-width corridor
    # with flat caps should have roughly length * 30.48 m of area.
    response = client.post(
        "/api/v1/land/corridor",
        json={
            "coordinates": [[0, 0], [0, 0.01]],
            "widthM": 30.48,
            "cap": "flat",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["areaM2"] == pytest.approx(1105.74 * 30.48, rel=0.004)
    rounded = client.post(
        "/api/v1/land/corridor",
        json={
            "coordinates": [[0, 0], [0, 0.01]],
            "widthM": 30.48,
            "cap": "round",
        },
    ).json()
    assert rounded["areaM2"] - response.json()["areaM2"] == pytest.approx(
        math.pi * 15.24**2, rel=0.02
    )


@pytest.mark.parametrize(
    "coordinates,width",
    [
        ([[0, 0], [0, 0]], 10),
        ([[0, 0], [181, 1]], 10),
        ([[0, 0], [0, 1]], 0),
        ([[0, 0], [0, 1]], -10),
        ([[179, 0], [-179, 0]], 10),
        ([[0, 0], [0, 10]], 10),
    ],
)
def test_corridor_rejects_degenerate_or_unsupported_extents(
    client: TestClient,
    coordinates: list[list[float]],
    width: float,
) -> None:
    response = client.post(
        "/api/v1/land/corridor", json={"coordinates": coordinates, "widthM": width}
    )
    assert response.status_code == 422, response.text


def test_boundary_operations_and_empty_intersection(client: TestClient) -> None:
    a, b = polygon(), polygon(-122.135, -122.125)
    intersect = client.post(
        "/api/v1/land/operations", json={"operation": "intersection", "left": a, "right": b}
    )
    union = client.post(
        "/api/v1/land/operations", json={"operation": "union", "left": a, "right": b}
    )
    assert intersect.status_code == union.status_code == 200
    assert union.json()["areaM2"] == pytest.approx(intersect.json()["areaM2"] * 3, rel=0.001)
    empty = client.post(
        "/api/v1/land/operations", json={"operation": "difference", "left": a, "right": a}
    )
    assert empty.status_code == 422


def test_land_reads_are_not_public() -> None:
    app = create_app(Settings(api_write_token="land-pilot-token"))
    with TestClient(app) as client:
        assert client.get("/api/v1/land").status_code == 401
        assert client.get("/api/v1/land/11111111-1111-4111-8111-111111111111").status_code == 401
        assert client.post("/api/v1/land/corridor", json={}).status_code == 401


def test_cross_origin_boundary_revision_preflight() -> None:
    app = create_app(Settings(api_cors_origins=["https://world.example"]))
    with TestClient(app) as client:
        response = client.options(
            "/api/v1/land/11111111-1111-4111-8111-111111111111",
            headers={
                "Origin": "https://world.example",
                "Access-Control-Request-Method": "PUT",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://world.example"


def test_split_preserves_holes_and_merges_kept_pieces(client: TestClient) -> None:
    outer = polygon()
    outer["coordinates"].append(
        [
            [-122.138, 47.642],
            [-122.136, 47.642],
            [-122.136, 47.644],
            [-122.138, 47.644],
            [-122.138, 47.642],
        ]
    )
    saved = client.post("/api/v1/land", json={**BODY, "boundary": outer}).json()
    payload = {"boundary": outer, "coordinates": [[-122.135, 47.63], [-122.135, 47.66]]}
    preview = client.post("/api/v1/land/split", json=payload)
    assert preview.status_code == 200, preview.text
    parts = preview.json()["parts"]
    assert len(parts) == 2 and preview.json()["selection"] is None
    assert sum(part["areaM2"] for part in parts) == pytest.approx(saved["areaM2"], rel=1e-6)
    assert sorted(len(part["boundary"]["coordinates"]) for part in parts) == [1, 2]
    chosen = client.post("/api/v1/land/split", json={**payload, "keepParts": [0]}).json()
    assert chosen["selection"]["areaM2"] == pytest.approx(parts[0]["areaM2"], rel=1e-6)
    all_parts = client.post("/api/v1/land/split", json={**payload, "keepParts": [0, 1]}).json()
    assert all_parts["selection"]["areaM2"] == pytest.approx(saved["areaM2"], rel=1e-6)
    assert client.get(f"/api/v1/land/{saved['id']}").json() == saved
    assert client.post("/api/v1/land/split", json={**payload, "keepParts": [2]}).status_code == 422


def test_split_rejects_missed_cut_and_keeps_disconnected_land(client: TestClient) -> None:
    a, b = polygon(), polygon(-122.12, -122.11)
    boundary = {"type": "MultiPolygon", "coordinates": [a["coordinates"], b["coordinates"]]}
    missed = client.post(
        "/api/v1/land/split",
        json={"boundary": boundary, "coordinates": [[-122.125, 47.63], [-122.125, 47.66]]},
    )
    assert missed.status_code == 422 and "did not divide" in missed.text
    split = client.post(
        "/api/v1/land/split",
        json={"boundary": boundary, "coordinates": [[-122.135, 47.63], [-122.135, 47.66]]},
    )
    assert split.status_code == 200, split.text
    assert len(split.json()["parts"]) == 3
    for coordinates in ([[0, 0], [0, 0]], [[179, 0], [-179, 0]], [[0, 91], [1, 91]]):
        assert (
            client.post(
                "/api/v1/land/split", json={"boundary": a, "coordinates": coordinates}
            ).status_code
            == 422
        )
