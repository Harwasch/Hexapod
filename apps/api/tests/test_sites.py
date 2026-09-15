from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import SQUARE, site_payload


def test_create_and_read_site(client: TestClient) -> None:
    response = client.post("/api/v1/sites", json=site_payload())
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["slug"] == "test-site"
    assert body["boundary"]["type"] == "MultiPolygon"
    assert body["areaM2"] > 10_000
    assert -122.139 < body["centroid"]["longitude"] < -122.137
    assert body["assets"][0]["provider"] == "cesium-ion"
    assert body["assets"][0]["source"] == {"type": "cesium-ion", "assetId": 4547222}
    assert body["cameraBookmarks"][0]["isDefault"] is True
    assert body["license"]["spdxId"] == "CC-BY-4.0"

    listed = client.get("/api/v1/sites").json()
    assert len(listed) == 1
    assert listed[0]["representations"] == ["gaussian-splat"]
    assert listed[0]["latestObservedAt"].startswith("2025-06-01")

    fetched = client.get(f"/api/v1/sites/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == body["id"]
    by_slug = client.get("/api/v1/sites/by-slug/test-site")
    assert by_slug.status_code == 200


def test_slug_uniqueness_and_conflict(client: TestClient) -> None:
    first = client.post("/api/v1/sites", json=site_payload()).json()
    second = client.post("/api/v1/sites", json=site_payload()).json()
    assert first["slug"] == "test-site"
    assert second["slug"] == "test-site-2"
    conflict = client.post("/api/v1/sites", json=site_payload(slug="test-site"))
    assert conflict.status_code == 409
    bad_slug = client.post("/api/v1/sites", json=site_payload(slug="Not A Slug"))
    assert bad_slug.status_code == 422


def test_update_site_recomputes_centroid(client: TestClient) -> None:
    site = client.post("/api/v1/sites", json=site_payload()).json()
    moved = {
        "type": "Polygon",
        "coordinates": [[[10, 10], [11, 10], [11, 11], [10, 11], [10, 10]]],
    }
    response = client.patch(
        f"/api/v1/sites/{site['id']}", json={"name": "Renamed", "boundary": moved}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["name"] == "Renamed"
    assert 10 < body["centroid"]["longitude"] < 11
    assert body["updatedAt"] >= body["createdAt"]


def test_delete_site_cascades(client: TestClient) -> None:
    site = client.post("/api/v1/sites", json=site_payload()).json()
    assert client.get(f"/api/v1/sites/{site['id']}/assets").json()
    assert client.delete(f"/api/v1/sites/{site['id']}").status_code == 204
    assert client.get(f"/api/v1/sites/{site['id']}").status_code == 404
    assert client.get("/api/v1/assets").json() == []


def test_site_not_found_is_problem_json(client: TestClient) -> None:
    response = client.get("/api/v1/sites/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json()["title"] == "Not found"


def test_site_with_tiles_url_asset_validates_url(client: TestClient) -> None:
    payload = site_payload(
        assets=[
            {
                "name": "Mesh",
                "representation": "mesh",
                "source": {"type": "3d-tiles-url", "url": "ftp://example.com/tileset.json"},
            }
        ]
    )
    response = client.post("/api/v1/sites", json=payload)
    assert response.status_code == 422
    payload = site_payload(
        assets=[
            {
                "name": "Mesh",
                "representation": "mesh",
                "source": {
                    "type": "3d-tiles-url",
                    "url": "https://user:pw@example.com/tileset.json",
                },
            }
        ]
    )
    response = client.post("/api/v1/sites", json=payload)
    assert response.status_code == 422
    assert "credentials" in response.json()["detail"]


def test_multipolygon_boundary(client: TestClient) -> None:
    boundary = {"type": "MultiPolygon", "coordinates": [SQUARE["coordinates"]]}
    response = client.post("/api/v1/sites", json=site_payload(boundary=boundary))
    assert response.status_code == 201
    assert response.json()["boundary"]["type"] == "MultiPolygon"
