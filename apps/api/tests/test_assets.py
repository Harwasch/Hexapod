from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import SQUARE, site_payload


def test_asset_crud_and_site_relationship(client: TestClient) -> None:
    site = client.post("/api/v1/sites", json=site_payload(assets=[])).json()
    payload = {
        "siteId": site["id"],
        "name": "Mesh",
        "representation": "mesh",
        "source": {"type": "3d-tiles-url", "url": "https://example.com/tiles/tileset.json"},
        "footprint": SQUARE,
        "observedAt": "2025-05-01T12:00:00Z",
        "validFrom": "2025-05-01T00:00:00Z",
        "validTo": "2026-05-01T00:00:00Z",
        "resolution": {"groundSampleDistanceM": 0.01, "description": "1 cm GSD"},
        "crs": {"horizontal": "EPSG:4978"},
        "attribution": [{"text": "Captured by drone"}],
        "renderConfig": {"maximumScreenSpaceError": 8, "clipsWorld": True},
        "provenance": {"sourceOrganization": "Acme", "sourceUrl": "https://example.com"},
    }
    created = client.post("/api/v1/assets", json=payload)
    assert created.status_code == 201, created.text
    asset = created.json()
    assert asset["provider"] == "3d-tiles-url"
    assert asset["footprint"]["type"] == "MultiPolygon"
    assert asset["renderConfig"]["maximumScreenSpaceError"] == 8
    assert asset["provenance"]["sourceOrganization"] == "Acme"

    assert client.get(f"/api/v1/sites/{site['id']}/assets").json()[0]["id"] == asset["id"]
    assert (
        client.get("/api/v1/assets", params={"siteId": site["id"]}).json()[0]["id"] == asset["id"]
    )

    updated = client.patch(
        f"/api/v1/assets/{asset['id']}",
        json={"name": "Mesh v2", "renderConfig": {"maximumScreenSpaceError": 4}},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "Mesh v2"
    assert updated.json()["renderConfig"]["maximumScreenSpaceError"] == 4
    assert updated.json()["provenance"]["sourceOrganization"] == "Acme"

    assert client.delete(f"/api/v1/assets/{asset['id']}").status_code == 204
    assert client.get(f"/api/v1/assets/{asset['id']}").status_code == 404


def test_asset_rejects_unknown_site(client: TestClient) -> None:
    payload = {
        "siteId": "00000000-0000-0000-0000-000000000000",
        "name": "Mesh",
        "representation": "mesh",
        "source": {"type": "cesium-ion", "assetId": 1},
    }
    assert client.post("/api/v1/assets", json=payload).status_code == 404


def test_asset_validation_errors(client: TestClient) -> None:
    base = {"name": "x", "representation": "mesh", "source": {"type": "cesium-ion", "assetId": 1}}
    assert (
        client.post("/api/v1/assets", json={**base, "representation": "hologram"}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/assets", json={**base, "source": {"type": "cesium-ion", "assetId": -1}}
        ).status_code
        == 422
    )
    bad_window = {**base, "validFrom": "2026-01-01T00:00:00Z", "validTo": "2025-01-01T00:00:00Z"}
    assert client.post("/api/v1/assets", json=bad_window).status_code == 422
    assert client.post("/api/v1/assets", json={**base, "unknownField": 1}).status_code == 422
