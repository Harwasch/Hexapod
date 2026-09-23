from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import site_payload


def test_bookmark_lifecycle(client: TestClient) -> None:
    site = client.post("/api/v1/sites", json=site_payload()).json()
    listed = client.get(f"/api/v1/sites/{site['id']}/bookmarks").json()
    assert len(listed) == 1 and listed[0]["isDefault"] is True

    created = client.post(
        f"/api/v1/sites/{site['id']}/bookmarks",
        json={
            "name": "Close-up",
            "longitude": -122.138,
            "latitude": 47.6445,
            "height": 140,
            "pitch": -15,
            "isDefault": True,
        },
    )
    assert created.status_code == 201, created.text
    listed = client.get(f"/api/v1/sites/{site['id']}/bookmarks").json()
    assert [b["isDefault"] for b in listed] == [False, True]

    bad = client.post(
        f"/api/v1/sites/{site['id']}/bookmarks",
        json={"name": "Bad", "longitude": 500, "latitude": 0, "height": 0},
    )
    assert bad.status_code == 422

    assert (
        client.delete(f"/api/v1/sites/{site['id']}/bookmarks/{created.json()['id']}").status_code
        == 204
    )
    assert (
        client.delete(f"/api/v1/sites/{site['id']}/bookmarks/{created.json()['id']}").status_code
        == 404
    )
    assert (
        client.get("/api/v1/sites/00000000-0000-0000-0000-000000000000/bookmarks").status_code
        == 404
    )
