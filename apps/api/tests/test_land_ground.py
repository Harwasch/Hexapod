from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from shapely.geometry import shape

from tests.test_land import BODY
from tests.test_workspaces import IdentityClient, headers
from tests.test_workspaces import identity_client as _identity_client

identity_client = _identity_client

SOURCE = {"method": "drawn", "label": "Walked with a GPS", "meaning": "physical-feature"}


def feature(name: str, category: str, geometry: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "requestKey": str(uuid.uuid4()),
        "name": name,
        "category": category,
        "geometry": geometry,
        "source": SOURCE,
        "status": "confirmed",
        **extra,
    }


# About 300 m of fence along the land's south edge, a paddock inside it, a trough and an old track.
FENCE = {"type": "LineString", "coordinates": [[-122.139, 47.641], [-122.135, 47.641]]}
PADDOCK = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.138, 47.642],
            [-122.136, 47.642],
            [-122.136, 47.643],
            [-122.138, 47.643],
            [-122.138, 47.642],
        ]
    ],
}
TROUGH = {"type": "Point", "coordinates": [-122.137, 47.645]}
TRACK = {"type": "LineString", "coordinates": [[-122.138, 47.646], [-122.133, 47.646]]}


def test_ground_turns_the_boundary_and_features_into_plannable_zones(client: TestClient) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    path = f"/api/v1/land/{land['id']}/features"
    for body in (
        feature("North fence", "other", FENCE),
        feature("Paddock", "vegetation", PADDOCK),
        feature("Trough", "water", TROUGH),
        feature("Old track", "transport", TRACK, status="retired"),
    ):
        assert client.post(path, json=body).status_code == 201
    response = client.get(f"/api/v1/land/{land['id']}/ground")
    assert response.status_code == 200, response.text
    ground = response.json()
    assert ground["landId"] == land["id"] and ground["boundaryRevision"] == 1
    assert ground["skipped"] == 1, "the point is counted; the retired track is left out"
    items = {item["name"]: item for item in ground["items"]}
    assert set(items) == {"My land", "North fence", "Paddock"}
    boundary = items["My land"]
    assert boundary["zoneId"] == "L-" + land["id"].replace("-", "")[:8]
    assert boundary["kind"] == "boundary" and boundary["revision"] == 1
    assert boundary["featureId"] is None and boundary["line"] is None
    assert boundary["areaM2"] == pytest.approx(land["areaM2"])
    assert "Saved boundary, revision 1" in boundary["source"]
    assert "not a verified property boundary" in boundary["source"]
    paddock = items["Paddock"]
    assert paddock["kind"] == "feature" and paddock["category"] == "vegetation"
    assert shape(paddock["footprint"]).equals(shape(PADDOCK))
    assert paddock["zoneId"] == "L-" + paddock["featureId"].replace("-", "")[:8]
    fence = items["North fence"]
    assert fence["line"] == FENCE and fence["bufferM"] == 10
    assert fence["lengthM"] == pytest.approx(300, rel=0.01)
    # A 10 m corridor along ~300 m of line.
    assert fence["areaM2"] == pytest.approx(300 * 10, rel=0.05)
    assert fence["status"] == "confirmed" and "Walked with a GPS" in fence["source"]
    assert "assumed 10 m work corridor" in fence["source"]


def test_ground_is_scoped_to_the_workspace(identity_client: IdentityClient) -> None:
    client, token = identity_client
    alice = headers(token())
    workspace = client.post("/api/v1/workspaces", headers=alice, json={"name": "Ranch"}).json()[
        "id"
    ]
    land = client.post("/api/v1/land", json=BODY, headers=headers(token(), workspace)).json()
    assert (
        client.get(f"/api/v1/land/{land['id']}/ground", headers=headers(token(), workspace)).json()[
            "items"
        ][0]["kind"]
        == "boundary"
    )
    bob = headers(token("bob"))
    other = client.post("/api/v1/workspaces", headers=bob, json={"name": "Other"}).json()["id"]
    assert (
        client.get(
            f"/api/v1/land/{land['id']}/ground", headers=headers(token("bob"), other)
        ).status_code
        == 404
    )
