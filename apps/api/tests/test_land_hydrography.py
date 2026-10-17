from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import shape

from app.services import land_hydrography
from app.services.land_hydrography import parse
from tests.test_land import BODY
from tests.test_workspaces import IdentityClient, headers
from tests.test_workspaces import identity_client as _identity_client

identity_client = _identity_client

# A 0.01 degree square near Santa Ynez, CA (about 915 m east-west, 1,109 m north-south).
SQUARE: dict[str, Any] = {
    "type": "Polygon",
    "coordinates": [
        [[-120.05, 34.69], [-120.04, 34.69], [-120.04, 34.70], [-120.05, 34.70], [-120.05, 34.69]]
    ],
}


def flowline(
    identifier: str | None, fcode: int, points: list[list[float]], name: Any = None
) -> Any:
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": points},
        "properties": {"permanent_identifier": identifier, "gnis_name": name, "fcode": fcode},
    }


def collection(**extra: Any) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            # Crosses the land west to east and runs on for a kilometre each side.
            flowline("{CREEK}", 46003, [[-120.06, 34.695], [-120.03, 34.695]], "Figueroa Creek"),
            flowline("P-1", 46006, [[-120.046, 34.692], [-120.045, 34.693]]),
            flowline("E-1", 46007, [[-120.048, 34.697], [-120.047, 34.698]], " "),
            flowline("D-1", 33600, [[-120.044, 34.691], [-120.043, 34.692]]),
            # About 390 m north of the land: outside the 100 m clip.
            flowline("FAR", 46007, [[-120.05, 34.7035], [-120.04, 34.7035]]),
            # Dips into the clip twice from the north: two pieces of one flowline.
            flowline(
                "W-1",
                46007,
                [
                    [-120.049, 34.705],
                    [-120.047, 34.6995],
                    [-120.045, 34.705],
                    [-120.043, 34.6995],
                    [-120.041, 34.705],
                ],
            ),
            flowline(None, 46003, [[-120.046, 34.692], [-120.045, 34.693]]),
        ],
        **extra,
    }


@pytest.fixture(autouse=True)
def empty_cache() -> Iterator[None]:
    land_hydrography._cache.clear()
    yield
    land_hydrography._cache.clear()


def upstream(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[httpx.Request], httpx.Response]
) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answer(request)

    monkeypatch.setattr(
        land_hydrography,
        "_client",
        lambda: httpx.Client(transport=httpx.MockTransport(serve)),
    )
    return seen


def test_flowlines_are_classified_clipped_to_the_land_plus_100_m_and_split() -> None:
    features, truncated = parse(collection(), shape(SQUARE))
    assert not truncated
    by_id = {feature.id: feature for feature in features}
    assert set(by_id) == {"{CREEK}", "P-1", "E-1", "D-1", "W-1", "W-1/2"}
    creek = by_id["{CREEK}"]
    assert (creek.kind, creek.name, creek.description) == (
        "intermittent",
        "Figueroa Creek",
        "Stream/river, intermittent",
    )
    # 0.01 degrees of longitude at 34.695 N plus the 100 m clip margin on each side.
    assert creek.length_m == pytest.approx(916.2 + 200, rel=0.01)
    assert creek.geometry.coordinates[0][0] == pytest.approx(-120.0510935, abs=2e-5)
    assert features[0] is creek  # Longest first.
    assert (by_id["P-1"].kind, by_id["E-1"].kind) == ("perennial", "ephemeral")
    assert by_id["E-1"].name is None  # A blank GNIS name is no name.
    assert (by_id["D-1"].kind, by_id["D-1"].description) == ("other", "Canal/ditch")
    assert by_id["W-1/2"].permanent_identifier == "W-1"
    _, limited = parse(collection(properties={"exceededTransferLimit": True}), shape(SQUARE))
    assert limited


def test_land_hydrography_is_cited_cached_per_revision_and_private(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = upstream(monkeypatch, lambda request: httpx.Response(200, json=collection()))
    land = client.post("/api/v1/land", json={**BODY, "boundary": SQUARE}).json()
    url = f"/api/v1/land/{land['id']}/hydrography"
    response = client.get(url)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    body = response.json()
    assert body["status"] == "available" and body["boundaryRevision"] == 1
    assert body["bufferM"] == 100 and body["truncated"] is False
    assert body["message"].startswith("5 mapped flowlines within 100 m of this land")
    assert "Figueroa Creek" in body["message"]
    assert body["source"]["provider"] == "usgs-nhd"
    assert body["source"]["license"] == "Public domain (U.S. Government work)"
    assert body["source"]["attribution"].startswith("U.S. Geological Survey")
    assert body["source"]["linkUrl"].endswith("/nhd/MapServer/6")
    assert body["features"][0]["permanentIdentifier"] == "{CREEK}"
    params = seen[0].url.params
    assert seen[0].url.path == "/arcgis/rest/services/nhd/MapServer/6/query"
    assert (params["f"], params["outSR"], params["resultRecordCount"]) == ("geojson", "4326", "500")
    west, _, _, north = map(float, params["geometry"].split(","))
    assert west == pytest.approx(-120.05 - 100 / (111_320 * 0.8222), abs=2e-5)
    assert north == pytest.approx(34.70 + 100 / 110_574, abs=1e-6)
    assert str(body["source"]["url"]).startswith(str(seen[0].url).split("?")[0])
    assert client.get(url).json() == body and len(seen) == 1  # Kept for the hour.
    moved = {
        "type": "Polygon",
        "coordinates": [[[x - 0.001, y] for x, y in SQUARE["coordinates"][0]]],
    }
    revised = client.put(
        f"/api/v1/land/{land['id']}", json={**BODY, "boundary": moved, "expectedRevision": 1}
    )
    assert revised.status_code == 200, revised.text
    assert client.get(url).json()["boundaryRevision"] == 2 and len(seen) == 2


@pytest.mark.parametrize(
    "answer",
    [
        lambda request: httpx.Response(500),
        lambda request: httpx.Response(200, json={"error": {"code": 400, "message": "Bad"}}),
        lambda request: httpx.Response(200, content=b"<html>maintenance</html>"),
    ],
)
def test_an_upstream_failure_answers_unavailable_and_is_not_kept(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    answer: Callable[[httpx.Request], httpx.Response],
) -> None:
    seen = upstream(monkeypatch, answer)
    land = client.post("/api/v1/land", json={**BODY, "boundary": SQUARE}).json()
    url = f"/api/v1/land/{land['id']}/hydrography"
    response = client.get(url)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable" and body["features"] == []
    assert body["message"] == "The USGS hydrography service did not answer. Try again in a minute."
    client.get(url)
    assert len(seen) == 2  # A failure is asked again, not cached.


def test_outside_the_us_says_so_without_asking_and_an_empty_answer_is_honest(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = upstream(monkeypatch, lambda request: httpx.Response(200, json={"features": []}))
    paris = {
        "type": "Polygon",
        "coordinates": [
            [[2.35, 48.85], [2.36, 48.85], [2.36, 48.86], [2.35, 48.86], [2.35, 48.85]]
        ],
    }
    land = client.post("/api/v1/land", json={**BODY, "boundary": paris}).json()
    body = client.get(f"/api/v1/land/{land['id']}/hydrography").json()
    assert body["status"] == "uncovered" and "United States only" in body["message"]
    assert seen == []
    land = client.post("/api/v1/land", json={**BODY, "boundary": SQUARE}).json()
    body = client.get(f"/api/v1/land/{land['id']}/hydrography").json()
    assert body["status"] == "empty"
    assert body["message"] == "The USGS NHD maps no streams or ditches within 100 m of this land."


def test_hydrography_is_scoped_to_the_workspace(
    identity_client: IdentityClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream(monkeypatch, lambda request: httpx.Response(200, json=collection()))
    client, token = identity_client
    alice_space = client.post(
        "/api/v1/workspaces", json={"name": "Alice"}, headers=headers(token())
    ).json()["id"]
    alice = headers(token(), alice_space)
    land = client.post("/api/v1/land", json={**BODY, "boundary": SQUARE}, headers=alice).json()
    url = f"/api/v1/land/{land['id']}/hydrography"
    assert client.get(url, headers=alice).status_code == 200
    bob_space = client.post(
        "/api/v1/workspaces", json={"name": "Bob"}, headers=headers(token("bob"))
    ).json()["id"]
    # The cached answer is never served across workspaces.
    assert client.get(url, headers=headers(token("bob"), bob_space)).status_code == 404
    assert client.get(url).status_code == 401
