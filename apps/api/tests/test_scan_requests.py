"""Site scan requests (`/api/v1/scan-requests`): the landing page's form, and the list read by
the admin console.

The POST is open to anyone, so these hold what makes that safe: it is validated (name, email,
outline, parcel ids), its outline is area-capped and stored as one valid geometry, a filled
honeypot is answered like any other request and dropped, the body is capped, each client is
rate-limited, and every accepted request gets the same answer. The list needs the write token.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.config import Settings
from app.main import create_app
from app.models.scan_request import ScanRequest
from app.services.ratelimit import SCAN_REQUESTS, RateLimits

URL = "/api/v1/scan-requests"
WRITE = "the-write-token"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def client(db: Session, clock: Clock) -> Iterator[TestClient]:
    app = create_app(Settings(api_write_token=WRITE, fly_app_name="twin-api"))
    app.state.rate_limits = RateLimits(clock=clock)

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    with TestClient(app) as test_client:
        yield test_client


def square(lon: float, lat: float, half: float = 0.001) -> list[list[list[float]]]:
    return [
        [
            [lon - half, lat - half],
            [lon + half, lat - half],
            [lon + half, lat + half],
            [lon - half, lat + half],
            [lon - half, lat - half],
        ]
    ]


def request(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "notes": "Organization: Oak Bench Ranch\nBroom on the north slope.",
        "boundary": {"type": "Polygon", "coordinates": square(-122.2711, 37.8044)},
        "parcelIds": ["ca-parcel/06001/2-51-1"],
        "website": "",
    }
    body.update(overrides)
    return body


def send(client: TestClient, body: dict[str, Any], ip: str = "203.0.113.7") -> Response:
    return client.post(URL, json=body, headers={"Fly-Client-IP": ip})


def stored(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(ScanRequest)) or 0


def listed(client: TestClient) -> list[dict[str, Any]]:
    response = client.get(URL, headers={"Authorization": f"Bearer {WRITE}"})
    assert response.status_code == 200, response.text
    rows: list[dict[str, Any]] = response.json()
    return rows


def test_a_request_is_stored_without_a_token_and_read_back_with_one(
    client: TestClient, db: Session
) -> None:
    response = send(client, request())
    assert response.status_code == 202, response.text
    assert response.json() == {"status": "received"}
    (row,) = listed(client)
    assert row["name"] == "Ada Lovelace"
    assert row["email"] == "ada@example.com"
    assert row["notes"].endswith("Broom on the north slope.")
    assert row["parcelIds"] == ["ca-parcel/06001/2-51-1"]
    assert row["boundary"]["type"] == "MultiPolygon"
    assert 30_000 < row["areaM2"] < 40_000  # 0.002 degrees square at 37.8 N
    assert row["id"] and row["createdAt"]


def test_adjacent_parcels_are_stored_as_one_valid_outline(client: TestClient) -> None:
    # Two parcels sharing an edge: as a MultiPolygon that is invalid, so they are dissolved.
    boundary = {
        "type": "MultiPolygon",
        "coordinates": [square(-122.2711, 37.8044), square(-122.2691, 37.8044)],
    }
    ids = ["ca-parcel/06001/2-51-1", "ca-parcel/06001/2-51-2"]
    assert send(client, request(boundary=boundary, parcelIds=ids)).status_code == 202
    (row,) = listed(client)
    assert row["boundary"]["type"] == "MultiPolygon"
    assert len(row["boundary"]["coordinates"]) == 1
    assert row["parcelIds"] == ids


def test_acres_alone_are_a_request_with_no_outline(client: TestClient) -> None:
    body = request(boundary=None, parcelIds=[], notes="Area (acres): 40")
    assert send(client, body).status_code == 202
    (row,) = listed(client)
    assert row["boundary"] is None and row["areaM2"] is None
    assert row["notes"] == "Area (acres): 40"


def test_a_filled_honeypot_gets_the_same_answer_and_is_dropped(
    client: TestClient, db: Session
) -> None:
    real = send(client, request())
    bot = send(client, request(website="https://spam.example"), ip="198.51.100.4")
    assert bot.status_code == real.status_code == 202
    assert bot.json() == real.json()
    assert stored(db) == 1


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"email": "ada@"}, "email"),
        ({"email": "not an email"}, "email"),
        ({"name": "   "}, "name"),
        ({"name": "x" * 201}, "name"),
        ({"notes": "x" * 5001}, "notes"),
        ({"boundary": None}, "body"),  # parcel ids without the outline they make up
        ({"parcelIds": ["../../etc/passwd"]}, "parcelIds"),
        ({"parcelIds": [f"ca-parcel/06001/{n}" for n in range(51)]}, "parcelIds"),
        ({"boundary": {"type": "Point", "coordinates": [-122.27, 37.8]}}, "boundary"),
    ],
)
def test_malformed_requests_are_refused(
    client: TestClient, db: Session, overrides: dict[str, Any], field: str
) -> None:
    response = send(client, request(**overrides))
    assert response.status_code == 422, response.text
    assert field in response.text
    assert stored(db) == 0


def test_an_outline_larger_than_the_cap_is_refused(client: TestClient, db: Session) -> None:
    # About 0.2 degrees square: some 390 km2, past the 50 km2 cap.
    big = {"type": "Polygon", "coordinates": square(-122.2711, 37.8044, half=0.1)}
    response = send(client, request(boundary=big, parcelIds=[]))
    assert response.status_code == 422
    assert "at most 50 km2" in response.text
    assert stored(db) == 0


def test_a_body_over_the_cap_is_refused_before_it_is_read(client: TestClient, db: Session) -> None:
    body = request(notes="x" * 300_000)
    response = client.post(URL, json=body)
    assert response.status_code == 413
    assert stored(db) == 0


def test_each_client_is_rate_limited_tightly(client: TestClient, db: Session, clock: Clock) -> None:
    for _ in range(SCAN_REQUESTS.burst):
        assert send(client, request()).status_code == 202
    limited = send(client, request())
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) == 60
    assert send(client, request(), ip="198.51.100.4").status_code == 202
    clock.now += 60
    assert send(client, request()).status_code == 202
    assert stored(db) == SCAN_REQUESTS.burst + 2


def test_the_list_needs_the_write_token(client: TestClient) -> None:
    send(client, request())
    assert client.get(URL).status_code == 401
    assert client.get(URL, headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get(URL, headers={"Authorization": f"Bearer {WRITE}"})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"


def test_the_list_is_newest_first_and_bounded(client: TestClient, clock: Clock) -> None:
    for name in ("First", "Second", "Third"):
        send(client, request(name=name))
        clock.now += 60
    assert [row["name"] for row in listed(client)] == ["Third", "Second", "First"]
    response = client.get(f"{URL}?limit=2", headers={"Authorization": f"Bearer {WRITE}"})
    assert [row["name"] for row in response.json()] == ["Third", "Second"]
    assert (
        client.get(f"{URL}?limit=501", headers={"Authorization": f"Bearer {WRITE}"}).status_code
        == 422
    )
