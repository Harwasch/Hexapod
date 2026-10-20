"""The public parcel lookup (`POST /api/v1/public/parcels/near`): the landing page's site finder.

Open to anyone, so the tests hold what makes that safe: no token is needed and none is
looked at, the search area is capped, each client is rate-limited, a county whose terms
restrict redistribution is never asked, and outside California no provider is called.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.api.v1.public import parcel_http
from app.config import Settings
from app.main import create_app
from app.models.land import LandArea
from app.research.providers.parcels import NO_PUBLIC_REUSE
from app.services.ratelimit import PUBLIC_PARCELS, RateLimits
from tests.test_land_selection import ALAMEDA, square

URL = "/api/v1/public/parcels/near"
OAKLAND = (-122.2711, 37.8044)
SANTA_ROSA = (-122.7141, 38.4405)
PLACERVILLE = (-120.7983, 38.7296)


class Clock:
    """Time that moves only when a test says so."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


Answer = Callable[[httpx.Request], httpx.Response]


def parcels(lon: float, lat: float) -> Answer:
    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "type": "FeatureCollection",
                "features": [
                    {"id": 1, "geometry": square(lon, lat), "properties": {"APN": "012-345-678"}}
                ],
            },
        )

    return answer


def refuse(request: httpx.Request) -> httpx.Response:
    pytest.fail(f"No parcel service should have been asked: {request.url}")


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def provider() -> dict[str, Answer]:
    """What the county services answer; a test sets `provider["answer"]`."""
    return {"answer": refuse}


@pytest.fixture
def client(db: Session, clock: Clock, provider: dict[str, Answer]) -> Iterator[TestClient]:
    # A write token is configured, as in production: the route must not ask for it.
    app = create_app(Settings(api_write_token="the-write-token", fly_app_name="twin-api"))
    app.state.rate_limits = RateLimits(clock=clock)

    def override_db() -> Iterator[Session]:
        yield db

    def override_http() -> Iterator[httpx.Client]:
        transport = httpx.MockTransport(lambda request: provider["answer"](request))
        with httpx.Client(transport=transport) as stub:
            yield stub

    app.dependency_overrides[_db] = override_db
    app.dependency_overrides[parcel_http] = override_http
    with TestClient(app) as test_client:
        yield test_client


def ask(
    client: TestClient, point: tuple[float, float], ip: str = "203.0.113.7", **extra: object
) -> Response:
    body = {"point": {"type": "Point", "coordinates": list(point)}, **extra}
    return client.post(URL, json=body, headers={"Fly-Client-IP": ip})


def test_finds_california_parcels_without_a_token_or_a_workspace(
    client: TestClient, provider: dict[str, Answer], db: Session
) -> None:
    seen: list[httpx.Request] = []
    answer = parcels(*OAKLAND)

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answer(request)

    provider["answer"] = record
    response = ask(client, OAKLAND)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "available"
    (parcel,) = result["candidates"]
    assert parcel["id"] == "ca-parcel/06001/012-345-678"
    # The same provenance the signed-in picker shows, attribution included.
    assert parcel["source"]["meaning"] == "recorded-parcel"
    assert parcel["source"]["attribution"] == "Alameda County Information Technology Department"
    assert ALAMEDA + "/query" in str(seen[0].url)
    # Only the parcel number is asked for: no owner or other attributes cross this route.
    assert seen[0].url.params["outFields"] == "APN"
    # Nothing was written.
    assert db.scalar(select(func.count()).select_from(LandArea)) == 0


def test_outside_california_no_provider_is_called(client: TestClient) -> None:
    for point in ((2.35, 48.86), (-122.135, 47.645)):
        response = ask(client, point)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["status"] == "uncovered" and result["candidates"] == []
        assert "California" in result["message"]


def test_a_county_whose_terms_restrict_reuse_is_never_asked(client: TestClient) -> None:
    assert "06017" in NO_PUBLIC_REUSE
    response = ask(client, PLACERVILLE)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "uncovered"
    assert "El Dorado County does not allow" in result["message"]


def test_sonoma_parcels_carry_the_attribution_its_licence_asks_for(
    client: TestClient, provider: dict[str, Answer]
) -> None:
    provider["answer"] = parcels(*SANTA_ROSA)
    (parcel,) = ask(client, SANTA_ROSA).json()["candidates"]
    assert parcel["source"]["attribution"].startswith("County of Sonoma")


def test_the_search_area_is_capped(client: TestClient, provider: dict[str, Answer]) -> None:
    assert ask(client, OAKLAND, radiusM=301).status_code == 422
    assert ask(client, OAKLAND, radiusM=0).status_code == 422
    provider["answer"] = parcels(*OAKLAND)
    assert ask(client, OAKLAND, radiusM=300).status_code == 200


def test_each_client_is_rate_limited_and_recovers(
    client: TestClient, provider: dict[str, Answer], clock: Clock
) -> None:
    provider["answer"] = parcels(*OAKLAND)
    for _ in range(PUBLIC_PARCELS.burst):
        assert ask(client, OAKLAND).status_code == 200
    limited = ask(client, OAKLAND)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) == 5
    # Another visitor has a bucket of their own.
    assert ask(client, OAKLAND, ip="198.51.100.4").status_code == 200
    # One lookup is back after five seconds (12 a minute).
    clock.now += 5
    assert ask(client, OAKLAND).status_code == 200
    assert ask(client, OAKLAND).status_code == 429


def test_a_rejected_request_still_spends_a_token(client: TestClient) -> None:
    # The bucket is in front of validation's work too: malformed floods are limited.
    for _ in range(PUBLIC_PARCELS.burst):
        ask(client, OAKLAND, radiusM=5000)
    assert ask(client, OAKLAND, radiusM=5000).status_code == 429
