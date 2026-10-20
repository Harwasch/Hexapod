"""The public parcel lookup (`POST /api/v1/public/parcels/near`): the landing page's site finder.

Open to anyone, so the tests hold what makes that safe: no token is needed and none is
looked at, the search area and the body are capped, each client is rate-limited, the counties
asked are an explicit list (a county whose terms restrict redistribution never, unverified ones
only behind a setting), a required disclaimer travels with its parcels, answers are cached,
lookups running at once are capped, and outside California no provider is called.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

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
from app.research.providers.parcels import (
    NO_PUBLIC_REUSE,
    PUBLIC_UNVERIFIED,
    PUBLIC_VERIFIED,
    available_sources,
)
from app.schemas.geojson import Point
from app.schemas.land_selection import CandidateResult, PublicParcelRequest
from app.services.land_selection import PUBLIC_TIMEOUT_S
from app.services.public_parcels import Busy, PublicParcels
from app.services.ratelimit import PUBLIC_PARCELS, RateLimits
from tests.test_land_selection import ALAMEDA, square

URL = "/api/v1/public/parcels/near"
OAKLAND = (-122.2711, 37.8044)
SANTA_ROSA = (-122.7141, 38.4405)
PLACERVILLE = (-120.7983, 38.7296)
BAKERSFIELD = (-119.0187, 35.3733)
SAN_RAFAEL = (-122.5311, 37.9735)


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


@contextmanager
def make_client(
    db: Session, clock: Clock, provider: dict[str, Answer], **settings: object
) -> Iterator[TestClient]:
    # A write token is configured, as in production: the route must not ask for it.
    app = create_app(
        Settings(api_write_token="the-write-token", fly_app_name="twin-api", **settings)  # type: ignore[arg-type]
    )
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


@pytest.fixture
def client(db: Session, clock: Clock, provider: dict[str, Answer]) -> Iterator[TestClient]:
    with make_client(db, clock, provider) as test_client:
        yield test_client


@pytest.fixture
def client_unverified(
    db: Session, clock: Clock, provider: dict[str, Answer]
) -> Iterator[TestClient]:
    with make_client(db, clock, provider, public_parcels_unverified=True) as test_client:
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


def test_every_available_county_is_placed_in_exactly_one_public_list() -> None:
    available = {source.fips for source in available_sources()}
    never = set(NO_PUBLIC_REUSE)
    assert not PUBLIC_VERIFIED & PUBLIC_UNVERIFIED
    assert not (PUBLIC_VERIFIED | PUBLIC_UNVERIFIED) & never
    assert PUBLIC_VERIFIED | PUBLIC_UNVERIFIED | never == available


def test_unverified_counties_and_sonoma_are_asked_only_behind_the_setting(
    client: TestClient, client_unverified: TestClient, provider: dict[str, Answer]
) -> None:
    for point, county in ((SAN_RAFAEL, "Marin"), (SANTA_ROSA, "Sonoma")):
        result = ask(client, point).json()
        assert result["status"] == "uncovered", county
        assert f"{county} County parcel records are not offered here yet" in result["message"]
    provider["answer"] = parcels(*SANTA_ROSA)
    (parcel,) = ask(client_unverified, SANTA_ROSA).json()["candidates"]
    # Sonoma's licence asks for this attribution wherever its data is shared.
    assert parcel["source"]["attribution"].startswith("County of Sonoma")
    # El Dorado stays out whatever the setting says.
    assert "does not allow" in ask(client_unverified, PLACERVILLE).json()["message"]


def test_kerns_disclaimer_travels_with_its_parcels(
    client: TestClient, provider: dict[str, Answer]
) -> None:
    provider["answer"] = parcels(*BAKERSFIELD)
    result = ask(client, BAKERSFIELD).json()
    assert result["candidates"]
    (notice,) = result["notices"]
    assert notice["source"] == "Kern County parcels"
    assert notice["text"].startswith("SUMMARY:")
    assert "without a copy of this disclaimer statement" in notice["text"]
    # A county whose terms ask for nothing more sends no notice.
    provider["answer"] = parcels(*OAKLAND)
    assert ask(client, OAKLAND).json()["notices"] == []


def test_a_county_gets_the_public_timeout_not_the_signed_in_one(
    client: TestClient, provider: dict[str, Answer]
) -> None:
    seen: list[httpx.Request] = []
    answer = parcels(*OAKLAND)

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answer(request)

    provider["answer"] = record
    ask(client, OAKLAND)
    assert seen[0].extensions["timeout"]["read"] == PUBLIC_TIMEOUT_S
    assert PUBLIC_TIMEOUT_S <= 8


def test_answers_are_cached_by_the_rounded_point(
    client: TestClient, provider: dict[str, Answer]
) -> None:
    asked: list[httpx.Request] = []
    answer = parcels(*OAKLAND)

    def record(request: httpx.Request) -> httpx.Response:
        asked.append(request)
        return answer(request)

    provider["answer"] = record
    first = ask(client, OAKLAND).json()
    # A tap a metre away is the same question.
    again = ask(client, (OAKLAND[0] + 0.00001, OAKLAND[1])).json()
    assert again == first
    assert len(asked) == 1
    # Another radius is another question.
    ask(client, OAKLAND, radiusM=100)
    assert len(asked) == 2


def test_a_county_that_did_not_answer_is_asked_again(
    client: TestClient, provider: dict[str, Answer]
) -> None:
    provider["answer"] = lambda request: httpx.Response(503)
    assert ask(client, OAKLAND).json()["status"] == "unavailable"
    provider["answer"] = parcels(*OAKLAND)
    assert ask(client, OAKLAND).json()["status"] == "available"


def test_when_every_lookup_slot_is_taken_the_answer_is_busy(client: TestClient) -> None:
    client.app.state.public_parcels = PublicParcels(max_concurrent=1, queue_wait_s=0.01)  # type: ignore[attr-defined]
    shared: PublicParcels = client.app.state.public_parcels  # type: ignore[attr-defined]
    started, release = threading.Event(), threading.Event()
    empty = CandidateResult(status="empty", message="", candidates=[])

    def slow() -> CandidateResult:
        started.set()
        release.wait(5)
        return empty

    held = threading.Thread(target=lambda: shared.lookup((0.0, 0.0, 250.0, False), slow))
    held.start()
    started.wait(5)
    try:
        response = ask(client, OAKLAND)
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "5"
        with pytest.raises(Busy):
            shared.lookup((1.0, 1.0, 250.0, False), lambda: empty)
    finally:
        release.set()
        held.join(5)
    # The slot is free again once the slow lookup ends.
    assert shared.lookup((1.0, 1.0, 250.0, False), lambda: empty) is empty


def test_the_cache_key_rounds_to_about_eleven_metres() -> None:
    def key(lon: float, lat: float) -> tuple[float, float, float, bool]:
        return PublicParcels.key(
            PublicParcelRequest(point=Point(coordinates=[lon, lat])), include_unverified=False
        )

    assert key(-122.27111, 37.80441) == key(-122.27109, 37.80439)
    assert key(-122.2711, 37.8044) != key(-122.2713, 37.8044)


def test_a_large_or_unmeasured_body_is_refused_before_it_is_read(client: TestClient) -> None:
    big = (
        '{"point": {"type": "Point", "coordinates": [-122.27, 37.8]}, "pad": "' + "x" * 5000 + '"}'
    )
    response = client.post(URL, content=big, headers={"Content-Type": "application/json"})
    assert response.status_code == 413

    def chunks() -> Iterator[bytes]:
        yield b'{"point": {"type": "Point", "coordinates": [-122.27, 37.8]}}'

    response = client.post(URL, content=chunks(), headers={"Content-Type": "application/json"})
    assert response.status_code == 411


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
