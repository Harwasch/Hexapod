"""Site scan requests (`/api/v1/scan-requests`): the landing page's form, and the list read by
the admin console.

The POST is open to anyone, so these hold what makes that safe: it is validated (name, email,
outline, parcel ids), its outline is area-capped and stored as one valid geometry, a filled
honeypot is answered like any other request and dropped, the body is capped, each client is
rate-limited, and every accepted request gets the same answer. The list needs the write token.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import DataError
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.config import Settings
from app.db import get_engine
from app.main import create_app
from app.models.scan_request import ScanRequest
from app.schemas.scan_request import MAX_RINGS, MAX_VERTICES
from app.services import scan_requests as service
from app.services.ratelimit import SCAN_REQUESTS, RateLimits
from tests.conftest import TEST_DATABASE_URL

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


def make_client(db: Session, clock: Clock, **settings: Any) -> TestClient:
    values: dict[str, Any] = {"api_write_token": WRITE, "fly_app_name": "twin-api", **settings}
    app = create_app(Settings(**values))
    app.state.rate_limits = RateLimits(clock=clock)

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    return TestClient(app)


@pytest.fixture
def client(db: Session, clock: Clock) -> Iterator[TestClient]:
    with make_client(db, clock) as test_client:
        yield test_client


def comb_up(x0: float, y0: float, teeth: int, pitch: float, length: float) -> list[Any]:
    """A comb whose fingers point north: a valid polygon on its own."""
    points = [[x0, y0]]
    x = x0
    for _ in range(teeth):
        points += [[x, y0 + length], [x + pitch / 2, y0 + length], [x + pitch / 2, y0 + pitch / 4]]
        x += pitch
        points += [[x, y0 + pitch / 4]]
    points += [[x, y0 - pitch / 4], [x0, y0 - pitch / 4], [x0, y0]]
    return [points]


def comb_right(x0: float, y0: float, teeth: int, pitch: float, length: float) -> list[Any]:
    """A comb whose fingers point east: laid over `comb_up`, every finger crosses every other."""
    points = [[x0, y0]]
    y = y0
    for _ in range(teeth):
        points += [[x0 + length, y], [x0 + length, y + pitch / 2], [x0 + pitch / 4, y + pitch / 2]]
        y += pitch
        points += [[x0 + pitch / 4, y]]
    points += [[x0 - pitch / 4, y], [x0 - pitch / 4, y0], [x0, y0]]
    return [points]


def crossed_combs(teeth: int) -> dict[str, Any]:
    pitch = 0.0002
    length = teeth * pitch
    return {
        "type": "MultiPolygon",
        "coordinates": [
            comb_up(-122.27, 37.80, teeth, pitch, length),
            comb_right(-122.27 - pitch, 37.80 + pitch / 3, teeth, pitch, length + 2 * pitch),
        ],
    }


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


# ---- Review of 2026-10-20 (Codex, confirmed against a local Postgres) ---------------------


def test_crossed_combs_are_refused_in_seconds_not_minutes(client: TestClient, db: Session) -> None:
    # 11 KB of two comb-shaped parcels whose fingers cross: dissolving them took 16.5 s
    # unbounded (measured 2026-10-20), 49 s at 60 fingers. Under the statement timeout it is
    # refused as an outline that could not be processed, and nothing is stored.
    started = time.monotonic()
    response = send(client, request(boundary=crossed_combs(40), parcelIds=[]))
    elapsed = time.monotonic() - started
    assert response.status_code == 422, response.text
    assert "could not be processed" in response.text
    assert elapsed < 6, elapsed
    assert stored(db) == 0
    # The session is usable again: the next request is stored.
    assert send(client, request(), ip="198.51.100.4").status_code == 202


def test_vertex_and_ring_caps_refuse_before_postgis(
    client: TestClient, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_database(*_: object, **__: object) -> None:
        pytest.fail("an outline over the caps must be refused before PostGIS sees it")

    monkeypatch.setattr(service, "clean_outline", no_database)
    many = MAX_VERTICES // 2 + 1
    ring = [[-122.27 + 0.00001 * i, 37.8] for i in range(many)]
    ring += [[-122.27 + 0.00001 * (many - 1 - i), 37.801] for i in range(many)]
    ring.append(ring[0])
    response = send(client, request(boundary={"type": "Polygon", "coordinates": [ring]}))
    assert response.status_code == 422
    assert "vertices" in response.text
    holes = [square(-122.2711, 37.8044, half=0.01)[0]] + [
        square(-122.2711 + 0.0005 * (i % 20), 37.8044 + 0.0005 * (i // 20), half=0.0001)[0]
        for i in range(MAX_RINGS)
    ]
    response = send(
        client, request(boundary={"type": "Polygon", "coordinates": holes}), ip="198.51.100.4"
    )
    assert response.status_code == 422
    assert "rings" in response.text
    assert stored(db) == 0


def test_a_sliver_that_would_not_survive_storage_is_refused(
    client: TestClient, db: Session
) -> None:
    # Two billionths of a degree wide: stored as sent, ST_AsGeoJSON's 9 digits flatten it, and
    # every read of the list failed. Reduced to the stored precision, it is no area at all.
    sliver = [[-122.27, 37.8], [-122.26, 37.8], [-122.26, 37.800000002], [-122.27, 37.8]]
    response = send(client, request(boundary={"type": "Polygon", "coordinates": [sliver]}))
    assert response.status_code == 422, response.text
    assert stored(db) == 0


def test_one_unreadable_row_does_not_take_the_list_down(client: TestClient, db: Session) -> None:
    send(client, request(name="Readable"))
    # A row written before the precision guard: a sliver two ten-billionths of a degree wide,
    # which ST_AsGeoJSON's nine decimals flatten into a ring that is not a polygon.
    db.execute(
        text(
            "INSERT INTO scan_requests (id, name, email, notes, boundary, parcel_ids) VALUES ("
            "gen_random_uuid(), 'Sliver', 'sliver@example.com', '', ST_Multi(ST_GeomFromText("
            "'POLYGON((-122.27 37.8, -122.26 37.8, -122.26 37.8000000002, -122.27 37.8))', 4326)),"
            " '{}')"
        )
    )
    db.commit()
    rows = {row["name"]: row for row in listed(client)}
    assert rows["Readable"]["boundary"]["type"] == "MultiPolygon"
    assert rows["Sliver"]["boundary"] is None
    assert rows["Sliver"]["boundaryUnreadable"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"notes": "fine\x00then a NUL"},
        {"name": "Ada\x07"},
        {"email": "ada\x00@example.com"},
    ],
)
def test_control_characters_are_refused_at_validation(
    client: TestClient, db: Session, overrides: dict[str, Any]
) -> None:
    response = send(client, request(**overrides))
    assert response.status_code == 422
    assert "control characters" in response.text
    assert stored(db) == 0


def test_a_database_refusal_is_generic_and_carries_no_personal_data(
    client: TestClient, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_: object) -> None:
        raise DataError("INSERT ...", {"email": "ada@example.com"}, Exception("bad byte"))

    monkeypatch.setattr(db, "add", refuse)
    response = send(client, request())
    assert response.status_code == 422
    assert "ada@example.com" not in response.text
    assert "could not be stored" in response.text


def test_the_engine_keeps_sql_parameters_out_of_its_errors() -> None:
    # The reviewed case: a NUL in a text value, refused by the driver. Without hide_parameters
    # the error's message lists every parameter of the statement, the email among them.
    engine = get_engine(TEST_DATABASE_URL)
    with engine.connect() as connection, pytest.raises(DataError) as raised:
        connection.execute(
            text("SELECT CAST(:email AS text), CAST(:notes AS text)"),
            {"email": "secret@example.com", "notes": "a NUL\x00here"},
        )
    assert "secret@example.com" not in str(raised.value)
    assert "parameters hidden" in str(raised.value)


def test_malformed_bodies_spend_tokens_too(client: TestClient) -> None:
    for _ in range(SCAN_REQUESTS.burst):
        response = client.post(
            URL,
            content=b"{not json",
            headers={"Content-Type": "application/json", "Fly-Client-IP": "203.0.113.9"},
        )
        assert response.status_code == 422
    response = client.post(
        URL,
        content=b"{not json",
        headers={"Content-Type": "application/json", "Fly-Client-IP": "203.0.113.9"},
    )
    assert response.status_code == 429


def test_a_daily_cap_across_all_clients_says_try_later(db: Session, clock: Clock) -> None:
    with make_client(db, clock, scan_requests_daily_cap=2) as client:
        assert send(client, request(), ip="198.51.100.1").status_code == 202
        # A honeypot is not stored, so it does not count.
        assert send(client, request(website="x"), ip="198.51.100.2").status_code == 202
        assert send(client, request(), ip="198.51.100.3").status_code == 202
        capped = send(client, request(), ip="198.51.100.4")
        assert capped.status_code == 503
        assert "try again later" in capped.text.lower()
        assert int(capped.headers["Retry-After"]) >= 60
        assert stored(db) == 2


def test_the_list_refuses_when_no_write_token_is_configured(db: Session, clock: Clock) -> None:
    with make_client(db, clock, api_write_token=None) as client:
        assert send(client, request()).status_code == 202
        response = client.get(URL)
        assert response.status_code == 403
        assert "API_WRITE_TOKEN" in response.text


@pytest.mark.parametrize(
    "email",
    [
        "person@example.com?bcc=x",
        "person@example.com&cc=x",
        "a b@example.com",
        "person@localhost",
        "<person@example.com>",
        "person@example..com",
    ],
)
def test_email_is_validated_strictly(client: TestClient, db: Session, email: str) -> None:
    assert send(client, request(email=email)).status_code == 422
    assert stored(db) == 0


def test_requests_can_be_pruned_by_age(engine_inspector: Any) -> None:
    indexes = {index["name"]: index["column_names"] for index in engine_inspector}
    assert indexes.get("ix_scan_requests_created_at") == ["created_at"]


@pytest.fixture
def engine_inspector(db: Session) -> Any:
    return inspect(db.get_bind()).get_indexes("scan_requests")
