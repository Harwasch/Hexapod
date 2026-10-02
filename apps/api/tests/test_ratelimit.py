"""The per-client rate limits (app/services/ratelimit.py), and what they must never catch.

The limits guard three costs: the phone key's PBKDF2, the bucket walk, and step-log
reads. What they must not do is get in the way of a person -- above all the phone page,
which polls while a run is going. Its interval is read out of its own source here, so a
change to it that the limits would not survive fails in this file rather than on a phone.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.config import Settings
from app.main import create_app
from app.models import Capture, Job, JobStep
from app.models.enums import CaptureKind, RunStatus
from app.services import phone_key
from app.services.ratelimit import (
    PHONE_KEY,
    RECONCILE,
    STEP_LOG,
    Limit,
    RateLimited,
    RateLimits,
    TokenBucket,
)
from app.storage import S3Storage, get_storage

KEY = "abcd-efgh-jkmn"
WRITE = "the-write-token"
UPLOAD_PAGE = Path(__file__).resolve().parents[3] / "apps/web/src/upload/main.ts"


class Clock:
    """Time that moves only when a test says so."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def client(db: Session, storage: S3Storage, clock: Clock) -> Iterator[TestClient]:
    settings = Settings(
        api_write_token=WRITE,
        api_phone_key_hash=phone_key.hash_key(KEY, salt=b"0123456789abcdef", iterations=1000),
    )
    app = create_app(settings)
    app.state.rate_limits = RateLimits(clock=clock)

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    app.dependency_overrides[get_storage] = lambda: storage
    with TestClient(app) as test_client:
        yield test_client


def bearer(token: str, ip: str = "203.0.113.7") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Fly-Client-IP": ip}


# --- the bucket ------------------------------------------------------------------------


def test_a_bucket_allows_its_burst_then_refills_at_its_rate(clock: Clock) -> None:
    bucket = TokenBucket(Limit(burst=3, per_minute=6), clock=clock)
    for _ in range(3):
        bucket.take("a")
    with pytest.raises(RateLimited) as refused:
        bucket.take("a")
    assert refused.value.retry_after_s == pytest.approx(10.0)  # 6 a minute: one per 10 s
    bucket.take("b")  # another client has its own bucket
    clock.now += 10
    bucket.take("a")
    with pytest.raises(RateLimited):
        bucket.take("a")
    clock.now += 3600
    for _ in range(3):  # refilled to the burst, and not past it
        bucket.take("a")
    with pytest.raises(RateLimited):
        bucket.take("a")


def test_a_returned_token_does_not_count(clock: Clock) -> None:
    bucket = TokenBucket(Limit(burst=2, per_minute=1), clock=clock)
    for _ in range(50):
        bucket.take("a")
        bucket.give_back("a")
    bucket.take("a")
    bucket.take("a")
    with pytest.raises(RateLimited):
        bucket.take("a")


def test_the_bucket_forgets_the_least_recent_client_first(clock: Clock) -> None:
    bucket = TokenBucket(Limit(burst=1, per_minute=1), clock=clock, max_clients=2)
    bucket.take("a")
    bucket.take("b")
    bucket.take("c")  # "a" is forgotten, and comes back with a full bucket
    bucket.take("a")
    with pytest.raises(RateLimited):
        bucket.take("c")


# --- the phone key ---------------------------------------------------------------------


def test_wrong_phone_keys_are_limited_and_right_ones_never_are(
    client: TestClient, clock: Clock
) -> None:
    for _ in range(3 * PHONE_KEY.burst):
        assert client.post("/api/v1/phone/check", headers=bearer(KEY)).status_code == 204
    for _ in range(PHONE_KEY.burst):
        assert client.post("/api/v1/phone/check", headers=bearer("wrong")).status_code == 401
    refused = client.post("/api/v1/phone/check", headers=bearer("wrong"))
    assert refused.status_code == 429
    assert int(refused.headers["Retry-After"]) >= 1
    assert refused.json()["title"] == "Too many requests"
    # The limit is per client: another address is unaffected...
    assert client.post("/api/v1/phone/check", headers=bearer(KEY, "198.51.100.1")).status_code == (
        204
    )
    # ...and it holds on every phone route, before the hash is computed at all.
    assert client.post("/api/v1/phone/captures", json={}, headers=bearer(KEY)).status_code == 429
    clock.now += 60 / PHONE_KEY.per_minute
    assert client.post("/api/v1/phone/check", headers=bearer(KEY)).status_code == 204


def test_without_the_proxy_header_the_socket_address_is_the_client(
    db: Session, clock: Clock
) -> None:
    """`API_CLIENT_IP_HEADER` empty: a forged Fly-Client-IP buys nothing."""
    settings = Settings(
        api_client_ip_header="",
        api_phone_key_hash=phone_key.hash_key(KEY, salt=b"0123456789abcdef", iterations=1000),
    )
    app = create_app(settings)
    app.state.rate_limits = RateLimits(clock=clock)
    app.dependency_overrides[_db] = lambda: db
    with TestClient(app) as test_client:
        statuses = [
            test_client.post(
                "/api/v1/phone/check", headers=bearer("wrong", f"192.0.2.{index}")
            ).status_code
            for index in range(PHONE_KEY.burst + 1)
        ]
    assert statuses[-1] == 429


# --- the phone page's polling ------------------------------------------------------------


def poll_interval_s() -> float:
    match = re.search(r"const POLL_MS = ([\d_]+);", UPLOAD_PAGE.read_text(encoding="utf-8"))
    assert match, f"POLL_MS is no longer declared in {UPLOAD_PAGE}; update this test"
    return int(match.group(1).replace("_", "")) / 1000


def test_the_phone_page_polling_trips_nothing(
    client: TestClient, db: Session, storage: S3Storage, clock: Clock
) -> None:
    """The page, run at its own interval: the key typed once, a capture made, and then
    the reads it polls (apps/web/src/upload/main.ts: `follow` and `refreshMine`). It
    reads no step log -- but if it read one on every tick it would still be far inside
    that limit, so that is polled too. Twenty minutes: long enough to have spent every
    burst twice over had nothing refilled, so passing is about the sustained rate."""
    interval = poll_interval_s()
    assert interval >= 5, "the phone page polls faster than these limits were sized for"
    assert 60 / interval < STEP_LOG.per_minute

    phone = bearer(KEY)
    assert client.post("/api/v1/phone/check", headers=phone).status_code == 204
    capture = client.post("/api/v1/phone/captures", json={}, headers=phone).json()["capture"]
    job_id, step_id = _job_with_a_log(db, storage, uuid.UUID(capture["id"]))

    ticks = int(20 * 60 / interval)
    for _ in range(ticks):
        clock.now += interval
        for response in (
            client.get(f"/api/v1/jobs?captureId={capture['id']}&limit=5", headers=phone),
            client.get("/api/v1/captures?limit=100", headers=phone),
            client.get("/api/v1/jobs?limit=100", headers=phone),
            client.get(f"/api/v1/jobs/{job_id}/steps/{step_id}/log", headers=phone),
        ):
            assert response.status_code == 200, response.text


def _job_with_a_log(db: Session, storage: S3Storage, capture_id: uuid.UUID) -> tuple[str, str]:
    capture = db.get(Capture, capture_id)
    assert capture is not None
    job = Job(capture_id=capture.id, recipe="splat-ingest", recipe_version="1", params={})
    db.add(job)
    db.flush()
    step = JobStep(
        job_id=job.id,
        stage_id="normalize",
        ordinal=0,
        impl="x",
        status=RunStatus.IN_PROGRESS,
        log_key=f"runs/{job.id}/normalize/log.txt",
    )
    db.add(step)
    db.commit()
    storage.put_object(step.log_key or "", b"working\n", "text/plain")
    return str(job.id), str(step.id)


# --- reconciliation and step logs --------------------------------------------------------


def test_reconciliation_is_limited_after_the_token_check(client: TestClient, clock: Clock) -> None:
    # Refused for the token, these spend nothing: the limit is for those who hold it.
    for _ in range(3 * RECONCILE.burst):
        assert client.post("/api/v1/storage/reconciliation").status_code == 401
    auth = bearer(WRITE)
    for _ in range(RECONCILE.burst):
        assert client.post("/api/v1/storage/reconciliation", headers=auth).status_code == 200
    refused = client.post("/api/v1/storage/reconciliation", headers=auth)
    assert refused.status_code == 429
    assert refused.headers["Retry-After"] == str(int(60 / RECONCILE.per_minute))
    clock.now += 60 / RECONCILE.per_minute
    assert client.post("/api/v1/storage/reconciliation", headers=auth).status_code == 200


def test_reconciliation_is_no_longer_an_open_get(client: TestClient) -> None:
    assert client.get("/api/v1/storage/reconciliation").status_code == 405


def test_step_log_reads_are_limited(
    client: TestClient, db: Session, storage: S3Storage, clock: Clock
) -> None:
    capture = Capture(slug="logs", name="Logs", kind=CaptureKind.VIDEO)
    db.add(capture)
    db.commit()
    job_id, step_id = _job_with_a_log(db, storage, capture.id)
    url = f"/api/v1/jobs/{job_id}/steps/{step_id}/log"
    for _ in range(STEP_LOG.burst):
        assert client.get(url, headers=bearer("anyone")).status_code == 200
    assert client.get(url, headers=bearer("anyone")).status_code == 429
    clock.now += 1
    assert client.get(url, headers=bearer("anyone")).status_code == 200
