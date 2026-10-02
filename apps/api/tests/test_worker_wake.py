"""Queueing a job starts the worker's machine: the other half of the worker's idle exit.

Fly's Machines API is replaced by an `httpx.MockTransport` that keeps a small table of
machines, so what is asserted is the conversation -- which machines are listed, which are
started, with what credentials -- and never a real app. The routes are tested for *when*
they ask: after a job is committed, and not after a request that queued nothing.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.config import Settings
from app.main import create_app
from app.models import Capture, Job
from app.models.capture import CaptureFile
from app.models.enums import CaptureKind, RunStatus, UploadStatus
from app.services import phone_key, worker_wake
from app.services.worker_wake import Wake, after_enqueue, wake_workers
from tests.test_job_api import failed_run

APP = "twin-api"


def fly_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "fly_api_token": "FlyV1 fm2_secret",
        "fly_app_name": APP,
        "queue_check_url": None,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def machine(machine_id: str, state: str, group: str = "worker") -> dict[str, Any]:
    return {
        "id": machine_id,
        "state": state,
        "config": {"metadata": {"fly_process_group": group}},
    }


class FakeFly:
    """The Machines API, as far as waking is concerned: list, and start."""

    def __init__(self, machines: list[dict[str, Any]]) -> None:
        self.machines = {m["id"]: m for m in machines}
        self.requests: list[httpx.Request] = []
        self.fail_start: set[str] = set()
        self.status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "nope"})
        path = request.url.path
        if request.method == "GET" and path == f"/v1/apps/{APP}/machines":
            return httpx.Response(200, json=list(self.machines.values()))
        if request.method == "POST" and path.endswith("/start"):
            machine_id = path.split("/")[-2]
            if machine_id in self.fail_start:
                return httpx.Response(412, json={"error": "failed_precondition"})
            self.machines[machine_id]["state"] = "started"
            return httpx.Response(200, json={"previous_state": "stopped"})
        if path.endswith("/start") or "hc-ping" in str(request.url):
            return httpx.Response(200, text="OK")
        return httpx.Response(404)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def started(self) -> list[str]:
        return [
            r.url.path.split("/")[-2]
            for r in self.requests
            if r.method == "POST" and r.url.path.endswith("/start")
        ]


# --- the call itself ----------------------------------------------------------------


def test_with_no_token_nothing_is_called() -> None:
    fly = FakeFly([machine("w1", "stopped")])
    assert wake_workers(fly_settings(fly_api_token=None), transport=fly.transport) is None
    assert wake_workers(fly_settings(fly_app_name=None), transport=fly.transport) is None
    assert after_enqueue(fly_settings(fly_api_token=None), transport=fly.transport) is None
    assert fly.requests == []


def test_stopped_and_suspended_worker_machines_are_started_and_nothing_else() -> None:
    fly = FakeFly(
        [
            machine("w-stopped", "stopped"),
            machine("w-suspended", "suspended"),
            machine("w-gone", "destroyed"),
            machine("app-stopped", "stopped", group="app"),
        ]
    )

    wake = wake_workers(fly_settings(), transport=fly.transport)

    assert wake == Wake(started=("w-stopped", "w-suspended"), up=())
    assert fly.started() == ["w-stopped", "w-suspended"]
    # A macaroon token goes in as Fly prints it; anything else is a bearer token.
    assert {r.headers["authorization"] for r in fly.requests} == {"FlyV1 fm2_secret"}
    plain = FakeFly([machine("w", "stopped")])
    wake_workers(fly_settings(fly_api_token="legacy"), transport=plain.transport)
    assert plain.requests[0].headers["authorization"] == "Bearer legacy"


def test_a_worker_already_up_is_looked_at_again_and_started_if_it_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The race: the job was committed just as the worker decided to exit, so the first
    look finds it still `started`. The second finds it stopped, and starts it."""
    fly = FakeFly([machine("w1", "started")])
    later: list[tuple[float, Callable[[], object]]] = []

    wake = after_enqueue(
        fly_settings(),
        transport=fly.transport,
        later=lambda delay, call: later.append((delay, call)),
    )

    assert wake == Wake(started=(), up=("w1",))
    assert fly.started() == []
    assert [delay for delay, _ in later] == [worker_wake.RECHECK_AFTER_S]

    fly.machines["w1"]["state"] = "stopped"  # it went, without the job
    later[0][1]()
    assert fly.started() == ["w1"]


def test_a_wake_that_starts_everything_does_not_look_again() -> None:
    fly = FakeFly([machine("w1", "stopped")])
    later: list[object] = []
    after_enqueue(fly_settings(), transport=fly.transport, later=lambda *a: later.append(a))
    assert later == []


def test_failures_are_logged_never_raised(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # alembic/env.py's `fileConfig` (the `engine` fixture runs it, earlier in a full run)
    # disables every logger that already exists, this one included.
    monkeypatch.setattr(worker_wake.log, "disabled", False)
    down = FakeFly([machine("w1", "stopped")])
    down.status = 503
    assert wake_workers(fly_settings(), transport=down.transport) is None

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    assert wake_workers(fly_settings(), transport=httpx.MockTransport(unreachable)) is None
    garbled = httpx.MockTransport(lambda request: httpx.Response(200, text="<html>"))
    assert wake_workers(fly_settings(), transport=garbled) is None

    # One machine refusing to start does not stop the others.
    partly = FakeFly([machine("w1", "stopped"), machine("w2", "stopped")])
    partly.fail_start = {"w1"}
    assert wake_workers(fly_settings(), transport=partly.transport) == Wake(started=("w2",))
    assert "could not start worker machine w1" in caplog.text
    assert "fm2_secret" not in caplog.text


def test_the_queue_check_gets_its_start_ping() -> None:
    fly = FakeFly([machine("w1", "stopped")])
    after_enqueue(
        fly_settings(queue_check_url="https://hc-ping.com/abc/"),
        transport=fly.transport,
        later=lambda *a: None,
    )
    pinged = [str(r.url) for r in fly.requests if r.url.host == "hc-ping.com"]
    assert pinged == ["https://hc-ping.com/abc/start"]

    # And without a Fly token, the ping alone.
    only = FakeFly([])
    after_enqueue(
        fly_settings(fly_api_token=None, queue_check_url="https://hc-ping.com/abc"),
        transport=only.transport,
    )
    assert [str(r.url) for r in only.requests] == ["https://hc-ping.com/abc/start"]


# --- when the routes ask --------------------------------------------------------------

KEY = "abcd-efgh-jkmn"
PHONE = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
def wakes() -> list[int]:
    """Every wake a route scheduled, each recording how many queued jobs a session of its
    own could see at that moment -- which is how "after the commit" is checked."""
    return []


@pytest.fixture
def client(
    db: Session, engine: Engine, wakes: list[int], monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    def recording() -> None:
        with Session(engine) as fresh:
            wakes.append(
                fresh.scalar(select(func.count(Job.id)).where(Job.status == RunStatus.NOT_STARTED))
                or 0
            )

    monkeypatch.setattr(worker_wake, "after_enqueue", recording)
    settings = Settings(
        api_phone_key_hash=phone_key.hash_key(KEY, salt=b"0123456789abcdef", iterations=1000),
    )
    app = create_app(settings)

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    with TestClient(app) as test_client:
        yield test_client


def uploaded(db: Session, capture_id: uuid.UUID) -> None:
    db.add(
        CaptureFile(
            capture_id=capture_id,
            filename="scan.ply",
            content_type="application/octet-stream",
            bytes=10,
            storage_key=f"captures/{capture_id}/source/scan.ply",
            status=UploadStatus.COMPLETE,
        )
    )
    db.commit()


def test_processing_a_capture_wakes_the_worker_after_the_job_is_committed(
    client: TestClient, db: Session, wakes: list[int]
) -> None:
    capture = Capture(slug="yard", name="Yard", kind=CaptureKind.GAUSSIAN_SPLAT)
    db.add(capture)
    db.commit()
    uploaded(db, capture.id)

    response = client.post(
        f"/api/v1/captures/{capture.id}/process", json={"recipe": "splat-ingest"}
    )
    assert response.status_code == 202, response.text
    assert wakes == [1]

    # A request that queues nothing wakes nothing: here, the same capture again.
    again = client.post(f"/api/v1/captures/{capture.id}/process", json={"recipe": "splat-ingest"})
    assert again.status_code == 409
    assert wakes == [1]


def test_a_retry_wakes_the_worker(client: TestClient, db: Session, wakes: list[int]) -> None:
    job = failed_run(db, slug="orchard-retry")

    response = client.post(f"/api/v1/jobs/{job.id}/retry", json={})
    assert response.status_code == 200, response.text
    assert wakes == [1]
    missing = client.post(f"/api/v1/jobs/{uuid.uuid4()}/retry", json={})
    assert missing.status_code == 404
    assert wakes == [1]


def test_a_phone_run_wakes_the_worker(client: TestClient, db: Session, wakes: list[int]) -> None:
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    uploaded(db, uuid.UUID(mine["id"]))

    response = client.post(
        f"/api/v1/phone/captures/{mine['id']}/process",
        json={"recipe": "photo-reconstruct"},
        headers=PHONE,
    )
    assert response.status_code == 202, response.text
    assert wakes == [1]
    refused = client.post(
        f"/api/v1/phone/captures/{mine['id']}/process", json={"recipe": "x"}, headers=PHONE
    )
    assert refused.status_code == 409
    assert wakes == [1]


def test_the_machine_listing_is_what_the_fly_api_documents() -> None:
    """The shape relied on, written down once: `config.metadata.fly_process_group`."""
    listing = json.loads(json.dumps([machine("w1", "stopped")]))
    fly = FakeFly(listing)
    assert wake_workers(fly_settings(), transport=fly.transport) == Wake(started=("w1",))
    assert fly.requests[0].url == httpx.URL(f"https://api.machines.dev/v1/apps/{APP}/machines")
    assert fly.requests[1].url.path == f"/v1/apps/{APP}/machines/w1/start"
