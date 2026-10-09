"""CPU-only lease and cost controls; all provider/gateway calls are mocked."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.worlds.config import WorldsSettings
from app.worlds.lifecycle import Lifecycle
from app.worlds.store import Store
from tests import test_worlds as worlds_fixtures
from tests.test_worlds import connect, start

setup = worlds_fixtures.setup


def record_worker(db, timestamp=1000, **overrides):
    value = {
        "id": "worker",
        "provider": "runpod",
        "providerId": "pod-id",
        "managed": True,
        "status": "ready",
        "gatewayUrl": "https://worker.example.test",
        "estimatedHourlyCost": 1,
        "createdAt": datetime.fromtimestamp(timestamp, UTC).isoformat(),
        "idleDeadline": timestamp + 300,
        "hardDeadline": timestamp + 3600,
    }
    value.update(overrides)
    return db.put("worker", value)


def record_session(db, **overrides):
    value = {
        "id": "session",
        "workerId": "worker",
        "status": "playing",
        "leaseExpiresAt": 1090,
        "heartbeatIntervalSeconds": 20,
        "hardDeadline": 4600,
    }
    value.update(overrides)
    return db.put("session", value)


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    config = WorldsSettings(_env_file=None, data_dir=tmp_path, max_worker_hourly_cost=2)
    db = Store(tmp_path)
    calls = []
    monkeypatch.setattr(
        "app.worlds.lifecycle.request",
        lambda method, url, **kw: calls.append((method, url)) or httpx.Response(204),
    )
    monkeypatch.setattr(
        "app.worlds.providers.RunPodProvider.destroy",
        lambda self, worker: calls.append(("destroy", worker["providerId"])),
    )
    return Lifecycle(config, db), db, calls


def test_browser_disappears_lease_then_idle_worker_terminated(lifecycle):
    runner, db, calls = lifecycle
    record_worker(db)
    record_session(db)
    runner.tick(1091)
    assert db.get("session", "session")["status"] == "stopped"
    assert calls == [("DELETE", "https://worker.example.test/sessions/session")]
    runner.tick(1301)
    assert db.get("worker", "worker")["status"] == "destroyed"
    assert calls[-1] == ("destroy", "pod-id")


def test_passive_reads_and_inactive_heartbeat_cannot_extend_lease(setup):
    client, config, _ = setup
    session = start(client, connect(client))
    original = session["leaseExpiresAt"]
    client.get(f"/api/v1/worlds/sessions/{session['id']}")
    client.get(f"/api/v1/worlds/sessions/{session['id']}/frame")
    response = client.post(
        f"/api/v1/worlds/sessions/{session['id']}/heartbeat", json={"active": False}
    )
    assert response.status_code == 200
    assert response.json()["leaseExpiresAt"] == original
    assert Store(config.data_dir).get("session", session["id"])["leaseExpiresAt"] == original


def test_expired_lease_cannot_revive_and_heartbeat_honors_hard_limit(lifecycle):
    _, db, _ = lifecycle
    record_worker(db, hardDeadline=1120)
    record_session(db)
    renewed = db.heartbeat("session", 1080, 90, 300, active=True)
    assert renewed["leaseExpiresAt"] == 1120
    assert db.get("worker", "worker")["idleDeadline"] == 1120
    assert db.heartbeat("session", 1121, 90, 300, active=True) is None


def test_hard_lifetime_terminates_even_with_live_lease(lifecycle):
    runner, db, calls = lifecycle
    record_worker(db, hardDeadline=1200, idleDeadline=9999)
    record_session(db, leaseExpiresAt=9999)
    runner.tick(1201)
    assert calls == [("destroy", "pod-id")]
    assert db.get("session", "session")["status"] == "stopped"
    assert db.get("worker", "worker")["terminationReason"] == "hard-lifetime"


def test_restart_reconciles_unknown_pod_without_new_provisioning(lifecycle, monkeypatch):
    runner, db, calls = lifecycle
    record_worker(db, providerId=None, status="unknown", operationName="worlds-unique-intent")
    monkeypatch.setattr(
        "app.worlds.providers.RunPodProvider.find_worker",
        lambda self, name: {
            "providerId": "late-pod",
            "status": "starting",
            "managed": True,
            "gatewayUrl": "https://late.example.test",
            "estimatedHourlyCost": 1,
        },
    )
    # A fresh process sees the same durable intent and destroys the late allocation.
    Lifecycle(runner.config, Store(runner.config.data_dir)).tick(1100)
    assert calls == [("destroy", "late-pod")]
    assert db.get("worker", "worker")["status"] == "destroyed"


def test_cleanup_failure_retries_without_freeing_reserved_capacity(lifecycle, monkeypatch):
    runner, db, calls = lifecycle
    record_worker(db, idleDeadline=1001)

    def fail(*args):
        raise HTTPException(502, "redacted")

    monkeypatch.setattr("app.worlds.providers.RunPodProvider.destroy", fail)
    runner.tick(1100)
    assert db.get("worker", "worker")["status"] == "terminating"
    assert (
        db.reserve_worker({"id": "another", "managed": True, "status": "provisioning"}, 1) is False
    )
    monkeypatch.setattr(
        "app.worlds.providers.RunPodProvider.destroy",
        lambda self, worker: calls.append(("destroy", worker["providerId"])),
    )
    runner.tick(db.get("worker", "worker")["cleanupUntil"] + 1)
    assert db.get("worker", "worker")["status"] == "destroyed"


def test_external_compute_never_destroyed_by_reaper(lifecycle):
    runner, db, calls = lifecycle
    record_worker(db, managed=False, hardDeadline=1001)
    record_session(db)
    runner.tick(1300)
    assert all(call[0] != "destroy" for call in calls)
    assert db.get("session", "session")["status"] == "stopped"


def enable_provision(client, config):
    config.runpod_allow_provision = True
    config.data_persistent = True
    config.max_worker_hourly_cost = 2
    config.runpod_api_key = SecretStr("test-provider-key")
    config.runpod_template_id = "approved"
    config.gateway_token = SecretStr("test-worker-token-at-least-32-characters")
    client.app.state.worlds_lifecycle = SimpleNamespace(running=True)


def test_provision_failclosed_without_running_reaper(setup):
    client, config, calls = setup
    enable_provision(client, config)
    client.app.state.worlds_lifecycle.running = False
    assert client.post("/api/v1/worlds/workers", json={"provider": "runpod"}).status_code == 503
    assert calls == []


def test_price_cap_terminates_new_worker_before_usable_response(setup, monkeypatch):
    client, config, _ = setup
    enable_provision(client, config)
    monkeypatch.setattr(
        "app.worlds.providers.RunPodProvider.create_worker",
        lambda *args, **kw: {
            "providerId": "pricey",
            "managed": True,
            "estimatedHourlyCost": 3,
            "gatewayUrl": "https://pricey.example.test",
            "status": "starting",
        },
    )
    deleted = []
    monkeypatch.setattr(
        "app.worlds.providers.RunPodProvider.destroy",
        lambda self, value: deleted.append(value["providerId"]),
    )
    response = client.post("/api/v1/worlds/workers", json={"provider": "runpod"})
    assert response.status_code == 409
    assert deleted == ["pricey"]
    assert Store(config.data_dir).list("worker")[0]["status"] == "destroyed"


def test_gpu_selection_must_match_operator_allowlist(setup):
    client, config, calls = setup
    enable_provision(client, config)
    assert (
        client.post(
            "/api/v1/worlds/workers", json={"provider": "runpod", "gpuTypeId": "expensive GPU"}
        ).status_code
        == 422
    )
    assert calls == []


def test_explicit_env_file_and_environment_precedence(tmp_path, monkeypatch):
    path = tmp_path / "worlds.env"
    path.write_text("WORLD_SESSION_LEASE_SECONDS=120\nWORLD_MAX_MANAGED_WORKERS=2\n")
    monkeypatch.setenv("WORLD_ENV_FILE", str(path))
    monkeypatch.setenv("WORLD_MAX_MANAGED_WORKERS", "1")
    config = WorldsSettings.load()
    assert config.session_lease_seconds == 120
    assert config.max_managed_workers == 1


def test_readiness_is_authenticated_non_provisioning_and_reports_limits(setup):
    client, _, calls = setup
    value = client.get("/api/v1/worlds/readiness").json()
    assert value["lifecycle"]["sessionLeaseSeconds"] == 90
    assert value["provisioningEnabled"] is False
    assert all(call.method == "GET" for call in calls)
    assert (
        client.get("/api/v1/worlds/readiness", headers={"Authorization": "Bearer bad"}).status_code
        == 401
    )


def test_lifespan_does_not_create_worlds_files_when_unconfigured(tmp_path, monkeypatch):
    from fastapi import FastAPI

    from app.worlds.lifecycle import worlds_lifespan

    config = WorldsSettings(_env_file=None, data_dir=tmp_path / "unused")
    monkeypatch.setattr(WorldsSettings, "load", lambda: config)
    app = FastAPI(lifespan=worlds_lifespan)
    app.state.settings = SimpleNamespace(api_write_token=None)
    with TestClient(app):
        assert not hasattr(app.state, "worlds_lifecycle")
    assert not config.data_dir.exists()


def test_cleanup_rechecks_latest_heartbeat_atomically(lifecycle):
    _, db, _ = lifecycle
    record_worker(db)
    record_session(db)
    assert db.heartbeat("session", 1080, 90, 300, active=True)
    # A reaper scanned the old 1090 deadline before the concurrent heartbeat.
    assert db.claim_cleanup("session", "session", 1091, expired_field="leaseExpiresAt") is None
    assert db.claim_cleanup("worker", "worker", 1301, expired_field="idleDeadline") is None
    assert db.claim_cleanup("worker", "worker", 1381, expired_field="idleDeadline") is not None
    # Once cleanup has atomically claimed the worker it cannot be revived.
    db.patch("session", "session", {"leaseExpiresAt": 2000})
    assert db.heartbeat("session", 1382, 90, 300, active=True) is None


def test_concurrent_provision_reservations_enforce_fleet_limit(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    db = Store(tmp_path)
    barrier = Barrier(8)

    def reserve(index):
        barrier.wait(timeout=5)
        return db.reserve_worker({"id": str(index), "managed": True, "status": "provisioning"}, 2)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(reserve, range(8)))
    assert sum(results) == 2
    assert len(db.records("worker")) == 2


def test_pooled_transport_reuses_client_without_persisting_request_auth(monkeypatch):
    from app.worlds import providers

    original = httpx.Client
    created = []
    tokens = []

    def handler(req):
        tokens.append(req.headers.get("authorization"))
        return httpx.Response(204)

    def build(**kwargs):
        created.append(1)
        return original(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "Client", build)
    providers.start_transport()
    try:
        providers.request("GET", "https://one.example.test", token="first-token")
        providers.request("GET", "https://two.example.test", token="different-token")
        providers.request("GET", "https://one.example.test")
    finally:
        providers.stop_transport()
    assert len(created) == 1
    assert tokens == ["Bearer first-token", "Bearer different-token", None]


def test_reaper_health_requires_a_recent_successful_cycle(lifecycle, monkeypatch):
    import time

    runner, _, _ = lifecycle
    monkeypatch.setattr(runner.thread, "is_alive", lambda: True)
    runner.last_tick = time.time()
    assert runner.running is False
    runner.last_success = time.time()
    assert runner.running is True
    runner.last_success = time.time() - 181
    assert runner.running is False


def test_stop_and_detach_reserve_worker_before_concurrent_session_claim(lifecycle, monkeypatch):
    _, db, _ = lifecycle
    monkeypatch.setattr("app.worlds.store.time.time", lambda: 1001)
    record_worker(db)
    outcome, _ = db.begin_shutdown("worker", "stopping", require_idle=True)
    assert outcome == "claimed"
    assert (
        db.claim_session({"id": "new", "workerId": "worker", "status": "starting"}) == "not-ready"
    )
    assert db.records("session") == []
    db.put("worker", record_worker(db, managed=False))
    outcome, _ = db.begin_shutdown("worker", "detaching", require_idle=True)
    assert outcome == "claimed"
    assert (
        db.claim_session({"id": "new", "workerId": "worker", "status": "starting"}) == "not-ready"
    )


def test_winning_session_claim_prevents_stop_and_detach(lifecycle, monkeypatch):
    _, db, _ = lifecycle
    monkeypatch.setattr("app.worlds.store.time.time", lambda: 1001)
    record_worker(db)
    assert db.claim_session({"id": "new", "workerId": "worker", "status": "starting"}) == "claimed"
    assert db.begin_shutdown("worker", "stopping", require_idle=True)[0] == "occupied"
    assert db.begin_shutdown("worker", "detaching", require_idle=True)[0] == "occupied"
    assert db.get("worker", "worker")["status"] == "ready"


def test_uncertain_stop_retries_and_keeps_storage_deadline(lifecycle, monkeypatch):
    runner, db, calls = lifecycle
    record_worker(db, status="stopping", cleanupError="unconfirmed stop")
    monkeypatch.setattr(
        "app.worlds.providers.RunPodProvider.stop",
        lambda self, worker: calls.append(("stop", worker["providerId"])),
    )
    runner.tick(1100)
    assert calls == [("stop", "pod-id")]
    assert db.get("worker", "worker")["status"] == "stopped"
    runner.tick(1301)
    assert calls[-1] == ("destroy", "pod-id")
    assert db.get("worker", "worker")["status"] == "destroyed"


def test_explicit_missing_config_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_ENV_FILE", str(tmp_path / "missing.env"))
    with pytest.raises(RuntimeError, match="WORLD_ENV_FILE"):
        WorldsSettings.load()
