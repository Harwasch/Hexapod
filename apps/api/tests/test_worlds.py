"""Worlds API tests need no PostgreSQL, GPU, credentials or provider calls."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.services.errors import UnauthorizedError
from app.worlds.config import WorldsSettings, validate_url
from app.worlds.providers import RunPodProvider
from app.worlds.router import router, settings
from app.worlds.store import Store
from tests.worlds_types import WorldsSetup, require_record


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WorldsSetup:
    config = WorldsSettings(
        _env_file=None,
        data_dir=tmp_path,
        local_gateway_url="http://127.0.0.1:8181",
        gateway_token="test-worker-token",
    )
    calls = []
    session_ids = []

    def handle(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        assert req.headers["authorization"] == "Bearer test-worker-token"
        path = req.url.path
        if path == "/health":
            return httpx.Response(
                200, json={"status": "ready", "models": [{"id": "astronex-world"}]}
            )
        if path == "/sessions" and req.method == "POST":
            body = json.loads(req.content)
            session_ids.append(body["id"])
            return httpx.Response(
                200, json={"id": body["id"], "status": "loading", "internalPath": "/secret/path"}
            )
        if path.endswith("/frame"):
            return httpx.Response(
                200,
                content=b"fake-jpeg",
                headers={"content-type": "image/jpeg", "x-frame-id": "12"},
            )
        if path.endswith("/actions"):
            return httpx.Response(
                200,
                json={
                    "accepted": True,
                    "mechanism": "chunk-boundary",
                    "revision": 3,
                    "prompt": "private",
                },
            )
        if path.endswith("/offer"):
            return httpx.Response(501, json={"secret": "must never leak"})
        if path.endswith("/snapshot"):
            return httpx.Response(200, json={"resumeKind": "visual-checkpoint", "frameId": 12})
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"status": "playing", "frameId": 12})

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original_client(**kw, transport=httpx.MockTransport(handle))
    )
    app = FastAPI()
    app.state.settings = SimpleNamespace(api_write_token="test-write-token")
    app.add_exception_handler(
        UnauthorizedError,
        lambda req, exc: JSONResponse(status_code=401, content={"detail": str(exc)}),
    )
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[settings] = lambda: config
    client = TestClient(app, headers={"Authorization": "Bearer test-write-token"})
    return client, config, calls


def connect(client: TestClient) -> dict[str, Any]:
    response = client.post("/api/v1/worlds/workers", json={"provider": "local"})
    assert response.status_code == 201, response.text
    return cast(dict[str, Any], response.json())


def start(client: TestClient, worker: Any) -> dict[str, Any]:
    response = client.post(
        "/api/v1/worlds/sessions",
        json={"workerId": worker["id"], "modelId": "astronex-world", "prompt": "redwood forest"},
    )
    assert response.status_code == 201, response.text
    return cast(dict[str, Any], response.json())


def test_requires_token_for_private_reads_and_writes(setup: WorldsSetup) -> None:
    client, _, calls = setup
    for path in ["providers", "catalog", "workers", "sessions"]:
        response = client.get(f"/api/v1/worlds/{path}", headers={"Authorization": "Bearer bad"})
        assert response.status_code == 401
    assert not calls


def test_gateway_session_full_lifecycle(setup: WorldsSetup) -> None:
    client, config, calls = setup
    worker = connect(client)
    assert worker["status"] == "ready"
    assert "gatewayUrl" not in worker
    session = start(client, worker)
    assert "internalPath" not in session
    sid = session["id"]
    assert client.get(f"/api/v1/worlds/sessions/{sid}").json()["status"] == "playing"
    action = client.post(
        f"/api/v1/worlds/sessions/{sid}/actions", json={"type": "native", "action": "forward"}
    )
    assert action.json()["accepted"] is True
    assert action.json()["revision"] == 3
    assert "prompt" not in action.json()
    frame = client.get(f"/api/v1/worlds/sessions/{sid}/frame")
    assert frame.content == b"fake-jpeg"
    assert frame.headers["cache-control"] == "no-store"
    assert frame.headers["x-frame-id"] == "12"
    assert (
        client.post(f"/api/v1/worlds/sessions/{sid}/snapshot").json()["resumeKind"]
        == "visual-checkpoint"
    )
    assert client.delete(f"/api/v1/worlds/workers/{worker['id']}").status_code == 409
    assert client.delete(f"/api/v1/worlds/sessions/{sid}").json()["status"] == "stopped"
    calls_before = len(calls)
    assert client.delete(f"/api/v1/worlds/sessions/{sid}").status_code == 200
    assert len(calls) == calls_before
    assert (
        client.post(f"/api/v1/worlds/sessions/{sid}/actions", json={"type": "pause"}).status_code
        == 409
    )
    removed = client.delete(f"/api/v1/worlds/workers/{worker['id']}").json()
    assert removed["status"] == "detached"
    assert "continues" in removed["message"]
    assert b"redwood forest" not in (config.data_dir / "metadata.sqlite3").read_bytes()


def test_cloud_configuration_missing_does_not_provision(setup: WorldsSetup) -> None:
    client, config, calls = setup
    config.runpod_allow_provision = False
    response = client.post("/api/v1/worlds/workers", json={"provider": "runpod"})
    assert response.status_code == 503
    assert calls == []
    providers = client.get("/api/v1/worlds/providers").json()["providers"]
    assert providers[0]["id"] == "runpod"
    assert providers[0]["configured"] is False


def test_unsupported_webrtc_is_honest_and_redacted(setup: WorldsSetup) -> None:
    client, _, _ = setup
    session = start(client, connect(client))
    response = client.post(
        f"/api/v1/worlds/sessions/{session['id']}/offer", json={"type": "offer", "sdp": "offer"}
    )
    assert response.status_code == 501
    assert "must never leak" not in response.text


def test_input_prevents_urls_and_paths(setup: WorldsSetup) -> None:
    client, _, _ = setup
    worker = connect(client)
    for value in ["http://169.254.169.254/latest/meta-data", "/etc/passwd"]:
        response = client.post(
            "/api/v1/worlds/sessions",
            json={
                "workerId": worker["id"],
                "modelId": "astronex-world",
                "prompt": "test",
                "inputs": {"images": [value]},
            },
        )
        assert response.status_code == 422


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example.com",
        "https://u:p@example.com",
        "https://example.com?token=secret",
        "file:///etc/passwd",
    ],
)
def test_gateway_url_security(url: str) -> None:
    with pytest.raises(ValueError):
        validate_url(url)


def test_runpod_uses_fixed_template_and_token_only_on_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = WorldsSettings(
        _env_file=None,
        data_dir=tmp_path,
        runpod_api_key="provider-secret",
        gateway_token="worker-secret",
        runpod_allow_provision=True,
        runpod_template_id="approved-template",
    )
    seen = []

    def fake_request(
        method: str, url: str, *, token: Any = None, payload: Any = None, **kw: Any
    ) -> httpx.Response:
        seen.append((method, url, token, payload))
        return httpx.Response(
            200, json={"id": "pod123", "costPerHr": 1.25, "desiredStatus": "RUNNING"}
        )

    monkeypatch.setattr("app.worlds.providers.request", fake_request)
    provider = RunPodProvider(config)
    worker = provider.create_worker()
    assert worker["managed"] is True
    assert worker["gatewayUrl"] == "https://pod123-8789.proxy.runpod.net"
    body = seen[0][3]
    assert body["templateId"] == "approved-template"
    assert body["env"] == {
        "WORLD_GATEWAY_TOKEN": "worker-secret",
        "WORLD_MODEL_ID": "astronex-world",
        "PORT": "8789",
        "WORLD_BIND": "0.0.0.0",
    }
    assert "provider-secret" not in json.dumps(body)
    provider.status(worker)
    provider.stop(worker)
    provider.destroy(worker)
    assert [(call[0], call[1]) for call in seen] == [
        ("POST", "https://rest.runpod.io/v1/pods"),
        ("GET", "https://rest.runpod.io/v1/pods/pod123"),
        ("POST", "https://rest.runpod.io/v1/pods/pod123/stop"),
        ("DELETE", "https://rest.runpod.io/v1/pods/pod123"),
    ]


def test_failed_session_can_be_cleaned_up_without_orphaning_worker(
    setup: WorldsSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    client, _, _ = setup
    worker = connect(client)

    def missing(*args: Any, **kwargs: Any) -> Any:
        raise HTTPException(404, "not found")

    monkeypatch.setattr("app.worlds.router.gateway", missing)
    response = client.post(
        "/api/v1/worlds/sessions",
        json={"workerId": worker["id"], "modelId": "astronex-world", "prompt": "forest"},
    )
    assert response.status_code == 404
    session = client.get("/api/v1/worlds/sessions").json()["sessions"][0]
    assert session["status"] == "error"
    assert client.delete(f"/api/v1/worlds/sessions/{session['id']}").status_code == 200
    assert client.delete(f"/api/v1/worlds/workers/{worker['id']}").status_code == 200


def test_unknown_provisioning_outcome_never_claims_teardown(
    setup: WorldsSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import FastAPI, HTTPException

    client, config, _ = setup
    config.runpod_allow_provision = True
    config.data_persistent = True
    config.max_worker_hourly_cost = 2
    from pydantic import SecretStr

    config.runpod_api_key = SecretStr("test-provider-key")
    config.runpod_template_id = "approved"
    config.gateway_token = SecretStr("test-worker-token-at-least-32-characters")
    cast(FastAPI, client.app).state.worlds_lifecycle = SimpleNamespace(running=True)

    def timeout(*args: Any, **kwargs: Any) -> Any:
        raise HTTPException(504, "timeout")

    monkeypatch.setattr("app.worlds.providers.RunPodProvider.create_worker", timeout)
    assert client.post("/api/v1/worlds/workers", json={"provider": "runpod"}).status_code == 504
    worker = client.get("/api/v1/worlds/workers").json()["workers"][0]
    assert worker["status"] == "unknown"
    response = client.delete(f"/api/v1/worlds/workers/{worker['id']}")
    assert response.status_code == 409
    assert "console" in response.text


def test_error_body_never_exposes_upstream_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from app.worlds.providers import request

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(
            **kw,
            transport=httpx.MockTransport(
                lambda req: httpx.Response(500, text="secret-token prompt-image")
            ),
        ),
    )
    with pytest.raises(HTTPException) as exc:
        request("GET", "https://example.test")
    assert "secret-token" not in str(exc.value.detail)
    assert "prompt-image" not in str(exc.value.detail)


def test_response_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from app.worlds.providers import request

    original = httpx.Client
    monkeypatch.setattr("app.worlds.providers.MAX_RESPONSE_BYTES", 20)
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(
            **kw, transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"a" * 21))
        ),
    )
    with pytest.raises(HTTPException) as exc:
        request("GET", "https://example.test")
    assert exc.value.status_code == 502


def test_active_sessions_check_is_not_limited_to_latest_500(tmp_path: Path) -> None:

    db = Store(tmp_path)
    db.put("session", {"id": "old", "workerId": "busy-worker", "status": "playing"})
    for index in range(501):
        db.put("session", {"id": str(index), "workerId": "other", "status": "stopped"})
    assert db.has_active_sessions("busy-worker") is True
    assert db.has_active_sessions("other") is False


def test_destroy_managed_worker_recovers_crashed_sessions(
    setup: WorldsSetup, monkeypatch: pytest.MonkeyPatch
) -> None:

    client, config, _ = setup
    db = Store(config.data_dir)
    db.put(
        "worker",
        {
            "id": "managed",
            "provider": "runpod",
            "providerId": "pod-id",
            "status": "ready",
            "managed": True,
            "gatewayUrl": "https://example.test",
        },
    )
    db.put("session", {"id": "crashed-session", "workerId": "managed", "status": "error"})
    destroyed = []
    monkeypatch.setattr(
        "app.worlds.providers.RunPodProvider.destroy",
        lambda self, value: destroyed.append(value["providerId"]),
    )
    response = client.delete("/api/v1/worlds/workers/managed")
    assert response.status_code == 200
    assert destroyed == ["pod-id"]
    assert require_record(db.get("session", "crashed-session"))["status"] == "stopped"


def test_sessions_cannot_start_on_stopped_worker(setup: WorldsSetup) -> None:

    client, config, _ = setup
    db = Store(config.data_dir)
    db.put(
        "worker",
        {
            "id": "stopped",
            "provider": "local",
            "status": "stopped",
            "gatewayUrl": "http://127.0.0.1:8181",
        },
    )
    response = client.post(
        "/api/v1/worlds/sessions",
        json={"workerId": "stopped", "modelId": "astronex-world", "prompt": "forest"},
    )
    assert response.status_code == 409
    assert db.list("session") == []


def test_unreachable_external_worker_does_not_claim_unknown_provisioning(
    setup: WorldsSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    client, _, _ = setup

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise HTTPException(502, "unreachable")

    monkeypatch.setattr("app.worlds.providers.GatewayProvider.create_worker", unavailable)
    assert client.post("/api/v1/worlds/workers", json={"provider": "local"}).status_code == 502
    worker = client.get("/api/v1/worlds/workers").json()["workers"][0]
    assert worker["status"] == "error"
    assert client.delete(f"/api/v1/worlds/workers/{worker['id']}").status_code == 200


def test_busy_worker_rejects_second_session_before_gateway_call(setup: WorldsSetup) -> None:
    client, _, calls = setup
    worker = connect(client)
    first = start(client, worker)
    before = len(calls)
    response = client.post(
        "/api/v1/worlds/sessions",
        json={"workerId": worker["id"], "modelId": "astronex-world", "prompt": "second tab"},
    )
    assert response.status_code == 409
    assert "active or unresolved" in response.json()["detail"]
    assert len(calls) == before
    assert len(client.get("/api/v1/worlds/sessions").json()["sessions"]) == 1
    assert client.delete(f"/api/v1/worlds/sessions/{first['id']}").status_code == 200
    assert start(client, worker)["id"] != first["id"]


def test_worker_claim_is_atomic_across_concurrent_connections(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    db = Store(tmp_path)
    db.put("worker", {"id": "shared", "status": "ready"})
    barrier = Barrier(8)

    def claim(index: int) -> Any:
        # Separate Store instances model distinct request handlers/API processes;
        # the transaction lock belongs to SQLite, not a process-local mutex.
        connection = Store(tmp_path)
        barrier.wait(timeout=5)
        return connection.claim_session(
            {"id": str(index), "workerId": "shared", "status": "starting"}
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(claim, range(8)))
    assert results.count("claimed") == 1
    assert results.count("occupied") == 7
    assert len(db.list("session")) == 1
    winner = db.list("session")[0]
    winner["status"] = "error"
    db.put("session", winner)
    assert (
        db.claim_session({"id": "retry", "workerId": "shared", "status": "starting"}) == "occupied"
    )
    winner["status"] = "stopped"
    db.put("session", winner)
    assert (
        db.claim_session({"id": "retry", "workerId": "shared", "status": "starting"}) == "claimed"
    )


@pytest.mark.parametrize(
    "value",
    ["https://example.test:not-a-port", "https://example.test:65536", "https://example.test\n"],
)
def test_invalid_gateway_configuration_is_validated_before_http(value: Any) -> None:
    with pytest.raises(ValueError):
        validate_url(value)


def test_malformed_gateway_health_is_sanitized(
    setup: WorldsSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    from app.worlds.providers import GatewayProvider

    _, config, _ = setup
    monkeypatch.setattr(
        "app.worlds.providers.request",
        lambda *args, **kwargs: httpx.Response(200, json={"status": []}),
    )
    with pytest.raises(HTTPException) as exc:
        GatewayProvider("local", config).create_worker()
    assert exc.value.status_code == 503


def test_boolean_provider_price_is_unknown() -> None:
    from app.worlds.providers import hourly_cost

    assert hourly_cost({"costPerHr": True}) is None
    assert hourly_cost({"costPerHr": "0.74"}) == 0.74


def test_creation_body_is_bounded_before_json_decoding(setup: WorldsSetup) -> None:
    client, _, calls = setup
    response = client.post(
        "/api/v1/worlds/sessions",
        content=b"not-json",
        headers={"Content-Type": "application/json", "Content-Length": str(8 * 1024 * 1024)},
    )
    assert response.status_code == 413
    assert calls == []


def test_chunked_body_cannot_bypass_worlds_request_limit(setup: WorldsSetup) -> None:
    client, _, calls = setup

    def chunks() -> Iterator[bytes]:
        yield b"x" * (128 * 1024)
        yield b"x"

    response = client.post(
        "/api/v1/worlds/workers", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert calls == []


@pytest.mark.parametrize("authorization", ["", "Bearer incorrect"])
def test_unauthorized_json_is_rejected_before_stream_read(
    setup: WorldsSetup, monkeypatch: pytest.MonkeyPatch, authorization: str
) -> None:
    from starlette.requests import Request

    client, _, calls = setup

    async def unread(_self: Any) -> AsyncIterator[bytes]:
        pytest.fail("Unauthorized Worlds payload must not be consumed")
        yield b""

    monkeypatch.setattr(Request, "stream", unread)
    response = client.post(
        "/api/v1/worlds/sessions",
        content=b"{large inline image}",
        headers={"Authorization": authorization, "Content-Type": "application/json"},
    )
    assert response.status_code == 401
    assert calls == []


def test_continuous_worker_status_keeps_revisions_without_echoing_private_input() -> None:
    from app.worlds.router import session_update

    metadata = {
        "queuedRevision": 5,
        "appliedRevision": 3,
        "generatingRevision": 4,
        "bufferedChunks": 1,
        "totalGeneratedFrames": 97,
        "totalGenerationSeconds": 9.5,
        "modelLoadSeconds": 12.0,
        "modelLoadCount": 1,
        "transformerLoadCount": 1,
        "promptEncodingCount": 4,
        "continuity": "latent-overlap",
        "residency": "gpu",
        "renderedWindowCount": 4,
        "samplingSeconds": 2.5,
        "promptTruncated": True,
        "peakVRAMBytes": 123456,
        "audioAvailable": True,
        "interactionMode": "continuous-chunks",
    }
    result = session_update({}, {**metadata, "prompt": "private", "audio": "/private/audio.pcm"})
    assert result == metadata
