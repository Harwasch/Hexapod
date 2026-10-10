"""Worlds reconstruction remains isolated and private, with no live GPU invocation."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image
from starlette import formparsers
from starlette.datastructures import Headers

from app.api.deps import require_write_token
from app.config import Settings
from app.services.errors import UnauthorizedError
from app.worlds import reconstruction as recon


@pytest.fixture(autouse=True)
def no_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_RECONSTRUCTION_GATEWAY_URL", raising=False)
    monkeypatch.delenv("WORLD_RECONSTRUCTION_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("WORLD_ENV_FILE", raising=False)


def source(name: str = "/Users/alice/private.png", content: bytes | None = None) -> UploadFile:
    if content is None:
        output = io.BytesIO()
        Image.new("RGB", (2, 2), "blue").save(output, format="PNG")
        content = output.getvalue()
    return UploadFile(
        io.BytesIO(content), filename=name, headers=Headers({"content-type": "image/png"})
    )


def test_unconfigured_never_calls_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", lambda **_kwargs: pytest.fail("No remote call expected"))
    assert recon.capabilities()["configured"] is False
    with pytest.raises(HTTPException) as raised:
        recon.create_reconstruction([source()], '{"frames":[{}]}')
    assert raised.value.status_code == 503


def test_auth_dependency_protects_reads_and_writes() -> None:
    app = FastAPI()
    app.include_router(recon.router, prefix="/api/v1")

    def refuse() -> None:
        raise HTTPException(401, "Access token required")

    app.dependency_overrides[require_write_token] = refuse
    with TestClient(app) as client:
        assert client.get("/api/v1/worlds/reconstructions/capabilities").status_code == 401
        assert client.get("/api/v1/worlds/reconstructions/job_1").status_code == 401
        assert client.delete("/api/v1/worlds/reconstructions/job_1").status_code == 401
        assert (
            client.post(
                "/api/v1/worlds/reconstructions", files={"files": ("image.png", b"x", "image/png")}
            ).status_code
            == 401
        )


@pytest.mark.parametrize("authorization", [None, "Bearer wrong-token"])
def test_auth_rejects_upload_before_spooling(
    monkeypatch: pytest.MonkeyPatch, authorization: str | None
) -> None:
    app = FastAPI()
    app.state.settings = Settings(api_write_token="private-api-token")
    app.include_router(recon.router)
    app.add_exception_handler(
        UnauthorizedError,
        lambda request, error: JSONResponse(status_code=401, content={"detail": "Unauthorized"}),
    )
    monkeypatch.setattr(
        formparsers,
        "SpooledTemporaryFile",
        lambda *args, **kwargs: pytest.fail("Unauthorized multipart must not be parsed"),
    )
    with TestClient(app) as client:
        response = client.post(
            "/worlds/reconstructions",
            headers={"Authorization": authorization} if authorization else {},
            files={"files": ("large.png", b"private-media", "image/png")},
        )
    assert response.status_code == 401


def test_only_minimum_media_metadata_reaches_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_RECONSTRUCTION_GATEWAY_URL", "https://worker.example")
    monkeypatch.setenv("WORLD_RECONSTRUCTION_GATEWAY_TOKEN", "private-worker-token")
    captured: dict[str, Any] = {}
    original_client = httpx.Client

    def handle(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode(errors="replace")
        captured["auth"] = request.headers["authorization"]
        captured["url"] = str(request.url)
        return httpx.Response(202, json={"id": "job_1", "status": "queued", "artifacts": []})

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    job = recon.create_reconstruction(
        [source()],
        json.dumps(
            {
                "frames": [{"timestampMs": 100, "email": "alice@example.test"}],
                "prompt": "private prompt",
                "projectId": "private-project",
            }
        ),
    )
    assert job.status == "queued"
    assert captured["url"] == "https://worker.example/reconstructions"
    assert captured["auth"] == "Bearer private-worker-token"
    assert "source-00000.png" in captured["body"]
    assert '"timestampMs": 100' in captured["body"]
    for private in ["alice", "private.png", "private prompt", "private-project"]:
        assert private not in captured["body"]


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example",
        "https://user:secret@worker.example",
        "https://worker.example?token=secret",
    ],
)
def test_invalid_operator_url_refused(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("WORLD_RECONSTRUCTION_GATEWAY_URL", url)
    assert recon.capabilities()["configured"] is False


@pytest.mark.parametrize("token", [None, "", "   "])
def test_remote_reconstruction_requires_authentication(
    monkeypatch: pytest.MonkeyPatch, token: str | None
) -> None:
    monkeypatch.setenv("WORLD_RECONSTRUCTION_GATEWAY_URL", "https://worker.example")
    if token is not None:
        monkeypatch.setenv("WORLD_RECONSTRUCTION_GATEWAY_TOKEN", token)
    assert recon.capabilities()["configured"] is False
    with pytest.raises(HTTPException) as raised:
        recon.connection()
    assert raised.value.status_code == 503
    assert "WORLD_RECONSTRUCTION_GATEWAY_TOKEN" in raised.value.detail


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:9000",
        "http://127.0.0.1:9000",
        "http://[::1]:9000",
        "https://localhost:9000",
    ],
)
def test_loopback_reconstruction_can_run_without_worker_token(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setenv("WORLD_RECONSTRUCTION_GATEWAY_URL", url)
    assert recon.connection() == (url, None)
    assert recon.capabilities()["configured"] is True


def test_operator_env_file_and_environment_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "worlds.env"
    config.write_text(
        "WORLD_RECONSTRUCTION_GATEWAY_URL=https://worker.example\n"
        "WORLD_RECONSTRUCTION_GATEWAY_TOKEN=file-token\n"
    )
    monkeypatch.setenv("WORLD_ENV_FILE", str(config))
    assert recon.connection() == ("https://worker.example", "file-token")
    monkeypatch.setenv("WORLD_RECONSTRUCTION_GATEWAY_TOKEN", "environment-token")
    assert recon.connection() == ("https://worker.example", "environment-token")


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///tmp/scene.ply",
        "https://user:secret@files.example/scene.ply",
        "http://remote.example/scene.ply",
    ],
)
def test_unsafe_artifact_urls_refused(url: str) -> None:
    with pytest.raises(HTTPException) as raised:
        recon.normalized_job(
            {"id": "job_1", "status": "completed", "artifacts": [{"format": "ply", "url": url}]},
            None,
        )
    assert raised.value.status_code == 502


def test_worker_secrets_and_raw_failures_never_returned() -> None:
    with pytest.raises(HTTPException):
        recon.normalized_job(
            {
                "id": "job_1",
                "status": "completed",
                "artifacts": [
                    {"format": "ply", "url": "https://files.example/scene.ply?token=secret"}
                ],
            },
            "secret",
        )
    failed = recon.normalized_job(
        {"id": "job_1", "status": "failed", "error": "Traceback /home/alice secret"}, None
    )
    assert "alice" not in (failed.error or "")


def test_size_limit_and_metadata_are_checked_before_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_RECONSTRUCTION_GATEWAY_URL", "http://localhost:8001")
    monkeypatch.setattr(recon, "MAX_UPLOAD_BYTES", 4)
    monkeypatch.setattr(httpx, "Client", lambda **_kwargs: pytest.fail("No remote call expected"))
    with pytest.raises(HTTPException) as raised:
        recon.create_reconstruction([source(content=b"12345")], '{"frames":[{}]}')
    assert raised.value.status_code == 413
    with pytest.raises(HTTPException) as raised:
        recon.create_reconstruction([source()], '{"frames":[]}')
    assert raised.value.status_code == 422


def test_chunked_upload_limit_closes_spooled_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recon, "MAX_MULTIPART_BYTES", 512)
    app = FastAPI()
    app.include_router(recon.router)
    opened = []
    original = SpooledTemporaryFile

    def spool(*args: Any, **kwargs: Any) -> Any:
        file = original(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", spool)

    async def chunks() -> AsyncIterator[bytes]:
        yield (
            b'--test\r\nContent-Disposition: form-data; name="files"; filename="image.png"\r\n'
            b"Content-Type: image/png\r\n\r\nfirst"
        )
        yield b"x" * 513

    async def send() -> Any:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/worlds/reconstructions",
                headers={"content-type": "multipart/form-data; boundary=test"},
                content=chunks(),
            )

    response = asyncio.run(send())
    assert response.status_code == 413
    assert opened, "The test must open a temporary file before crossing the streaming limit"
    assert all(file.closed for file in opened)


def test_declared_oversized_upload_refused_before_spooling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recon, "MAX_MULTIPART_BYTES", 10)
    monkeypatch.setattr(
        formparsers,
        "SpooledTemporaryFile",
        lambda *args, **kwargs: pytest.fail("Must reject before parsing"),
    )
    app = FastAPI()
    app.include_router(recon.router)
    with TestClient(app) as client:
        response = client.post(
            "/worlds/reconstructions",
            content=b"x" * 20,
            headers={"content-type": "multipart/form-data; boundary=test"},
        )
    assert response.status_code == 413


def test_image_exif_removed_and_orientation_applied() -> None:
    image = Image.new("RGB", (2, 3), "red")
    exif = Image.Exif()
    exif[274] = 6
    exif[315] = "private-alice"
    original = io.BytesIO()
    image.save(original, format="JPEG", exif=exif)
    clean = recon.image_without_metadata(original.getvalue(), "image/jpeg")
    assert b"private-alice" not in clean
    with Image.open(io.BytesIO(clean)) as decoded:
        assert decoded.size == (3, 2)
        assert not decoded.getexif()
        assert not decoded.info


def test_invalid_image_and_stage_metadata_are_not_forwarded() -> None:
    with pytest.raises(HTTPException) as raised:
        recon.image_without_metadata(b"not an image", "image/png")
    assert raised.value.status_code == 422
    result = recon.normalized_job(
        {"id": "job_1", "status": "running", "stage": "Reading /home/alice/private-image.png"}, None
    )
    assert result.stage is None


def test_optional_geometry_reports_use_safe_scoped_download_urls() -> None:
    payload = {
        "id": "job_1",
        "status": "completed",
        "diagnosticsUrl": "https://worker.example/report?signature=abc",
        "camerasUrl": "http://localhost:8790/cameras?signature=xyz",
    }
    job = recon.normalized_job(payload, "private-token")
    assert job.diagnostics_url == payload["diagnosticsUrl"]
    assert job.cameras_url == payload["camerasUrl"]
    for unsafe in (
        "file:///etc/passwd",
        "https://user:password@worker.example/report",
        "https://worker.example/report?token=private-token",
    ):
        with pytest.raises(HTTPException) as raised:
            recon.normalized_job({**payload, "diagnosticsUrl": unsafe}, "private-token")
        assert raised.value.status_code == 502


@pytest.mark.parametrize(
    "response", [httpx.Response(204), httpx.Response(200, json={"deleted": True})]
)
def test_delete_requires_and_returns_worker_confirmation(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response
) -> None:
    monkeypatch.setenv("WORLD_RECONSTRUCTION_GATEWAY_URL", "http://localhost:9000")
    called = []

    def upstream(method: str, url: str, **kwargs: Any) -> Any:
        called.append((method, url))
        return response

    monkeypatch.setattr(recon, "request", upstream)
    assert recon.delete_reconstruction("job_1") == {"deleted": True}
    assert called == [("DELETE", "http://localhost:9000/reconstructions/job_1")]


def test_delete_does_not_claim_success_without_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_RECONSTRUCTION_GATEWAY_URL", "http://localhost:9000")
    monkeypatch.setattr(
        recon, "request", lambda *a, **k: httpx.Response(202, json={"status": "deleting"})
    )
    with pytest.raises(HTTPException) as raised:
        recon.delete_reconstruction("job_1")
    assert raised.value.status_code == 502
    assert "may still exist" in raised.value.detail
