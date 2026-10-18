"""CPU-only service lifecycle fixtures. These do not run or assess reconstruction."""

import asyncio
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette import formparsers

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "worlds_reconstruction_server", ROOT / "server.py"
)
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)
TOKEN = "reconstruction-test-token-that-is-not-a-secret"


def fixture_command(job):
    # Structural files prove lifecycle/download plumbing only. This fixture is not
    # an inference model, and these bytes must never be used as a quality result.
    code = "from pathlib import Path; import sys; p=Path(sys.argv[1])/'output'; p.mkdir(); [(p/n).write_bytes(b'CPU lifecycle test fixture') for n in ['scene.ply','points.ply','diagnostics.json','cameras.json']]"
    return [sys.executable, "-c", code, str(job.directory)]


@pytest.fixture
def engine(tmp_path):
    result = server.Engine(
        tmp_path,
        TOKEN,
        "http://127.0.0.1:8790",
        tmp_path / "source",
        tmp_path / "weights",
        command=fixture_command,
        ready_check=lambda: True,
        retention=60,
    )
    yield result
    result.close()


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(output, format="PNG")
    return output.getvalue()


def wait_complete(engine, identifier):
    limit = time.monotonic() + 5
    while time.monotonic() < limit:
        job = engine.get(identifier)
        if job.status in ("completed", "failed"):
            return job
        time.sleep(0.02)
    pytest.fail("CPU lifecycle fixture did not finish")


def test_auth_precedes_multipart_parsing_and_health_loads_no_model(engine):
    with TestClient(server.create_app(engine)) as client:
        assert client.get("/health").status_code == 401
        assert (
            client.post("/reconstructions", content=b"not multipart").status_code == 401
        )
        response = client.get("/health", headers={"Authorization": "Bearer " + TOKEN})
        assert response.status_code == 200
        assert (
            response.json()["models"][0]["readiness"]
            == "artifacts-verified-gpu-unverified"
        )
        assert not engine.jobs


def test_end_to_end_upload_signed_result_and_confirmed_delete(engine):
    with TestClient(server.create_app(engine, ["http://localhost:5173"])) as client:
        response = client.post(
            "/reconstructions",
            headers={"Authorization": "Bearer " + TOKEN},
            files=[
                ("files", ("private-alice.png", image_bytes(), "image/png")),
                ("files", ("private-bob.png", image_bytes(), "image/png")),
            ],
            data={
                "metadata": json.dumps({"prompt": "private prompt", "frames": [{}, {}]})
            },
        )
        assert response.status_code == 202
        identifier = response.json()["id"]
        job = wait_complete(engine, identifier)
        assert job.status == "completed"
        # The GPU subprocess has terminated; wait for immediate input cleanup.
        job.thread.join(timeout=2)
        assert not (job.directory / "inputs").exists()
        status = client.get(
            f"/reconstructions/{identifier}",
            headers={"Authorization": "Bearer " + TOKEN},
        ).json()
        assert len(status["artifacts"]) == 2
        assert TOKEN not in json.dumps(status)
        url = urlsplit(status["artifacts"][0]["url"])
        download = client.get(
            url.path + "?" + url.query, headers={"Origin": "http://localhost:5173"}
        )
        assert download.status_code == 200
        assert (
            download.headers["access-control-allow-origin"] == "http://localhost:5173"
        )
        assert client.get(url.path + "?expires=1&signature=bad").status_code == 403
        assert (
            client.get(
                url.path, params={"expires": int(time.time()) + 100, "signature": "é"}
            ).status_code
            == 403
        )
        assert client.delete(f"/reconstructions/{identifier}").status_code == 401
        deletion = client.delete(
            f"/reconstructions/{identifier}",
            headers={"Authorization": "Bearer " + TOKEN},
        )
        assert deletion.json() == {"deleted": True}
        assert not job.directory.exists()
        assert client.get(url.path + "?" + url.query).status_code == 404


def test_cancel_running_subprocess_removes_partial_files(engine):
    engine.command = lambda job: [sys.executable, "-c", "import time; time.sleep(30)"]
    job = engine.reserve()
    engine.start(job)
    limit = time.monotonic() + 3
    while job.process is None and time.monotonic() < limit:
        time.sleep(0.01)
    assert job.process is not None
    assert engine.delete(job.id) == {"deleted": True}
    assert job.process.poll() is not None
    assert not job.directory.exists()


def test_queue_and_unconfigured_engine_fail_before_inference(engine):
    first = engine.reserve()
    second = engine.reserve()
    with pytest.raises(server.HTTPException) as error:
        engine.reserve()
    assert error.value.status_code == 429
    engine.delete(first.id)
    engine.delete(second.id)
    engine.ready_check = lambda: False
    with pytest.raises(server.HTTPException) as error:
        engine.reserve()
    assert error.value.status_code == 503


def test_restart_marks_incomplete_jobs_failed_and_removes_sensitive_media(tmp_path):
    identifier = "a" * 32
    directory = tmp_path / identifier
    (directory / "inputs").mkdir(parents=True)
    (directory / "inputs" / "private.png").write_bytes(b"sensitive")
    (directory / "job.json").write_text(
        json.dumps(
            {
                "created": time.time(),
                "status": "running",
                "stage": "Reconstructing scene",
                "progress": 0.2,
            }
        )
    )
    engine = server.Engine(
        tmp_path,
        TOKEN,
        "http://localhost:8790",
        tmp_path / "source",
        tmp_path / "weights",
        ready_check=lambda: False,
    )
    try:
        assert engine.get(identifier).status == "failed"
        assert not (directory / "inputs").exists()
    finally:
        engine.close()


def test_inference_environment_contains_no_provider_or_control_credentials(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "private-provider-token")
    monkeypatch.setenv("WORLD_RECONSTRUCTION_GATEWAY_TOKEN", TOKEN)
    monkeypatch.setenv("API_WRITE_TOKEN", "private-api-token")
    environment = server.child_environment()
    assert not set(environment) & {
        "RUNPOD_API_KEY",
        "WORLD_RECONSTRUCTION_GATEWAY_TOKEN",
        "API_WRITE_TOKEN",
    }
    assert environment["HF_HUB_OFFLINE"] == "1"


def test_hosted_download_url_requires_tls(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        server.Engine(
            tmp_path,
            TOKEN,
            "http://remote.example",
            tmp_path / "source",
            tmp_path / "weights",
        )


def test_chunked_upload_failure_closes_spools_and_removes_reserved_job(
    engine, monkeypatch
):
    monkeypatch.setattr(server, "MAX_WIRE", 512)
    opened = []
    original = formparsers.SpooledTemporaryFile

    def spool(*args, **kwargs):
        file = original(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", spool)

    async def chunks():
        yield b'--test\r\nContent-Disposition: form-data; name="files"; filename="private.png"\r\nContent-Type: image/png\r\n\r\nfirst'
        yield b"x" * 513

    async def send():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.create_app(engine)),
            base_url="http://test",
        ) as client:
            return await client.post(
                "/reconstructions",
                headers={
                    "Authorization": "Bearer " + TOKEN,
                    "Content-Type": "multipart/form-data; boundary=test",
                },
                content=chunks(),
            )

    response = asyncio.run(send())
    assert response.status_code == 413
    assert opened and all(file.closed for file in opened)
    assert not engine.jobs
    assert not list(engine.root.glob("*/inputs"))


def test_failed_engine_never_exposes_partial_geometry(engine):
    engine.command = lambda job: [sys.executable, "-c", "raise SystemExit(1)"]
    job = engine.reserve()
    engine.start(job)
    wait_complete(engine, job.id)
    job.thread.join(timeout=2)
    assert engine.public(job)["status"] == "failed"
    assert engine.public(job)["artifacts"] == []
    assert not (job.directory / "inputs").exists()
