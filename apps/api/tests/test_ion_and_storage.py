from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.services.ion import IonClient, IonError
from app.services.urls import UrlValidationError, validate_dataset_url
from app.storage import NullStorage, StorageUnavailableError, build_storage


def test_ion_status_reports_capabilities(client: TestClient) -> None:
    body = client.get("/api/v1/ion/status").json()
    assert body["reconstruction"]["createJobs"] is False
    assert body["reconstruction"]["registerAssets"] is True
    assert "sourceType" in body["reconstruction"]["createJobsReason"]


def test_ion_asset_requires_token(client: TestClient) -> None:
    response = client.get("/api/v1/ion/assets/1")
    assert response.status_code in {503, 200}


def test_ion_client_parses_asset() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        assert request.url.path == "/v1/assets/42"
        return httpx.Response(
            200,
            json={
                "id": 42,
                "name": "Job",
                "type": "3DTILES",
                "status": "IN_PROGRESS",
                "percentComplete": 40,
            },
        )

    settings = Settings(CESIUM_ION_SERVER_TOKEN="secret", _env_file=None)
    ion = IonClient(settings, transport=httpx.MockTransport(handler))
    asset = ion.get_asset(42)
    assert asset.status == "IN_PROGRESS" and asset.percent_complete == 40
    assert ion.capabilities().monitor_jobs is True


def test_ion_client_maps_errors() -> None:
    settings = Settings(CESIUM_ION_SERVER_TOKEN="secret", _env_file=None)
    ion = IonClient(settings, transport=httpx.MockTransport(lambda _: httpx.Response(404)))
    with pytest.raises(IonError) as excinfo:
        ion.get_asset(1)
    assert excinfo.value.status_code == 404
    unconfigured = IonClient(Settings(_env_file=None))
    with pytest.raises(IonError) as excinfo:
        unconfigured.get_asset(1)
    assert excinfo.value.status_code == 503


def test_storage_factory() -> None:
    assert isinstance(build_storage(Settings(_env_file=None)), NullStorage)
    with pytest.raises(StorageUnavailableError):
        NullStorage().put_object("k", b"x", "text/plain")
    s3 = build_storage(
        Settings(
            OBJECT_STORAGE_ENDPOINT_URL="http://localhost:9000",
            OBJECT_STORAGE_BUCKET="b",
            OBJECT_STORAGE_ACCESS_KEY="a",
            OBJECT_STORAGE_SECRET_KEY="s",
            OBJECT_STORAGE_PUBLIC_URL="http://cdn.example.com/b",
            _env_file=None,
        )
    )
    assert s3.name == "s3" and s3.public_url("x/y.png") == "http://cdn.example.com/b/x/y.png"


def test_thumbnail_upload_without_storage_is_503(client: TestClient) -> None:
    from app.storage import NullStorage, get_storage
    from tests.conftest import site_payload

    # Assert the unconfigured case explicitly rather than relying on OBJECT_STORAGE_* being
    # absent from the ambient environment: a developer who followed the README has a .env
    # that configures storage, and this test would then fail on a connection error instead.
    client.app.dependency_overrides[get_storage] = NullStorage  # type: ignore[attr-defined]

    site = client.post("/api/v1/sites", json=site_payload()).json()
    response = client.post(
        f"/api/v1/sites/{site['id']}/thumbnail", files={"file": ("t.png", b"\x89PNG", "image/png")}
    )
    assert response.status_code == 503
    bad_type = client.post(
        f"/api/v1/sites/{site['id']}/thumbnail", files={"file": ("t.txt", b"hi", "text/plain")}
    )
    assert bad_type.status_code == 415
    client.app.dependency_overrides.pop(get_storage, None)  # type: ignore[attr-defined]


def test_url_validation() -> None:
    assert validate_dataset_url("https://example.com/tileset.json")
    for bad in ("ftp://x.com/a", "https://user:pw@x.com/a", "https:///nohost"):
        with pytest.raises(UrlValidationError):
            validate_dataset_url(bad)


def test_health(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok" and body["database"] is True
    assert body["postgisVersion"]
