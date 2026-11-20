"""Phone-key authority: exhaust the published surface, then exercise owned resources."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from typing import Any

import boto3
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.api.v1.ask import reasoner
from app.config import Settings
from app.main import create_app
from app.models import Capture
from app.schemas.ask import SpeechEvent
from app.services import phone_key
from app.storage import S3Storage, get_storage
from tests.conftest import site_payload
from tests.test_phone import uploaded
from tests.test_quality_api import _finish

KEY = "app-phone-key"
WRITE = "test-write-token"
PHONE = {"Authorization": f"Bearer {KEY}"}
ADMIN = {"Authorization": f"Bearer {WRITE}"}
BUNDLE = {"name": "Phone bundle", "kind": "video"}
FILE = {"filename": "capture.hexcap", "contentType": "application/x-hex-capture", "bytes": 3}
QUESTION = {"id": "q", "question": "What is here?", "keyframes": [], "deviceEntities": []}

# Independent of production's allowlist. Every schema operation is visited, even public
# reads and public POSTs, plus the non-schema alias. Updating this is a scope decision.
ALLOW = {
    ("POST", "/api/v1/phone/check"),
    ("POST", "/api/v1/phone/captures"),
    ("POST", "/api/v1/phone/captures/{capture_id}/process"),
    ("POST", "/api/v1/phone/captures/{capture_id}/refine"),
    ("POST", "/api/v1/phone/captures/{capture_id}/stop"),
    ("PUT", "/api/v1/captures/bundles/{bundle_id}"),
    ("PUT", "/api/v1/captures/{capture_id}/bundle-upload"),
    ("POST", "/api/v1/captures/{capture_id}/files/{file_id}/parts"),
    ("POST", "/api/v1/captures/{capture_id}/files/{file_id}/complete"),
    ("POST", "/api/v1/captures/{capture_id}/process-bundle"),
    ("GET", "/api/v1/captures/{capture_id}"),
    ("POST", "/api/v1/ask"),
    ("POST", "/v1/ask"),
    ("GET", "/api/v1/properties/{site_id}/entities"),
    ("POST", "/api/v1/properties/{site_id}/entities/search"),
}


def settings(**kwargs: Any) -> Settings:
    return Settings(
        _env_file=None,
        api_write_token=WRITE,
        api_phone_key_hash=phone_key.hash_key(KEY, salt=b"test-app-key", iterations=1000),
        # No model key in tests: the scripted reasoner answers without one.
        ask_reasoner="mock",
        api_phone_daily_captures=10,
        **kwargs,
    )


def operations() -> list[tuple[str, str]]:
    spec = create_app(settings()).openapi()
    return [
        (method.upper(), path)
        for path, methods in spec["paths"].items()
        for method in methods
        if method.upper() in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
    ] + [("POST", "/v1/ask")]


@pytest.fixture(scope="module")
def surface_client() -> Iterator[TestClient]:
    app = create_app(settings())

    def no_db() -> None:
        pytest.fail("a refused phone request must not reach the database")

    app.dependency_overrides[_db] = no_db
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize(
    ("method", "template"),
    [op for op in operations() if op[0] in {"POST", "PUT", "PATCH", "DELETE"}],
)
def test_every_route_outside_phone_allowlist_refuses_key(
    surface_client: TestClient, method: str, template: str
) -> None:
    if (method, template) in ALLOW:
        return  # Every allowed row is exercised with real resources below.
    # Reads are not elevated by a phone key (see restrict_phone_routes); every write is refused.
    path = re.sub(r"\{[^}]+\}", str(uuid.uuid4()), template)
    response = surface_client.request(method, path, headers=PHONE, json={})
    assert response.status_code in (401, 403), (
        method,
        template,
        response.status_code,
        response.text,
    )


class Answer:
    async def answer(self, *_: Any) -> Any:
        yield SpeechEvent(text="A saved observation.")


@pytest.fixture
def app_client(db: Session, storage: S3Storage) -> Iterator[TestClient]:
    app = create_app(settings())
    app.dependency_overrides[_db] = lambda: db
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[reasoner] = Answer
    with TestClient(app) as client:
        yield client


def test_every_allowed_route_accepts_phone_key(app_client: TestClient, db: Session) -> None:
    client = app_client
    seen = set()

    def call(method: str, template: str, expected: int, payload: Any = None, **ids: str) -> Any:
        response = client.request(method, template.format(**ids), headers=PHONE, json=payload)
        assert response.status_code == expected, (method, template, response.text)
        seen.add((method, template))
        return response.json() if response.content else None

    call("POST", "/api/v1/phone/check", 204)
    phone = call("POST", "/api/v1/phone/captures", 201, {})["capture"]
    uploaded(db, phone["id"])
    db.commit()
    call(
        "POST",
        "/api/v1/phone/captures/{capture_id}/process",
        202,
        {"recipe": "photo-reconstruct"},
        capture_id=phone["id"],
    )
    call("POST", "/api/v1/phone/captures/{capture_id}/stop", 200, capture_id=phone["id"])
    # Simulate the worker finishing a quality-checked preview before Refine.
    _finish(db, phone["id"])
    call("POST", "/api/v1/phone/captures/{capture_id}/refine", 202, {}, capture_id=phone["id"])

    bundle_id = str(uuid.uuid4())
    capture = call("PUT", "/api/v1/captures/bundles/{bundle_id}", 200, BUNDLE, bundle_id=bundle_id)
    cid = capture["id"]
    assert capture["metadata"]["origin"] == "phone-key"
    call("GET", "/api/v1/captures/{capture_id}", 200, capture_id=cid)
    file = call("PUT", "/api/v1/captures/{capture_id}/bundle-upload", 200, FILE, capture_id=cid)
    call(
        "POST",
        "/api/v1/captures/{capture_id}/files/{file_id}/parts",
        200,
        {"firstPartNumber": 1},
        capture_id=cid,
        file_id=file["id"],
    )
    s3 = boto3.client("s3", region_name="us-east-1")
    # The API's presigned PUT runs against moto storage, not against a live bucket.
    from tests.conftest import WORKER_BUCKET

    etag = s3.upload_part(
        Bucket=WORKER_BUCKET,
        Key=file["storageKey"],
        UploadId=file["uploadId"],
        PartNumber=1,
        Body=b"abc",
    )["ETag"]
    call(
        "POST",
        "/api/v1/captures/{capture_id}/files/{file_id}/complete",
        200,
        {"parts": [{"partNumber": 1, "etag": etag}]},
        capture_id=cid,
        file_id=file["id"],
    )
    call("POST", "/api/v1/captures/{capture_id}/process-bundle", 202, capture_id=cid)

    site = client.post("/api/v1/sites", headers=ADMIN, json=site_payload()).json()
    owned = db.get(Capture, uuid.UUID(cid))
    assert owned is not None
    owned.site_id = uuid.UUID(site["id"])
    db.commit()
    call("GET", "/api/v1/properties/{site_id}/entities", 200, site_id=site["id"])
    call("POST", "/api/v1/properties/{site_id}/entities/search", 200, {}, site_id=site["id"])
    for path in ("/v1/ask", "/api/v1/ask"):
        # SSE responses are asserted separately from the JSON helper.
        response = client.post(path, headers=PHONE, json={**QUESTION, "id": path, "scanId": cid})
        assert response.status_code == 200 and '"type":"done"' in response.text
        seen.add(("POST", path))
        property_response = client.post(
            path,
            headers=PHONE,
            json={
                **QUESTION,
                "id": path + "-property",
                "scope": "property",
                "siteId": site["id"],
            },
        )
        assert property_response.status_code == 200, property_response.text
    assert seen == ALLOW


def test_phone_cannot_reach_other_capture_or_site(app_client: TestClient, db: Session) -> None:
    client = app_client
    bid = str(uuid.uuid4())
    other = client.put(f"/api/v1/captures/bundles/{bid}", headers=ADMIN, json=BUNDLE).json()
    cid = other["id"]
    file = client.put(f"/api/v1/captures/{cid}/bundle-upload", headers=ADMIN, json=FILE).json()
    site = client.post("/api/v1/sites", headers=ADMIN, json=site_payload()).json()
    rows: list[tuple[str, str, Any]] = [
        ("PUT", f"/api/v1/captures/bundles/{bid}", BUNDLE),
        ("GET", f"/api/v1/captures/{cid}", None),
        ("PUT", f"/api/v1/captures/{cid}/bundle-upload", FILE),
        ("POST", f"/api/v1/captures/{cid}/files/{file['id']}/parts", {"firstPartNumber": 1}),
        ("POST", f"/api/v1/captures/{cid}/files/{file['id']}/complete", {"parts": []}),
        ("POST", f"/api/v1/captures/{cid}/process-bundle", None),
        ("GET", f"/api/v1/properties/{site['id']}/entities", None),
        ("POST", f"/api/v1/properties/{site['id']}/entities/search", {}),
    ]
    for method, path, payload in rows:
        response = client.request(method, path, headers=PHONE, json=payload)
        assert response.status_code in (401, 403), (method, path, response.text)
    for path in ("/v1/ask", "/api/v1/ask"):
        for scope in ({"scanId": cid}, {"scope": "property", "siteId": site["id"]}, {}):
            response = client.post(path, headers=PHONE, json={**QUESTION, **scope})
            assert response.status_code in (401, 403), response.text
    # Public read behavior and write-token access remain unchanged.
    assert client.get(f"/api/v1/captures/{cid}").status_code == 200
    assert client.get(f"/api/v1/captures/{cid}", headers=ADMIN).status_code == 200
    assert client.get(f"/api/v1/properties/{site['id']}/entities", headers=ADMIN).status_code == 200


def test_bundle_daily_cap_counts_both_entry_points_and_allows_resume(
    app_client: TestClient,
) -> None:
    client = app_client
    client.app.state.settings.api_phone_daily_captures = 2  # type: ignore[attr-defined]
    bid = str(uuid.uuid4())
    url = f"/api/v1/captures/bundles/{bid}"
    assert client.put(url, headers=PHONE, json=BUNDLE).status_code == 200
    assert client.post("/api/v1/phone/captures", headers=PHONE, json={}).status_code == 201
    assert client.put(url, headers=PHONE, json=BUNDLE).status_code == 200
    assert (
        client.put(
            f"/api/v1/captures/bundles/{uuid.uuid4()}", headers=PHONE, json=BUNDLE
        ).status_code
        == 409
    )
    assert client.post("/api/v1/phone/captures", headers=PHONE, json={}).status_code == 409
    assert (
        client.put(
            f"/api/v1/captures/bundles/{uuid.uuid4()}", headers=ADMIN, json=BUNDLE
        ).status_code
        == 200
    )
