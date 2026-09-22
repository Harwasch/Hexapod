"""The phone-handoff token: what it can do, and the much longer list of what it cannot.

Every assertion here is about *this API's* authorisation, which is the thing that has to
hold. Nothing here says anything about object storage: moto served a PUT with no
signature at all in A0, so a test that pointed at it and watched a part land would prove
nothing about credentials. tests/test_storage_minio.py covers presigned-URL auth against
a real server; what is covered here is the decision the API makes before it ever hands a
presigned URL out.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import Any

import boto3
import pytest
import requests
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy.orm import Session

from app.api.deps import HANDOFF_RENEWAL_HEADER, _db
from app.config import Settings
from app.main import create_app
from app.services import captures as capture_service
from app.services import handoff
from app.services.errors import UnauthorizedError
from app.storage import S3Storage, get_storage

TOKEN = "correct-horse-battery-staple"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
BUCKET = "twin-handoff-test"
REGION = "us-east-1"
PART_SIZE = capture_service.PART_SIZE_BYTES
BODY = b"\x07" * (PART_SIZE + 512)

SETTINGS = Settings(api_write_token=TOKEN)
KEY = handoff.key_for(SETTINGS)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def storage() -> Iterator[S3Storage]:
    with mock_aws():
        boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
        yield S3Storage(
            bucket=BUCKET,
            endpoint_url=None,
            access_key="key",
            secret_key="secret",
            region=REGION,
            public_base_url="https://cdn.example.com/twin-handoff-test",
        )


@pytest.fixture
def client(db: Session, storage: S3Storage) -> Iterator[TestClient]:
    """An app with a real write token configured, so the gate is actually closed."""
    app = create_app(SETTINGS)

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    app.dependency_overrides[get_storage] = lambda: storage
    with TestClient(app) as test_client:
        yield test_client


def make_capture(client: TestClient, name: str = "Backyard maple") -> str:
    response = client.post("/api/v1/captures", json={"name": name, "kind": "video"}, headers=AUTH)
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def mint(client: TestClient, capture_id: str) -> dict[str, Any]:
    response = client.post(f"/api/v1/captures/{capture_id}/handoff", headers=AUTH)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


# --- minting -----------------------------------------------------------------


def test_minting_a_handoff_needs_the_write_token(client: TestClient) -> None:
    capture_id = make_capture(client)
    assert client.post(f"/api/v1/captures/{capture_id}/handoff").status_code == 401


def test_a_handoff_for_a_capture_that_does_not_exist_is_a_404(client: TestClient) -> None:
    missing = uuid.uuid4()
    assert client.post(f"/api/v1/captures/{missing}/handoff", headers=AUTH).status_code == 404


def test_the_handoff_carries_a_url_a_qr_code_and_a_deadline(client: TestClient) -> None:
    capture_id = make_capture(client)
    body = mint(client, capture_id)

    assert body["captureId"] == capture_id
    # The token rides in the fragment: never sent to a server, never in an access log.
    assert body["url"].endswith(f"/upload.html#{body['token']}")
    assert "#" in body["url"]
    assert body["expiresIn"] == handoff.TOKEN_TTL_SECONDS
    # Rendered server-side, so the console gains no QR dependency.
    assert body["qrSvg"].startswith("<svg")
    assert body["qrSvg"].endswith("</svg>")


def test_the_token_names_its_capture_and_nothing_secret(client: TestClient) -> None:
    """The payload is readable — that is deliberate, so the phone page knows which
    capture it is uploading to without a second round trip. What it is not is
    *forgeable*: the signature is over every field."""
    capture_id = make_capture(client)
    token = mint(client, capture_id)["token"]
    version, capture_hex, _expires, _ceiling, _signature = token.split(".")
    assert version == handoff.FORMAT_VERSION
    assert uuid.UUID(hex=capture_hex) == uuid.UUID(capture_id)
    assert TOKEN not in token  # the write token is not recoverable from it


# --- what it can do ----------------------------------------------------------


def test_a_handoff_token_uploads_to_its_own_capture(client: TestClient) -> None:
    """The whole point, end to end: register, presign, PUT, complete — no write token."""
    capture_id = make_capture(client)
    phone = bearer(mint(client, capture_id)["token"])

    registered = client.post(
        f"/api/v1/captures/{capture_id}/files",
        json={"filename": "IMG_0001.MOV", "bytes": len(BODY), "contentType": "video/quicktime"},
        headers=phone,
    )
    assert registered.status_code == 201, registered.text
    payload = registered.json()
    file_id = payload["file"]["id"]
    window = payload["upload"]

    parts = []
    for part in window["parts"]:
        start = (part["partNumber"] - 1) * window["partSize"]
        chunk = BODY[start : start + window["partSize"]]
        put = requests.put(part["url"], data=chunk, timeout=10)
        assert put.status_code == 200
        parts.append({"partNumber": part["partNumber"], "etag": put.headers["ETag"].strip('"')})

    if window["nextPartNumber"] is not None:
        more = client.post(
            f"/api/v1/captures/{capture_id}/files/{file_id}/parts",
            json={"firstPartNumber": window["nextPartNumber"]},
            headers=phone,
        )
        assert more.status_code == 200, more.text

    completed = client.post(
        f"/api/v1/captures/{capture_id}/files/{file_id}/complete",
        json={"parts": parts},
        headers=phone,
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["status"] == "complete"

    # And the capture is now ready to process — which the phone still cannot start.
    assert client.get(f"/api/v1/captures/{capture_id}").json()["status"] == "not-started"


def test_an_authorised_upload_response_carries_the_next_token(client: TestClient) -> None:
    """A 12 GB upload outlives a ten-minute token, so each response renews it."""
    capture_id = make_capture(client)
    first = mint(client, capture_id)["token"]

    registered = client.post(
        f"/api/v1/captures/{capture_id}/files",
        json={"filename": "a.ply", "bytes": 10},
        headers=bearer(first),
    )
    assert registered.status_code == 201
    renewed = registered.headers[HANDOFF_RENEWAL_HEADER]
    # Deterministic in (capture, deadlines), so a renewal inside the same second is
    # byte-identical; what matters is that its deadline has moved forward with the clock.
    now = int(time.time())
    assert handoff.verify(KEY, renewed, capture_id=uuid.UUID(capture_id), now=now).expires_at >= (
        handoff.verify(KEY, first, capture_id=uuid.UUID(capture_id), now=now).expires_at
    )
    assert (
        handoff.renew(
            KEY,
            handoff.verify(KEY, first, capture_id=uuid.UUID(capture_id), now=now),
            now=now + 300,
        )
        != first
    )

    # The renewed token works, and on this capture only.
    file_id = registered.json()["file"]["id"]
    again = client.post(
        f"/api/v1/captures/{capture_id}/files/{file_id}/parts",
        json={"firstPartNumber": 1},
        headers=bearer(renewed),
    )
    assert again.status_code == 200


def test_renewal_cannot_outrun_the_ceiling() -> None:
    """The sliding window slides, but not forever: the ceiling is signed into the token
    and copied unchanged into every renewal."""
    capture_id = uuid.uuid4()
    issued_at = 1_800_000_000
    token = handoff.issue(KEY, capture_id, now=issued_at)

    # Walk the chain in ten-minute steps for a day. It must die at the ceiling.
    now = issued_at
    last_expiry = 0
    for _ in range(200):
        now += handoff.TOKEN_TTL_SECONDS - 1
        try:
            claims = handoff.verify(KEY, token, capture_id=capture_id, now=now)
        except UnauthorizedError:  # any refusal ends the chain, which is the point
            break
        last_expiry = claims.expires_at
        nxt = handoff.renew(KEY, claims, now=now)
        if nxt is None:
            break
        token = nxt
    assert now < issued_at + handoff.RENEWABLE_FOR_SECONDS + handoff.TOKEN_TTL_SECONDS
    assert last_expiry <= issued_at + handoff.RENEWABLE_FOR_SECONDS


# --- what it cannot do -------------------------------------------------------


def test_the_same_token_is_refused_against_a_different_capture(client: TestClient) -> None:
    """The scope check, and the reason the capture id is inside the signature."""
    mine = make_capture(client, "Mine")
    theirs = make_capture(client, "Theirs")
    phone = bearer(mint(client, mine)["token"])

    # It works on its own capture...
    assert (
        client.post(
            f"/api/v1/captures/{mine}/files", json={"filename": "a.ply"}, headers=phone
        ).status_code
        == 201
    )
    # ...and is refused on the neighbouring one, which exists and is uploadable.
    refused = client.post(
        f"/api/v1/captures/{theirs}/files", json={"filename": "a.ply"}, headers=phone
    )
    assert refused.status_code == 401
    assert refused.headers["WWW-Authenticate"] == "Bearer"
    # Nothing was created on the capture it was pointed at.
    assert client.get(f"/api/v1/captures/{theirs}").json()["files"] == []


def test_an_expired_token_is_refused(client: TestClient) -> None:
    capture_id = make_capture(client)
    stale = handoff.issue(
        KEY, uuid.UUID(capture_id), now=int(time.time()) - handoff.TOKEN_TTL_SECONDS - 1
    )
    response = client.post(
        f"/api/v1/captures/{capture_id}/files", json={"filename": "a.ply"}, headers=bearer(stale)
    )
    assert response.status_code == 401


def test_a_token_past_its_ceiling_is_refused(client: TestClient) -> None:
    """Even with a long TTL, the ceiling ends it."""
    capture_id = make_capture(client)
    now = int(time.time())
    token = handoff.issue(
        KEY,
        uuid.UUID(capture_id),
        now=now,
        ttl=handoff.RENEWABLE_FOR_SECONDS * 10,
        renewable_until=now - 1,
    )
    response = client.post(
        f"/api/v1/captures/{capture_id}/files", json={"filename": "a.ply"}, headers=bearer(token)
    )
    assert response.status_code == 401


def tampered(token: str) -> list[str]:
    """Every field of a token, edited in the way an attacker would want to edit it."""
    version, capture_hex, expires, ceiling, signature = token.split(".")
    other = uuid.uuid4().hex
    far = str(int(expires) + 10 * 365 * 24 * 3600)
    flipped = ("b" if signature[0] != "b" else "c") + signature[1:]
    return [
        f"{version}.{other}.{expires}.{ceiling}.{signature}",  # a different capture
        f"{version}.{capture_hex}.{far}.{ceiling}.{signature}",  # never expire
        f"{version}.{capture_hex}.{expires}.{far}.{signature}",  # renew forever
        f"{version}.{capture_hex}.{expires}.{ceiling}.{flipped}",  # forge the signature
        f"h2.{capture_hex}.{expires}.{ceiling}.{signature}",  # a format that does not exist
        f"{version}.{capture_hex}.{expires}.{ceiling}",  # drop the signature
        f"{version}.{capture_hex}.{expires}.{ceiling}.",  # empty signature
        token[:-1],  # truncate it
        "",  # nothing at all
    ]


def test_a_tampered_token_is_refused(client: TestClient) -> None:
    capture_id = make_capture(client)
    token = mint(client, capture_id)["token"]
    for candidate in tampered(token):
        response = client.post(
            f"/api/v1/captures/{capture_id}/files",
            json={"filename": "a.ply"},
            headers=bearer(candidate),
        )
        assert response.status_code == 401, f"accepted a tampered token: {candidate!r}"
    assert client.get(f"/api/v1/captures/{capture_id}").json()["files"] == []


def test_a_token_signed_with_another_key_is_refused(client: TestClient) -> None:
    """Rotating `API_HANDOFF_SECRET` invalidates outstanding links, as it should."""
    capture_id = make_capture(client)
    foreign = handoff.issue(
        handoff.derive_key("some other deployment's secret"),
        uuid.UUID(capture_id),
        now=int(time.time()),
    )
    response = client.post(
        f"/api/v1/captures/{capture_id}/files", json={"filename": "a.ply"}, headers=bearer(foreign)
    )
    assert response.status_code == 401


NON_UPLOAD_WRITES: list[tuple[str, str, dict[str, Any] | None]] = [
    ("post", "/api/v1/captures", {"name": "smuggled", "kind": "video"}),
    ("post", "/api/v1/captures/{capture}/process", {"recipe": "splat-ingest"}),
    ("post", "/api/v1/captures/{capture}/handoff", None),
    ("post", "/api/v1/jobs/{uuid}/cancel", None),
    ("post", "/api/v1/sites", {"name": "smuggled"}),
    ("patch", "/api/v1/sites/{uuid}", {}),
    ("delete", "/api/v1/sites/{uuid}", None),
    ("post", "/api/v1/layers", {"name": "x"}),
    ("delete", "/api/v1/layers/{uuid}", None),
    ("post", "/api/v1/assets", {"name": "x"}),
    ("delete", "/api/v1/assets/{uuid}", None),
    ("post", "/api/v1/plans", {"projectId": "p"}),
    ("delete", "/api/v1/plans/{uuid}", None),
    ("post", "/api/v1/agent/plan-draft", {"goal": "x", "projectId": "p"}),
]


@pytest.mark.parametrize(("method", "template", "body"), NON_UPLOAD_WRITES)
def test_a_handoff_token_reaches_no_other_mutating_endpoint(
    client: TestClient, method: str, template: str, body: dict[str, Any] | None
) -> None:
    """Including, pointedly, `process` and `handoff` on its *own* capture: the token
    uploads, and that is the entire verb list."""
    capture_id = make_capture(client)
    phone = bearer(mint(client, capture_id)["token"])
    url = template.replace("{capture}", capture_id).replace(
        "{uuid}", "00000000-0000-0000-0000-000000000000"
    )
    response = client.request(method, url, json=body, headers=phone)
    assert response.status_code == 401, f"{method.upper()} {url} -> {response.status_code}"
    # And it created nothing on the way past.
    assert client.get("/api/v1/captures").json()[0]["id"] == capture_id
    assert len(client.get("/api/v1/captures").json()) == 1


def test_the_write_token_still_works_on_the_upload_endpoints(client: TestClient) -> None:
    """The console does not hold a handoff token, and must not need one."""
    capture_id = make_capture(client)
    response = client.post(
        f"/api/v1/captures/{capture_id}/files", json={"filename": "a.ply"}, headers=AUTH
    )
    assert response.status_code == 201
    # A write-token upload is not a handoff, so there is nothing to renew.
    assert HANDOFF_RENEWAL_HEADER not in response.headers


def test_an_upload_endpoint_with_no_credential_is_refused(client: TestClient) -> None:
    capture_id = make_capture(client)
    assert (
        client.post(f"/api/v1/captures/{capture_id}/files", json={"filename": "a.ply"}).status_code
        == 401
    )


# --- the unconfigured deployment ---------------------------------------------


def test_with_no_write_token_the_upload_endpoints_stay_open(db: Session) -> None:
    """A fresh checkout has no token and writes are open; the phone page must work there
    too, or local development needs a secret to test a file picker."""
    app = create_app(Settings(api_write_token=None))

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    with TestClient(app) as open_client:
        created = open_client.post("/api/v1/captures", json={"name": "x", "kind": "video"})
        assert created.status_code == 201
        handoff_response = open_client.post(f"/api/v1/captures/{created.json()['id']}/handoff")
        assert handoff_response.status_code == 201
        assert handoff_response.json()["token"]
