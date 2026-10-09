from __future__ import annotations

import hashlib
import io
import uuid
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session

from app.analysis.archive_image import MAX_SOURCE_BYTES, normalize
from app.models.land_archive_image import LandArchiveImageBlob
from app.models.research import Evidence
from app.models.workspace import PILOT_WORKSPACE_ID
from app.research.providers.archives import commons
from app.research.providers.base import SourceContext
from app.schemas.land_archives import ArchiveMedia
from app.services import land_archive_images as images
from app.services.errors import InvalidInputError
from app.services.identity import principal_id
from tests.test_land_archives import transport
from tests.test_research import start
from tests.test_terrain import BOUNDARY
from tests.test_workspaces import ISSUER, IdentityClient, headers
from tests.test_workspaces import identity_client as _identity_client

identity_client = _identity_client


def source() -> ArchiveMedia:
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        result = commons(SourceContext(BOUNDARY), client).evidence[0][1].media
    assert result is not None
    return result


def original() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (6, 4), "green").save(output, format="JPEG")
    return output.getvalue()


def snapshot() -> images.ImageSnapshot:
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=original(), headers={"ETag": '"version-1"'})
        )
    ) as client:
        return images.retrieve(source(), client)


def add_archive(
    client: TestClient, db: Session, auth: dict[str, str] | None = None
) -> tuple[uuid.UUID, str]:
    land, _, run, _ = start(client, auth)
    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        evidence = commons(SourceContext(BOUNDARY), http).evidence[0][1]
    row = Evidence(
        run_id=uuid.UUID(run["id"]), source_key="archive", content=evidence.model_dump(mode="json")
    )
    db.add(row)
    db.commit()
    return row.id, land["id"]


def test_bounded_capture_normalizes_orientation_and_preserves_exact_source_bytes() -> None:
    result = snapshot()
    assert result.original == original()
    assert result.metadata.source_sha256 == hashlib.sha256(result.original).hexdigest()
    assert result.metadata.sha256 == hashlib.sha256(result.preview).hexdigest()
    assert result.metadata.source_etag == '"version-1"'
    with Image.open(io.BytesIO(result.preview)) as preview:
        assert preview.format == "PNG" and preview.size == (6, 4)
        assert not preview.getexif()
    output = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (6, 4), "red").save(output, format="JPEG", exif=exif)
    oriented, info = normalize(output.getvalue())
    assert (info["width"], info["height"]) == (4, 6)
    with Image.open(io.BytesIO(oriented)) as image:
        assert not image.getexif()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(302, headers={"Location": "http://127.0.0.1/private"}),
        httpx.Response(200, headers={"Content-Length": str(MAX_SOURCE_BYTES + 1)}),
        httpx.Response(200, content=b"<svg onload='alert(1)'/>"),
    ],
)
def test_capture_rejects_redirects_oversize_and_non_image_content(response: httpx.Response) -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return response

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(InvalidInputError),
    ):
        images.retrieve(source(), client)
    assert len(requests) == 1 and requests[0].startswith("https://upload.wikimedia.org/")


def test_rejects_large_pixel_dimensions_before_decoding() -> None:
    output = io.BytesIO()
    Image.new("1", (4000, 3000)).save(output, format="PNG")
    with pytest.raises(ValueError, match="8-megapixel"):
        normalize(output.getvalue())


def test_saved_image_retry_is_immutable_private_and_cascades_with_land(
    client: TestClient, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_id, land_id = add_archive(client, db)
    endpoint = f"/api/v1/research/evidence/{evidence_id}/image"
    assert client.get(endpoint).json() is None
    result = snapshot()
    calls = []

    def retrieve(*args: Any) -> images.ImageSnapshot:
        calls.append(True)
        return result

    monkeypatch.setattr(images, "retrieve", retrieve)
    first = client.post(endpoint)
    assert first.status_code == 200, first.text
    assert first.json()["sourceSha256"] == hashlib.sha256(result.original).hexdigest()
    assert client.post(endpoint).json() == first.json()
    assert len(calls) == 1
    assert client.get(endpoint).json() == first.json()
    for path, expected in [("preview", result.preview), ("original", result.original)]:
        response = client.get(endpoint + "/" + path)
        assert response.content == expected
        assert response.headers["Cache-Control"] == "private, no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert client.delete(f"/api/v1/land/{land_id}").status_code == 204
    db.expire_all()
    assert db.get(LandArchiveImageBlob, evidence_id) is None
    assert client.get(endpoint + "/original").status_code == 404


def test_workspace_quota_rejects_image_without_partial_blobs(
    client: TestClient, db: Session
) -> None:
    evidence_id, _ = add_archive(client, db)
    with pytest.raises(InvalidInputError, match="allowance"):
        images.save(db, PILOT_WORKSPACE_ID, evidence_id, snapshot(), quota_bytes=1)
    db.rollback()
    assert db.get(LandArchiveImageBlob, evidence_id) is None


def test_image_access_follows_workspace_and_member_roles(
    identity_client: IdentityClient, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, token = identity_client
    owner = headers(token())
    workspace = client.post(
        "/api/v1/workspaces", headers=owner, json={"name": "Archive research"}
    ).json()["id"]
    owner = headers(token(), workspace)
    evidence_id, _ = add_archive(client, db, owner)
    endpoint = f"/api/v1/research/evidence/{evidence_id}/image"
    result = snapshot()
    monkeypatch.setattr(images, "retrieve", lambda *args: result)
    assert client.post(endpoint, headers=owner).status_code == 200
    bob_id = principal_id(ISSUER, "bob")
    assert (
        client.put(
            "/api/v1/workspaces/members",
            headers=owner,
            json={"principalId": bob_id, "role": "viewer"},
        ).status_code
        == 200
    )
    viewer = headers(token("bob"), workspace)
    assert client.get(endpoint + "/preview", headers=viewer).status_code == 200
    assert client.post(endpoint, headers=viewer).status_code == 403
    other = client.post(
        "/api/v1/workspaces", headers=headers(token("bob")), json={"name": "Other"}
    ).json()["id"]
    for suffix in ("", "/preview", "/original"):
        assert (
            client.get(endpoint + suffix, headers=headers(token("bob"), other)).status_code == 404
        )
    assert client.delete(f"/api/v1/workspaces/members/{bob_id}", headers=owner).status_code == 204
    assert client.get(endpoint + "/preview", headers=viewer).status_code == 404
