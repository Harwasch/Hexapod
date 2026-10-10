from __future__ import annotations

import hashlib
import io
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session

from app.models.land_image_registration import LandImageRegistrationBlob
from app.models.research import Evidence
from app.models.workspace import PILOT_WORKSPACE_ID
from app.schemas.land_image_registrations import ImageRegistrationCreate
from app.services import land_archive_images as images
from app.services import land_image_registrations as registrations
from app.services.errors import InvalidInputError
from app.services.identity import principal_id
from app.services.land_rasters import TMS
from tests.test_image_registration import request
from tests.test_land_archive_images import add_archive, snapshot
from tests.test_workspaces import ISSUER, IdentityClient, headers
from tests.test_workspaces import identity_client as _identity_client

identity_client = _identity_client


def prepare(
    client: TestClient,
    db: Session,
    auth: dict[str, str] | None = None,
    workspace_id: uuid.UUID = PILOT_WORKSPACE_ID,
) -> tuple[str, str, ImageRegistrationCreate]:
    evidence_id, land_id = add_archive(client, db, auth)
    evidence = db.get(Evidence, evidence_id)
    assert evidence is not None
    evidence.content = {
        **evidence.content,
        "media": {**evidence.content["media"], "kind": "historical-map"},
    }
    result = snapshot()
    images.save(db, workspace_id, evidence_id, result, quota_bytes=100000)
    db.commit()
    return str(evidence_id), land_id, request(6, 4, result.metadata.sha256)


def test_alignment_preview_save_retry_tiles_download_and_cascade(
    client: TestClient, db: Session
) -> None:
    evidence_id, land_id, payload = prepare(client, db)
    endpoint = f"/api/v1/research/evidence/{evidence_id}/image/registrations"
    body = payload.model_dump(mode="json", by_alias=True)
    preview = client.post(endpoint + "/preview", json=body)
    assert preview.status_code == 200, preview.text
    assert preview.json()["controlPointCoverage"] == 1
    saved = client.post(endpoint, json=body)
    assert saved.status_code == 201, saved.text
    data = saved.json()
    assert client.post(endpoint, json=body).json() == data
    assert client.get(endpoint).json() == [data]
    assert client.post(endpoint, json={**body, "name": "Changed"}).status_code == 409
    assert client.post(endpoint, json={**body, "imageSha256": "0" * 64}).status_code == 409
    base = f"/api/v1/research/image-registrations/{data['id']}"
    download = client.get(base + "/download")
    assert hashlib.sha256(download.content).hexdigest() == data["sha256"]
    assert download.headers["Cache-Control"] == "private, no-store"
    assert client.get(base).json() == data
    point = payload.points[-1]
    tile = TMS.tile(point.longitude, point.latitude, 19)
    visible = client.get(f"{base}/tiles/{tile.z}/{tile.x}/{tile.y}.png")
    assert visible.status_code == 200, visible.text
    with Image.open(io.BytesIO(visible.content)) as image:
        assert image.size == (256, 256) and image.getchannel("A").getbbox() is not None
    blank = client.get(base + "/tiles/19/0/0.png")
    with Image.open(io.BytesIO(blank.content)) as image:
        assert image.getchannel("A").getextrema() == (0, 0)
    assert client.get(base + "/tiles/23/0/0.png").status_code == 404
    assert client.delete(f"/api/v1/land/{land_id}").status_code == 204
    db.expire_all()
    assert db.get(LandImageRegistrationBlob, uuid.UUID(data["id"])) is None
    assert client.get(base + "/download").status_code == 404


def test_archive_and_alignment_share_storage_quota(client: TestClient, db: Session) -> None:
    evidence_id, _, payload = prepare(client, db)
    eid = uuid.UUID(evidence_id)
    result = registrations.preview(db, PILOT_WORKSPACE_ID, eid, payload)
    used = images.workspace_used_bytes(db, PILOT_WORKSPACE_ID)
    with pytest.raises(InvalidInputError, match="allowance"):
        registrations.save(
            db, PILOT_WORKSPACE_ID, eid, payload, result, (b"synthetic raster", 6, 4), used
        )
    db.rollback()
    registrations.save(
        db, PILOT_WORKSPACE_ID, eid, payload, result, (b"synthetic raster", 6, 4), used + 100
    )
    db.commit()
    assert images.workspace_used_bytes(db, PILOT_WORKSPACE_ID) == used + len(b"synthetic raster")
    other_id, _ = add_archive(client, db)
    with pytest.raises(InvalidInputError, match="allowance"):
        images.save(db, PILOT_WORKSPACE_ID, other_id, snapshot(), used * 2)
    db.rollback()


def test_map_only_with_pinned_image_hash(client: TestClient, db: Session) -> None:
    evidence_id, _ = add_archive(client, db)
    endpoint = f"/api/v1/research/evidence/{evidence_id}/image/registrations/preview"
    body = request(6, 4).model_dump(mode="json", by_alias=True)
    response = client.post(endpoint, json=body)
    assert response.status_code == 422 and "map-sheet" in response.text
    evidence = db.get(Evidence, evidence_id)
    assert evidence is not None
    evidence.content = {
        **evidence.content,
        "media": {**evidence.content["media"], "kind": "historical-map"},
    }
    db.commit()
    response = client.post(endpoint, json=body)
    assert response.status_code == 422 and "Save the archive" in response.text


def test_alignment_workspace_and_readonly_access(
    identity_client: IdentityClient, db: Session
) -> None:
    client, token = identity_client
    owner = headers(token())
    workspace = client.post(
        "/api/v1/workspaces", headers=owner, json={"name": "Map research"}
    ).json()["id"]
    owner = headers(token(), workspace)
    evidence_id, _, payload = prepare(client, db, owner, uuid.UUID(workspace))
    endpoint = f"/api/v1/research/evidence/{evidence_id}/image/registrations"
    body = payload.model_dump(mode="json", by_alias=True)
    saved = client.post(endpoint, headers=owner, json=body)
    assert saved.status_code == 201, saved.text
    base = f"/api/v1/research/image-registrations/{saved.json()['id']}"
    assert (
        client.put(
            "/api/v1/workspaces/members",
            headers=owner,
            json={"principalId": principal_id(ISSUER, "bob"), "role": "viewer"},
        ).status_code
        == 200
    )
    viewer = headers(token("bob"), workspace)
    assert client.post(endpoint + "/preview", headers=viewer, json=body).status_code == 200
    assert client.post(endpoint, headers=viewer, json=body).status_code == 403
    assert client.get(base + "/download", headers=viewer).status_code == 200
    other = client.post(
        "/api/v1/workspaces", headers=headers(token("bob")), json={"name": "Other"}
    ).json()["id"]
    foreign = headers(token("bob"), other)
    for path in (endpoint, base, base + "/download", base + "/tiles/0/0/0.png"):
        assert client.get(path, headers=foreign).status_code == 404
    assert client.post(endpoint + "/preview", headers=foreign, json=body).status_code == 404
