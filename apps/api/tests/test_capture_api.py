"""The capture upload surface, end to end against moto.

What these prove is the *lifecycle*: that a file registered through the API can be
uploaded to the URLs the API hands out, window by window, and completed, aborted or
resumed. What they deliberately do not prove is authorisation of those URLs -- moto
served a PUT with no signature at all in A0 -- so nothing here asserts that a bad
signature is refused. tests/test_storage_minio.py does that against real MinIO in CI.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import boto3
import pytest
import requests
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.main import create_app
from app.models import Artifact, Job, JobStep
from app.models.enums import CaptureStatus
from app.services import captures as capture_service
from app.services import recipes as recipe_service
from app.storage import S3Storage, get_storage

BUCKET = "twin-test"
REGION = "us-east-1"
PART_SIZE = capture_service.PART_SIZE_BYTES
#: One full part plus a little: the smallest file that is genuinely multipart, since
#: S3 rejects a non-final part under 5 MiB.
BODY = b"\x00" * (PART_SIZE + 1024)


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
            public_base_url="https://cdn.example.com/twin-test",
        )


@pytest.fixture
def client(db: Session, storage: S3Storage) -> Iterator[TestClient]:
    app = create_app()

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    app.dependency_overrides[get_storage] = lambda: storage
    with TestClient(app) as test_client:
        yield test_client


def make_capture(client: TestClient, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": "Backyard maple",
        "kind": "video",
        "device": "iPhone 15 Pro",
        "provenance": {"sourceOrganization": "Tests"},
    }
    payload.update(overrides)
    response = client.post("/api/v1/captures", json=payload)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def register(client: TestClient, capture_id: str, size: int = len(BODY)) -> dict[str, Any]:
    response = client.post(
        f"/api/v1/captures/{capture_id}/files",
        json={"filename": "IMG_0001.MOV", "contentType": "video/quicktime", "bytes": size},
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def put_part(part: dict[str, Any], chunk: bytes) -> dict[str, Any]:
    """Upload one part to the presigned URL and report it the way a browser would."""
    response = requests.put(part["url"], data=chunk, timeout=30)
    assert response.status_code == 200, response.text
    # The browser can only read this header because of ExposeHeaders: ["ETag"] in the
    # bucket CORS rule; without it multipart cannot be completed at all.
    return {"partNumber": part["partNumber"], "etag": response.headers["ETag"]}


def slice_for(number: int, part_size: int, body: bytes = BODY) -> bytes:
    return body[(number - 1) * part_size : number * part_size]


# --- captures ---------------------------------------------------------------


def test_create_capture_derives_a_slug(client: TestClient) -> None:
    capture = make_capture(client, name="Backyard Maple — June")
    assert capture["slug"] == "backyard-maple-june"
    assert capture["status"] == "awaiting-files"
    assert capture["files"] == []
    assert capture["provenance"]["sourceOrganization"] == "Tests"

    again = make_capture(client, name="Backyard Maple — June")
    assert again["slug"] == "backyard-maple-june-2"


def test_explicit_slug_collision_is_a_conflict(client: TestClient) -> None:
    make_capture(client, slug="maple")
    response = client.post(
        "/api/v1/captures", json={"name": "Other", "kind": "video", "slug": "maple"}
    )
    assert response.status_code == 409


def test_list_is_newest_first_and_pages(client: TestClient) -> None:
    names = ["first", "second", "third"]
    for name in names:
        make_capture(client, name=name)

    listed = client.get("/api/v1/captures").json()
    assert [c["name"] for c in listed] == ["third", "second", "first"]

    page = client.get("/api/v1/captures", params={"limit": 1, "offset": 1}).json()
    assert [c["name"] for c in page] == ["second"]
    assert client.get("/api/v1/captures", params={"limit": 0}).status_code == 422


def test_get_capture_carries_files_and_jobs(client: TestClient) -> None:
    capture = make_capture(client)
    upload_file(client, capture["id"])
    queued = client.post(
        f"/api/v1/captures/{capture['id']}/process", json={"recipe": "photo-reconstruct"}
    ).json()

    detail = client.get(f"/api/v1/captures/{capture['id']}").json()
    assert [f["status"] for f in detail["files"]] == ["complete"]
    assert [j["id"] for j in detail["jobs"]] == [queued["id"]]
    assert detail["jobs"][0]["steps"] == []


def test_unknown_capture_is_404(client: TestClient) -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/api/v1/captures/{missing}").status_code == 404
    assert (
        client.post(f"/api/v1/captures/{missing}/files", json={"filename": "a.ply"}).status_code
        == 404
    )


# --- the multipart lifecycle -------------------------------------------------


def upload_file(client: TestClient, capture_id: str) -> dict[str, Any]:
    """Register, upload every part, complete. Returns the completed file row."""
    registered = register(client, capture_id)
    window = registered["upload"]
    parts = [put_part(p, slice_for(p["partNumber"], window["partSize"])) for p in window["parts"]]
    response = client.post(
        f"/api/v1/captures/{capture_id}/files/{registered['file']['id']}/complete",
        json={"parts": parts},
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_multipart_lifecycle_through_the_api(client: TestClient, storage: S3Storage) -> None:
    capture = make_capture(client)
    registered = register(client, capture["id"])
    file, window = registered["file"], registered["upload"]

    assert file["status"] == "in-progress"
    assert file["storageKey"].startswith(f"captures/{capture['id']}/source/{file['id']}/")
    assert file["storageKey"].endswith("IMG_0001.MOV")
    assert window["partSize"] == PART_SIZE
    assert window["partsTotal"] == 2
    assert [p["partNumber"] for p in window["parts"]] == [1, 2]
    assert window["nextPartNumber"] is None  # 2 parts fits inside one window
    assert window["expiresIn"] == 3600
    assert all("X-Amz-Signature=" in p["url"] for p in window["parts"])

    parts = [put_part(p, slice_for(p["partNumber"], window["partSize"])) for p in window["parts"]]
    # Out of order on purpose: the storage layer sorts before it sends.
    completed = client.post(
        f"/api/v1/captures/{capture['id']}/files/{file['id']}/complete",
        json={"parts": list(reversed(parts))},
    ).json()

    assert completed["status"] == "complete"
    assert completed["bytes"] == len(BODY)
    assert completed["partsCompleted"] == 2
    assert completed["uploadId"] is None
    # S3's multipart ETag: a digest of digests, with the part count appended.
    assert completed["checksum"].endswith("-2")
    assert storage.get_object(file["storageKey"]) == BODY

    # The capture is now uploaded but unprocessed.
    assert client.get(f"/api/v1/captures/{capture['id']}").json()["status"] == "not-started"


def test_a_client_supplied_checksum_wins(client: TestClient) -> None:
    capture = make_capture(client)
    registered = register(client, capture["id"])
    window = registered["upload"]
    parts = [put_part(p, slice_for(p["partNumber"], window["partSize"])) for p in window["parts"]]
    completed = client.post(
        f"/api/v1/captures/{capture['id']}/files/{registered['file']['id']}/complete",
        json={"parts": parts, "checksum": "sha256:abc"},
    ).json()
    assert completed["checksum"] == "sha256:abc"


def test_completing_twice_is_a_conflict(client: TestClient) -> None:
    capture = make_capture(client)
    file = upload_file(client, capture["id"])
    response = client.post(
        f"/api/v1/captures/{capture['id']}/files/{file['id']}/complete",
        json={"parts": [{"partNumber": 1, "etag": "whatever"}]},
    )
    assert response.status_code == 409


# --- windows and resume ------------------------------------------------------


def test_presigning_is_windowed_and_resumable(
    client: TestClient, storage: S3Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A window of one, so a two-part upload takes two presign calls.

    This is the 1536-part case in miniature: the first response carries a bounded run
    of URLs and says where the next window starts, and the client comes back for it --
    which is the same call it makes to resume after being interrupted.
    """
    monkeypatch.setattr(capture_service, "PRESIGN_WINDOW_PARTS", 1)
    capture = make_capture(client)
    registered = register(client, capture["id"])
    file = registered["file"]
    first = registered["upload"]

    assert [p["partNumber"] for p in first["parts"]] == [1]
    assert first["nextPartNumber"] == 2

    parts = [put_part(first["parts"][0], slice_for(1, first["partSize"]))]

    # Resume: ask for the window starting at the first part not yet confirmed.
    second = client.post(
        f"/api/v1/captures/{capture['id']}/files/{file['id']}/parts",
        json={"firstPartNumber": 2, "count": 64},  # count is clamped to the window
    )
    assert second.status_code == 200, second.text
    window = second.json()
    assert [p["partNumber"] for p in window["parts"]] == [2]
    assert window["nextPartNumber"] is None
    assert window["uploadId"] == first["uploadId"]

    # Asking for a later window is the only progress signal the API gets.
    detail = client.get(f"/api/v1/captures/{capture['id']}").json()
    assert detail["files"][0]["partsCompleted"] == 1

    parts.append(put_part(window["parts"][0], slice_for(2, window["partSize"])))
    completed = client.post(
        f"/api/v1/captures/{capture['id']}/files/{file['id']}/complete", json={"parts": parts}
    ).json()
    assert completed["status"] == "complete"
    assert storage.get_object(file["storageKey"]) == BODY


def test_re_presigning_a_window_does_not_rewind_progress(client: TestClient) -> None:
    capture = make_capture(client)
    file = register(client, capture["id"])["file"]
    base = f"/api/v1/captures/{capture['id']}/files/{file['id']}/parts"
    client.post(base, json={"firstPartNumber": 2})
    client.post(base, json={"firstPartNumber": 1})  # a retry of an earlier part
    detail = client.get(f"/api/v1/captures/{capture['id']}").json()
    assert detail["files"][0]["partsCompleted"] == 1


def test_a_window_past_the_end_is_rejected(client: TestClient) -> None:
    capture = make_capture(client)
    file = register(client, capture["id"])["file"]
    response = client.post(
        f"/api/v1/captures/{capture['id']}/files/{file['id']}/parts",
        json={"firstPartNumber": 3},  # the file is 2 parts long
    )
    assert response.status_code == 422


def test_an_undeclared_size_windows_forever(client: TestClient) -> None:
    """No `bytes` means no part count: the client uploads until it runs out."""
    capture = make_capture(client)
    response = client.post(f"/api/v1/captures/{capture['id']}/files", json={"filename": "scan.ply"})
    window = response.json()["upload"]
    assert window["partsTotal"] is None
    assert len(window["parts"]) == capture_service.PRESIGN_WINDOW_PARTS
    assert window["nextPartNumber"] == capture_service.PRESIGN_WINDOW_PARTS + 1


def test_part_size_grows_past_s3s_part_limit() -> None:
    """78 GiB is the most 10 000 parts of 8 MiB can carry; past that, parts get bigger."""
    assert capture_service.choose_part_size(12 * 1000**3) == PART_SIZE
    assert capture_service.count_parts(12 * 1000**3, PART_SIZE) == 1431
    huge = 200 * 1024**3
    part_size = capture_service.choose_part_size(huge)
    assert part_size > PART_SIZE
    total = capture_service.count_parts(huge, part_size)
    assert total is not None and total <= capture_service.MAX_MULTIPART_PARTS


# --- abort -------------------------------------------------------------------


def test_abort_kills_the_upload_and_marks_the_row(client: TestClient, storage: S3Storage) -> None:
    capture = make_capture(client)
    registered = register(client, capture["id"])
    file, window = registered["file"], registered["upload"]
    put_part(window["parts"][0], slice_for(1, window["partSize"]))

    aborted = client.post(f"/api/v1/captures/{capture['id']}/files/{file['id']}/abort")
    assert aborted.status_code == 200, aborted.text
    assert aborted.json()["status"] == "aborted"
    assert aborted.json()["uploadId"] is None

    assert storage.head_object(file["storageKey"]) is None
    # Nothing landed, so the capture is still waiting for bytes.
    assert client.get(f"/api/v1/captures/{capture['id']}").json()["status"] == "awaiting-files"
    # And the dead upload cannot be presigned or completed.
    assert (
        client.post(
            f"/api/v1/captures/{capture['id']}/files/{file['id']}/parts", json={}
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/api/v1/captures/{capture['id']}/files/{file['id']}/complete",
            json={"parts": [{"partNumber": 1, "etag": "x"}]},
        ).status_code
        == 409
    )


def test_a_completed_file_cannot_be_aborted(client: TestClient) -> None:
    capture = make_capture(client)
    file = upload_file(client, capture["id"])
    response = client.post(f"/api/v1/captures/{capture['id']}/files/{file['id']}/abort")
    assert response.status_code == 409


# --- process -----------------------------------------------------------------


def test_process_queues_a_job_and_runs_nothing(client: TestClient, db: Session) -> None:
    capture = make_capture(client)
    upload_file(client, capture["id"])

    response = client.post(
        f"/api/v1/captures/{capture['id']}/process",
        json={
            # Keyed by stage id, which is the shape `Recipe.with_params` takes. A flat
            # `{"sh": 3}` used to be accepted here and silently ignored by the worker.
            "recipe": "splat-ingest",
            "params": {"package": {"max_gaussians": 120000}},
            "provider": "modal",
            "tier": "a10",
        },
    )
    assert response.status_code == 202, response.text
    job = response.json()

    assert job["status"] == "not-started"
    assert job["recipe"] == "splat-ingest"
    # Resolved from the recipe file, not supplied and not a literal in this service.
    assert job["recipeVersion"] == recipe_service.version_of("splat-ingest")
    assert job["params"] == {"package": {"max_gaussians": 120000}}
    assert job["captureId"] == capture["id"]
    # Nothing has run and nothing has claimed it.
    assert job["steps"] == []
    assert job["claimedBy"] is None and job["claimedAt"] is None
    assert job["leaseExpiresAt"] is None
    assert job["finishedAt"] is None
    assert db.scalar(select(func.count()).select_from(JobStep)) == 0
    assert db.scalar(select(func.count()).select_from(Artifact)) == 0
    assert db.scalar(select(func.count()).select_from(Job)) == 1
    # The capture is queued, not processing: only a worker moves it on.
    assert client.get(f"/api/v1/captures/{capture['id']}").json()["status"] == "not-started"


def test_process_needs_an_uploaded_file(client: TestClient) -> None:
    capture = make_capture(client)
    register(client, capture["id"])  # registered, never completed
    response = client.post(
        f"/api/v1/captures/{capture['id']}/process", json={"recipe": "splat-ingest"}
    )
    assert response.status_code == 409


def test_process_rejects_an_unknown_recipe(client: TestClient) -> None:
    capture = make_capture(client)
    upload_file(client, capture["id"])
    response = client.post(
        f"/api/v1/captures/{capture['id']}/process", json={"recipe": "make-it-nice"}
    )
    assert response.status_code == 422


def test_one_run_at_a_time(client: TestClient) -> None:
    capture = make_capture(client)
    upload_file(client, capture["id"])
    body = {"recipe": "splat-ingest"}
    assert client.post(f"/api/v1/captures/{capture['id']}/process", json=body).status_code == 202
    assert client.post(f"/api/v1/captures/{capture['id']}/process", json=body).status_code == 409


def test_files_cannot_be_added_once_a_capture_is_processing(
    client: TestClient, db: Session
) -> None:
    capture = make_capture(client)
    upload_file(client, capture["id"])
    job = client.post(
        f"/api/v1/captures/{capture['id']}/process", json={"recipe": "splat-ingest"}
    ).json()
    # Stand in for the worker: only it moves a capture to in-progress.
    row = db.get(Job, job["id"])
    assert row is not None
    row.capture.status = CaptureStatus.IN_PROGRESS
    db.commit()

    response = client.post(f"/api/v1/captures/{capture['id']}/files", json={"filename": "late.mov"})
    assert response.status_code == 409


# --- jobs --------------------------------------------------------------------


def test_jobs_can_be_read_listed_and_filtered(client: TestClient) -> None:
    capture = make_capture(client)
    upload_file(client, capture["id"])
    job = client.post(
        f"/api/v1/captures/{capture['id']}/process", json={"recipe": "splat-ingest"}
    ).json()

    assert client.get(f"/api/v1/jobs/{job['id']}").json()["id"] == job["id"]
    assert [j["id"] for j in client.get("/api/v1/jobs").json()] == [job["id"]]
    by_capture = client.get("/api/v1/jobs", params={"captureId": capture["id"]}).json()
    assert [j["id"] for j in by_capture] == [job["id"]]
    assert client.get("/api/v1/jobs", params={"status": "not-started"}).json() != []
    assert client.get("/api/v1/jobs", params={"status": "complete"}).json() == []
    other = make_capture(client, name="elsewhere")
    assert client.get("/api/v1/jobs", params={"captureId": other["id"]}).json() == []


def test_cancel_transitions_once(client: TestClient) -> None:
    capture = make_capture(client)
    upload_file(client, capture["id"])
    job = client.post(
        f"/api/v1/captures/{capture['id']}/process", json={"recipe": "splat-ingest"}
    ).json()

    cancelled = client.post(f"/api/v1/jobs/{job['id']}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["finishedAt"] is not None

    assert client.post(f"/api/v1/jobs/{job['id']}/cancel").status_code == 409
    # A cancelled run does not block the next one.
    assert (
        client.post(
            f"/api/v1/captures/{capture['id']}/process", json={"recipe": "splat-ingest"}
        ).status_code
        == 202
    )


def test_unknown_job_is_404(client: TestClient) -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/api/v1/jobs/{missing}").status_code == 404
    assert client.post(f"/api/v1/jobs/{missing}/cancel").status_code == 404
