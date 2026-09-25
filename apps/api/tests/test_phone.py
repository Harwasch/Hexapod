"""The phone key: capture from a phone with a short shared key and nothing else."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.config import Settings
from app.main import create_app
from app.models.capture import CaptureFile
from app.models.enums import ArtifactKind, RunStatus, UploadStatus
from app.models.job import Artifact, Job, JobStep
from app.services import phone_key
from app.storage import S3Storage, get_storage

KEY = "abcd-efgh-jkmn"
WRITE = "the-write-token"
PHONE = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
def client(db: Session, storage: S3Storage) -> Iterator[TestClient]:
    settings = Settings(
        api_write_token=WRITE,
        # A low iteration count keeps the suite fast; the stored format is the real one.
        api_phone_key_hash=phone_key.hash_key(KEY, salt=b"0123456789abcdef", iterations=1000),
        api_phone_daily_captures=3,
    )
    app = create_app(settings)

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    app.dependency_overrides[get_storage] = lambda: storage
    with TestClient(app) as test_client:
        yield test_client


def uploaded(db: Session, capture_id: str) -> None:
    """A finished upload, written directly: the part PUTs go to storage, not this API."""
    db.add(
        CaptureFile(
            capture_id=uuid.UUID(capture_id),
            filename="walk.mov",
            content_type="video/quicktime",
            bytes=10,
            storage_key=f"captures/{capture_id}/source/walk.mov",
            status=UploadStatus.COMPLETE,
        )
    )
    db.flush()


def test_the_key_is_checked_and_typed_the_way_a_phone_types(client: TestClient) -> None:
    assert client.post("/api/v1/phone/check", headers=PHONE).status_code == 204
    loose = {"Authorization": f"Bearer  {KEY.upper()} "}
    assert client.post("/api/v1/phone/check", headers=loose).status_code == 204
    wrong = {"Authorization": "Bearer abcd-efgh-jkmm"}
    assert client.post("/api/v1/phone/check", headers=wrong).status_code == 401
    assert client.post("/api/v1/phone/check").status_code == 401


def test_a_phone_capture_is_placed_where_the_phone_was_and_can_upload(
    client: TestClient,
) -> None:
    created = client.post(
        "/api/v1/phone/captures",
        json={"lat": 44.9778, "lon": -93.265, "accuracyM": 8},
        headers=PHONE,
    )
    assert created.status_code == 201, created.text
    body = created.json()
    capture = body["capture"]
    assert capture["metadata"] == {
        "origin": "phone-key",
        "lat": 44.9778,
        "lon": -93.265,
        "locationAccuracyM": 8.0,
    }
    # The upload token is an ordinary handoff token for this capture: the existing
    # upload routes take it, with no change to them.
    upload = {"Authorization": f"Bearer {body['uploadToken']}"}
    registered = client.post(
        f"/api/v1/captures/{capture['id']}/files",
        json={"filename": "walk.mov", "contentType": "video/quicktime", "bytes": 10},
        headers=upload,
    )
    assert registered.status_code == 201, registered.text


def test_the_key_starts_runs_only_on_its_own_captures(client: TestClient, db: Session) -> None:
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    uploaded(db, mine["id"])
    other = client.post(
        "/api/v1/captures",
        json={"name": "desktop", "kind": "video"},
        headers={"Authorization": f"Bearer {WRITE}"},
    ).json()
    uploaded(db, other["id"])

    run = {"recipe": "photo-reconstruct"}
    ok = client.post(f"/api/v1/phone/captures/{mine['id']}/process", json=run, headers=PHONE)
    assert ok.status_code == 202, ok.text
    assert ok.json()["recipe"] == "photo-reconstruct"
    refused = client.post(f"/api/v1/phone/captures/{other['id']}/process", json=run, headers=PHONE)
    assert refused.status_code == 401
    odd = client.post(
        f"/api/v1/phone/captures/{mine['id']}/process", json={"recipe": "x"}, headers=PHONE
    )
    assert odd.status_code == 409


def test_the_key_stops_a_run_on_its_own_captures_only(client: TestClient, db: Session) -> None:
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    uploaded(db, mine["id"])
    other = client.post(
        "/api/v1/captures",
        json={"name": "desktop", "kind": "video"},
        headers={"Authorization": f"Bearer {WRITE}"},
    ).json()
    uploaded(db, other["id"])
    run = {"recipe": "photo-reconstruct"}
    started = client.post(f"/api/v1/phone/captures/{mine['id']}/process", json=run, headers=PHONE)
    client.post(
        f"/api/v1/captures/{other['id']}/jobs",
        json=run,
        headers={"Authorization": f"Bearer {WRITE}"},
    )

    assert client.post(f"/api/v1/phone/captures/{mine['id']}/stop").status_code == 401
    refused = client.post(f"/api/v1/phone/captures/{other['id']}/stop", headers=PHONE)
    assert refused.status_code == 401
    stopped = client.post(f"/api/v1/phone/captures/{mine['id']}/stop", headers=PHONE)
    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["id"] == started.json()["id"]
    assert stopped.json()["status"] == "cancelled"
    again = client.post(f"/api/v1/phone/captures/{mine['id']}/stop", headers=PHONE)
    assert again.status_code == 409
    # Stopped, it can be started again.
    retried = client.post(f"/api/v1/phone/captures/{mine['id']}/process", json=run, headers=PHONE)
    assert retried.status_code == 202, retried.text


def test_the_key_cannot_do_what_the_write_token_does(client: TestClient) -> None:
    assert (
        client.post(
            "/api/v1/captures", json={"name": "n", "kind": "video"}, headers=PHONE
        ).status_code
        == 401
    )


def test_the_daily_limit_holds(client: TestClient) -> None:
    for _ in range(3):
        assert client.post("/api/v1/phone/captures", json={}, headers=PHONE).status_code == 201
    over = client.post("/api/v1/phone/captures", json={}, headers=PHONE)
    assert over.status_code == 409
    assert "limit" in over.json()["detail"]


def test_with_no_key_configured_the_phone_routes_are_closed(db: Session) -> None:
    app = create_app(Settings(api_write_token=WRITE))

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    with TestClient(app) as test_client:
        assert test_client.post("/api/v1/phone/check", headers=PHONE).status_code == 401


def test_a_phone_sets_only_the_options_it_is_allowed(client: TestClient, db: Session) -> None:
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    uploaded(db, mine["id"])
    url = f"/api/v1/phone/captures/{mine['id']}/process"

    def start(params: dict[str, object], recipe: str = "photo-reconstruct") -> Response:
        return client.post(url, json={"recipe": recipe, "params": params}, headers=PHONE)

    for refused in (
        {"train": {"trainer": "/bin/sh"}},
        {"train": {"cap_max": 10}},
        {"train": {"cap_max": True}},
        {"pose": {"matcher": "sequential"}},
        {"normalize": "fast"},
    ):
        response = start(refused)
        assert response.status_code == 409, (refused, response.text)
    assert start({"normalize": {"up_axis": "sideways"}}, "splat-ingest").status_code == 409

    chosen = {
        "normalize": {"max_side": 2400},
        "train": {"schedule_floor": 1.0, "cap_max": 1_000_000},
        "package": {"max_gaussians": 800_000},
    }
    ok = start(chosen)
    assert ok.status_code == 202, ok.text
    job = db.get(Job, uuid.UUID(ok.json()["id"]))
    assert job is not None and job.params == chosen


def test_a_finished_capture_downloads_its_placed_splat(client: TestClient, db: Session) -> None:
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    capture_id = uuid.UUID(mine["id"])
    url = f"/api/v1/captures/{capture_id}/splat.ply"
    assert client.get(url, follow_redirects=False).status_code == 404

    job = Job(capture_id=capture_id, recipe="photo-reconstruct", recipe_version="7", params={})
    job.status = RunStatus.COMPLETE
    job.finished_at = datetime.now(tz=UTC)
    db.add(job)
    db.flush()
    for ordinal, stage, key in (
        (3, "train", "runs/x/train/trained.ply"),
        (6, "place", "runs/x/place/canonical.ply"),
    ):
        step = JobStep(job_id=job.id, stage_id=stage, ordinal=ordinal, impl=stage)
        step.status = RunStatus.COMPLETE
        db.add(step)
        db.flush()
        kind = ArtifactKind.SPLAT if stage == "place" else ArtifactKind.METADATA
        db.add(Artifact(job_step_id=step.id, kind=kind, storage_key=key, bytes=1))
    db.commit()

    response = client.get(url, follow_redirects=False)
    assert response.status_code == 307
    assert "runs/x/place/canonical.ply" in response.headers["location"]
