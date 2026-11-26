"""A completed phone upload stores bytes; only an explicit action buys training."""

import uuid
from collections.abc import Iterator
from unittest.mock import Mock

import boto3
import pytest
from fastapi import BackgroundTasks, Request
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.api.v1 import captures as capture_routes
from app.api.v1 import phone as phone_routes
from app.config import Settings
from app.main import create_app
from app.models import Capture, CaptureFile, Job
from app.models.enums import RunStatus
from app.schemas.capture import CaptureFileComplete
from app.schemas.job import JobCreate
from app.services import capture_bundles, captures, jobs, phone_access, recipes, worker_wake
from app.services.errors import ConflictError
from app.storage import S3Storage, get_storage
from tests.conftest import WORKER_BUCKET
from tests.test_phone_app_access import ADMIN, PHONE, settings


@pytest.fixture
def app_client(db: Session, storage: S3Storage) -> Iterator[TestClient]:
    app = create_app(settings(phone_upload_auto_process=False))
    app.dependency_overrides[_db] = lambda: db
    app.dependency_overrides[get_storage] = lambda: storage
    with TestClient(app) as client:
        yield client


def upload(client: TestClient) -> str:
    capture = client.put(
        f"/api/v1/captures/bundles/{uuid.uuid4()}",
        headers=PHONE,
        json={"name": "Stored phone scan", "kind": "video"},
    ).json()
    base = f"/api/v1/captures/{capture['id']}"
    file = client.put(
        base + "/bundle-upload",
        headers=PHONE,
        json={"filename": "capture.hexcap", "contentType": "application/x-hex-capture", "bytes": 3},
    ).json()
    etag = boto3.client("s3", region_name="us-east-1").upload_part(
        Bucket=WORKER_BUCKET,
        Key=file["storageKey"],
        UploadId=file["uploadId"],
        PartNumber=1,
        Body=b"abc",
    )["ETag"]
    response = client.post(
        base + f"/files/{file['id']}/complete",
        headers=PHONE,
        json={"parts": [{"partNumber": 1, "etag": etag}]},
    )
    assert response.status_code == 200, response.text
    return str(capture["id"])


def test_upload_auto_process_defaults_off() -> None:
    assert (
        Settings(_env_file=None, phone_upload_auto_process=False).phone_upload_auto_process is False
    )
    assert Settings.model_fields["phone_upload_auto_process"].default is False


@pytest.mark.parametrize("automatic", [False, True])
def test_phone_upload_waits_for_refine_and_double_press_recovers_job(
    app_client: TestClient, db: Session, monkeypatch: pytest.MonkeyPatch, automatic: bool
) -> None:
    app_client.app.state.settings.phone_upload_auto_process = automatic  # type: ignore[attr-defined]
    wake = Mock()
    monkeypatch.setattr(worker_wake, "schedule", wake)
    cid = upload(app_client)
    detail = app_client.get(f"/api/v1/captures/{cid}", headers=PHONE).json()
    assert detail["status"] == "not-started"  # Existing stored/unprocessed state.
    assert len(detail["jobs"]) == int(automatic)
    assert db.scalar(select(func.count()).select_from(Job)) == int(automatic)
    assert wake.call_count == int(automatic)

    url = f"/api/v1/captures/{cid}/process-bundle"
    first = app_client.post(url, headers=PHONE)
    assert first.status_code == 202, first.text
    expected = uuid.uuid5(uuid.UUID(cid), "photo-reconstruct")
    assert first.json()["id"] == str(expected)
    second = app_client.post(url, headers=PHONE)
    assert second.status_code == 202, second.text
    assert second.json()["id"] == first.json()["id"]
    card = app_client.post(
        f"/api/v1/captures/{cid}/process", headers=ADMIN, json={"recipe": "photo-reconstruct"}
    )
    assert card.status_code == 409, card.text
    assert db.scalar(select(func.count()).select_from(Job)) == 1


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("origin", ["phone-key", "desktop"])
def test_upload_completion_dispatches_only_when_opted_in(
    monkeypatch: pytest.MonkeyPatch, automatic: bool, origin: str
) -> None:
    capture = Capture(
        id=uuid.uuid4(), metadata_={"origin": origin, "bundleFormat": "hex-capture-v1"}
    )
    file = CaptureFile(capture=capture)
    monkeypatch.setattr(captures, "complete_file", Mock(return_value=file))
    read = Mock()
    monkeypatch.setattr(captures, "file_to_read", Mock(return_value=read))
    process = Mock()
    wake = Mock()
    monkeypatch.setattr(capture_bundles, "process", process)
    monkeypatch.setattr(phone_access, "check_run", Mock())
    monkeypatch.setattr(worker_wake, "schedule", wake)
    settings_value = Settings(_env_file=None, phone_upload_auto_process=automatic)
    result = capture_routes.complete_file(
        capture.id,
        uuid.uuid4(),
        CaptureFileComplete(parts=[{"partNumber": 1, "etag": "x"}]),
        Mock(spec=Session),
        Mock(),
        settings_value,
        BackgroundTasks(),
    )
    assert result is read
    assert process.call_count == wake.call_count == int(automatic and origin == "phone-key")


def test_on_demand_locks_before_creating_and_preserves_legacy_job_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = Mock(spec=Session)
    cid = uuid.uuid4()
    monkeypatch.setattr(jobs, "get_capture", Mock())
    new = Job(id=uuid.uuid5(cid, "photo-reconstruct"))
    create = Mock(return_value=new)
    monkeypatch.setattr(jobs, "create_job", create)
    db.scalar.return_value = None
    payload = JobCreate(recipe="photo-reconstruct")
    assert jobs.create_on_demand(db, cid, payload) is new
    assert "FOR UPDATE" in str(db.execute.call_args.args[0])
    assert create.call_args.kwargs["job_id"] == new.id
    for status in (RunStatus.NOT_STARTED, RunStatus.IN_PROGRESS, RunStatus.COMPLETE):
        old = Job(id=uuid.uuid4(), status=status)
        db.scalar.return_value = old
        assert jobs.create_on_demand(db, cid, payload) is old
    create.assert_called_once()


def test_stopped_unclaimed_phone_job_restarts_with_same_id(monkeypatch: pytest.MonkeyPatch) -> None:
    db = Mock(spec=Session)
    cid = uuid.uuid4()
    job = Job(id=uuid.uuid4(), status=RunStatus.CANCELLED, steps=[], error="Stopped")
    db.scalar.return_value = job
    monkeypatch.setattr(jobs, "get_capture", Mock())
    monkeypatch.setattr(recipes, "check_overrides", Mock())
    create = Mock()
    monkeypatch.setattr(jobs, "create_job", create)
    assert (
        jobs.create_on_demand(db, cid, JobCreate(recipe="photo-reconstruct"), retry_failed=True)
        is job
    )
    assert job.status == RunStatus.NOT_STARTED and job.error is None
    create.assert_not_called()
    db.commit.assert_called_once()


def test_legacy_phone_process_is_idempotent(app_client: TestClient, db: Session) -> None:
    from tests.test_phone import uploaded

    cid = app_client.post("/api/v1/phone/captures", headers=PHONE, json={}).json()["capture"]["id"]
    uploaded(db, cid)
    db.commit()
    url = f"/api/v1/phone/captures/{cid}/process"
    first = app_client.post(url, headers=PHONE, json={"recipe": "photo-reconstruct"})
    second = app_client.post(url, headers=PHONE, json={"recipe": "photo-reconstruct"})
    assert first.status_code == second.status_code == 202
    assert (
        first.json()["id"]
        == second.json()["id"]
        == str(uuid.uuid5(uuid.UUID(cid), "photo-reconstruct"))
    )
    assert db.scalar(select(func.count()).select_from(Job)) == 1


@pytest.mark.parametrize("bundle", [False, True])
def test_admin_process_preserves_new_run_options(
    monkeypatch: pytest.MonkeyPatch, bundle: bool
) -> None:
    db = Mock(spec=Session)
    cid = uuid.uuid4()
    metadata = {"origin": "phone-key"}
    if bundle:
        metadata["bundleFormat"] = "hex-capture-v1"
    monkeypatch.setattr(captures, "get_capture", Mock(return_value=Capture(metadata_=metadata)))
    create = Mock(return_value=Job())
    monkeypatch.setattr(jobs, "create_job", create)
    monkeypatch.setattr(capture_bundles, "process", Mock(return_value=Job()))
    monkeypatch.setattr(jobs, "create_on_demand", Mock(return_value=Job()))
    monkeypatch.setattr(jobs, "job_to_read", Mock())
    monkeypatch.setattr(worker_wake, "schedule", Mock())
    payload = JobCreate(
        recipe="photo-reconstruct",
        params={"train": {"schedule_scale": 0.5}},
        provider="modal",
        tier="l4",
    )
    capture_routes.process_capture(cid, payload, db, BackgroundTasks())
    create.assert_called_once_with(db, cid, payload)


@pytest.mark.parametrize("bundle", [False, True])
def test_admin_new_run_after_finished_phone_job_keeps_overrides(
    app_client: TestClient, db: Session, bundle: bool
) -> None:
    from tests.test_phone import uploaded

    if bundle:
        cid = upload(app_client)
    else:
        cid = app_client.post("/api/v1/phone/captures", headers=PHONE, json={}).json()["capture"][
            "id"
        ]
        uploaded(db, cid)
        db.commit()
    url = f"/api/v1/captures/{cid}/process"
    first = app_client.post(url, headers=ADMIN, json={"recipe": "photo-reconstruct"})
    assert first.status_code == 202, first.text
    job = db.get(Job, uuid.UUID(first.json()["id"]))
    assert job is not None
    job.status = RunStatus.COMPLETE
    db.commit()
    payload = {
        "recipe": "photo-reconstruct",
        "params": {"train": {"schedule_scale": 0.5}},
        "provider": "modal",
        "tier": "l4",
    }
    second = app_client.post(url, headers=ADMIN, json=payload)
    assert second.status_code == 202, second.text
    assert second.json()["id"] != first.json()["id"]
    for key, value in payload.items():
        assert second.json()[key] == value
    assert db.scalar(select(func.count()).select_from(Job)) == 2


@pytest.mark.parametrize("status", [RunStatus.ERROR, RunStatus.CANCELLED])
def test_phone_bundle_refine_requeues_dead_job_in_place(
    app_client: TestClient, db: Session, status: RunStatus
) -> None:
    cid = upload(app_client)
    url = f"/api/v1/captures/{cid}/process-bundle"
    first = app_client.post(url, headers=PHONE)
    assert first.status_code == 202, first.text
    job = db.get(Job, uuid.UUID(first.json()["id"]))
    assert job is not None
    job.status = status
    db.commit()
    retried = app_client.post(url, headers=PHONE)
    assert retried.status_code == 202, retried.text
    assert retried.json()["id"] == first.json()["id"]
    assert retried.json()["status"] == "not-started"
    assert db.scalar(select(func.count()).select_from(Job)) == 1


def request_for_refine(credential: str, header: str | None) -> Request:
    headers = [] if header is None else [(b"x-hex-refine", header.encode())]
    request = Request({"type": "http", "headers": headers})
    request.state.api_credential = credential
    return request


@pytest.mark.parametrize("credential", ["phone", "write"])
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("header", [None, "0", "1"])
def test_cloud_refine_gate_distinguishes_phone_and_admin(
    monkeypatch: pytest.MonkeyPatch, credential: str, automatic: bool, header: str | None
) -> None:
    process = Mock(return_value=Job())
    monkeypatch.setattr(capture_bundles, "process", process)
    monkeypatch.setattr(captures, "get_capture", Mock())
    monkeypatch.setattr(phone_access, "check_run", Mock())
    monkeypatch.setattr(jobs, "job_to_read", Mock())
    monkeypatch.setattr(worker_wake, "schedule", Mock())
    args = (
        uuid.uuid4(),
        Mock(spec=Session),
        BackgroundTasks(),
        request_for_refine(credential, header),
        Settings(_env_file=None, phone_upload_auto_process=automatic),
    )
    if credential == "phone" and not automatic and header != "1":
        with pytest.raises(ConflictError, match="Update the app to refine in the cloud"):
            capture_routes.process_bundle(*args)
        process.assert_not_called()
    else:
        capture_routes.process_bundle(*args)
        assert process.call_args.kwargs["phone"] == (credential == "phone")


@pytest.mark.parametrize("header", [None, "1"])
def test_legacy_phone_process_requires_explicit_refine(
    monkeypatch: pytest.MonkeyPatch, header: str | None
) -> None:
    create = Mock(return_value=Job())
    monkeypatch.setattr(jobs, "create_on_demand", create)
    monkeypatch.setattr(jobs, "job_to_read", Mock())
    monkeypatch.setattr(
        captures, "get_capture", Mock(return_value=Capture(metadata_={"origin": "phone-key"}))
    )
    monkeypatch.setattr(phone_access, "check_run", Mock())
    monkeypatch.setattr(worker_wake, "schedule", Mock())
    args = (
        uuid.uuid4(),
        JobCreate(recipe="photo-reconstruct"),
        Mock(spec=Session),
        BackgroundTasks(),
        request_for_refine("phone", header),
        Settings(_env_file=None, phone_upload_auto_process=False),
    )
    if header is None:
        with pytest.raises(ConflictError, match="Update the app to refine in the cloud"):
            phone_routes.process_phone_capture(*args)
        create.assert_not_called()
    else:
        phone_routes.process_phone_capture(*args)
        create.assert_called_once()


@pytest.mark.parametrize("bundle", [False, True])
def test_phone_gate_blocks_legacy_automatic_requests(
    app_client: TestClient, db: Session, bundle: bool
) -> None:
    from tests.test_phone import uploaded

    if bundle:
        cid = upload(app_client)
        url = f"/api/v1/captures/{cid}/process-bundle"
        payload = {}
    else:
        cid = app_client.post("/api/v1/phone/captures", headers=PHONE, json={}).json()["capture"][
            "id"
        ]
        uploaded(db, cid)
        db.commit()
        url = f"/api/v1/phone/captures/{cid}/process"
        payload = {"recipe": "photo-reconstruct"}
    old_headers = {key: value for key, value in PHONE.items() if key != "X-Hex-Refine"}
    refused = app_client.post(url, headers=old_headers, json=payload)
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"] == "Update the app to refine in the cloud"
    assert db.scalar(select(func.count()).select_from(Job)) == 0
    accepted = app_client.post(url, headers={**PHONE, "X-Hex-Refine": "1"}, json=payload)
    assert accepted.status_code == 202, accepted.text
    assert db.scalar(select(func.count()).select_from(Job)) == 1
