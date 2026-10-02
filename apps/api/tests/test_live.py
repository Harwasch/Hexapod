"""The live viewer's data path, worker side and API side.

The container logs `live-cameras:` and `live-splat:` lines (tools/pipeline/live.py); the
supervisor's heartbeat copies the newest of each onto the running step as `metrics.live`;
`GET /captures/{id}/live` assembles them across the run and signs the snapshot's URL.
"""

from __future__ import annotations

import base64
import json
import struct
import uuid
from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.main import create_app
from app.models import Capture, Job, JobStep
from app.models.enums import CaptureKind, RunStatus
from app.schemas.job import JobRead
from app.storage import NullStorage, S3Storage, get_storage
from app.worker import steps as step_service
from app.worker.pipeline_bridge import Workdir
from app.worker.runner import JobSupervisor
from tests.conftest import site_payload

BUCKET = "twin-live-test"
SPLAT_KEY = "runs/run-1/train/checkpoint/live/splat_003000_1700000000.spz"


def cameras_payload(registered: int = 2, *, final: bool = False) -> dict[str, object]:
    """What `live.encode_model` logs, for two cameras and one point."""
    cameras = b"".join(
        struct.pack("<3h6b", 1000 * i, 0, 0, 0, 0, 127, 0, -127, 0) for i in range(registered)
    )
    points = struct.pack("<3h3B", 0, 0, 32767, 200, 100, 50)
    return {
        "v": 1,
        "seq": registered,
        "final": final,
        "registered": registered,
        "frames": 40,
        "origin": [0.0, 0.0, 0.0],
        "scale": 10.0,
        "up": [0.0, -1.0, 0.0],
        "aspect": 1.3333,
        "cameraCount": registered,
        "cameras": base64.b64encode(cameras).decode(),
        "pointCount": 1,
        "pointsTotal": 812,
        "points": base64.b64encode(points).decode(),
    }


def splat_payload(step: int = 3000, key: str = SPLAT_KEY) -> dict[str, object]:
    return {
        "v": 1,
        "step": step,
        "total": 30000,
        "count": 100000,
        "of": 412733,
        "bytes": 1843200,
        "key": key,
        "up": [0.0, -1.0, 0.0],
    }


def run_of(db: Session, slug: str) -> tuple[Capture, Job]:
    capture = Capture(slug=slug, name="Garden tree", kind=CaptureKind.IMAGES)
    db.add(capture)
    db.commit()
    job = Job(capture_id=capture.id, recipe="photo-reconstruct", recipe_version="1", params={})
    job.status = RunStatus.IN_PROGRESS
    db.add(job)
    db.commit()
    return capture, job


# --- the worker ----------------------------------------------------------------------


def test_the_heartbeat_copies_the_newest_live_lines_onto_the_running_step(
    db: Session, tmp_path: Path
) -> None:
    _capture, job = run_of(db, "live-worker")
    step = step_service.start_step(db, job.id, stage_id="pose", ordinal=1, impl="colmap", attempt=1)
    log_path = Workdir(tmp_path).log_path("pose")
    log_path.parent.mkdir(parents=True)
    log_path.write_text("$ colmap mapper\n", encoding="utf-8")

    JobSupervisor._report_progress(db, step, tmp_path)
    assert "live" not in step.metrics

    with log_path.open("a", encoding="utf-8") as handle:
        for registered in (2, 3):
            handle.write("live-cameras: " + json.dumps(cameras_payload(registered)) + "\n")
        handle.write("I20260927 mapper.cc:12] Registering image #9\n")
    JobSupervisor._report_progress(db, step, tmp_path)
    db.expire_all()
    row = db.get(JobStep, step.id)
    assert row is not None
    assert row.metrics["live"]["cameras"]["registered"] == 3
    # Nothing new: no write.
    assert step_service.report_live(db, row, {"cameras": row.metrics["live"]["cameras"]}) is False

    # The final model is logged just before the stage ends and is kept once it has.
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("live-cameras: " + json.dumps(cameras_payload(4, final=True)) + "\n")
    JobSupervisor._report_progress(db, row, tmp_path)
    step_service.finish_step(
        db, row, metrics={"registered": 4}, log_key=None, checkpoint_key=None, artifacts=[]
    )
    db.expire_all()
    done = db.get(JobStep, step.id)
    assert done is not None
    assert done.metrics["registered"] == 4
    assert done.metrics["live"]["cameras"]["final"] is True
    # ...but not in what every job poll carries.
    assert "live" not in JobRead.model_validate(db.get(Job, job.id)).steps[0].metrics

    # A stage that runs again starts with nothing to show.
    again = step_service.start_step(
        db, job.id, stage_id="pose", ordinal=1, impl="colmap", attempt=2
    )
    assert again.metrics == {}


def test_a_splat_line_is_merged_beside_the_progress(db: Session, tmp_path: Path) -> None:
    _capture, job = run_of(db, "live-train")
    step = step_service.start_step(
        db, job.id, stage_id="train", ordinal=3, impl="gsplat", attempt=1
    )
    log_path = Workdir(tmp_path).log_path("train")
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "live-splat: " + json.dumps(splat_payload(3000)) + "\n"
        "loss=0.04| :  12%|#  | 3600/30000 [10:00<1:10:00,  6.00it/s]\n"
        # Not a key the pipeline writes: never stored, never signed.
        "live-splat: " + json.dumps(splat_payload(3500, key="captures/c/original.mov")) + "\n",
        encoding="utf-8",
    )

    JobSupervisor._report_progress(db, step, tmp_path)

    db.expire_all()
    row = db.get(JobStep, step.id)
    assert row is not None
    assert row.metrics["progress"]["done"] == 3600
    assert row.metrics["live"] == {"splat": splat_payload(3000)}


# --- the API -------------------------------------------------------------------------


@pytest.fixture
def storage() -> Iterator[S3Storage]:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield S3Storage(
            bucket=BUCKET,
            endpoint_url=None,
            access_key="key",
            secret_key="secret",
            region="us-east-1",
            public_base_url="https://cdn.example.com/twin-live-test",
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


def training_run(db: Session, slug: str) -> tuple[Capture, Job]:
    capture, job = run_of(db, slug)
    db.add(
        JobStep(
            job_id=job.id,
            stage_id="pose",
            ordinal=1,
            impl="colmap",
            status=RunStatus.COMPLETE,
            attempt=1,
            metrics={"registered": 2, "live": {"cameras": cameras_payload(final=True)}},
        )
    )
    step = step_service.start_step(
        db, job.id, stage_id="train", ordinal=3, impl="gsplat", attempt=1
    )
    step_service.report_progress(
        db, step, {"done": 3600, "total": 30000, "elapsedS": 600, "remainingS": 4200}
    )
    step_service.report_live(db, step, {"splat": splat_payload()})
    return capture, job


def test_live_state_puts_trainings_splat_among_the_poses_cameras(
    client: TestClient, db: Session, storage: S3Storage
) -> None:
    capture, job = training_run(db, "live-api")
    storage.put_object(SPLAT_KEY, b"spz bytes", "application/octet-stream")

    response = client.get(f"/api/v1/captures/{capture.id}/live")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["jobId"] == str(job.id) and body["status"] == "in-progress"
    assert body["captureName"] == "Garden tree" and body["siteId"] is None
    assert body["stage"]["stageId"] == "train"
    assert body["stage"]["progress"] == {
        "done": 3600,
        "total": 30000,
        "elapsedS": 600,
        "remainingS": 4200,
    }
    assert body["stepsDone"] == 1 and body["stepsStarted"] == 2
    assert body["cameras"]["stageId"] == "pose" and body["cameras"]["cameraCount"] == 2
    assert body["cameras"]["cameras"] == cameras_payload()["cameras"]
    splat = body["splat"]
    assert splat["step"] == 3000 and splat["count"] == 100000
    assert splat["name"] == "splat_003000_1700000000.spz"
    assert SPLAT_KEY in splat["url"] and "Signature" in splat["url"]
    # The same thing by job.
    assert client.get(f"/api/v1/jobs/{job.id}/live").json()["splat"]["step"] == 3000


def test_a_snapshot_not_uploaded_yet_has_no_url(client: TestClient, db: Session) -> None:
    capture, _job = training_run(db, "live-not-yet")

    splat = client.get(f"/api/v1/captures/{capture.id}/live").json()["splat"]

    assert splat["step"] == 3000 and splat["url"] is None


def test_a_finished_run_names_the_site_its_scan_is_on(client: TestClient, db: Session) -> None:
    capture, job = training_run(db, "live-done")
    created = client.post("/api/v1/sites", json=site_payload(slug="live-done-site"))
    assert created.status_code == 201, created.text
    site_id = uuid.UUID(created.json()["id"])
    capture.site_id = site_id
    job.status = RunStatus.COMPLETE
    db.commit()

    body = client.get(f"/api/v1/captures/{capture.id}/live").json()

    assert body["status"] == "complete" and body["siteId"] == str(site_id)


def test_a_capture_with_no_run_is_a_404(client: TestClient, db: Session) -> None:
    capture = Capture(slug="live-none", name="Nothing yet", kind=CaptureKind.IMAGES)
    db.add(capture)
    db.commit()

    assert client.get(f"/api/v1/captures/{capture.id}/live").status_code == 404
    assert client.get(f"/api/v1/captures/{uuid.uuid4()}/live").status_code == 404
    assert client.get(f"/api/v1/jobs/{uuid.uuid4()}/live").status_code == 404


def test_without_a_bucket_the_state_is_still_served(db: Session) -> None:
    capture, _job = training_run(db, "live-no-bucket")
    app = create_app()

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    app.dependency_overrides[get_storage] = NullStorage
    with TestClient(app) as test_client:
        body = test_client.get(f"/api/v1/captures/{capture.id}/live").json()

    assert body["cameras"]["registered"] == 2
    assert body["splat"]["url"] is None
