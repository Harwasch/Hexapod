"""The two endpoints A7 added: retry a run from a stage, and read a step's log.

Both exist because of something the worker does. Retry is the *human* counterpart to the
worker's automatic one — the worker stops after `worker_max_attempts` and dead-letters,
and a person asking again resets that budget. The log endpoint is what makes
`job_steps.log_key` useful: logs go to object storage, never into the database.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.main import create_app
from app.models import Capture, Job, JobStep
from app.models.enums import CaptureKind, RunStatus
from app.storage import S3Storage, get_storage

BUCKET = "twin-jobs-test"


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
            public_base_url="https://cdn.example.com/twin-jobs-test",
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


def failed_run(db: Session, *, slug: str = "orchard") -> Job:
    """A job that ran two stages, completed the first and dead-lettered on the second."""
    capture = Capture(slug=slug, name="Orchard", kind=CaptureKind.GAUSSIAN_SPLAT)
    db.add(capture)
    db.commit()
    job = Job(
        capture_id=capture.id,
        recipe="splat-ingest",
        recipe_version="0.1.0",
        status=RunStatus.ERROR,
        error="stage 'package' has been attempted 3 times without completing",
        claimed_by="worker-a",
    )
    db.add(job)
    db.flush()
    db.add(
        JobStep(
            job_id=job.id,
            stage_id="normalize",
            ordinal=0,
            impl="ingest_splat",
            status=RunStatus.COMPLETE,
            attempt=1,
            log_key="runs/x/normalize/log.txt",
        )
    )
    db.add(
        JobStep(
            job_id=job.id,
            stage_id="package",
            ordinal=1,
            impl="splat_tiles",
            status=RunStatus.ERROR,
            attempt=3,
        )
    )
    db.commit()
    return job


def test_retry_resumes_at_the_failed_stage_and_gives_it_its_budget_back(
    client: TestClient, db: Session
) -> None:
    job = failed_run(db)

    response = client.post(f"/api/v1/jobs/{job.id}/retry", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not-started"
    assert body["error"] is None
    assert body["claimedBy"] is None and body["leaseExpiresAt"] is None
    steps = {step["stageId"]: step for step in body["steps"]}
    # The stage that succeeded keeps its row; the worker will skip it.
    assert steps["normalize"]["status"] == "complete"
    assert steps["normalize"]["attempt"] == 1
    # The one that failed starts over with the whole attempt budget: a retry that failed
    # instantly on the already-spent count would say nothing new.
    assert steps["package"]["status"] == "not-started"
    assert steps["package"]["attempt"] == 0


def test_retry_can_name_an_earlier_stage(client: TestClient, db: Session) -> None:
    job = failed_run(db)

    response = client.post(f"/api/v1/jobs/{job.id}/retry", json={"fromStage": "normalize"})

    assert response.status_code == 200
    steps = {step["stageId"]: step for step in response.json()["steps"]}
    assert steps["normalize"]["status"] == "not-started"
    assert steps["package"]["status"] == "not-started"


def test_retry_refuses_a_stage_the_job_never_had(client: TestClient, db: Session) -> None:
    job = failed_run(db)

    response = client.post(f"/api/v1/jobs/{job.id}/retry", json={"fromStage": "train"})

    # ValueError -> 422 Invalid input, the same shape every other bad payload gets.
    assert response.status_code == 422
    assert "train" in response.json()["detail"]


def test_retry_refuses_a_job_that_is_still_running(client: TestClient, db: Session) -> None:
    job = failed_run(db)
    job.status = RunStatus.IN_PROGRESS
    db.commit()

    response = client.post(f"/api/v1/jobs/{job.id}/retry", json={})

    assert response.status_code == 409


def test_a_step_log_is_read_out_of_object_storage(
    client: TestClient, db: Session, storage: S3Storage
) -> None:
    job = failed_run(db)
    storage.put_object("runs/x/normalize/log.txt", b"$ ingest\nread 1.2M gaussians\n", "text/plain")
    step = next(s for s in job.steps if s.stage_id == "normalize")

    response = client.get(f"/api/v1/jobs/{job.id}/steps/{step.id}/log")

    assert response.status_code == 200
    body = response.json()
    assert body["stageId"] == "normalize"
    assert body["logKey"] == "runs/x/normalize/log.txt"
    assert "1.2M gaussians" in body["text"]


def test_a_step_with_no_log_is_a_404_not_an_empty_string(client: TestClient, db: Session) -> None:
    job = failed_run(db)
    step = next(s for s in job.steps if s.stage_id == "package")

    assert client.get(f"/api/v1/jobs/{job.id}/steps/{step.id}/log").status_code == 404
    # And a step belonging to another job is not readable through this job's URL.
    other = failed_run(db, slug="second-orchard")
    stray = next(s for s in other.steps if s.stage_id == "normalize")
    assert client.get(f"/api/v1/jobs/{job.id}/steps/{stray.id}/log").status_code == 404
    assert client.get(f"/api/v1/jobs/{uuid.uuid4()}/steps/{stray.id}/log").status_code == 404
