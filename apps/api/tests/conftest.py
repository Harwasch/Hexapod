from __future__ import annotations

import gc
import os
from collections.abc import Generator, Iterator

import boto3
import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from alembic import command
from app.api.deps import _db
from app.config import get_settings
from app.main import create_app
from app.storage import S3Storage

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL") or get_settings().test_database_url
TABLES = (
    "land_image_registration_blobs",
    "land_image_registrations",
    "land_archive_image_blobs",
    "land_archive_images",
    "land_raster_blobs",
    "land_rasters",
    "land_document_ocr",
    "land_document_links",
    "land_document_pages",
    "land_document_blobs",
    "land_documents",
    "land_action_revisions",
    "land_actions",
    "land_feature_inspections",
    "land_feature_revisions",
    "land_features",
    "land_scenario_revisions",
    "land_scenarios",
    "land_research_events",
    "land_research_artifacts",
    "land_findings",
    "land_evidence",
    "land_research_messages",
    "land_research_runs",
    "land_investigations",
    "workspace_memberships",
    "workspaces",
    "land_boundary_revisions",
    "land_areas",
    "artifacts",
    "job_steps",
    "jobs",
    "capture_files",
    "captures",
    "plan_revisions",
    "plans",
    "camera_bookmarks",
    "assets",
    "layers",
    "sites",
)


@pytest.fixture(scope="session")
def alembic_config() -> Config:
    config = Config("alembic.ini")
    os.environ["ALEMBIC_DATABASE_URL"] = TEST_DATABASE_URL
    return config


@pytest.fixture(scope="session")
def engine(alembic_config: Config) -> Iterator[Engine]:
    command.upgrade(alembic_config, "head")
    engine = create_engine(TEST_DATABASE_URL, future=True)
    yield engine
    engine.dispose()


@pytest.fixture
def db(engine: Engine) -> Iterator[Session]:
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        with engine.begin() as connection:
            connection.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))


#: The bucket the worker's tests write to, inside moto.
WORKER_BUCKET = "twin-worker-test"


@pytest.fixture
def frozen_heap() -> Iterator[None]:
    """Keep the garbage collector off everything the suite made before this test.

    The worker's tests run real leases at test speed: `FAST_LEASE_S` is 1.5 s, renewed
    every 0.2 s from `claim.LeaseKeeper`'s thread, while a second slot polls the queue
    every 0.05 s and takes any lease that has lapsed. A full collection stops every
    thread in the process for as long as it takes to walk the heap, and by the time
    `uv run pytest` reaches `test_worker*.py` the earlier tests have left some 3.6 M
    objects in it: a collection there took 3.4 s, measured, and CPython ran one every
    minute or two (it grows along the suite -- 0.2 s at `test_bookmarks`, 1.5 s at
    `test_sidecar_attach`). The database's clock does not stop with the process, so the
    lease lapsed with its keeper frozen; when the process resumed, the idle slot's poll
    and the keeper's renewal raced for the row, and the slot won about half the time:
    `job claimed 2 times, by ['worker-a/0', 'worker-a/1']`, in CI on 2026-10-05, now and
    then, and never when the module runs alone, where the heap is small.

    That is not the worker's to survive. Its lease is 30 s and its heap a fraction of
    this one, and a process frozen for longer than its lease is meant to lose its jobs
    (`claim.LeaseKeeper`). So for the length of the test the heap from before it is moved
    out of the collector's sight (`gc.freeze`: a list splice, not a walk) and then given
    back. Collections during the test still run and still free what the test makes, and
    walk only that: none took over 50 ms in a full run. No lease, timing or assertion
    changes.
    """
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()


@pytest.fixture
def sessions(engine: Engine, frozen_heap: None) -> sessionmaker[Session]:
    """A session factory, for the worker: it opens its own sessions per job.

    Every test that runs the worker in this process asks for it, so each of them runs
    with the earlier tests' heap frozen (`frozen_heap`)."""
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def storage() -> Iterator[S3Storage]:
    """A real `S3Storage` against moto, for the worker's uploads and its transfer.

    Here rather than in one test module because two of them need it since B1b: the
    supervisor's own tests and the cloud seam's.
    """
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=WORKER_BUCKET)
        yield S3Storage(
            bucket=WORKER_BUCKET,
            endpoint_url=None,
            access_key="key",
            secret_key="secret",
            region="us-east-1",
            public_base_url="https://cdn.example.com/twin-worker-test",
        )


@pytest.fixture
def client(db: Session) -> Generator[TestClient, None, None]:
    app = create_app()

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    with TestClient(app) as test_client:
        yield test_client


SQUARE = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.139, 47.644],
            [-122.137, 47.644],
            [-122.137, 47.645],
            [-122.139, 47.645],
            [-122.139, 47.644],
        ]
    ],
}


def site_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "Test Site",
        "description": "A test site",
        "boundary": SQUARE,
        "attribution": [{"text": "Test attribution", "organization": "Tests"}],
        "license": {"name": "CC BY 4.0", "spdxId": "CC-BY-4.0"},
        "assets": [
            {
                "name": "Splat",
                "representation": "gaussian-splat",
                "source": {"type": "cesium-ion", "assetId": 4547222},
                "defaultVisible": True,
                "observedAt": "2025-06-01T00:00:00Z",
            }
        ],
        "cameraBookmarks": [
            {
                "name": "Overview",
                "longitude": -122.138,
                "latitude": 47.6445,
                "height": 300,
                "isDefault": True,
            }
        ],
    }
    payload.update(overrides)
    return payload
