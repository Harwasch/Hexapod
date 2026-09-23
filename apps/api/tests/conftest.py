from __future__ import annotations

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
def sessions(engine: Engine) -> sessionmaker[Session]:
    """A session factory, for the worker: it opens its own sessions per job."""
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
