"""The data console's read surface: reconciliation, outputs, and the recipe catalogue.

Run against moto. What moto is trusted for here is S3's *bookkeeping* -- which keys
exist, what a listing pages, what a HEAD says about an object that is not there -- and
that is exactly what reconciliation is made of. Nothing below asserts authorisation:
moto serves an unsigned request, so a test that "proved" a refusal would be testing the
mock (tests/test_storage_objects.py states the same rule).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.main import create_app
from app.models import Artifact, Capture, CaptureFile, Job, JobStep
from app.models.enums import ArtifactKind, CaptureKind, RunStatus, UploadStatus
from app.services import recipes as recipe_service
from app.storage import NullStorage, S3Storage, get_storage

BUCKET = "twin-test"
REGION = "us-east-1"


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


def make_capture(db: Session, storage: S3Storage, slug: str = "back-paddock") -> Capture:
    capture = Capture(slug=slug, name="Back paddock", kind=CaptureKind.GAUSSIAN_SPLAT)
    db.add(capture)
    db.flush()
    key = f"captures/{capture.id}/source/scan.ply"
    storage.put_object(key, b"ply\n", "application/octet-stream")
    db.add(
        CaptureFile(
            capture_id=capture.id,
            filename="scan.ply",
            bytes=4,
            storage_key=key,
            status=UploadStatus.COMPLETE,
        )
    )
    db.commit()
    return capture


def make_run(
    db: Session,
    storage: S3Storage,
    capture: Capture,
    *,
    params: dict[str, Any] | None = None,
    upload: bool = True,
) -> Job:
    """A finished run with one packaged-tileset artifact and one step log."""
    job = Job(
        capture_id=capture.id,
        recipe="splat-ingest",
        recipe_version="2",
        params=params or {},
        status=RunStatus.COMPLETE,
        provider="modal",
        tier="l4",
    )
    db.add(job)
    db.flush()
    step = JobStep(
        job_id=job.id,
        stage_id="package",
        ordinal=2,
        impl="splat_tiles",
        status=RunStatus.COMPLETE,
        log_key=f"runs/{job.id}/package/log.txt",
    )
    db.add(step)
    db.flush()
    key = f"runs/{job.id}/package/splat"
    db.add(
        Artifact(
            job_step_id=step.id,
            kind=ArtifactKind.TILES_3D,
            storage_key=key,
            bytes=175_000,
            checksum="sha256:abc",
            content_type="application/json",
        )
    )
    db.commit()
    if upload:
        # A directory artifact is one row whose key is the prefix its members live under.
        storage.put_object(f"{key}/tileset.json", b"{}", "application/json")
        storage.put_object(f"{key}/splat.glb", b"glb", "model/gltf-binary")
        storage.put_object(step.log_key or "", b"ok\n", "text/plain")
    return job


# --- reconciliation -----------------------------------------------------------


def test_reconciliation_finds_a_deliberate_orphan_and_a_deliberate_absence(
    client: TestClient, db: Session, storage: S3Storage
) -> None:
    """The two failures this view exists for, both staged on purpose.

    The orphan is what a run that uploaded its outputs and died before committing its
    step leaves behind: bytes in the bucket that no row has ever heard of. The missing
    object is the opposite and worse -- a row a site points at, whose object is gone, so
    the globe has a dead tileset and nothing else in the system would notice.
    """
    capture = make_capture(db, storage)
    healthy = make_run(db, storage, capture)

    # Deliberately orphaned: an object under a run id with no job row at all.
    ghost = uuid.uuid4()
    storage.put_object(f"runs/{ghost}/train/point_cloud.ply", b"x" * 64, "application/octet-stream")

    # Deliberately missing: a second run's artifact row, with nothing uploaded for it.
    other = Capture(slug="north-gate", name="North gate", kind=CaptureKind.GAUSSIAN_SPLAT)
    db.add(other)
    db.flush()
    db.commit()
    absent = make_run(db, storage, other, upload=False)

    body = client.get("/api/v1/storage/reconciliation").json()

    orphans = {row["key"] for row in body["orphans"]}
    assert orphans == {f"runs/{ghost}/train/point_cloud.ply"}
    assert body["bytesOrphaned"] == 64
    assert body["orphans"][0]["reason"].startswith(f"under runs/{ghost}")

    missing = sorted((row["kind"], row["key"]) for row in body["missing"])
    assert missing == sorted(
        [
            ("artifact", f"runs/{absent.id}/package/splat"),
            ("step-log", f"runs/{absent.id}/package/log.txt"),
        ]
    )
    # The healthy run is in neither list, and its members are matched rather than counted
    # one by one against the row: three objects, one artifact row.
    assert not any(str(healthy.id) in key for _, key in missing)
    assert not any(str(healthy.id) in key for key in orphans)
    assert body["matched"] == 4  # 1 capture file + 2 tileset members + 1 log
    assert body["truncated"] is False
    assert body["prefixes"] == ["captures/", "runs/"]


def test_reconciliation_does_not_call_an_unfinished_upload_missing(
    client: TestClient, db: Session, storage: S3Storage
) -> None:
    """A multipart upload in flight has a row and no object yet. That is not a fault."""
    capture = Capture(slug="in-flight", name="In flight", kind=CaptureKind.VIDEO)
    db.add(capture)
    db.flush()
    db.add(
        CaptureFile(
            capture_id=capture.id,
            filename="clip.mov",
            storage_key=f"captures/{capture.id}/source/clip.mov",
            status=UploadStatus.IN_PROGRESS,
            upload_id="u-1",
        )
    )
    db.commit()

    body = client.get("/api/v1/storage/reconciliation").json()
    assert body["missing"] == []
    assert body["rowsChecked"] == 0


def test_reconciliation_caps_the_walk_and_says_so(
    client: TestClient, db: Session, storage: S3Storage
) -> None:
    capture = make_capture(db, storage)
    for index in range(5):
        storage.put_object(f"runs/{uuid.uuid4()}/train/{index}.bin", b"x", "application/x-binary")

    body = client.get("/api/v1/storage/reconciliation", params={"maxObjects": 3}).json()
    assert body["scanned"] == 3
    assert body["truncated"] is True
    # The row half is exact regardless: it asks storage per row rather than reading the walk.
    assert body["rowsChecked"] == 1
    assert body["missing"] == []
    assert capture.slug == "back-paddock"


def test_reconciliation_needs_a_bucket(db: Session) -> None:
    """With no object storage configured this is a 503, not an empty all-clear.

    The distinction matters: a console that showed "0 orphans, 0 missing" for a
    deployment it could not read would be reporting a clean bucket it never looked at.
    """
    app = create_app()

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    app.dependency_overrides[get_storage] = NullStorage
    with TestClient(app) as client:
        response = client.get("/api/v1/storage/reconciliation")
    assert response.status_code == 503, response.text


# --- outputs ------------------------------------------------------------------


def test_artifacts_carry_their_run_and_what_references_them(
    client: TestClient, db: Session, storage: S3Storage
) -> None:
    capture = make_capture(db, storage)
    job = make_run(db, storage, capture)

    # A site pointing at the tileset, exactly as `register` writes it.
    site = client.post(
        "/api/v1/sites",
        json={
            "name": "Back paddock",
            "boundary": {
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
            },
            "assets": [
                {
                    "name": "Back paddock splat",
                    "representation": "gaussian-splat",
                    "source": {
                        "type": "3d-tiles-url",
                        "url": storage.public_url(f"runs/{job.id}/package/splat/tileset.json"),
                    },
                }
            ],
        },
    )
    assert site.status_code == 201, site.text

    rows = client.get("/api/v1/artifacts").json()
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "3d-tiles"
    assert row["jobId"] == str(job.id)
    assert row["recipe"] == "splat-ingest"
    assert row["stageId"] == "package"
    assert row["impl"] == "splat_tiles"
    assert row["captureName"] == "Back paddock"
    assert [reference["kind"] for reference in row["references"]] == ["site-asset"]
    assert row["references"][0]["siteSlug"] == site.json()["slug"]

    # The cleanup list: nothing, because the one artifact there is is referenced.
    assert client.get("/api/v1/artifacts", params={"unreferenced": True}).json() == []
    assert len(client.get("/api/v1/artifacts", params={"unreferenced": False}).json()) == 1


def test_unreferenced_artifacts_are_the_cleanup_list(
    client: TestClient, db: Session, storage: S3Storage
) -> None:
    capture = make_capture(db, storage)
    make_run(db, storage, capture)
    rows = client.get("/api/v1/artifacts", params={"unreferenced": True}).json()
    assert [row["storageKey"].split("/")[-1] for row in rows] == ["splat"]
    assert rows[0]["references"] == []


# --- the recipe catalogue, and launching a run through it ---------------------


def test_the_catalogue_is_the_recipe_files(client: TestClient) -> None:
    body = client.get("/api/v1/recipes").json()
    names = {recipe["name"] for recipe in body["recipes"]}
    assert {"splat-ingest", "photo-reconstruct"} <= names

    ingest = next(r for r in body["recipes"] if r["name"] == "splat-ingest")
    package = next(stage for stage in ingest["stages"] if stage["id"] == "package")
    # The defaults an override is merged over come from the file, not from this API.
    assert package["params"]["max_gaussians"] == 400000
    assert package["gpu"] is None

    reconstruct = next(r for r in body["recipes"] if r["name"] == "photo-reconstruct")
    train = next(stage for stage in reconstruct["stages"] if stage["id"] == "train")
    # The only routing signal there is.
    assert train["gpu"] == {"tier": "l4", "preemptible": True}

    assert {provider["name"] for provider in body["providers"]} >= {"modal", "vast"}


def test_launching_a_run_with_an_override_stores_it_on_the_job(
    client: TestClient, db: Session, storage: S3Storage
) -> None:
    """The console's whole reason to exist: run this capture again, differently.

    The override is stored verbatim on `jobs.params`, keyed by stage id, which is what
    `app.worker.params` merges over the recipe's own defaults (`Recipe.with_params`).
    A run whose parameters were not written down is a run that cannot be compared with
    another, which is what Phase B needs this screen for.
    """
    capture = make_capture(db, storage)

    response = client.post(
        f"/api/v1/captures/{capture.id}/process",
        json={
            "recipe": "splat-ingest",
            "params": {"package": {"max_gaussians": 50000, "opacity_min": 0.1}},
            "provider": "runpod-community",
            "tier": "l4",
        },
    )
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["params"] == {"package": {"max_gaussians": 50000, "opacity_min": 0.1}}
    assert job["provider"] == "runpod-community"
    assert job["tier"] == "l4"
    # Stamped from the recipe file, so two runs can be told apart by what they ran.
    assert job["recipeVersion"] == recipe_service.version_of("splat-ingest")

    stored = db.get(Job, uuid.UUID(job["id"]))
    assert stored is not None
    assert stored.params == {"package": {"max_gaussians": 50000, "opacity_min": 0.1}}


def test_an_override_naming_a_stage_the_recipe_lacks_is_refused_in_its_own_words(
    client: TestClient, db: Session, storage: S3Storage
) -> None:
    """A6 built this check; the console surfaces it rather than repeating it.

    The point is not that the request fails. It is that it fails *here*, with the recipe's
    own message naming the stages it does have -- instead of queueing a run that the
    worker refuses minutes later with the same words nobody is watching for.
    """
    capture = make_capture(db, storage)
    response = client.post(
        f"/api/v1/captures/{capture.id}/process",
        json={"recipe": "splat-ingest", "params": {"train": {"iterations": 1}}},
    )
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "train" in detail
    assert "does not have" in detail
    # Names the stages it *does* have, which is what makes the message actionable.
    assert "georeference" in detail and "package" in detail


def test_two_runs_of_one_capture_are_comparable(
    client: TestClient, db: Session, storage: S3Storage
) -> None:
    """The Phase B requirement, at the API: one capture, several runs, all their params.

    B3 compares `mask: none` against `robust` against `imc` over the same windy capture.
    That is this: the runs are listed together, each carrying the full resolved parameter
    set it ran with, so what differs between them is readable rather than remembered.
    """
    capture = make_capture(db, storage)
    first = make_run(db, storage, capture, params={"package": {"max_gaussians": 400000}})
    second = make_run(db, storage, capture, params={"package": {"max_gaussians": 50000}})

    runs = client.get("/api/v1/jobs", params={"captureId": str(capture.id)}).json()
    assert {run["id"] for run in runs} == {str(first.id), str(second.id)}
    by_id = {run["id"]: run for run in runs}
    assert by_id[str(first.id)]["params"]["package"]["max_gaussians"] == 400000
    assert by_id[str(second.id)]["params"]["package"]["max_gaussians"] == 50000
    # Same recipe and version: the only difference between them is the parameter, which
    # is the condition a comparison has to be able to state.
    assert len({run["recipe"] for run in runs}) == 1
    assert len({run["recipeVersion"] for run in runs}) == 1
