"""The capture and job spine: rows, cascades, the slug constraint, and lease reclaim."""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    Artifact,
    ArtifactKind,
    Capture,
    CaptureFile,
    CaptureKind,
    CaptureStatus,
    GeorefMethod,
    Job,
    JobStep,
    RunStatus,
    ScaleSource,
    Site,
    UploadStatus,
)
from app.models.base import utcnow
from app.worker.claim import claimable
from tests.conftest import site_payload


def make_capture(db: Session, slug: str = "back-paddock", **overrides: object) -> Capture:
    capture = Capture(slug=slug, name="Back paddock", kind=CaptureKind.VIDEO, **overrides)
    db.add(capture)
    db.commit()
    return capture


def make_run(db: Session, capture: Capture) -> tuple[Job, JobStep, Artifact]:
    job = Job(
        capture_id=capture.id,
        recipe="photo-reconstruct",
        recipe_version="1.2.0",
        params={"train": {"iterations": 30000}, "normalize": {"fps": 4}},
        provider="modal",
        tier="l4",
    )
    db.add(job)
    db.flush()
    step = JobStep(job_id=job.id, stage_id="train", ordinal=3, impl="gsplat")
    db.add(step)
    db.flush()
    artifact = Artifact(
        job_step_id=step.id,
        kind=ArtifactKind.SPLAT,
        storage_key="captures/back-paddock/jobs/1/splat/point_cloud.ply",
        bytes=18_000_000_000,
        checksum="sha256:abc",
        content_type="application/octet-stream",
    )
    db.add(artifact)
    db.commit()
    return job, step, artifact


def test_capture_defaults_and_full_row(db: Session) -> None:
    capture = make_capture(
        db,
        description="Two orbits at dusk",
        device="iPhone 15 Pro",
        sensor="lidar",
        captured_at=utcnow(),
        temporal_extent={"start": "2026-09-01T00:00:00Z", "end": "2026-09-01T00:04:00Z"},
        georef_method=GeorefMethod.ARKIT,
        scale_source=ScaleSource.ARKIT,
        uncertainty_m=0.35,
        metadata_={"notes": "windy"},
        attribution=[{"text": "H. Waschura"}],
        license={"name": "CC BY 4.0", "spdxId": "CC-BY-4.0"},
        provenance={"sourceOrganization": "Field team"},
    )
    db.refresh(capture)

    assert capture.status is CaptureStatus.AWAITING_FILES
    assert capture.site_id is None
    assert capture.metadata_ == {"notes": "windy"}
    assert capture.uncertainty_m == 0.35
    assert capture.georef_method is GeorefMethod.ARKIT
    assert capture.scale_source is ScaleSource.ARKIT
    assert capture.created_at is not None and capture.updated_at is not None
    assert capture.files == [] and capture.jobs == []


def test_capture_file_tracks_multipart_state(db: Session) -> None:
    capture = make_capture(db)
    # A0's sizing: 12 GB at 8 MiB parts is 1536 parts, counted rather than rowed.
    file = CaptureFile(
        capture_id=capture.id,
        filename="IMG_0042.MOV",
        content_type="video/quicktime",
        bytes=12_884_901_888,
        storage_key="captures/back-paddock/source/IMG_0042.MOV",
        upload_id="2~abcdef",
        parts_total=1536,
        parts_completed=1200,
        status=UploadStatus.IN_PROGRESS,
    )
    db.add(file)
    db.commit()
    db.refresh(file)

    assert file.bytes == 12_884_901_888
    assert file.parts_total == 1536 and file.parts_completed == 1200
    assert db.scalar(select(func.count()).select_from(CaptureFile)) == 1

    default_status = CaptureFile(
        capture_id=capture.id,
        filename="IMG_0043.MOV",
        storage_key="captures/back-paddock/source/IMG_0043.MOV",
    )
    db.add(default_status)
    db.commit()
    db.refresh(default_status)
    assert default_status.status is UploadStatus.NOT_STARTED
    assert default_status.parts_completed == 0


def test_job_step_and_artifact_defaults(db: Session) -> None:
    capture = make_capture(db)
    job, step, artifact = make_run(db, capture)
    db.refresh(job)
    db.refresh(step)

    assert job.status is RunStatus.NOT_STARTED
    assert job.claimed_by is None and job.lease_expires_at is None
    assert job.params["train"]["iterations"] == 30000
    assert step.status is RunStatus.NOT_STARTED
    assert step.attempt == 1
    assert step.metrics == {}
    assert step.preempted_at is None and step.checkpoint_key is None
    assert job.steps == [step]
    assert step.artifacts == [artifact]
    assert capture.jobs == [job]


def test_capture_delete_cascades_to_files_jobs_steps_and_artifacts(db: Session) -> None:
    capture = make_capture(db)
    db.add(
        CaptureFile(
            capture_id=capture.id,
            filename="scan.ply",
            storage_key="captures/back-paddock/source/scan.ply",
        )
    )
    db.commit()
    make_run(db, capture)

    db.delete(capture)
    db.commit()

    for model in (Capture, CaptureFile, Job, JobStep, Artifact):
        assert db.scalar(select(func.count()).select_from(model)) == 0


def test_job_delete_cascades_to_steps_and_artifacts_but_keeps_capture(db: Session) -> None:
    capture = make_capture(db)
    job, _step, _artifact = make_run(db, capture)

    db.delete(job)
    db.commit()

    assert db.scalar(select(func.count()).select_from(Job)) == 0
    assert db.scalar(select(func.count()).select_from(JobStep)) == 0
    assert db.scalar(select(func.count()).select_from(Artifact)) == 0
    assert db.scalar(select(func.count()).select_from(Capture)) == 1


def test_capture_slug_is_unique(db: Session) -> None:
    make_capture(db)
    db.add(Capture(slug="back-paddock", name="Duplicate", kind=CaptureKind.IMAGES))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_job_step_ordinal_is_unique_per_job(db: Session) -> None:
    capture = make_capture(db)
    job, _step, _artifact = make_run(db, capture)
    db.add(JobStep(job_id=job.id, stage_id="train-again", ordinal=3, impl="gsplat"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_deleting_a_site_clears_the_capture_link(db: Session, client: TestClient) -> None:
    created = client.post("/api/v1/sites", json=site_payload())
    assert created.status_code == 201, created.text
    site = db.scalar(select(Site))
    assert site is not None
    capture = make_capture(db, site_id=site.id)

    db.delete(site)
    db.commit()
    db.refresh(capture)

    # A capture outlives the site it became: SET NULL, not CASCADE.
    assert capture.site_id is None
    assert db.scalar(select(func.count()).select_from(Capture)) == 1


def test_lease_expiry_query_selects_unstarted_and_abandoned_jobs(db: Session) -> None:
    """The claim loop's actual predicate: a job is claimable when it has not started, or
    when whoever claimed it has let the lease lapse. A0 #2 -- the claim is a committed
    lease, so a frozen worker's rows come back after the lease, not after a ~2 h 51 min
    TCP keepalive timeout.

    A7 made this the worker's own predicate rather than a second copy of it: the query
    below is `app.worker.claim.claimable`, so a change to the worker's idea of claimable
    that disagrees with this test fails here."""
    capture = make_capture(db)
    now = utcnow()

    def job(suffix: str, **fields: object) -> Job:
        row = Job(
            capture_id=capture.id, recipe=f"recipe-{suffix}", recipe_version="1.0.0", **fields
        )
        db.add(row)
        return row

    unstarted = job("unstarted")
    abandoned = job(
        "abandoned",
        status=RunStatus.IN_PROGRESS,
        claimed_by="worker-frozen",
        claimed_at=now - timedelta(seconds=30),
        lease_expires_at=now - timedelta(seconds=3),
    )
    alive = job(
        "alive",
        status=RunStatus.IN_PROGRESS,
        claimed_by="worker-healthy",
        claimed_at=now - timedelta(seconds=1),
        lease_expires_at=now + timedelta(seconds=3),
    )
    # A worker that shut down mid-run lets go of the lease rather than leaving it to
    # lapse. `NULL < now()` is NULL, so this row is only claimable because `claimable`
    # names the case; without it the job would be stranded in-progress forever.
    handed_back = job("handed-back", status=RunStatus.IN_PROGRESS, claimed_at=now)
    finished = job("finished", status=RunStatus.COMPLETE, duration_s=41.5)
    cancelled = job("cancelled", status=RunStatus.CANCELLED)
    db.commit()

    rows = db.scalars(select(Job).where(claimable(now)).order_by(Job.created_at)).all()

    assert {row.id for row in rows} == {unstarted.id, abandoned.id, handed_back.id}
    assert alive.id not in {row.id for row in rows}
    assert finished.id not in {row.id for row in rows}
    assert cancelled.id not in {row.id for row in rows}
