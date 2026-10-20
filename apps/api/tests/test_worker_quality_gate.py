"""A failed quality gate rejects the epoch, once (M3b.2; the M3b review's fix 8).

The real supervisor, child process and recipe executor, with `tests/recipes/t-gate.yaml`:
a gate the recipe turned on fails the run on its first attempt, the job says which gate,
and the capture is kept as a `rejected` epoch of the site it was run in -- with a
footprint, and the gate where the API shows it -- or, with no site, as nothing but the
job's error. A later run of the same capture that passes publishes that same epoch.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Capture, Epoch, Job, Site
from app.models.enums import CaptureStatus, RunStatus
from app.services import epochs
from app.storage import S3Storage
from app.worker.runner import JobSupervisor
from tests.test_asset_scale import LAT, LON
from tests.test_registration_epochs import _capture, run
from tests.test_worker import config, queue_job, steps_by_stage
from tests.test_worker_stops import claimed

GATE = "max_check_target_rmse_m"


def _run(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    capture: Capture,
    params: dict[str, dict[str, object]] | None = None,
) -> Job:
    job = queue_job(db, capture, "t-gate")
    job.params = params or {}
    db.commit()
    claimed(sessions, job)
    JobSupervisor(sessions, storage, config(tmp_path, max_attempts=3)).run(job.id)
    db.expire_all()
    done = db.get(Job, job.id)
    assert done is not None
    return done


def _site_with_a_frame(db: Session, storage: S3Storage, tmp_path: Path) -> uuid.UUID:
    first = _capture(db, "first")
    run(db, storage, tmp_path, first)
    assert first.site_id is not None
    return first.site_id


def test_a_failed_gate_is_one_attempt_and_a_rejected_epoch(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    client: TestClient,
) -> None:
    site_id = _site_with_a_frame(db, storage, tmp_path)
    capture = _capture(
        db, "gated", metadata_={"siteId": str(site_id), "lat": LAT + 1e-4, "lon": LON}
    )

    job = _run(db, sessions, storage, tmp_path, capture)

    assert job.status is RunStatus.ERROR
    assert "failed a quality gate the recipe turned on" in (job.error or "")
    assert f"quality gate {GATE}: measured 0.31 m, limit 0.12 m" in (job.error or "")
    assert steps_by_stage(db, job.id)["align"].attempt == 1, "a gate is never retried"
    db.refresh(capture)
    assert capture.status is CaptureStatus.ERROR

    epoch = epochs.epoch_of_capture(db, capture.id)
    assert epoch is not None
    assert (epoch.site_id, epoch.status, epoch.ordinal, epoch.job_id) == (
        site_id,
        "rejected",
        2,
        job.id,
    )
    assert (epoch.join_method, epoch.observed_at_source) == ("explicit", "created")
    assert epoch.alignment is not None
    assert epoch.alignment["verdict"] == "unaligned"
    assert epoch.alignment["rejected"] == {
        "stage": "align",
        "jobId": str(job.id),
        "gate": GATE,
        "measured": 0.31,
        "limit": 0.12,
        "unit": "m",
        "reason": "the check targets' RMSE is over the limit the recipe set",
    }
    # The footprint: the located point buffered by 25 m, about pi * 25^2 square metres.
    area = db.scalar(
        select(func.ST_Area(func.geography(Epoch.footprint))).where(Epoch.id == epoch.id)
    )
    assert area is not None and abs(float(area) - 3.14159 * 25.0**2) < 0.02 * 3.14159 * 25.0**2

    [_one, shown] = client.get(f"/api/v1/sites/{site_id}/epochs").json()
    assert shown["status"] == "rejected" and shown["currentAssetId"] is None
    assert shown["quality"] is None
    assert shown["alignment"]["label"] == f"Capture, rejected: quality gate {GATE} failed"
    assert shown["alignment"]["verdict"] == "unaligned"


def test_a_later_run_that_passes_publishes_the_rejected_epoch(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    client: TestClient,
) -> None:
    site_id = _site_with_a_frame(db, storage, tmp_path)
    capture = _capture(db, "gated", metadata_={"siteId": str(site_id)})
    _run(db, sessions, storage, tmp_path / "a", capture)
    rejected = epochs.epoch_of_capture(db, capture.id)
    assert rejected is not None and rejected.status == "rejected"

    again = _run(db, sessions, storage, tmp_path / "b", capture, {"align": {"fail": False}})

    assert again.status is RunStatus.COMPLETE, again.error
    db.expire_all()
    epoch = epochs.epoch_of_capture(db, capture.id)
    assert epoch is not None
    assert (epoch.id, epoch.status, epoch.job_id) == (rejected.id, "published", again.id)
    # A Lane 1 run has no align stage of its own: unaligned, and saying why.
    assert epoch.alignment is not None and "rejected" not in epoch.alignment
    [_one, shown] = client.get(f"/api/v1/sites/{site_id}/epochs").json()
    assert shown["status"] == "published" and shown["currentAssetId"] is not None


def test_a_rejected_capture_of_no_site_makes_no_site(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db, "nowhere")
    sites_before = db.scalar(select(func.count()).select_from(Site))

    job = _run(db, sessions, storage, tmp_path, capture)

    assert job.status is RunStatus.ERROR and GATE in (job.error or "")
    assert steps_by_stage(db, job.id)["align"].attempt == 1
    assert epochs.epoch_of_capture(db, capture.id) is None
    assert db.scalar(select(func.count()).select_from(Site)) == sites_before


def _rejection() -> epochs.Rejection:
    return epochs.Rejection(stage_id="align", gate={"gate": GATE, "measured": 0.31, "limit": 0.12})


def test_with_no_located_point_the_footprint_is_the_sites_boundary(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    site_id = _site_with_a_frame(db, storage, tmp_path)
    capture = _capture(db, "unlocated", metadata_={"siteId": str(site_id)})
    epoch = epochs.record_rejected(db, uuid.uuid4(), capture, _rejection())
    db.commit()
    assert epoch is not None and epoch.job_id is None  # a run nobody queued is not named
    same = db.scalar(
        select(func.ST_Equals(Epoch.footprint, Site.boundary))
        .join(Site, Site.id == Epoch.site_id)
        .where(Epoch.id == epoch.id)
    )
    assert same is True


def test_a_published_epoch_is_never_demoted_and_a_frameless_site_gets_none(
    db: Session, storage: S3Storage, tmp_path: Path, client: TestClient
) -> None:
    from tests.conftest import site_payload

    site_id = _site_with_a_frame(db, storage, tmp_path)
    [published] = list(db.scalars(select(Epoch).where(Epoch.site_id == site_id)))
    first = db.get(Capture, published.capture_id)
    assert first is not None
    assert epochs.record_rejected(db, uuid.uuid4(), first, _rejection()) is None
    db.commit()
    db.refresh(published)
    assert published.status == "published" and "rejected" not in (published.alignment or {})

    frameless = client.post("/api/v1/sites", json=site_payload(slug="frameless", assets=[]))
    assert frameless.status_code == 201, frameless.text
    capture = _capture(db, "named", metadata_={"siteId": frameless.json()["id"]})
    assert epochs.record_rejected(db, uuid.uuid4(), capture, _rejection()) is None
