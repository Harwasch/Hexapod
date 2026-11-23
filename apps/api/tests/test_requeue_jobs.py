"""Correction units and PostGIS coverage (the coordinator runs the DB cases)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Capture, CaptureFile, Job
from app.models.enums import CaptureKind, RunStatus, UploadStatus
from app.schemas.job import JobCreate
from app.scripts import requeue_jobs
from app.services import capture_bundles, captures, jobs, worker_wake
from app.services.errors import ConflictError, InvalidInputError


def uploaded_bundle() -> Capture:
    return Capture(
        id=uuid.uuid4(),
        slug=f"test-{uuid.uuid4()}",
        name="Phone bundle",
        kind=CaptureKind.VIDEO,
        metadata_={"bundleFormat": capture_bundles.FORMAT},
        files=[
            CaptureFile(
                filename="capture.hexcap",
                content_type=capture_bundles.CONTENT_TYPE,
                bytes=100,
                storage_key=f"captures/{uuid.uuid4()}/source/capture.hexcap",
                status=UploadStatus.COMPLETE,
            )
        ],
    )


def test_fresh_job_uses_bundle_params_and_stable_id(monkeypatch: pytest.MonkeyPatch) -> None:
    db = Mock(spec=Session)
    db.get.return_value = None
    db.scalar.return_value = None
    capture = uploaded_bundle()
    monkeypatch.setattr(capture_bundles, "locked_capture", lambda *_: capture)
    create = Mock(return_value=Job(id=requeue_jobs.job_id(capture.id)))
    monkeypatch.setattr(jobs, "create_job", create)
    assert requeue_jobs.requeue(db, capture.id, apply=False) is None
    create.assert_not_called()
    job = requeue_jobs.requeue(db, capture.id, apply=True)
    assert job is create.return_value
    sent = create.call_args
    assert sent.args[2].recipe == "photo-reconstruct"
    assert sent.args[2].params == capture_bundles.PARAMS
    assert sent.kwargs["job_id"] == requeue_jobs.job_id(capture.id)
    assert requeue_jobs.job_id(uuid.uuid4()) != requeue_jobs.job_id(capture.id)


@pytest.mark.parametrize("status", list(RunStatus))
def test_repeat_returns_same_correction_even_after_completion(
    monkeypatch: pytest.MonkeyPatch, status: RunStatus
) -> None:
    db = Mock(spec=Session)
    capture = uploaded_bundle()
    job = Job(id=requeue_jobs.job_id(capture.id), status=status)
    db.get.return_value = job
    monkeypatch.setattr(capture_bundles, "locked_capture", lambda *_: capture)
    create = Mock()
    monkeypatch.setattr(jobs, "create_job", create)
    assert requeue_jobs.requeue(db, capture.id, apply=True) is job
    create.assert_not_called()


@pytest.mark.parametrize("bad", ["kind", "format", "incomplete", "filename", "content-type", "two"])
def test_refuses_non_bundles_or_incomplete_archives(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    db = Mock(spec=Session)
    capture = uploaded_bundle()
    if bad == "kind":
        capture.kind = CaptureKind.IMAGES
    elif bad == "format":
        capture.metadata_ = {}
    elif bad == "incomplete":
        capture.files[0].status = UploadStatus.IN_PROGRESS
    elif bad == "filename":
        capture.files[0].filename = "video.mov"
    elif bad == "content-type":
        capture.files[0].content_type = "video/quicktime"
    else:
        capture.files.append(CaptureFile())
    monkeypatch.setattr(captures, "get_capture", lambda *_: capture)
    create = Mock()
    monkeypatch.setattr(jobs, "create_job", create)
    with pytest.raises((InvalidInputError, ConflictError)):
        requeue_jobs.requeue(db, capture.id, apply=True)
    create.assert_not_called()


def test_active_job_blocks_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    db = Mock(spec=Session)
    db.get.return_value = None
    db.scalar.return_value = uuid.uuid4()
    capture = uploaded_bundle()
    monkeypatch.setattr(capture_bundles, "locked_capture", lambda *_: capture)
    with pytest.raises(ConflictError, match="queued or running"):
        requeue_jobs.requeue(db, capture.id, apply=False)


def test_db_fresh_run_leaves_failed_history_and_is_durably_idempotent(db: Session) -> None:
    capture = uploaded_bundle()
    db.add(capture)
    db.commit()
    old = jobs.create_job(
        db,
        capture.id,
        JobCreate(recipe="photo-reconstruct", params=capture_bundles.PARAMS),
    )
    old.status = RunStatus.ERROR
    old.error = "normalize exhausted 3 attempts"
    db.commit()
    assert requeue_jobs.requeue(db, capture.id, apply=False) is None
    assert db.scalar(select(func.count()).select_from(Job)) == 1
    fresh = requeue_jobs.requeue(db, capture.id, apply=True)
    assert fresh is not None and fresh.id != old.id
    assert fresh.params == capture_bundles.PARAMS
    assert fresh.steps == [] and fresh.status == RunStatus.NOT_STARTED
    for status in (RunStatus.NOT_STARTED, RunStatus.COMPLETE, RunStatus.ERROR):
        fresh.status = status
        db.commit()
        db.expire_all()
        repeated = requeue_jobs.requeue(db, capture.id, apply=True)
        assert repeated is not None and repeated.id == fresh.id
        assert db.scalar(select(func.count()).select_from(Job)) == 2
    db.refresh(old)
    assert old.status == RunStatus.ERROR and old.error == "normalize exhausted 3 attempts"


def test_db_other_active_job_blocks_apply(db: Session) -> None:
    capture = uploaded_bundle()
    db.add(capture)
    db.commit()
    jobs.create_job(
        db,
        capture.id,
        JobCreate(recipe="photo-reconstruct", params=capture_bundles.PARAMS),
    )
    with pytest.raises(ConflictError):
        requeue_jobs.requeue(db, capture.id, apply=True)
    assert db.scalar(select(func.count()).select_from(Job)) == 1


@pytest.mark.parametrize(
    "apply,status", [(False, None), (True, RunStatus.NOT_STARTED), (True, RunStatus.COMPLETE)]
)
def test_cli_deduplicates_ids_and_wakes_only_after_commit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    apply: bool,
    status: RunStatus | None,
) -> None:
    from contextlib import contextmanager

    from app import db as database

    capture_id = uuid.uuid4()
    events: list[str] = []

    @contextmanager
    def session() -> Iterator[Session]:
        events.append("open")
        yield Mock(spec=Session)
        events.append("closed")

    def queue(db: Session, identifier: uuid.UUID, *, apply: bool) -> Job | None:
        assert identifier == capture_id
        events.append("committed" if apply else "dry")
        return (
            Job(id=requeue_jobs.job_id(capture_id), status=status) if status is not None else None
        )

    monkeypatch.setattr(database, "get_session_factory", lambda: session)
    monkeypatch.setattr(requeue_jobs, "requeue", queue)
    monkeypatch.setattr(worker_wake, "after_enqueue", lambda: events.append("wake"))
    args = ["--capture", str(capture_id), "--capture", str(capture_id)]
    requeue_jobs.main([*args, *(["--apply"] if apply else [])])
    expected = ["open", "committed" if apply else "dry", "closed"]
    if apply and status == RunStatus.NOT_STARTED:
        expected.append("wake")
    assert events == expected
    output = capsys.readouterr().out
    assert output.count(f"capture {capture_id}") == 1
    assert output.endswith("REQUEUE-JOBS-OK\n")


def test_db_concurrent_corrections_create_one_job(
    db: Session, sessions: sessionmaker[Session]
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with sessions() as db:
        capture = uploaded_bundle()
        db.add(capture)
        db.commit()
        identifier = capture.id

    def queue(_: int) -> uuid.UUID:
        with sessions() as db:
            job = requeue_jobs.requeue(db, identifier, apply=True)
            assert job is not None
            return job.id

    with ThreadPoolExecutor(max_workers=2) as pool:
        identifiers = list(pool.map(queue, range(2)))
    assert identifiers == [requeue_jobs.job_id(identifier)] * 2
    with sessions() as db:
        assert db.scalar(select(func.count()).select_from(Job)) == 1
