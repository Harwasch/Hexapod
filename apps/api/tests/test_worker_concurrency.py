"""One worker process supervising two jobs at once (`WORKER_CONCURRENCY`).

The same real machinery as `test_worker.py` -- real recipe processes, real claims,
leases and heartbeats against Postgres -- with two slots. What is asked is what has to
stay true per job when jobs share a process: each is claimed once and by its own slot,
each lease is kept alive by its own heartbeat, a cancel stops only its own job, and a
stop hands every job back.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models import Capture, CaptureFile, Job
from app.models.enums import RunStatus, UploadStatus
from app.services import jobs as job_service
from app.storage import S3Storage
from app.worker import loop as worker_loop
from app.worker.claim import claim_next
from app.worker.config import WorkerConfig
from app.worker.loop import Worker
from app.worker.runner import JobSupervisor
from tests.test_worker import (
    FAST_LEASE_S,
    config,
    make_capture,
    queue_job,
    steps_by_stage,
    wait_until,
)


def _running_two(db: Session, job: Job) -> bool:
    step = steps_by_stage(db, job.id).get("two")
    return step is not None and step.status is RunStatus.IN_PROGRESS


def _job(db: Session, job: Job) -> Job:
    db.expire_all()
    found = db.get(Job, job.id)
    assert found is not None
    return found


def test_the_setting_reaches_the_worker_and_one_slot_keeps_its_plain_id(tmp_path: Path) -> None:
    settings = Settings(worker_concurrency=3, worker_workdir=str(tmp_path))
    resolved = WorkerConfig.from_settings(settings)

    assert resolved.concurrency == 3
    assert resolved.for_slot(1).worker_id == f"{resolved.worker_id}/1"
    single = config(tmp_path)
    assert single.for_slot(0) is single


def test_two_slots_run_two_jobs_at_once_each_with_its_own_claim_and_lease(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    capture = make_capture(db)
    first = queue_job(db, capture, "t-slow")
    second = queue_job(db, capture, "t-slow")
    worker = Worker(sessions, storage, config(tmp_path, concurrency=2))
    stop = threading.Event()
    ran: list[int] = []
    loop = threading.Thread(target=lambda: ran.append(worker.run_forever(stop=stop)), daemon=True)
    loop.start()
    try:
        # Both 30-second stages running at the same time: two slots, not a queue.
        assert wait_until(lambda: _running_two(db, first) and _running_two(db, second))
        owners = {_job(db, first).claimed_by, _job(db, second).claimed_by}
        assert owners == {"worker-a/0", "worker-a/1"}

        # Each lease is kept alive by its own slot's heartbeat, well past its length.
        leases = {job.id: _job(db, job).lease_expires_at for job in (first, second)}
        time.sleep(FAST_LEASE_S * 2)
        for job in (first, second):
            renewed = _job(db, job)
            assert renewed.status is RunStatus.IN_PROGRESS
            assert renewed.lease_expires_at is not None
            assert renewed.lease_expires_at > leases[job.id]  # type: ignore[operator]
        # And neither is claimable by anybody else while it is held.
        other = sessions()
        assert claim_next(other, worker_id="worker-b", lease_s=30) is None
        other.close()

        # Cancelling one stops that one, and only that one.
        job_service.cancel_job(db, first.id)
        assert wait_until(lambda: _job(db, first).lease_expires_at is None)
        assert steps_by_stage(db, first.id)["two"].status is RunStatus.CANCELLED
        assert _running_two(db, second)
        assert _job(db, second).status is RunStatus.IN_PROGRESS
    finally:
        # A stop hands the job still running back, lease cleared, for the next worker.
        stop.set()
        loop.join(timeout=30)
    assert not loop.is_alive()
    handed_back = _job(db, second)
    assert handed_back.status is RunStatus.IN_PROGRESS
    assert handed_back.claimed_by is None and handed_back.lease_expires_at is None
    assert _job(db, first).status is RunStatus.CANCELLED
    # Both slots' supervisions ended (one "cancelled", one "lost" -- handed back), which
    # is what the loop counts, as a one-slot worker always has.
    assert ran == [2]


def test_two_slots_finish_the_queue_and_max_jobs_counts_across_them(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    capture = make_capture(db)
    jobs = [queue_job(db, capture, "t-three") for _ in range(3)]
    worker = Worker(sessions, storage, config(tmp_path, concurrency=2))

    assert worker.run_forever(max_jobs=2) == 2

    statuses = sorted(_job(db, job).status.value for job in jobs)
    # Two ran -- not three, although two slots were free when the second finished.
    assert statuses == ["complete", "complete", "not-started"]
    assert worker.run_forever(max_jobs=1) == 1
    assert {_job(db, job).status for job in jobs} == {RunStatus.COMPLETE}


def test_an_idle_slot_never_takes_a_job_another_slot_is_running(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """One job, two slots: the idle slot polls the queue the whole time the busy one
    works, and must never reclaim it (production dead-lettered jobs this way)."""
    capture = make_capture(db)
    only = queue_job(db, capture, "t-slow")
    worker = Worker(sessions, storage, config(tmp_path, concurrency=2))
    stop = threading.Event()
    loop = threading.Thread(target=lambda: worker.run_forever(stop=stop), daemon=True)
    loop.start()
    try:
        assert wait_until(lambda: _running_two(db, only))
        owner = _job(db, only).claimed_by
        time.sleep(FAST_LEASE_S * 3)
        after = _job(db, only)
        assert after.status is RunStatus.IN_PROGRESS, after.error
        assert after.claimed_by == owner
        assert steps_by_stage(db, only.id)["one"].attempt == 1
    finally:
        stop.set()
        loop.join(timeout=30)


# --------------------------------------------------------------------------------------
# The supervisor's own I/O must not let its lease lapse (the 2026-09-27 dead-letters)
# --------------------------------------------------------------------------------------
#
# A supervisor does blocking work of its own between heartbeats: it downloads the capture
# before the first stage (`_seed`), and uploads every artifact and the log of a stage that
# finished before it writes that step's row (`_apply`) -- in production, normalize's ~100
# frames to R2, one `put_object` at a time. The test stages above upload a few bytes to
# moto, so none of that ever crossed a lease. These make it slow, as R2 from Fly is.

#: Longer than the lease, per object: a stage's artifact plus its log keep the supervisor
#: away from its heartbeat for more than twice `FAST_LEASE_S`, which an idle slot polling
#: every `idle_s` sees as a lapsed lease -- unless something else is renewing it.
SLOW_TRANSFER_S = FAST_LEASE_S * 1.2


@pytest.fixture
def claims(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[str, uuid.UUID]]]:
    """Every successful claim the worker's slots make, as (slot id, job id)."""
    made: list[tuple[str, uuid.UUID]] = []
    guard = threading.Lock()

    def recording(db: Session, *, worker_id: str, lease_s: float) -> Job | None:
        job = claim_next(db, worker_id=worker_id, lease_s=lease_s)
        if job is not None:
            with guard:
                made.append((worker_id, job.id))
        return job

    # Where the loop looks it up: `app.worker.loop` imported the name.
    monkeypatch.setattr(worker_loop, "claim_next", recording)
    yield made


def _slow_uploads(monkeypatch: pytest.MonkeyPatch, storage: S3Storage) -> None:
    real = storage.put_object

    def slow(key: str, data: bytes, content_type: str) -> object:
        if key.startswith("runs/"):
            time.sleep(SLOW_TRANSFER_S)
        return real(key, data, content_type)

    monkeypatch.setattr(storage, "put_object", slow)


def _run_until_settled(worker: Worker, db: Session, jobs: list[Job], timeout: float) -> None:
    """Run every slot until each job has a terminal status, then stop the worker."""
    stop = threading.Event()
    loop = threading.Thread(target=lambda: worker.run_forever(stop=stop), daemon=True)
    loop.start()
    try:
        wait_until(
            lambda: all(
                _job(db, job).status in (RunStatus.COMPLETE, RunStatus.ERROR) for job in jobs
            ),
            timeout=timeout,
        )
    finally:
        stop.set()
        loop.join(timeout=30)
    assert not loop.is_alive()


def _ran_once_and_finished(
    db: Session, job: Job, claims: list[tuple[str, uuid.UUID]], stages: set[str]
) -> None:
    owners = [slot for slot, claimed in claims if claimed == job.id]
    finished = _job(db, job)
    assert finished.status is RunStatus.COMPLETE, finished.error
    # One claim: never taken by the other slot while its own was still working on it.
    assert len(owners) == 1, f"job claimed {len(owners)} times, by {owners}"
    assert finished.claimed_by == owners[0]
    assert finished.lease_expires_at is None
    rows = steps_by_stage(db, job.id)
    assert set(rows) == stages
    assert {stage: row.attempt for stage, row in rows.items()} == dict.fromkeys(stages, 1)


def test_an_idle_slot_does_not_reclaim_a_job_whose_supervisor_is_uploading(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    claims: list[tuple[str, uuid.UUID]],
) -> None:
    """The production failure, reproduced: the busy slot uploads a finished stage's
    outputs for longer than the lease, the idle slot takes the job, the stage runs again
    as attempt 2 -- and so on, until "attempted 3 times"."""
    _slow_uploads(monkeypatch, storage)
    capture = make_capture(db)
    only = queue_job(db, capture, "t-three")
    worker = Worker(sessions, storage, config(tmp_path, concurrency=2))

    _run_until_settled(worker, db, [only], timeout=60)

    _ran_once_and_finished(db, only, claims, {"one", "two", "three"})
    # Nothing is left renewing a lease once the job is over.
    assert not [t for t in threading.enumerate() if t.name.startswith("lease-")]


def test_two_slots_with_slow_uploads_run_two_jobs_once_each(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    claims: list[tuple[str, uuid.UUID]],
) -> None:
    """Two jobs, two slots, every stage's uploads crossing the lease: the slot whose job
    finishes first goes idle and polls while the other is still uploading."""
    _slow_uploads(monkeypatch, storage)
    capture = make_capture(db)
    jobs = [queue_job(db, capture, "t-three"), queue_job(db, capture, "t-two")]
    worker = Worker(sessions, storage, config(tmp_path, concurrency=2))

    _run_until_settled(worker, db, jobs, timeout=90)

    _ran_once_and_finished(db, jobs[0], claims, {"one", "two", "three"})
    _ran_once_and_finished(db, jobs[1], claims, {"one", "two"})
    assert {slot for slot, _ in claims} == {"worker-a/0", "worker-a/1"}


def test_an_idle_slot_does_not_reclaim_a_job_whose_capture_is_downloading(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    claims: list[tuple[str, uuid.UUID]],
) -> None:
    """The same before the first stage: fetching the capture's upload into the workdir
    (a video of gigabytes, in production) happens between the claim and the first
    heartbeat."""
    capture = make_capture(db)
    _upload(db, storage, capture, "IMG_0001.MOV", b"not really a video")
    real = storage.download_file

    def slow(key: str, target: Path) -> int:
        time.sleep(FAST_LEASE_S * 2.5)
        return real(key, target)

    monkeypatch.setattr(storage, "download_file", slow)
    only = queue_job(db, capture, "t-upload")
    worker = Worker(sessions, storage, config(tmp_path, concurrency=2))

    _run_until_settled(worker, db, [only], timeout=60)

    _ran_once_and_finished(db, only, claims, {"one"})
    seeded = tmp_path / "runs" / str(only.id) / "inputs" / "upload" / "IMG_0001.MOV"
    assert seeded.read_bytes() == b"not really a video"


def _upload(db: Session, storage: S3Storage, capture: Capture, name: str, data: bytes) -> None:
    key = f"captures/{capture.id}/{name}"
    storage.put_object(key, data, "application/octet-stream")
    db.add(
        CaptureFile(
            capture_id=capture.id,
            filename=name,
            storage_key=key,
            bytes=len(data),
            status=UploadStatus.COMPLETE,
        )
    )
    db.commit()


def test_a_resumed_job_fetches_the_files_an_interrupted_download_did_not(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """A worker that stopped mid-download left `inputs/upload/` neither empty nor
    complete: one file in place, the next as boto's temporary beside it. The job's next
    worker used to see a non-empty directory and run the recipe on what was there."""
    capture = make_capture(db)
    _upload(db, storage, capture, "part-1.MOV", b"first")
    _upload(db, storage, capture, "part-2.MOV", b"second")
    job = queue_job(db, capture, "t-upload")
    inputs = tmp_path / "runs" / str(job.id) / "inputs" / "upload"
    inputs.mkdir(parents=True)
    (inputs / "part-1.MOV").write_bytes(b"first")
    (inputs / "part-2.MOV.3fa1c2d0").write_bytes(b"sec")
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=30) is not None
    session.close()

    assert JobSupervisor(sessions, storage, config(tmp_path)).run(job.id) == "complete"

    assert (inputs / "part-1.MOV").read_bytes() == b"first"
    assert (inputs / "part-2.MOV").read_bytes() == b"second"
