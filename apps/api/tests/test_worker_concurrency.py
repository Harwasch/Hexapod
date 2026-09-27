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
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models import Job
from app.models.enums import RunStatus
from app.services import jobs as job_service
from app.storage import S3Storage
from app.worker.claim import claim_next
from app.worker.config import WorkerConfig
from app.worker.loop import Worker
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
