"""What the worker does about work nobody is watching: the 2026-10 review's findings.

`tests/test_worker_stops.py` drove the audit's answers for a recipe process and its GPU
call when the worker stops in the ordinary ways -- a deploy, a cancel, a crash, a lost
workdir. The review found the cracks between them, and each test here was written to
fail before its fix:

* a job **cancelled between two attempts** is closed out as a cancel, not dropped as
  "lost";
* an **exception in the supervisor's own loop** stops the recipe process before it goes
  anywhere -- a cancel when the job is about to be dead-lettered, a detach when somebody
  else will resume it -- instead of leaving it running beside the next job;
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session, sessionmaker

from app.models import Job, JobStep
from app.models.enums import RunStatus
from app.services import jobs as job_service
from app.storage import S3Storage
from app.worker.runner import JobSupervisor
from tests.test_worker import config, make_capture, queue_job, steps_by_stage, wait_until
from tests.test_worker_stops import Pings, claimed, in_background, recipe_process, watched

# --------------------------------------------------------------------------------------
# A cancel that lands while the worker waits to try a stage again
# --------------------------------------------------------------------------------------


def test_a_job_cancelled_during_the_retry_backoff_is_closed_out_as_cancelled(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """The stage failed, and the worker is waiting `retry_backoff_s` before the next
    attempt when the job is cancelled. The heartbeat after the wait reads CANCELLED, and
    that is a cancel to finish -- the lease cleared, the check told -- not a job "lost"
    to nobody, which skipped all of it and left the dead-man's switch to alert."""
    job = queue_job(db, make_capture(db), "t-doomed")
    claimed(sessions, job)
    pings = Pings()
    cfg = config(
        tmp_path, heartbeat_url="https://hc.example/run", max_attempts=3, retry_backoff_s=3.0
    )
    thread, result = in_background(watched(sessions, storage, cfg, pings), job.id)

    def failed_once() -> bool:
        row = steps_by_stage(db, job.id).get("two")
        return row is not None and row.status is RunStatus.ERROR and row.attempt == 1

    assert wait_until(failed_once), "the stage never failed"
    job_service.cancel_job(db, job.id)
    thread.join(timeout=20)

    assert result == ["cancelled"]
    db.expire_all()
    cancelled = db.get(Job, job.id)
    assert cancelled is not None and cancelled.status is RunStatus.CANCELLED
    assert cancelled.lease_expires_at is None
    assert steps_by_stage(db, job.id)["two"].attempt == 1, "no second attempt was started"
    assert wait_until(lambda: "https://hc.example/run" in pings.urls(), timeout=5)
    assert "https://hc.example/run/fail" not in pings.urls()


def test_a_stop_during_the_retry_backoff_lets_go_of_the_job_at_once(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    job = queue_job(db, make_capture(db), "t-doomed")
    claimed(sessions, job)
    stop = threading.Event()
    cfg = config(tmp_path, max_attempts=3, retry_backoff_s=30.0)
    thread, result = in_background(JobSupervisor(sessions, storage, cfg), job.id, stop)
    assert wait_until(lambda: steps_by_stage(db, job.id).get("two") is not None)
    assert wait_until(lambda: steps_by_stage(db, job.id)["two"].status is RunStatus.ERROR)

    started = time.monotonic()
    stop.set()
    thread.join(timeout=10)

    assert result == ["lost"] and time.monotonic() - started < 5
    db.expire_all()
    handed_back = db.get(Job, job.id)
    assert handed_back is not None and handed_back.status is RunStatus.IN_PROGRESS
    assert handed_back.claimed_by is None and handed_back.lease_expires_at is None


# --------------------------------------------------------------------------------------
# An exception in the supervisor's own loop
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("taken", "signal_heard", "outcome"),
    [
        # Still this worker's: the failure dead-letters the job, so the call goes too.
        (False, "CancelRequested", "error"),
        # Reclaimed meanwhile: the new holder resumes it, so the call is left for it.
        (True, "DetachRequested", "lost"),
    ],
)
def test_an_exception_in_the_watch_loop_stops_the_recipe_process_before_it_propagates(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    taken: bool,
    signal_heard: str,
    outcome: str,
) -> None:
    """A database error in the heartbeat's own writes (here `_report_progress`) used to
    escape to `_report_supervisor_failure` with the recipe process still running: the job
    was dead-lettered and its workdir tidied under a live process, its GPU call kept
    billing, and the slot went on to claim another job beside the orphan."""
    job = queue_job(db, make_capture(db), "t-notes-stop")
    claimed(sessions, job)

    def breaks(_db: Session, step: JobStep, _workdir: Path) -> None:
        if step.stage_id != "two":
            return
        if taken:
            with sessions() as other:
                other.execute(update(Job).where(Job.id == job.id).values(claimed_by="worker-b"))
                other.commit()
        raise RuntimeError("server closed the connection unexpectedly")

    monkeypatch.setattr(JobSupervisor, "_report_progress", staticmethod(breaks))

    result = JobSupervisor(sessions, storage, config(tmp_path)).run(job.id)

    assert result == outcome
    assert recipe_process() is None, "the recipe process outlived its supervisor"
    said = tmp_path / "runs" / str(job.id) / "stages" / "two" / "work" / "stopped-by.txt"
    assert said.read_text() == signal_heard
