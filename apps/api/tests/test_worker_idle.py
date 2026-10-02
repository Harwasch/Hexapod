"""An idle worker polls less, and then goes away -- without stranding a job.

Polling every 2 s forever kept Neon's compute awake around the clock (it needs five idle
minutes to scale to zero) and the worker machine running for nothing. So the poll backs
off, and after `idle_exit_s` with nothing running the worker exits 0 and its machine
stops; the API starts it again when it queues a job (`tests/test_worker_wake.py`).

The thing that must not happen is the job that arrives as the worker leaves. These run the
real loop against Postgres, with real recipe processes, at test-sized timings.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models import Job
from app.models.enums import RunStatus
from app.services import jobs as job_service
from app.storage import S3Storage
from app.worker import loop as worker_loop
from app.worker.claim import anything_claimable
from app.worker.config import WorkerConfig
from app.worker.loop import Worker, _Idle, poll_delay
from tests.conftest import TEST_DATABASE_URL
from tests.test_worker import API_ROOT, RECIPES, config, make_capture, queue_job, wait_until

#: Idle timings for the loop tests: exit after well under a second of nothing.
IDLE: dict[str, object] = {
    "idle_s": 0.05,
    "idle_backoff_after_s": 0.2,
    "idle_max_s": 0.2,
    "idle_exit_s": 0.6,
}


def _status(db: Session, job: Job) -> RunStatus:
    db.expire_all()
    found = db.get(Job, job.id)
    assert found is not None
    return found.status


def _run(worker: Worker, stop: threading.Event | None = None) -> tuple[threading.Thread, list[int]]:
    ran: list[int] = []
    thread = threading.Thread(target=lambda: ran.append(worker.run_forever(stop=stop)), daemon=True)
    thread.start()
    return thread, ran


# --- the schedule -------------------------------------------------------------------


def test_the_poll_backs_off_after_a_minute_and_stops_at_thirty_seconds() -> None:
    def delay(idle_for: float) -> float:
        return poll_delay(idle_for, idle_s=2.0, backoff_after_s=60.0, max_s=30.0)

    assert [delay(t) for t in (0, 30, 59.9)] == [2.0, 2.0, 2.0]
    assert [delay(t) for t in (60, 119, 120, 180, 240)] == [4.0, 4.0, 8.0, 16.0, 30.0]
    # A worker idle for a month does not overflow anything.
    assert delay(30 * 86_400) == 30.0
    # A ceiling at or below the base delay means no backoff at all.
    assert poll_delay(600, idle_s=2.0, backoff_after_s=60.0, max_s=2.0) == 2.0


def test_the_settings_reach_the_worker_and_a_checkout_default_is_documented(
    tmp_path: Path,
) -> None:
    resolved = WorkerConfig.from_settings(Settings(worker_workdir=str(tmp_path)))
    assert resolved.idle_exit_s == 900.0
    assert (resolved.idle_backoff_after_s, resolved.idle_max_s) == (60.0, 30.0)
    # A worker built directly -- every test, any embedding -- polls forever.
    assert config(tmp_path).idle_exit_s == 0.0
    example = (API_ROOT.parents[1] / ".env.example").read_text()
    assert "WORKER_IDLE_EXIT_S=0" in example


def test_the_idle_clock_counts_from_the_last_job_and_never_while_one_runs() -> None:
    now = [100.0]
    idle = _Idle(clock=lambda: now[0])
    empty = lambda: False  # noqa: E731

    now[0] = 200.0
    assert idle.idle_for() == 100.0
    assert idle.enter()
    assert idle.idle_for() == 0.0
    # Busy: however long, no exit.
    now[0] = 10_000.0
    assert not idle.should_exit(60, empty)
    idle.leave(ran=True)
    # The clock restarted when the job ended.
    assert not idle.should_exit(60, empty)
    now[0] += 61
    # The queue, asked once more, has a job: stay.
    assert not idle.should_exit(60, lambda: True)
    assert idle.should_exit(60, empty)
    # Decided: no slot may start another claim, and the decision sticks.
    assert not idle.enter()
    assert idle.should_exit(60, lambda: True)


def test_an_empty_poll_does_not_restart_the_idle_clock() -> None:
    now = [0.0]
    idle = _Idle(clock=lambda: now[0])
    for _ in range(5):
        now[0] += 30
        assert idle.enter()
        idle.leave(ran=False)
    assert idle.idle_for() == 150.0


# --- the loop -------------------------------------------------------------------------


def test_an_idle_worker_runs_what_is_queued_and_then_exits_on_its_own(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    capture = make_capture(db)
    job = queue_job(db, capture, "t-two")
    worker = Worker(sessions, storage, config(tmp_path, "worker-a", **IDLE))

    thread, ran = _run(worker)
    thread.join(timeout=30)

    assert not thread.is_alive(), "the worker never exited"
    assert ran == [1]
    assert worker.exited_idle
    assert _status(db, job) is RunStatus.COMPLETE


def test_a_job_committed_as_the_worker_decides_to_exit_keeps_it_up(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The race: the last poll found nothing, and a job lands before the worker goes.
    The look at the queue just before exiting is what catches it."""
    capture = make_capture(db)
    real = anything_claimable
    late: list[Job] = []

    def a_job_arrives_first(session: Session) -> bool:
        if not late:
            late.append(queue_job(db, capture, "t-two"))
        return real(session)

    monkeypatch.setattr(worker_loop, "anything_claimable", a_job_arrives_first)
    worker = Worker(sessions, storage, config(tmp_path, "worker-a", **IDLE))

    thread, ran = _run(worker)
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert late, "the worker never got as far as deciding to exit"
    assert _status(db, late[0]) is RunStatus.COMPLETE
    assert ran == [1]
    assert worker.exited_idle


def test_a_worker_does_not_exit_while_any_slot_holds_a_job(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    capture = make_capture(db)
    job = queue_job(db, capture, "t-slow")
    worker = Worker(sessions, storage, config(tmp_path, "worker-a", concurrency=2, **IDLE))

    thread, _ = _run(worker)
    try:
        assert wait_until(lambda: _status(db, job) is RunStatus.IN_PROGRESS)
        # Far longer than idle_exit_s, with the other slot polling an empty queue.
        time.sleep(2.0)
        assert thread.is_alive()
        assert not worker.exited_idle
        # Once the job is over, the worker is idle, and goes.
        job_service.cancel_job(db, job.id)
        thread.join(timeout=30)
    finally:
        assert not thread.is_alive()
    assert worker.exited_idle


def test_a_stop_still_interrupts_a_backed_off_wait(
    sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """SIGTERM's behaviour is unchanged: a slot asleep for the longest backoff wakes on the
    stop flag, not at the end of its sleep."""
    worker = Worker(
        sessions,
        storage,
        config(tmp_path, idle_s=0.05, idle_backoff_after_s=0.05, idle_max_s=600.0),
    )
    stop = threading.Event()
    thread, ran = _run(worker, stop)
    time.sleep(0.5)  # well into the backoff: the slot's next wait is ten minutes

    started = time.monotonic()
    stop.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert time.monotonic() - started < 5
    assert ran == [0]
    assert not worker.exited_idle


def test_the_process_exits_0_when_idle(tmp_path: Path) -> None:
    """Exit status 0 is the contract with fly.toml's restart policy (`on-failure`): an
    idle exit leaves the machine stopped, where a crash would have it restarted."""
    env = dict(os.environ)
    env.update(
        {
            "DATABASE_URL": TEST_DATABASE_URL,
            "WORKER_WORKDIR": str(tmp_path / "runs"),
            "WORKER_RECIPE_DIR": str(RECIPES),
            "WORKER_IMPL_MODULES": "tests.worker_stages",
            "WORKER_RUNNER": "local",
            "WORKER_IDLE_S": "0.05",
            "WORKER_IDLE_EXIT_S": "1",
            "PYTHONPATH": str(API_ROOT),
            "OBJECT_STORAGE_ENDPOINT_URL": "",
            "OBJECT_STORAGE_BUCKET": "",
            "OBJECT_STORAGE_ACCESS_KEY": "",
            "OBJECT_STORAGE_SECRET_KEY": "",
        }
    )
    process = subprocess.run(
        [sys.executable, "-u", "-m", "app.worker"],
        cwd=str(API_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "exiting so the machine can stop" in process.stderr
