"""A GPU call left running by a job cancelled while no worker held it is still stopped.

The 2026-10 review's first finding. `POST /jobs/{id}/cancel` only sets the status; a
worker holding the job sees it and cancels its call. A worker that was not there -- it
crashed, ran out of restarts, or was stopped by `fly machine stop` after detaching the
call for a successor -- never did: a cancelled job is never claimed again, and the step's
copy of its calls was only read on a claim. So the worker now reaps the recorded calls of
every job that is over, when it starts and on a slow tick (`app/worker/reaper.py`), and
the API wakes it on a cancel (`app/services/worker_wake.py`).

The calls go to `tests/fake_modal`, as in `tests/test_worker_stops.py`: a process of their
own that outlives the worker that spawned them and can be cancelled by id from another.
Here the reaper runs in the test's own process -- it is the worker's, not a recipe
process's -- so the stand-in `modal` is importable here too (`modal_here`).
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import _db
from app.config import Settings
from app.main import create_app
from app.models import Capture, Job, JobStep
from app.models.enums import RunStatus
from app.services import jobs as job_service
from app.services import phone_key, worker_wake
from app.storage import S3Storage
from app.worker.loop import Worker
from app.worker.pipeline_bridge import CALL_BOOK, CallBook, CallRecord
from app.worker.runner import JobSupervisor
from tests.test_worker import config, make_capture, queue_job, steps_by_stage, wait_until
from tests.test_worker_stops import (
    FAKE_MODAL,
    alive,
    call_running,
    claimed,
    fake_modal,  # noqa: F401 - a fixture, used by name
    in_background,
    ledger,
    modal_config,
)
from tests.test_worker_wake import KEY, PHONE, FakeFly, fly_settings, machine


@pytest.fixture
def modal_here(fake_modal: Path, monkeypatch: pytest.MonkeyPatch) -> Path:  # noqa: F811
    """`tests/fake_modal`'s `modal`, importable in this process for as long as the test
    runs: the reaper cancels from the worker's own process."""
    spec = importlib.util.spec_from_file_location("modal", FAKE_MODAL / "modal" / "__init__.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "modal", module)
    return fake_modal


def detached_and_cancelled(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path, state: Path
) -> tuple[Job, str]:
    """`fly machine stop` mid-train: the worker detaches the call for a successor, lets go
    of the job and exits, and nobody succeeds it. Then the job is cancelled."""
    job = queue_job(db, make_capture(db), "t-gpu-slow")
    claimed(sessions, job)
    workdir = tmp_path / "runs" / str(job.id)
    stop = threading.Event()
    thread, result = in_background(
        JobSupervisor(sessions, storage, modal_config(tmp_path)), job.id, stop
    )
    assert wait_until(lambda: call_running(workdir, state), timeout=60), "never dispatched"
    call_id = call_running(workdir, state)
    assert call_id is not None
    stop.set()
    thread.join(timeout=30)
    assert result == ["lost"]
    job_service.cancel_job(db, job.id)
    return job, call_id


@pytest.mark.parametrize("workdir", ["on this volume", "gone with another machine"])
def test_the_call_of_a_job_cancelled_while_no_worker_held_it_is_reaped_once(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    modal_here: Path,
    workdir: str,
) -> None:
    job, call_id = detached_and_cancelled(db, sessions, storage, tmp_path, modal_here)
    pid = int((modal_here / call_id / "pid").read_text())
    assert alive(pid) and ledger(modal_here, "cancelled.txt") == []
    if workdir != "on this volume":
        # The worker that comes up is on another machine: all it has is the row's copy.
        shutil.rmtree(tmp_path / "runs" / str(job.id))
    worker = Worker(sessions, storage, modal_config(tmp_path, "worker-b"))

    assert worker.reap_abandoned() == [call_id]

    assert ledger(modal_here, "cancelled.txt") == [call_id]
    assert (modal_here / call_id / "cancelled").read_text() == "True", "containers terminated"
    assert wait_until(lambda: not alive(pid), timeout=5), "the GPU went on running"
    train = steps_by_stage(db, job.id)["train"]
    assert "remoteCalls" not in train.metrics
    assert train.metrics["reapedCalls"][call_id]["how"] == "cancelled"
    assert not (tmp_path / "runs" / str(job.id) / "stages" / "train" / CALL_BOOK).exists()

    # Idempotent: a second pass -- or a second worker's -- has nothing to do.
    assert worker.reap_abandoned() == []
    assert ledger(modal_here, "cancelled.txt") == [call_id]


def test_a_job_its_worker_cancelled_itself_leaves_nothing_to_reap(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    modal_here: Path,
) -> None:
    """The ordinary cancel: the supervisor's recipe process cancels the call, and the row's
    copy of it goes too -- the reaper finds nothing, and cancels nothing a second time."""
    job = queue_job(db, make_capture(db), "t-gpu-slow")
    claimed(sessions, job)
    workdir = tmp_path / "runs" / str(job.id)
    thread, result = in_background(JobSupervisor(sessions, storage, modal_config(tmp_path)), job.id)
    assert wait_until(lambda: call_running(workdir, modal_here), timeout=60), "never dispatched"
    job_service.cancel_job(db, job.id)
    thread.join(timeout=30)
    assert result == ["cancelled"]

    assert Worker(sessions, storage, modal_config(tmp_path, "worker-b")).reap_abandoned() == []
    assert len(ledger(modal_here, "cancelled.txt")) == 1
    assert "remoteCalls" not in steps_by_stage(db, job.id)["train"].metrics


def test_a_job_a_worker_may_still_be_closing_is_left_to_it_and_a_stale_record_expires(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """A cancelled job whose lease has not run out for a whole lease is some worker's to
    close, with the call's billing; and a record older than any function runs is struck
    off without asking the provider (which here does not even exist)."""
    job = queue_job(db, make_capture(db), "t-gpu-slow")
    job.status = RunStatus.CANCELLED
    job.lease_expires_at = datetime.now(tz=UTC) + timedelta(seconds=20)
    old = time.time() - 2 * 24 * 3600
    record = {"id": "fc-long-gone", "provider": "modal", "tier": "l4", "submittedAt": old}
    db.add(
        JobStep(
            job_id=job.id,
            stage_id="train",
            ordinal=1,
            impl="t_gpu_slow",
            status=RunStatus.CANCELLED,
            metrics={"remoteCalls": {"": record}},
        )
    )
    db.commit()
    worker = Worker(sessions, storage, modal_config(tmp_path, "worker-b"))

    assert worker.reap_abandoned() == []
    assert "remoteCalls" in steps_by_stage(db, job.id)["train"].metrics, "held: left alone"

    db.execute(update(Job).where(Job.id == job.id).values(lease_expires_at=None))
    db.commit()
    assert worker.reap_abandoned() == []
    train = steps_by_stage(db, job.id)["train"]
    assert "remoteCalls" not in train.metrics
    assert train.metrics["reapedCalls"]["fc-long-gone"]["how"] == "expired"


def test_the_worker_reaps_when_it_starts_and_on_its_tick(
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passes: list[float] = []
    monkeypatch.setattr(Worker, "reap_abandoned", lambda self: passes.append(time.monotonic()))
    worker = Worker(sessions, storage, config(tmp_path, reap_every_s=0.1))
    stop = threading.Event()
    loop = threading.Thread(target=lambda: worker.run_forever(stop=stop), daemon=True)
    started = time.monotonic()
    loop.start()

    assert wait_until(lambda: len(passes) >= 3, timeout=5)
    stop.set()
    loop.join(timeout=10)

    assert not loop.is_alive()
    assert passes[0] - started < 0.5, "the first pass is at start-up, before any tick"


def test_a_book_whose_cancel_failed_is_kept_orphaned_for_the_next_pass(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """A provider this deployment no longer dispatches to cannot be asked: the call stays
    in the book for the next pass, and the book is marked orphaned, so a Retry's recipe
    process cancels it instead of re-attaching to a call of a cancelled run."""
    job = queue_job(db, make_capture(db), "t-gpu-slow")
    job.status = RunStatus.CANCELLED
    db.commit()
    book = tmp_path / "runs" / str(job.id) / "stages" / "train" / CALL_BOOK
    handle = {"id": "fc-elsewhere", "provider": "runpod", "tier": "a100"}
    CallBook(book, calls={"": CallRecord.from_dict({**handle, "submittedAt": time.time()})}).save()

    assert Worker(sessions, storage, modal_config(tmp_path, "worker-b")).reap_abandoned() == []

    kept = CallBook.read(book)
    assert kept.orphaned and [r.handle.id for r in kept.calls.values()] == ["fc-elsewhere"]


# --------------------------------------------------------------------------------------
# The API wakes the worker on a cancel
# --------------------------------------------------------------------------------------


def test_waking_for_a_cancel_starts_the_worker_and_does_not_ping_the_queue_check() -> None:
    fly = FakeFly([machine("w1", "stopped")])

    wake = worker_wake.after_cancel(
        fly_settings(queue_check_url="https://hc-ping.com/abc"),
        transport=fly.transport,
        later=lambda *a: None,
    )

    assert wake == worker_wake.Wake(started=("w1",))
    assert fly.started() == ["w1"]
    assert not [r for r in fly.requests if r.url.host == "hc-ping.com"], "nothing was queued"

    up = FakeFly([machine("w1", "started")])
    later: list[float] = []
    worker_wake.after_cancel(
        fly_settings(), transport=up.transport, later=lambda delay, _call: later.append(delay)
    )
    assert later == [worker_wake.RECHECK_AFTER_S], "it may have been on its way out"


class Cancels:
    """A client whose cancel-wakes are recorded, each with the status of `job` as a
    session of its own saw it at that moment -- which is how "after the commit" is
    checked."""

    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.job: uuid.UUID | None = None
        self.seen: list[str] = []


@pytest.fixture
def cancels(db: Session, engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[Cancels]:
    settings = Settings(
        api_phone_key_hash=phone_key.hash_key(KEY, salt=b"0123456789abcdef", iterations=1000),
    )
    app = create_app(settings)

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    with TestClient(app) as client:
        recorded = Cancels(client)

        def recording() -> None:
            with Session(engine) as fresh:
                status = fresh.scalar(select(Job.status).where(Job.id == recorded.job))
                recorded.seen.append(status.value if status is not None else "missing")

        monkeypatch.setattr(worker_wake, "after_cancel", recording)
        yield recorded


def test_cancelling_a_job_a_worker_had_claimed_wakes_the_worker_after_the_commit(
    db: Session, sessions: sessionmaker[Session], cancels: Cancels
) -> None:
    queued = queue_job(db, make_capture(db, "never-claimed"), "t-three")
    cancels.job = queued.id
    assert cancels.client.post(f"/api/v1/jobs/{queued.id}/cancel").status_code == 200
    assert cancels.seen == [], "never claimed: no worker can have left a call"

    running = queue_job(db, make_capture(db, "claimed"), "t-gpu-slow")
    claimed(sessions, running)
    cancels.job = running.id
    assert cancels.client.post(f"/api/v1/jobs/{running.id}/cancel").status_code == 200
    assert cancels.seen == ["cancelled"]


def test_the_phones_stop_wakes_the_worker_too(
    db: Session, sessions: sessionmaker[Session], cancels: Cancels
) -> None:
    mine = cancels.client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()
    capture = db.get(Capture, uuid.UUID(mine["capture"]["id"]))
    assert capture is not None
    job = queue_job(db, capture)
    claimed(sessions, job)
    cancels.job = job.id

    response = cancels.client.post(f"/api/v1/phone/captures/{capture.id}/stop", headers=PHONE)

    assert response.status_code == 200, response.text
    assert cancels.seen == ["cancelled"]
