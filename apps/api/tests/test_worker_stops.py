"""Stopping a run, reading a failure before retrying it, and what the worker tells and keeps.

The first half is the 2026-10 audit's R2, end to end and with nothing mocked inside the
worker: a real supervisor, a real recipe process, the real `CloudRunner` and the real
`ModalAdapter` -- dispatching to `tests/fake_modal`, a stand-in for the `modal` client
whose every call is a process of its own (running `remote.execute`, as a container does)
with its state on disk. That is the one property that matters here: the call goes on
when the process that spawned it is stopped, and another process can pick it up by id.

* a **deploy** (the worker's stop flag) leaves the call running, and the next worker
  re-attaches to it -- one call, one attempt;
* a **cancel** stops the call within seconds and still counts what it cost;
* a **crash** (the recipe process SIGKILLed) leaves the call too, and the retry adopts it;
* a **lost workdir** leaves only the database's copy of the call's id, which is enough
  to cancel it.

Then the retry policy (R3), the dollar cap, the recipe process's stderr (R18), the
dead-man's switch and the disk guard (R8).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import signal
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.models import Job
from app.models.enums import RunStatus
from app.services import jobs as job_service
from app.storage import S3Storage
from app.worker import alerts, retry
from app.worker.child import Interrupts
from app.worker.claim import claim_next
from app.worker.config import WorkerConfig
from app.worker.disk import GB, DiskGuard
from app.worker.loop import Worker
from app.worker.pipeline_bridge import (
    CALL_BOOK,
    AttemptLedger,
    CallBook,
    CancelRequested,
    DetachRequested,
)
from app.worker.runner import CHILD_STDERR, JobSupervisor, Terminal
from tests.test_worker import config, make_capture, queue_job, steps_by_stage, wait_until

FAKE_MODAL = Path(__file__).resolve().parent / "fake_modal"


@pytest.fixture
def fake_modal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The stand-in `modal` on the recipe process's path, and its calls' state dir."""
    state = tmp_path / "modal"
    state.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setenv("FAKE_MODAL_DIR", str(state))
    monkeypatch.setenv("FAKE_MODAL_TRANSFER", str(shared))
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", f"{FAKE_MODAL}:{existing}" if existing else str(FAKE_MODAL))
    yield state
    for pid in state.glob("*/pid"):
        with contextlib.suppress(ProcessLookupError, PermissionError, ValueError):
            os.killpg(int(pid.read_text()), signal.SIGKILL)


def modal_config(tmp_path: Path, worker_id: str = "worker-a", **overrides: object) -> WorkerConfig:
    defaults: dict[str, object] = {
        "runner": "cloud",
        "cloud_providers": ("modal",),
        "modal_app": "twin-test",
        "cloud_poll_s": 0.1,
        "checkpoint_every_s": 0.2,
        "cloud_transfer_dir": tmp_path / "shared",
        "terminate_grace_s": 10.0,
    }
    defaults.update(overrides)
    return config(tmp_path, worker_id, **defaults)


def claimed(sessions: sessionmaker[Session], job: Job, worker_id: str = "worker-a") -> None:
    session = sessions()
    try:
        assert claim_next(session, worker_id=worker_id, lease_s=30) is not None
    finally:
        session.close()


def ledger(state: Path, name: str) -> list[str]:
    path = state / name
    return path.read_text().split() if path.is_file() else []


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie is dead for this purpose: it has stopped running and only awaits its reap.
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.is_file() and stat.read_text().split(") ", 1)[-1].startswith("Z"))


def in_background(
    supervisor: JobSupervisor, job_id: uuid.UUID, stop: threading.Event | None = None
) -> tuple[threading.Thread, list[Terminal]]:
    result: list[Terminal] = []
    thread = threading.Thread(
        target=lambda: result.append(supervisor.run(job_id, stop=stop)), daemon=True
    )
    thread.start()
    return thread, result


def call_running(workdir: Path, state: Path, step: str = "step 2") -> str | None:
    """The id of the train stage's call once its container has done some work."""
    book = CallBook.read(workdir / "stages" / "train" / CALL_BOOK)
    for record in book.calls.values():
        log = state / record.handle.id / "log.txt"
        if log.is_file() and step in log.read_text():
            return record.handle.id
    return None


def recipe_process() -> int | None:
    """This test process's recipe process (the supervisor runs in one of its threads)."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
            parent = int((entry / "stat").read_text().split(") ", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if b"app.worker.child" in command and parent == os.getpid():
            return int(entry.name)
    return None


# --------------------------------------------------------------------------------------
# A deploy, a cancel, a crash and a lost workdir, with a GPU call in flight
# --------------------------------------------------------------------------------------


def test_a_deploy_leaves_the_gpu_call_running_and_the_next_worker_re_attaches_to_it(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    fake_modal: Path,
) -> None:
    job = queue_job(db, make_capture(db), "t-gpu-slow")
    claimed(sessions, job)
    workdir = tmp_path / "runs" / str(job.id)
    stop = threading.Event()
    thread, result = in_background(
        JobSupervisor(sessions, storage, modal_config(tmp_path)), job.id, stop
    )
    assert wait_until(lambda: call_running(workdir, fake_modal), timeout=60), "never dispatched"
    call_id = call_running(workdir, fake_modal)
    assert call_id is not None

    stop.set()
    thread.join(timeout=30)

    assert result == ["lost"]
    book = CallBook.read(workdir / "stages" / "train" / CALL_BOOK)
    assert book.detached and [r.handle.id for r in book.calls.values()] == [call_id]
    assert ledger(fake_modal, "cancelled.txt") == [], "a deploy must not cancel the GPU"
    pid = int((fake_modal / call_id / "pid").read_text())
    assert alive(pid), "the call is still running, on nobody's watch for the moment"
    step = steps_by_stage(db, job.id)["train"]
    assert step.metrics["detached"] is True
    assert step.metrics["remoteCalls"][""]["id"] == call_id
    db.expire_all()
    handed_back = db.get(Job, job.id)
    assert handed_back is not None and handed_back.lease_expires_at is None

    claimed(sessions, job, "worker-b")
    outcome = JobSupervisor(sessions, storage, modal_config(tmp_path, "worker-b")).run(job.id)

    assert outcome == "complete"
    assert ledger(fake_modal, "spawned.txt") == [call_id], "one call, not two"
    assert call_id in ledger(fake_modal, "from_id.txt")
    train = steps_by_stage(db, job.id)["train"]
    assert train.attempt == 1, "a deploy does not spend an attempt"
    assert "remoteCalls" not in train.metrics and "detached" not in train.metrics
    trained = json.loads((workdir / "stages" / "train" / "out" / "trained.json").read_text())
    assert trained["pid"] == pid and trained["iterations"] == 12
    (entry,) = AttemptLedger.read(workdir / "stages" / "train" / "attempts.json").entries
    assert entry.call == call_id and entry.state == "succeeded"
    assert not (workdir / "stages" / "train" / CALL_BOOK).exists()
    db.expire_all()
    finished = db.get(Job, job.id)
    assert finished is not None and finished.cost_usd is not None


def test_cancelling_a_job_stops_its_gpu_call_and_counts_what_it_cost(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    fake_modal: Path,
) -> None:
    job = queue_job(db, make_capture(db), "t-gpu-slow")
    claimed(sessions, job)
    workdir = tmp_path / "runs" / str(job.id)
    thread, result = in_background(JobSupervisor(sessions, storage, modal_config(tmp_path)), job.id)
    assert wait_until(lambda: call_running(workdir, fake_modal), timeout=60), "never dispatched"
    call_id = call_running(workdir, fake_modal)
    assert call_id is not None
    pid = int((fake_modal / call_id / "pid").read_text())

    started = time.monotonic()
    job_service.cancel_job(db, job.id)
    thread.join(timeout=30)

    assert result == ["cancelled"]
    assert ledger(fake_modal, "cancelled.txt") == [call_id]
    assert (fake_modal / call_id / "cancelled").read_text() == "True", "containers terminated"
    assert wait_until(lambda: not alive(pid), timeout=5), "the GPU went on running"
    # A poll interval, the recipe process's way out, and one request to the provider.
    assert time.monotonic() - started < 10
    (entry,) = AttemptLedger.read(workdir / "stages" / "train" / "attempts.json").entries
    assert entry.call == call_id and entry.detail.startswith("cancelled:")
    db.expire_all()
    cancelled = db.get(Job, job.id)
    assert cancelled is not None and cancelled.status is RunStatus.CANCELLED
    assert cancelled.cost_usd is not None and float(cancelled.cost_usd) > 0


def test_a_recipe_process_killed_mid_call_leaves_the_call_for_the_retry_to_adopt(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    fake_modal: Path,
) -> None:
    """A crash, not a deploy: nothing runs on the way out, the attempt is spent -- a stage
    that keeps killing its worker must run out of them -- but the call the dead process
    left is picked up rather than started again beside itself."""
    job = queue_job(db, make_capture(db), "t-gpu-slow")
    claimed(sessions, job)
    workdir = tmp_path / "runs" / str(job.id)
    thread, result = in_background(JobSupervisor(sessions, storage, modal_config(tmp_path)), job.id)
    assert wait_until(lambda: call_running(workdir, fake_modal), timeout=60), "never dispatched"
    call_id = call_running(workdir, fake_modal)
    child = recipe_process()
    assert child is not None

    os.kill(child, signal.SIGKILL)
    thread.join(timeout=60)

    assert result == ["complete"]
    assert ledger(fake_modal, "spawned.txt") == [call_id]
    assert call_id in ledger(fake_modal, "from_id.txt")
    assert ledger(fake_modal, "cancelled.txt") == []
    assert steps_by_stage(db, job.id)["train"].attempt == 2


def test_a_call_whose_workdir_is_gone_is_cancelled_by_the_id_the_database_kept(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    fake_modal: Path,
) -> None:
    """The job is reclaimed on a machine without its workdir: the run starts over, and the
    call the old one left -- found by id on the step's row -- is cancelled first."""
    job = queue_job(db, make_capture(db), "t-gpu-slow")
    claimed(sessions, job)
    workdir = tmp_path / "runs" / str(job.id)
    stop = threading.Event()
    thread, _ = in_background(
        JobSupervisor(sessions, storage, modal_config(tmp_path)), job.id, stop
    )
    assert wait_until(lambda: call_running(workdir, fake_modal), timeout=60), "never dispatched"
    old = call_running(workdir, fake_modal)
    stop.set()
    thread.join(timeout=30)
    shutil.rmtree(workdir)

    claimed(sessions, job, "worker-b")
    outcome = JobSupervisor(sessions, storage, modal_config(tmp_path, "worker-b")).run(job.id)

    assert outcome == "complete"
    assert ledger(fake_modal, "cancelled.txt") == [old]
    spawned = ledger(fake_modal, "spawned.txt")
    assert len(spawned) == 2 and spawned[0] == old
    assert "cancelled call" in (workdir / "stages" / "train" / "log.txt").read_text()


# --------------------------------------------------------------------------------------
# The signal says why
# --------------------------------------------------------------------------------------


def test_the_recipe_process_hears_a_cancel_as_a_cancel_and_a_shutdown_as_a_detach(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    def workdir_of(job: Job) -> Path:
        return tmp_path / "runs" / str(job.id)

    def running(job: Job) -> bool:
        row = steps_by_stage(db, job.id).get("two")
        return row is not None and row.status is RunStatus.IN_PROGRESS

    cancelled = queue_job(db, make_capture(db, "cancelled"), "t-notes-stop")
    claimed(sessions, cancelled)
    thread, _ = in_background(JobSupervisor(sessions, storage, config(tmp_path)), cancelled.id)
    assert wait_until(lambda: running(cancelled))
    time.sleep(0.5)
    job_service.cancel_job(db, cancelled.id)
    thread.join(timeout=20)
    said = workdir_of(cancelled) / "stages" / "two" / "work" / "stopped-by.txt"
    assert said.read_text() == "CancelRequested"

    detached = queue_job(db, make_capture(db, "detached"), "t-notes-stop")
    claimed(sessions, detached)
    stop = threading.Event()
    thread, result = in_background(
        JobSupervisor(sessions, storage, config(tmp_path)), detached.id, stop
    )
    assert wait_until(lambda: running(detached))
    time.sleep(0.5)
    # A shutdown signalled to the whole process group reaches the recipe process too. It
    # is the worker's to read, not the recipe process's: nothing stops yet.
    child = recipe_process()
    assert child is not None
    os.kill(child, signal.SIGTERM)
    time.sleep(0.5)
    said = workdir_of(detached) / "stages" / "two" / "work" / "stopped-by.txt"
    assert not said.exists() and running(detached)
    stop.set()
    thread.join(timeout=20)
    assert result == ["lost"]
    assert said.read_text() == "DetachRequested"


def test_a_stop_that_lands_while_a_call_is_being_written_down_waits_for_it() -> None:
    """Between a call being created and its id reaching the book, a stop would leave a
    call nobody can find; `held` lets the block finish and then raises it. A cancel that
    arrives meanwhile wins over a detach."""
    interrupts = Interrupts()
    finished: list[str] = []
    with pytest.raises(DetachRequested), interrupts.held():
        interrupts._on_detach(signal.SIGUSR1, None)
        finished.append("written down")
    assert finished == ["written down"]

    with pytest.raises(CancelRequested), interrupts.held():
        interrupts._on_detach(signal.SIGUSR1, None)
        interrupts._on_cancel(signal.SIGUSR2, None)
    with pytest.raises(CancelRequested):
        interrupts._on_cancel(signal.SIGUSR2, None)


# --------------------------------------------------------------------------------------
# Reading a failure before retrying it
# --------------------------------------------------------------------------------------

REMOTE = "recipe 'r', stage 'train' (impl 'gsplat') failed on 'modal': "
OOM_LINE = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"


@pytest.mark.parametrize(
    ("error_type", "error", "log", "kind"),
    [
        ("RemoteTimeoutError", REMOTE + "FunctionTimeoutError: 6h", "", "timeout"),
        ("CostCapError", "the run has been billed $20.10", "", "cost-cap"),
        ("StageContractError", "asked for input 'poses'", "", "bad-input"),
        ("RemoteStageError", REMOTE + "UnknownImplError: no impl", "", "bad-input"),
        ("RemoteStageError", REMOTE + "CalledProcessError: exit 1", OOM_LINE, "oom"),
        ("StageFailedError", "failed: CUDA error: out of memory", "", "oom"),
        # Narrow on purpose: a stage's own ValueError is bad input and a dead trainer
        # alike, and outputs that did not come back may be a transfer that failed.
        ("RemoteStageError", REMOTE + "ValueError: the trainer wrote no .ply", "", "other"),
        ("MissingArtifactError", "wrote nothing at out/trained.ply", "", "other"),
        ("StageFailedError", "boom", "", "other"),
    ],
)
def test_a_failure_is_classified_narrowly(error_type: str, error: str, log: str, kind: str) -> None:
    assert retry.classify(error_type, error, log).kind == kind


def test_out_of_memory_is_retried_once_and_only_with_a_cap_to_lower(tmp_path: Path) -> None:
    history = retry.History.of(tmp_path)
    log = f"gsplat: cap_max auto -> 2000000; ...\n[b0] gsplat: cap_max auto -> 900000\n{OOM_LINE}"
    first = retry.classify("RemoteStageError", REMOTE + "CalledProcessError", log)

    assert first.cap_max == 2_000_000, "the stage's own budget, not a block's"
    decision = retry.decide("train", first, history)
    assert decision.retry and decision.params == {"cap_max": 1_400_000}

    history.add(1, first)
    again = retry.decide("train", first, retry.History.of(tmp_path))
    assert not again.retry and "again" in again.why

    uncapped = retry.decide("train", retry.Failure("oom"), retry.History.of(tmp_path / "x"))
    assert not uncapped.retry and "no gaussian cap" in uncapped.why


def run_one(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    recipe: str,
    params: dict[str, dict[str, object]] | None = None,
    **overrides: object,
) -> tuple[Job, Terminal]:
    job = queue_job(db, make_capture(db, f"c-{uuid.uuid4().hex[:8]}"), recipe)
    if params:
        job.params = params
        db.commit()
    claimed(sessions, job)
    cfg = config(tmp_path, "worker-a", **overrides)
    outcome = JobSupervisor(sessions, storage, cfg).run(job.id)
    db.expire_all()
    refreshed = db.get(Job, job.id)
    assert refreshed is not None
    return refreshed, outcome


def test_out_of_gpu_memory_is_retried_once_at_a_lower_gaussian_cap(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    job, outcome = run_one(db, sessions, storage, tmp_path, "t-oom", max_attempts=3)

    assert outcome == "complete"
    # 0.7 of the 1,000,000 its log said it trained at, written where a person sees it.
    assert job.params == {"train": {"cap_max": 700_000}}
    train = steps_by_stage(db, job.id)["train"]
    assert train.attempt == 2 and train.metrics["capMax"] == 700_000


def test_out_of_gpu_memory_twice_is_not_retried_a_third_time(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    job, outcome = run_one(
        db, sessions, storage, tmp_path, "t-oom", {"train": {"fits": 0}}, max_attempts=5
    )

    assert outcome == "error"
    assert "ran out of GPU memory again" in (job.error or "")
    assert steps_by_stage(db, job.id)["train"].attempt == 2


@pytest.mark.parametrize(
    ("recipe", "says"),
    [("t-timeout", "ran out of time"), ("t-contract", "its own code")],
)
def test_a_timeout_or_a_broken_stage_is_not_retried(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    recipe: str,
    says: str,
) -> None:
    job, outcome = run_one(db, sessions, storage, tmp_path, recipe, max_attempts=3)

    assert outcome == "error"
    assert says in (job.error or "")
    assert steps_by_stage(db, job.id)["train"].attempt == 1, "the budget was not spent on it"


def test_any_other_failure_keeps_the_attempts_it_had(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    job, outcome = run_one(db, sessions, storage, tmp_path, "t-doomed", max_attempts=3)

    assert outcome == "error"
    assert "attempted 3 times" in (job.error or "")


def test_a_run_over_its_dollar_cap_has_its_call_cancelled_and_is_not_retried(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    fake_modal: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dollar a second against a $3 cap: the call is cancelled a few seconds in, and the
    job is dead-lettered saying why, not retried into the same wall."""
    monkeypatch.setenv("PIPELINE_GPU_RATES", "modal:l4=3600")
    job = queue_job(db, make_capture(db), "t-gpu-slow")
    claimed(sessions, job)

    outcome = JobSupervisor(
        sessions, storage, modal_config(tmp_path, cost_cap_usd=3.0, max_attempts=3)
    ).run(job.id)

    assert outcome == "error"
    db.expire_all()
    dead = db.get(Job, job.id)
    assert dead is not None
    assert "cost cap" in (dead.error or "") and "WORKER_JOB_COST_CAP_USD" in (dead.error or "")
    assert len(ledger(fake_modal, "spawned.txt")) == 1
    assert ledger(fake_modal, "cancelled.txt") == ledger(fake_modal, "spawned.txt")
    assert steps_by_stage(db, job.id)["train"].attempt == 1
    assert dead.cost_usd is not None and float(dead.cost_usd) >= 3.0


# --------------------------------------------------------------------------------------
# What the recipe process said on its way out
# --------------------------------------------------------------------------------------


def test_a_recipe_process_that_dies_says_why_in_the_jobs_error(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    job, outcome = run_one(db, sessions, storage, tmp_path, "t-dies", max_attempts=1)

    assert outcome == "error"
    assert "exited with code 9" in (job.error or "")
    assert "could not map its weights" in (job.error or "")
    kept = tmp_path / "runs" / str(job.id) / CHILD_STDERR
    assert "could not map its weights" in kept.read_text()


# --------------------------------------------------------------------------------------
# The dead-man's switch
# --------------------------------------------------------------------------------------


class Pings:
    """A recording sender, safe from the ping threads."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def __call__(self, url: str, body: str) -> None:
        with self._lock:
            self.sent.append((url, body))

    def urls(self) -> list[str]:
        with self._lock:
            return [url for url, _ in self.sent]


def watched(
    sessions: sessionmaker[Session], storage: S3Storage, cfg: WorkerConfig, pings: Pings
) -> JobSupervisor:
    supervisor = JobSupervisor(sessions, storage, cfg)
    supervisor.sender = pings
    return supervisor


def test_an_active_run_pings_start_alive_and_success_and_the_queue_check(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    job = queue_job(db, make_capture(db), "t-slow")
    job.params = {"two": {"seconds": 1.0}}
    db.commit()
    claimed(sessions, job)
    pings = Pings()
    cfg = config(
        tmp_path,
        heartbeat_url="https://hc.example/run/",
        queue_check_url="https://hc.example/queue",
        heartbeat_every_s=0.2,
    )

    assert watched(sessions, storage, cfg, pings).run(job.id) == "complete"

    assert wait_until(lambda: any("complete" in body for _, body in pings.sent), timeout=5)
    urls = pings.urls()
    assert set(urls[:2]) == {"https://hc.example/queue", "https://hc.example/run/start"}
    assert urls.count("https://hc.example/run") >= 2, "alive while running, and done"
    assert "https://hc.example/run/fail" not in urls


def test_a_failed_run_pings_fail_with_why_and_a_stopped_one_pings_nothing_at_the_end(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    doomed = queue_job(db, make_capture(db, "doomed"), "t-doomed")
    claimed(sessions, doomed)
    pings = Pings()
    cfg = config(tmp_path, heartbeat_url="https://hc.example/run", max_attempts=1)
    assert watched(sessions, storage, cfg, pings).run(doomed.id) == "error"
    assert wait_until(lambda: "https://hc.example/run/fail" in pings.urls(), timeout=5)
    (body,) = [body for url, body in pings.sent if url.endswith("/fail")]
    assert "broken on purpose" in body

    slow = queue_job(db, make_capture(db, "slow"), "t-slow")
    claimed(sessions, slow)
    pings = Pings()
    stop = threading.Event()
    supervisor = watched(sessions, storage, config(tmp_path, heartbeat_url="https://hc/x"), pings)
    thread, result = in_background(supervisor, slow.id, stop)
    assert wait_until(lambda: "two" in steps_by_stage(db, slow.id))
    stop.set()
    thread.join(timeout=20)
    time.sleep(0.3)
    assert result == ["lost"]
    assert pings.urls() == ["https://hc/x/start"], "the next worker's /start, or silence"


def test_a_ping_that_hangs_or_fails_never_holds_up_or_fails_the_worker() -> None:
    def hangs(_url: str, _body: str) -> None:
        time.sleep(5.0)

    def fails(_url: str, _body: str) -> None:
        raise OSError("the monitoring service is down")

    for sender in (hangs, fails):
        started = time.monotonic()
        with alerts.RunWatch("https://hc/x", uuid.uuid4(), every_s=0.05, sender=sender) as watch:
            time.sleep(0.2)
            watch.finish("error", "boom")
        assert time.monotonic() - started < 1.0


def test_an_idle_worker_pings_nothing(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pings = Pings()
    monkeypatch.setattr(alerts, "send", pings)
    cfg = config(tmp_path, heartbeat_url="https://hc/x", queue_check_url="https://hc/q")
    assert Worker(sessions, storage, cfg).run_one() is None
    time.sleep(0.2)
    assert pings.sent == []


# --------------------------------------------------------------------------------------
# Room on the volume
# --------------------------------------------------------------------------------------


def workdir_with_scratch(root: Path, job_id: uuid.UUID) -> Path:
    workdir = root / str(job_id)
    (workdir / "inputs" / "upload").mkdir(parents=True)
    (workdir / "inputs" / "upload" / "video.mov").write_bytes(b"v" * 64)
    for part in ("work", "out"):
        (workdir / "stages" / "one" / part).mkdir(parents=True)
        (workdir / "stages" / "one" / part / "x.bin").write_bytes(b"x")
    return workdir


def finished_job(db: Session, status: RunStatus, ended: datetime | None) -> Job:
    job = queue_job(db, make_capture(db, f"d-{uuid.uuid4().hex[:8]}"), "t-three")
    job.status = status
    job.finished_at = ended
    db.commit()
    return job


def test_short_of_room_the_worker_tidies_and_evicts_old_finished_runs_and_nothing_else(
    db: Session, sessions: sessionmaker[Session], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runs"
    now = datetime.now(tz=UTC)
    old = finished_job(db, RunStatus.COMPLETE, now - timedelta(days=10))
    recent = finished_job(db, RunStatus.ERROR, now - timedelta(days=1))
    running = finished_job(db, RunStatus.IN_PROGRESS, None)
    for job in (old, recent, running):
        workdir_with_scratch(root, job.id)
    stray = workdir_with_scratch(root, uuid.uuid4())
    week_ago = (now - timedelta(days=9)).timestamp()
    os.utime(stray, (week_ago, week_ago))
    guard = DiskGuard(sessions, root, min_free_gb=5, evict_after_days=7)
    # Room appears once the old run's workdir is gone, and not before.
    monkeypatch.setattr(
        guard, "free_bytes", lambda: 10 * GB if not (root / str(old.id)).exists() else GB
    )

    assert guard.room_to_claim()

    assert not (root / str(old.id)).exists(), "evicted: finished, and older than a week"
    kept = root / str(recent.id)
    assert not (kept / "inputs").exists() and not (kept / "stages" / "one" / "work").exists()
    assert (kept / "stages" / "one" / "out" / "x.bin").is_file(), "retry-from-stage reads out/"
    busy = root / str(running.id)
    assert (busy / "inputs" / "upload" / "video.mov").is_file(), "a live run is never touched"
    assert (busy / "stages" / "one" / "work" / "x.bin").is_file()


def test_with_nothing_left_to_evict_the_worker_does_not_claim_and_says_so(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    queued = queue_job(db, make_capture(db), "t-three")
    worker = Worker(sessions, storage, config(tmp_path, min_free_gb=5))
    monkeypatch.setattr(worker._disk, "free_bytes", lambda: GB)
    # The session's alembic upgrade ran `fileConfig`, which disables every logger that
    # existed before it; the worker's did.
    monkeypatch.setattr(logging.getLogger("app.worker"), "disabled", False)

    with caplog.at_level(logging.ERROR, logger="app.worker"):
        assert worker.run_one() is None

    assert "NOT CLAIMING" in caplog.text
    db.expire_all()
    still = db.get(Job, queued.id)
    assert still is not None and still.status is RunStatus.NOT_STARTED
