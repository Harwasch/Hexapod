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
* a **stage that finished while its worker was uploading it** is finished on resume from
  its `step.json`, not run (and billed) again, and a stop no longer waits out a long
  upload;
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import delete, update
from sqlalchemy.orm import Session, sessionmaker

from app.models import Artifact, Capture, Job, JobStep
from app.models.enums import RunStatus
from app.services import jobs as job_service
from app.storage import S3Storage
from app.worker import registration, retry
from app.worker.disk import GB
from app.worker.loop import Worker
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


# --------------------------------------------------------------------------------------
# A stage that finished while its worker was uploading it
# --------------------------------------------------------------------------------------


def runs_of(tmp_path: Path, job: Job, stage: str) -> int:
    counter = tmp_path / "runs" / str(job.id) / "stages" / stage / "work" / "runs.txt"
    return len(counter.read_text().splitlines()) if counter.is_file() else 0


def killed_while_uploading(db: Session, tmp_path: Path, job: Job, stage: str, after: str) -> None:
    """What a worker SIGKILLed in the middle of `stage`'s upload leaves, made from a run
    that finished: the recipe process wrote `step.json` and reported the stage, the row is
    still `in-progress` with nothing uploaded, the stage after it never ran, and the
    lease is free."""
    workdir = tmp_path / "runs" / str(job.id)
    written = datetime.fromtimestamp(
        (workdir / "stages" / stage / "step.json").stat().st_mtime, tz=UTC
    )
    rows = steps_by_stage(db, job.id)
    row = rows[stage]
    db.execute(delete(Artifact).where(Artifact.job_step_id == row.id))
    row.status = RunStatus.IN_PROGRESS
    row.started_at = written - timedelta(seconds=5)
    row.finished_at = None
    row.metrics = {}
    db.delete(rows[after])
    for leftover in ("step.json", "out"):
        path = workdir / "stages" / after / leftover
        if path.is_dir():
            for member in path.iterdir():
                member.unlink()
            path.rmdir()
        elif path.exists():
            path.unlink()
    db.execute(
        update(Job)
        .where(Job.id == job.id)
        .values(
            status=RunStatus.IN_PROGRESS,
            claimed_by=None,
            lease_expires_at=None,
            finished_at=None,
        )
    )
    db.commit()


def test_a_stage_that_finished_during_its_upload_is_finished_on_resume_not_run_again(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """SIGTERM mid-upload was killed at fly.toml's 30 s `kill_timeout` with the step row
    still `in-progress`, and the next worker ran the finished stage again -- for `train`,
    the GPU billed twice. The `step.json` that attempt wrote, with `out/` matching its
    checksums, is the proof it finished: the upload is redone and the row finished."""
    job = queue_job(db, make_capture(db), "t-three")
    claimed(sessions, job)
    assert JobSupervisor(sessions, storage, config(tmp_path)).run(job.id) == "complete"
    killed_while_uploading(db, tmp_path, job, "two", after="three")

    claimed(sessions, job, "worker-b")
    outcome = JobSupervisor(sessions, storage, config(tmp_path, "worker-b")).run(job.id)

    assert outcome == "complete"
    assert runs_of(tmp_path, job, "two") == 1, "the finished stage ran again"
    rows = steps_by_stage(db, job.id)
    assert rows["two"].status is RunStatus.COMPLETE and rows["two"].attempt == 1
    assert [a.storage_key for a in rows["two"].artifacts] == [f"runs/{job.id}/two/second.json"]
    assert storage.head_object(f"runs/{job.id}/two/second.json") is not None
    assert rows["three"].status is RunStatus.COMPLETE


@pytest.mark.parametrize("why", ["out/ changed", "an earlier run's step.json", "another attempt"])
def test_a_step_json_that_is_not_this_attempts_proof_is_not_taken_for_one(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path, why: str
) -> None:
    """The proof has to be this attempt's: written after the row's attempt started, by
    that attempt, over the `out/` that is there now. A Refine re-runs `train` beside the
    preview's `step.json`, and that one must never be mistaken for the new run's."""
    job = queue_job(db, make_capture(db), "t-three")
    claimed(sessions, job)
    assert JobSupervisor(sessions, storage, config(tmp_path)).run(job.id) == "complete"
    killed_while_uploading(db, tmp_path, job, "two", after="three")
    row = steps_by_stage(db, job.id)["two"]
    if why == "out/ changed":
        out = tmp_path / "runs" / str(job.id) / "stages" / "two" / "out" / "second.json"
        out.write_text("{}\n")
    elif why == "an earlier run's step.json":
        row.started_at = datetime.now(tz=UTC)
    else:
        row.attempt = 2
    db.commit()

    claimed(sessions, job, "worker-b")
    outcome = JobSupervisor(sessions, storage, config(tmp_path, "worker-b")).run(job.id)

    assert outcome == "complete"
    assert runs_of(tmp_path, job, "two") == 2, why


def test_a_stop_lands_between_member_uploads_and_the_next_worker_finishes_the_stage(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deploy that arrives while a stage's directory is going up stops between two
    members, rather than at the end of the whole directory -- or at the SIGKILL, thirty
    seconds in -- and the next worker picks the finished stage up from its `step.json`."""
    job = queue_job(db, make_capture(db), "t-many")
    claimed(sessions, job)
    stop = threading.Event()
    real = storage.upload_file
    uploaded: list[str] = []
    lock = threading.Lock()

    def slow(key: str, source: Path, content_type: str, **kwargs: object) -> object:
        if "/many/" in key:
            with lock:
                uploaded.append(key)
            stop.set()  # the deploy's SIGTERM, as the first member goes up
            time.sleep(0.1)
        return real(key, source, content_type, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(storage, "upload_file", slow)

    result = JobSupervisor(sessions, storage, config(tmp_path)).run(job.id, stop=stop)

    assert result == "lost"
    assert 0 < len(uploaded) < 40, "the stop waited for the whole directory"
    row = steps_by_stage(db, job.id)["many"]
    assert row.status is RunStatus.IN_PROGRESS and row.metrics.get("detached") is True

    monkeypatch.setattr(storage, "upload_file", real)
    claimed(sessions, job, "worker-b")
    outcome = JobSupervisor(sessions, storage, config(tmp_path, "worker-b")).run(job.id)

    assert outcome == "complete"
    assert runs_of(tmp_path, job, "many") == 1
    listed = storage.list_objects(f"runs/{job.id}/many/many/")
    assert len(listed.objects) == 40
    assert steps_by_stage(db, job.id)["many"].attempt == 1


def test_a_stop_during_publishing_lets_go_and_the_next_worker_publishes(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publishing a large capture is minutes of copies. Every stage is finished on its row
    by then, so a deploy that arrives during it lets go of the job (rather than being
    SIGKILLed at the 30 s `kill_timeout`), and the next worker publishes and registers
    it, running no stage again."""
    capture = make_capture(db, slug="orchard")
    job = queue_job(db, capture, "t-ingest")
    claimed(sessions, job)
    stop = threading.Event()
    real = registration.publish_outputs

    def publishing(*args: Any, **kwargs: Any) -> Any:
        stop.set()  # the deploy's SIGTERM, while the copies are going on
        return real(*args, **kwargs)

    monkeypatch.setattr(registration, "publish_outputs", publishing)
    result = JobSupervisor(sessions, storage, config(tmp_path)).run(job.id, stop=stop)

    assert result == "lost"
    db.expire_all()
    handed_back = db.get(Job, job.id)
    assert handed_back is not None and handed_back.status is RunStatus.IN_PROGRESS
    assert handed_back.lease_expires_at is None
    finished = {stage: row.attempt for stage, row in steps_by_stage(db, job.id).items()}
    assert all(row.status is RunStatus.COMPLETE for row in steps_by_stage(db, job.id).values())

    monkeypatch.setattr(registration, "publish_outputs", real)
    claimed(sessions, job, "worker-b")
    outcome = JobSupervisor(sessions, storage, config(tmp_path, "worker-b")).run(job.id)

    assert outcome == "complete"
    db.expire_all()
    registered = db.get(Capture, capture.id)
    assert registered is not None and registered.site_id is not None
    again = {stage: row.attempt for stage, row in steps_by_stage(db, job.id).items()}
    assert again == finished, "no stage ran again"


# --------------------------------------------------------------------------------------
# A failure is read from what failed, not from everything the attempt logged
# --------------------------------------------------------------------------------------

REMOTE = "recipe 'r', stage 'train' (impl 'gsplat') failed on 'modal': "
OOM_LINE = "[b3] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"


def test_an_out_of_memory_the_attempt_got_past_does_not_condemn_its_later_failure() -> None:
    """One piece of a fan-out ran out of memory and was resubmitted alone, and finished;
    the attempt then failed on something else entirely. The whole attempt's log has the
    out-of-memory in it, and read that way the failure was "oom": with no cap to lower,
    not retried at all. The verdict comes from the error and the log's last lines."""
    log = "\n".join(
        [
            "gsplat: cap_max auto -> 1000000; ...",
            OOM_LINE,
            "cloud: part b3 failed on call 1 of 3; resubmitting it alone",
            *[f"[b{n % 4}] step {n} of 3000" for n in range(200)],
            "cloud: 4 of 4 part(s) finished",
            "join: merging 4 blocks",
            "Traceback (most recent call last):",
            "ValueError: the merged splat has no gaussians inside the support mask",
        ]
    )
    failure = retry.classify("RemoteStageError", REMOTE + "CalledProcessError: exit 1", log)
    assert failure.kind == "other"

    # The same out-of-memory as the attempt's last words is one, with the stage's cap.
    last = retry.classify("RemoteStageError", REMOTE + "CalledProcessError", f"{log}\n{OOM_LINE}")
    assert last == retry.Failure("oom", cap_max=1_000_000)


# --------------------------------------------------------------------------------------
# Short of room, a detached run whose workdir is here is still picked up
# --------------------------------------------------------------------------------------


def detached_job(db: Session, slug: str) -> Job:
    """A run a deploy let go of mid-train: in progress, no lease, its step detached."""
    job = queue_job(db, make_capture(db, slug), "t-gpu-slow")
    job.status = RunStatus.IN_PROGRESS
    db.add(
        JobStep(
            job_id=job.id,
            stage_id="train",
            ordinal=1,
            impl="t_gpu_slow",
            attempt=1,
            status=RunStatus.IN_PROGRESS,
            metrics={"detached": True, "remoteCalls": {"": {"id": "fc-1", "provider": "modal"}}},
        )
    )
    db.commit()
    return job


def test_short_of_room_the_worker_still_re_attaches_to_a_detached_run_whose_workdir_is_here(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below `WORKER_MIN_FREE_GB` with nothing left to evict, the worker claimed nothing --
    including a run a deploy had detached, which needs no download: its inputs are in its
    workdir and its GPU call is running, waiting to be re-attached to. That one is still
    claimed; a queued job, and a detached one whose workdir is on another volume (it
    would start over from the upload), still wait for room."""
    queued = queue_job(db, make_capture(db, "queued"), "t-three")
    here = detached_job(db, "here")
    elsewhere = detached_job(db, "elsewhere")
    (tmp_path / "runs" / str(here.id) / "inputs").mkdir(parents=True)
    worker = Worker(sessions, storage, config(tmp_path, min_free_gb=5))
    monkeypatch.setattr(worker._disk, "free_bytes", lambda: GB)

    assert worker.claim() == here.id
    assert worker.claim() is None

    db.expire_all()
    for waiting in (queued, elsewhere):
        row = db.get(Job, waiting.id)
        assert row is not None and row.claimed_by is None
