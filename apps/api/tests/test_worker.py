"""The worker: claiming, leasing, running, cancelling, retrying and giving up.

These run the real thing. `JobSupervisor` starts a real `app.worker.child` process, which
runs a real recipe out of `tests/recipes/` with real stage implementations from
`tests/worker_stages.py`; the two tests about a worker dying start `python -m app.worker`
as a process and kill it. Nothing here mocks the executor, because the questions being
asked — does a killed worker's job come back, does cancel stop a running stage — are
exactly the questions a mock would answer by construction.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from moto import mock_aws
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Artifact, Capture, Job, JobStep, Site
from app.models.enums import CaptureKind, CaptureStatus, RunStatus
from app.services import jobs as job_service
from app.storage import NullStorage, ObjectStorage, S3Storage
from app.worker.claim import Heartbeat, claim_next, heartbeat, release
from app.worker.config import WorkerConfig
from app.worker.loop import Worker
from app.worker.runner import JobSupervisor, Terminal
from tests.conftest import TEST_DATABASE_URL

BUCKET = "twin-worker-test"
RECIPES = Path(__file__).resolve().parent / "recipes"
API_ROOT = Path(__file__).resolve().parents[1]

# The suite must not wait on production-sized timings: the lease, the heartbeat and the
# retry backoff are all configuration, and the point of the test is the behaviour.
FAST_LEASE_S = 1.5
FAST_POLL_S = 0.2


@pytest.fixture
def sessions(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def storage() -> Iterator[S3Storage]:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield S3Storage(
            bucket=BUCKET,
            endpoint_url=None,
            access_key="key",
            secret_key="secret",
            region="us-east-1",
            public_base_url="https://cdn.example.com/twin-worker-test",
        )


def config(tmp_path: Path, worker_id: str = "worker-a", **overrides: object) -> WorkerConfig:
    defaults: dict[str, object] = {
        "workdir_root": tmp_path / "runs",
        "worker_id": worker_id,
        "runner": "local",
        "recipe_dir": RECIPES,
        "impl_modules": ("tests.worker_stages",),
        "lease_s": FAST_LEASE_S,
        "poll_s": FAST_POLL_S,
        "idle_s": 0.05,
        "retry_backoff_s": 0.0,
        "terminate_grace_s": 2.0,
    }
    defaults.update(overrides)
    return WorkerConfig(**defaults)  # type: ignore[arg-type]


def make_capture(db: Session, slug: str = "back-paddock") -> Capture:
    capture = Capture(slug=slug, name="Back paddock", kind=CaptureKind.GAUSSIAN_SPLAT)
    db.add(capture)
    db.commit()
    return capture


def queue_job(db: Session, capture: Capture, recipe: str = "t-three") -> Job:
    job = Job(capture_id=capture.id, recipe=recipe, recipe_version="1", params={})
    db.add(job)
    db.commit()
    return job


def steps_by_stage(db: Session, job_id: uuid.UUID) -> dict[str, JobStep]:
    db.expire_all()
    rows = db.scalars(select(JobStep).where(JobStep.job_id == job_id)).all()
    return {row.stage_id: row for row in rows}


def wait_until(predicate: object, timeout: float = 20.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(interval)
    return False


# --------------------------------------------------------------------------------------
# The claim loop
# --------------------------------------------------------------------------------------


def test_two_workers_never_claim_the_same_job(
    db: Session, engine: Engine, sessions: sessionmaker[Session]
) -> None:
    """`SKIP LOCKED` picks the row; the claim commits immediately. Four workers racing on
    eight jobs must end with eight distinct claims and no job claimed twice."""
    capture = make_capture(db)
    for index in range(8):
        db.add(Job(capture_id=capture.id, recipe=f"t-three-{index}", recipe_version="1"))
    db.commit()

    claimed: list[tuple[str, uuid.UUID]] = []
    guard = threading.Lock()
    start = threading.Barrier(4)

    def worker(name: str) -> None:
        session = sessions()
        try:
            start.wait(timeout=10)
            while True:
                job = claim_next(session, worker_id=name, lease_s=120)
                if job is None:
                    return
                with guard:
                    claimed.append((name, job.id))
        finally:
            session.close()

    threads = [threading.Thread(target=worker, args=(f"worker-{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    ids = [job_id for _, job_id in claimed]
    assert len(ids) == 8, f"every job claimed exactly once, got {len(ids)}"
    assert len(set(ids)) == 8, "a job was claimed twice"
    db.expire_all()
    assert all(job.status is RunStatus.IN_PROGRESS for job in db.scalars(select(Job)).all())


def test_a_job_whose_lease_lapses_is_claimable_again(
    db: Session, sessions: sessionmaker[Session]
) -> None:
    """A0 #2's whole point: the claim is a lease, so a worker that stopped existing does
    not hold the row until TCP gives up on it."""
    capture = make_capture(db)
    job = queue_job(db, capture)
    first, second = sessions(), sessions()

    assert claim_next(first, worker_id="worker-a", lease_s=0.3) is not None
    assert claim_next(second, worker_id="worker-b", lease_s=30) is None, "the lease is held"

    time.sleep(0.5)
    reclaimed = claim_next(second, worker_id="worker-b", lease_s=30)

    assert reclaimed is not None and reclaimed.id == job.id
    assert reclaimed.claimed_by == "worker-b"
    first.close()
    second.close()


def test_a_heartbeat_holds_the_lease_and_reports_what_it_could_not_hold(
    db: Session, sessions: sessionmaker[Session]
) -> None:
    capture = make_capture(db)
    job = queue_job(db, capture)
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=0.3) is not None

    assert heartbeat(session, job.id, worker_id="worker-a", lease_s=30) is Heartbeat.HELD
    # Still held after the original lease would have expired, because it was pushed.
    time.sleep(0.5)
    other = sessions()
    assert claim_next(other, worker_id="worker-b", lease_s=30) is None

    job_service.cancel_job(db, job.id)
    assert heartbeat(session, job.id, worker_id="worker-a", lease_s=30) is Heartbeat.CANCELLED

    # And a job taken by somebody else reads as lost rather than being written over.
    release(session, job.id, worker_id="worker-a")
    db.expire_all()
    stolen = db.get(Job, job.id)
    assert stolen is not None
    stolen.status = RunStatus.IN_PROGRESS
    stolen.claimed_by = "worker-b"
    db.commit()
    assert heartbeat(session, job.id, worker_id="worker-a", lease_s=30) is Heartbeat.LOST
    session.close()
    other.close()


# --------------------------------------------------------------------------------------
# Running a recipe
# --------------------------------------------------------------------------------------


def run_job(
    sessions: sessionmaker[Session],
    storage: ObjectStorage,
    job_id: uuid.UUID,
    cfg: WorkerConfig,
) -> Terminal:
    return JobSupervisor(sessions, storage, cfg).run(job_id)


def test_step_rows_appear_as_stages_run_not_in_one_batch_at_the_end(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """The panel polls `GET /jobs`; a stage list that materialises at the end is not live.

    The supervisor runs in a thread while this session watches the table, and the test
    passes only if it *observed* the run part-done — one or two completed steps — rather
    than inferring it from the final state.
    """
    capture = make_capture(db)
    job = queue_job(db, capture, "t-three")
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=30) is not None
    session.close()

    observations: list[tuple[int, int]] = []
    finished = threading.Event()

    def watch() -> None:
        watcher = sessions()
        try:
            while not finished.is_set():
                # expire_on_commit is off, so identity-mapped rows would otherwise be
                # read back at the values this session first saw.
                watcher.expire_all()
                rows = watcher.scalars(select(JobStep).where(JobStep.job_id == job.id)).all()
                observations.append(
                    (len(rows), sum(1 for r in rows if r.status is RunStatus.COMPLETE))
                )
                watcher.commit()
                time.sleep(0.02)
        finally:
            watcher.close()

    watcher_thread = threading.Thread(target=watch, daemon=True)
    watcher_thread.start()
    outcome = run_job(sessions, storage, job.id, config(tmp_path))
    finished.set()
    watcher_thread.join(timeout=5)

    assert outcome == "complete"
    partial = {complete for _, complete in observations if 0 < complete < 3}
    assert partial, f"never saw the run part-finished; observed {sorted(set(observations))}"
    growing = {total for total, _ in observations}
    assert growing & {1, 2}, "the step rows all appeared at once"
    assert len(steps_by_stage(db, job.id)) == 3


def test_a_finished_run_records_steps_artifacts_and_logs(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    capture = make_capture(db)
    job = queue_job(db, capture, "t-three")
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=30) is not None
    session.close()

    assert run_job(sessions, storage, job.id, config(tmp_path)) == "complete"

    db.expire_all()
    finished = db.get(Job, job.id)
    assert finished is not None
    assert finished.status is RunStatus.COMPLETE
    assert finished.finished_at is not None
    assert finished.duration_s is not None and finished.duration_s >= 0
    assert finished.lease_expires_at is None

    steps = steps_by_stage(db, job.id)
    assert sorted(steps) == ["one", "three", "two"]
    assert [steps[name].ordinal for name in ("one", "two", "three")] == [0, 1, 2]
    for step in steps.values():
        assert step.status is RunStatus.COMPLETE
        assert step.attempt == 1
        assert step.log_key is not None
        # The log is in object storage, not in the database.
        assert storage.get_object(step.log_key).decode()
        assert step.metrics["runner"] == "local"
        assert step.metrics["runs"] == 1

    artifacts = db.scalars(select(Artifact).join(JobStep).where(JobStep.job_id == job.id)).all()
    assert {a.storage_key for a in artifacts} == {
        f"runs/{job.id}/one/first.json",
        f"runs/{job.id}/two/second.json",
        f"runs/{job.id}/three/third.json",
    }
    for artifact in artifacts:
        assert artifact.checksum is not None and artifact.checksum.startswith("sha256:")
        assert storage.get_object(artifact.storage_key)


def test_the_register_stage_describes_and_the_worker_registers(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """The decision: `register` writes `registration.json`, the worker performs it.

    The pipeline never opens a session or an HTTP client; the site, its asset and the
    capture's georeference all land here, in the process that already holds the claim.
    """
    capture = make_capture(db, slug="orchard")
    job = queue_job(db, capture, "t-ingest")
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=30) is not None
    session.close()

    assert run_job(sessions, storage, job.id, config(tmp_path)) == "complete"

    db.expire_all()
    registered = db.get(Capture, capture.id)
    assert registered is not None
    assert registered.status is CaptureStatus.COMPLETE
    assert registered.site_id is not None
    assert registered.georef_method is not None and registered.georef_method.value == "manual"
    # `manual_placement` writes scaleSource "source", which is not a ScaleSource. An
    # unknown scale source is what `unresolved` means.
    assert registered.scale_source is not None
    assert registered.scale_source.value == "unresolved"
    assert registered.uncertainty_m == pytest.approx(1.5)

    site = db.get(Site, registered.site_id)
    assert site is not None
    assert site.slug == "orchard-run"
    assert site.name == "The Orchard"
    assert len(site.assets) == 1
    assert str(site.assets[0].source["url"]).endswith(f"runs/{job.id}/package/splat/tileset.json")
    # And the tileset really is in the bucket at that key.
    assert json.loads(storage.get_object(f"runs/{job.id}/package/splat/tileset.json"))


def test_a_stage_that_fails_every_time_dead_letters_and_says_why(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    capture = make_capture(db)
    job = queue_job(db, capture, "t-doomed")
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=30) is not None
    session.close()

    assert run_job(sessions, storage, job.id, config(tmp_path, max_attempts=2)) == "error"

    db.expire_all()
    dead = db.get(Job, job.id)
    assert dead is not None
    assert dead.status is RunStatus.ERROR
    assert "attempted 2 times" in (dead.error or "")
    assert "'two'" in (dead.error or "")
    assert "broken on purpose" in (dead.error or "")
    steps = steps_by_stage(db, job.id)
    assert steps["one"].status is RunStatus.COMPLETE
    assert steps["two"].status is RunStatus.ERROR
    assert steps["two"].attempt == 2, "the budget was spent, not exceeded"
    assert steps["two"].preempted_at is not None
    # The capture says so too, rather than sitting at `not-started` forever.
    assert db.get(Capture, capture.id).status is CaptureStatus.ERROR  # type: ignore[union-attr]


def test_a_stage_resumes_from_its_checkpoint_on_the_next_attempt(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """A6 keeps `checkpoint/` across attempts and wipes `out/`. This is what that buys:
    the second attempt finishes, and the stage before it is not run again."""
    capture = make_capture(db)
    job = queue_job(db, capture, "t-resume")
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=30) is not None
    session.close()

    assert run_job(sessions, storage, job.id, config(tmp_path)) == "complete"

    steps = steps_by_stage(db, job.id)
    assert steps["two"].attempt == 2
    assert steps["two"].status is RunStatus.COMPLETE
    assert steps["one"].attempt == 1
    workdir = tmp_path / "runs" / str(job.id)
    # The skipped stage's implementation ran exactly once across both attempts.
    assert (workdir / "stages" / "one" / "work" / "runs.txt").read_text() == "x\n"
    resumed = json.loads((workdir / "stages" / "two" / "out" / "resumed.json").read_text())
    assert resumed == {"attempt": 2, "resumedFromCheckpoint": True, "stage": "two"}


def test_cancelling_a_job_stops_the_stage_that_is_running(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """The stage sleeps for 30 s. Cancellation has to reach it mid-stage, not between
    stages, so the supervisor kills the process running it — within one poll interval."""
    capture = make_capture(db)
    job = queue_job(db, capture, "t-slow")
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=30) is not None
    session.close()

    result: list[Terminal] = []
    runner = threading.Thread(
        target=lambda: result.append(run_job(sessions, storage, job.id, config(tmp_path))),
        daemon=True,
    )
    runner.start()
    assert wait_until(
        lambda: (
            steps_by_stage(db, job.id).get("two") is not None
            and steps_by_stage(db, job.id)["two"].status is RunStatus.IN_PROGRESS
        )
    ), "the slow stage never started"

    started = time.monotonic()
    job_service.cancel_job(db, job.id)
    runner.join(timeout=20)
    elapsed = time.monotonic() - started

    assert result == ["cancelled"]
    # One poll interval (0.2 s here, 2 s in production) plus however long the child takes
    # to die on SIGTERM. Bounded well under a second, and nowhere near the 30 s the stage
    # asked to sleep for -- which is the whole point of running the recipe in a process
    # that can be signalled rather than in a thread that cannot.
    # Measured at 0.15 s with this configuration.
    assert elapsed < 3, f"cancellation took {elapsed:.2f}s"
    db.expire_all()
    cancelled = db.get(Job, job.id)
    assert cancelled is not None
    assert cancelled.status is RunStatus.CANCELLED
    steps = steps_by_stage(db, job.id)
    assert steps["one"].status is RunStatus.COMPLETE
    assert steps["two"].status is RunStatus.CANCELLED
    # The part-finished workdir is left where it is: the completed stage's outputs are
    # still there, and the cancelled stage produced no artifact rows.
    workdir = tmp_path / "runs" / str(job.id)
    assert (workdir / "stages" / "one" / "out" / "first.json").is_file()
    assert not db.scalars(
        select(Artifact).join(JobStep).where(JobStep.job_id == job.id, JobStep.stage_id == "two")
    ).all()


# --------------------------------------------------------------------------------------
# A worker that dies
# --------------------------------------------------------------------------------------


def spawn_worker(tmp_path: Path, worker_id: str) -> subprocess.Popen[str]:
    """A real `python -m app.worker`, against the test database."""
    env = dict(os.environ)
    env.update(
        {
            "DATABASE_URL": TEST_DATABASE_URL,
            "WORKER_WORKDIR": str(tmp_path / "runs"),
            "WORKER_RECIPE_DIR": str(RECIPES),
            "WORKER_IMPL_MODULES": "tests.worker_stages",
            "WORKER_RUNNER": "local",
            "WORKER_LEASE_S": str(FAST_LEASE_S),
            "WORKER_POLL_S": str(FAST_POLL_S),
            "WORKER_IDLE_S": "0.05",
            "WORKER_MAX_ATTEMPTS": "3",
            "WORKER_RETRY_BACKOFF_S": "0",
            "PYTHONPATH": str(API_ROOT),
            "HOSTNAME": worker_id,
            # No bucket: moto only exists inside the test process, and the repository's
            # .env points at a MinIO that is not running here. The worker degrades to
            # no logs and no artifact rows, which the test below does not ask about.
            "OBJECT_STORAGE_ENDPOINT_URL": "",
            "OBJECT_STORAGE_BUCKET": "",
            "OBJECT_STORAGE_ACCESS_KEY": "",
            "OBJECT_STORAGE_SECRET_KEY": "",
        }
    )
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "app.worker", "--once"],
        cwd=str(API_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )


def test_a_worker_killed_mid_stage_leaves_a_job_another_worker_picks_up(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """The case A0 measured, end to end.

    A real worker claims the job and starts the slow stage. The whole process group is
    SIGKILLed — a dead container, not a polite shutdown. Nothing runs to tidy up, so the
    job stays `in-progress` until its lease lapses, and then it is simply claimable
    again. The worker that takes it skips the stage that already completed and re-runs
    the one that was interrupted, as attempt 2.
    """
    capture = make_capture(db)
    job = queue_job(db, capture, "t-slow")

    victim = spawn_worker(tmp_path, "worker-victim")
    try:
        assert wait_until(
            lambda: (
                steps_by_stage(db, job.id).get("two") is not None
                and steps_by_stage(db, job.id)["two"].status is RunStatus.IN_PROGRESS
            )
        ), "the victim worker never got to the slow stage"
        os.killpg(os.getpgid(victim.pid), signal.SIGKILL)
    finally:
        victim.wait(timeout=10)

    db.expire_all()
    stranded = db.get(Job, job.id)
    assert stranded is not None
    assert stranded.status is RunStatus.IN_PROGRESS, "a killed worker leaves the row as it was"
    assert stranded.claimed_by is not None
    assert stranded.lease_expires_at is not None

    # Nobody may take it while the lease stands...
    taker = sessions()
    assert claim_next(taker, worker_id="worker-b", lease_s=30) is None
    # ...and everybody may once it lapses.
    assert wait_until(
        lambda: claim_next(taker, worker_id="worker-b", lease_s=120) is not None,
        timeout=FAST_LEASE_S + 5,
    ), "the lease never lapsed"
    taker.close()

    # Swap the 30 s sleep for a run that will finish, so the reclaim can be seen through.
    assert run_job(
        sessions,
        storage,
        job.id,
        config(tmp_path, worker_id="worker-b", recipe_dir=RECIPES),
    ) in ("complete", "error", "cancelled")

    steps = steps_by_stage(db, job.id)
    assert steps["one"].status is RunStatus.COMPLETE
    assert steps["one"].attempt == 1, "the completed stage was not run again"
    assert steps["two"].attempt == 2, "the interrupted stage is on its second attempt"
    workdir = tmp_path / "runs" / str(job.id)
    assert (workdir / "stages" / "one" / "work" / "runs.txt").read_text() == "x\n"


def test_the_recipe_process_stops_itself_if_the_worker_disappears(
    db: Session, tmp_path: Path
) -> None:
    """A SIGKILLed worker cannot clean up, so the process running the recipe watches its
    own parent: orphaned, it exits rather than keeping a workdir another worker is about
    to resume."""
    capture = make_capture(db)
    job = queue_job(db, capture, "t-slow")

    victim = spawn_worker(tmp_path, "worker-victim")
    assert wait_until(lambda: steps_by_stage(db, job.id).get("two") is not None), (
        "the worker never started the recipe"
    )
    children = _descendants(victim.pid)
    assert children, "the worker did not start a recipe process"
    os.kill(victim.pid, signal.SIGKILL)
    victim.wait(timeout=10)

    assert wait_until(lambda: not any(_alive(pid) for pid in children), timeout=10), (
        "the orphaned recipe process kept running"
    )


def _descendants(pid: int) -> list[int]:
    if shutil.which("pgrep") is None:
        pytest.skip("pgrep is not installed, so the recipe process cannot be found")
    found = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, check=False)
    return [int(line) for line in found.stdout.split()]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


# --------------------------------------------------------------------------------------
# The loop, and the pieces that degrade
# --------------------------------------------------------------------------------------


def test_the_loop_claims_runs_and_stops_after_the_jobs_it_was_given(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    capture = make_capture(db)
    first = queue_job(db, capture, "t-three")
    worker = Worker(sessions, storage, config(tmp_path))

    assert worker.run_forever(max_jobs=1) == 1

    db.expire_all()
    assert db.get(Job, first.id).status is RunStatus.COMPLETE  # type: ignore[union-attr]
    # Nothing left to claim, so the loop stops rather than spinning.
    assert worker.run_one() is None


def test_a_worker_asked_to_stop_lets_go_of_the_job_mid_stage(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """SIGTERM is not "finish this two-hour stage first".

    The supervisor stops the recipe process and clears the lease, so the next worker can
    take the job at once instead of waiting it out -- and the stage that had completed is
    still in the workdir, so it will be skipped rather than re-run.
    """
    capture = make_capture(db)
    job = queue_job(db, capture, "t-slow")
    worker = Worker(sessions, storage, config(tmp_path))
    stop = threading.Event()
    result: list[Terminal | None] = []
    runner = threading.Thread(target=lambda: result.append(worker.run_one(stop=stop)), daemon=True)
    runner.start()
    assert wait_until(
        lambda: (
            steps_by_stage(db, job.id).get("two") is not None
            and steps_by_stage(db, job.id)["two"].status is RunStatus.IN_PROGRESS
        )
    ), "the slow stage never started"

    stop.set()
    runner.join(timeout=20)

    assert result == ["lost"]
    db.expire_all()
    handed_back = db.get(Job, job.id)
    assert handed_back is not None
    assert handed_back.status is RunStatus.IN_PROGRESS
    assert handed_back.claimed_by is None and handed_back.lease_expires_at is None
    # Claimable straight away, with no lease to wait out.
    taker = sessions()
    assert claim_next(taker, worker_id="worker-b", lease_s=30) is not None
    taker.close()


def test_a_deployment_with_no_bucket_still_runs_it_just_has_no_logs(
    db: Session, sessions: sessionmaker[Session], tmp_path: Path
) -> None:
    capture = make_capture(db)
    job = queue_job(db, capture, "t-three")
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=30) is not None
    session.close()

    assert run_job(sessions, NullStorage(), job.id, config(tmp_path)) == "complete"

    steps = steps_by_stage(db, job.id)
    assert all(step.status is RunStatus.COMPLETE for step in steps.values())
    assert all(step.log_key is None for step in steps.values())
    assert not db.scalars(select(Artifact).join(JobStep).where(JobStep.job_id == job.id)).all()


def test_a_recipe_that_does_not_resolve_is_dead_lettered_immediately(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    capture = make_capture(db)
    job = queue_job(db, capture, "t-does-not-exist")
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=30) is not None
    session.close()

    assert run_job(sessions, storage, job.id, config(tmp_path)) == "error"

    db.expire_all()
    dead = db.get(Job, job.id)
    assert dead is not None
    assert dead.status is RunStatus.ERROR
    assert "did not resolve" in (dead.error or "")
