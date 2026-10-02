"""The claim loop: `SKIP LOCKED` to pick the row, a committed lease to hold it.

A0 #2 settled the shape of this and it is not open for reinvention, so the reasoning is
repeated where the code is:

    `SELECT ... FOR UPDATE SKIP LOCKED` held for the job's duration releases in 0.02 s
    when the worker is SIGKILLed — and holds the row for about **2 h 51 min** when the
    worker is merely frozen (SIGSTOP, a wedged host, a network partition), because that
    is when the default TCP keepalives finally give up. The open transaction also pins
    the vacuum horizon the whole time: 50 000 dead tuples measured unreclaimable. A
    committed claim plus a lease reclaims in 3.06 s however the worker died.

So `SKIP LOCKED` selects exactly one row and the `UPDATE` around it **commits
immediately**. Holding the job afterwards is a matter of pushing `lease_expires_at`
forward (:func:`heartbeat`), not of holding a lock. Every timestamp here comes from the
database's clock rather than from a worker's, so two workers cannot disagree about whether
a lease has lapsed -- and from `statement_timestamp()`, not `now()`, because `now()` is
when the *transaction* began, and a supervisor's session can hold one open for as long as
a download takes (see :func:`_clock`).
"""

from __future__ import annotations

import enum
import logging
import threading
import uuid
from collections.abc import Collection
from datetime import datetime

from sqlalchemy import ColumnElement, and_, exists, func, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.models import Job
from app.models.enums import RunStatus

log = logging.getLogger("app.worker")


def claimable(now: datetime | None = None) -> ColumnElement[bool]:
    """The predicate: a job nobody has started, or one whose lease has lapsed.

    Spelled once, here, and used by both the claim loop and
    ``tests/test_capture_models.py::test_lease_expiry_query_selects_unstarted_and_abandoned_jobs``
    — a second copy of it in a test is a second copy that can disagree with the worker.

    `now` is for tests that want to reason about a fixed moment. The worker passes
    nothing and gets the database's clock.
    """
    moment: ColumnElement[datetime] | datetime = _clock() if now is None else now
    return or_(
        Job.status == RunStatus.NOT_STARTED,
        and_(
            Job.status == RunStatus.IN_PROGRESS,
            # A NULL lease is claimable, not stuck. `NULL < now()` is NULL, so without
            # this an in-progress row whose lease was cleared -- by `release`, when a
            # worker shuts down mid-run -- would never be picked up by anybody again.
            or_(Job.lease_expires_at.is_(None), Job.lease_expires_at < moment),
        ),
    )


def _clock() -> ColumnElement[datetime]:
    """The database's time at the start of the statement being run.

    Not `now()`, which is the start of the *transaction*. A heartbeat is one `UPDATE` on
    a session that may have opened its transaction long before: the supervisor reads the
    job, then downloads the capture -- a multi-gigabyte video, minutes -- and its first
    heartbeat after that, in the same transaction, wrote `now() + lease` as a lease that
    had already run out. It reported HELD, and a second slot polling the queue could claim
    the job a moment later -- half of what dead-lettered two-slot jobs on 2026-09-27
    (`LeaseKeeper` says the other half; `test_a_heartbeat_renews_from_when_it_runs...`).
    """
    clock: ColumnElement[datetime] = func.statement_timestamp()
    return clock


def _expiry(lease_s: float) -> ColumnElement[datetime]:
    """`statement_timestamp() + lease`, computed in the database. `make_interval` rather
    than string interpolation, so the lease is a bound parameter like everything else."""
    interval = func.make_interval(0, 0, 0, 0, 0, 0, lease_s)
    expiry: ColumnElement[datetime] = _clock() + interval
    return expiry


def claim_next(
    db: Session,
    *,
    worker_id: str,
    lease_s: float,
    recipes: Collection[str] | None = None,
) -> Job | None:
    """Take the oldest claimable job, or return None.

    One statement and one commit. The inner `SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1`
    is what stops two workers choosing the same row — the second one skips it rather than
    blocking — and the `UPDATE` that wraps it is what turns the momentary lock into a
    lease that outlives the transaction.

    `recipes`, when given, narrows "oldest claimable" to jobs of those recipes: the
    oldest *splat-ingest* rather than the oldest job. It is how a CPU-only slot
    (`WORKER_CPU_ONLY_SLOTS`, `loop.Worker`) takes a one-minute ingest that would
    otherwise queue behind a two-hour training run, without ever taking the training
    run itself. It is a filter inside the same `SKIP LOCKED` select, not a second query,
    so the claim stays one statement and a filtered slot and an unfiltered one can no
    more both win a row than two workers can. None means any recipe; an empty set means
    none, and asks the database nothing.
    """
    if recipes is not None and not recipes:
        return None
    eligible = claimable()
    if recipes is not None:
        eligible = and_(eligible, Job.recipe.in_(sorted(recipes)))
    candidate = (
        select(Job.id)
        .where(eligible)
        .order_by(Job.created_at, Job.id)
        .limit(1)
        .with_for_update(skip_locked=True)
        .scalar_subquery()
    )
    claimed = db.execute(
        update(Job)
        .where(Job.id == candidate)
        .values(
            status=RunStatus.IN_PROGRESS,
            claimed_by=worker_id,
            claimed_at=func.now(),
            lease_expires_at=_expiry(lease_s),
        )
        .returning(Job.id)
        .execution_options(synchronize_session=False)
    ).scalar_one_or_none()
    db.commit()
    if claimed is None:
        return None
    return db.get(Job, claimed)


def anything_claimable(db: Session) -> bool:
    """Is there a job any worker could claim right now? One `EXISTS`, then commit.

    What an idle worker asks immediately before it exits (`loop._Idle`): the last poll
    found nothing, but a job committed between that poll and the exit would otherwise sit
    with nobody running to take it until the next enqueue wakes a machine. This narrows
    that window to the moment between this statement and the process ending; the API's
    wake call closes the rest (`app/services/worker_wake.py`). Unfiltered by recipe on
    purpose: a worker exits as a whole, so a job any of its slots could take keeps it up.
    """
    found = bool(db.scalar(select(exists().where(claimable()))))
    db.commit()
    return found


class Heartbeat(enum.Enum):
    """What a heartbeat found."""

    #: The lease was pushed forward; this worker still owns the job.
    HELD = "held"
    #: Somebody cancelled the job. Stop the run and leave the status alone.
    CANCELLED = "cancelled"
    #: The job is no longer this worker's — reclaimed, or already finished elsewhere.
    LOST = "lost"


def heartbeat(db: Session, job_id: uuid.UUID, *, worker_id: str, lease_s: float) -> Heartbeat:
    """Push the lease forward, and say what happened if it could not be pushed.

    Conditional on `claimed_by` and on the job still being in progress, so a worker that
    was frozen long enough to be reclaimed learns it has lost the job the next time it
    wakes up, instead of writing over the worker that took it.

    Commits — both when it renews and when it does not. A heartbeat that left its
    transaction open would be the very thing A0 measured as pinning the vacuum horizon.
    """
    renewed = db.execute(
        update(Job)
        .where(
            Job.id == job_id,
            Job.claimed_by == worker_id,
            Job.status == RunStatus.IN_PROGRESS,
        )
        .values(lease_expires_at=_expiry(lease_s))
        .returning(Job.id)
        .execution_options(synchronize_session=False)
    ).scalar_one_or_none()
    db.commit()
    if renewed is not None:
        return Heartbeat.HELD
    status = db.scalar(select(Job.status).where(Job.id == job_id))
    db.commit()
    return Heartbeat.CANCELLED if status is RunStatus.CANCELLED else Heartbeat.LOST


def holder(db: Session, job_id: uuid.UUID) -> str:
    """Who has a job now and in what state, for the log line of a slot that lost it."""
    found = db.execute(
        select(Job.status, Job.claimed_by, Job.lease_expires_at).where(Job.id == job_id)
    ).one_or_none()
    db.commit()
    if found is None:
        return "status=missing"
    status, owner, expiry = found
    return f"status={status.value} claimed_by={owner} lease_expires_at={expiry}"


class LeaseKeeper:
    """Keeps one job's lease alive from a thread of its own, for as long as it is held.

    The supervisor's loop beats the lease every `poll_s` -- when it is in the loop. It is
    also the thread that fetches the capture before the first stage and uploads every
    finished stage's artifacts and log before it writes the step's row, and for a
    multi-gigabyte video or normalize's hundred frames going to R2 that is longer than
    the lease. With one worker on the machine nothing else was polling, so the lapse
    went unseen. With two slots in one process the other slot polls every `idle_s`,
    finds the lease lapsed within a second, claims the job and runs its stage again as
    the next attempt -- and the two slots take it from each other until the stage has
    "been attempted 3 times": the 2026-09-27 dead-letters, which
    `tests/test_worker_concurrency.py` reproduces with slow uploads and a slow download.

    So the lease no longer depends on what the supervising thread happens to be doing.
    This renews it on its own session every `interval_s`, which makes the lease mean
    what A0 wanted from it -- *this worker process is alive and still has the job* --
    while the supervisor's own heartbeat stays what notices a cancel or a loss. It needs
    no care at the end: the renewal is conditional on `claimed_by` and `in-progress`, so
    a finished, cancelled, released or reclaimed job is never renewed, and on the first
    renewal that finds it so this thread stops.

    A frozen process freezes this thread with it (SIGSTOP, a wedged host) and a killed
    one takes it along, so a worker that has really gone still loses the job after one
    lease, exactly as before. What it does not do is give up a job whose supervisor is
    alive but stuck -- an upload that never returns -- which the heartbeat never did
    either for a recipe process that is stuck; a cancel still stops that one.
    """

    def __init__(
        self,
        sessions: sessionmaker[Session],
        job_id: uuid.UUID,
        *,
        worker_id: str,
        lease_s: float,
        interval_s: float,
    ) -> None:
        self._sessions = sessions
        self._job_id = job_id
        self._worker_id = worker_id
        self._lease_s = lease_s
        self._interval_s = interval_s
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"lease-{worker_id}-{job_id}", daemon=True
        )

    def __enter__(self) -> LeaseKeeper:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._done.set()
        self._thread.join()

    def _run(self) -> None:
        db = self._sessions()
        try:
            while not self._done.wait(self._interval_s):
                try:
                    beat = heartbeat(
                        db, self._job_id, worker_id=self._worker_id, lease_s=self._lease_s
                    )
                except Exception:
                    # A dropped connection is the next renewal's problem, not the job's:
                    # the lease has two more intervals to run before anyone may take it.
                    log.warning(
                        "lease renewal failed: job=%s holder=%s",
                        self._job_id,
                        self._worker_id,
                        exc_info=True,
                    )
                    db.rollback()
                    continue
                if beat is not Heartbeat.HELD:
                    return
        finally:
            db.close()


def release(db: Session, job_id: uuid.UUID, *, worker_id: str) -> None:
    """Let go of a job without finishing it, so the next worker need not wait out the lease.

    Used on a clean shutdown and between attempts. It clears the lease rather than the
    status: an `in-progress` job with no lease is claimable immediately, which is exactly
    what :func:`claimable` says.
    """
    db.execute(
        update(Job)
        .where(Job.id == job_id, Job.claimed_by == worker_id)
        .values(claimed_by=None, lease_expires_at=None)
        .execution_options(synchronize_session=False)
    )
    db.commit()
