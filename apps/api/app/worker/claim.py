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
forward (:func:`heartbeat`), not of holding a lock. Every timestamp here comes from
`now()` on the database rather than from a worker's clock, so two workers cannot disagree
about whether a lease has lapsed.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import ColumnElement, and_, func, or_, select, update
from sqlalchemy.orm import Session

from app.models import Job
from app.models.enums import RunStatus


def claimable(now: datetime | None = None) -> ColumnElement[bool]:
    """The predicate: a job nobody has started, or one whose lease has lapsed.

    Spelled once, here, and used by both the claim loop and
    ``tests/test_capture_models.py::test_lease_expiry_query_selects_unstarted_and_abandoned_jobs``
    — a second copy of it in a test is a second copy that can disagree with the worker.

    `now` is for tests that want to reason about a fixed moment. The worker passes
    nothing and gets the database's clock.
    """
    moment: ColumnElement[datetime] | datetime = func.now() if now is None else now
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


def _expiry(lease_s: float) -> ColumnElement[datetime]:
    """`now() + lease`, computed in the database. `make_interval` rather than string
    interpolation, so the lease is a bound parameter like everything else."""
    interval = func.make_interval(0, 0, 0, 0, 0, 0, lease_s)
    expiry: ColumnElement[datetime] = func.now() + interval
    return expiry


def claim_next(db: Session, *, worker_id: str, lease_s: float) -> Job | None:
    """Take the oldest claimable job, or return None.

    One statement and one commit. The inner `SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1`
    is what stops two workers choosing the same row — the second one skips it rather than
    blocking — and the `UPDATE` that wraps it is what turns the momentary lock into a
    lease that outlives the transaction.
    """
    candidate = (
        select(Job.id)
        .where(claimable())
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
