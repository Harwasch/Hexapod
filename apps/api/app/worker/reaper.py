"""Cancelling the remote calls of runs that nobody is going to supervise again.

A GPU call is stopped by the worker that watches it: the recipe process cancels it when
the job is cancelled, and the supervisor cancels whatever is still written down when it
closes a run. Both need a worker *holding the job* when the cancel lands. Until the
2026-10 review, one that was not there missed it for good: `POST /jobs/{id}/cancel` only
sets the status, a cancelled job is never claimed again, and the step's copy of its calls
(`metrics.remoteCalls`) was only ever read on a claim. So a call kept training for nobody
-- up to the function's six-hour limit on an L4 -- whenever the cancel fell in one of
these windows:

* the worker crashed: thirty seconds of lease, plus fly.toml's restart;
* the worker's restarts ran out (`[[restart]]` on-failure gives up);
* a SIGTERM that was not a deploy -- `fly machine stop`: the worker detaches the call
  for a successor, exits 0, and the machine stays stopped with nobody to succeed it.

So `Reaper.reap` runs when the worker starts and every `reap_every_s` after
(`loop.Worker`), and the API wakes a stopped worker when it cancels a job a worker had
claimed (`app/services/worker_wake.py`), so the start-up pass is what answers that
cancel. A pass cancels, by id through the provider's adapter (`CloudRunner.
cancel_recorded`), every call still recorded for a job that is over -- cancelled,
dead-lettered or finished:

* in the workdir's books (`stages/<id>/calls.json`), when the run's workdir is on this
  volume -- the full records, the request included;
* on the step rows (`metrics.remoteCalls`), which is all a worker on another machine has.

Each call cancelled is struck off and written down as reaped (`metrics.reapedCalls`, by
id, with when), so a second pass -- or a second worker's -- finds nothing to do: the
whole pass is idempotent, and a cancel of a call that is already over is a no-op at the
provider. A call whose cancel fails is kept for the next pass; one recorded more than
`GIVE_UP_AFTER_S` ago is struck off as expired without asking, because no function runs
that long and its record would otherwise be retried forever.

Nothing here touches a job some worker may still be supervising: only a job whose lease
is clear or ran out more than a lease ago. A worker that is alive sees a cancel within a
tick and cancels its own call -- with its billing -- and then clears the lease itself.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Collection, Mapping
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import ColumnElement, and_, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Job, JobStep
from app.models.enums import RunStatus
from app.storage import ObjectStorage
from app.worker.cloud import build_runners
from app.worker.config import WorkerConfig
from app.worker.pipeline_bridge import CALL_BOOK, CallBook, CallRecord, CloudRunner
from app.worker.steps import REAPED_CALLS, REMOTE_CALLS

log = logging.getLogger("app.worker")

#: The statuses a run is over in: nothing will claim it, so nothing else will cancel its
#: calls. The same three as the disk guard's (`disk.FINISHED`).
OVER = (RunStatus.COMPLETE, RunStatus.ERROR, RunStatus.CANCELLED)
#: A record older than this is struck off without a cancel: Modal's longest function
#: timeout is a day, and this deployment's is six hours (`infra/modal/app.py`).
GIVE_UP_AFTER_S = 24 * 3600.0


def calls_runner(storage: ObjectStorage, config: WorkerConfig, sandbox: Path) -> CloudRunner | None:
    """A `CloudRunner` built as the recipe process builds one, for cancelling calls by
    record from the supervisor's side; None where this worker dispatches nowhere."""
    if config.runner != "cloud" or not config.cloud_providers:
        return None
    runners = build_runners(
        storage,
        providers=config.cloud_providers,
        sandbox=sandbox,
        impl_modules=config.impl_modules,
        modal_app=config.modal_app,
        transfer_dir=config.cloud_transfer_dir,
    )
    return runners.gpu if isinstance(runners.gpu, CloudRunner) else None


class Reaper:
    """One per worker process: `reap` is safe to call from any thread, at any time."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        storage: ObjectStorage,
        config: WorkerConfig,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._sessions = sessions
        self._storage = storage
        self._config = config
        self._clock = clock

    def reap(self) -> list[str]:
        """One pass. Returns the ids of the calls it cancelled."""
        runner = calls_runner(self._storage, self._config, self._config.sandbox_for("reaper"))
        if runner is None:
            return []
        cancelled = self._reap_workdirs(runner)
        cancelled += self._reap_rows(runner, done=set(cancelled))
        if cancelled:
            log.warning(
                "worker %s: cancelled remote call(s) %s left running by jobs that are over",
                self._config.worker_id,
                ", ".join(cancelled),
            )
        return cancelled

    # --- the books in this volume's workdirs ----------------------------------------

    def _reap_workdirs(self, runner: CloudRunner) -> list[str]:
        """Every book in a workdir on this volume whose job is over (or gone)."""
        books: dict[uuid.UUID, list[Path]] = {}
        root = self._config.workdir_root
        if root.is_dir():
            for entry in root.iterdir():
                try:
                    job_id = uuid.UUID(entry.name)
                except ValueError:
                    continue
                found = sorted(entry.glob(f"stages/*/{CALL_BOOK}"))
                if found:
                    books[job_id] = found
        if not books:
            return []
        over = self._jobs_over(books)
        cancelled: list[str] = []
        for job_id, paths in books.items():
            if job_id not in over:
                continue
            outcomes: dict[str, str] = {}
            for path in paths:
                book = CallBook.read(path)
                stale = [r for r in book.calls.values() if self._expired(r)]
                live = [r for r in book.calls.values() if not self._expired(r)]
                result = runner.cancel_recorded(live)
                outcomes.update({r.handle.id: "expired" for r in stale})
                outcomes.update({cid: "cancelled" for cid, ok in result.items() if ok})
                kept = {
                    slot: record
                    for slot, record in book.calls.items()
                    if record.handle.id not in outcomes
                }
                # A call whose cancel failed stays, for the next pass -- and the book is
                # marked orphaned, so a Retry's recipe process cancels it rather than
                # re-attaching to a call of a run that was cancelled.
                CallBook(path, calls=kept, orphaned=bool(kept)).save()
                cancelled += [cid for cid, ok in result.items() if ok]
            self._strike(job_id, outcomes)
        return cancelled

    # --- the copies on the step rows -------------------------------------------------

    def _reap_rows(self, runner: CloudRunner, *, done: Collection[str]) -> list[str]:
        """Every call still on the row of a step of a job that is over."""
        db = self._sessions()
        try:
            rows = db.execute(
                select(JobStep.job_id, JobStep.metrics)
                .join(Job, Job.id == JobStep.job_id)
                .where(JobStep.metrics.has_key(REMOTE_CALLS), self._over_and_unheld())
            ).all()
            db.commit()
        finally:
            db.close()
        cancelled: list[str] = []
        for job_id, metrics in rows:
            records = _records(metrics)
            if not records:
                continue
            stale = [r for r in records if self._expired(r)]
            live = [r for r in records if not self._expired(r) and r.handle.id not in done]
            result = runner.cancel_recorded(live)
            outcomes = {r.handle.id: "expired" for r in stale}
            outcomes.update({r.handle.id: "cancelled" for r in records if r.handle.id in done})
            outcomes.update({cid: "cancelled" for cid, ok in result.items() if ok})
            for record in live:
                if not result.get(record.handle.id):
                    log.warning(
                        "worker %s: could not cancel call %s on %r of job %s, which is over; "
                        "trying again on the next pass",
                        self._config.worker_id,
                        record.handle.id,
                        record.handle.provider,
                        job_id,
                    )
            self._strike(job_id, outcomes)
            cancelled += [cid for cid, ok in result.items() if ok]
        return cancelled

    # --- helpers ----------------------------------------------------------------------

    def _over_and_unheld(self) -> ColumnElement[bool]:
        """A job that is over, and that no worker can be in the middle of closing: its
        lease cleared, or run out for longer than a lease (a worker that is alive renews
        its lease until it sees the cancel, then clears it). The database's clock, as
        every lease comparison is (`claim._clock`)."""
        grace = func.make_interval(0, 0, 0, 0, 0, 0, self._config.lease_s)
        return and_(
            Job.status.in_(OVER),
            or_(
                Job.lease_expires_at.is_(None),
                Job.lease_expires_at < func.statement_timestamp() - grace,
            ),
        )

    def _jobs_over(self, job_ids: Collection[uuid.UUID]) -> set[uuid.UUID]:
        """Of `job_ids`, those that are over and unheld -- or have no row at all."""
        db = self._sessions()
        try:
            known = set(db.scalars(select(Job.id).where(Job.id.in_(list(job_ids)))).all())
            over = set(
                db.scalars(
                    select(Job.id).where(Job.id.in_(list(job_ids)), self._over_and_unheld())
                ).all()
            )
            db.commit()
        finally:
            db.close()
        return over | (set(job_ids) - known)

    def _expired(self, record: CallRecord) -> bool:
        return bool(record.submitted_at) and self._clock() - record.submitted_at > GIVE_UP_AFTER_S

    def _strike(self, job_id: uuid.UUID, outcomes: Mapping[str, str]) -> None:
        strike_calls(self._sessions, job_id, outcomes)


def strike_calls(
    sessions: sessionmaker[Session], job_id: uuid.UUID, outcomes: Mapping[str, str]
) -> None:
    """Take the calls in `outcomes` (id -> how) off the job's step rows, and write them
    down as reaped. Under a row lock and from what the rows say now, so two passes -- two
    workers' -- cannot undo each other, and idempotent: a call already struck is not
    there to strike."""
    if not outcomes:
        return
    when = datetime.now(tz=UTC).isoformat(timespec="seconds")
    db = sessions()
    try:
        for step in db.scalars(
            select(JobStep).where(JobStep.job_id == job_id).with_for_update()
        ).all():
            metrics = dict(step.metrics or {})
            calls = metrics.get(REMOTE_CALLS)
            if not isinstance(calls, dict):
                continue
            gone = {
                slot: str(entry["id"])
                for slot, entry in calls.items()
                if isinstance(entry, dict) and entry.get("id") in outcomes
            }
            if not gone:
                continue
            reaped = dict(metrics.get(REAPED_CALLS) or {})
            reaped.update({cid: {"at": when, "how": outcomes[cid]} for cid in gone.values()})
            kept = {slot: entry for slot, entry in calls.items() if slot not in gone}
            if kept:
                metrics[REMOTE_CALLS] = kept
            else:
                metrics.pop(REMOTE_CALLS, None)
            metrics[REAPED_CALLS] = reaped
            step.metrics = metrics
        db.commit()
    finally:
        db.close()


def _records(metrics: object) -> list[CallRecord]:
    """The calls a step row's `metrics.remoteCalls` names, as records with no request."""
    calls = metrics.get(REMOTE_CALLS) if isinstance(metrics, dict) else None
    if not isinstance(calls, dict):
        return []
    records: list[CallRecord] = []
    for entry in calls.values():
        if isinstance(entry, dict) and entry.get("id"):
            try:
                records.append(CallRecord.from_dict(entry))
            except (KeyError, TypeError, ValueError):
                continue
    return records
