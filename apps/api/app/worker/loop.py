"""The poll loop: claim a job, run it, claim the next one -- in one slot or several.

A slot is exactly what a whole worker used to be: it claims one job, supervises it to a
terminal status (its own `JobSupervisor`, its own database session, its own recipe
process, its own heartbeat on its own lease) and then claims the next. `WORKER_CONCURRENCY`
slots run side by side in one process, each a thread, so a machine whose jobs spend most
of their time waiting on a GPU somewhere else is not idle while a second capture queues.

What keeps that correct is that nothing about a job is shared between slots:

* **the claim** is `claim_next`'s `SKIP LOCKED` update, which two slots can no more
  both win than two workers can;
* **the identity** is per slot (`WorkerConfig.for_slot`, `host:pid/<n>`), so a lease one
  slot let lapse and another reclaimed is *lost* to the first -- its next heartbeat fails
  on `claimed_by` -- rather than silently owned by both;
* **cancellation** is read by the heartbeat of the slot holding that job and stops that
  job's recipe process only;
* **stopping** (SIGTERM) sets one event every slot watches: each stops its own recipe
  process on its next tick and clears its own lease.

The slots' threads only wait -- on a child's pipe, a database round trip, a sleep -- so
the GIL is not what bounds this. Memory is: every slot's job has a recipe process of its
own, and the worker's README says what each costs and what `WORKER_CONCURRENCY` the
2 GB machine can take.
"""

from __future__ import annotations

import logging
import threading
import uuid

from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.db import get_session_factory
from app.storage import ObjectStorage, build_storage
from app.storage.factory import build_publish_storage
from app.worker.claim import claim_next
from app.worker.cloud import check_dispatchable
from app.worker.config import WorkerConfig
from app.worker.runner import JobSupervisor, Terminal

log = logging.getLogger("app.worker")


class Worker:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        storage: ObjectStorage,
        config: WorkerConfig,
        publish_storage: ObjectStorage | None = None,
    ) -> None:
        self._sessions = session_factory
        self._storage = storage
        self._config = config
        self._publish_storage = publish_storage

    @staticmethod
    def from_settings(settings: Settings | None = None) -> Worker:
        resolved = settings or get_settings()
        # Before anything else, and before a job is claimed: see `check_dispatchable`.
        check_dispatchable(resolved.worker_cloud_providers)
        return Worker(
            session_factory=get_session_factory(resolved.database_url),
            storage=build_storage(resolved),
            config=WorkerConfig.from_settings(resolved),
            publish_storage=build_publish_storage(resolved),
        )

    @property
    def worker_id(self) -> str:
        return self._config.worker_id

    @property
    def slots(self) -> int:
        return max(1, self._config.concurrency)

    def claim(self, config: WorkerConfig | None = None) -> uuid.UUID | None:
        """Take the next claimable job, committing the claim immediately (A0 #2)."""
        resolved = config or self._config
        db = self._sessions()
        try:
            job = claim_next(db, worker_id=resolved.worker_id, lease_s=resolved.lease_s)
            return job.id if job is not None else None
        finally:
            db.close()

    def run_one(
        self, stop: threading.Event | None = None, *, config: WorkerConfig | None = None
    ) -> Terminal | None:
        """Claim and run one job. None when the queue was empty."""
        resolved = config or self._config
        job_id = self.claim(resolved)
        if job_id is None:
            return None
        return JobSupervisor(self._sessions, self._storage, resolved, self._publish_storage).run(
            job_id, stop=stop
        )

    def run_forever(
        self, *, max_jobs: int | None = None, stop: threading.Event | None = None
    ) -> int:
        """Run until asked to stop, or until `max_jobs` have been run, in every slot.

        `max_jobs` is what makes the loop testable and what `--once` is built on; without
        it this is the long-lived process C1 deploys beside the API. It counts jobs
        across all slots, and a slot reserves its place in it *before* claiming, so a
        two-slot worker asked for one job claims one.
        """
        halt = stop or threading.Event()
        budget = _Budget(max_jobs)
        if self.slots == 1:
            self._slot(self._config, halt, budget)
            return budget.done
        threads = [
            threading.Thread(
                target=self._slot,
                args=(self._config.for_slot(slot), halt, budget),
                name=f"worker-slot-{slot}",
                daemon=True,
            )
            for slot in range(self.slots)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return budget.done

    def _slot(self, config: WorkerConfig, halt: threading.Event, budget: _Budget) -> None:
        """One slot's loop: claim, supervise, repeat. A slot never takes the others down:
        `JobSupervisor.run` already turns a failure into the job's error, and anything
        that escapes it here is logged and the slot carries on."""
        while not halt.is_set():
            if not budget.reserve():
                return
            try:
                outcome = self.run_one(stop=halt, config=config)
            except Exception:
                log.exception("worker %s: the slot failed outside a job", config.worker_id)
                outcome = None
            if outcome is None:
                budget.unreserve()
                halt.wait(config.idle_s)
                continue
            budget.finish()


class _Budget:
    """`max_jobs` shared between slots: a place is reserved before a claim, given back
    if the queue was empty, and counted once the job has run."""

    def __init__(self, limit: int | None) -> None:
        self._limit = limit
        self._reserved = 0
        self.done = 0
        self._lock = threading.Lock()

    def reserve(self) -> bool:
        with self._lock:
            if self._limit is not None and self._reserved >= self._limit:
                return False
            self._reserved += 1
            return True

    def unreserve(self) -> None:
        with self._lock:
            self._reserved -= 1

    def finish(self) -> None:
        with self._lock:
            self.done += 1
