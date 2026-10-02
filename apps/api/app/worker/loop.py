"""The poll loop: claim a job, run it, claim the next one -- in one slot or several.

A slot is exactly what a whole worker used to be: it claims one job, supervises it to a
terminal status (its own `JobSupervisor`, its own database session, its own recipe
process, its own lease kept alive by a `claim.LeaseKeeper` thread of its own) and then
claims the next. `WORKER_CONCURRENCY` slots run side by side in one process, each a
thread, so a machine whose jobs spend most of their time waiting on a GPU somewhere else
is not idle while a second capture queues.

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

**CPU-only slots** (`WORKER_CPU_ONLY_SLOTS`) are slots like any other that claim only a
recipe with no `gpu:` stage -- `splat-ingest`, whose whole run is a minute of packing on
this machine. With one general slot and one of these, an ingest no longer queues behind a
two-hour training run, and two training runs still never run at once: a second
concurrent video would be a second GPU billed and a second capture on a 20 GB volume
sized for one. Which recipes qualify is read from the recipes when the worker starts
(`cpu_only_recipes`), so a recipe that gains a `gpu:` stage stops qualifying without a
list to keep in step.

**Idle.** An empty queue is polled every `idle_s` for the first `idle_backoff_after_s`,
then less and less often, up to `idle_max_s` (`poll_delay`): a queue read every two
seconds forever is a database that never scales to zero. And once nothing has run or been
claimed for `idle_exit_s` the worker exits 0, the machine stops (fly.toml restarts it only
on a failure), and the API starts it again when it next queues a job
(`app/services/worker_wake.py`). The race at the edge of that -- a job committed just as
the worker decides to go -- is narrowed here, by asking the database once more under the
lock every claim takes (`_Idle.should_exit`), and closed there, by the wake call looking
again a little later at a machine it found still running.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.db import get_session_factory
from app.storage import ObjectStorage, build_storage
from app.storage.factory import build_publish_storage
from app.worker.child import resolve_recipe
from app.worker.claim import anything_claimable, claim_next
from app.worker.cloud import check_dispatchable
from app.worker.config import WorkerConfig
from app.worker.disk import DiskGuard
from app.worker.pipeline_bridge import recipe_dir
from app.worker.reaper import Reaper
from app.worker.runner import JobSupervisor, Terminal

log = logging.getLogger("app.worker")


def cpu_only_recipes(local_dir: Path | None = None) -> frozenset[str]:
    """Every recipe this worker can run whose stages declare no `gpu:` -- what a CPU-only
    slot may claim.

    Read from the recipes themselves, by the same lookup a job's recipe goes through
    (`child.resolve_recipe`: the deployment's own directory first, then the shipped
    ones), so a deployment that overrides `splat-ingest` with a GPU stage of its own is
    judged by its version. `gpu:` is the pipeline's only routing signal and a run's
    `params` cannot add one (`Recipe.with_params` overrides parameters, not stages), so
    a recipe's answer here is every one of its runs' answer. A recipe that will not load
    is left out with a warning rather than stopping the worker: a general slot claims it
    and dead-letters it with the pipeline's own message, as it always has.
    """
    names = {path.stem for path in recipe_dir().glob("*.yaml")}
    if local_dir is not None and local_dir.is_dir():
        names |= {path.stem for path in local_dir.glob("*.yaml")}
    found: set[str] = set()
    for name in sorted(names):
        try:
            recipe = resolve_recipe(name, str(local_dir) if local_dir else None)
        except Exception as error:
            # A `RecipeError` for a bad shape, but also yaml's own error for bad syntax
            # (the loader does not wrap it) and an unreadable file: none of them is a
            # reason for the worker not to start.
            log.warning(
                "worker: recipe %s does not load; no CPU-only slot takes it: %s", name, error
            )
            continue
        if not any(stage.gpu for stage in recipe.stages):
            found.add(name)
    return frozenset(found)


def poll_delay(idle_for: float, *, idle_s: float, backoff_after_s: float, max_s: float) -> float:
    """How long an idle slot waits before asking for work again.

    `idle_s` while the worker has been idle for less than `backoff_after_s` -- a job
    queued a moment after the last one finished is picked up as fast as it always was --
    then doubling every further `backoff_after_s`, capped at `max_s`. With the defaults
    (2 s, 60 s, 30 s): 2 s for the first minute, then 4, 8 and 16 s a minute at a time,
    and 30 s from the fifth minute on. Seconds of latency on a job that arrives at a
    worker that has been idle for minutes, against a database asked thirty times a
    minute on nobody's behalf.
    """
    if max_s <= idle_s:
        return idle_s
    if backoff_after_s <= 0:
        return max_s
    if idle_for < backoff_after_s:
        return idle_s
    periods = min(32, int((idle_for - backoff_after_s) // backoff_after_s) + 1)
    return float(min(max_s, idle_s * 2**periods))


def pool_size_for(slots: int) -> int:
    """Database connections a worker with `slots` slots can hold at once.

    Each slot holds up to three while it works -- its supervisor's session, its
    `LeaseKeeper`'s, and its claim session briefly before them -- plus the one the idle
    check opens. SQLAlchemy's default (5, and 10 more on overflow) covers four slots;
    past that a slot would wait on the pool, and a heartbeat that waits is a lease that
    lapses.
    """
    return max(5, 3 * slots + 1)


class Worker:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        storage: ObjectStorage,
        config: WorkerConfig,
        publish_storage: ObjectStorage | None = None,
        *,
        cpu_recipes: frozenset[str] | None = None,
    ) -> None:
        self._sessions = session_factory
        self._storage = storage
        self._publish_storage = publish_storage
        #: True once `run_forever` has returned because the worker was idle, not stopped.
        self.exited_idle = False
        self._cpu_recipes: frozenset[str] = frozenset()
        if config.cpu_only_slots > 0:
            # At start-up, once: which recipes a CPU-only slot may take. `cpu_recipes` is
            # for a test that wants to say so rather than read the shipped recipes.
            self._cpu_recipes = (
                cpu_recipes if cpu_recipes is not None else cpu_only_recipes(config.recipe_dir)
            )
            if not self._cpu_recipes:
                log.warning(
                    "worker: WORKER_CPU_ONLY_SLOTS=%d, but every recipe declares a `gpu:` "
                    "stage; running without CPU-only slots",
                    config.cpu_only_slots,
                )
                config = replace(config, cpu_only_slots=0)
        self._config = config
        # One for the process: its slots share the volume, and the eviction's lock.
        self._disk = DiskGuard(
            session_factory,
            config.workdir_root,
            min_free_gb=config.min_free_gb,
            evict_after_days=config.evict_after_days,
        )
        self._reaper = Reaper(session_factory, storage, config)

    @staticmethod
    def from_settings(settings: Settings | None = None) -> Worker:
        resolved = settings or get_settings()
        # Before anything else, and before a job is claimed: see `check_dispatchable`.
        check_dispatchable(resolved.worker_cloud_providers)
        config = WorkerConfig.from_settings(resolved)
        return Worker(
            session_factory=get_session_factory(
                resolved.database_url, pool_size=pool_size_for(config.total_slots)
            ),
            storage=build_storage(resolved),
            config=config,
            publish_storage=build_publish_storage(resolved),
        )

    @property
    def worker_id(self) -> str:
        return self._config.worker_id

    @property
    def slots(self) -> int:
        return self._config.total_slots

    @property
    def cpu_recipes(self) -> frozenset[str]:
        """What the CPU-only slots claim; empty when there are none."""
        return self._cpu_recipes if self._config.cpu_only_slots > 0 else frozenset()

    def slot_configs(self) -> list[WorkerConfig]:
        """One configuration per slot: the general ones first, then the CPU-only ones."""
        if self.slots == 1:
            return [self._config]
        general = max(1, self._config.concurrency)
        return [self._config.for_slot(slot) for slot in range(general)] + [
            self._config.for_slot(general + slot, recipes=self._cpu_recipes)
            for slot in range(self._config.cpu_only_slots)
        ]

    def claim(self, config: WorkerConfig | None = None) -> uuid.UUID | None:
        """Take the next claimable job, committing the claim immediately (A0 #2).

        Not when the workdir's volume is short of room (`app.worker.disk`): the job
        stays queued -- an empty claim, as far as the loop is concerned -- rather than
        failing on a full disk after its download.
        """
        resolved = config or self._config
        if not self._disk.room_to_claim():
            return None
        db = self._sessions()
        try:
            job = claim_next(
                db,
                worker_id=resolved.worker_id,
                lease_s=resolved.lease_s,
                recipes=resolved.recipes,
            )
            if job is None:
                return None
            # With `runner.JobSupervisor`'s "lost job" line, what tells a reclaim from a
            # first claim in the logs: the same job id claimed a second time.
            log.info("worker %s: claimed job %s (%s)", resolved.worker_id, job.id, job.recipe)
            return job.id
        finally:
            db.close()

    def reap_abandoned(self) -> list[str]:
        """Cancel the remote calls of runs that are over (`app.worker.reaper`): what the
        worker does when it starts and every `reap_every_s`. Never raises -- a provider
        that cannot be reached is the next pass's to try again, not a reason to stop
        claiming -- and returns the ids of the calls it cancelled."""
        try:
            return self._reaper.reap()
        except Exception:
            log.exception("worker %s: the pass over abandoned remote calls failed", self.worker_id)
            return []

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
        """Run until asked to stop, until `max_jobs` have been run, or until idle for
        `idle_exit_s` -- in every slot.

        `max_jobs` is what makes the loop testable and what `--once` is built on; without
        it this is the long-lived process C1 deploys beside the API. It counts jobs
        across all slots, and a slot reserves its place in it *before* claiming, so a
        two-slot worker asked for one job claims one.

        An idle exit sets `stop` (or the loop's own flag) once no slot holds a job, so a
        slot asleep between polls wakes and returns at once rather than at its next poll;
        `exited_idle` says that is why this returned.
        """
        halt = stop or threading.Event()
        budget = _Budget(max_jobs)
        idle = _Idle()
        configs = self.slot_configs()
        for config in configs:
            if config.recipes is not None:
                log.info(
                    "worker %s: a CPU-only slot, claiming only %s",
                    config.worker_id,
                    ", ".join(sorted(config.recipes)),
                )
        # The calls of runs that ended while no worker was watching are cancelled now,
        # before the first claim -- this start may be the API waking the worker for a
        # cancel, which is the whole of its errand -- and every `reap_every_s` after.
        self.reap_abandoned()
        finished = threading.Event()
        reaping = threading.Thread(
            target=self._reap_every, args=(halt, finished), name="worker-reaper", daemon=True
        )
        reaping.start()
        try:
            if len(configs) == 1:
                self._slot(configs[0], halt, budget, idle)
                return budget.done
            threads = [
                threading.Thread(
                    target=self._slot,
                    args=(config, halt, budget, idle),
                    name=f"worker-slot-{slot}",
                    daemon=True,
                )
                for slot, config in enumerate(configs)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            return budget.done
        finally:
            finished.set()
            reaping.join(timeout=30)

    def _reap_every(self, halt: threading.Event, finished: threading.Event) -> None:
        """The reaper's tick, on a thread of its own: a slot may be two hours into a job."""
        while not finished.wait(self._config.reap_every_s):
            if halt.is_set():
                return
            self.reap_abandoned()

    def _slot(
        self, config: WorkerConfig, halt: threading.Event, budget: _Budget, idle: _Idle
    ) -> None:
        """One slot's loop: claim, supervise, repeat. A slot never takes the others down:
        `JobSupervisor.run` already turns a failure into the job's error, and anything
        that escapes it here is logged and the slot carries on."""
        while not halt.is_set():
            if not budget.reserve():
                return
            if not idle.enter():
                # Another slot has decided the worker is going: nothing more is claimed.
                budget.unreserve()
                return
            outcome: Terminal | None = None
            try:
                outcome = self.run_one(stop=halt, config=config)
            except Exception:
                log.exception("worker %s: the slot failed outside a job", config.worker_id)
            finally:
                idle.leave(ran=outcome is not None)
            if outcome is None:
                budget.unreserve()
                if self._idle_exit(config, idle):
                    halt.set()
                    return
                halt.wait(
                    poll_delay(
                        idle.idle_for(),
                        idle_s=config.idle_s,
                        backoff_after_s=config.idle_backoff_after_s,
                        max_s=config.idle_max_s,
                    )
                )
                continue
            budget.finish()

    def _idle_exit(self, config: WorkerConfig, idle: _Idle) -> bool:
        """Has the worker been idle for `idle_exit_s`, with the queue still empty?"""
        if config.idle_exit_s <= 0:
            return False

        def still_queued() -> bool:
            db = self._sessions()
            try:
                return anything_claimable(db)
            except Exception:
                # Unsure is not empty: a worker that cannot read the queue stays up.
                log.warning(
                    "worker %s: could not check the queue before exiting",
                    config.worker_id,
                    exc_info=True,
                )
                return True
            finally:
                db.close()

        if not idle.should_exit(config.idle_exit_s, still_queued):
            return False
        if not self.exited_idle:
            self.exited_idle = True
            log.info(
                "worker %s: nothing has run or been queued for %.0f s; exiting so the "
                "machine can stop (WORKER_IDLE_EXIT_S=0 polls forever). Queueing a job "
                "starts it again.",
                self._config.worker_id,
                config.idle_exit_s,
            )
        return True


class _Idle:
    """Whether the whole worker is idle, and since when: shared by every slot.

    A slot is *busy* from the moment it starts a claim until its job ends, so "nothing in
    progress" is a count of zero and a claim can never be under way unseen. The idle
    clock restarts when a job ends, not when a poll finds nothing. The exit decision and
    the start of every claim take the same lock, so no slot can begin a claim between the
    final look at the queue and the decision to go.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._busy = 0
        self._since = clock()
        self._exiting = False

    def enter(self) -> bool:
        """Mark a slot busy before it claims; False once the worker has decided to exit."""
        with self._lock:
            if self._exiting:
                return False
            self._busy += 1
            return True

    def leave(self, *, ran: bool) -> None:
        with self._lock:
            self._busy -= 1
            if ran:
                self._since = self._clock()

    def idle_for(self) -> float:
        """Seconds since the last job ended (or the worker started); 0 while one runs."""
        with self._lock:
            return 0.0 if self._busy else self._clock() - self._since

    def should_exit(self, after_s: float, still_queued: Callable[[], bool]) -> bool:
        """True when no slot is busy, none has run anything for `after_s`, and the queue
        -- asked once more, under the lock -- is still empty. Every call after the first
        True is True too, so the other slots follow."""
        with self._lock:
            if self._exiting:
                return True
            if self._busy or self._clock() - self._since < after_s:
                return False
            if still_queued():
                return False
            self._exiting = True
            return True


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
