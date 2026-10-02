"""What a worker was configured with, resolved once.

Separate from :class:`app.config.Settings` because a test wants to build one directly —
a two-second lease and no retry backoff make the lease and dead-letter paths testable in
a second rather than in a minute.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field, replace
from pathlib import Path

from app.config import REPO_ROOT, Settings, get_settings


def default_worker_id() -> str:
    """Host and pid. Two workers on one machine differ; the same worker restarted does not
    pretend to be the one that died, because the pid changed."""
    return f"{socket.gethostname()}:{os.getpid()}"


@dataclass(frozen=True)
class WorkerConfig:
    workdir_root: Path
    worker_id: str = field(default_factory=default_worker_id)
    #: "stub", "local" or "cloud" — which RunnerSet the child process builds.
    runner: str = "stub"
    #: For "cloud": where a GPU stage goes, preferred first, fallback last.
    cloud_providers: tuple[str, ...] = ()
    #: Preemptions of one stage before it is moved to the next provider in that list.
    preemptions_before_fallback: int = 2
    modal_app: str = ""
    cloud_poll_s: float = 5.0
    checkpoint_every_s: float = 60.0
    #: A directory both this worker and the machine running the stage can see. Unset,
    #: a dispatched stage's bytes go through the bucket.
    cloud_transfer_dir: Path | None = None
    recipe_dir: Path | None = None
    #: Modules the child imports before planning, so their `@stage_impl`s are registered.
    impl_modules: tuple[str, ...] = ()
    lease_s: float = 30.0
    poll_s: float = 2.0
    idle_s: float = 2.0
    max_attempts: int = 3
    #: Attempts a stage gets *on top of* `max_attempts` for having had its machine taken
    #: back. A preemption is not the stage failing, so it should not spend the budget
    #: meant for a stage that is; but the ceiling stays hard, because a placement that
    #: keeps losing the box after the fallback is a problem with the placement.
    max_preemptions: int = 4
    retry_backoff_s: float = 2.0
    #: How long a cancelled child is given to die politely before it is killed.
    terminate_grace_s: float = 5.0
    #: Drop `inputs/` and every stage's `work/` once a run has finished successfully (see
    #: `JobSupervisor._tidy`). On by default; off only for tests that read scratch files.
    tidy_finished_runs: bool = True
    #: Jobs supervised at once, one slot each (`loop.Worker.run_forever`).
    concurrency: int = 1
    #: Slots *beside* those that only claim a recipe with no `gpu:` stage (today
    #: `splat-ingest`), so a one-minute ingest is not queued behind a two-hour training run
    #: and two training runs never share the machine. `loop.Worker` works out which
    #: recipes qualify, from the recipes themselves, when it starts.
    cpu_only_slots: int = 0
    #: The recipes this configuration's slot may claim; None is any. Set per slot by
    #: `for_slot`, never from the environment: it is derived, not configured.
    recipes: frozenset[str] | None = None
    #: The idle poll backs off: `idle_s` for the first `idle_backoff_after_s` of an empty
    #: queue, then doubling a period at a time up to `idle_max_s` (`loop.poll_delay`).
    idle_backoff_after_s: float = 60.0
    idle_max_s: float = 30.0
    #: Exit, with status 0, once nothing has been running or claimed for this long, so
    #: the machine stops and the database can scale to zero; the API starts it again when
    #: it queues a job (`app/services/worker_wake.py`). 0 polls forever, which is what a
    #: worker built directly -- a test, a checkout with nothing to restart it -- gets.
    idle_exit_s: float = 0.0

    @staticmethod
    def from_settings(settings: Settings | None = None) -> WorkerConfig:
        resolved = settings or get_settings()
        root = Path(resolved.worker_workdir)
        recipes = resolved.worker_recipe_dir
        return WorkerConfig(
            workdir_root=root if root.is_absolute() else REPO_ROOT / root,
            runner=resolved.worker_runner,
            cloud_providers=tuple(resolved.worker_cloud_providers),
            preemptions_before_fallback=resolved.worker_preemptions_before_fallback,
            modal_app=resolved.worker_modal_app,
            cloud_poll_s=resolved.worker_cloud_poll_s,
            checkpoint_every_s=resolved.worker_checkpoint_every_s,
            cloud_transfer_dir=(
                Path(resolved.worker_cloud_transfer_dir)
                if resolved.worker_cloud_transfer_dir
                else None
            ),
            recipe_dir=Path(recipes) if recipes else None,
            impl_modules=tuple(resolved.worker_impl_modules),
            lease_s=resolved.worker_lease_s,
            poll_s=resolved.worker_poll_s,
            idle_s=resolved.worker_idle_s,
            max_attempts=resolved.worker_max_attempts,
            max_preemptions=resolved.worker_max_preemptions,
            retry_backoff_s=resolved.worker_retry_backoff_s,
            concurrency=resolved.worker_concurrency,
            cpu_only_slots=resolved.worker_cpu_only_slots,
            idle_backoff_after_s=resolved.worker_idle_backoff_after_s,
            idle_max_s=resolved.worker_idle_max_s,
            idle_exit_s=resolved.worker_idle_exit_s,
        )

    @property
    def total_slots(self) -> int:
        """Every slot the worker runs: the general ones and the CPU-only ones."""
        return max(1, self.concurrency) + max(0, self.cpu_only_slots)

    def for_slot(self, slot: int, *, recipes: frozenset[str] | None = None) -> WorkerConfig:
        """The configuration one slot of a concurrent worker supervises its job with.

        The same in every way but the id, which gets the slot's number: a claim, a
        heartbeat and `_still_ours` all compare `claimed_by` with this id, so two slots
        sharing one would each take the other's job for its own the moment a lease
        lapsed and was reclaimed inside the same process. With one id per slot they
        behave exactly as separate workers do, which is what `SKIP LOCKED` is built for.
        A single-slot worker keeps the plain id it always had.

        `recipes` is what a CPU-only slot is restricted to; the general slots pass None.
        Numbering runs on across both kinds (`/0` general, `/1` CPU-only, for one of
        each), so an id in `claimed_by` names exactly one slot.
        """
        if self.total_slots <= 1:
            return self
        return replace(self, worker_id=f"{self.worker_id}/{slot}", recipes=recipes)

    def workdir_for(self, job_id: object) -> Path:
        return self.workdir_root / str(job_id)

    def sandbox_for(self, job_id: object) -> Path:
        """Where an adapter that runs a stage on this machine lays out its sandbox.

        Beside the workdir rather than inside it: the workdir is the run's record and
        `prepare_stage` has rules about what lives there, whereas a sandbox is a
        provider's scratch and is nobody's artifact.
        """
        return self.workdir_root / f"{job_id}.sandbox"
