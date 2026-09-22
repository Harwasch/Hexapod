"""What a worker was configured with, resolved once.

Separate from :class:`app.config.Settings` because a test wants to build one directly —
a two-second lease and no retry backoff make the lease and dead-letter paths testable in
a second rather than in a minute.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
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
    #: "stub" or "local" — which RunnerSet the child process builds.
    runner: str = "stub"
    recipe_dir: Path | None = None
    #: Modules the child imports before planning, so their `@stage_impl`s are registered.
    impl_modules: tuple[str, ...] = ()
    lease_s: float = 30.0
    poll_s: float = 2.0
    idle_s: float = 2.0
    max_attempts: int = 3
    retry_backoff_s: float = 2.0
    #: How long a cancelled child is given to die politely before it is killed.
    terminate_grace_s: float = 5.0

    @staticmethod
    def from_settings(settings: Settings | None = None) -> WorkerConfig:
        resolved = settings or get_settings()
        root = Path(resolved.worker_workdir)
        recipes = resolved.worker_recipe_dir
        return WorkerConfig(
            workdir_root=root if root.is_absolute() else REPO_ROOT / root,
            runner=resolved.worker_runner,
            recipe_dir=Path(recipes) if recipes else None,
            impl_modules=tuple(resolved.worker_impl_modules),
            lease_s=resolved.worker_lease_s,
            poll_s=resolved.worker_poll_s,
            idle_s=resolved.worker_idle_s,
            max_attempts=resolved.worker_max_attempts,
            retry_backoff_s=resolved.worker_retry_backoff_s,
        )

    def workdir_for(self, job_id: object) -> Path:
        return self.workdir_root / str(job_id)
