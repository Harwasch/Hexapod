"""Room on the workdir's volume, checked before a job is claimed rather than discovered
half way through one.

The worker's volume is 20 GB (fly.toml) and a run's workdir outlives it on purpose --
"retry from this stage" reads the stages that finished out of it. A successful run is
tidied down to its outputs, but nothing ever deleted a workdir, so the volume fills a run
at a time; and a worker that claims a 12 GB capture onto a disk with 3 GB left fails it
somewhere in `normalize` with ENOSPC, after the download, spending an attempt and an
hour for nothing.

So before a claim, `DiskGuard.room_to_claim` looks at the free space. Below
`min_free_gb` it makes room, in this order, and looks again:

1. **tidy** every finished, failed or cancelled run: drop `inputs/` (a copy of what is
   still in the bucket; `_seed` fetches it again for a retry) and each stage's `work/`
   (scratch, by A6's own definition) -- what `JobSupervisor._tidy` does at the end of a
   run, for the runs that ended before it did that;
2. **evict** whole workdirs of runs that ended more than `evict_after_days` ago, oldest
   first, until there is room. A retry of one of those starts over from the upload,
   which is the honest answer for a run nobody has touched in a week.

If that is still not enough it does not claim, and says so at error level -- once a
minute, not once a poll -- so the job stays queued (and `QUEUE_CHECK_URL`, if set,
alerts) instead of failing on a full disk. A run that is not over is never touched: only
a job whose row says it is complete, failed or cancelled, or a directory no job owns.

**Except a detached run whose workdir is here** (`resumable_here`, the 2026-10 review). A
deploy that stops the worker mid-train leaves the GPU call running for the next worker
to re-attach to, the stage's row marked `detached`, and the workdir -- inputs and all --
on this volume. Resuming it downloads nothing, and refusing it left the call to finish
for nobody, so it is claimed even when nothing else is. What its outputs need when they
come home is the same room the run was always going to need.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import ColumnElement, and_, exists, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Job, JobStep
from app.models.enums import RunStatus
from app.worker.pipeline_bridge import Workdir
from app.worker.steps import DETACHED

log = logging.getLogger("app.worker")

GB = 1024**3
#: The statuses a run is over in. Anything else may still be using its workdir.
FINISHED = (RunStatus.COMPLETE, RunStatus.ERROR, RunStatus.CANCELLED)
#: How often "not claiming: the disk is full" is said, at most.
COMPLAIN_EVERY_S = 60.0


class DiskGuard:
    """One per worker process; its slots share it, and its lock."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        root: Path,
        *,
        min_free_gb: float,
        evict_after_days: float,
    ) -> None:
        self._sessions = sessions
        self._root = root
        self._min_free = int(min_free_gb * GB)
        self._evict_after = timedelta(days=evict_after_days)
        self._lock = threading.Lock()
        self._complained = float("-inf")

    def free_bytes(self) -> int:
        self._root.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(self._root).free

    def room_to_claim(self) -> bool:
        """True when there is room for another run, after making some if there was not."""
        if self._min_free <= 0:
            return True
        with self._lock:
            if self.free_bytes() >= self._min_free:
                return True
            freed = self.make_room()
            free = self.free_bytes()
            if free >= self._min_free:
                log.warning(
                    "worker: the workdir volume was short of %.1f GB free; made room by "
                    "removing %s, %.1f GB free now",
                    self._min_free / GB,
                    ", ".join(freed) or "nothing",
                    free / GB,
                )
                return True
            now = time.monotonic()
            if now - self._complained >= COMPLAIN_EVERY_S:
                self._complained = now
                log.error(
                    "worker: NOT CLAIMING -- only %.1f GB free on %s, below the %.1f GB "
                    "WORKER_MIN_FREE_GB, and no finished run older than %s is left to "
                    "evict (a detached run whose workdir is here is still resumed). "
                    "Extend the volume (fly volumes extend) or remove workdirs.",
                    free / GB,
                    self._root,
                    self._min_free / GB,
                    self._evict_after,
                )
            return False

    def resumable_here(self) -> ColumnElement[bool] | None:
        """What a worker short of room may still claim: a run in progress, with a stage
        a deploy detached (`steps.DETACHED`: its call left running to be re-attached to),
        whose workdir is on this volume. A condition for `claim.claim_next`'s `only`, or
        None when no workdir here could be one -- then nothing is claimed at all."""
        here: list[uuid.UUID] = []
        if self._root.is_dir():
            for entry in self._root.iterdir():
                try:
                    job_id = uuid.UUID(entry.name)
                except ValueError:
                    continue
                if Workdir(entry).inputs_dir.is_dir():
                    here.append(job_id)
        if not here:
            return None
        detached = exists(
            select(JobStep.id).where(
                JobStep.job_id == Job.id,
                JobStep.status == RunStatus.IN_PROGRESS,
                JobStep.metrics.contains({DETACHED: True}),
            )
        )
        return and_(Job.id.in_(here), Job.status == RunStatus.IN_PROGRESS, detached)

    def make_room(self) -> list[str]:
        """Tidy every finished run, then evict old ones until there is room. Returns what
        was removed, for the log."""
        finished = self._finished_runs()
        removed: list[str] = []
        for job_id in sorted(finished):
            for path in _scratch(self._root / str(job_id)):
                if _remove(path):
                    removed.append(str(path.relative_to(self._root)))
        if self.free_bytes() >= self._min_free:
            return removed
        cutoff = datetime.now(tz=UTC) - self._evict_after
        old = sorted((ended, job_id) for job_id, ended in finished.items() if ended <= cutoff)
        for _, job_id in old:
            for path in (self._root / str(job_id), self._root / f"{job_id}.sandbox"):
                if _remove(path):
                    removed.append(path.name)
            if self.free_bytes() >= self._min_free:
                break
        return removed

    def _finished_runs(self) -> dict[uuid.UUID, datetime]:
        """Every workdir on the volume whose run is over, with when it ended.

        A directory no job owns any more (the row was deleted) is over too, as of when it
        was last written. Anything whose job is queued or running is left out.
        """
        on_disk: dict[uuid.UUID, Path] = {}
        if self._root.is_dir():
            for entry in self._root.iterdir():
                try:
                    on_disk[uuid.UUID(entry.name)] = entry
                except ValueError:
                    continue
        if not on_disk:
            return {}
        db = self._sessions()
        try:
            rows = db.execute(
                select(Job.id, Job.status, Job.finished_at, Job.updated_at).where(
                    Job.id.in_(list(on_disk))
                )
            ).all()
            db.commit()
        finally:
            db.close()
        known = {row.id: row for row in rows}
        finished: dict[uuid.UUID, datetime] = {}
        for job_id, path in on_disk.items():
            row = known.get(job_id)
            if row is None:
                stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
                finished[job_id] = stamp
            elif row.status in FINISHED:
                finished[job_id] = row.finished_at or row.updated_at
        return finished


def _scratch(workdir_root: Path) -> list[Path]:
    """What a finished run's workdir can lose and keep "retry from this stage": the same
    two things `JobSupervisor._tidy` drops."""
    workdir = Workdir(workdir_root)
    paths = [workdir.inputs_dir]
    if workdir.stages_dir.is_dir():
        paths += [stage / "work" for stage in sorted(workdir.stages_dir.iterdir())]
    return [path for path in paths if path.is_dir()]


def _remove(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        shutil.rmtree(path)
    except OSError:
        log.warning("worker: could not remove %s to make room", path, exc_info=True)
        return False
    return True
