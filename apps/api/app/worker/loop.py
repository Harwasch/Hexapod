"""The poll loop: claim a job, run it, claim the next one.

Deliberately one job at a time. A worker that ran several would have to multiplex its
heartbeats, and the way to run two jobs at once here is to start two workers — which the
claim loop already supports, because that is the thing `SKIP LOCKED` is for.
"""

from __future__ import annotations

import threading
import uuid

from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.db import get_session_factory
from app.storage import ObjectStorage, build_storage
from app.storage.factory import build_publish_storage
from app.worker.claim import claim_next
from app.worker.config import WorkerConfig
from app.worker.runner import JobSupervisor, Terminal


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
        return Worker(
            session_factory=get_session_factory(resolved.database_url),
            storage=build_storage(resolved),
            config=WorkerConfig.from_settings(resolved),
            publish_storage=build_publish_storage(resolved),
        )

    @property
    def worker_id(self) -> str:
        return self._config.worker_id

    def claim(self) -> uuid.UUID | None:
        """Take the next claimable job, committing the claim immediately (A0 #2)."""
        db = self._sessions()
        try:
            job = claim_next(db, worker_id=self._config.worker_id, lease_s=self._config.lease_s)
            return job.id if job is not None else None
        finally:
            db.close()

    def run_one(self, stop: threading.Event | None = None) -> Terminal | None:
        """Claim and run one job. None when the queue was empty."""
        job_id = self.claim()
        if job_id is None:
            return None
        return JobSupervisor(
            self._sessions, self._storage, self._config, self._publish_storage
        ).run(job_id, stop=stop)

    def run_forever(
        self, *, max_jobs: int | None = None, stop: threading.Event | None = None
    ) -> int:
        """Run until asked to stop, or until `max_jobs` have been run.

        `max_jobs` is what makes the loop testable and what `--once` is built on; without
        it this is the long-lived process C1 deploys beside the API.
        """
        done = 0
        halt = stop or threading.Event()
        while not halt.is_set() and (max_jobs is None or done < max_jobs):
            if self.run_one(stop=halt) is None:
                halt.wait(self._config.idle_s)
                continue
            done += 1
        return done
