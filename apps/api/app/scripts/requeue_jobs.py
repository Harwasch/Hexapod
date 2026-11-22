"""Queue one fresh bundle reconstruction per capture after the ARKit pose fix.

Run on the deployed API machine as user api with its own DATABASE_URL. No storage
reads or write token are needed. Dry run unless --apply; each capture commits
independently, so an interrupted batch can be repeated safely. Never invoke this
as part of deploy: applying queues paid pipeline work.
"""

from __future__ import annotations

import argparse
import sys
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Job
from app.schemas.job import JobCreate
from app.services import capture_bundles, jobs, worker_wake
from app.services.errors import ConflictError

# Fixed correction identity. Do not change it to retry a failed correction job:
# use the existing job retry command/route, preserving this tool's idempotency.
CORRECTION = "hex-capture-arkit-pose-v1"


def job_id(capture_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"hexapod:{CORRECTION}:{capture_id}")


def requeue(db: Session, capture_id: uuid.UUID, *, apply: bool = False) -> Job | None:
    """Return the correction's job, or None for a dry run that would create one.

    Serialize concurrent invocations on the capture row. The deterministic job ID is
    committed in the same transaction as the new job, so even a lost response cannot
    enqueue another reconstruction. create_job also enforces one active run per capture.
    """
    capture_bundles.completed_capture(db, capture_id)
    existing = db.get(Job, job_id(capture_id))
    if existing is not None:
        return existing
    running = db.scalar(
        select(Job.id).where(Job.capture_id == capture_id, Job.status.in_(jobs.ACTIVE_STATUSES))
    )
    if running is not None:
        raise ConflictError(f"capture {capture_id} already has job {running} queued or running")
    if not apply:
        return None
    return jobs.create_job(
        db,
        capture_id,
        JobCreate(recipe="photo-reconstruct", params=capture_bundles.PARAMS),
        job_id=job_id(capture_id),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="append", type=uuid.UUID, required=True)
    parser.add_argument("--apply", action="store_true", help="queue fresh jobs and wake the worker")
    args = parser.parse_args(argv)
    from app.db import get_session_factory

    for capture_id in dict.fromkeys(args.capture):
        with get_session_factory()() as db:
            job = requeue(db, capture_id, apply=args.apply)
            identifier = job.id if job is not None else job_id(capture_id)
            status = job.status.value if job is not None else "would queue"
            wake = args.apply and job is not None and job.status in jobs.ACTIVE_STATUSES
        sys.stdout.write(f"capture {capture_id}: job {identifier} ({status})\n")
        if wake:
            # create_job committed before this call; repeated applies can recover a
            # failed wake too. A CLI has no FastAPI background-task response boundary.
            worker_wake.after_enqueue()
    if not args.apply:
        sys.stdout.write("dry run; nothing written and no worker wake requested\n")
    sys.stdout.write("REQUEUE-JOBS-OK\n")


if __name__ == "__main__":
    main()
