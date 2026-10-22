from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from collections.abc import Callable

from app.config import get_settings
from app.db import get_session_factory
from app.research import queue
from app.research.worker import ResearchWorker

log = logging.getLogger("twin.research")


def poll(
    worker: ResearchWorker,
    *,
    idle_exit_s: float,
    once: bool = False,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Back off empty claims, then exit cleanly; each completed run resets idleness.

    The last claim happens at the idle deadline before deciding to exit. Enqueue's
    delayed Machines API recheck covers a job committed just after that claim.
    """
    idle_since = clock()
    while True:
        worked = worker.run_once()
        if once:
            return
        if worked:
            idle_since = clock()
            continue
        idle_for = clock() - idle_since
        if idle_exit_s > 0 and idle_for >= idle_exit_s:
            if worker.has_pending_work():
                sleep(2)  # A locked or exhausted run must not cause a busy poll loop.
                continue
            log.info("Research idle for %.0f s; exiting until the next enqueue.", idle_for)
            return
        delay = min(30.0, 2.0 * 2 ** min(4, int(idle_for // 60)))
        sleep(min(delay, idle_exit_s - idle_for) if idle_exit_s > 0 else delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent land research worker")
    parser.add_argument("--once", action="store_true", help="Process at most one queued run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Not one line per source request.
    settings = get_settings()
    sessions = get_session_factory(settings.database_url)
    worker = ResearchWorker(sessions, settings)
    try:
        with sessions() as db:
            queue.prune_presence(db)
    except Exception as error:
        log.warning("Research worker presence pruning failed: %s", type(error).__name__)
    # A deploy or `kill` sends SIGTERM: leave through the finally below, so the worker is
    # reported stopped rather than going quietly stale. A run cut off mid-way is recovered by
    # its lease (queue.claim), as it always was.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    log.info(
        "Research worker %s polling for queued runs; active runs check in every %d s.",
        worker.worker_id,
        queue.CHECK_IN_EVERY_S,
    )
    try:
        poll(worker, idle_exit_s=settings.research_idle_exit_s, once=args.once)
    except KeyboardInterrupt:
        pass  # Ctrl+C in `pnpm dev:research`: stop quietly, through the finally.
    finally:
        worker.stop()
        log.info("Research worker %s stopped.", worker.worker_id)


if __name__ == "__main__":
    main()
