from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from app.config import get_settings
from app.db import get_session_factory
from app.research import queue
from app.research.worker import ResearchWorker

log = logging.getLogger("twin.research")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent land research worker")
    parser.add_argument("--once", action="store_true", help="Process at most one queued run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
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
        "Research worker %s polling for queued runs; checking in every %d s.",
        worker.worker_id,
        queue.CHECK_IN_EVERY_S,
    )
    try:
        while True:
            worked = worker.run_once()
            if args.once:
                return
            if not worked:
                time.sleep(2)
    except KeyboardInterrupt:
        pass  # Ctrl+C in `pnpm dev:research`: stop quietly, through the finally.
    finally:
        worker.stop()
        log.info("Research worker %s stopped.", worker.worker_id)


if __name__ == "__main__":
    main()
