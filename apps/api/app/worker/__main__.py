"""`python -m app.worker` — the long-running process C1 deploys beside the API.

SIGTERM and SIGINT set the stop flag rather than killing the process. A worker that is
supervising a run stops the recipe process on its next tick, clears the lease on the job
it holds and exits — so the next worker can take that job immediately rather than waiting
the lease out, and resumes it from the stages already in its workdir.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from types import FrameType

from app.worker.loop import Worker

log = logging.getLogger("app.worker")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once", action="store_true", help="run a single job (or exit if none is queued)"
    )
    parser.add_argument("--max-jobs", type=int, default=None, help="stop after this many jobs")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), stream=sys.stderr)

    worker = Worker.from_settings()
    stop = threading.Event()

    def _stop(signum: int, _frame: FrameType | None) -> None:
        log.info("worker %s: signal %s, finishing the current tick", worker.worker_id, signum)
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    if args.once:
        outcome = worker.run_one()
        log.info("worker %s: %s", worker.worker_id, outcome or "nothing queued")
        return 0
    log.info("worker %s: polling", worker.worker_id)
    ran = worker.run_forever(max_jobs=args.max_jobs, stop=stop)
    log.info("worker %s: ran %d job(s)", worker.worker_id, ran)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
