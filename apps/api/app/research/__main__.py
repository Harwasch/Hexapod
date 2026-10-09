from __future__ import annotations

import argparse
import time

from app.config import get_settings
from app.db import get_session_factory
from app.research.worker import ResearchWorker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent land research worker")
    parser.add_argument("--once", action="store_true", help="Process at most one queued run")
    args = parser.parse_args()
    settings = get_settings()
    worker = ResearchWorker(get_session_factory(settings.database_url), settings)
    while True:
        worked = worker.run_once()
        if args.once:
            return
        if not worked:
            time.sleep(2)


if __name__ == "__main__":
    main()
