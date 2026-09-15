from __future__ import annotations

import logging
import sys

from app.db import get_session_factory
from app.seed import seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

with get_session_factory()() as session:
    result = seed(session)
sys.stdout.write(f"seed complete: {result}\n")
