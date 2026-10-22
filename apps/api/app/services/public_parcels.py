"""What keeps the public parcel lookup cheap: a short-lived answer cache and a cap on how many
lookups run at once across every visitor.

The lookup (`POST /api/v1/public/parcels/near`, app/api/v1/public.py) asks one or two county
services and makes one database query for an answer. The per-client bucket (ratelimit.py)
stops one visitor asking too often; it cannot stop many visitors, or one slow county, from
holding every worker and database connection at once. So:

* **Answers are cached** for `CACHE_TTL_S`, keyed by the point rounded to `CACHE_DECIMALS`
  (about 11 m) and the radius. A visitor tapping around one spot, or two visitors at the same
  address, ask the counties once. Only settled answers are kept: a county that did not answer
  (`unavailable`) is asked again next time.
* **At most `MAX_CONCURRENT` lookups run at once.** A lookup that cannot start within
  `QUEUE_WAIT_S` is refused as busy (503 with Retry-After) rather than queued behind a slow
  county.

In-process and per machine, like the rate limits, and kept on `app.state` for the same reason:
a test's app starts empty.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from app.schemas.land_selection import CandidateResult, PublicParcelRequest

#: Seconds a settled answer is reused. County parcel layers change monthly at most.
CACHE_TTL_S = 600.0
#: Answers kept before the oldest is dropped.
CACHE_ENTRIES = 2_000
#: Decimal places of longitude and latitude in a cache key: 4 is about 11 m.
CACHE_DECIMALS = 4
#: Lookups running at once across all visitors. Each holds a worker thread for up to two county
#: timeouts (land_selection.PUBLIC_TIMEOUT_S), and the API has a few dozen.
MAX_CONCURRENT = 4
#: Seconds a lookup waits for a free slot before it is refused as busy.
QUEUE_WAIT_S = 2.0
#: What a refused lookup is told to wait (s).
BUSY_RETRY_S = 5

CacheKey = tuple[float, float, float]


class Busy(Exception):  # noqa: N818 - a condition, answered 503, not an error
    """Every lookup slot is taken."""


class PublicParcels:
    """The cache and the concurrency cap one app uses, made in `create_app`."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_concurrent: int = MAX_CONCURRENT,
        queue_wait_s: float = QUEUE_WAIT_S,
    ) -> None:
        self._clock = clock
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._queue_wait_s = queue_wait_s
        self._answers: OrderedDict[CacheKey, tuple[float, CandidateResult]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(request: PublicParcelRequest) -> CacheKey:
        lon, lat = request.point.coordinates[:2]
        return (round(lon, CACHE_DECIMALS), round(lat, CACHE_DECIMALS), request.radius_m)

    def lookup(self, key: CacheKey, ask: Callable[[], CandidateResult]) -> CandidateResult:
        """The cached answer for `key`, or `ask()`'s, run in a free slot (raises `Busy`)."""
        cached = self._cached(key)
        if cached is not None:
            return cached
        if not self._slots.acquire(timeout=self._queue_wait_s):
            raise Busy
        try:
            answer = ask()
        finally:
            self._slots.release()
        if answer.status != "unavailable":
            with self._lock:
                self._answers[key] = (self._clock(), answer)
                self._answers.move_to_end(key)
                while len(self._answers) > CACHE_ENTRIES:
                    self._answers.popitem(last=False)
        return answer

    def _cached(self, key: CacheKey) -> CandidateResult | None:
        with self._lock:
            entry = self._answers.get(key)
            if entry is None:
                return None
            stored, answer = entry
            if self._clock() - stored > CACHE_TTL_S:
                del self._answers[key]
                return None
            return answer
