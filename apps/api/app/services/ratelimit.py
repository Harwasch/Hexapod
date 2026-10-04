"""Per-client token buckets for the few routes whose cost is the attack.

Most of this API is cheap to call and open to read, and is left alone. Three things are
not cheap:

* **The phone key.** Every route behind it hashes the supplied key with 200,000 rounds of
  PBKDF2 (app/services/phone_key.py) -- deliberately, so a leaked hash is slow to attack.
  The same cost made every request with a wrong key ~0.1 s of CPU on a shared-CPU machine,
  with nothing stopping a loop of them, and nothing slowing a guesser down. A bucket here
  counts *wrong* keys: a request takes a token before the hash and gives it back when the
  key was right, so a phone holding the key is never limited however much it does, and a
  guesser gets `PHONE_KEY` attempts a minute.
* **Storage reconciliation** walks up to 50,000 objects in the bucket per call.
* **Step logs** are read out of object storage on every request.

In-process and per-machine, on purpose: the API runs as one Fly machine (fly.toml), and a
limit that needs Redis to work is a limit nobody deploys. A second machine would give each
client a bucket per machine -- looser, never stricter. Buckets live on `app.state`, so an
app made in a test starts with full ones.

**None of the phone page's polling is limited.** apps/web/src/upload/main.ts polls
`GET /jobs` and `GET /captures` every `POLL_MS` (10 s) while a run is going, and calls
`/phone/check` once, when the key is typed; tests/test_ratelimit.py holds both facts.
"""

from __future__ import annotations

import ipaddress
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Request

from app.config import Settings


@dataclass(frozen=True)
class Limit:
    """`burst` requests at once, refilled at `per_minute`."""

    burst: int
    per_minute: float


#: Wrong phone keys per client. A person mistyping gets ten tries a minute; a guesser at
#: ~59 bits of key gets nowhere at 14,400 a day.
PHONE_KEY = Limit(burst=10, per_minute=10)
#: A bucket walk. The console asks only when somebody presses the button.
RECONCILE = Limit(burst=3, per_minute=4)
#: Step logs. The console reads one when its drawer opens, and never polls them.
STEP_LOG = Limit(burst=60, per_minute=120)

#: Clients remembered per bucket before the least recently seen is forgotten. Forgetting
#: one only ever gives it a full bucket again, so the bound is on memory, not on safety.
MAX_CLIENTS = 10_000


class RateLimited(Exception):  # noqa: N818 - a condition, raised as a response, not an error
    """Too many requests from one client. Answered 429 with `Retry-After`."""

    def __init__(self, retry_after_s: float) -> None:
        super().__init__(f"Too many requests. Try again in {max(1, math.ceil(retry_after_s))} s.")
        self.retry_after_s = retry_after_s


class TokenBucket:
    """One `Limit`, kept per client key, safe to share across the threadpool."""

    def __init__(
        self,
        limit: Limit,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = MAX_CLIENTS,
    ) -> None:
        self.limit = limit
        self._rate = limit.per_minute / 60.0
        self._clock = clock
        self._max_clients = max_clients
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def _level(self, key: str, now: float) -> float:
        tokens, updated = self._buckets.get(key, (float(self.limit.burst), now))
        return min(float(self.limit.burst), tokens + (now - updated) * self._rate)

    def take(self, key: str) -> None:
        """Spend one token for `key`, or raise `RateLimited` saying when one will be back."""
        with self._lock:
            now = self._clock()
            tokens = self._level(key, now)
            if tokens < 1.0:
                self._store(key, tokens, now)
                raise RateLimited((1.0 - tokens) / self._rate)
            self._store(key, tokens - 1.0, now)

    def give_back(self, key: str) -> None:
        """Return the token a request took: it turned out not to count (a right key)."""
        with self._lock:
            now = self._clock()
            self._store(key, min(float(self.limit.burst), self._level(key, now) + 1.0), now)

    def _store(self, key: str, tokens: float, now: float) -> None:
        self._buckets[key] = (tokens, now)
        self._buckets.move_to_end(key)
        while len(self._buckets) > self._max_clients:
            self._buckets.popitem(last=False)


class RateLimits:
    """The buckets one app uses, made in `create_app` and kept on `app.state`."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.phone_key = TokenBucket(PHONE_KEY, clock=clock)
        self.reconcile = TokenBucket(RECONCILE, clock=clock)
        self.step_log = TokenBucket(STEP_LOG, clock=clock)


#: The header Fly's proxy sets on every request, which is `API_CLIENT_IP_HEADER`'s default.
FLY_CLIENT_IP = "fly-client-ip"
#: How much of an IPv6 address is one client. A subscriber is routinely handed a whole
#: /64 -- it is the smallest subnet SLAAC works in -- and every one of its 2^64 addresses
#: is theirs to send from, so a bucket per address was a fresh bucket per request for
#: anybody who cared to rotate. An IPv4 address stays one client.
IPV6_CLIENT_PREFIX = 64


def client_key(request: Request, settings: Settings) -> str:
    """Who is asking, as well as this process can tell: an IPv4 address, or an IPv6 /64.

    Behind Fly every connection arrives from Fly's proxy, so the socket's peer address is
    the same for everyone and a bucket keyed on it would be one bucket for the world.
    Fly puts the real client's address in `Fly-Client-IP` and sets that header itself on
    every request, so through Fly it cannot be forged -- and *only* through Fly: anywhere
    else it is a header any client may send, with a new value per request, and keying on
    it there was no limit at all. So the default header is believed only when the process
    is on Fly (`FLY_APP_NAME`, which Fly sets on every machine and nothing else does), and
    the socket's peer is the client everywhere else. `API_CLIENT_IP_HEADER` set to another
    proxy's header is that operator's statement that their proxy sets it, and is believed;
    set empty, the socket's peer is always the client.
    """
    header = settings.api_client_ip_header.strip()
    believed = bool(header) and (header.lower() != FLY_CLIENT_IP or bool(settings.fly_app_name))
    if believed:
        forwarded = request.headers.get(header, "").strip()
        if forwarded:
            return _client_of(forwarded)
    return _client_of(request.client.host) if request.client is not None else "unknown"


def _client_of(address: str) -> str:
    """An address as the client it stands for: IPv4 as it is, IPv4 inside IPv6 as the
    IPv4 address it carries, any other IPv6 address as its /64. Anything that is not an
    address at all is kept as it came, so it is still a key of its own."""
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return address
    if isinstance(parsed, ipaddress.IPv6Address):
        if parsed.ipv4_mapped is not None:
            return str(parsed.ipv4_mapped)
        network = ipaddress.IPv6Network((parsed, IPV6_CLIENT_PREFIX), strict=False)
        return str(network)
    return str(parsed)


def limits_of(request: Request) -> RateLimits:
    limits: RateLimits = request.app.state.rate_limits
    return limits
