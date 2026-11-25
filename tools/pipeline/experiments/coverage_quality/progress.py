"""Flushed phase timing and a fail-fast wall-clock budget; no GPU dependencies."""

from __future__ import annotations

import signal
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class Progress:
    # Upper bounds, further clipped to the whole evaluation deadline. Visibility is
    # cumulative across test views; its projection is checked after the first batches.
    LIMITS = {
        "setup": 600,
        "cameras": 30,
        "ply": 90,
        "lpips": 180,
        "test_render": 600,
        "visibility": 360,
        "features": 600,
        "fit": 90,
        "write": 60,
    }

    def __init__(self, seconds: float) -> None:
        if seconds <= 30:
            raise ValueError("evaluation needs >30 seconds including partial-output reserve")
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.used: dict[str, float] = {}
        self.finalization_reserve = min(120.0, max(45.0, seconds * 0.25))
        self.log(
            f"budget={seconds:.0f}s; phase ceilings={self.LIMITS}; partial-output reserve=15s; "
            f"early-work fit/write reserve={self.finalization_reserve:.0f}s"
        )

    @property
    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        sys.stdout.write(f"{stamp} +{time.monotonic() - self.started:.1f}s {message}\n")
        sys.stdout.flush()

    def require(self, seconds: float, message: str) -> None:
        if seconds + 15 > self.remaining:
            raise TimeoutError(f"{message}: need {seconds:.1f}s, remaining {self.remaining:.1f}s")

    def estimate(self, done: int, total: int, began: float, *, tail: float = 30) -> None:
        if done < 4:
            return
        elapsed = time.monotonic() - began
        projected = elapsed * (total - done) / done
        self.log(
            f"visibility {done}/{total} cameras; {elapsed / done:.3f}s/camera; "
            f"projected remaining={projected:.1f}s + tail={tail:.0f}s"
        )
        self.require(
            projected + max(tail, self.finalization_reserve),
            "visibility projection exceeds deadline",
        )
        if projected + elapsed > self.LIMITS["visibility"] - self.used.get("visibility", 0):
            raise TimeoutError("visibility projection exceeds cumulative phase ceiling")

    @contextmanager
    def phase(self, name: str, detail: str = "") -> Iterator[None]:
        self.require(1, name)
        reserve = 15 if name in {"fit", "write"} else self.finalization_reserve
        limit = min(self.LIMITS[name] - self.used.get(name, 0), self.remaining - reserve)
        if limit <= 0:
            raise TimeoutError(f"{name} exhausted phase budget")
        self.log(f"{name} start {detail}; ceiling remaining={limit:.1f}s")
        began = time.monotonic()
        previous = signal.getsignal(signal.SIGALRM)

        def expired(signum: int, frame: Any) -> None:
            raise TimeoutError(f"{name} exceeded {limit:.1f}s phase/deadline budget")

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, limit)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            elapsed = time.monotonic() - began
            self.used[name] = self.used.get(name, 0) + elapsed
            self.log(f"{name} end duration={elapsed:.1f}s")
