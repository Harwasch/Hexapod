"""Modal calls a run spawned, cancelled when the run stops short of success.

A spawned `modal.FunctionCall` runs, and bills, until it finishes, whatever becomes of the
process that spawned it. `modal run` stops its own ephemeral app when it exits, which takes
that app's containers with it, but a call on a *deployed* app goes on: `infra/modal/dream.py`
queues every clip on `hexapod-world-models` (Wan 2.2 and Cosmos on H100s) up front, and a
run that raised, or a GitHub job that was cancelled, left all of them running to the end. An
earlier run sat 4 h 25 min on failing Cosmos starts.

`SpawnedCalls` keeps every call a run spawns until its result is collected:

    calls = SpawnedCalls()
    with calls.guard():
        call = calls.spawn("clip camp-gentle", cls.clip, request)
        response = calls.get(call, timeout=3600)

* **Inside `guard()`**, SIGINT and SIGTERM raise `Stopped`, a `BaseException`, so the run's
  own `except Exception` blocks (one failed clip does not stop the rest) let it through.
* **Leaving the guard by any exception** cancels every call still outstanding -- all at
  once, each outcome logged, waited on for at most `cancel_timeout_s` -- and re-raises it;
  a stop is re-raised as `RunStopped`, an `Exception`, because `modal run` reports one and
  exits 1 but exits 0 after a KeyboardInterrupt. Leaving it normally cancels nothing.
* **`get`** collects a result. A call that fails on its own is recorded (`failures`) and its
  error raised exactly as it came: the caller reports it, and nothing here turns it into a
  cancellation. A call with no result within `timeout` is cancelled, since nobody will
  collect it, and recorded the same way.
* **Signals during `spawn`** wait until the new call is tracked, so a stop cannot fall
  between Modal creating a call and this list knowing about it.

GitHub stops a cancelled (or timed-out) step with SIGINT, SIGTERM 7.5 s later and SIGKILL
2.5 s after that, and only to the step's top process (hence `exec` in dream.yml):
`CANCEL_TIMEOUT_S` keeps the cancels inside the first interval.

The same rule as `tools/pipeline/cloud.py` (`except BaseException`: cancel every piece still
out there, then raise) and the same call as `ModalAdapter.cancel`
(`terminate_containers=True`: Modal's default cancels the input but leaves its container, and
the bill, running). Standard library only: dream.py's local entrypoint imports it where
nothing but `modal` is installed.
"""

from __future__ import annotations

import contextlib
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

#: How long the cancels of a stop are waited on, together (GitHub's SIGINT-to-SIGTERM is 7.5 s).
CANCEL_TIMEOUT_S = 5.0

SIGNALS = (signal.SIGINT, signal.SIGTERM)


class Stopped(BaseException):
    """SIGINT or SIGTERM inside `SpawnedCalls.guard`."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(signal.Signals(signum).name)


class RunStopped(RuntimeError):
    """What `guard` raises for a stop, after cancelling."""


def _say(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


@dataclass
class _Entry:
    label: str
    call: Any
    #: running (not collected, not cancelled), succeeded, failed (on its own), cancelled.
    state: str = "running"
    detail: str = ""


class SpawnedCalls:
    """Every `FunctionCall` a run spawned, and what became of it. One `guard` per instance."""

    def __init__(
        self,
        log: Callable[[str], None] = _say,
        cancel_timeout_s: float = CANCEL_TIMEOUT_S,
        terminate_containers: bool = True,
    ) -> None:
        self.log = log
        self.cancel_timeout_s = cancel_timeout_s
        self.terminate_containers = terminate_containers
        self._entries: list[_Entry] = []
        self._by_call: dict[int, _Entry] = {}
        self._failures: list[dict] = []
        self._lock = threading.Lock()
        self._signum: int | None = None  # the first SIGINT / SIGTERM seen inside the guard
        self._cleaning = False  # the guard is cancelling: further signals are only logged
        self._defer = 0  # inside `spawn`: a signal waits until the call is tracked

    # --- spawning and collecting ----------------------------------------------------------

    def track(self, label: str, call: Any) -> Any:
        """Keep `call` (spawned elsewhere) until its result is collected."""
        entry = _Entry(label, call)
        with self._lock:
            self._entries.append(entry)
            self._by_call[id(call)] = entry
        return call

    def spawn(self, label: str, function: Any, *args: Any, **kwargs: Any) -> Any:
        """`function.spawn(*args, **kwargs)`, tracked under `label`."""
        self._check()
        with self._signals_deferred():
            return self.track(label, function.spawn(*args, **kwargs))

    def get(self, call: Any, timeout: float | None = None) -> Any:
        """`call.get(timeout)`: its result, or its own error raised as it came (and recorded
        in `failures`). No result within `timeout`: the call is cancelled, recorded, and a
        `TimeoutError` naming it is raised."""
        entry = self._by_call.get(id(call))
        if entry is None:
            raise ValueError("get: that call was not spawned or tracked here")
        self._check()
        try:
            result = call.get(timeout=timeout)
        except TimeoutError as error:
            if error.args:  # the call's own TimeoutError; Modal's "not yet" is a bare one
                self._failed(entry, error)
                raise
            detail = "no result" if timeout is None else f"no result within {timeout:g} s"
            self._record(entry, detail)
            self.log(f"{entry.label}: {detail}, cancelling it")
            self._cancel([entry])
            raise TimeoutError(f"{entry.label}: {detail}; cancelled") from error
        except Exception as error:
            self._failed(entry, error)
            raise
        # Anything else (a stop) leaves the call outstanding, for the guard to cancel.
        entry.state = "succeeded"
        self._check()  # a stop swallowed while this call was waited on
        return result

    def outstanding(self) -> list[str]:
        """The labels of calls neither collected nor cancelled."""
        with self._lock:
            return [e.label for e in self._entries if e.state == "running"]

    def failures(self) -> list[dict]:
        """`{"call", "detail"}` for each call that failed on its own or was given up on."""
        return list(self._failures)

    def report(self) -> list[dict]:
        """`{"call", "state", "detail"}` for every call, in the order spawned."""
        with self._lock:
            return [{"call": e.label, "state": e.state, "detail": e.detail} for e in self._entries]

    # --- cancelling -----------------------------------------------------------------------

    def cancel_outstanding(self, reason: str, keep: Iterable[Any] = ()) -> int:
        """Cancel every call still outstanding but those in `keep`; returns how many Modal
        confirmed. Best effort: a cancel that raises or does not answer in time is logged,
        and the call stays outstanding."""
        kept = {id(call) for call in keep}
        with self._lock:
            chosen = [e for e in self._entries if e.state == "running" and id(e.call) not in kept]
        if not chosen:
            return 0
        self.log(f"cancelling {len(chosen)} outstanding Modal call(s) ({reason})")
        return self._cancel(chosen)

    def _cancel(self, entries: list[_Entry]) -> int:
        """All at once on daemon threads (a hung cancel must not hold up the exit), waited on
        for at most `cancel_timeout_s` in all."""
        outcomes: dict[int, str] = {}

        def cancel(k: int, entry: _Entry) -> None:
            try:
                entry.call.cancel(terminate_containers=self.terminate_containers)
            except Exception as error:  # noqa: BLE001 - logged; the others still go out
                outcomes[k] = f"cancel failed: {error!r}"[:500]
            else:
                outcomes[k] = "cancelled"

        threads = [
            threading.Thread(target=cancel, args=(k, e), name=f"cancel {e.label}", daemon=True)
            for k, e in enumerate(entries)
        ]
        deadline = time.monotonic() + self.cancel_timeout_s
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        done = dict(outcomes)
        confirmed = 0
        for k, entry in enumerate(entries):
            outcome = done.get(k, f"no answer to the cancel within {self.cancel_timeout_s:g} s")
            if outcome == "cancelled":
                confirmed += 1
                entry.state = "cancelled"
            entry.detail = f"{entry.detail}; {outcome}" if entry.detail else outcome
            self.log(f"  {entry.label}: {outcome}")
        return confirmed

    # --- stopping -------------------------------------------------------------------------

    @contextlib.contextmanager
    def guard(self) -> Iterator[SpawnedCalls]:
        """SIGINT and SIGTERM raise `Stopped`; any exception out of the block cancels every
        call still outstanding and is re-raised (a stop as `RunStopped`)."""
        self._signum, self._cleaning = None, False
        previous = self._install()
        try:
            yield self
            if self._signum is not None:  # a stop that something swallowed
                raise Stopped(self._signum)
        except BaseException as error:
            if isinstance(error, SystemExit) and not error.code:
                raise
            self._cleaning = True
            if isinstance(error, Stopped):
                why = f"stopped by {error}"
            elif isinstance(error, KeyboardInterrupt):
                why = "stopped by KeyboardInterrupt"
            else:
                why = f"the run raised {type(error).__name__}: {error}"[:300]
            left = len(self.outstanding())
            try:
                confirmed = self.cancel_outstanding(why)
            except Exception as cancelling:  # noqa: BLE001 - never in place of `error`
                self.log(f"cancelling raised {cancelling!r}")
                confirmed = 0
            if isinstance(error, (Stopped, KeyboardInterrupt)):
                raise RunStopped(
                    f"{why}; {confirmed} of {left} outstanding Modal call(s) cancelled"
                ) from error
            raise
        else:
            if left := self.outstanding():
                self.log(f"{len(left)} call(s) never collected, left running: {', '.join(left)}")
        finally:
            self._restore(previous)

    def _install(self) -> dict[int, Any]:
        if threading.current_thread() is not threading.main_thread():
            self.log("not on the main thread: SIGINT and SIGTERM keep their own handlers")
            return {}
        return {signum: signal.signal(signum, self._handle) for signum in SIGNALS}

    @staticmethod
    def _restore(previous: dict[int, Any]) -> None:
        for signum, handler in previous.items():
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)

    def _handle(self, signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        if self._signum is not None or self._cleaning:
            self._note(f"{name}: already stopping; the cancels go on")
            return
        self._signum = signum
        if self._defer:
            self._note(f"{name}: stopping once the call being spawned is tracked")
            return
        self._note(f"{name}: stopping")
        raise Stopped(signum)

    def _note(self, line: str) -> None:
        """`log` from a signal handler, which may have interrupted a write to the same
        stream (a reentrant write raises): a lost line, never a lost stop."""
        with contextlib.suppress(Exception):
            self.log(line)

    def _check(self) -> None:
        """Raise a stop that was deferred during a spawn, or swallowed on its way out."""
        if self._signum is not None and not self._cleaning and not self._defer:
            raise Stopped(self._signum)

    @contextlib.contextmanager
    def _signals_deferred(self) -> Iterator[None]:
        self._defer += 1
        try:
            yield
        finally:
            self._defer -= 1
            self._check()

    # --- recording ------------------------------------------------------------------------

    def _failed(self, entry: _Entry, error: BaseException) -> None:
        entry.state = "failed"
        self._record(entry, repr(error)[:2000])
        self.log(f"{entry.label} failed: {error!r}"[:2000])

    def _record(self, entry: _Entry, detail: str) -> None:
        entry.detail = detail
        self._failures.append({"call": entry.label, "detail": detail})
