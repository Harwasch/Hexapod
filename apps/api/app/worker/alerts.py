"""A dead-man's switch for the runs that are going on, and nothing else.

`WORKER_HEARTBEAT_URL` is a healthchecks.io-style check (any service that takes the same
three URLs works). For each job the worker supervises:

    claimed             <url>/start     the check expects to hear again soon
    every minute        <url>           still alive
    finished            <url>           done
    failed              <url>/fail      alert now
    stopped, or lost    (nothing)       the next worker's /start, or the check's silence

So a worker that dies mid-run -- OOM-killed, a host gone, a deadlock that stops the
lease keeper too -- goes quiet, and the check alerts after its grace period, whether or
not anything ever restarts it. An idle worker sends nothing at all, so an empty queue is
never an alert and a check is not woken on a schedule for no reason.

`QUEUE_CHECK_URL` is the other half. The API starts that check (`<url>/start`) when it
queues a job, and the worker completes it (`<url>`) when it claims one: a job queued and
never claimed -- no worker running, a worker that cannot claim (a full disk) -- alerts.

**A ping never blocks the worker and never fails a job.** Each is sent from a short-lived
thread of its own with a short timeout, and anything it raises is logged and dropped: the
point of a dead-man's switch is to report trouble, and one that could cause it -- a slow
monitoring service holding up a lease renewal, an exception ending a supervision -- is
worse than none.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from types import TracebackType

import httpx

log = logging.getLogger("app.worker")

#: How long one ping may take before it is abandoned.
PING_TIMEOUT_S = 5.0
#: The most of a failure's message sent with `/fail` (healthchecks.io keeps 100 kB).
BODY_LIMIT = 10_000

#: Sends one ping: (url, body). Replaced in tests.
Sender = Callable[[str, str], None]


def send(url: str, body: str) -> None:
    """POST `body` to `url`, in this thread, giving up after `PING_TIMEOUT_S`."""
    httpx.post(url, content=body.encode("utf-8"), timeout=PING_TIMEOUT_S)


def ping(url: str, body: str = "", *, sender: Sender = send) -> threading.Thread:
    """Send one ping from a thread of its own; returns the thread (tests join it)."""

    def run() -> None:
        try:
            sender(url, body[:BODY_LIMIT])
        except Exception as error:
            log.warning("worker: ping to %s failed: %r", _redacted(url), error)

    thread = threading.Thread(target=run, name="ping", daemon=True)
    thread.start()
    return thread


def _redacted(url: str) -> str:
    """A check URL carries its secret in the path; the log says which host, not which check."""
    head, _, _ = url.partition("://")
    host = url.split("://", 1)[-1].split("/", 1)[0]
    return f"{head}://{host}/..."


class RunWatch:
    """One job's pings: `/start` on entry, one every `every_s` until `finish`, and the end.

    A context manager around the whole supervision, beside `claim.LeaseKeeper` and for
    the same reason: the periodic ping has to go on whatever the supervising thread is
    doing (downloading the capture, uploading a stage), so it has a thread of its own.
    `finish` sends the last word; leaving without one (the worker stopping, the job lost
    to another slot) sends nothing, which is what lets the check notice.
    """

    def __init__(
        self,
        url: str | None,
        job_id: uuid.UUID,
        *,
        queue_url: str | None = None,
        every_s: float = 60.0,
        sender: Sender = send,
    ) -> None:
        self._url = url.rstrip("/") if url else None
        self._queue_url = queue_url.rstrip("/") if queue_url else None
        self._job_id = job_id
        self._every_s = every_s
        self._sender = sender
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        #: Every ping thread started, so a test can wait for them.
        self.sent: list[threading.Thread] = []

    def __enter__(self) -> RunWatch:
        if self._queue_url:
            self._ping(self._queue_url, f"job {self._job_id} claimed")
        if self._url:
            self._ping(f"{self._url}/start", f"job {self._job_id} claimed")
            self._thread = threading.Thread(
                target=self._beat, name=f"ping-{self._job_id}", daemon=True
            )
            self._thread.start()
        return self

    def _beat(self) -> None:
        assert self._url is not None
        while not self._done.wait(self._every_s):
            self._ping(self._url, f"job {self._job_id} running")

    def _ping(self, url: str, body: str) -> None:
        self.sent.append(ping(url, body, sender=self._sender))

    def _stop(self) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def finish(self, outcome: str, detail: str = "") -> None:
        """The run's last word. `complete` and `cancelled` are a success -- the worker did
        what it was asked -- `error` is a failure, and `lost` is silence: the job is not
        over, and either the next worker's `/start` or the check's grace period says so."""
        self._stop()
        if not self._url:
            return
        message = f"job {self._job_id} {outcome}" + (f": {detail}" if detail else "")
        if outcome == "error":
            self._ping(f"{self._url}/fail", message)
        elif outcome in ("complete", "cancelled"):
            self._ping(self._url, message)

    def __exit__(
        self,
        _type: type[BaseException] | None,
        _value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._stop()
