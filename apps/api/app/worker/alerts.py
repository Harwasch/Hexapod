"""A dead-man's switch for the runs that are going on, and nothing else.

`WORKER_HEARTBEAT_URL` is a healthchecks.io-style check (any service that takes the same
three URLs works). For each job the worker supervises:

    claimed             <url>/start     a run has started: it must end within the grace
    every minute        <url>/start     still running: the grace starts again from now
    finished            <url>           done
    failed              <url>/fail      alert now
    stopped, or lost    (nothing)       the next worker's /start, or the grace running out

So a worker that dies mid-run -- OOM-killed, a host gone, a deadlock that stops the
lease keeper too -- goes quiet, and the check alerts once its grace time has passed since
the last `/start`, whether or not anything ever restarts it. An idle worker sends nothing
at all, so an empty queue is never an alert.

**Why the keep-alive is a `/start`, and how the check must be set.** healthchecks.io
watches two clocks. The *period* runs from the last success ping and alerts when nothing
has arrived for period + grace; a `/start` opens a run, and the run alerts when no
success or failure follows within the *grace* -- each new `/start` restarting it. The
keep-alive used to be a success ping, which only ever fed the period: a short period
alerted whenever the worker sat idle for longer than it (which it does, by design, for
days), and a period long enough to stay quiet through that was also long enough to miss
a run that went quiet for hours. As a `/start` it feeds the grace, which is the clock
about a run. So the check wants:

* **period: long** -- 30 days (or more). It is only "the worker has not finished a job
  in a month", and must not fire because nobody uploaded anything this week.
* **grace: a few minutes** -- 5. Longer than the one-minute keep-alive, with room for a
  ping or two lost to a slow network, and longer than a deploy takes to hand a running
  job to the next worker (whose claim sends the next `/start`). That is how long a run
  that dies takes to alert.

`QUEUE_CHECK_URL` is the other half. The API starts that check (`<url>/start`) when it
queues a job onto an idle worker, and the worker completes it (`<url>`) whenever it
claims one: a job queued and never claimed -- no worker running, a worker that cannot
claim (a full disk) -- alerts. It wants the same long period, and a grace longer than a
cold start: the machine has to boot and the worker claim (10 minutes is generous). The
API does not start it for a job queued behind one a worker is running -- that job waits
for as long as the run takes, two hours of training and more, and a check started for it
would alert after its grace every time (`app/services/worker_wake.py`).

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
            # The kind of failure, not its message: an httpx error's message can name the
            # URL it was sent to, and the path of a check URL is its secret.
            log.warning("worker: ping to %s failed: %s", _redacted(url), type(error).__name__)

    thread = threading.Thread(target=run, name="ping", daemon=True)
    thread.start()
    return thread


def _redacted(url: str) -> str:
    """A check URL carries its secret in the path; the log says which host, not which check."""
    head, _, _ = url.partition("://")
    host = url.split("://", 1)[-1].split("/", 1)[0]
    return f"{head}://{host}/..."


class RunWatch:
    """One job's pings: `/start` on entry, `/start` again every `every_s` until `finish`,
    and the end -- success or `/fail`.

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
        # A `/start`, not a success: each one restarts the check's grace for this run, so
        # a run that goes quiet alerts a grace after its last word, while the check's long
        # period stays a matter of weeks for a worker that is idle (module note).
        assert self._url is not None
        while not self._done.wait(self._every_s):
            self._ping(f"{self._url}/start", f"job {self._job_id} running")

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
