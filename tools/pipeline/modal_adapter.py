"""A `ProviderAdapter` for Modal. **Checked against the SDK; still never run against Modal.**

Read that sentence carefully, because it is a smaller claim than it looks. The previous
version of this file said its Modal calls were written *from memory of Modal's Python
SDK, not from its documentation*, because both were unreachable. They are reachable now,
so every symbol below has been checked against `modal==1.5.5` -- its installed source,
its docstrings, and https://modal.com/docs. That moves this file from *guessed* to
*read*. It does not move it to *verified*: no line here has been executed against Modal,
because that needs an account and a token this repository does not have. In the three
states `README.md` tracks, `ModalAdapter` is still **unproven**.

What the check found, since "we looked and it was fine" would be the least useful
possible report:

* **"Not finished yet" is the builtin `TimeoutError`, and this file got that wrong
  twice.** `modal/_functions.py` (`poll_function`) raises a bare `TimeoutError()` --
  the builtin, since that module never imports Modal's own -- when a zero-timeout
  `get` finds no output. An earlier revision of this adapter read it as Modal's
  `modal.exception.TimeoutError` and classified the builtin as a failure, so the first
  real run on Modal (the smoke in `.github/workflows/modal.yml`) was dead-lettered on
  its first poll while the GPU was still training. The builtin now means "keep
  polling", and `remote.execute` converts a stage's own `TimeoutError` into a
  `RuntimeError` inside the container, so the two can no longer be confused.
* **None of the three guessed preemption markers exist.** `PreemptedError`,
  `FunctionInterrupted` and `InterruptedError` are not in `modal.exception`. Preemption
  arrives as `InternalFailure`: `modal/_functions.py` says so in as many words --
  *"This will retry when the server returns GENERIC_STATUS_INTERNAL_FAILURE, i.e. lost
  inputs or worker preemption"* -- and `_process_result` raises `InternalFailure` for
  that status.
* **Two exception classes inherit `modal.exception.TimeoutError`.**
  `FunctionTimeoutError` (the stage outran the function's own limit) and
  `OutputExpiredError` (the result was garbage-collected before we asked for it) are both
  subclasses of it. Classifying by `isinstance` against Modal's `TimeoutError` would have
  reported both as *still running*, forever. `_STATES` below matches on the exact class,
  by qualified name, for exactly this reason.
* **`gpu="A10G"` is not a Modal tier.** Modal's identifier is `A10`; `A10G` is AWS's
  instance name. The full set is in `GPU_NAMES`.
* **`logs` is implementable and was returning nothing.** `FunctionCall.logs` is a
  property giving a manager with `fetch`, `tail` and `stream`. The old note here said
  Modal's log API "was not reachable to be written against"; it is now.
* **`cancel()` alone keeps the container -- and the bill -- running.** The real
  signature is `cancel(terminate_containers: bool = False)`. The old docstring worried
  that "a provider whose `cancel` does nothing is a provider that keeps billing"; the
  default argument is precisely that provider. We pass `True`.

Two things the check settled that are not defects but change what can be claimed:

* **Billed seconds.** There is no per-call billed-seconds figure in the SDK. What Modal
  exposes is `modal.billing.workspace_billing_report(start=..., end=..., resolution=...)`,
  which reports **cost** (a `Decimal`, broken down by resource) per Modal object per
  interval, at a resolution of hours or days. That is a workspace-level reconciliation
  feed, not something a poll loop can attribute to one `FunctionCall`. So a finished
  call is billed from **the container's own clock** (`cloud.container_billing`): from
  the container's start (a cold call) or the call's entry (a warm one) to the call's
  end, which is what Modal bills a GPU-second for -- and the wait before it, which Modal
  does not bill, is reported apart as `queueS` (Poll.queue_s). One part of job 33bc1bff
  waited 1,697 s for an L4 and the old figure counted every second of it as cost. A call
  that did not report those clocks (failed, preempted, still running, or an older
  image) keeps the old proxy -- wall time since `submit` -- and says so
  (`billing: wall-proxy`). Neither counts the container's idle scale-down window after
  its last call, which Modal does bill and which belongs to no call.
* **`interruptible = False` is right, for a different reason than it said.** The old
  comment claimed "Modal's own tiers are not interruptible". They are: `nonpreemptible`
  is a real parameter of `@app.function` and it defaults to `False`. What makes Modal the
  sensible last entry in a `Placement` is that the client absorbs preemption below this
  seam -- it retries an internal failure up to `MAX_INTERNAL_FAILURE_COUNT` (8) before
  raising -- so by the time `poll` sees `InternalFailure`, Modal has already given up on
  resuming it eight times. The `preempted` branch is therefore rare rather than routine,
  and `CloudRunner`'s checkpoint-and-resume is the right response to it either way.

Still genuinely unverified, because only a real run can settle them:

* That a preempted attempt surfaces as `InternalFailure` **to the caller**, rather than
  being retried invisibly until it succeeds or the function times out. The source says
  the status means preemption; nobody here has watched a box get reclaimed.
* Whether `logs.fetch()` on a large call is fast enough to sit inside a poll loop.
* The remote half, which now exists: `infra/modal/app.py` deploys one `run_stage_<tier>`
  per tier, and its body is `tools/pipeline/remote.py` -- ordinary code, driven over a
  `Transfer` by `tests/test_remote.py`. What is unverified is the wrapper around it: the
  image, the `gpu=` string and the secret. That the App builds locally is all anybody
  here has checked, and a nonsense GPU name builds locally too.

`providers.py` is the source of the price, and for Modal it has one surveyed figure: an
A100 hour. An L4 or A10 run therefore records the seconds it was billed and no cost,
which is deliberate -- see that module.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from cloud import (
    BILLING_CONTAINER,
    BILLING_PROXY,
    Poll,
    RemoteHandle,
    RemoteState,
    StageRequest,
    container_billing,
)
from contracts import FANOUT_PARAM
from providers import Rate, provider

__all__ = ["GPU_FALLBACKS", "ModalAdapter", "fallback_tier", "tier_of_gpu"]

#: The SDK release every symbol in this module was read against. Recorded because
#: "checked" is only meaningful with a version attached, and because the next person to
#: find a mismatch should know whether the SDK moved or this file was wrong.
MODAL_VERSION_CHECKED = "1.5.5"

#: Tier -> the string Modal's `gpu=` argument wants, from
#: https://modal.com/docs/guide/gpu. Counts are appended as `:n` (`"A100:4"`), which the
#: remote half does; a tier here names one GPU. `a10g` is deliberately absent: it is not
#: a Modal identifier, and a tier that silently maps to nothing is worse than a KeyError.
GPU_NAMES: Mapping[str, str] = {
    "t4": "T4",
    "l4": "L4",
    "a10": "A10",
    "l40s": "L40S",
    # `A100-80GB`, not the bare `A100`, which Modal reads as the 40 GB part and prices
    # 19% lower. `providers.py` carries $2.50 for this tier, which is the 80 GB figure,
    # so a bare `A100` here would be a table whose price and whose hardware disagree.
    # The smaller part is reachable, but only by asking for it by name.
    "a100": "A100-80GB",
    "a100-40gb": "A100-40GB",
    "h100": "H100",
    "h200": "H200",
    "b200": "B200",
    "b300": "B300",
}

#: A fan-out part's GPU, ranked: tier asked for -> the tiers Modal may start the part on,
#: first free wins (`gpu=[...]` in `infra/modal/app.py`, a ranked list in modal 1.5.5's
#: `_Function.from_local`). A part otherwise waits for its one tier: in job 33bc1bff one
#: of two L4 parts waited 1,697 s -- 28 minutes of a 30-minute `max_pending_s` -- while
#: the other trained. Price-aware, on the L4 as the reference:
#:
#: * **L40S, in.** $1.95/h against $0.80, 2.44x the price, and it trained ~2.3x faster on
#:   our benchmark, which put a block's cost ~17% above the L4's (the list prices alone
#:   give +6% on the training seconds; the start, fetch and syncs are not sped up) -- for
#:   a part that ends in under half the time, instead of one that has not started. 48 GB,
#:   so a block sized for the L4's 24 GB fits with room.
#: * **A10, out, until measured.** $1.10/h, 1.38x the price, 24 GB; nothing here has timed
#:   gsplat on one. Its memory bandwidth (600 GB/s against 300) suggests it breaks even,
#:   but a slower-per-dollar GPU that is merely free would make a part cost more *and* end
#:   later than a short wait for an L4 would.
#:
#: Parts only: a single run, a head and a join keep their one tier (the L4 by default),
#: because they are not waited on by a sibling that is already paid for.
GPU_FALLBACKS: Mapping[str, tuple[str, ...]] = {
    "l4": ("l4", "l40s"),
}


def fallback_tier(tier: str) -> str:
    """The name `infra/modal/app.py` deploys `tier`'s fallback list under, after
    `run_stage_`: `run_stage_l4_fallback`."""
    return f"{tier}_fallback"


def tier_of_gpu(name: str, chain: Sequence[str]) -> str | None:
    """Which of `chain`'s tiers the GPU `nvidia-smi` named is (`NVIDIA L40S` -> `l40s`).

    Matched on the name's words, so `L4` is never taken for `L40S` or the other way; an
    A100's memory is checked too (`NVIDIA A100-SXM4-80GB`). None when no tier matches."""
    words = name.upper().replace("NVIDIA", " ").replace("-", " ").split()
    for tier in chain:
        want = GPU_NAMES.get(tier, "").upper().split("-")
        if want and want[0] in words and all(part in name.upper() for part in want[1:]):
            return tier
    return None


#: Tiers with no GPU at all: tier -> (Modal physical cores, memory in MiB). Modal's
#: `cpu=` counts physical cores, two vCPUs each. `cpu4` is where `pose` runs: COLMAP's
#: feature extraction peaked at 1.7 GB on real 1080p frames, more than the worker's
#: 2 GB machine can spare, and a per-second container costs $0.25 an hour only while a
#: capture is being posed, where a Fly machine big enough would cost $124 a month
#: whether anything ran or not. Deployed from its own image, which carries COLMAP.
CPU_TIERS: Mapping[str, tuple[float, int]] = {
    "cpu4": (4.0, 8192),
}

#: Fully-qualified exception class name -> what it means for a submitted stage.
#:
#: Qualified, and matched on the *exact* class rather than by `isinstance`, for two
#: reasons that are both real bugs avoided. First, `FunctionTimeoutError` and
#: `OutputExpiredError` are subclasses of `modal.exception.TimeoutError`, so an
#: `isinstance` check would read a stage that blew its time limit as still running.
#: Second, the module matters, because the builtin and Modal's own are different
#: classes that both carry the name `TimeoutError`.
_STATES: Mapping[str, RemoteState | None] = {
    # No result yet: what Modal 1.5.5's zero-timeout `get` raises (`poll_function`).
    # Unambiguous only because `remote.execute` never lets a stage's own TimeoutError
    # out of the container as itself.
    "builtins.TimeoutError": None,
    # Modal's own, kept in case another client version raises it for the same thing.
    "modal.exception.TimeoutError": None,
    # The container was reclaimed and Modal exhausted its own retries.
    "modal.exception.InternalFailure": "preempted",
    # The stage outran the deployed function's `timeout=`.
    "modal.exception.FunctionTimeoutError": "failed",
    # We asked too late; Modal had already dropped the result.
    "modal.exception.OutputExpiredError": "failed",
}

#: The prefix of the first line `run_stage` prints inside the container. Its appearance
#: in a call's log is what moves the call from `pending` to `running`.
START_MARKER = "run_stage: "

#: How many log lines to hold for one call, and to hand back from one `logs` call.
#: `logs.fetch()` re-reads a call's whole history, so an unbounded tail would grow the
#: poll loop's cost with the stage's chattiness; a training stage that prints a line per
#: step would be unbounded indeed. The bound is on what is *held*: `since` stays an index
#: into the whole log, however much of it has scrolled out of the held tail.
MAX_LOG_LINES = 5_000


@dataclass
class _Call:
    request: StageRequest
    call: Any
    started_at: float
    #: The wall clock at submit (`cloud.container_billing` compares it with the
    #: container's own clock) and the tiers the call may run on.
    submitted_at: float = 0.0
    chain: tuple[str, ...] = ()
    state: RemoteState = "running"
    tier: str = ""
    billing: str = BILLING_PROXY
    queue_s: float | None = None
    detail: str = ""
    billed_s: float = 0.0
    metrics: Mapping[str, Any] = field(default_factory=dict)
    summary: str = ""
    #: The newest lines of the last successful fetch, and the index of the first of them
    #: in the whole log.
    lines: list[str] = field(default_factory=list)
    first: int = 0
    #: Whether a log fetch has ever succeeded. Without one, "no start line yet" is not
    #: evidence of anything, and the call is reported running as it always was.
    logs_read: bool = False
    #: Sticky: the start line scrolls out of `lines` once a run logs more than
    #: `MAX_LOG_LINES`, and a stage that has started does not stop having started.
    started: bool = False


class ModalAdapter:
    """Modal, as far as this repository can describe it without having run it.

    Every method is shaped exactly like `FakeAdapter`'s and `SubprocessAdapter`'s, which
    was the one claim this file used to make: the protocol fits a real provider. It now
    makes a second, still modest one -- the Modal calls are the ones the SDK documents,
    checked against `MODAL_VERSION_CHECKED`. Whether they behave as documented against a
    live workspace is the part that remains unproven.
    """

    #: Matches `providers.PROVIDERS`, so `jobs.provider` and the console's price table
    #: agree about which host this is.
    name = "modal"
    #: Not that Modal's containers cannot be reclaimed -- they run preemptible unless a
    #: function sets `nonpreemptible=True` -- but that the client retries a reclaimed
    #: input up to eight times before the caller hears about it. Preemption is absorbed
    #: below this seam, which is what makes Modal the sensible thing to fall back *to*.
    interruptible = False

    def __init__(
        self,
        app_name: str,
        function_name: str = "run_stage",  # a prefix; see `_function`
        *,
        rates: Mapping[str, Rate] | None = None,
        timeout_s: float = 6 * 3600.0,
        environment_name: str | None = None,
        #: Start a fan-out's parts on the first free GPU of `GPU_FALLBACKS` rather than
        #: only their own tier. Off, every call waits for exactly the tier it asked for.
        part_fallback: bool = True,
    ) -> None:
        self._app_name = app_name
        self._function_name = function_name
        self._rates = dict(rates or {})
        self._timeout_s = timeout_s
        self._environment_name = environment_name
        self._part_fallback = part_fallback
        self._calls: dict[str, _Call] = {}

    # --- the protocol -------------------------------------------------------------

    def rate(self, tier: str) -> Rate | None:
        """The surveyed rate, unless the deployment supplied its own."""
        supplied = self._rates.get(tier)
        if supplied is not None:
            return supplied
        listed = provider(self.name)
        return None if listed is None else listed.rate(tier)

    def submit(self, request: StageRequest) -> RemoteHandle:
        """Spawn the deployed function and hand back its call id.

        `spawn` is the non-blocking call -- "Conceptually similar to
        `multiprocessing.pool.apply_async`", per its docstring -- and returns a
        `FunctionCall` whose `object_id` is the durable handle: `FunctionCall.from_id`
        reconstructs it in a later process, which is what makes a handle outlive the
        supervisor that created it.
        """
        chain = self._chain(request)
        submitted_at = time.time()
        if len(chain) > 1:
            try:
                call = self._function(fallback_tier(request.tier)).spawn(request.to_dict())
            except Exception as error:
                # A deployment older than the fallback functions: the part waits for its
                # own tier, as every part did before. Anything else is a real failure.
                if type(error).__name__ != "NotFoundError":
                    raise
                chain = (request.tier,)
                call = self._function(request.tier).spawn(request.to_dict())
        else:
            call = self._function(request.tier).spawn(request.to_dict())
        handle = RemoteHandle(id=str(call.object_id), provider=self.name, tier=request.tier)
        self._calls[handle.id] = _Call(
            request=request,
            call=call,
            started_at=time.monotonic(),
            submitted_at=submitted_at,
            chain=chain,
        )
        return handle

    def _chain(self, request: StageRequest) -> tuple[str, ...]:
        """The tiers this call may run on: a part's fallback list, or its own tier."""
        role = request.params.get(FANOUT_PARAM)
        is_part = isinstance(role, Mapping) and role.get("role") == "part"
        if self._part_fallback and is_part and request.tier in GPU_FALLBACKS:
            return GPU_FALLBACKS[request.tier]
        return (request.tier,)

    def poll(self, handle: RemoteHandle) -> Poll:
        run = self._calls[handle.id]
        if run.state in ("succeeded", "failed", "preempted"):
            return self._result(run)
        # Wall time until the call reports its own clocks: see the module docstring. A
        # proxy, and labelled as one (`billing`), replaced when the call finishes.
        run.billed_s = time.monotonic() - run.started_at
        try:
            # `timeout=0` is documented as the way to "poll for an output immediately".
            outcome = run.call.get(timeout=0)
        except Exception as error:  # every outcome of a poll arrives as one
            state = self._state_of(error)
            if state is None:
                return Poll(state=self._liveness(run), billed_s=run.billed_s, billing=BILLING_PROXY)
            run.state = state
            run.detail = f"{type(error).__name__}: {error}"
            return self._result(run)
        run.state = "succeeded"
        run.metrics = dict((outcome or {}).get("metrics") or {})
        run.summary = str((outcome or {}).get("summary") or "")
        figured = container_billing(run.metrics, run.submitted_at, run.billed_s)
        if figured is not None:
            run.billed_s, run.queue_s = figured
            run.billing = BILLING_CONTAINER
        run.tier = self._ran_on(run)
        return self._result(run)

    @staticmethod
    def _ran_on(run: _Call) -> str:
        """The tier the call ran on: its own, unless a fallback list let Modal choose, in
        which case the GPU the container reported (`remoteGpu`). One it cannot place is
        priced at the dearest of the list, so a cost is never understated."""
        if len(run.chain) <= 1:
            return ""
        gpu = run.metrics.get("remoteGpu")
        found = tier_of_gpu(gpu, run.chain) if isinstance(gpu, str) else None
        if found is not None:
            return found
        listed = provider("modal")

        def price(tier: str) -> float:
            rate = listed.rate(tier) if listed is not None else None
            return rate.usd_per_hour if rate is not None else 0.0

        return max(run.chain, key=price)

    def logs(self, handle: RemoteHandle, *, since: int = 0) -> Sequence[str]:
        """Lines from index `since` onward, via `FunctionCall.logs.fetch()`.

        The protocol indexes by line and Modal's API filters by timestamp, so this reads
        the call's whole history each time and slices. That is the honest implementation
        of an index-based contract over a time-based source, and it is bounded by
        `MAX_LOG_LINES` rather than by trust. A log fetch that fails is not a stage that
        failed: the poll loop keeps its verdict and simply has nothing new to tail.

        At most `MAX_LOG_LINES` come back per call, the *oldest* from `since`, so a caller
        that advances by what it got never skips a line: a big backlog arrives over a few
        polls rather than with a hole in it. (Slicing the held tail by `since` instead went
        silent for good once a stage passed `MAX_LOG_LINES` -- every pose solve does, and
        the live viewer then never heard of the finished cameras.)
        """
        run = self._calls.get(handle.id)
        if run is None:
            return ()
        since = max(since, 0)
        # Suppressed rather than handled: logs are diagnostics, and a log service that is
        # down must not turn into a verdict on a stage that is running perfectly well.
        # The previously fetched lines stay, so a tail goes quiet rather than truncating.
        with suppress(Exception):
            entries = list(run.call.logs.fetch())
            # Looked for in the whole fetch, before the tail is bounded: a stage that logs
            # thousands of lines before the first fetch would otherwise have its start
            # line trimmed away before it was ever seen.
            run.started = run.started or any(
                START_MARKER in str(getattr(entry, "message", entry)) for entry in entries
            )
            every = self._lines(entries)
            run.first = max(len(every) - MAX_LOG_LINES, 0)
            run.lines = every[run.first :]
            run.logs_read = True
            return tuple(every[since : since + MAX_LOG_LINES])
        start = max(since - run.first, 0)
        return tuple(run.lines[start : start + MAX_LOG_LINES])

    def cancel(self, handle: RemoteHandle) -> None:
        run = self._calls.get(handle.id)
        if run is None or run.state in ("succeeded", "failed", "preempted"):
            return
        # `terminate_containers` defaults to False, which cancels the input but leaves
        # the container alive -- and billing. This is the argument that makes `cancel`
        # mean "stop paying for it", which is what the protocol says it must mean.
        run.call.cancel(terminate_containers=True)
        run.state = "failed"
        run.detail = "cancelled"

    # --- helpers ------------------------------------------------------------------

    def _function(self, tier: str) -> Any:
        """The deployed function for one tier, as `run_stage_<tier>`.

        One function per tier, because Modal fixes a function's GPU at decoration time
        and a single deployed `run_stage` therefore cannot serve an L4 request and an
        A100 one. `infra/modal/app.py` registers them under exactly these names, from
        the same `GPU_NAMES` table above, so a tier this adapter can ask for is a tier
        that was deployed.
        """
        try:
            import modal  # optional, and absent everywhere this actually runs
        except ImportError as error:  # pragma: no cover - modal is not a dependency here
            raise RuntimeError(
                "ModalAdapter needs the `modal` package, which is deliberately not a "
                "dependency of tools/pipeline: no part of this adapter has ever been run, "
                "so nothing here should pull a client library into the image. Install it "
                "in the deployment that actually uses Modal"
            ) from error
        return modal.Function.from_name(
            self._app_name,
            f"{self._function_name}_{tier}",
            environment_name=self._environment_name,
        )

    @staticmethod
    def _lines(entries: Iterable[Any]) -> list[str]:
        """`LogEntry` objects to lines.

        An entry's `message` can carry several lines or a trailing newline, so it is
        split rather than appended whole: `since` indexes lines, and an index that
        sometimes means "entry" would skip or repeat output.
        """
        lines: list[str] = []
        for entry in entries:
            lines.extend(str(getattr(entry, "message", entry)).rstrip("\n").split("\n"))
        return lines

    @staticmethod
    def _liveness(run: _Call) -> RemoteState:
        """`pending` until the stage's own first line appears, then `running`.

        Modal answers "no output yet" identically for a stage that is training and for
        a container that crash-loops before the function body is reached, so the log is
        the only thing that tells them apart. `run_stage` prints `run_stage: <impl> for
        stage ...` before anything else, from inside the function; a container that
        never gets that far never prints it. `CloudRunner` gives up on a stage that stays
        pending too long. When the log cannot be read at all this answers `running`, so
        a log outage cannot get a healthy stage cancelled.
        """
        if not run.logs_read:
            return "running"
        return "running" if run.started else "pending"

    @staticmethod
    def _state_of(error: BaseException) -> RemoteState | None:
        """Terminal state, or None for "no result yet, keep polling".

        Matched on the exact qualified class name; `_STATES` says why. Anything not in
        that table is a failure, which is the safe default: an unrecognised exception
        that meant "still running" costs one dead-lettered stage, while an unrecognised
        exception treated as "still running" costs a poll loop that never ends.
        """
        kind = type(error)
        return _STATES.get(f"{kind.__module__}.{kind.__qualname__}", "failed")

    @staticmethod
    def _result(run: _Call) -> Poll:
        metrics = {
            key: value
            for key, value in run.metrics.items()
            if isinstance(value, bool | int | float | str)
        }
        return Poll(
            state=run.state,
            billed_s=run.billed_s,
            detail=run.detail,
            metrics=metrics if run.state == "succeeded" else None,
            summary=run.summary,
            billing=run.billing,
            queue_s=run.queue_s,
            tier=run.tier,
        )
