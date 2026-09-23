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

* **`poll` classified every running call as failed.** It caught the builtin
  `TimeoutError`. `modal.exception.TimeoutError` -- which is what
  `modal/_functions.py` raises from a zero-timeout poll -- inherits from `modal.Error`,
  **not** from the builtin. The `except TimeoutError` clause never fired, so the first
  poll of a perfectly healthy stage fell through to `except Exception` and was
  dead-lettered. This was the fatal one.
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
  feed, not something a poll loop can attribute to one `FunctionCall`. So `poll` still
  reports wall time between `submit` and the poll, and this is still not the billed
  number: queue time is not billed, a container that outlives the call is. It is a proxy,
  labelled as one, and `providers.py` refuses to price an unsurveyed tier, so the
  overstatement stops at seconds and never becomes an invented dollar figure.
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

from cloud import Poll, RemoteHandle, RemoteState, StageRequest
from providers import Rate, provider

__all__ = ["ModalAdapter"]

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

#: Fully-qualified exception class name -> what it means for a submitted stage.
#:
#: Qualified, and matched on the *exact* class rather than by `isinstance`, for two
#: reasons that are both real bugs avoided. First, `FunctionTimeoutError` and
#: `OutputExpiredError` are subclasses of `modal.exception.TimeoutError`, so an
#: `isinstance` check would read a stage that blew its time limit as still running.
#: Second, the module matters: a stage whose own code raises the *builtin* `TimeoutError`
#: has that exception deserialised and re-raised here, and it must be a failure, not a
#: poll that never terminates. `builtins.TimeoutError` is absent from this table, so it
#: falls to the default, which is `failed`.
_STATES: Mapping[str, RemoteState | None] = {
    # No result yet. The only entry that means "keep polling".
    "modal.exception.TimeoutError": None,
    # The container was reclaimed and Modal exhausted its own retries.
    "modal.exception.InternalFailure": "preempted",
    # The stage outran the deployed function's `timeout=`.
    "modal.exception.FunctionTimeoutError": "failed",
    # We asked too late; Modal had already dropped the result.
    "modal.exception.OutputExpiredError": "failed",
}

#: How many log lines to hold for one call. `logs.fetch()` re-reads a call's whole
#: history, so an unbounded tail would grow the poll loop's cost with the stage's
#: chattiness; a training stage that prints a line per step would be unbounded indeed.
MAX_LOG_LINES = 5_000


@dataclass
class _Call:
    request: StageRequest
    call: Any
    started_at: float
    state: RemoteState = "running"
    detail: str = ""
    billed_s: float = 0.0
    metrics: Mapping[str, Any] = field(default_factory=dict)
    summary: str = ""
    lines: list[str] = field(default_factory=list)


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
    ) -> None:
        self._app_name = app_name
        self._function_name = function_name
        self._rates = dict(rates or {})
        self._timeout_s = timeout_s
        self._environment_name = environment_name
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
        function = self._function(request.tier)
        call = function.spawn(request.to_dict())
        handle = RemoteHandle(id=str(call.object_id), provider=self.name, tier=request.tier)
        self._calls[handle.id] = _Call(request=request, call=call, started_at=time.monotonic())
        return handle

    def poll(self, handle: RemoteHandle) -> Poll:
        run = self._calls[handle.id]
        if run.state in ("succeeded", "failed", "preempted"):
            return self._result(run)
        # Wall time, not the billed figure: see the module docstring. A proxy, and
        # labelled as one, because there is no per-call billed-seconds number to read.
        run.billed_s = time.monotonic() - run.started_at
        try:
            # `timeout=0` is documented as the way to "poll for an output immediately".
            outcome = run.call.get(timeout=0)
        except Exception as error:  # every outcome of a poll arrives as one
            state = self._state_of(error)
            if state is None:
                return Poll(state="running", billed_s=run.billed_s)
            run.state = state
            run.detail = f"{type(error).__name__}: {error}"
            return self._result(run)
        run.state = "succeeded"
        run.metrics = dict((outcome or {}).get("metrics") or {})
        run.summary = str((outcome or {}).get("summary") or "")
        return self._result(run)

    def logs(self, handle: RemoteHandle, *, since: int = 0) -> Sequence[str]:
        """Lines from index `since` onward, via `FunctionCall.logs.fetch()`.

        The protocol indexes by line and Modal's API filters by timestamp, so this reads
        the call's whole history each time and slices. That is the honest implementation
        of an index-based contract over a time-based source, and it is bounded by
        `MAX_LOG_LINES` rather than by trust. A log fetch that fails is not a stage that
        failed: the poll loop keeps its verdict and simply has nothing new to tail.
        """
        run = self._calls.get(handle.id)
        if run is None:
            return ()
        # Suppressed rather than handled: logs are diagnostics, and a log service that is
        # down must not turn into a verdict on a stage that is running perfectly well.
        # The previously fetched lines stay, so a tail goes quiet rather than truncating.
        with suppress(Exception):
            run.lines = self._tail(run.call.logs.fetch())
        return tuple(run.lines[max(since, 0) :])

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
    def _tail(entries: Iterable[Any]) -> list[str]:
        """`LogEntry` objects to lines, newest-bounded.

        An entry's `message` can carry several lines or a trailing newline, so it is
        split rather than appended whole: `since` indexes lines, and an index that
        sometimes means "entry" would skip or repeat output.
        """
        lines: list[str] = []
        for entry in entries:
            lines.extend(str(getattr(entry, "message", entry)).rstrip("\n").split("\n"))
            if len(lines) > MAX_LOG_LINES * 2:
                del lines[:-MAX_LOG_LINES]
        return lines[-MAX_LOG_LINES:]

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
        )
