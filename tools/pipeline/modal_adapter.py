"""A `ProviderAdapter` for Modal. **No line of this file has ever run against Modal.**

Read that sentence again before trusting anything below it. This module is a design
sketch, written to show that the protocol in `cloud.py` has room for a real provider and
to be the starting point for whoever wires one up. It is not a working integration and it
is not covered by a test that talks to Modal, because nothing in this repository can:
`modal.com` and `api.modal.com` are unreachable from the environment this was written in
(connection failure, not a rejected request), so there was no way to run it once, and a
single unverified adapter is a sketch while two would be twice the unverified code for no
extra proof.

Worse, and more usefully said plainly: the Modal calls below are written **from memory of
Modal's Python SDK, not from its documentation**, which was equally unreachable. Treat
every symbol -- `modal.Function.from_name`, `FunctionCall.from_id`, `spawn`, `get`, the
exception names -- as a guess at the right shape rather than a checked fact. The parts
this repository *can* stand behind are the parts on this side of the seam: the five
methods, their signatures, and what `CloudRunner` does with each answer.

Specifically unverified, and worth checking first:

* **how Modal reports a preemption.** The whole point of `poll` returning `preempted`
  separately from `failed` is that they are different events; `_state_of` below guesses
  that a preemption surfaces as a retryable/interrupted exception class. If Modal
  actually signals it another way -- a field on the call, a specific exception, nothing
  at all -- this is the line to fix, and getting it wrong means a preempted stage is
  dead-lettered as a failure instead of resuming.
* **billed seconds.** `poll` reports wall time here. Modal bills per second of container
  runtime, which is not the same as the wall time between `spawn` and a poll: queue time
  is not billed and a container that outlives the call is. A real integration should read
  the number Modal reports rather than time it locally, or the cost will be wrong in the
  direction that flatters us.
* **the remote half.** `submit` assumes a deployed Modal function that takes a
  `StageRequest` dict, fetches the inputs and checkpoint from object storage itself,
  runs `run_stage`, syncs `checkpoint/` on an interval, and uploads `out/`. That function
  is not in this repository. `SubprocessAdapter` is the worked example of what it has to
  do.

`providers.py` is the source of the price, and for Modal it has one surveyed figure: an
A100 hour. An L4 or A10G run therefore records the seconds it was billed and no cost,
which is deliberate -- see that module.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cloud import Poll, RemoteHandle, RemoteState, StageRequest
from providers import Rate, provider

__all__ = ["ModalAdapter"]

#: Tier -> what Modal's `gpu=` argument is believed to want. Unverified, like the rest.
GPU_NAMES: Mapping[str, str] = {"l4": "L4", "a10g": "A10G", "a100": "A100", "h100": "H100"}

#: Exception class names that are *guessed* to mean "the container was taken back".
#: See the module docstring: this is the single most important thing to verify.
PREEMPTION_MARKERS: tuple[str, ...] = (
    "PreemptedError",
    "InterruptedError",
    "FunctionInterrupted",
)


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
    is the one claim made for this file: the protocol fits a real provider. Whether these
    particular Modal calls are the right ones is unknown.
    """

    #: Matches `providers.PROVIDERS`, so `jobs.provider` and the console's price table
    #: agree about which host this is.
    name = "modal"
    #: Modal's own tiers are not interruptible; this is why it is the sensible last entry
    #: in a `Placement`, the one a stage falls back *to*.
    interruptible = False

    def __init__(
        self,
        app_name: str,
        function_name: str = "run_stage",
        *,
        rates: Mapping[str, Rate] | None = None,
        timeout_s: float = 6 * 3600.0,
    ) -> None:
        self._app_name = app_name
        self._function_name = function_name
        self._rates = dict(rates or {})
        self._timeout_s = timeout_s
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
        function = self._function()
        # UNVERIFIED: that `spawn` is the non-blocking call, that it returns an object
        # with `.object_id`, and that a plain dict is an acceptable argument.
        call = function.spawn(request.to_dict())
        handle = RemoteHandle(id=str(call.object_id), provider=self.name, tier=request.tier)
        self._calls[handle.id] = _Call(request=request, call=call, started_at=time.monotonic())
        return handle

    def poll(self, handle: RemoteHandle) -> Poll:
        run = self._calls[handle.id]
        if run.state in ("succeeded", "failed", "preempted"):
            return self._result(run)
        run.billed_s = time.monotonic() - run.started_at  # UNVERIFIED: see the docstring.
        try:
            # UNVERIFIED: that `get(timeout=0)` is the way to ask "is it done yet" without
            # blocking, and that it raises rather than returning a sentinel while running.
            outcome = run.call.get(timeout=0)
        except TimeoutError:
            return Poll(state="running", billed_s=run.billed_s)
        except Exception as error:  # anything Modal raises is a verdict on the call
            run.state = self._state_of(error)
            run.detail = f"{type(error).__name__}: {error}"
            return self._result(run)
        run.state = "succeeded"
        run.metrics = dict((outcome or {}).get("metrics") or {})
        run.summary = str((outcome or {}).get("summary") or "")
        return self._result(run)

    def logs(self, handle: RemoteHandle, *, since: int = 0) -> Sequence[str]:
        """Not implemented: Modal's log API was not reachable to be written against.

        Returning nothing is honest and harmless -- `CloudRunner` tails whatever it is
        given -- whereas a guess here would put invented lines in a stage's log.
        """
        return ()

    def cancel(self, handle: RemoteHandle) -> None:
        run = self._calls.get(handle.id)
        if run is None or run.state in ("succeeded", "failed", "preempted"):
            return
        # UNVERIFIED: the method name. A provider whose `cancel` does nothing is a
        # provider that keeps billing, which is why this is worth checking early.
        run.call.cancel()
        run.state = "failed"
        run.detail = "cancelled"

    # --- helpers ------------------------------------------------------------------

    def _function(self) -> Any:
        try:
            import modal  # optional, and absent everywhere this actually runs
        except ImportError as error:  # pragma: no cover - modal is not a dependency here
            raise RuntimeError(
                "ModalAdapter needs the `modal` package, which is deliberately not a "
                "dependency of tools/pipeline: no part of this adapter has ever been run, "
                "so nothing here should pull a client library into the image. Install it "
                "in the deployment that actually uses Modal"
            ) from error
        # UNVERIFIED: that this is how a deployed function is looked up by name.
        return modal.Function.from_name(self._app_name, self._function_name)

    @staticmethod
    def _state_of(error: BaseException) -> RemoteState:
        """Preempted or failed. A guess, and the most consequential one in the file."""
        return "preempted" if type(error).__name__ in PREEMPTION_MARKERS else "failed"

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
