"""Running a stage somewhere else: the two seams, and the runner that drives them.

`CloudRunner` is a `BaseRunner` like any other. It overrides `_invoke` and nothing else,
so wiping `out/`, verifying the declared `produces`, hashing the artifacts and writing
`step.json` all still happen exactly once, in `BaseRunner`, for the cloud path too. There
is no second code path; there is one code path with a different `_invoke`.

Two protocols, both defined here, both injected, because `tools/pipeline` may not grow a
`boto3` and may not import `apps/api`:

* **`ProviderAdapter`** -- submit a stage, poll it, read its logs, cancel it, price a
  tier. `poll` returns `preempted` as a state of its own, separate from `failed`, because
  that distinction is the entire point: a preempted attempt is ordinary operation that
  resumes, a failed one is a bug that retries and then gives up.
* **`Transfer`** -- move a stage's inputs and checkpoint out to wherever the provider can
  read them, and its outputs back. The worker supplies an S3 implementation over the
  `ObjectStorage` it already has; tests supply one over a local directory.

This is the convention `app/worker/registration.py` already set: **the pipeline describes,
the worker performs.** The pipeline knows a stage's inputs must be somewhere the GPU box
can read them and under which key; it does not know what a bucket is.

Preemption is not an error path. The flow is:

    attempt N   push checkpoint/ -> poll -> `preempted` -> pull checkpoint/ back
                -> record the attempt and what it billed -> raise PreemptedError
    attempt N+1 `prepare_stage` wipes out/ and keeps checkpoint/ -> push it again
                -> the remote picks up where it left off

and a stage that keeps losing its box moves to the reliable provider rather than being
retried on the cheap one until the cheap one has cost more (`Placement`).

**Fanning out.** A stage may split into pieces that each want a machine of their own
(`contracts.FanOut`; block training is the one that does). The runner, not the stage and
not the provider, runs them: the head call answers with the pieces, the runner submits
them concurrently -- at most `parallel` at once -- through the same adapter, polls them
all from this one thread, retries a piece that failed or was preempted on its own while
the others carry on, brings each finished piece's result home into `checkpoint/`, and
then makes the join call that writes the stage's outputs. Every call is a line in the
attempt ledger, so what a stage cost is still the sum of what was billed, pieces included.

It is here rather than inside a GPU container because a container that spawns the pieces
and waits for them is a GPU billed to wait, and rather than on a CPU orchestrator in
Modal because retrying, falling back to another provider, pricing and the checkpoint all
already live in this runner: a second orchestrator would need its own copy of each, and
its children's billed seconds would never reach `attempts.json`.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

import live
import progress
from artifacts import ArtifactDecl, checksum_of
from contracts import (
    FANOUT_METRIC,
    FANOUT_PARAM,
    FanOut,
    FanOutPart,
    MetricValue,
    StageContext,
    StageOutcome,
)
from errors import NoRunnerError, PreemptedError, RemoteStageError
from phases import flat, parse_flat
from plan import PlannedStage
from providers import Rate
from runners import BaseRunner
from workdir import Workdir

__all__ = [
    "Attempt",
    "AttemptLedger",
    "CloudRunner",
    "Placement",
    "Poll",
    "ProviderAdapter",
    "RemoteHandle",
    "RemoteState",
    "RunCost",
    "StageKeys",
    "StageRequest",
    "Transfer",
    "call_phases",
    "run_cost",
]

#: Where a submitted stage is. `preempted` is deliberately not a kind of `failed`.
RemoteState = Literal["pending", "running", "succeeded", "failed", "preempted"]

#: The states a poll loop stops on.
TERMINAL: tuple[RemoteState, ...] = ("succeeded", "failed", "preempted")


# --- what crosses the seam ---------------------------------------------------------


@dataclass(frozen=True)
class StageKeys:
    """The object-storage keys one stage uses, all derived from `checkpoint_key`.

    `StageContext.checkpoint_key` is the workdir contract's `runs/<run>/<stage>/checkpoint`
    (A6 put it there for exactly this step), so everything else this runner needs is a
    sibling of it and no second naming scheme is invented.
    """

    root: str
    stage: str
    checkpoint: str

    @staticmethod
    def of(context: StageContext) -> StageKeys:
        stage = context.checkpoint_key.removesuffix("/checkpoint")
        return StageKeys(
            root=stage.removesuffix(f"/{context.stage_id}"),
            stage=stage,
            checkpoint=context.checkpoint_key,
        )

    @property
    def outputs(self) -> str:
        """Where the remote puts `out/`. Under `transfer/` so it cannot collide with the
        per-artifact keys the worker uploads finished artifacts to."""
        return f"{self.stage}/transfer/out"

    def input(self, workdir_relative: str) -> str:
        """An input's key, from its path in the workdir.

        Keyed by path rather than by consuming stage, so an artifact two stages read is
        uploaded once and the second stage finds it already there.
        """
        return f"{self.root}/transfer/{workdir_relative}"

    def part_checkpoint(self, part: str) -> str:
        """One fanned-out piece's own checkpoint key. A sibling of the stage's, never under
        it: every running piece syncs its `checkpoint/` wholesale, and N of them sharing a
        key would each replace what the others had synced."""
        return f"{self.stage}/parts/{part}/checkpoint"

    def part_outputs(self, part: str) -> str:
        return f"{self.stage}/parts/{part}/out"

    def part_root(self, part: str) -> str:
        return f"{self.stage}/parts/{part}"


@dataclass(frozen=True)
class StageRequest:
    """Everything a provider is told about the stage it is being asked to run.

    Deliberately not a `PlannedStage`: a remote box gets the facts it needs to run one
    stage and no view of the recipe around it. Everything here is JSON-serialisable,
    because for a real provider it becomes the payload of an HTTP call.
    """

    recipe: str
    run_id: str
    stage_id: str
    impl: str
    attempt: int
    tier: str
    preemptible: bool
    params: Mapping[str, Any]
    #: Artifact name -> the transfer key its bytes are under.
    inputs: Mapping[str, str]
    #: What this stage declares it produces. The declarations rather than the names,
    #: so a machine that has never seen the recipe knows whether to write a file or a
    #: directory and which of its members are not optional.
    produces: tuple[ArtifactDecl, ...]
    #: Where the remote reads its checkpoint from on start and syncs it back to as it
    #: runs. The sync is on an interval: a checkpoint that only appears at the end is
    #: worth nothing to an attempt that does not reach the end.
    checkpoint_key: str
    #: Where the remote puts the contents of `out/` when it finishes.
    outputs_key: str
    #: How often the remote should sync `checkpoint/` while it runs.
    checkpoint_every_s: float = 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "runId": self.run_id,
            "stageId": self.stage_id,
            "impl": self.impl,
            "attempt": self.attempt,
            "tier": self.tier,
            "preemptible": self.preemptible,
            "params": dict(self.params),
            "inputs": dict(self.inputs),
            "produces": [decl.to_dict() for decl in self.produces],
            "checkpointKey": self.checkpoint_key,
            "outputsKey": self.outputs_key,
            "checkpointEveryS": self.checkpoint_every_s,
        }

    @staticmethod
    def from_dict(document: Mapping[str, Any]) -> StageRequest:
        return StageRequest(
            recipe=str(document["recipe"]),
            run_id=str(document["runId"]),
            stage_id=str(document["stageId"]),
            impl=str(document["impl"]),
            attempt=int(document["attempt"]),
            tier=str(document["tier"]),
            preemptible=bool(document["preemptible"]),
            params=dict(document.get("params") or {}),
            inputs={str(k): str(v) for k, v in dict(document.get("inputs") or {}).items()},
            produces=tuple(
                ArtifactDecl.from_dict(entry) for entry in document.get("produces") or ()
            ),
            checkpoint_key=str(document["checkpointKey"]),
            outputs_key=str(document["outputsKey"]),
            checkpoint_every_s=float(document.get("checkpointEveryS", 60.0)),
        )


@dataclass(frozen=True)
class RemoteHandle:
    """What `submit` hands back: enough to poll, read and cancel, and nothing more."""

    id: str
    provider: str
    tier: str


@dataclass(frozen=True)
class Poll:
    """One look at a submitted stage.

    `billed_s` is cumulative and is what the provider says it has charged for so far, not
    what a stopwatch here measured -- queue time a provider does not bill for is not a
    cost, and a provider that bills a whole minute for eleven seconds did bill a minute.
    A preempted attempt reports what it burned before it was taken away, which is the
    number that must not be quietly dropped.
    """

    state: RemoteState
    billed_s: float = 0.0
    detail: str = ""
    metrics: Mapping[str, MetricValue] | None = None
    summary: str = ""


@runtime_checkable
class ProviderAdapter(Protocol):
    """One GPU host, behind five methods.

    Every method is called from the supervising process, never from the stage. An adapter
    holds credentials and talks to an API; it does not import the pipeline's executor and
    it never touches the workdir -- the bytes move through `Transfer`.
    """

    #: The name recorded in `jobs.provider`; matches `providers.PROVIDERS` where the
    #: adapter speaks for a real host.
    name: str
    #: True where being killed mid-stage is ordinary. `Placement` reads this to decide
    #: which adapter is the one to fall back *to*.
    interruptible: bool

    def rate(self, tier: str) -> Rate | None:
        """What an hour of `tier` costs here, or None where nobody has measured it.

        None is a real answer and must stay one: it produces a run that records the
        seconds it was billed and no cost, instead of a plausible-looking invented one.
        """
        ...

    def submit(self, request: StageRequest) -> RemoteHandle: ...

    def poll(self, handle: RemoteHandle) -> Poll: ...

    def logs(self, handle: RemoteHandle, *, since: int = 0) -> Sequence[str]:
        """Log lines from index `since` onward, so a poll loop can tail without repeating."""
        ...

    def cancel(self, handle: RemoteHandle) -> None:
        """Stop it and stop paying for it. Must be safe to call on a finished stage."""
        ...


@runtime_checkable
class Transfer(Protocol):
    """Moving a stage's bytes between this workdir and somewhere the provider can read.

    Four methods, because that is all the runner needs. `put` and `get` take a file or a
    directory and are symmetric: whatever `put` stored under a key, `get` reconstructs at
    the target. Both return the number of bytes moved, which is what makes a transfer
    visible in a stage log instead of being a silent pause.
    """

    def put(self, key: str, source: Path) -> int: ...

    def get(self, key: str, target: Path) -> int:
        """Restore `key` at `target`. A key that holds nothing is not an error: it
        returns 0 and leaves the target alone, which is a first attempt with no
        checkpoint to resume from."""
        ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None: ...


# --- the attempt ledger ------------------------------------------------------------


@dataclass(frozen=True)
class Attempt:
    """One attempt at one stage: where it ran, how it ended, what it cost.

    Written for **every** attempt, preempted ones included. A cost number that counts
    only the attempt that happened to succeed is a cost number that hides the two hours
    the cheap tier lost, which is the exact failure this step exists to make visible.
    """

    attempt: int
    provider: str
    tier: str
    state: RemoteState
    billed_s: float
    usd: float | None = None
    rate_source: str = ""
    detail: str = ""
    #: Which call of a fanned-out attempt this was: a piece's id, or `join`. Empty for the
    #: stage's own (head) call, so a stage that never fans out writes the ledger it always
    #: did. Several entries share one `attempt` number when an attempt fanned out.
    part: str = ""

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "attempt": self.attempt,
            "provider": self.provider,
            "tier": self.tier,
            "state": self.state,
            "billedS": round(self.billed_s, 6),
            "usd": None if self.usd is None else round(self.usd, 6),
            "rateSource": self.rate_source,
            "detail": self.detail,
        }
        if self.part:
            document["part"] = self.part
        return document

    @staticmethod
    def from_dict(document: Mapping[str, Any]) -> Attempt:
        usd = document.get("usd")
        state = cast("RemoteState", document.get("state", "failed"))
        return Attempt(
            attempt=int(document["attempt"]),
            provider=str(document["provider"]),
            tier=str(document["tier"]),
            state=state,
            billed_s=float(document.get("billedS", 0.0)),
            usd=None if usd is None else float(usd),
            rate_source=str(document.get("rateSource", "")),
            detail=str(document.get("detail", "")),
            part=str(document.get("part", "")),
        )


@dataclass(frozen=True)
class AttemptLedger:
    """Every attempt at one stage, in order, in `stages/<id>/attempts.json`.

    It lives beside `step.json` rather than inside `checkpoint/` or `work/` because it
    must outlive both an attempt (`out/` is wiped) and a machine (the supervisor reads it
    after the run to fill in `jobs.cost_usd`), and because a stage implementation has no
    business seeing it.
    """

    entries: tuple[Attempt, ...] = ()

    @staticmethod
    def read(path: Path) -> AttemptLedger:
        if not path.is_file():
            return AttemptLedger()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            return AttemptLedger(tuple(Attempt.from_dict(entry) for entry in document["attempts"]))
        except (ValueError, KeyError, TypeError):
            # A truncated ledger -- the machine died mid-write -- must not stop the next
            # attempt. It costs the history of what was already paid for, which is
            # recorded again from the next attempt onward.
            return AttemptLedger()

    def append(self, entry: Attempt) -> AttemptLedger:
        return AttemptLedger((*self.entries, entry))

    def write(self, path: Path) -> None:
        document = {"attempts": [entry.to_dict() for entry in self.entries]}
        path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", "utf-8")

    @property
    def preemptions(self) -> int:
        return sum(1 for entry in self.entries if entry.state == "preempted")

    @property
    def billed_s(self) -> float:
        return sum(entry.billed_s for entry in self.entries)

    @property
    def unpriced_s(self) -> float:
        return sum(entry.billed_s for entry in self.entries if entry.usd is None)

    @property
    def usd(self) -> float | None:
        priced = [entry.usd for entry in self.entries if entry.usd is not None]
        return sum(priced) if priced else None

    @property
    def last(self) -> Attempt | None:
        return self.entries[-1] if self.entries else None


@dataclass(frozen=True)
class RunCost:
    """What a whole run was billed, across every stage and every attempt of it."""

    billed_s: float = 0.0
    unpriced_s: float = 0.0
    usd: float | None = None
    provider: str | None = None
    tier: str | None = None
    attempts: int = 0
    preemptions: int = 0

    @property
    def complete(self) -> bool:
        """True when every billed second was priced. False means `usd` is a floor."""
        return self.unpriced_s == 0.0


def run_cost(workdir: Workdir) -> RunCost:
    """Add up every stage's ledger. This is what fills `jobs.cost_usd`.

    `provider` and `tier` are the *last* attempt's, because that is where the work ended
    up -- a run that started on a community host and finished on Modal is a Modal run
    that has a preemption history, and the history is in the step metrics.
    """
    total = RunCost()
    for path in workdir.attempt_ledgers():
        ledger = AttemptLedger.read(path)
        if not ledger.entries:
            continue
        usd = total.usd
        stage_usd = ledger.usd
        combined = usd if stage_usd is None else (stage_usd if usd is None else usd + stage_usd)
        last = ledger.entries[-1]
        total = replace(
            total,
            billed_s=total.billed_s + ledger.billed_s,
            unpriced_s=total.unpriced_s + ledger.unpriced_s,
            usd=combined,
            provider=last.provider,
            tier=last.tier,
            attempts=total.attempts + len(ledger.entries),
            preemptions=total.preemptions + ledger.preemptions,
        )
    return total


# --- placement ---------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """Which provider this attempt goes to, given how often the work has been lost.

    A cheap interruptible box that keeps being taken away is not cheap. After
    `preemptions_before_fallback` preemptions the stage moves to the next adapter, and the
    last one in the list should be a reliable host -- `Placement.of` says so out loud if
    it is not. Without this, a two-hour stage on a host that preempts hourly is retried
    until the attempt budget is gone, having paid for the same two hours three times and
    produced nothing.
    """

    adapters: tuple[ProviderAdapter, ...]
    preemptions_before_fallback: int = 2

    @staticmethod
    def of(*adapters: ProviderAdapter, preemptions_before_fallback: int = 2) -> Placement:
        if not adapters:
            raise NoRunnerError(
                "a CloudRunner needs at least one ProviderAdapter; it was given none"
            )
        if len(adapters) > 1 and adapters[-1].interruptible:
            raise NoRunnerError(
                f"the last provider in a placement is the one a preempted stage falls back "
                f"to, so it must not itself be interruptible -- {adapters[-1].name!r} is. "
                f"Put a reliable provider last"
            )
        return Placement(tuple(adapters), preemptions_before_fallback)

    def adapter_for(self, preemptions: int) -> ProviderAdapter:
        if not self.adapters:
            raise NoRunnerError("a CloudRunner needs at least one ProviderAdapter")
        step = max(1, self.preemptions_before_fallback)
        return self.adapters[min(preemptions // step, len(self.adapters) - 1)]


# --- the runner --------------------------------------------------------------------


class CloudRunner(BaseRunner):
    """A GPU stage, run on somebody else's machine.

    Overrides `_invoke` and nothing else. Everything a `LocalRunner` stage gets --
    a cleared `out/`, a verified `produces`, hashed artifacts, a `step.json` -- a cloud
    stage gets too, from the same code, because it is the same code.
    """

    name = "cloud"

    def __init__(
        self,
        placement: Placement,
        transfer: Transfer,
        *,
        poll_interval_s: float = 2.0,
        #: How often the remote is asked to sync `checkpoint/` back to object storage.
        #: The cost of getting this wrong is asymmetric: too often wastes bandwidth, too
        #: rarely throws away everything computed since the last sync when the box goes.
        checkpoint_every_s: float = 60.0,
        max_wait_s: float = 24 * 3600.0,
        #: How long a stage may sit `pending` -- submitted, but no sign it has started --
        #: before it is cancelled. Much shorter than `max_wait_s`, because the two fail
        #: differently: a slow stage is making progress, while a container that
        #: crash-loops on import (the first real Modal deploy did, on `IndexError: 2`)
        #: never starts and never returns, and `max_wait_s` would hold the worker for a
        #: day on it. Thirty minutes covers a cold pull of the ~10 GB training image and
        #: a GPU queue; past that, a person should be told rather than kept waiting.
        max_pending_s: float = 30 * 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        #: The wall clock, for comparing a call's submit time with the remote's own
        #: (`remote.execute`'s `remoteStartedAt`): the queue and the container's start.
        wall: Callable[[], float] = time.time,
        #: Offer the head call a fan-out (`contracts.FANOUT_PARAM`). Off only for a
        #: deployment whose remote image predates the protocol -- which ignores the param
        #: and runs the stage whole anyway, so leaving it on costs nothing there either.
        fan_out: bool = True,
        #: The most pieces of one stage in flight at once, whatever the stage asks for:
        #: the deployment's ceiling. Modal caps a workspace's concurrent GPUs by plan, and
        #: a piece beyond the cap queues as `pending` -- where `max_pending_s` would cancel
        #: it after thirty minutes -- so this should stay under that cap divided by the
        #: number of runs the worker may have training at once.
        max_parallel: int = 8,
        #: Calls a piece gets within one attempt before the attempt gives up on it. A
        #: preempted or failed piece is retried alone while the others keep running; one
        #: that fails every time is a bug, and the attempt then ends (after the others
        #: finish, so their work is kept) for the worker's own retry to resume.
        part_attempts: int = 3,
    ) -> None:
        self._placement = placement
        self._transfer = transfer
        self._poll_interval_s = poll_interval_s
        self._checkpoint_every_s = checkpoint_every_s
        self._max_wait_s = max_wait_s
        self._max_pending_s = max_pending_s
        self._clock = clock
        self._sleep = sleep
        self._wall = wall
        self._fan_out = fan_out
        self._max_parallel = max(1, max_parallel)
        self._part_attempts = max(1, part_attempts)

    def _invoke(self, stage: PlannedStage, context: StageContext) -> StageOutcome:
        ledger = AttemptLedger.read(context.attempts_path)
        first = len(ledger.entries)
        adapter = self._placement.adapter_for(ledger.preemptions)
        keys = StageKeys.of(context)
        tier = context.gpu_tier or "cpu"
        context.log(
            f"cloud: stage {context.stage_id!r} attempt {context.attempt} on "
            f"{adapter.name!r} ({tier}); {ledger.preemptions} preemption(s) so far"
        )
        staging = time.monotonic()
        self._stage_in(stage, context, keys)
        staged_s = time.monotonic() - staging
        context.log(f"cloud: inputs and checkpoint staged in {staged_s:.1f} s")
        params = dict(context.params)
        if self._fan_out:
            params[FANOUT_PARAM] = {"role": "head"}
        adapter, poll, submitted = self._call(stage, context, keys, tier, params)
        fanned: dict[str, MetricValue] = {}
        spec = FanOut.parse((poll.metrics or {}).get(FANOUT_METRIC)) if self._fan_out else None
        if spec is not None:
            fanned["headPhases"] = flat(call_phases(submitted, poll))
            fanned.update(self._fan_out_parts(stage, context, keys, tier, spec))
            # Every piece's result is in `checkpoint/` now; the join call starts from it.
            sending = time.monotonic()
            sent = self._transfer.put(keys.checkpoint, context.checkpoint_dir)
            fanned["joinSendS"] = round(time.monotonic() - sending, 2)
            context.log(f"cloud: sent {sent} byte(s) of checkpoint for the join")
            joined = {**context.params, FANOUT_PARAM: {"role": "join"}}
            adapter, poll, submitted = self._call(stage, context, keys, tier, joined, part="join")
            for part in spec.parts:
                self._transfer.delete(keys.part_root(part.id))
        fetching = time.monotonic()
        moved = self._transfer.get(keys.outputs, context.out_dir)
        back_s = time.monotonic() - fetching
        context.log(
            f"cloud: {moved} byte(s) of output(s) came back from {adapter.name!r} in {back_s:.1f} s"
        )
        ledger = AttemptLedger.read(context.attempts_path)
        metrics = self._metrics(ledger.entries[first:], ledger, poll)
        metrics.update(fanned)
        # The last call's billed seconds, phase by phase (`call_phases`): the one call, or
        # the join. A fan-out's head and parts are `headPhases` and `fanOutPhases`.
        metrics["remotePhases"] = flat(call_phases(submitted, poll))
        # The worker's side of the transfer, beside the remote's own (`remote*S`): with
        # `billedS` they account for a remote stage's wall time end to end.
        metrics["stageInS"] = round(staged_s, 2)
        metrics["outputsBackS"] = round(back_s, 2)
        return StageOutcome(metrics=metrics, summary=poll.summary)

    def _request(
        self,
        stage: PlannedStage,
        context: StageContext,
        keys: StageKeys,
        tier: str,
        params: Mapping[str, Any],
        *,
        checkpoint_key: str,
        outputs_key: str,
        produces: tuple[ArtifactDecl, ...],
    ) -> StageRequest:
        return StageRequest(
            recipe=context.recipe,
            run_id=context.run_id,
            stage_id=context.stage_id,
            impl=context.impl,
            attempt=context.attempt,
            tier=tier,
            preemptible=bool(stage.gpu and stage.gpu.preemptible),
            params=dict(params),
            inputs={name: keys.input(path) for name, path in stage.inputs.items()},
            produces=produces,
            checkpoint_key=checkpoint_key,
            outputs_key=outputs_key,
            checkpoint_every_s=self._checkpoint_every_s,
        )

    def _call(
        self,
        stage: PlannedStage,
        context: StageContext,
        keys: StageKeys,
        tier: str,
        params: Mapping[str, Any],
        *,
        part: str = "",
    ) -> tuple[ProviderAdapter, Poll, float]:
        """One call on the stage's own checkpoint key: the head, or a fan-out's join.
        Raises unless it succeeded, having brought the checkpoint home and recorded it.
        Returns the wall-clock time it was submitted at, too (`call_phases`)."""
        ledger = AttemptLedger.read(context.attempts_path)
        adapter = self._placement.adapter_for(ledger.preemptions)
        request = self._request(
            stage,
            context,
            keys,
            tier,
            params,
            checkpoint_key=keys.checkpoint,
            outputs_key=keys.outputs,
            produces=stage.impl.produces,
        )
        submitted = self._wall()
        handle = adapter.submit(request)
        poll = self._watch(adapter, handle, context)
        # Whatever the remote last synced comes home before anything else is decided, so
        # the next attempt resumes from it. Disable this and a preempted stage restarts.
        self._restore_checkpoint(context)
        entry = self._record(adapter, context, tier, poll, ledger, part=part)
        if poll.state == "preempted":
            raise PreemptedError(
                context.recipe, context.stage_id, context.attempt, adapter.name, entry.billed_s
            )
        if poll.state != "succeeded":
            raise RemoteStageError(
                context.recipe, context.stage_id, context.impl, adapter.name, poll.detail
            )
        return adapter, poll, submitted

    # --- fanning out -----------------------------------------------------------------

    def _fan_out_parts(
        self,
        stage: PlannedStage,
        context: StageContext,
        keys: StageKeys,
        tier: str,
        spec: FanOut,
    ) -> dict[str, MetricValue]:
        """Run every piece, at most `parallel` at once; raise if any is lost for good.

        One thread polls them all: the adapters' `poll` does not block (Modal's is a
        zero-timeout `get`), so N pieces cost N polls a round, not N threads. A piece that
        ends in anything but success is resubmitted alone, up to `part_attempts` calls;
        a piece that runs out of calls does not stop the others, because each finished
        piece is kept in `checkpoint/` and the next attempt starts only what is missing.
        """
        parallel = min(spec.parallel, self._max_parallel)
        context.log(
            f"cloud: fanning out {len(spec.parts)} part(s) of stage {context.stage_id!r}, "
            f"{parallel} at a time ({', '.join(p.id for p in spec.parts)})"
        )
        queue: deque[FanOutPart] = deque(spec.parts)
        calls: dict[str, int] = {part.id: 0 for part in spec.parts}
        billed: dict[str, float] = {part.id: 0.0 for part in spec.parts}
        running: dict[str, _PartRun] = {}
        lost: dict[str, tuple[ProviderAdapter, Poll]] = {}
        #: Each finished piece's billed seconds, phase by phase (`call_phases`).
        timed: dict[str, str] = {}
        collect_s = 0.0
        tail = _FanOutLog(context)
        started = self._clock()
        peak = 0
        try:
            while queue or running:
                while queue and len(running) < parallel:
                    part = queue.popleft()
                    calls[part.id] += 1
                    running[part.id] = self._submit_part(
                        stage, context, keys, tier, spec, part, calls[part.id]
                    )
                peak = max(peak, len(running))
                for part_id, run in list(running.items()):
                    poll = self._poll_part(run, context, tail)
                    if poll.state not in TERMINAL:
                        continue
                    del running[part_id]
                    tail.finished(part_id)
                    if poll.state == "succeeded":
                        collecting = time.monotonic()
                        poll = self._collect_part(context, keys, run.part, poll)
                        collected = time.monotonic() - collecting
                        collect_s += collected
                        timed[part_id] = flat(
                            call_phases(run.submitted_at, poll, collect_s=collected)
                        )
                    entry = self._record(
                        run.adapter,
                        context,
                        tier,
                        poll,
                        AttemptLedger.read(context.attempts_path),
                        part=part_id,
                    )
                    billed[part_id] += entry.billed_s
                    if poll.state == "succeeded":
                        continue
                    if calls[part_id] < self._part_attempts:
                        context.log(
                            f"cloud: part {part_id} {poll.state} on call {calls[part_id]} of "
                            f"{self._part_attempts}; resubmitting it alone"
                        )
                        # To the front: a piece that has already cost a call is the one
                        # most likely to decide when the whole stage ends.
                        queue.appendleft(run.part)
                    else:
                        lost[part_id] = (run.adapter, poll)
                if queue or running:
                    self._sleep(self._poll_interval_s)
        except BaseException:
            # Whatever stops the loop -- a cancelled job, the worker shutting down -- stops
            # paying for every piece still out there, not just the one being polled.
            for run in running.values():
                run.adapter.cancel(run.handle)
            raise
        wall = self._clock() - started
        total = sum(billed.values())
        context.log(
            f"cloud: {len(spec.parts) - len(lost)} of {len(spec.parts)} part(s) finished in "
            f"{wall:.0f} s of wall time, {total:.0f} s billed across them "
            f"({', '.join(f'{k} {v:.0f} s' for k, v in billed.items())})"
        )
        if lost:
            detail = "; ".join(
                f"part {part_id} {poll.state} on {adapter.name!r}: {poll.detail or 'no detail'}"
                for part_id, (adapter, poll) in lost.items()
            )
            adapter = next(iter(lost.values()))[0]
            if all(poll.state == "preempted" for _, poll in lost.values()):
                raise PreemptedError(
                    context.recipe,
                    context.stage_id,
                    context.attempt,
                    adapter.name,
                    sum(billed[part_id] for part_id in lost),
                )
            raise RemoteStageError(
                context.recipe, context.stage_id, context.impl, adapter.name, detail
            )
        return {
            "fanOutParts": len(spec.parts),
            "fanOutParallel": parallel,
            "fanOutPeak": peak,
            "fanOutCalls": sum(calls.values()),
            "fanOutWallS": round(wall, 1),
            "fanOutBilledS": round(total, 3),
            "fanOutBilledByPart": ",".join(f"{k}:{v:.1f}" for k, v in billed.items()),
            # Where each piece's billed seconds went (`call_phases`), and the runner's own
            # time bringing the pieces home -- not billed, but the other pieces wait on it.
            "fanOutPhases": "; ".join(f"{k} {v}" for k, v in timed.items()),
            "fanOutCollectS": round(collect_s, 2),
        }

    def _submit_part(
        self,
        stage: PlannedStage,
        context: StageContext,
        keys: StageKeys,
        tier: str,
        spec: FanOut,
        part: FanOutPart,
        call: int,
    ) -> _PartRun:
        """The shared members of `checkpoint/` onto the piece's own key, then submit it.

        Rebuilt for every call, so a retry never starts from what a lost call left there.
        """
        bundle = context.work_dir / "fanout" / part.id
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True)
        for member in spec.share:
            _copy_member(context.checkpoint_dir / member, bundle / member)
        checkpoint = keys.part_checkpoint(part.id)
        self._transfer.delete(checkpoint)
        if any(bundle.iterdir()):
            self._transfer.put(checkpoint, bundle)
        params = {**context.params, FANOUT_PARAM: {"role": "part", "part": part.id}}
        request = self._request(
            stage,
            context,
            keys,
            tier,
            params,
            checkpoint_key=checkpoint,
            outputs_key=keys.part_outputs(part.id),
            produces=(),
        )
        adapter = self._placement.adapter_for(AttemptLedger.read(context.attempts_path).preemptions)
        submitted = self._wall()
        handle = adapter.submit(request)
        context.log(f"cloud: part {part.id} submitted to {adapter.name!r} (call {call})")
        return _PartRun(
            part=part,
            adapter=adapter,
            handle=handle,
            started=self._clock(),
            submitted_at=submitted,
        )

    def _poll_part(self, run: _PartRun, context: StageContext, tail: _FanOutLog) -> Poll:
        poll = run.adapter.poll(run.handle)
        lines = run.adapter.logs(run.handle, since=run.cursor)
        run.cursor += len(lines)
        for line in lines:
            tail.line(run.part.id, line)
        if poll.state in TERMINAL:
            return poll
        return self._overdue(run.adapter, run.handle, poll, self._clock() - run.started)

    def _collect_part(
        self, context: StageContext, keys: StageKeys, part: FanOutPart, poll: Poll
    ) -> Poll:
        """A finished piece's result, from its key into the stage's `checkpoint/`.

        Only the paths it declared (`FanOutPart.collect`), each fetched on its own and
        replaced whole. Fetched member by member rather than the piece's whole key, which
        also holds the shared members it was sent (a block's prior) and its live
        snapshots: bytes that went out with it and are already here. A result that did not
        come back makes the call a failure: the piece claimed to finish and there is
        nothing for the join to use, so it runs again.
        """
        incoming = context.work_dir / "fanout" / f"{part.id}.incoming"
        if incoming.exists():
            shutil.rmtree(incoming)
        incoming.mkdir(parents=True)
        for member in part.collect:
            self._transfer.get(f"{keys.part_checkpoint(part.id)}/{member}", incoming / member)
        missing = [member for member in part.collect if not (incoming / member).exists()]
        if missing:
            shutil.rmtree(incoming, ignore_errors=True)
            return replace(
                poll,
                state="failed",
                detail=f"part {part.id} finished but its {', '.join(missing)} did not come back",
            )
        for member in part.collect:
            target = context.checkpoint_dir / member
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            (incoming / member).rename(target)
        shutil.rmtree(incoming, ignore_errors=True)
        return poll

    # --- the pieces, each of which a test can reach --------------------------------

    def _stage_in(self, stage: PlannedStage, context: StageContext, keys: StageKeys) -> None:
        """Inputs, then the checkpoint. An input already up is not sent twice.

        The workdir's `checkpoint/` is the truth about how far this stage got; object
        storage is only how the bytes travel. So an empty one **clears** the remote copy
        rather than leaving it: a stale object that quietly resumed a stage the workdir
        says is starting over would be a resume nobody asked for and nobody could see.
        """
        for name, relative in sorted(stage.inputs.items()):
            key = keys.input(relative)
            source = context.input(name)
            # "Already up" means the same bytes, not the same key. Keys are per run and
            # per path, so a stage re-run inside a finished run (retry-from-stage, the
            # phone's Refine) produces a *new* artifact at the *old* key -- and a
            # downstream remote stage that found the key and skipped the upload would
            # quietly read the previous run's `trained.ply`. The checksum rides beside
            # the input as a small object of its own.
            digest = checksum_of(source)
            if self._transfer.exists(key) and self._remote_digest(key, context) == digest:
                continue
            self._transfer.delete(key)
            sent = self._transfer.put(key, source)
            self._put_digest(key, digest, context)
            context.log(f"cloud: sent input {name!r} ({sent} bytes) to {key}")
        if context.has_checkpoint:
            sent = self._transfer.put(keys.checkpoint, context.checkpoint_dir)
            context.log(
                f"cloud: sent {sent} byte(s) of checkpoint to the provider (an earlier "
                f"attempt's state, or a seed an earlier run left for this one)"
            )
        else:
            self._transfer.delete(keys.checkpoint)

    def _remote_digest(self, key: str, context: StageContext) -> str | None:
        """The checksum recorded beside an uploaded input, or None if there is none."""
        local = context.work_dir / "transfer-digests" / "remote"
        if local.exists():
            local.unlink()
        if self._transfer.get(_digest_key(key), local) == 0 or not local.is_file():
            return None
        return local.read_text(encoding="utf-8").strip() or None

    def _put_digest(self, key: str, digest: str, context: StageContext) -> None:
        local = context.work_dir / "transfer-digests" / "local"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(digest + "\n", encoding="utf-8")
        self._transfer.put(_digest_key(key), local)

    def _restore_checkpoint(self, context: StageContext) -> int:
        """Bring back whatever the remote last synced.

        This one call is the whole resume story. Remove it and `prepare_stage` keeping
        `checkpoint/` buys nothing, because the state the remote built lives on the
        remote and arrives only through here -- the next attempt then starts from zero
        and the stage never finishes. `tests/test_cloud_preemption.py` holds it to that.

        It *replaces* `checkpoint/` with what came back rather than copying over it: a
        file the remote deleted -- a block run's finished splats once they are merged
        (`blocks.py`), an earlier attempt's live snapshots -- would otherwise survive here
        and be sent out again with every later attempt and run. Fetched beside it first,
        so a transfer that fails half way leaves the local checkpoint as it was; and
        nothing at the key (a remote that never synced) leaves it alone too.
        """
        incoming = context.checkpoint_dir.with_name(f"{context.checkpoint_dir.name}.incoming")
        if incoming.exists():
            shutil.rmtree(incoming)
        moved = self._transfer.get(context.checkpoint_key, incoming)
        if not incoming.is_dir() or not any(incoming.iterdir()):
            shutil.rmtree(incoming, ignore_errors=True)
            return moved
        for member in list(context.checkpoint_dir.iterdir()):
            if member.is_dir() and not member.is_symlink():
                shutil.rmtree(member)
            else:
                member.unlink()
        for member in list(incoming.iterdir()):
            member.rename(context.checkpoint_dir / member.name)
        incoming.rmdir()
        return moved

    def _watch(self, adapter: ProviderAdapter, handle: RemoteHandle, context: StageContext) -> Poll:
        """Poll until it ends, tailing the log. Anything that stops this cancels first:
        a stage nobody is watching any more is a stage nobody has stopped paying for."""
        started = self._clock()
        cursor = 0
        poll = Poll(state="pending")
        try:
            while True:
                poll = adapter.poll(handle)
                cursor = self._tail(adapter, handle, context, cursor)
                if poll.state in TERMINAL:
                    return poll
                poll = self._overdue(adapter, handle, poll, self._clock() - started)
                if poll.state in TERMINAL:
                    return poll
                self._sleep(self._poll_interval_s)
        except BaseException:
            adapter.cancel(handle)
            raise

    def _overdue(
        self, adapter: ProviderAdapter, handle: RemoteHandle, poll: Poll, waited: float
    ) -> Poll:
        """`poll`, or a cancelled failure when the call has pended or run too long."""
        if poll.state == "pending" and waited >= self._max_pending_s:
            adapter.cancel(handle)
            return replace(
                poll,
                state="failed",
                detail=(
                    f"the stage never started on {adapter.name!r}: still pending "
                    f"after {self._max_pending_s:.0f}s, so it was cancelled. "
                    f"A container that crashes on start (check the provider's "
                    f"logs for this app) or no {handle.tier} capacity"
                ),
            )
        if waited >= self._max_wait_s:
            adapter.cancel(handle)
            return replace(
                poll,
                state="failed",
                detail=(
                    f"the stage was still {poll.state} after {self._max_wait_s:.0f}s "
                    f"on {adapter.name!r} and was cancelled"
                ),
            )
        return poll

    @staticmethod
    def _tail(
        adapter: ProviderAdapter, handle: RemoteHandle, context: StageContext, cursor: int
    ) -> int:
        lines = adapter.logs(handle, since=cursor)
        for line in lines:
            context.log(line)
        return cursor + len(lines)

    @staticmethod
    def _record(
        adapter: ProviderAdapter,
        context: StageContext,
        tier: str,
        poll: Poll,
        ledger: AttemptLedger,
        *,
        part: str = "",
    ) -> Attempt:
        """Price this call and write it down, whatever state it ended in."""
        rate = adapter.rate(tier)
        entry = Attempt(
            attempt=context.attempt,
            provider=adapter.name,
            tier=tier,
            state=poll.state,
            billed_s=poll.billed_s,
            usd=None if rate is None else rate.usd_for(poll.billed_s),
            rate_source="" if rate is None else rate.source,
            detail=poll.detail,
            part=part,
        )
        ledger.append(entry).write(context.attempts_path)
        which = f" (part {part})" if part else ""
        billed = (
            f"cloud: {poll.state}{which}; billed {entry.billed_s:.1f}s on {adapter.name!r} ({tier})"
        )
        if rate is None or entry.usd is None:
            context.log(
                f"{billed}; no rate is recorded for that tier, so this attempt is counted "
                f"but not priced"
            )
        else:
            context.log(
                f"{billed} = ${entry.usd:.4f} at ${rate.usd_per_hour:.2f}/h ({entry.rate_source})"
            )
        return entry

    @staticmethod
    def _metrics(
        calls: Sequence[Attempt], ledger: AttemptLedger, poll: Poll
    ) -> dict[str, MetricValue]:
        """This attempt's facts and the stage's running totals, on the successful step.

        The totals are the point: `billedS` alone would say the successful attempt took
        eleven minutes and say nothing about the two preempted attempts before it. An
        attempt that fanned out made several calls, some at the same time; `billedS` and
        `costUsd` are their sum, because each was billed for its own GPU-seconds.
        """
        metrics: dict[str, MetricValue] = dict(poll.metrics or {})
        metrics.pop(FANOUT_METRIC, None)
        last = calls[-1]
        metrics["provider"] = last.provider
        metrics["tier"] = last.tier
        metrics["billedS"] = round(sum(call.billed_s for call in calls), 3)
        metrics["stageBilledS"] = round(ledger.billed_s, 3)
        metrics["preemptions"] = ledger.preemptions
        priced = [call.usd for call in calls if call.usd is not None]
        if priced:
            metrics["costUsd"] = round(sum(priced), 6)
            metrics["rateSource"] = next(c.rate_source for c in calls if c.usd is not None)
        stage_usd = ledger.usd
        if stage_usd is not None:
            metrics["stageCostUsd"] = round(stage_usd, 6)
        if ledger.unpriced_s > 0:
            metrics["unpricedS"] = round(ledger.unpriced_s, 3)
        return metrics


@dataclass
class _PartRun:
    """One call of one fanned-out piece, in flight."""

    part: FanOutPart
    adapter: ProviderAdapter
    handle: RemoteHandle
    started: float
    #: The wall clock at submit (`call_phases`); None where nothing recorded it.
    submitted_at: float | None = None
    cursor: int = 0


@dataclass
class _FanOutLog:
    """The pieces' logs, interleaved into the stage's, each line tagged with its piece.

    Two kinds of line are read by the worker off the stage log, newest first, and N pieces
    printing them at once would make each flicker between pieces. So:

    * a progress line (`progress.parse`, the trainer's tqdm) is kept only from the piece
      that is furthest *behind*: the stage ends when the slowest piece does, so that is
      the bar and the time remaining a person should see;
    * a live splat (`live.parse_line`) is kept only when it is at least as far along as
      the most advanced piece's -- the viewer shows the best-trained block of the scene
      rather than jumping between blocks. The others are logged without the tag.
    """

    context: StageContext
    done_share: dict[str, float] = field(default_factory=dict)
    live_share: dict[str, float] = field(default_factory=dict)

    def line(self, part: str, line: str) -> None:
        tagged = f"[{part}] {line}"
        reading = progress.parse(line)
        if reading is not None and reading.total > 0:
            self.done_share[part] = reading.done / reading.total
            if self.done_share[part] > min(self.done_share.values()):
                return
        parsed = live.parse_line(line)
        if parsed is not None and parsed[0] == "splat":
            payload = parsed[1]
            share = float(payload.get("step") or 0) / max(1.0, float(payload.get("total") or 1))
            best = max(self.live_share.values(), default=0.0)
            self.live_share[part] = share
            if share < best:
                self.context.log(
                    f"[{part}] live: a snapshot at step {payload.get('step')} of "
                    f"{payload.get('total')} is not shown; another block is further along"
                )
                return
        self.context.log(tagged)

    def finished(self, part: str) -> None:
        self.done_share.pop(part, None)


def _number(metrics: Mapping[str, MetricValue], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def call_phases(
    submitted_at: float | None, poll: Poll, *, collect_s: float | None = None
) -> dict[str, float]:
    """One call's billed seconds (`poll.billed_s`), phase by phase, from what the remote
    reported (`remote.execute`, `infra/modal/app.py`) and the stage's own `callPhases`.

    Additive, in order, to the billed figure:

    * `start` -- submit to the function body: the GPU queue, the container's cold start
      and its image (or nothing, on a warm container: `cold` says which);
    * `import` -- the pipeline's imports in the container;
    * `fetch` -- inputs and checkpoint down;
    * the stage's own phases (`phases.py`: `stageDataset`, `budget`, `prepare`, and a
      block's `dataset`, `seed`, `train`, `post`, or the join's `merge`, `eval`,
      `holdout`), with `stageOther` for whatever the stage did not time -- or `stage`
      whole when it timed nothing;
    * `finalSync` and `output` -- the checkpoint's last sync and `out/` up;
    * `rest` -- what none of that covers: the result reaching the runner (a poll
      interval, a log fetch), and any skew between the two clocks.

    Not in the sum: `bgSync`, the checkpoint syncer's own seconds while the stage ran (on
    its own thread, beside the stage), and `collect`, the runner bringing a finished
    piece's result home after the call has stopped billing.
    """
    metrics: Mapping[str, MetricValue] = poll.metrics or {}
    out: dict[str, float] = {}
    entered = _number(metrics, "remoteEnteredAt") or _number(metrics, "remoteStartedAt")
    if submitted_at is not None and entered is not None:
        out["start"] = max(0.0, entered - submitted_at)
    for name, key in (("import", "remoteImportS"), ("fetch", "remoteFetchS")):
        value = _number(metrics, key)
        if value is not None:
            out[name] = value
    stage = _number(metrics, "remoteStageS")
    own = parse_flat(metrics.get("callPhases"))
    if own:
        out.update(own)
        if stage is not None:
            out["stageOther"] = max(0.0, stage - sum(own.values()))
    elif stage is not None:
        out["stage"] = stage
    final = _number(metrics, "remoteFinalSyncS")
    output = _number(metrics, "remoteOutputS")
    if final is not None or output is not None:
        out["finalSync"] = final or 0.0
        out["output"] = output or 0.0
    elif (upload := _number(metrics, "remoteUploadS")) is not None:
        out["upload"] = upload
    out["rest"] = max(0.0, poll.billed_s - sum(out.values()))
    background = _number(metrics, "remoteSyncS")
    if background is not None:
        out["bgSync"] = background
    if collect_s is not None:
        out["collect"] = collect_s
    call = _number(metrics, "containerCall")
    if call is not None:
        out["cold"] = 1.0 if call <= 1 else 0.0
    return out


def _copy_member(source: Path, target: Path) -> None:
    """A file or a directory under `checkpoint/`, copied; nothing if it is not there."""
    if source.is_dir():
        shutil.copytree(source, target)
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _digest_key(key: str) -> str:
    """Where an input's checksum is kept: a sibling object, never a member of the input.

    A sibling rather than `<key>/...` so a directory input's listing -- which is what
    the remote fetches -- never contains it.
    """
    return f"{key}.sha256"
