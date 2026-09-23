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
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

from artifacts import ArtifactDecl
from contracts import MetricValue, StageContext, StageOutcome
from errors import NoRunnerError, PreemptedError, RemoteStageError
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "provider": self.provider,
            "tier": self.tier,
            "state": self.state,
            "billedS": round(self.billed_s, 6),
            "usd": None if self.usd is None else round(self.usd, 6),
            "rateSource": self.rate_source,
            "detail": self.detail,
        }

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
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._placement = placement
        self._transfer = transfer
        self._poll_interval_s = poll_interval_s
        self._checkpoint_every_s = checkpoint_every_s
        self._max_wait_s = max_wait_s
        self._clock = clock
        self._sleep = sleep

    def _invoke(self, stage: PlannedStage, context: StageContext) -> StageOutcome:
        ledger = AttemptLedger.read(context.attempts_path)
        adapter = self._placement.adapter_for(ledger.preemptions)
        keys = StageKeys.of(context)
        tier = context.gpu_tier or "cpu"
        context.log(
            f"cloud: stage {context.stage_id!r} attempt {context.attempt} on "
            f"{adapter.name!r} ({tier}); {ledger.preemptions} preemption(s) so far"
        )
        self._stage_in(stage, context, keys)
        request = StageRequest(
            recipe=context.recipe,
            run_id=context.run_id,
            stage_id=context.stage_id,
            impl=context.impl,
            attempt=context.attempt,
            tier=tier,
            preemptible=bool(stage.gpu and stage.gpu.preemptible),
            params=dict(context.params),
            inputs={name: keys.input(path) for name, path in stage.inputs.items()},
            produces=stage.impl.produces,
            checkpoint_key=keys.checkpoint,
            outputs_key=keys.outputs,
            checkpoint_every_s=self._checkpoint_every_s,
        )
        handle = adapter.submit(request)
        poll = self._watch(adapter, handle, context)
        # Whatever the remote last synced comes home before anything else is decided, so
        # the next attempt resumes from it. Disable this and a preempted stage restarts.
        self._restore_checkpoint(context)
        entry = self._record(adapter, context, tier, poll, ledger)
        if poll.state == "preempted":
            raise PreemptedError(
                context.recipe, context.stage_id, context.attempt, adapter.name, entry.billed_s
            )
        if poll.state != "succeeded":
            raise RemoteStageError(
                context.recipe, context.stage_id, context.impl, adapter.name, poll.detail
            )
        moved = self._transfer.get(keys.outputs, context.out_dir)
        context.log(f"cloud: {moved} byte(s) of output(s) came back from {adapter.name!r}")
        return StageOutcome(
            metrics=self._metrics(entry, ledger.append(entry), poll),
            summary=poll.summary,
        )

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
            if self._transfer.exists(key):
                continue
            sent = self._transfer.put(key, context.input(name))
            context.log(f"cloud: sent input {name!r} ({sent} bytes) to {key}")
        if context.has_checkpoint:
            sent = self._transfer.put(keys.checkpoint, context.checkpoint_dir)
            context.log(f"cloud: resuming -- sent {sent} byte(s) of checkpoint to the provider")
        else:
            self._transfer.delete(keys.checkpoint)

    def _restore_checkpoint(self, context: StageContext) -> int:
        """Bring back whatever the remote last synced.

        This one call is the whole resume story. Remove it and `prepare_stage` keeping
        `checkpoint/` buys nothing, because the state the remote built lives on the
        remote and arrives only through here -- the next attempt then starts from zero
        and the stage never finishes. `tests/test_cloud_preemption.py` holds it to that.
        """
        return self._transfer.get(context.checkpoint_key, context.checkpoint_dir)

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
                if self._clock() - started >= self._max_wait_s:
                    adapter.cancel(handle)
                    return replace(
                        poll,
                        state="failed",
                        detail=(
                            f"the stage was still {poll.state} after {self._max_wait_s:.0f}s "
                            f"on {adapter.name!r} and was cancelled"
                        ),
                    )
                self._sleep(self._poll_interval_s)
        except BaseException:
            adapter.cancel(handle)
            raise

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
    ) -> Attempt:
        """Price this attempt and write it down, whatever state it ended in."""
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
        )
        ledger.append(entry).write(context.attempts_path)
        billed = f"cloud: {poll.state}; billed {entry.billed_s:.1f}s on {adapter.name!r} ({tier})"
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
    def _metrics(entry: Attempt, ledger: AttemptLedger, poll: Poll) -> dict[str, MetricValue]:
        """This attempt's facts and the stage's running totals, on the successful step.

        The totals are the point: `billedS` alone would say the successful attempt took
        eleven minutes and say nothing about the two preempted attempts before it.
        """
        metrics: dict[str, MetricValue] = dict(poll.metrics or {})
        metrics["provider"] = entry.provider
        metrics["tier"] = entry.tier
        metrics["billedS"] = round(entry.billed_s, 3)
        metrics["stageBilledS"] = round(ledger.billed_s, 3)
        metrics["preemptions"] = ledger.preemptions
        if entry.usd is not None:
            metrics["costUsd"] = round(entry.usd, 6)
            metrics["rateSource"] = entry.rate_source
        stage_usd = ledger.usd
        if stage_usd is not None:
            metrics["stageCostUsd"] = round(stage_usd, 6)
        if ledger.unpriced_s > 0:
            metrics["unpricedS"] = round(ledger.unpriced_s, 3)
        return metrics
