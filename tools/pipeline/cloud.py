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

**A call outlives the process that made it.** A remote call is somebody else's machine,
and it does not stop because the process polling it did. So every call is written down
the moment it is submitted -- `stages/<id>/calls.json`, the `CallBook` -- and taken off
again when it has ended and been paid for. What happens to it when this process is
stopped depends on why (`errors.StopRequested`, raised by the worker's signals):

* **cancelled**, or the worker lost its lease: the call is cancelled (Modal: its
  containers terminated), what it billed so far goes into the ledger, and it is struck
  from the book;
* **detached** -- the worker is shutting down for a deploy: the call is left running and
  the book says so. The next process to run the stage finds it there and *re-attaches*
  (`Reattachable`: Modal's `FunctionCall.from_id`) instead of staging the stage again
  and paying for a second call;
* **killed** outright: nothing runs, the book is still there, and the next process
  re-attaches the same way.

A call that cannot be re-attached is cancelled, and a call recorded for a stage the run
is not about to resume is cancelled before anything else runs (`CloudRunner.reap`).
Each attempt also uses keys of its own (`runners.per_attempt`), so a call that escaped
all of that -- its record lost with the disk -- cannot write over the next attempt's
checkpoint or outputs.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
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
from errors import (
    CostCapError,
    DetachRequested,
    NoRunnerError,
    PreemptedError,
    RemoteStageError,
    RemoteTimeoutError,
)
from phases import flat, parse_flat
from plan import PlannedStage
from providers import Rate
from runners import BaseRunner, per_attempt
from workdir import Workdir

__all__ = [
    "CALL_BOOK",
    "Attempt",
    "AttemptLedger",
    "CallBook",
    "CallRecord",
    "CloudRunner",
    "Placement",
    "Poll",
    "ProviderAdapter",
    "Reattachable",
    "RemoteHandle",
    "RemoteState",
    "RunCost",
    "StageKeys",
    "StageRequest",
    "Transfer",
    "billed_from",
    "call_phases",
    "container_billing",
    "run_cost",
]

#: Where a submitted stage is. `preempted` is deliberately not a kind of `failed`.
RemoteState = Literal["pending", "running", "succeeded", "failed", "preempted"]

#: The states a poll loop stops on.
TERMINAL: tuple[RemoteState, ...] = ("succeeded", "failed", "preempted")

#: How a call's billed seconds were arrived at (`Poll.billing`, `Attempt.billing`):
#: from the container's own clock -- its start (or, warm, the call's entry) to the call's
#: end, which is what a per-second provider bills -- or the runner's wall time since the
#: submit, a proxy that also counts the wait for a GPU and the runner noticing the end.
BILLING_CONTAINER = "container"
BILLING_PROXY = "wall-proxy"


# --- what crosses the seam ---------------------------------------------------------


#: `runs/<run>/<stage>/checkpoint`, with or without an attempt's suffix (`per_attempt`).
_CHECKPOINT_TAIL = re.compile(r"/checkpoint(?:-a\d+)?$")


@dataclass(frozen=True)
class StageKeys:
    """The object-storage keys one attempt at one stage uses, derived from `checkpoint_key`.

    `StageContext.checkpoint_key` is the workdir contract's `runs/<run>/<stage>/checkpoint`
    (A6 put it there for exactly this step), so everything else this runner needs is a
    sibling of it and no second naming scheme is invented.

    Every key a remote call *writes* is per attempt (`runners.per_attempt`): the context's
    checkpoint key already is, and the outputs and each piece's keys follow it. A call
    nobody stopped -- one whose record was lost with the disk -- then writes only over its
    own attempt's keys, never the next attempt's checkpoint or outputs. Attempt 1 keeps
    the keys it always had. The inputs are not per attempt: a remote only reads them, and
    they are checked by checksum before every use (`CloudRunner._stage_in`).
    """

    root: str
    stage: str
    checkpoint: str
    attempt: int = 1

    @staticmethod
    def of(context: StageContext) -> StageKeys:
        stage = _CHECKPOINT_TAIL.sub("", context.checkpoint_key)
        return StageKeys(
            root=stage.removesuffix(f"/{context.stage_id}"),
            stage=stage,
            checkpoint=context.checkpoint_key,
            attempt=context.attempt,
        )

    @property
    def outputs(self) -> str:
        """Where the remote puts `out/`. Under `transfer/` so it cannot collide with the
        per-artifact keys the worker uploads finished artifacts to."""
        return f"{self.stage}/transfer/{per_attempt('out', self.attempt)}"

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
        return f"{self.part_root(part)}/{per_attempt('checkpoint', self.attempt)}"

    def part_outputs(self, part: str) -> str:
        return f"{self.part_root(part)}/{per_attempt('out', self.attempt)}"

    def part_root(self, part: str) -> str:
        """Everything of one piece's, every attempt's: what the join deletes after it."""
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
    #: `BILLING_CONTAINER` or `BILLING_PROXY`; empty when the adapter does not say.
    billing: str = ""
    #: Seconds between the submit and the start of billing -- the wait for a GPU -- when
    #: the adapter could tell them apart (`billing == BILLING_CONTAINER`). Not a cost.
    queue_s: float | None = None
    #: The tier the call actually ran on, when the provider may pick one of several
    #: (`modal_adapter.GPU_FALLBACKS`); empty means the tier that was asked for.
    tier: str = ""
    #: True when a `failed` call ran out of time -- the deployed function's own limit
    #: (Modal's `FunctionTimeoutError`), the runner's `max_wait_s` or its deadline --
    #: rather than failing at something. `CloudRunner` raises `RemoteTimeoutError` for it,
    #: which the worker does not retry: the next attempt would outrun the same limit.
    timed_out: bool = False


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
class Reattachable(Protocol):
    """An adapter whose calls outlive the process that submitted them.

    Optional, and deliberately not part of `ProviderAdapter`: a provider whose call is a
    machine somewhere else (Modal) can hand a new process the same call from its id, and
    a deploy then costs the stage nothing; `SubprocessAdapter`'s call is a child of the
    process that is going away, and there is nothing to re-attach to. `CloudRunner`
    leaves a call running across a deploy only for an adapter that has this, and cancels
    it for one that does not.
    """

    def reattach(
        self, handle: RemoteHandle, request: StageRequest | None, submitted_at: float
    ) -> None:
        """Make `handle` pollable, readable and cancellable in this process.

        `request` is what the call was submitted with; None when only a cancel is wanted
        and the request was not kept (a call known only from the database, its workdir
        gone). `submitted_at` is the wall clock at the original submit, so a billed-time
        proxy keeps counting from there rather than from now.
        """
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
    #: How `billed_s` was arrived at (`BILLING_CONTAINER` or `BILLING_PROXY`), and the
    #: wait for a GPU before it, which is not billed (`Poll.queue_s`).
    billing: str = ""
    queue_s: float | None = None
    #: The provider's id for the call (`RemoteHandle.id`). What makes writing an entry
    #: idempotent: a process killed after it wrote one and before it struck the call from
    #: the `CallBook` leaves the next process to re-attach to a call that is already paid
    #: for, and it must not be paid for twice.
    call: str = ""

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
        if self.billing:
            document["billing"] = self.billing
        if self.queue_s is not None:
            document["queueS"] = round(self.queue_s, 3)
        if self.call:
            document["call"] = self.call
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
            billing=str(document.get("billing", "")),
            queue_s=None if document.get("queueS") is None else float(document["queueS"]),
            call=str(document.get("call", "")),
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


# --- the calls that are out there --------------------------------------------------

#: The `CallBook`'s file, in `stages/<id>/` beside `attempts.json`.
CALL_BOOK = "calls.json"

#: The slot of a stage's own call -- the only call, or a fan-out's head -- in a book; a
#: fan-out's pieces are under their part ids, and its join under `JOIN`.
HEAD = ""
JOIN = "join"


@dataclass(frozen=True)
class CallRecord:
    """One remote call that has been submitted and not yet ended and been paid for."""

    handle: RemoteHandle
    #: The wall clock at the submit (epoch seconds).
    submitted_at: float
    #: The stage attempt that submitted it, which decides the keys it writes to.
    attempt: int
    #: What it was submitted with: its checkpoint and output keys, which an attempt that
    #: re-attaches to it must read back from rather than its own. None for a call known
    #: only by id (`CallBook.orphaned`), which can be cancelled and nothing else.
    request: StageRequest | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.handle.id,
            "provider": self.handle.provider,
            "tier": self.handle.tier,
            "submittedAt": round(self.submitted_at, 3),
            "attempt": self.attempt,
            "request": None if self.request is None else self.request.to_dict(),
        }

    @staticmethod
    def from_dict(document: Mapping[str, Any]) -> CallRecord:
        request = document.get("request")
        return CallRecord(
            handle=RemoteHandle(
                id=str(document["id"]),
                provider=str(document["provider"]),
                tier=str(document.get("tier", "")),
            ),
            submitted_at=float(document.get("submittedAt", 0.0)),
            attempt=int(document.get("attempt", 1)),
            request=StageRequest.from_dict(request) if isinstance(request, Mapping) else None,
        )


@dataclass
class CallBook:
    """`stages/<id>/calls.json`: every remote call of one stage that is out there now.

    Written the moment a call is submitted and struck off once it has ended, been brought
    home and been entered in the ledger, so whatever is in it is a call a process may
    still be paying for. That is the whole of what makes a stopped worker's calls
    findable: the process that submitted them may be gone, and the provider's id is the
    only handle there is. Also the fan-out a stage was in the middle of (`fan_out`, the
    head's answer, and `parts_done`), so a process that picks it up resumes the pieces
    rather than asking the head again.

    `detached` says the last process left on purpose (a deploy) and expected the next
    one to re-attach; `orphaned` says the calls are known only from the database, the
    workdir that had this book having been lost, and are to be cancelled, not resumed.
    The worker's supervisor reads the book too, to copy the calls onto the step's row.
    """

    path: Path
    calls: dict[str, CallRecord] = field(default_factory=dict)
    detached: bool = False
    orphaned: bool = False
    #: The head's `contracts.FANOUT_METRIC`, as it answered it, while pieces are running.
    fan_out: MetricValue | None = None
    parts_done: list[str] = field(default_factory=list)

    @staticmethod
    def beside(attempts_path: Path) -> Path:
        """The book of the stage whose ledger is `attempts_path`."""
        return attempts_path.with_name(CALL_BOOK)

    @staticmethod
    def read(path: Path) -> CallBook:
        """The book at `path`, or an empty one. A file that will not parse is a process
        that died mid-write, which `save`'s rename makes impossible; it reads as empty
        rather than stopping the stage, at the cost of forgetting a call."""
        if not path.is_file():
            return CallBook(path)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            calls = {
                str(slot): CallRecord.from_dict(entry)
                for slot, entry in dict(document.get("calls") or {}).items()
            }
            fan_out = document.get("fanOut")
            return CallBook(
                path=path,
                calls=calls,
                detached=bool(document.get("detached", False)),
                orphaned=bool(document.get("orphaned", False)),
                fan_out=fan_out if isinstance(fan_out, str | int | float | bool) else None,
                parts_done=[str(part) for part in document.get("partsDone") or ()],
            )
        except (OSError, ValueError, KeyError, TypeError):
            return CallBook(path)

    @property
    def empty(self) -> bool:
        return not self.calls and self.fan_out is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": {slot: record.to_dict() for slot, record in sorted(self.calls.items())},
            "detached": self.detached,
            "orphaned": self.orphaned,
            "fanOut": self.fan_out,
            "partsDone": list(self.parts_done),
        }

    def save(self) -> None:
        """Written beside and renamed into place, so a process killed mid-write leaves
        the previous book rather than half of one. An empty book is no file at all."""
        if self.empty:
            self.path.unlink(missing_ok=True)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(f"{self.path.name}.{os.getpid()}.partial")
        partial.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n", "utf-8")
        partial.replace(self.path)

    def put(self, slot: str, record: CallRecord) -> None:
        self.calls[slot] = record
        self.save()

    def drop(self, slot: str) -> None:
        if self.calls.pop(slot, None) is not None:
            self.save()

    def end_fan_out(self) -> None:
        """The fan-out is over, one way or the other: forget its pieces' progress."""
        self.fan_out = None
        self.parts_done = []
        self.save()


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
        #: Wrapped around each submit (and re-attach) together with writing the call into
        #: the `CallBook`. The worker's recipe process passes a context that holds a stop
        #: signal back until it exits (`app/worker/child.py`), so a cancel or a deploy can
        #: never land between a call being created and its id being written down -- the
        #: one moment at which a call would be out there with nobody able to find it.
        shield: Callable[[], contextlib.AbstractContextManager[object]] = contextlib.nullcontext,
        #: The most a run may be billed, across every stage and attempt, before this
        #: runner refuses to submit another call and cancels a running one whose cost
        #: would go over it (`CostCapError`). None for no cap. Priced only where a rate
        #: is known: an unpriced call is not stopped by a figure nobody has.
        cost_cap_usd: float | None = None,
        #: A call whose log reports training progress (tqdm's `done/total [elapsed<left`)
        #: is cancelled as timed out once it has run `deadline_factor` times the time its
        #: newest progress line projected for the planned steps, plus `deadline_slack_s`
        #: for what the trainer does after its last step. A trainer that is slow and keeps
        #: printing moves its own projection and is never stopped by this; one that has
        #: stopped printing -- hung -- is stopped long before the provider's own limit.
        #: 0 turns it off.
        deadline_factor: float = 2.0,
        deadline_slack_s: float = 30 * 60.0,
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
        self._shield = shield
        self._cost_cap_usd = cost_cap_usd if cost_cap_usd and cost_cap_usd > 0 else None
        self._deadline_factor = max(0.0, deadline_factor)
        self._deadline_slack_s = max(0.0, deadline_slack_s)
        #: The ledger entries of the attempt being run: every call it made or re-attached
        #: to, in order. Reset by `_invoke`; `_record` adds to it.
        self._made: list[Attempt] = []

    def _invoke(self, stage: PlannedStage, context: StageContext) -> StageOutcome:
        self._made = []
        ledger = AttemptLedger.read(context.attempts_path)
        adapter = self._placement.adapter_for(ledger.preemptions)
        keys = StageKeys.of(context)
        tier = context.gpu_tier or "cpu"
        book = CallBook.read(CallBook.beside(context.attempts_path))
        context.log(
            f"cloud: stage {context.stage_id!r} attempt {context.attempt} on "
            f"{adapter.name!r} ({tier}); {ledger.preemptions} preemption(s) so far"
        )
        try:
            return self._run(stage, context, keys, tier, book)
        except DetachRequested:
            # The calls are left in the book for the next process; so is the fan-out.
            raise
        except BaseException:
            # Every other way out of an attempt ends what it was in the middle of. Its
            # calls have been cancelled and struck off already (`_watch`); one whose
            # cancel failed stays, for the next attempt to cancel or re-attach to.
            book.end_fan_out()
            raise

    def _run(
        self,
        stage: PlannedStage,
        context: StageContext,
        keys: StageKeys,
        tier: str,
        book: CallBook,
    ) -> StageOutcome:
        """The attempt proper: stage in, the head call, any fan-out and its join, the
        outputs home -- picking up wherever a previous process left `book`."""
        staged_s = 0.0
        if book.empty:
            # Before the inputs go up, not only before the submit: a run already at its
            # cap should not spend minutes uploading for a call it will not make. A call
            # picked up from the book is not refused here -- it is already running and
            # billing, and `_watch` holds it to the cap with its running cost included.
            self._within_cap(context, "the stage")
            staging = time.monotonic()
            self._stage_in(stage, context, keys)
            staged_s = time.monotonic() - staging
            context.log(f"cloud: inputs and checkpoint staged in {staged_s:.1f} s")
        else:
            # Nothing is staged again: the calls already have their inputs, and pushing
            # `checkpoint/` would write over the one a running call is syncing to.
            book.detached = False
            book.save()
            waiting = ", ".join(
                f"{slot or 'the stage'} ({record.handle.id})"
                for slot, record in sorted(book.calls.items())
            )
            context.log(
                f"cloud: picking up where the last process left this stage: "
                f"{waiting or 'no call in flight'}"
                + ("; its fan-out is part done" if book.fan_out is not None else "")
            )
        fanned: dict[str, MetricValue] = {}
        spec: FanOut | None = None
        called: _Called | None = None
        if book.fan_out is not None:
            spec = FanOut.parse(book.fan_out)
        else:
            params = dict(context.params)
            if self._fan_out:
                params[FANOUT_PARAM] = {"role": "head"}
            called = self._call(stage, context, keys, tier, params, book)
            raw = (called.poll.metrics or {}).get(FANOUT_METRIC)
            spec = FanOut.parse(raw) if self._fan_out else None
            if spec is not None and raw is not None:
                fanned["headPhases"] = flat(call_phases(called.submitted_at, called.poll))
                head_host = (called.poll.metrics or {}).get("remoteHost")
                if isinstance(head_host, str) and head_host:
                    fanned["headHost"] = head_host
                # Written before the head is struck off: a process stopped from here on
                # resumes the pieces instead of asking the head again.
                book.fan_out = raw
                book.calls.pop(HEAD, None)
                book.save()
        if spec is not None:
            if JOIN not in book.calls:
                fanned.update(self._fan_out_parts(stage, context, keys, tier, spec, book))
                # Every piece's result is in `checkpoint/` now; the join starts from it.
                sending = time.monotonic()
                self._transfer.delete(keys.checkpoint)
                sent = self._transfer.put(keys.checkpoint, context.checkpoint_dir)
                fanned["joinSendS"] = round(time.monotonic() - sending, 2)
                context.log(f"cloud: sent {sent} byte(s) of checkpoint for the join")
            joined = {**context.params, FANOUT_PARAM: {"role": "join"}}
            called = self._call(stage, context, keys, tier, joined, book, part=JOIN)
            for part in spec.parts:
                self._transfer.delete(keys.part_root(part.id))
        if called is None:  # pragma: no cover - a fan-out always ends with its join
            raise RemoteStageError(
                context.recipe, context.stage_id, context.impl, "cloud", "no call was made"
            )
        fetching = time.monotonic()
        # From the call's own keys, which are an earlier attempt's when it was re-attached
        # to after a worker died mid-attempt.
        moved = self._transfer.get(called.request.outputs_key, context.out_dir)
        back_s = time.monotonic() - fetching
        context.log(
            f"cloud: {moved} byte(s) of output(s) came back from {called.adapter.name!r} "
            f"in {back_s:.1f} s"
        )
        # Home, verified by `BaseRunner` next, and paid for: nothing is out there now.
        book.calls.clear()
        book.end_fan_out()
        ledger = AttemptLedger.read(context.attempts_path)
        metrics = self._metrics(self._made, ledger, called.poll)
        metrics.update(fanned)
        # The last call's billed seconds, phase by phase (`call_phases`): the one call, or
        # the join. A fan-out's head and parts are `headPhases` and `fanOutPhases`.
        metrics["remotePhases"] = flat(call_phases(called.submitted_at, called.poll))
        # The worker's side of the transfer, beside the remote's own (`remote*S`): with
        # `billedS` they account for a remote stage's wall time end to end.
        metrics["stageInS"] = round(staged_s, 2)
        metrics["outputsBackS"] = round(back_s, 2)
        # Which key `out/` came home from: the call's, which is an earlier attempt's for a
        # call re-attached to across a dead worker. The worker copies the artifacts into
        # place from exactly this key (`app/worker/outputs.py`); guessing it from the
        # attempt number was wrong for an adopted call, and after a Retry reset the
        # attempts could name another run's leftover outputs of the same size.
        metrics["outputsKey"] = called.request.outputs_key
        return StageOutcome(metrics=metrics, summary=called.poll.summary)

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
        book: CallBook,
        *,
        part: str = HEAD,
    ) -> _Called:
        """One call on the stage's own checkpoint key: the head, or a fan-out's join --
        re-attached to if `book` says a previous process left it running, submitted and
        written into `book` otherwise. Raises unless it succeeded, having brought the
        checkpoint home and recorded what it billed. A call that succeeded stays in the
        book until its outputs are home (`_run`)."""
        ledger = AttemptLedger.read(context.attempts_path)
        adopted = self._adopt(context, book, part)
        if adopted is None:
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
            self._within_cap(context, f"call {part or 'for the stage'}")
            handle, submitted = self._submit(adapter, request, context, tier, book, part)
            context.log(f"cloud: submitted to {adapter.name!r} as call {handle.id}")
            started = self._clock()
        else:
            adapter, handle, request, submitted = adopted
            started = self._clock() - max(0.0, self._wall() - submitted)
        poll = self._watch(adapter, handle, context, book, part, tier, started)
        # Whatever the remote last synced comes home before anything else is decided, so
        # the next attempt resumes from it. Disable this and a preempted stage restarts.
        self._restore_checkpoint(context, request.checkpoint_key)
        entry = self._record(adapter, context, tier, poll, ledger, part=part, call=handle.id)
        if poll.state != "succeeded":
            book.drop(part)
        if poll.state == "preempted":
            raise PreemptedError(
                context.recipe, context.stage_id, context.attempt, adapter.name, entry.billed_s
            )
        if poll.state != "succeeded":
            failure = RemoteTimeoutError if poll.timed_out else RemoteStageError
            raise failure(context.recipe, context.stage_id, context.impl, adapter.name, poll.detail)
        return _Called(adapter=adapter, poll=poll, submitted_at=submitted, request=request)

    def _submit(
        self,
        adapter: ProviderAdapter,
        request: StageRequest,
        context: StageContext,
        tier: str,
        book: CallBook,
        slot: str,
    ) -> tuple[RemoteHandle, float]:
        """Submit and write the call into the book, with any stop held back until both are
        done (`shield`) -- and if one was, deal with the call before letting it through, as
        `_watch` would have: it is out there now, and written down."""
        handle: RemoteHandle | None = None
        try:
            with self._shield():
                submitted = self._wall()
                handle = adapter.submit(request)
                book.put(slot, CallRecord(handle, submitted, context.attempt, request))
        except BaseException as error:
            if handle is not None and slot in book.calls:
                self._stopped(adapter, handle, context, tier, None, book, slot, error)
            raise
        return handle, submitted

    # --- calls a previous process left running ---------------------------------------

    def _adapter_named(self, name: str) -> ProviderAdapter | None:
        return next((adapter for adapter in self._placement.adapters if adapter.name == name), None)

    def _adopt(
        self, context: StageContext, book: CallBook, slot: str
    ) -> tuple[ProviderAdapter, RemoteHandle, StageRequest, float] | None:
        """The call `book` has in `slot`, made pollable here; None when there is none.

        A call that cannot be re-attached -- its provider is not configured here any more,
        or its adapter has no `reattach` -- is cancelled as far as that is possible and a
        new one is started, *unless* it was this same attempt's: a new call would then
        share its keys with one that may still be running, so the attempt fails instead
        and the next one starts clean on keys of its own.
        """
        record = book.calls.get(slot)
        if record is None:
            return None
        adapter = self._adapter_named(record.handle.provider)
        what = f"call {record.handle.id} on {record.handle.provider!r}"
        problem = ""
        if adapter is None:
            problem = "that provider is not configured here"
        elif not isinstance(adapter, Reattachable) or record.request is None:
            problem = "that provider's calls cannot be re-attached to"
        else:
            # No `shield`: the call is written down already, so a stop landing here leaves
            # nothing unaccounted for.
            try:
                adapter.reattach(record.handle, record.request, record.submitted_at)
            except Exception as error:
                problem = f"re-attaching failed: {error!r}"
        if not problem and adapter is not None and record.request is not None:
            age = max(0.0, self._wall() - record.submitted_at)
            context.log(
                f"cloud: re-attached to {what}, submitted {age:.0f} s ago by attempt "
                f"{record.attempt}; its log is read again from the start"
            )
            return adapter, record.handle, record.request, record.submitted_at
        if adapter is not None:
            self._cancel_quietly(adapter, record, context)
        book.drop(slot)
        if record.attempt == context.attempt:
            raise RemoteStageError(
                context.recipe,
                context.stage_id,
                context.impl,
                record.handle.provider,
                f"{what}, left by the last process, could not be picked up ({problem}), and "
                f"a new call of the same attempt would share its keys; the next attempt "
                f"starts one on its own",
            )
        context.log(f"cloud: {what} could not be picked up ({problem}); starting a new call")
        return None

    def _cancel_quietly(
        self, adapter: ProviderAdapter, record: CallRecord, context: StageContext | None
    ) -> bool:
        """Cancel a call known only from a record, as well as can be done. True if done."""
        try:
            if isinstance(adapter, Reattachable):
                adapter.reattach(record.handle, record.request, record.submitted_at)
            adapter.cancel(record.handle)
        except Exception as error:
            if context is not None:
                context.log(f"cloud: could not cancel call {record.handle.id}: {error!r}")
            return False
        return True

    def reap(self, workdir: Workdir, *, keep: str | None) -> list[str]:
        """Cancel every recorded call no stage of this run is about to pick up.

        Called by the worker's recipe process before it runs anything. `keep` is the first
        stage the run will execute -- the only one whose calls can be resumed, because
        every stage before it is skipped as done and every stage after it will be run
        again on new inputs, which makes any call of theirs work on stale ones. A book
        marked `orphaned` (its calls known only from the database) is cancelled even for
        `keep`: the workdir it belonged to is gone, and with it the run's inputs. Returns
        the ids cancelled, and says so in each stage's log.
        """
        cancelled: list[str] = []
        if not workdir.stages_dir.is_dir():
            return cancelled
        for path in sorted(workdir.stages_dir.glob(f"*/{CALL_BOOK}")):
            book = CallBook.read(path)
            if path.parent.name == keep and not book.orphaned:
                continue
            for record in list(book.calls.values()):
                adapter = self._adapter_named(record.handle.provider)
                done = adapter is not None and self._cancel_quietly(adapter, record, None)
                if done:
                    cancelled.append(record.handle.id)
                _append_log(
                    path.parent / "log.txt",
                    f"cloud: {'cancelled' if done else 'could not cancel'} call "
                    f"{record.handle.id} on {record.handle.provider!r}, left by an earlier "
                    f"process for a stage this run is not resuming",
                )
            book.calls.clear()
            book.end_fan_out()
        return cancelled

    # --- fanning out -----------------------------------------------------------------

    def _fan_out_parts(
        self,
        stage: PlannedStage,
        context: StageContext,
        keys: StageKeys,
        tier: str,
        spec: FanOut,
        book: CallBook,
    ) -> dict[str, MetricValue]:
        """Run every piece, at most `parallel` at once; raise if any is lost for good.

        One thread polls them all: the adapters' `poll` does not block (Modal's is a
        zero-timeout `get`), so N pieces cost N polls a round, not N threads. A piece that
        ends in anything but success is resubmitted alone, up to `part_attempts` calls;
        a piece that runs out of calls does not stop the others, because each finished
        piece is kept in `checkpoint/` and the next attempt starts only what is missing.

        Picked up from `book` when a previous process left the fan-out part done: the
        pieces it finished (`parts_done`, already in `checkpoint/`) are not run again, and
        the ones it left running are re-attached to rather than submitted twice.
        """
        parallel = min(spec.parallel, self._max_parallel)
        todo = [part for part in spec.parts if part.id not in book.parts_done]
        context.log(
            f"cloud: fanning out {len(todo)} part(s) of stage {context.stage_id!r}, "
            f"{parallel} at a time ({', '.join(p.id for p in todo)})"
            + (f"; {', '.join(book.parts_done)} already done" if book.parts_done else "")
        )
        queue: deque[FanOutPart] = deque(todo)
        calls: dict[str, int] = {part.id: 0 for part in spec.parts}
        billed: dict[str, float] = {part.id: 0.0 for part in spec.parts}
        running: dict[str, _PartRun] = {}
        lost: dict[str, tuple[ProviderAdapter, Poll]] = {}
        #: Each finished piece's billed seconds, phase by phase (`call_phases`), the tier
        #: it ran on, and its machine (`remoteHost`).
        timed: dict[str, str] = {}
        tiers: dict[str, str] = {}
        hosts: dict[str, str] = {}
        collect_s = 0.0
        tail = _FanOutLog(context)
        started = self._clock()
        peak = 0
        last: dict[str, Poll] = {}
        try:
            while queue or running:
                while queue and len(running) < parallel:
                    part = queue.popleft()
                    calls[part.id] += 1
                    running[part.id] = self._adopt_part(context, book, part) or (
                        self._submit_part(
                            stage, context, keys, tier, spec, part, calls[part.id], book
                        )
                    )
                peak = max(peak, len(running))
                for part_id, run in list(running.items()):
                    poll = self._poll_part(run, context, tail)
                    last[part_id] = poll
                    if poll.state not in TERMINAL:
                        continue
                    del running[part_id]
                    tail.finished(part_id)
                    if poll.state == "succeeded":
                        collecting = time.monotonic()
                        poll = self._collect_part(context, keys, run, poll)
                        collected = time.monotonic() - collecting
                        collect_s += collected
                        timed[part_id] = flat(
                            call_phases(run.submitted_at, poll, collect_s=collected)
                        )
                        tiers[part_id] = poll.tier or tier
                        host = (poll.metrics or {}).get("remoteHost")
                        if isinstance(host, str) and host:
                            hosts[part_id] = host
                    entry = self._record(
                        run.adapter,
                        context,
                        tier,
                        poll,
                        AttemptLedger.read(context.attempts_path),
                        part=part_id,
                        call=run.handle.id,
                    )
                    billed[part_id] += entry.billed_s
                    if poll.state == "succeeded":
                        # In `checkpoint/` and paid for: a process that picks this
                        # fan-out up from here does not run it again.
                        book.parts_done.append(part_id)
                        book.calls.pop(part_id, None)
                        book.save()
                        continue
                    book.drop(part_id)
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
                if running:
                    self._parts_within_cap(context, running, last, tier, book)
                if queue or running:
                    self._sleep(self._poll_interval_s)
        except CostCapError:
            # Over the cap: `_parts_within_cap` has cancelled what was running, or a new
            # piece was refused before its submit -- and the ones already out are stopped.
            self._cancel_parts(context, tier, book, running, last, "the run's cost cap")
            raise
        except DetachRequested:
            if all(isinstance(run.adapter, Reattachable) for run in running.values()):
                book.detached = True
                book.save()
                context.log(
                    f"cloud: the worker is shutting down; part(s) "
                    f"{', '.join(running) or 'none'} are left running for the next one"
                )
                raise
            self._cancel_parts(context, tier, book, running, last, "the worker is shutting down")
            raise
        except BaseException as error:
            # Whatever stops the loop -- a cancelled job, a lost lease -- stops paying for
            # every piece still out there, not just the one being polled.
            self._cancel_parts(context, tier, book, running, last, _why(error))
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
            # The tier each piece ran on (a fallback list may start one on another GPU),
            # and the machine it had (`host.HostWatch`): a slow piece says why.
            "fanOutTiers": ",".join(f"{k}:{v}" for k, v in tiers.items()),
            "fanOutHosts": "; ".join(f"{k} {v}" for k, v in hosts.items()),
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
        book: CallBook,
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
        self._within_cap(context, f"part {part.id}")
        handle, submitted = self._submit(adapter, request, context, tier, book, part.id)
        context.log(
            f"cloud: part {part.id} submitted to {adapter.name!r} (call {call}) as {handle.id}"
        )
        return _PartRun(
            part=part,
            adapter=adapter,
            handle=handle,
            started=self._clock(),
            submitted_at=submitted,
            request=request,
        )

    def _adopt_part(
        self, context: StageContext, book: CallBook, part: FanOutPart
    ) -> _PartRun | None:
        """The piece a previous process left running, re-attached to; None when there is
        none, or it could not be (and a new call is wanted)."""
        adopted = self._adopt(context, book, part.id)
        if adopted is None:
            return None
        adapter, handle, request, submitted = adopted
        return _PartRun(
            part=part,
            adapter=adapter,
            handle=handle,
            started=self._clock() - max(0.0, self._wall() - submitted),
            submitted_at=submitted,
            request=request,
        )

    def _parts_within_cap(
        self,
        context: StageContext,
        running: Mapping[str, _PartRun],
        last: Mapping[str, Poll],
        tier: str,
        book: CallBook,
    ) -> None:
        """Every piece still running, cancelled, if together they would take the run past
        the cap: the pieces of one stage are billed at once, so they are priced at once."""
        if self._cost_cap_usd is None:
            return
        spent = self._spent(context)
        running_usd = 0.0
        for part_id, run in running.items():
            poll = last.get(part_id)
            rate = run.adapter.rate((poll.tier if poll else "") or tier)
            if rate is not None and poll is not None:
                running_usd += rate.usd_for(poll.billed_s)
        if spent + running_usd <= self._cost_cap_usd:
            return
        why = (
            f"the {len(running)} part(s) still running had cost ${running_usd:.2f}, which "
            f"would take the run past the cap, so they were cancelled"
        )
        self._cancel_parts(context, tier, book, running, last, "the run's cost cap")
        raise CostCapError(
            context.recipe, context.stage_id, self._spent(context), self._cost_cap_usd, why
        )

    def _cancel_parts(
        self,
        context: StageContext,
        tier: str,
        book: CallBook,
        running: Mapping[str, _PartRun],
        last: Mapping[str, Poll],
        why: str,
    ) -> None:
        """`_cancel` every piece in `running` that is still in the book."""
        for part_id, run in running.items():
            if part_id in book.calls:
                self._cancel(
                    run.adapter, run.handle, context, tier, last.get(part_id), book, part_id, why
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
        self, context: StageContext, keys: StageKeys, run: _PartRun, poll: Poll
    ) -> Poll:
        """A finished piece's result, from its key into the stage's `checkpoint/`.

        Only the paths it declared (`FanOutPart.collect`), each fetched on its own and
        replaced whole. Fetched member by member rather than the piece's whole key, which
        also holds the shared members it was sent (a block's prior) and its live
        snapshots: bytes that went out with it and are already here. A result that did not
        come back makes the call a failure: the piece claimed to finish and there is
        nothing for the join to use, so it runs again.

        From the key the call was submitted with, which is an earlier attempt's when the
        call was re-attached to after a worker died mid-attempt.
        """
        part = run.part
        source = run.request.checkpoint_key if run.request else keys.part_checkpoint(part.id)
        incoming = context.work_dir / "fanout" / f"{part.id}.incoming"
        if incoming.exists():
            shutil.rmtree(incoming)
        incoming.mkdir(parents=True)
        for member in part.collect:
            self._transfer.get(f"{source}/{member}", incoming / member)
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
        # Cleared either way, not only when there is nothing to send: the worker's bucket
        # transfer adds members and never removes one, and a key can be used twice -- a
        # retry from the panel starts its stages at attempt 1 again -- so what an earlier
        # use left there would ride along with this attempt's checkpoint.
        self._transfer.delete(keys.checkpoint)
        if context.has_checkpoint:
            sent = self._transfer.put(keys.checkpoint, context.checkpoint_dir)
            context.log(
                f"cloud: sent {sent} byte(s) of checkpoint to the provider (an earlier "
                f"attempt's state, or a seed an earlier run left for this one)"
            )

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

    def _restore_checkpoint(self, context: StageContext, key: str | None = None) -> int:
        """Bring back whatever the remote last synced -- to `key`, the call's own, which
        is an earlier attempt's for a call re-attached to across a dead worker.

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
        moved = self._transfer.get(key or context.checkpoint_key, incoming)
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

    def _watch(
        self,
        adapter: ProviderAdapter,
        handle: RemoteHandle,
        context: StageContext,
        book: CallBook,
        slot: str,
        tier: str,
        started: float,
    ) -> Poll:
        """Poll until it ends, tailing the log.

        Anything that stops this cancels first -- a stage nobody is watching any more is
        a stage nobody has stopped paying for -- except a detach, which leaves a call that
        can be re-attached to running and says so in the book (the module docstring has
        the three ways a process stops). A cancelled call is entered in the ledger with
        what it billed until then, so a cancelled run's cost is not a cost of zero.
        """
        cursor = 0
        poll = Poll(state="pending")
        deadline = _Deadline(self._deadline_factor, self._deadline_slack_s)
        try:
            while True:
                poll = adapter.poll(handle)
                cursor = self._tail(adapter, handle, context, cursor, deadline)
                if poll.state in TERMINAL:
                    # Once more: the lines that say why a call failed -- a traceback, CUDA
                    # running out of memory -- are its last, and can reach the provider's
                    # log after its verdict did. The worker reads them to decide on a retry.
                    self._tail(adapter, handle, context, cursor, deadline)
                    return poll
                poll = self._overdue(adapter, handle, poll, self._clock() - started)
                if poll.state in TERMINAL:
                    return poll
                poll = self._past_deadline(adapter, handle, poll, deadline)
                if poll.state in TERMINAL:
                    return poll
                self._call_within_cap(adapter, handle, context, tier, poll, book, slot)
                self._sleep(self._poll_interval_s)
        except BaseException as error:
            self._stopped(adapter, handle, context, tier, poll, book, slot, error)
            raise

    def _stopped(
        self,
        adapter: ProviderAdapter,
        handle: RemoteHandle,
        context: StageContext,
        tier: str,
        last: Poll | None,
        book: CallBook,
        slot: str,
        error: BaseException,
    ) -> None:
        """What becomes of a call when whatever was watching it stops with `error`.

        A detach leaves a call that can be re-attached to running and marks the book; a
        cost-cap stop has already cancelled it; anything else -- a cancel, a lost lease, a
        bug -- cancels it now. The caller re-raises `error`.
        """
        if isinstance(error, CostCapError):
            return  # cancelled and recorded on the way to raising it
        if isinstance(error, DetachRequested) and isinstance(adapter, Reattachable):
            book.detached = True
            book.save()
            context.log(
                f"cloud: the worker is shutting down; call {handle.id} on "
                f"{adapter.name!r} is left running for the next one to re-attach to"
            )
            return
        why = (
            "the worker is shutting down and this provider's calls cannot be re-attached to"
            if isinstance(error, DetachRequested)
            else _why(error)
        )
        self._cancel(adapter, handle, context, tier, last, book, slot, why)

    def _cancel(
        self,
        adapter: ProviderAdapter,
        handle: RemoteHandle,
        context: StageContext,
        tier: str,
        last: Poll | None,
        book: CallBook,
        slot: str,
        why: str,
    ) -> Attempt | None:
        """Stop paying for a call, enter what it billed until now in the ledger, and strike
        it from the book -- unless the cancel itself failed, in which case it stays there
        for the next process to try again. Never raises: it runs on the way out of a
        stage, and an exception here would replace the one that says why."""
        try:
            adapter.cancel(handle)
        except Exception as error:
            context.log(f"cloud: could not cancel call {handle.id} ({why}): {error!r}")
            return None
        context.log(f"cloud: cancelled call {handle.id} on {adapter.name!r}: {why}")
        seen = last or Poll(state="pending")
        ended = Poll(
            state="failed",
            billed_s=seen.billed_s,
            detail=f"cancelled: {why}",
            billing=seen.billing or BILLING_PROXY,
            queue_s=seen.queue_s,
            tier=seen.tier,
        )
        entry: Attempt | None = None
        try:
            ledger = AttemptLedger.read(context.attempts_path)
            entry = self._record(adapter, context, tier, ended, ledger, part=slot, call=handle.id)
        except OSError as error:
            context.log(f"cloud: could not record the cancelled call {handle.id}: {error!r}")
        book.drop(slot)
        return entry

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
                timed_out=True,
                detail=(
                    f"the stage was still {poll.state} after {self._max_wait_s:.0f}s "
                    f"on {adapter.name!r} and was cancelled"
                ),
            )
        return poll

    def _past_deadline(
        self, adapter: ProviderAdapter, handle: RemoteHandle, poll: Poll, deadline: _Deadline
    ) -> Poll:
        """`poll`, or a cancelled, timed-out failure when the call's own progress says it
        has run far past what its planned steps take (`deadline_factor`)."""
        over = deadline.overrun(self._clock())
        if over is None:
            return poll
        adapter.cancel(handle)
        return replace(
            poll,
            state="failed",
            timed_out=True,
            detail=(
                f"cancelled as overdue on {adapter.name!r}: its newest progress line "
                f"projected {deadline.expected_s:.0f} s for the planned steps, and it had "
                f"run {over:.0f} s, more than {self._deadline_factor:g}x that plus "
                f"{self._deadline_slack_s:.0f} s"
            ),
        )

    # --- the cost cap ----------------------------------------------------------------

    def _spent(self, context: StageContext) -> float:
        """What the whole run has been billed so far, by its ledgers: every stage's, every
        attempt's. A call still running is not in it yet."""
        root = context.attempts_path.parent.parent.parent
        return run_cost(Workdir(root)).usd or 0.0

    def _within_cap(self, context: StageContext, what: str) -> None:
        """Refuse to start `what` once the run's spend has reached the cap."""
        if self._cost_cap_usd is None:
            return
        spent = self._spent(context)
        if spent >= self._cost_cap_usd:
            raise CostCapError(
                context.recipe,
                context.stage_id,
                spent,
                self._cost_cap_usd,
                f"not starting {what}",
            )

    def _call_within_cap(
        self,
        adapter: ProviderAdapter,
        handle: RemoteHandle,
        context: StageContext,
        tier: str,
        poll: Poll,
        book: CallBook,
        slot: str,
    ) -> None:
        """Cancel a running call whose cost so far would take the run past the cap."""
        if self._cost_cap_usd is None:
            return
        rate = adapter.rate(poll.tier or tier)
        if rate is None:
            return  # unpriced: nothing to hold it to
        running = rate.usd_for(poll.billed_s)
        spent = self._spent(context)
        if spent + running <= self._cost_cap_usd:
            return
        self._cancel(adapter, handle, context, tier, poll, book, slot, "the run's cost cap")
        raise CostCapError(
            context.recipe,
            context.stage_id,
            self._spent(context),
            self._cost_cap_usd,
            f"call {handle.id} had cost ${running:.2f} while still running, which would "
            f"take the run past the cap, so it was cancelled",
        )

    def _tail(
        self,
        adapter: ProviderAdapter,
        handle: RemoteHandle,
        context: StageContext,
        cursor: int,
        deadline: _Deadline | None = None,
    ) -> int:
        lines = adapter.logs(handle, since=cursor)
        for line in lines:
            context.log(line)
            if deadline is not None:
                deadline.see(line, self._clock())
        return cursor + len(lines)

    def _record(
        self,
        adapter: ProviderAdapter,
        context: StageContext,
        tier: str,
        poll: Poll,
        ledger: AttemptLedger,
        *,
        part: str = "",
        call: str = "",
    ) -> Attempt:
        """Price this call and write it down, whatever state it ended in. At the tier it
        ran on, which a GPU fallback list can make other than the one asked for.

        Once per call: an entry with this call's id already in the ledger -- written by a
        process that was killed before it could strike the call from its book -- is the
        call's entry, and is returned rather than written a second time."""
        if call:
            for existing in ledger.entries:
                if existing.call == call:
                    self._made.append(existing)
                    return existing
        tier = poll.tier or tier
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
            billing=poll.billing,
            queue_s=poll.queue_s,
            call=call,
        )
        ledger.append(entry).write(context.attempts_path)
        self._made.append(entry)
        which = f" (part {part})" if part else ""
        how = f" ({poll.billing})" if poll.billing else ""
        waited = "" if poll.queue_s is None else f", after {poll.queue_s:.1f}s waiting for a GPU"
        billed = (
            f"cloud: {poll.state}{which}; billed {entry.billed_s:.1f}s{how}{waited} on "
            f"{adapter.name!r} ({tier})"
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
        # How those seconds were arrived at, and the wait for a GPU that preceded them --
        # a cost in time, not money (`Poll.queue_s`).
        bases = sorted({call.billing for call in calls if call.billing})
        if bases:
            metrics["billingBasis"] = ",".join(bases)
        queued = [call.queue_s for call in calls if call.queue_s is not None]
        if queued:
            metrics["queueS"] = round(sum(queued), 3)
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


@dataclass(frozen=True)
class _Called:
    """A call that succeeded: who ran it, its verdict, when it was submitted, and what it
    was submitted with -- whose keys, not this attempt's, are where its outputs are."""

    adapter: ProviderAdapter
    poll: Poll
    submitted_at: float
    request: StageRequest


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
    #: What it was submitted with; its checkpoint key is where its result is collected
    #: from (an earlier attempt's, for a piece re-attached to across a dead worker).
    request: StageRequest | None = None


@dataclass
class _Deadline:
    """How long a call's own progress lines say it should take, and whether it has run
    far past that (`CloudRunner`'s `deadline_factor`).

    Armed by the first tqdm line that has a rate (`progress.parse`: `remaining_s` known)
    and re-armed by every later one: `expected_s` is that line's elapsed plus remaining
    -- the projection for the planned steps at the rate the tool is actually going -- and
    `began` when its clock started on ours. So a trainer that slows down as its gaussians
    multiply moves its own deadline with it, and only one that stops reporting altogether
    is ever overdue. Off with a factor of 0, and silent for a call that prints no
    progress at all (a pose stage between its tools' bars, a join).
    """

    factor: float
    slack_s: float
    began: float | None = None
    expected_s: float = 0.0

    def see(self, line: str, now: float) -> None:
        if self.factor <= 0:
            return
        reading = progress.parse(line)
        if reading is None or reading.remaining_s is None or reading.done <= 0:
            return
        self.began = now - reading.elapsed_s
        self.expected_s = float(reading.elapsed_s + reading.remaining_s)

    def overrun(self, now: float) -> float | None:
        """Seconds since the tool started, once past the deadline; None until then."""
        if self.factor <= 0 or self.began is None:
            return None
        ran = now - self.began
        return ran if ran > self.factor * self.expected_s + self.slack_s else None


def _why(error: BaseException) -> str:
    """A cancel's reason, for the log and the ledger, from what stopped the watch."""
    reason = getattr(error, "reason", "")
    if reason == "cancelled":
        return "the job was cancelled or the worker lost its lease"
    if isinstance(reason, str) and reason:
        return reason
    return f"the process watching it stopped ({type(error).__name__})"


def _append_log(path: Path, line: str) -> None:
    """One line into a stage's log from outside its run (`CloudRunner.reap`)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip("\n") + "\n")
    except OSError:
        pass


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


def billed_from(metrics: Mapping[str, MetricValue]) -> float | None:
    """When the provider started billing a call (epoch seconds), as its container said.

    A cold container (`containerCall` 1) is billed from its own start: the sandbox's
    (`containerStartedAt`), or, from an image that does not report it, the moment the
    app's module was imported (`containerBootedAt`) -- later, so the bill is understated
    by the image's mount and Python's start, a few seconds. A warm container was already
    paid for by the call before; this one is billed from entering the function body. None
    when the call reported neither (an older image), and the proxy stands.
    """
    call = _number(metrics, "containerCall")
    entered = _number(metrics, "remoteEnteredAt") or _number(metrics, "remoteStartedAt")
    if call is None:
        return None
    if call <= 1:
        return (
            _number(metrics, "containerStartedAt")
            or _number(metrics, "containerBootedAt")
            or entered
        )
    return entered


def container_billing(
    metrics: Mapping[str, MetricValue], submitted_at: float, proxy_s: float
) -> tuple[float, float] | None:
    """(billed seconds, queue seconds) of a finished call, from the container's clock.

    Billed: `billed_from` to `remoteFinishedAt`. Queue: the submit (`submitted_at`, the
    runner's wall clock) to `billed_from` -- the wait for a GPU, which a per-second
    provider does not bill. Both are clamped to `proxy_s` (the runner's wall time from
    submit to seeing the result), which neither can exceed but for skew between the two
    clocks. None when the call did not report enough to say.
    """
    start = billed_from(metrics)
    finished = _number(metrics, "remoteFinishedAt")
    if start is None or finished is None or finished < start:
        return None
    billed = min(finished - start, proxy_s)
    queue = min(max(0.0, start - submitted_at), max(0.0, proxy_s - billed))
    return billed, queue


def call_phases(
    submitted_at: float | None, poll: Poll, *, collect_s: float | None = None
) -> dict[str, float]:
    """One call's billed seconds (`poll.billed_s`), phase by phase, from what the remote
    reported (`remote.execute`, `infra/modal/app.py`) and the stage's own `callPhases`.

    Additive, in order, to the billed figure:

    * `start` -- the container's start to the function body (a cold start: the image and
      Python; nothing on a warm container, `cold` says which). When the billed figure is
      the proxy (`poll.billing` is not `BILLING_CONTAINER`) it is the submit to the
      function body instead, the GPU queue included, because the proxy includes it;
    * `import` -- the pipeline's imports in the container;
    * `fetch` -- inputs and checkpoint down;
    * the stage's own phases (`phases.py`: `stageDataset`, `budget`, `prepare`, and a
      block's `dataset`, `seed`, `train`, `post`, or the join's `merge`, `eval`,
      `holdout`), with `stageOther` for whatever the stage did not time -- or `stage`
      whole when it timed nothing;
    * `finalSync` and `output` -- the checkpoint's last sync and `out/` up;
    * `rest` -- what none of that covers: on the proxy, the result reaching the runner
      (a poll interval, a log fetch); on the container's clock, the call's own tail; and
      any skew between the two clocks.

    Not in the sum: `queue`, the wait for a GPU before billing began (container basis
    only); `bgSync`, the checkpoint syncer's own seconds while the stage ran (on its own
    thread, beside the stage); and `collect`, the runner bringing a finished piece's
    result home after the call has stopped billing.
    """
    metrics: Mapping[str, MetricValue] = poll.metrics or {}
    out: dict[str, float] = {}
    entered = _number(metrics, "remoteEnteredAt") or _number(metrics, "remoteStartedAt")
    since = billed_from(metrics) if poll.billing == BILLING_CONTAINER else submitted_at
    if since is not None and entered is not None:
        out["start"] = max(0.0, entered - since)
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
    if poll.queue_s is not None:
        out["queue"] = poll.queue_s
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
