"""Which failures are worth another attempt, and what that attempt is told.

Until this, every failure got `worker_max_attempts` (3) -- under a six-hour Modal limit,
so a training stage that timed out was paid for three times, eighteen GPU-hours, and a
recipe whose stage asked for an input it never declared failed the same way three times
in a row. The worker now reads each failure before it decides, and the classes are kept
narrow on purpose: a failure put in the wrong one either spends money on a retry that
cannot help or gives up on one that could, and the second is the worse mistake, so
anything not plainly one of these keeps today's attempts.

* **CUDA ran out of memory** -- the failed attempt's own log says so (the trainer's
  traceback, tailed back from the provider; `CloudRunner` drains it after the verdict).
  Exactly **one** retry, with the gaussian cap the attempt trained at (its `gsplat:
  cap_max ...` line) times `OOM_CAP_SCALE`, written into `jobs.params` -- the existing
  per-run override, so it shows on the job and survives a worker restart. A second
  out-of-memory, or one with no cap in its log to lower, is not retried: the same run
  would run out the same way.
* **It ran out of time** (`RemoteTimeoutError`: the function's own limit, the runner's
  `max_wait_s`, or its overdue-trainer deadline): not retried. The next attempt would
  outrun the same limit, at the cost of the whole limit.
* **The recipe or its stage is wrong** (`errors.BAD_INPUT_ERRORS`, by name, for a stage
  that ran here or -- as the head of its provider's detail -- one that ran remotely):
  not retried.
* **The run reached its dollar cap** (`CostCapError`): not retried; it would start from
  the same spend.
* **Anything else**: the attempts it always had.

Which attempt's log: the stage log accumulates across attempts, so the supervisor notes
where it ended when each attempt started (`log_from`) and only what came after is read.
An out-of-memory in an earlier attempt that was then retried must not condemn a later
attempt that failed for an unrelated reason.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.worker.pipeline_bridge import BAD_INPUT_ERRORS

#: What a CUDA out-of-memory leaves in a log: torch's own message (`CUDA out of memory.
#: Tried to allocate ...`), its exception classes (`torch.OutOfMemoryError`, the older
#: `torch.cuda.OutOfMemoryError`), and the bare CUDA runtime error a kernel raises.
OOM_MARKERS: tuple[str, ...] = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "CUDA error: out of memory",
)

#: The training stage's budget line (`stages.train_gsplat`, `gaussian_budget.describe`):
#: `gsplat: cap_max auto -> 1234567; ...` or `gsplat: cap_max 400000 given; ...`.
CAP_LINE = re.compile(r"gsplat: cap_max (?:auto -> )?(\d+)")

#: The cap an out-of-memory retry trains at, as a share of the one that ran out.
OOM_CAP_SCALE = 0.7

#: The provider's own words at the head of a remote stage's failure: `RemoteStageError`'s
#: `... failed on 'modal': StageContractError: ...`.
_REMOTE_TYPE = re.compile(r"failed on '[^']+': ([A-Za-z_][A-Za-z0-9_]*)(?::|$)")

#: Where a stage's failures are kept, beside its ledger: what each attempt failed of.
HISTORY = "failures.json"

Kind = Literal["oom", "timeout", "bad-input", "cost-cap", "other"]


@dataclass(frozen=True)
class Failure:
    """One failed attempt, read."""

    kind: Kind
    #: For `oom`: the gaussian cap the attempt trained at, when its log says.
    cap_max: int | None = None


def classify(error_type: str, error: str, log: str) -> Failure:
    """What kind of failure this was, from what the stage raised and what it logged.

    `error_type` is the exception's class name as it crossed the line protocol, `error`
    its message, `log` the failed attempt's part of the stage log.
    """
    if error_type == "CostCapError":
        return Failure("cost-cap")
    if error_type == "RemoteTimeoutError":
        return Failure("timeout")
    remote = _REMOTE_TYPE.search(error) if error_type == "RemoteStageError" else None
    if error_type in BAD_INPUT_ERRORS or (remote and remote.group(1) in BAD_INPUT_ERRORS):
        return Failure("bad-input")
    if any(marker in log or marker in error for marker in OOM_MARKERS):
        caps = CAP_LINE.findall(log)
        # The first: the whole stage's budget, before a block run's own per-block lines.
        return Failure("oom", cap_max=int(caps[0]) if caps else None)
    return Failure("other")


@dataclass
class History:
    """`stages/<id>/failures.json`: what each failed attempt of one stage failed of.

    The supervisor's, not the pipeline's: it is how "exactly one retry" for an
    out-of-memory is kept across attempts and across a worker restart. Lost with the
    workdir, which restarts the run from nothing anyway.
    """

    path: Path
    entries: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def of(stage_dir: Path) -> History:
        path = stage_dir / HISTORY
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            entries = [dict(entry) for entry in document.get("failures") or ()]
        except (OSError, ValueError, TypeError, AttributeError):
            entries = []
        return History(path, entries)

    def add(self, attempt: int, failure: Failure) -> None:
        self.entries.append({"attempt": attempt, "kind": failure.kind, "capMax": failure.cap_max})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"failures": self.entries}, indent=1) + "\n", "utf-8")

    def had(self, kind: Kind) -> bool:
        return any(entry.get("kind") == kind for entry in self.entries)


@dataclass(frozen=True)
class Decision:
    """Another attempt, or not; and why not, in the words the job's error will carry."""

    retry: bool
    why: str = ""
    #: Parameters for the failed stage's next attempt, merged into `jobs.params[stage]`.
    params: dict[str, Any] = field(default_factory=dict)


def decide(stage_id: str, failure: Failure, history: History) -> Decision:
    """The policy, given what this failure was and what this stage failed of before.

    `history` is read before this failure is added to it.
    """
    if failure.kind == "cost-cap":
        return Decision(False, f"stage {stage_id!r} stopped at the run's cost cap")
    if failure.kind == "timeout":
        return Decision(
            False,
            f"stage {stage_id!r} ran out of time; another attempt would outrun the same "
            f"limit, so it is not retried",
        )
    if failure.kind == "bad-input":
        return Decision(
            False,
            f"stage {stage_id!r} failed on its recipe, its parameters or its own code, which "
            f"another attempt would not change, so it is not retried",
        )
    if failure.kind == "oom":
        if history.had("oom"):
            return Decision(
                False,
                f"stage {stage_id!r} ran out of GPU memory again after its gaussian cap was "
                f"lowered once; not retried a second time",
            )
        if failure.cap_max is None:
            return Decision(
                False,
                f"stage {stage_id!r} ran out of GPU memory, and its log names no gaussian cap "
                f"to lower, so another attempt would run out the same way",
            )
        lowered = max(1, int(failure.cap_max * OOM_CAP_SCALE))
        return Decision(True, params={"cap_max": lowered})
    return Decision(True)
