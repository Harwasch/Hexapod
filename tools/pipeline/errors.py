"""Every way the pipeline refuses, with enough context to name the offender.

Errors carry the recipe and stage they came from because the audience is somebody
reading a job's failure in the console (A7/A10), not a Python traceback.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Literal

__all__ = [
    "BAD_INPUT_ERRORS",
    "CancelRequested",
    "CostCapError",
    "DetachRequested",
    "DuplicateArtifactError",
    "DuplicateImplError",
    "MissingArtifactError",
    "MissingInputError",
    "NoRunnerError",
    "PipelineError",
    "PreemptedError",
    "QualityGateError",
    "RecipeError",
    "RemoteStageError",
    "RemoteTimeoutError",
    "ResumeError",
    "StageContractError",
    "StageFailedError",
    "StopRequested",
    "UndeclaredArtifactError",
    "UnknownImplError",
    "UnresolvedArtifactError",
    "check_gate",
    "parse_quality_gate",
]


class PipelineError(Exception):
    """Base class: anything this package raises deliberately."""


class RecipeError(PipelineError):
    """A recipe document is malformed, before any impl is looked up."""


class DuplicateImplError(PipelineError):
    """Two stage implementations registered under the same name."""


class UnknownImplError(PipelineError):
    def __init__(self, recipe: str, stage_id: str, impl: str, known: Iterable[str]) -> None:
        options = ", ".join(sorted(known)) or "none registered"
        super().__init__(
            f"recipe {recipe!r}, stage {stage_id!r}: unknown impl {impl!r}. "
            f"Registered impls: {options}"
        )
        self.recipe = recipe
        self.stage_id = stage_id
        self.impl = impl


class UnresolvedArtifactError(PipelineError):
    """A stage consumes an artifact no earlier stage produces."""

    def __init__(
        self, recipe: str, stage_id: str, impl: str, artifact: str, available: Iterable[str]
    ) -> None:
        have = ", ".join(sorted(available)) or "nothing"
        super().__init__(
            f"recipe {recipe!r}, stage {stage_id!r} (impl {impl!r}): consumes {artifact!r}, "
            f"which no earlier stage produces. Available at that point: {have}"
        )
        self.recipe = recipe
        self.stage_id = stage_id
        self.artifact = artifact


class DuplicateArtifactError(PipelineError):
    """Two stages produce the same artifact name."""

    def __init__(self, recipe: str, stage_id: str, artifact: str, first_stage: str) -> None:
        super().__init__(
            f"recipe {recipe!r}, stage {stage_id!r}: produces {artifact!r}, "
            f"already produced by stage {first_stage!r}"
        )
        self.recipe = recipe
        self.stage_id = stage_id
        self.artifact = artifact


class MissingInputError(PipelineError):
    """A recipe input was not seeded into the workdir before the run."""

    def __init__(self, recipe: str, name: str, path: str) -> None:
        super().__init__(f"recipe {recipe!r}: input {name!r} is missing from the workdir ({path})")
        self.recipe = recipe
        self.name = name


class MissingArtifactError(PipelineError):
    """A stage declared a `produces` it did not write. Always loud, never silent."""

    def __init__(self, recipe: str, stage_id: str, impl: str, artifact: str, path: str) -> None:
        super().__init__(
            f"recipe {recipe!r}, stage {stage_id!r} (impl {impl!r}): declared it produces "
            f"{artifact!r} but wrote nothing at {path}"
        )
        self.recipe = recipe
        self.stage_id = stage_id
        self.artifact = artifact


class UndeclaredArtifactError(PipelineError):
    """A stage wrote something into its output directory that it never declared."""

    def __init__(self, recipe: str, stage_id: str, impl: str, entries: Iterable[str]) -> None:
        names = ", ".join(sorted(entries))
        super().__init__(
            f"recipe {recipe!r}, stage {stage_id!r} (impl {impl!r}): wrote undeclared entries "
            f"into its output directory: {names}. Declare them in `produces` or write them to "
            f"the stage's work/ directory instead"
        )
        self.recipe = recipe
        self.stage_id = stage_id


class ResumeError(PipelineError):
    """A run was asked to skip a stage whose previous result is not in the workdir.

    Skipping is how A7 retries from a stage rather than from the beginning, and it is
    only honest while the earlier stage's `step.json` and outputs are still on disk. A
    workdir that has been cleaned up gets a fresh run, never a half one.
    """

    def __init__(self, recipe: str, stage_id: str, path: str) -> None:
        super().__init__(
            f"recipe {recipe!r}: cannot skip stage {stage_id!r} -- no previous result at {path}. "
            f"Run it again instead of resuming past it"
        )
        self.recipe = recipe
        self.stage_id = stage_id


class StageContractError(PipelineError):
    """A stage implementation asked for an artifact it never declared."""


class NoRunnerError(PipelineError):
    """No runner is configured for a stage -- typically a gpu: stage with no CloudRunner."""


class PreemptedError(PipelineError):
    """The provider took the machine back. **Not a failure**, and deliberately its own
    class so the worker can tell the two apart.

    On an interruptible tier this is ordinary operation: the attempt is over, whatever
    the remote last synced to `checkpoint/` has been brought home, and the next attempt
    continues from it. It still carries what the lost attempt billed, because that money
    was spent whether or not the work survived.
    """

    def __init__(
        self, recipe: str, stage_id: str, attempt: int, provider: str, billed_s: float
    ) -> None:
        super().__init__(
            f"recipe {recipe!r}, stage {stage_id!r}: attempt {attempt} was preempted by "
            f"{provider!r} after {billed_s:.1f}s of billed time. It resumes from its "
            f"checkpoint on the next attempt"
        )
        self.recipe = recipe
        self.stage_id = stage_id
        self.attempt = attempt
        self.provider = provider
        self.billed_s = billed_s


class RemoteStageError(PipelineError):
    """A stage run on a provider failed there. The provider's own words, carried back."""

    def __init__(self, recipe: str, stage_id: str, impl: str, provider: str, detail: str) -> None:
        super().__init__(
            f"recipe {recipe!r}, stage {stage_id!r} (impl {impl!r}) failed on "
            f"{provider!r}: {detail or 'no detail was reported'}"
        )
        self.recipe = recipe
        self.stage_id = stage_id
        self.impl = impl
        self.provider = provider


class StageFailedError(PipelineError):
    """A stage implementation raised. The cause is chained, the location is here."""

    def __init__(self, recipe: str, stage_id: str, impl: str, cause: BaseException) -> None:
        super().__init__(f"recipe {recipe!r}, stage {stage_id!r} (impl {impl!r}) failed: {cause}")
        self.recipe = recipe
        self.stage_id = stage_id
        self.impl = impl


class RemoteTimeoutError(RemoteStageError):
    """A remote stage ran out of time rather than failing at something.

    Its own class because the worker's retry policy treats it differently from a failure:
    a stage that outran the deployed function's own limit (Modal's `FunctionTimeoutError`,
    six hours in `infra/modal/app.py`), the runner's `max_wait_s`, or the deadline its
    planned steps imply (`CloudRunner`'s `deadline_factor`) will outrun it again on the
    next attempt, and every such attempt costs the whole limit in GPU time. A stage that is
    merely slow and then *finishes* never raises this.
    """


class CostCapError(PipelineError):
    """A run reached the dollar ceiling its worker was configured with.

    Raised before a call is submitted when what the run has already been billed is at or
    over the cap, and in place of a running call's result when the call was cancelled
    because its own running cost would have taken the run over it. Never retried: the
    next attempt would start from the same spend.
    """

    def __init__(
        self, recipe: str, stage_id: str, spent_usd: float, cap_usd: float, why: str
    ) -> None:
        super().__init__(
            f"recipe {recipe!r}, stage {stage_id!r}: the run has been billed "
            f"${spent_usd:.2f} against a cap of ${cap_usd:.2f} (WORKER_JOB_COST_CAP_USD); "
            f"{why}"
        )
        self.recipe = recipe
        self.stage_id = stage_id
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd


class QualityGateError(PipelineError):
    """A measured property of the capture failed a gate the recipe turned on; another
    attempt measures the same bytes the same way.

    The message is a fixed format so it survives the worker's line protocol and a remote
    provider's wrapping (`RemoteStageError` carries it inside its own text) and can be
    read back by `parse_quality_gate`:
    `quality gate <gate>: measured <m>[ <unit>], limit <l>[ <unit>] -- <reason>`.
    """

    def __init__(
        self, gate: str, measured: float, limit: float, *, unit: str = "", reason: str
    ) -> None:
        if not _GATE_NAME.fullmatch(gate):
            raise ValueError(f"a gate is named by one word (letters, digits, _.-), not {gate!r}")
        if unit and not _GATE_UNIT.fullmatch(unit):
            raise ValueError(f"a gate's unit is one token with no comma, not {unit!r}")
        suffix = f" {unit}" if unit else ""
        one_line = " ".join(reason.split())
        super().__init__(
            f"quality gate {gate}: measured {measured:.6g}{suffix}, limit {limit:.6g}{suffix} "
            f"-- {one_line}"
        )
        self.gate = gate
        self.measured = measured
        self.limit = limit
        self.unit = unit
        self.reason = one_line

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "measured": self.measured,
            "limit": self.limit,
            "unit": self.unit,
            "reason": self.reason,
        }


_GATE_NAME = re.compile(r"[A-Za-z0-9_.\-]+")
_GATE_UNIT = re.compile(r"[^\s,]+")
_NUMBER = r"[-+]?(?:inf|nan|\d+(?:\.\d*)?(?:e[-+]?\d+)?)"
_GATE_MESSAGE = re.compile(
    rf"quality gate (?P<gate>[A-Za-z0-9_.\-]+): measured (?P<measured>{_NUMBER})"
    rf"(?: (?P<unit>[^\s,]+))?, limit (?P<limit>{_NUMBER})(?: [^\s,]+)? -- (?P<reason>[^\n]*)",
    re.I,
)


def parse_quality_gate(text: str) -> dict[str, object] | None:
    """A `QualityGateError` message read back out of any text that carries one -- its own
    `str()`, or a `RemoteStageError`'s `"... failed on 'modal': QualityGateError: ..."` --
    as `{gate, measured, limit, unit, reason}`; None when there is none."""
    match = _GATE_MESSAGE.search(text)
    if match is None:
        return None
    return {
        "gate": match.group("gate"),
        "measured": float(match.group("measured")),
        "limit": float(match.group("limit")),
        "unit": match.group("unit") or "",
        "reason": match.group("reason").strip(),
    }


def check_gate(
    gate: str,
    measured: float | None,
    limit: float | None,
    *,
    fail_when: Literal["below", "above"],
    unit: str = "",
    reason: str,
) -> dict[str, object]:
    """Apply one gate, and say what it did: `{name, enabled, limit, measured, passed}`.

    A gate is **off** when its limit is None, which is every gate's default unless a
    recorded measurement backs a threshold (DECISIONS: tolerances). An enabled gate whose
    measurement does not exist (None, or not a finite number) is not applied -- `passed`
    None -- rather than failing a run for what it could not measure. Raises
    `QualityGateError` when enabled and the measurement is `below` / `above` the limit.
    """
    finite = measured is not None and math.isfinite(measured)
    record: dict[str, object] = {
        "name": gate,
        "enabled": limit is not None,
        "limit": limit,
        "measured": measured if finite else None,
        "passed": None,
    }
    if limit is None or measured is None or not finite:
        return record
    failed = measured < limit if fail_when == "below" else measured > limit
    record["passed"] = not failed
    if failed:
        raise QualityGateError(gate, measured, limit, unit=unit, reason=reason)
    return record


#: The errors that say the recipe, its parameters or its stage code are wrong rather than
#: that a run was unlucky: another attempt fails the same way, so the worker dead-letters
#: on the first one instead of spending the attempt budget on it. Kept to the classes
#: whose every raise is a contract or configuration mistake -- not `MissingArtifactError`
#: (outputs that did not come back can be a transfer that failed), not `ResumeError` (the
#: next attempt re-reads the workdir and does not skip), and not a stage's own
#: `ValueError`, which is raised for bad input and for a trainer that died alike. By name,
#: because a name is how a failure crosses the worker's line protocol.
BAD_INPUT_ERRORS: frozenset[str] = frozenset(
    {
        "RecipeError",
        "UnknownImplError",
        "UnresolvedArtifactError",
        "DuplicateArtifactError",
        "MissingInputError",
        "StageContractError",
        "NoRunnerError",
        # Not a mistake in the recipe, but just as deterministic: the gate measured the
        # capture's own bytes, and the next attempt measures the same bytes the same way.
        "QualityGateError",
    }
)


class StopRequested(BaseException):
    """The process running a recipe was asked to stop, and the subclass says why.

    A `BaseException`, like `KeyboardInterrupt`, and not a `PipelineError`: it is raised
    from a signal handler wherever the main thread happens to be, and that is often inside
    an `except Exception` that exists for a different reason -- `ModalAdapter.poll`
    classifies whatever its `get` raises, `ModalAdapter.logs` suppresses every exception.
    Caught there, a stop would be read as a failed stage or swallowed outright.

    The worker's supervisor raises one of the two subclasses in its recipe process with a
    signal (`app/worker/child.py`), and `CloudRunner` is what tells them apart: a cancel
    stops paying for the remote call, a detach leaves it running for the next worker.
    """

    reason = "stopped"


class CancelRequested(StopRequested):
    """The job was cancelled, or this worker lost its lease: stop the remote call now."""

    reason = "cancelled"


class DetachRequested(StopRequested):
    """The worker is shutting down (a deploy): leave the remote call running, write down
    where it is, and let the next worker re-attach to it instead of paying for it twice."""

    reason = "detached"
