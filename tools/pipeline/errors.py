"""Every way the pipeline refuses, with enough context to name the offender.

Errors carry the recipe and stage they came from because the audience is somebody
reading a job's failure in the console (A7/A10), not a Python traceback.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "DuplicateArtifactError",
    "DuplicateImplError",
    "MissingArtifactError",
    "MissingInputError",
    "NoRunnerError",
    "PipelineError",
    "PreemptedError",
    "RecipeError",
    "RemoteStageError",
    "ResumeError",
    "StageContractError",
    "StageFailedError",
    "UndeclaredArtifactError",
    "UnknownImplError",
    "UnresolvedArtifactError",
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
