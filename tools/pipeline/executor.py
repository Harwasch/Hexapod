"""Run a recipe: validate everything, then run the stages in order.

The executor knows about ordering, artifacts and runner selection. It knows nothing about
splats, poses, COLMAP or GPUs, which is why adding an implementation never touches it.

Two seams exist for the worker (A7), and neither of them lets the API into this project:

* an **observer** is told when each stage starts, finishes, is skipped or fails, so a
  caller can write a step transition *while the run is going on* rather than reading the
  RunResult at the end. It is a plain callback protocol -- no HTTP, no database, no
  knowledge of what the observer does with it;
* **skip** and **attempts** are how a run resumes. A stage whose previous `step.json` is
  still in the workdir can be skipped, and its result is read back rather than recomputed,
  so `artifacts.json` after a resumed run says the same thing as after a fresh one.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from artifacts import ArtifactManifest, ArtifactRef
from contracts import StepResult
from errors import ResumeError
from plan import Plan, PlannedStage, check_inputs, plan_recipe
from recipe import Recipe, load_recipe
from runners import RunnerSet
from workdir import Workdir

__all__ = ["RunResult", "StageObserver", "execute", "execute_recipe"]


class StageObserver(Protocol):
    """What a caller is told, as it happens.

    Every method may raise: the exception propagates out of `execute` untouched, which is
    how A7 stops a run between stages when the job has been cancelled.
    """

    def stage_started(self, stage: PlannedStage, attempt: int) -> None: ...

    def stage_finished(self, stage: PlannedStage, result: StepResult) -> None: ...

    def stage_skipped(self, stage: PlannedStage, result: StepResult) -> None: ...

    def stage_failed(self, stage: PlannedStage, attempt: int, error: BaseException) -> None: ...


@dataclass(frozen=True)
class RunResult:
    recipe: str
    recipe_version: int
    steps: tuple[StepResult, ...]
    artifacts: tuple[ArtifactRef, ...]

    @property
    def by_name(self) -> Mapping[str, ArtifactRef]:
        return {artifact.name: artifact for artifact in self.artifacts}

    def artifact(self, name: str) -> ArtifactRef:
        return self.by_name[name]


def execute(
    recipe: Recipe,
    workdir: Workdir,
    runners: RunnerSet,
    *,
    observer: StageObserver | None = None,
    skip: Collection[str] = (),
    attempts: Mapping[str, int] | None = None,
) -> RunResult:
    """Validate, then run every stage in order.

    Nothing is created and no stage is invoked until the whole recipe resolves: an unknown
    impl, an unproduced input or a GPU stage with no GPU runner all fail here, before the
    first stage does any work.

    `skip` names stages whose previous result in this workdir stands. `attempts` gives the
    attempt number to record for a stage that is about to run -- A2's `job_steps.attempt`,
    which counts a resumed or preempted stage rather than pretending it ran once.
    """
    plan = plan_recipe(recipe)
    check_inputs(recipe, workdir)
    for stage in plan.stages:
        runners.for_stage(stage)
    _check_resumable(plan, workdir, skip)
    workdir.stages_dir.mkdir(parents=True, exist_ok=True)
    _write_plan(plan, workdir)
    steps: list[StepResult] = []
    artifacts: list[ArtifactRef] = []
    for stage in plan.stages:
        if stage.id in skip:
            result = StepResult.read(workdir.step_path(stage.id))
            if observer is not None:
                observer.stage_skipped(stage, result)
        else:
            attempt = (attempts or {}).get(stage.id, 1)
            if observer is not None:
                observer.stage_started(stage, attempt)
            try:
                result = runners.for_stage(stage).run(stage, workdir, attempt)
            except BaseException as error:
                if observer is not None:
                    observer.stage_failed(stage, attempt, error)
                raise
            if observer is not None:
                observer.stage_finished(stage, result)
        steps.append(result)
        artifacts.extend(result.artifacts)
    ArtifactManifest(entries=tuple(artifacts)).write(workdir.manifest_path)
    return RunResult(
        recipe=recipe.name,
        recipe_version=recipe.version,
        steps=tuple(steps),
        artifacts=tuple(artifacts),
    )


def execute_recipe(
    name_or_path: str | Path,
    workdir: Path,
    runners: RunnerSet,
    seed: Mapping[str, Path] | None = None,
    *,
    observer: StageObserver | None = None,
    skip: Collection[str] = (),
    attempts: Mapping[str, int] | None = None,
) -> RunResult:
    """Convenience for the CLI and, in A7, for the worker: load, seed the inputs, run.

    `seed` maps a recipe input name to a file or directory to copy into the workdir. A7
    fetches the capture's bytes out of object storage into exactly these paths.
    """
    recipe = load_recipe(name_or_path)
    work = Workdir.create(Path(workdir))
    for name, source in (seed or {}).items():
        _seed(work, name, Path(source))
    return execute(recipe, work, runners, observer=observer, skip=skip, attempts=attempts)


def _check_resumable(plan: Plan, workdir: Workdir, skip: Collection[str]) -> None:
    """A stage may only be skipped if its previous result is still here.

    Checked with the rest of the planning, before anything runs, so a workdir that was
    cleaned up between attempts fails loudly rather than producing a run whose
    `artifacts.json` is quietly missing half of it.
    """
    known = {stage.id for stage in plan.stages}
    for stage_id in sorted(skip):
        if stage_id not in known:
            raise ResumeError(plan.recipe.name, stage_id, "<not a stage of this recipe>")
        path = workdir.step_path(stage_id)
        if not path.exists():
            raise ResumeError(plan.recipe.name, stage_id, workdir.relative(path))


def _seed(workdir: Workdir, name: str, source: Path) -> None:
    target = workdir.input_path(name)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
        return
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target / source.name)


def _write_plan(plan: Plan, workdir: Workdir) -> None:
    """`recipe.json`: the recipe exactly as executed, plus where each artifact comes from.

    A run is reproducible from this file alone -- which is what the admin console's
    side-by-side comparison of two runs (A10) reads.
    """
    document = {
        "recipe": plan.recipe.to_dict(),
        "origins": dict(sorted(plan.origins.items())),
        "gpuStages": list(plan.gpu_stages),
    }
    workdir.recipe_path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", "utf-8")
