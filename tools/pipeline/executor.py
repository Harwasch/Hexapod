"""Run a recipe: validate everything, then run the stages in order.

The executor knows about ordering, artifacts and runner selection. It knows nothing about
splats, poses, COLMAP or GPUs, which is why adding an implementation never touches it.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from artifacts import ArtifactManifest, ArtifactRef
from contracts import StepResult
from plan import Plan, check_inputs, plan_recipe
from recipe import Recipe, load_recipe
from runners import RunnerSet
from workdir import Workdir

__all__ = ["RunResult", "execute", "execute_recipe"]


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


def execute(recipe: Recipe, workdir: Workdir, runners: RunnerSet) -> RunResult:
    """Validate, then run every stage in order.

    Nothing is created and no stage is invoked until the whole recipe resolves: an unknown
    impl, an unproduced input or a GPU stage with no GPU runner all fail here, before the
    first stage does any work.
    """
    plan = plan_recipe(recipe)
    check_inputs(recipe, workdir)
    for stage in plan.stages:
        runners.for_stage(stage)
    workdir.stages_dir.mkdir(parents=True, exist_ok=True)
    _write_plan(plan, workdir)
    steps: list[StepResult] = []
    artifacts: list[ArtifactRef] = []
    for stage in plan.stages:
        result = runners.for_stage(stage).run(stage, workdir)
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
) -> RunResult:
    """Convenience for the CLI and, in A7, for the worker: load, seed the inputs, run.

    `seed` maps a recipe input name to a file or directory to copy into the workdir. A7
    fetches the capture's bytes out of object storage into exactly these paths.
    """
    recipe = load_recipe(name_or_path)
    work = Workdir.create(Path(workdir))
    for name, source in (seed or {}).items():
        _seed(work, name, Path(source))
    return execute(recipe, work, runners)


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
