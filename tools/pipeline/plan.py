"""Planning: resolve every impl and every artifact before a single stage runs.

Planning is the whole error-checking story of an ordered recipe. Walking the stages in
order with a set of artifacts that exist so far catches, without running anything:

  * an `impl` that is not registered;
  * a stage consuming an artifact nothing earlier produces;
  * two stages producing the same artifact;
  * a recipe input that was never seeded into the workdir.

A planned stage carries the workdir-relative path of each of its inputs, so a runner needs
nothing but the plan and the workdir to run a stage -- including, in B1, a runner on a
different machine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import registry
from errors import (
    DuplicateArtifactError,
    MissingInputError,
    UnknownImplError,
    UnresolvedArtifactError,
)
from recipe import GpuRequest, Recipe, RecipeStage
from registry import StageImpl
from workdir import Workdir

__all__ = ["Plan", "PlannedStage", "plan_recipe"]

INPUT_ORIGIN = "<input>"


@dataclass(frozen=True)
class PlannedStage:
    recipe: str
    index: int
    stage: RecipeStage
    impl: StageImpl
    inputs: Mapping[str, str]
    """Consumed artifact name -> path relative to the workdir root."""

    @property
    def id(self) -> str:
        return self.stage.id

    @property
    def params(self) -> Mapping[str, object]:
        return self.stage.params

    @property
    def gpu(self) -> GpuRequest | None:
        return self.stage.gpu


@dataclass(frozen=True)
class Plan:
    recipe: Recipe
    stages: tuple[PlannedStage, ...]
    origins: Mapping[str, str]
    """Artifact name -> the stage id that produces it, or `<input>`."""

    @property
    def gpu_stages(self) -> tuple[str, ...]:
        return tuple(stage.id for stage in self.stages if stage.gpu is not None)


def plan_recipe(recipe: Recipe) -> Plan:
    """Validate a recipe end to end. Raises before anything is run or written."""
    origins: dict[str, str] = {name: INPUT_ORIGIN for name in recipe.inputs}
    paths: dict[str, str] = {name: f"inputs/{name}" for name in recipe.inputs}
    planned: list[PlannedStage] = []
    for index, stage in enumerate(recipe.stages):
        impl = registry.lookup(stage.impl)
        if impl is None:
            raise UnknownImplError(recipe.name, stage.id, stage.impl, registry.known_impls())
        resolved: dict[str, str] = {}
        for name in impl.consumes:
            if name not in paths:
                raise UnresolvedArtifactError(
                    recipe.name, stage.id, impl.name, name, origins.keys()
                )
            resolved[name] = paths[name]
        for name in impl.optional_consumes:
            if name in paths:
                resolved[name] = paths[name]
        planned.append(
            PlannedStage(recipe=recipe.name, index=index, stage=stage, impl=impl, inputs=resolved)
        )
        for decl in impl.produces:
            if decl.name in origins:
                raise DuplicateArtifactError(recipe.name, stage.id, decl.name, origins[decl.name])
            origins[decl.name] = stage.id
            paths[decl.name] = f"stages/{stage.id}/out/{decl.name}"
    return Plan(recipe=recipe, stages=tuple(planned), origins=origins)


def check_inputs(recipe: Recipe, workdir: Workdir) -> None:
    """Every declared recipe input must already exist in the workdir."""
    for name in recipe.inputs:
        path = workdir.input_path(name)
        if not path.exists():
            raise MissingInputError(recipe.name, name, workdir.relative(path))
