"""The recipe format: an ordered list of stages, not a DAG.

    name: photo-reconstruct
    version: 1
    inputs: [upload]
    stages:
      - { id: normalize, impl: ffmpeg_frames, params: { fps: 4 } }
      - { id: train,     impl: gsplat, gpu: { tier: l4, preemptible: true } }

`inputs` are the artifacts the caller seeds into `<workdir>/inputs/` before the run -- for
both shipped recipes that is `upload`, the directory of files the browser put in object
storage. Every other artifact must be produced by a stage.

A DAG with `needs` edges was considered and rejected: the real graph is almost linear, and
the one genuine fan-out (mesh, point cloud and splat from the same poses) is three
consecutive stages. Ordering plus declared artifacts gives the same modularity and the same
error checking for a fraction of the executor.

`gpu:` is the only routing signal there is. A stage either declares it or it does not, and
that single fact chooses the runner.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from artifacts import validate_artifact_name
from errors import RecipeError

__all__ = ["GpuRequest", "Recipe", "RecipeStage", "load_recipe", "recipe_dir"]

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def recipe_dir() -> Path:
    return Path(__file__).resolve().parent / "recipes"


@dataclass(frozen=True)
class GpuRequest:
    """A stage's GPU requirement. Its presence is what routes the stage to a GPU runner."""

    tier: str
    preemptible: bool = False

    def to_dict(self) -> dict[str, object]:
        return {"tier": self.tier, "preemptible": self.preemptible}


@dataclass(frozen=True)
class RecipeStage:
    id: str
    impl: str
    params: Mapping[str, Any]
    gpu: GpuRequest | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "impl": self.impl,
            "params": dict(self.params),
            "gpu": self.gpu.to_dict() if self.gpu else None,
        }


@dataclass(frozen=True)
class Recipe:
    name: str
    version: int
    stages: tuple[RecipeStage, ...]
    inputs: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "inputs": list(self.inputs),
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def with_params(self, overrides: Mapping[str, Mapping[str, Any]]) -> Recipe:
        """A copy of this recipe with per-stage parameter overrides merged over its own.

        This is how a run carries the facts a recipe file cannot know: where the operator
        placed this capture, what sensor took it, what the site should be called. The
        recipe still decides which stages run and in what order -- an override can only
        change a value a stage already reads.

        Keyed by **stage id**, not by impl, because two stages may run the same impl. An
        override naming a stage this recipe does not have is refused: a coordinate that
        silently went nowhere would put the site in the Gulf of Guinea and say nothing.
        """
        unknown = sorted(set(overrides) - {stage.id for stage in self.stages})
        if unknown:
            known = ", ".join(stage.id for stage in self.stages)
            raise RecipeError(
                f"recipe {self.name!r}: parameter overrides name stage(s) "
                f"{', '.join(unknown)}, which it does not have. Its stages are: {known}"
            )
        if not overrides:
            return self
        stages = tuple(
            replace(stage, params={**stage.params, **dict(overrides.get(stage.id, {}))})
            for stage in self.stages
        )
        return replace(self, stages=stages)

    @staticmethod
    def from_dict(document: Mapping[str, Any]) -> Recipe:
        name = _require_str(document, "name")
        version = document.get("version", 1)
        if not isinstance(version, int) or isinstance(version, bool):
            raise RecipeError(f"recipe {name!r}: `version` must be an integer")
        raw_inputs = document.get("inputs", [])
        if not isinstance(raw_inputs, Sequence) or isinstance(raw_inputs, str):
            raise RecipeError(f"recipe {name!r}: `inputs` must be a list of artifact names")
        inputs = tuple(validate_artifact_name(_as_str(name, "inputs", item)) for item in raw_inputs)
        raw_stages = document.get("stages")
        if not isinstance(raw_stages, Sequence) or not raw_stages:
            raise RecipeError(f"recipe {name!r}: `stages` must be a non-empty list")
        stages = tuple(_stage_from(name, index, item) for index, item in enumerate(raw_stages))
        seen: set[str] = set()
        for stage in stages:
            if stage.id in seen:
                raise RecipeError(f"recipe {name!r}: duplicate stage id {stage.id!r}")
            seen.add(stage.id)
        description = document.get("description", "")
        if not isinstance(description, str):
            raise RecipeError(f"recipe {name!r}: `description` must be a string")
        return Recipe(
            name=name,
            version=version,
            stages=stages,
            inputs=inputs,
            description=description,
        )

    @staticmethod
    def from_yaml(path: Path) -> Recipe:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise RecipeError(f"{path}: expected a YAML mapping")
        return Recipe.from_dict(document)


def load_recipe(name_or_path: str | Path) -> Recipe:
    """Load a shipped recipe by name (`splat-ingest`) or any recipe by path."""
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"}:
        if not candidate.exists():
            raise RecipeError(f"no recipe file at {candidate}")
        return Recipe.from_yaml(candidate)
    shipped = recipe_dir() / f"{name_or_path}.yaml"
    if not shipped.exists():
        available = ", ".join(sorted(p.stem for p in recipe_dir().glob("*.yaml"))) or "none"
        raise RecipeError(f"unknown recipe {str(name_or_path)!r}. Shipped recipes: {available}")
    return Recipe.from_yaml(shipped)


def _require_str(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise RecipeError(f"recipe: `{key}` is required and must be a non-empty string")
    return value


def _as_str(recipe: str, key: str, value: Any) -> str:
    if not isinstance(value, str):
        raise RecipeError(f"recipe {recipe!r}: `{key}` entries must be strings, got {value!r}")
    return value


def _stage_from(recipe: str, index: int, item: Any) -> RecipeStage:
    where = f"recipe {recipe!r}, stage #{index}"
    if not isinstance(item, Mapping):
        raise RecipeError(f"{where}: expected a mapping with `id` and `impl`")
    stage_id = _as_str(recipe, "id", item.get("id"))
    impl = _as_str(recipe, "impl", item.get("impl"))
    if not _ID_RE.match(stage_id):
        raise RecipeError(f"{where}: stage id {stage_id!r} must be lower_snake_case")
    if not _ID_RE.match(impl):
        raise RecipeError(f"{where}: impl {impl!r} must be lower_snake_case")
    params = item.get("params") or {}
    if not isinstance(params, Mapping):
        raise RecipeError(f"{where}: `params` must be a mapping")
    unknown = set(item) - {"id", "impl", "params", "gpu"}
    if unknown:
        raise RecipeError(f"{where}: unknown keys {sorted(unknown)}")
    return RecipeStage(id=stage_id, impl=impl, params=dict(params), gpu=_gpu_from(where, item))


def _gpu_from(where: str, item: Mapping[str, Any]) -> GpuRequest | None:
    gpu = item.get("gpu")
    if gpu is None:
        return None
    if not isinstance(gpu, Mapping):
        raise RecipeError(f"{where}: `gpu` must be a mapping with a `tier`")
    tier = gpu.get("tier")
    if not isinstance(tier, str) or not tier:
        raise RecipeError(f"{where}: `gpu.tier` is required (for example `l4`)")
    preemptible = bool(gpu.get("preemptible", False))
    unknown = set(gpu) - {"tier", "preemptible"}
    if unknown:
        raise RecipeError(f"{where}: unknown `gpu` keys {sorted(unknown)}")
    return GpuRequest(tier=tier, preemptible=preemptible)
