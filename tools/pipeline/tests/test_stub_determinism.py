"""StubRunner's outputs are byte-identical across runs, so tests can assert on them.

This is the property the whole CI story rests on: if the stub's bytes drifted between runs
the sibling project's byte-identity gate could not be extended to the pipeline, and no test
could assert on a stage's output without re-deriving it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import seeded_workdir, tree
from executor import execute
from recipe import Recipe, load_recipe
from runners import RunnerSet

RECIPES = ["splat-ingest", "photo-reconstruct"]


def _run(root: Path, name: str, *, params: dict[str, object] | None = None) -> dict[str, bytes]:
    recipe = load_recipe(name)
    if params:
        document: Any = json.loads(json.dumps(recipe.to_dict()))
        document["stages"][0]["params"].update(params)
        recipe = Recipe.from_dict(document)
    workdir = seeded_workdir(root)
    execute(recipe, workdir, RunnerSet.stubbed())
    outputs = tree(workdir.stages_dir)
    return {
        path: data
        for path, data in outputs.items()
        # step.json holds the stage's wall time, which is the one thing that cannot be
        # deterministic. Everything a later stage or a test reads is in out/ or in
        # artifacts.json.
        if "/out/" in path
    } | {"artifacts.json": workdir.manifest_path.read_bytes()}


@pytest.mark.parametrize("name", RECIPES)
def test_the_same_recipe_over_the_same_inputs_is_byte_identical(name: str, tmp_path: Path) -> None:
    first = _run(tmp_path / "a", name)
    second = _run(tmp_path / "b", name)

    assert first.keys() == second.keys()
    assert first == second


def test_a_changed_parameter_changes_the_bytes(tmp_path: Path) -> None:
    base = _run(tmp_path / "a", "photo-reconstruct")
    changed = _run(tmp_path / "b", "photo-reconstruct", params={"keep": 41})

    assert base.keys() == changed.keys()
    assert base != changed


def test_a_changed_input_changes_the_bytes(tmp_path: Path) -> None:
    base = _run(tmp_path / "a", "splat-ingest")
    other = seeded_workdir(tmp_path / "b")
    (other.input_path("upload") / "capture.ply").write_bytes(b"different bytes entirely")
    execute(load_recipe("splat-ingest"), other, RunnerSet.stubbed())

    assert base["artifacts.json"] != other.manifest_path.read_bytes()
