"""Both shipped recipes, end to end, on a machine with no GPU and no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import seeded_workdir
from executor import execute
from plan import plan_recipe
from recipe import load_recipe, recipe_dir
from runners import RunnerSet

RECIPES = ["splat-ingest", "photo-reconstruct"]


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_runs_end_to_end_under_the_stub_runner(name: str, tmp_path: Path) -> None:
    recipe = load_recipe(name)
    workdir = seeded_workdir(tmp_path / "run")

    result = execute(recipe, workdir, RunnerSet.stubbed())

    assert [step.stage_id for step in result.steps] == [s.id for s in recipe.stages]
    assert all(step.runner == "stub" for step in result.steps)
    # Every stage's declared produces exist, are non-empty, and are in the manifest.
    for artifact in result.artifacts:
        path = workdir.root / artifact.path
        assert path.exists()
        assert artifact.bytes > 0
        assert artifact.checksum.startswith("sha256:")
    assert json.loads(workdir.manifest_path.read_text())["artifacts"]
    assert json.loads(workdir.recipe_path.read_text())["recipe"]["name"] == name


@pytest.mark.parametrize("name", RECIPES)
def test_every_recipe_ends_in_a_registration(name: str, tmp_path: Path) -> None:
    workdir = seeded_workdir(tmp_path / "run")

    result = execute(load_recipe(name), workdir, RunnerSet.stubbed())

    # Both lanes converge: same canonical splat, same tileset, same registration.
    assert {"canonical.ply", "splat", "georef.json", "registration.json"} <= set(result.by_name)
    tiles = workdir.root / result.artifact("splat").path
    assert sorted(entry.name for entry in tiles.iterdir()) == ["splat.glb", "tileset.json"]


def test_photo_reconstruct_routes_only_its_gpu_stage_to_the_gpu_runner() -> None:
    plan = plan_recipe(load_recipe("photo-reconstruct"))

    assert plan.gpu_stages == ("train",)
    assert plan.origins["canonical.ply"] == "train"
    assert plan.origins["upload"] == "<input>"


def test_splat_ingest_needs_no_gpu_at_all() -> None:
    assert plan_recipe(load_recipe("splat-ingest")).gpu_stages == ()


def test_a_gpu_stage_with_no_gpu_runner_is_refused_before_anything_runs(tmp_path: Path) -> None:
    from errors import NoRunnerError

    workdir = seeded_workdir(tmp_path / "run")
    with pytest.raises(NoRunnerError, match="requires a l4 GPU"):
        execute(load_recipe("photo-reconstruct"), workdir, RunnerSet.local())
    assert not workdir.stages_dir.exists()


def test_every_shipped_recipe_is_valid() -> None:
    found = sorted(p.stem for p in recipe_dir().glob("*.yaml"))
    assert found == sorted(RECIPES)
    for name in found:
        plan_recipe(load_recipe(name))
