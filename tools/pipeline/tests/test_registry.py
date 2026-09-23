"""Adding an implementation is a decorator and a recipe edit. Nothing else moves.

These tests are the evidence for that claim: a new impl is registered here, in a test file,
and a recipe picks it up with no change to the executor, the runners, the recipe format or
the artifact contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import registry
from artifacts import ArtifactDecl
from conftest import make_recipe, seeded_workdir
from contracts import StageContext, StageOutcome
from errors import DuplicateImplError, UnresolvedArtifactError
from executor import execute
from plan import plan_recipe
from recipe import load_recipe
from registry import stage_impl
from runners import LocalRunner, RunnerSet

MEASUREMENTS = ArtifactDecl(
    "measurements.json", content_type="application/json", summary="an entirely new artifact"
)


@stage_impl(
    "t_measure",
    consumes=("georef.json",),
    produces=(MEASUREMENTS,),
    summary="a stage that did not exist when the executor was written",
)
def _measure(ctx: StageContext) -> StageOutcome:
    georef = json.loads(ctx.input("georef.json").read_text(encoding="utf-8"))
    ctx.output("measurements.json").write_text(
        json.dumps({"lat": georef["lat"]}, sort_keys=True) + "\n", encoding="utf-8"
    )
    return StageOutcome(metrics={"measured": 1})


def test_a_new_impl_is_a_decorator_and_a_recipe_edit(tmp_path: Path) -> None:
    recipe = make_recipe(
        [
            {"id": "georeference", "impl": "manual_placement", "params": {"lat": 46.84}},
            {"id": "measure", "impl": "t_measure"},
        ],
        inputs=[],
    )
    workdir = seeded_workdir(tmp_path / "run", upload=False)

    result = execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))

    assert [step.impl for step in result.steps] == ["manual_placement", "t_measure"]
    measured = workdir.root / result.artifact("measurements.json").path
    assert json.loads(measured.read_text())["lat"] == pytest.approx(46.84)
    # And the stub knows how to fake it without being told anything about it.
    stub_workdir = seeded_workdir(tmp_path / "stub", upload=False)
    execute(recipe, stub_workdir, RunnerSet.stubbed())
    assert (stub_workdir.root / "stages/measure/out/measurements.json").exists()


def test_swapping_mask_none_for_mask_robust_rewires_the_trainer(tmp_path: Path) -> None:
    document = json.loads(json.dumps(load_recipe("photo-reconstruct").to_dict()))
    before = plan_recipe(load_recipe("photo-reconstruct"))
    assert "masks" not in dict(before.stages[3].inputs)

    for stage in document["stages"]:
        if stage["id"] == "mask":
            stage["impl"] = "robust"
    from recipe import Recipe

    after = plan_recipe(Recipe.from_dict(document))

    # gsplat declares `masks` as an optional consume, so the same trainer now gets them --
    # with no change to the trainer, the executor, or any other stage.
    assert "masks" in dict(after.stages[3].inputs)
    assert after.origins["masks"] == "mask"
    execute(Recipe.from_dict(document), seeded_workdir(tmp_path / "run"), RunnerSet.stubbed())


def test_swapping_pose_colmap_for_glomap_needs_no_other_change() -> None:
    document = json.loads(json.dumps(load_recipe("photo-reconstruct").to_dict()))
    for stage in document["stages"]:
        if stage["id"] == "pose":
            stage["impl"] = "glomap"
    from recipe import Recipe

    plan = plan_recipe(Recipe.from_dict(document))

    assert plan.stages[1].impl.name == "glomap"
    assert plan.origins["poses"] == "pose"


def test_an_impl_swap_that_breaks_the_chain_is_caught_by_planning() -> None:
    # `arkit` needs the frames as well as the upload, so dropping the normalize stage is
    # refused at plan time rather than when the stage runs.
    recipe = make_recipe([{"id": "pose", "impl": "arkit"}], inputs=["upload"])

    with pytest.raises(UnresolvedArtifactError, match="consumes 'frames'"):
        plan_recipe(recipe)


def test_registering_the_same_name_twice_is_refused() -> None:
    with pytest.raises(DuplicateImplError, match="already registered"):

        @stage_impl("gsplat")
        def _another_gsplat(ctx: StageContext) -> StageOutcome:
            return StageOutcome()


def test_the_registry_knows_every_shipped_impl() -> None:
    shipped = {"colmap", "gsplat", "ingest_splat", "none", "splat_tiles"}
    assert shipped <= set(registry.known_impls())
