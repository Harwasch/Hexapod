"""The workdir and StepResult contract A7, A8 and B1 all build on."""

from __future__ import annotations

import json
from pathlib import Path

from artifacts import ArtifactDecl
from conftest import make_recipe, seeded_workdir
from contracts import StageContext, StageOutcome
from executor import execute
from plan import plan_recipe
from registry import stage_impl
from runners import LocalRunner, RunnerSet

STATE = ArtifactDecl("state.json", content_type="application/json")


@stage_impl("t_checkpoints", produces=(STATE,))
def _checkpoints(ctx: StageContext) -> StageOutcome:
    """Writes a checkpoint on the way, the way a long GPU stage would."""
    resumed = ctx.has_checkpoint
    (ctx.checkpoint_dir / "iteration.txt").write_text(str(ctx.attempt), encoding="utf-8")
    (ctx.work_dir / "scratch.bin").write_bytes(b"not an artifact")
    ctx.output("state.json").write_text(
        json.dumps({"attempt": ctx.attempt, "resumed": resumed}) + "\n", encoding="utf-8"
    )
    return StageOutcome(metrics={"resumed": resumed})


def test_a_stage_is_given_three_directories_and_only_one_of_them_is_an_artifact(
    tmp_path: Path,
) -> None:
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    recipe = make_recipe([{"id": "step", "impl": "t_checkpoints"}], inputs=[])

    result = execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))

    assert (workdir.out_dir("step") / "state.json").exists()
    assert (workdir.work_dir("step") / "scratch.bin").exists()
    assert (workdir.checkpoint_dir("step") / "iteration.txt").exists()
    # Only the declared output is an artifact. Scratch and checkpoints are not.
    assert [artifact.name for artifact in result.artifacts] == ["state.json"]


def test_a_second_attempt_keeps_the_checkpoint_and_clears_the_outputs(tmp_path: Path) -> None:
    """B1 runs on preemptible GPUs, where being killed is normal operation."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    recipe = make_recipe([{"id": "step", "impl": "t_checkpoints"}], inputs=[])
    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    # A killed attempt can leave a half-written output behind. Simulate one.
    (workdir.out_dir("step") / "half-written.tmp").write_bytes(b"truncated")

    planned = plan_recipe(recipe).stages[0]
    second = LocalRunner().run(planned, workdir, attempt=2)

    state = json.loads((workdir.out_dir("step") / "state.json").read_text())
    assert state == {"attempt": 2, "resumed": True}
    # The half-written file from the killed attempt is gone: it can never be mistaken for
    # a produced artifact.
    assert not (workdir.out_dir("step") / "half-written.tmp").exists()
    assert second.attempt == 2
    assert second.checkpoint_key == "runs/run/step/checkpoint"


def test_a_step_result_carries_artifacts_metrics_and_a_log(tmp_path: Path) -> None:
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    recipe = make_recipe([{"id": "step", "impl": "t_checkpoints"}], inputs=[])

    result = execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    step = result.steps[0]

    assert step.stage_id == "step"
    assert step.impl == "t_checkpoints"
    assert step.runner == "local"
    assert step.metrics == {"resumed": False}
    assert step.gpu_tier is None
    assert step.duration_s >= 0
    assert (workdir.root / step.log_path).exists()
    # ...and it is on disk, which is what A7 reads into job_step.
    written = json.loads(workdir.step_path("step").read_text())
    assert written["stageId"] == "step"
    assert written["artifacts"][0]["name"] == "state.json"
    assert written["checkpointKey"] == "runs/run/step/checkpoint"


def test_an_artifact_is_addressed_by_name_not_by_path(tmp_path: Path) -> None:
    workdir = seeded_workdir(tmp_path / "run")
    from recipe import load_recipe

    plan = plan_recipe(load_recipe("splat-ingest"))

    # A later stage refers to an earlier stage's output by artifact name; the executor is
    # the only thing that knows where it landed.
    package = next(stage for stage in plan.stages if stage.id == "package")
    assert dict(package.inputs) == {
        "canonical.ply": "stages/normalize/out/canonical.ply",
        "georef.json": "stages/georeference/out/georef.json",
    }
    assert plan.stages[0].inputs == {"upload": "inputs/upload"}
    assert workdir.input_path("upload").is_dir()


def test_a_stage_cannot_write_an_output_it_did_not_declare(tmp_path: Path) -> None:
    from errors import StageContractError

    workdir = seeded_workdir(tmp_path / "run", upload=False)
    recipe = make_recipe([{"id": "step", "impl": "t_checkpoints"}], inputs=[])
    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))

    planned = plan_recipe(recipe).stages[0]
    context = LocalRunner()._context(planned, workdir, 1)
    try:
        context.output("not-declared.json")
    except StageContractError as error:
        assert "does not declare in `produces`" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a StageContractError")
