"""The two seams A7 needed and A6 did not have: an observer, and skipping a stage.

Both exist so the worker can write a `job_step` row *as each stage runs* and retry from
the stage that failed rather than from the beginning. Neither lets the API into this
project: an observer is a callback, and `skip` is a set of stage ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifacts import ArtifactDecl
from conftest import make_recipe, seeded_workdir
from contracts import StageContext, StageOutcome, StepResult
from errors import ResumeError, StageFailedError
from executor import execute
from plan import PlannedStage
from recipe import Recipe
from registry import stage_impl
from runners import LocalRunner, RunnerSet

#: How many times each impl has actually executed, per test file run.
RUNS: dict[str, int] = {}


def _counting_impl(name: str) -> None:
    """Register `t_<name>`, producing `<name>.json` -- two stages may not produce the
    same artifact, so a three-stage recipe needs three impls."""
    artifact = f"{name}.json"

    @stage_impl(f"t_{name}", produces=(ArtifactDecl(artifact, content_type="application/json"),))
    def _counts(ctx: StageContext) -> StageOutcome:
        """Records how often it really ran, so a skipped stage is provably not re-run."""
        RUNS[ctx.stage_id] = RUNS.get(ctx.stage_id, 0) + 1
        ctx.output(artifact).write_text(
            json.dumps({"runs": RUNS[ctx.stage_id], "attempt": ctx.attempt}) + "\n",
            encoding="utf-8",
        )
        return StageOutcome(metrics={"runs": RUNS[ctx.stage_id]})


for _name in ("one", "two", "three"):
    _counting_impl(_name)


@stage_impl("t_explodes", produces=(ArtifactDecl("never.json"),))
def _explodes(ctx: StageContext) -> StageOutcome:
    raise RuntimeError("the GPU went away")


class Recorder:
    """A StageObserver that just writes down what it was told, in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, object]] = []

    def stage_started(self, stage: PlannedStage, attempt: int) -> None:
        self.events.append(("started", stage.id, attempt))

    def stage_finished(self, stage: PlannedStage, result: StepResult) -> None:
        self.events.append(("finished", stage.id, result.attempt))

    def stage_skipped(self, stage: PlannedStage, result: StepResult) -> None:
        self.events.append(("skipped", stage.id, result.attempt))

    def stage_failed(self, stage: PlannedStage, attempt: int, error: BaseException) -> None:
        self.events.append(("failed", stage.id, type(error).__name__))


def three_stages() -> Recipe:
    return make_recipe(
        [
            {"id": "one", "impl": "t_one"},
            {"id": "two", "impl": "t_two"},
            {"id": "three", "impl": "t_three"},
        ],
        inputs=[],
    )


def test_the_observer_is_told_before_the_stage_runs_not_after_the_run(tmp_path: Path) -> None:
    """The point of the seam: a `started` arrives before that stage's `finished`, and
    before the next stage exists at all. That is what makes the panel's stage list live
    instead of appearing in one batch at the end."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    recorder = Recorder()

    execute(three_stages(), workdir, RunnerSet.stubbed(), observer=recorder)

    assert [(kind, stage) for kind, stage, _ in recorder.events] == [
        ("started", "one"),
        ("finished", "one"),
        ("started", "two"),
        ("finished", "two"),
        ("started", "three"),
        ("finished", "three"),
    ]


def test_a_failing_stage_is_reported_and_the_run_stops_there(tmp_path: Path) -> None:
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    recorder = Recorder()
    recipe = make_recipe(
        [{"id": "one", "impl": "t_one"}, {"id": "boom", "impl": "t_explodes"}], inputs=[]
    )

    with pytest.raises(StageFailedError):
        execute(recipe, workdir, RunnerSet(cpu=LocalRunner()), observer=recorder)

    assert recorder.events[-1] == ("failed", "boom", "StageFailedError")


def test_an_observer_that_raises_stops_the_run_between_stages(tmp_path: Path) -> None:
    """How A7 notices a cancelled job: the observer refuses to start the next stage."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)

    class Stopper(Recorder):
        def stage_started(self, stage: PlannedStage, attempt: int) -> None:
            super().stage_started(stage, attempt)
            if stage.id == "two":
                raise KeyboardInterrupt("cancelled")

    stopper = Stopper()
    with pytest.raises(KeyboardInterrupt):
        execute(three_stages(), workdir, RunnerSet.stubbed(), observer=stopper)

    assert not workdir.step_path("three").exists()


def test_a_skipped_stage_is_not_re_run_and_its_artifacts_still_count(tmp_path: Path) -> None:
    """Retry from a stage: the earlier stages' outputs are still in the workdir, so the
    resumed run's `artifacts.json` says exactly what a fresh run's would."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    RUNS.clear()
    execute(three_stages(), workdir, RunnerSet(cpu=LocalRunner()))
    assert RUNS == {"one": 1, "two": 1, "three": 1}
    first = json.loads(workdir.manifest_path.read_text())

    recorder = Recorder()
    result = execute(
        three_stages(),
        workdir,
        RunnerSet(cpu=LocalRunner()),
        observer=recorder,
        skip={"one", "two"},
        attempts={"three": 2},
    )

    # Only the stage that was not skipped ran a second time, and it knows it is attempt 2.
    assert RUNS == {"one": 1, "two": 1, "three": 2}
    assert [(kind, stage) for kind, stage, _ in recorder.events] == [
        ("skipped", "one"),
        ("skipped", "two"),
        ("started", "three"),
        ("finished", "three"),
    ]
    assert [artifact.stage_id for artifact in result.artifacts] == ["one", "two", "three"]
    second = json.loads(workdir.manifest_path.read_text())
    assert [entry["name"] for entry in second["artifacts"]] == [
        entry["name"] for entry in first["artifacts"]
    ]
    assert json.loads((workdir.out_dir("three") / "three.json").read_text())["attempt"] == 2


def test_skipping_a_stage_with_no_previous_result_is_refused_before_anything_runs(
    tmp_path: Path,
) -> None:
    """A cleaned-up workdir gets a fresh run, not a half one."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    RUNS.clear()

    with pytest.raises(ResumeError, match="cannot skip stage 'one'"):
        execute(three_stages(), workdir, RunnerSet(cpu=LocalRunner()), skip={"one"})

    assert RUNS == {}
    assert not workdir.stages_dir.exists()


def test_a_step_result_survives_the_round_trip_through_json(tmp_path: Path) -> None:
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    execute(three_stages(), workdir, RunnerSet.stubbed())

    original = json.loads(workdir.step_path("two").read_text())
    assert StepResult.read(workdir.step_path("two")).to_dict() == original
