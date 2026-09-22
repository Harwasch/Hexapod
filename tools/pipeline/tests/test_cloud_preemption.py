"""Kill a stage mid-run and prove it *resumes* rather than restarts.

The assertion this file exists for is not "the stage ran twice". It is that the second
attempt did **five** iterations of a ten-iteration stage, because the first attempt had
already done five and they were not done again. Both attempts are cut off at the same
point, so ten iterations are only ever reached by continuing:

    attempt 1   five iterations, checkpointed; the machine is taken away
    attempt 2   resumes at five, does five more, finishes at ten

Disable the checkpoint restore and attempt 2 starts from zero, runs into the same cutoff
and is preempted again -- so the negative test below is not a mock's opinion, it is the
same run with one call removed. `test_without_the_restore_it_never_finishes` is that run.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from adapters import FakeAdapter, LocalTransfer, RemoteExecution
from artifacts import ArtifactDecl
from cloud import AttemptLedger, CloudRunner, Placement
from conftest import make_recipe, seeded_workdir
from contracts import StageContext, StageOutcome
from errors import PreemptedError
from executor import execute
from providers import Rate
from recipe import Recipe
from registry import stage_impl
from runners import RunnerSet
from workdir import Workdir

MODEL = ArtifactDecl("model.json", content_type="application/json")

#: How far the stage has to get. Both attempts are cut off after six polls, which is five
#: iterations -- so ten is reachable only by the second attempt continuing the first.
ITERATIONS = 10
CUTOFF_POLLS = 7


@stage_impl("t_trains", produces=(MODEL,), summary="a long GPU stage that checkpoints")
def t_trains(ctx: StageContext) -> StageOutcome:
    """Registered so the recipe plans and `produces` is verified.

    It never runs under `CloudRunner`: the work happens on the provider, and for
    `FakeAdapter` the provider is `training` below. `SubprocessAdapter`'s tests are where
    a registered implementation really is executed through the cloud seam.
    """
    raise AssertionError("t_trains must run on the provider, not here")


def training(run: RemoteExecution) -> Iterator[None]:
    """What the GPU box does: iterate, checkpointing after every iteration.

    It reads where it got to out of `checkpoint/`, which is the directory the adapter
    fills from object storage before it starts and syncs back to object storage while it
    runs. A stage that is killed loses only what it did since the last sync.
    """
    target = int(run.request.params["iterations"])
    progress = run.checkpoint_dir / "progress.json"
    resumed_at = (
        json.loads(progress.read_text(encoding="utf-8"))["iteration"] if progress.is_file() else 0
    )
    run.log(f"training: starting at iteration {resumed_at} of {target}")
    done = resumed_at
    while done < target:
        yield  # one unit of work
        done += 1
        progress.write_text(json.dumps({"iteration": done}), encoding="utf-8")
    run.log(f"training: reached {done}")
    run.output(MODEL.name).write_text(
        json.dumps({"iterations": done, "thisAttempt": done - resumed_at, "resumedAt": resumed_at}),
        encoding="utf-8",
    )


def one_gpu_stage() -> Recipe:
    return make_recipe(
        [
            {
                "id": "train",
                "impl": "t_trains",
                "params": {"iterations": ITERATIONS},
                "gpu": {"tier": "a100", "preemptible": True},
            }
        ],
        inputs=[],
    )


def cloud(tmp_path: Path, **overrides: object) -> tuple[CloudRunner, FakeAdapter]:
    """A CloudRunner over a FakeAdapter that always dies at the same iteration."""
    transfer = LocalTransfer(tmp_path / "bucket")
    settings: dict[str, object] = {
        "script": training,
        "preempt_after_polls": CUTOFF_POLLS,
        "seconds_per_tick": 1.0,
        "rates": {"a100": Rate(1.19, "a test")},
    }
    settings.update(overrides)
    adapter = FakeAdapter(transfer, tmp_path / "sandbox", **settings)  # type: ignore[arg-type]
    runner = CloudRunner(
        Placement((adapter,)),
        transfer,
        poll_interval_s=0.0,
        checkpoint_every_s=1.0,
        sleep=lambda _seconds: None,
    )
    return runner, adapter


def attempt(workdir: Workdir, runner: CloudRunner, number: int) -> None:
    execute(one_gpu_stage(), workdir, RunnerSet.cloud(runner), attempts={"train": number})


def test_a_preempted_stage_resumes_and_does_not_repeat_the_work(tmp_path: Path) -> None:
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    runner, adapter = cloud(tmp_path)

    with pytest.raises(PreemptedError) as preemption:
        attempt(workdir, runner, 1)

    # The machine went away mid-training. What came home is what the last interval sync
    # put in object storage -- real state, in the one directory `prepare_stage` keeps.
    checkpoint = workdir.checkpoint_dir("train") / "progress.json"
    assert json.loads(checkpoint.read_text())["iteration"] == 5
    assert not workdir.step_path("train").exists()
    assert preemption.value.attempt == 1

    attempt(workdir, runner, 2)

    model = json.loads((workdir.out_dir("train") / MODEL.name).read_text())
    # The assertion the whole step is for: ten iterations, of which the second attempt
    # did five. The other five were not done again.
    assert model == {"iterations": ITERATIONS, "thisAttempt": 5, "resumedAt": 5}
    assert json.loads(checkpoint.read_text())["iteration"] == ITERATIONS
    # And the second attempt really was cut from the same cloth as the first: the same
    # adapter, the same cutoff. It finished because it had less left to do.
    assert [request.attempt for request in adapter.submitted] == [1, 2]


def test_without_the_restore_it_never_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same run with `_restore_checkpoint` removed. It does not finish.

    This is what makes the test above an assertion about resuming rather than about
    running twice: take away the one call that brings the remote's checkpoint home and
    the second attempt starts at zero, hits the same cutoff, and is preempted again.
    """
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    runner, _ = cloud(tmp_path)
    monkeypatch.setattr(CloudRunner, "_restore_checkpoint", lambda _self, _context: 0)

    with pytest.raises(PreemptedError):
        attempt(workdir, runner, 1)
    assert not any(workdir.checkpoint_dir("train").iterdir())

    with pytest.raises(PreemptedError):
        attempt(workdir, runner, 2)

    assert not (workdir.out_dir("train") / MODEL.name).exists()


def test_every_attempt_is_billed_including_the_one_that_was_taken_away(tmp_path: Path) -> None:
    """A cost that counted only the attempt that succeeded would hide half the bill."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    runner, _ = cloud(tmp_path)

    with pytest.raises(PreemptedError):
        attempt(workdir, runner, 1)
    attempt(workdir, runner, 2)

    ledger = AttemptLedger.read(workdir.attempts_path("train"))
    assert [(entry.attempt, entry.state) for entry in ledger.entries] == [
        (1, "preempted"),
        (2, "succeeded"),
    ]
    # Six simulated seconds each: five iterations plus the poll that found it gone.
    assert [entry.billed_s for entry in ledger.entries] == [6.0, 6.0]
    assert ledger.preemptions == 1
    assert ledger.usd == pytest.approx(2 * 1.19 * 6.0 / 3600.0, abs=2e-6)
    # The successful step's metrics carry the stage's totals, not just its own attempt.
    step = json.loads(workdir.step_path("train").read_text())
    assert step["metrics"]["billedS"] == 6.0
    assert step["metrics"]["stageBilledS"] == 12.0
    assert step["metrics"]["preemptions"] == 1
    assert step["metrics"]["provider"] == "fake"


def test_the_checkpoint_interval_decides_how_much_is_lost(tmp_path: Path) -> None:
    """Sync every five simulated seconds instead of every one and four iterations are
    thrown away rather than none -- which is the trade that interval is."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = FakeAdapter(
        transfer,
        tmp_path / "sandbox",
        script=training,
        preempt_after_polls=CUTOFF_POLLS,
        seconds_per_tick=1.0,
    )
    runner = CloudRunner(
        Placement((adapter,)),
        transfer,
        poll_interval_s=0.0,
        checkpoint_every_s=5.0,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(PreemptedError):
        execute(one_gpu_stage(), workdir, RunnerSet.cloud(runner), attempts={"train": 1})

    progress = workdir.checkpoint_dir("train") / "progress.json"
    assert json.loads(progress.read_text())["iteration"] == 4
