"""Stopping a run when held-out quality has gone flat: the rule, and the hook that applies it.

**Nothing here trains anything.** The rule (`convergence.py`) is plain arithmetic over a
curve and is tested as that. The hook (`converge_trainer.py`) is tested against
`loop_stand_in/simple_trainer.py`, a stand-in with the *shape* of gsplat v1.5.3's training
loop -- step lists re-read every step, save then export then evaluate, `main` handed to
`gsplat.distributed.cli` -- and a scripted PSNR curve; it is evidence that the hook stops
a loop of that shape where the rule says and leaves the files where the stage reads
them, not that a real run converges where this one does. The strings the hook depends
on in the real trainer were checked against v1.5.3's file at the tag, and the image build
checks them again (`converge_trainer.py --self-check`).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

import converge_trainer
import convergence
import training
from executor import execute
from runners import LocalRunner, RunnerSet
from test_train_gsplat import seed_inputs, stand_in_params, train_recipe
from workdir import Workdir

LOOP_STAND_IN = Path(__file__).resolve().parent / "loop_stand_in" / "simple_trainer.py"


# --- the rule -----------------------------------------------------------------------------


def test_evaluations_are_a_coarse_curve_then_every_step_of_the_tail() -> None:
    steps = convergence.eval_steps(30_000, 25_000, every=500, curve_every=5_000)

    assert steps[:5] == [5_000, 10_000, 15_000, 20_000, 25_000]
    assert steps[-1] == 29_500  # the final one is always requested separately
    assert steps[5:] == list(range(25_500, 30_000, 500))


def test_no_evaluations_when_densification_runs_to_the_end() -> None:
    """A schedule shorter than densification has no tail to stop, so none are paid for."""
    assert convergence.eval_steps(300, 25_000) == []
    assert convergence.eval_steps(25_000, 25_000) == []


def curve(pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """(steps completed, psnr) as the zero-based (step, psnr) pairs `decide` reads."""
    return [(step - 1, psnr) for step, psnr in pairs]


def test_nothing_stops_while_densifying_however_flat_it_is() -> None:
    flat = curve([(5_000, 24.0), (10_000, 24.0), (15_000, 24.0), (20_000, 24.0)])

    decision = convergence.decide(flat, start=25_000, window=2_000, min_gain_db=0.05)

    assert (decision.stop, decision.reason) == (False, "densifying")


def test_a_full_window_after_densification_is_needed_before_anything_is_judged() -> None:
    tail = curve([(25_000, 24.0), (25_500, 24.0), (26_000, 24.0), (26_500, 24.0)])

    decision = convergence.decide(tail, start=25_000, window=2_000, min_gain_db=0.05)

    assert (decision.stop, decision.reason) == (False, "warming up")


def test_a_flat_tail_stops_and_a_rising_one_does_not() -> None:
    flat = curve([(25_000 + 500 * i, 24.0 + 0.005 * i) for i in range(5)])
    rising = curve([(25_000 + 500 * i, 24.0 + 0.05 * i) for i in range(5)])

    stop = convergence.decide(flat, start=25_000, window=2_000, min_gain_db=0.05)
    go = convergence.decide(rising, start=25_000, window=2_000, min_gain_db=0.05)

    assert (stop.stop, stop.reason, stop.gain_db) == (True, "converged", 0.02)
    assert (go.stop, go.reason, go.gain_db) == (False, "improving", 0.2)


def test_one_lucky_evaluation_before_the_window_does_not_make_a_plateau_look_worse() -> None:
    """Best against best: a spike before the window counts as the level to beat."""
    spiky = curve([(25_000, 24.3), (25_500, 24.0), (26_000, 24.1), (26_500, 24.2), (27_000, 24.25)])

    decision = convergence.decide(spiky, start=25_000, window=2_000, min_gain_db=0.05)

    assert decision.stop and decision.gain_db == pytest.approx(-0.05)


def test_a_diverged_evaluation_never_reads_as_converged() -> None:
    broken = curve([(25_000 + 500 * i, math.nan) for i in range(5)])

    decision = convergence.decide(broken, start=25_000, window=2_000, min_gain_db=0.05)

    assert not decision.stop


@pytest.mark.parametrize(
    "rule", [{"every": 0}, {"every": 500, "window": 100}, {"min_gain_db": -1.0}]
)
def test_a_rule_that_could_never_judge_is_refused(rule: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="converge"):
        convergence.Rule(**rule)  # type: ignore[arg-type]


def test_the_curve_is_read_in_step_order_as_steps_completed(tmp_path: Path) -> None:
    stats = tmp_path / "stats"
    stats.mkdir()
    for step, psnr in ((999, 21.0), (29999, 25.0), (9999, 23.0)):
        (stats / f"val_step{step:04d}.json").write_text(
            json.dumps({"psnr": psnr, "ssim": 0.8, "lpips": 0.2, "num_GS": 7})
        )
    (stats / "train_step29999_rank0.json").write_text("{}")
    (stats / "val_step1000.json").write_text("not json")

    points = convergence.read_curve(stats)

    assert [point["step"] for point in points] == [1_000, 10_000, 30_000]
    assert points[-1] == {"step": 30_000, "psnr": 25.0, "ssim": 0.8, "lpips": 0.2, "gaussians": 7}


def test_the_wrapper_takes_the_trainers_place_and_passes_its_argv_on_unchanged() -> None:
    argv = training.gsplat_argv(
        "python3", Path("/t/simple_trainer.py"), Path("/d"), Path("/r"), strategy="mcmc"
    )

    wrapped = training.converge_argv(
        argv, script=Path("/p/converge_trainer.py"), every=500, window=2_000, min_gain_db=0.05
    )

    assert wrapped[:4] == ["python3", "/p/converge_trainer.py", "--trainer", "/t/simple_trainer.py"]
    assert wrapped[wrapped.index("--") + 1 :] == argv[2:]
    assert wrapped[wrapped.index("--window") + 1] == "2000"


def test_extra_evaluations_follow_the_final_one_in_the_argv() -> None:
    argv = training.gsplat_argv(
        "p", Path("t"), Path("d"), Path("r"), max_steps=30_000, eval_steps=[25_000, 25_500, 30_000]
    )

    at = argv.index("--eval_steps")
    assert argv[at + 1 : at + 4] == ["30000", "25000", "25500"]  # the final step never twice


# --- the hook, against a loop of v1.5.3's shape --------------------------------------------


def converge_params(**overrides: object) -> dict[str, object]:
    """A 1,500-step maximum (30k x 0.05) whose densification ends at 1,250: the recipe's
    shape at a twentieth of the length, so the tail is 250 steps with an evaluation every
    25 and a 100-step window."""
    params = stand_in_params(
        trainer=str(LOOP_STAND_IN),
        iterations=30_000,
        schedule_scale=0.05,
        strategy="mcmc",
        cap_max=64,
        converge=True,
        live=False,
        extra_args=[],
    )
    params.update(overrides)
    return params


def run(tmp_path: Path, params: dict[str, object]) -> tuple[Workdir, dict[str, Any]]:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)
    execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))
    document = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    return workdir, document


def test_a_flat_tail_is_stopped_by_the_trainers_own_save(tmp_path: Path) -> None:
    workdir, document = run(tmp_path, converge_params(extra_args=["--plateau-at", "0.84"]))

    convergence_ = document["convergence"]
    assert convergence_["status"] == "on"
    assert convergence_["stepsMax"] == 1_500
    assert convergence_["refineStopIter"] == 1_250
    assert convergence_["stoppedEarly"] is True
    # Densification ends at 1,250; the first full 100-step window after it is judged at
    # 1,350, which is flat; the trainer saves, exports and evaluates at the next step.
    assert convergence_["stepsRun"] == 1_351
    assert document["iterations"] == 1_351
    hook = convergence_["hook"]
    assert hook["hooked"] is True and hook["decidedAtStep"] == 1_350
    assert "held-out PSNR gained" in str(convergence_["reason"])
    steps = [point["step"] for point in convergence_["curve"]]
    assert steps[0] == 250 and 1_250 in steps and steps[-1] == 1_351
    # The trainer's own PLY at the stopping step became trained.ply; the renders are gone.
    result = workdir.work_dir("train") / "gsplat"
    assert training.latest_ply(result) == result / "ply" / "point_cloud_1350.ply"
    assert (workdir.out_dir("train") / "trained.ply").is_file()
    assert not any((result / "renders").iterdir())
    step = json.loads(workdir.step_path("train").read_text())
    assert step["metrics"]["stoppedEarly"] is True
    assert step["metrics"]["stepsRun"] == 1_351
    assert step["metrics"]["stepsMax"] == 1_500


def test_a_batch_stops_at_the_same_point_of_its_images_in_half_the_steps(
    tmp_path: Path,
) -> None:
    """Two images a step: `--steps_scaler` is halved, and with it -- by the trainer's own
    `adjust_steps` -- densification's end and every evaluation step, while the wrapper
    multiplies its window by the same scaler. The unscaled `--eval_steps` the stage passes
    are the ones a batch of one gets; the stop lands at half the steps, where the same
    number of images have been trained on."""
    workdir, document = run(
        tmp_path, converge_params(batch_size=2, extra_args=["--plateau-at", "0.84"])
    )
    single_dir, single = run(tmp_path / "one", converge_params(extra_args=["--plateau-at", "0.84"]))

    batched, one = document["convergence"], single["convergence"]
    assert batched["stepsMax"] == 750 and one["stepsMax"] == 1_500
    assert batched["refineStopIter"] == 625 and one["refineStopIter"] == 1_250
    assert batched["stoppedEarly"] is True
    assert batched["hook"]["windowSteps"] == one["hook"]["windowSteps"] // 2
    assert abs(2 * batched["stepsRun"] - one["stepsRun"]) <= 30  # one evaluation apart
    log = workdir.log_path("train").read_text()
    evals = log.split("--eval_steps ")[1].split(" --")[0]
    single_log = single_dir.log_path("train").read_text()
    assert evals == single_log.split("--eval_steps ")[1].split(" --")[0]
    assert "--steps_scaler 0.025" in log and "--batch_size 2" in log


def test_a_rising_tail_runs_its_whole_schedule(tmp_path: Path) -> None:
    _workdir, document = run(tmp_path, converge_params())

    convergence_ = document["convergence"]
    assert convergence_["stoppedEarly"] is False
    assert convergence_["stepsRun"] == convergence_["stepsMax"] == 1_500
    assert convergence_["reason"] == "ran the full schedule (last decision: improving)"


def test_without_gsplat_the_wrapper_runs_the_trainer_whole_and_says_why(tmp_path: Path) -> None:
    """The plain stand-in has no `gsplat.distributed` to hook: a full run, not a failure."""
    params = converge_params(
        trainer=str(Path(__file__).resolve().parent / "gsplat_stand_in.py"),
        extra_args=["--ckpt-every", "500", "--gaussians", "64"],
    )

    _workdir, document = run(tmp_path, params)

    convergence_ = document["convergence"]
    assert convergence_["stoppedEarly"] is False
    assert convergence_["stepsRun"] == 1_500
    assert "gsplat.distributed does not import" in str(convergence_["reason"])


def test_a_schedule_shorter_than_densification_is_not_wrapped(tmp_path: Path) -> None:
    workdir, document = run(tmp_path, stand_in_params(strategy="mcmc", cap_max=64, converge=True))

    convergence_ = document["convergence"]
    assert str(convergence_["status"]).startswith("not applied: densification runs to step")
    assert convergence_["hook"] is None
    assert "converge_trainer.py" not in workdir.log_path("train").read_text()


def test_convergence_is_off_unless_asked_for(tmp_path: Path) -> None:
    workdir, document = run(tmp_path, converge_params(converge=None))

    convergence_ = document["convergence"]
    assert (convergence_["enabled"], convergence_["status"]) == (False, "off")
    assert convergence_["stepsRun"] == 1_500
    assert "converge_trainer.py" not in workdir.log_path("train").read_text()


def test_a_big_budget_lengthens_the_maximum_and_only_with_convergence(tmp_path: Path) -> None:
    """2M gaussians: a maximum 1.41x as long (`gaussian_budget.schedule_factor`)."""
    params = converge_params(cap_max=2_000_000, extra_args=["--plateau-at", "0.5"])

    _workdir, document = run(tmp_path, params)
    _other, plain = run(tmp_path / "plain", {**params, "converge": False})

    assert document["settings"]["scheduleFactor"] == 1.41
    assert document["convergence"]["stepsMax"] == training.scaled_steps(30_000, 0.0705)
    assert document["convergence"]["stoppedEarly"] is True
    assert plain["settings"]["scheduleFactor"] == 1.0
    assert plain["requestedIterations"] == 1_500


def test_the_self_check_accepts_a_loop_of_the_shape_it_hooks_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the image build runs against the real trainer; a loop that stopped
    re-reading one of its step lists is refused by name."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        assert converge_trainer.main(["--trainer", str(LOOP_STAND_IN), "--self-check"]) == 0
        reshaped = tmp_path / "simple_trainer.py"
        reshaped.write_text(LOOP_STAND_IN.read_text().replace("cfg.eval_steps]", "eval_at]"))
        assert converge_trainer.main(["--trainer", str(reshaped), "--self-check"]) == 1
        assert "no longer has" in capsys.readouterr().err
    finally:
        sys.modules.pop("gsplat.distributed", None)
        sys.modules.pop("gsplat", None)
