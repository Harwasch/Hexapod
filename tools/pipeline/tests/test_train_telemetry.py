"""Only stand-ins: observations must not change loop work, errors, or RNG state."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_convergence import converge_params, run

import converge_trainer
import live
import train_telemetry


def test_loop_records_scaled_plan_samples_evaluations_and_stop(tmp_path: Path) -> None:
    workdir, _ = run(tmp_path, converge_params(extra_args=["--plateau-at", "0.84"]))
    text = workdir.log_path("train").read_text()
    records = live.records(text)
    points = [p for kind, p in records if kind == "train"]
    assert points[0]["k"] == "p"
    assert points[0]["maxSteps"] == 1500
    assert points[0]["refineStop"] == 1250
    assert points[0]["rule"] == {"every": 25, "window": 100, "minGainDb": 0.05}
    samples = [p for p in points if p["k"] == "s"]
    assert [p["step"] for p in samples] == [1, 200, 400, 600, 800, 1000, 1200, 1351]
    assert samples[-1]["loss"] == {"total": 0.1, "l1": 0.07, "ssim": 0.03, "depth": 0.01}
    assert samples[-1]["gs"] == 64 and samples[-1]["lr"] > 0
    evaluations = [p for p in points if p["k"] == "e"]
    assert evaluations[-1]["step"] == 1351 and evaluations[-1]["psnr"] > 20
    assert evaluations[-1]["decision"] == "converged"
    assert points[-1]["k"] == "x" and points[-1]["step"] == 1351
    for line in text.splitlines():
        if train_telemetry.TAG in line:
            assert len(line[line.index(train_telemetry.TAG) :].encode()) < 512
    plan = next(p for kind, p in records if kind == "plan")
    assert plan["stepsMax"] == 1500 and plan["cap"] == 64
    assert plan["metresPerUnit"] is None and plan["usdPerHour"] is None


def test_no_attributes_and_broken_log_sink_do_not_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    class Runner:
        def train(self) -> int:
            return 42

    timing = converge_trainer.Timing(None)
    before = random.getstate()
    train_telemetry.install(Runner, object(), timing)
    assert Runner().train() == 42
    monkeypatch.setattr("builtins.print", lambda *a, **kw: (_ for _ in ()).throw(OSError("sink")))
    assert Runner().train() == 42
    assert random.getstate() == before


def test_optimizer_return_errors_rng_and_eval_export_time_are_preserved(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    cuda = SimpleNamespace(max_memory_allocated=lambda: 2 * 1024**3)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("telemetry drew from torch RNG")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=cuda, rand=forbidden, randn=forbidden, manual_seed=forbidden),
    )
    timing = converge_trainer.Timing(None)

    class Optimizer:
        param_groups = [{"lr": 0.01}]

        def step(self) -> int:
            clock[0] += 0.5
            return 17

    class Runner:
        optimizers = {"means": Optimizer()}

        def train(self) -> str:
            timing.train_entered = clock[0]
            for _ in range(200):
                assert self.optimizers["means"].step() == 17
                clock[0] += 2
                timing.eval_in_train_s += 1
                timing.ply_s += 1
            return "done"

    train_telemetry.install(Runner, SimpleNamespace(max_steps=200), timing)
    assert Runner().train() == "done"
    records = live.records(capsys.readouterr().out)
    sample = [p for _, p in records if p["k"] == "s"][-1]
    assert sample["its"] == 2.0 and sample["mem"] == 2
    assert sample["loss"] is None and sample["gs"] is None


def test_parser_retains_order_and_rejects_non_finite_or_malformed_records() -> None:
    rows = [{"v": 1, "k": "s", "step": step, "loss": {"future": 0.1}} for step in (1, 200)]
    text = "\n".join(live.TRAIN_TAG + json.dumps(row) for row in rows)
    assert [p for _, p in live.records(text)] == rows
    assert live.latest(text)["train"] == rows[-1]
    for row in (
        {"v": 1, "k": "s", "step": -1, "loss": {}},
        {"v": 1, "k": "e", "step": 1, "psnr": float("nan")},
    ):
        assert live.parse_line(live.TRAIN_TAG + json.dumps(row)) is None


def test_block_wrapper_observes_evals_with_the_rule_off(tmp_path: Path) -> None:
    import subprocess

    from test_convergence import LOOP_STAND_IN

    wrapper = Path(train_telemetry.__file__).with_name("block_trainer.py")
    result = subprocess.run(
        [
            sys.executable,
            str(wrapper),
            "--trainer",
            str(LOOP_STAND_IN),
            "--",
            "mcmc",
            "--data_dir",
            str(tmp_path),
            "--result_dir",
            str(tmp_path / "result"),
            "--steps_scaler",
            "0.01",
            "--eval_steps",
            "10000",
            "30000",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    points = [p for kind, p in live.records(result.stdout) if kind == "train"]
    assert points[0]["rule"] is None
    assert [p["step"] for p in points if p["k"] == "e"] == [100, 300]
    assert all(p["decision"] is None for p in points if p["k"] == "e")
    assert not any(p["k"] == "x" for p in points)


def test_plan_metric_scale_and_adapter_rate_are_explicit(tmp_path: Path) -> None:
    from typing import cast

    import train_plan
    from contracts import StageContext

    georef = tmp_path / "georef.json"
    georef.write_text(json.dumps({"scaleSource": "exif-aligned", "frame": {"scale": 2.5}}))
    lines: list[str] = []
    params = {"_live_usd_per_hour": 1.2, "_live_rate_tier": "l4"}
    ctx = cast(
        "StageContext",
        SimpleNamespace(
            has_input=lambda name: True,
            input=lambda name: georef,
            attempt=2,
            gpu_tier="l4",
            param=lambda name: params.get(name),
            log=lines.append,
        ),
    )
    train_plan.emit(
        ctx,
        budget=None,
        cap=1000000,
        steps=60000,
        factor=2,
        scaler=2,
        strategy="mcmc",
        rule=None,
        frames=40,
        sparse=200,
    )
    plan = live.latest(lines[-1])["plan"]
    assert plan["metresPerUnit"] == 2.5 and plan["usdPerHour"] == 1.2
    assert plan["refineStop"] == 50000 and plan["frames"] == 40
    georef.write_text(json.dumps({"scaleSource": "unresolved", "frame": {"scale": 1}}))
    train_plan.emit(
        ctx, budget=None, cap=None, steps=300, factor=1, scaler=0.01, strategy="mcmc", rule=None
    )
    assert live.latest(lines[-1])["plan"]["metresPerUnit"] is None
