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


@pytest.mark.parametrize("enabled", [False, True])
def test_quality_samples_share_completed_step_cadence_and_cannot_break_training(
    enabled: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import quality_trainer

    observed: list[int] = []

    def sample(runner: Any, step: int) -> dict[str, Any]:
        observed.append(step)
        assert bool(getattr(runner, "hex_settings", {})) == enabled
        if step == 200:
            raise OSError("optional observation failed")
        return {"step": step, "splatCount": 4, "farFieldShare": 0.25, "elongationShare": 0.5}

    monkeypatch.setattr(quality_trainer, "live_metrics", sample, raising=False)

    class Optimizer:
        def step(self) -> int:
            return 17

    class Runner:
        def __init__(self) -> None:
            self.optimizers = {"means": Optimizer()}
            if enabled:
                self.hex_settings = {"emit_metrics": True}

        def eval(self, index: int, stage: str = "val") -> None:
            pass

        def train(self) -> str:
            for _ in range(401):
                assert self.optimizers["means"].step() == 17
            return "done"

    converge_trainer.instrument({}, Runner, converge_trainer.Timing(None), SimpleNamespace())
    assert Runner().train() == "done"
    records = live.records(capsys.readouterr().out)
    assert observed == [1, 200, 400, 401]
    assert [p["step"] for k, p in records if k == "metrics"] == [1, 400, 401]
    assert [p["step"] for k, p in records if k == "train" and p["k"] == "s"] == observed


def test_stage_without_stop_rule_still_installs_live_hooks(tmp_path: Path) -> None:
    workdir, _ = run(tmp_path, converge_params(converge=False))
    records = live.records(workdir.log_path("train").read_text())
    assert any(k == "train" and p["k"] == "s" for k, p in records)


@pytest.mark.parametrize("enabled", [False, True])
def test_live_quality_math_with_current_parameters_without_torch(
    enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NumPy tensor stand-in: actual reductions, not a simulated CUDA run."""
    import contextlib

    import numpy as np

    import quality_trainer

    class Tensor:
        def __init__(self, value: Any) -> None:
            self.value = np.asarray(value, dtype=float)

        def __len__(self) -> int:
            return len(self.value)

        def __sub__(self, other: Any) -> Any:
            return self.value - other

        def detach(self) -> Any:
            return self

        def max(self, axis: int) -> Any:
            return SimpleNamespace(values=self.value.max(axis))

        def min(self, axis: int) -> Any:
            return SimpleNamespace(values=self.value.min(axis))

        def new_tensor(self, value: Any) -> Any:
            return np.asarray(value)

    def tensor(value: Any) -> Any:
        return Tensor(value)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            no_grad=contextlib.nullcontext,
            linalg=SimpleNamespace(vector_norm=lambda x, dim: np.linalg.norm(x, axis=dim)),
            cuda=SimpleNamespace(
                max_memory_allocated=lambda: 2e9,
                max_memory_reserved=lambda: 3e9,
                memory_allocated=lambda: 1e9,
                memory_reserved=lambda: 2e9,
            ),
        ),
    )
    runner = SimpleNamespace(
        splats={
            "means": tensor([[0, 0, 0], [100, 0, 0]]),
            "scales": tensor([[0, 0, 0], [1000, 0, -1000]]),
        },
        parser=SimpleNamespace(points=np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])),
    )
    if enabled:
        runner.hex_settings = {"center": [0, 0, 0], "radius": 1, "far_count": 2}
    payload = quality_trainer.live_metrics(runner, 200)
    count = 4 if enabled else 2
    assert payload["step"] == 200 and payload["splatCount"] == count
    assert payload["farFieldShare"] == (3 if enabled else 1) / count
    assert payload["elongationShare"] == 1 / count
    assert payload["trainMemoryCurve"][0]["peakAllocatedGb"] == 2
    assert live.parse_line("live-metrics: " + json.dumps(payload)) == ("metrics", payload)
    # Densification replaces Parameters; the next sample must read the new tensors.
    runner.splats = {"means": tensor([[0, 0, 0]]), "scales": tensor([[0, 0, 0]])}
    assert quality_trainer.live_metrics(runner, 400)["elongationShare"] == 0
