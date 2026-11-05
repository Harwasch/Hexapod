"""Container orchestration exercised on CPU with the stand-in trainer/evaluator."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from experiments.bench import worker
from experiments.bench.rehearsal import stand_in_params, synthetic_scene
from experiments.bench.scene import write_json


@pytest.mark.parametrize("fail_training", [False, True])
@pytest.mark.parametrize("fail_topdown", [False, True])
@pytest.mark.parametrize("fade", [False, True])
def test_training_sampler_scope_and_render_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_training: bool,
    fail_topdown: bool,
    fade: bool,
) -> None:
    import executor

    scene, run = tmp_path / "scene", tmp_path / "run"
    synthetic_scene(scene, "spool", "test")
    params = stand_in_params()
    if fade:
        params["_bench_quality"] = {"bar": "fade", "context_fade": 0.5}
    monkeypatch.setenv("GSPLAT_TRAINER", params["trainer"])
    monkeypatch.setenv("GSPLAT_PYTHON", sys.executable)
    active = False
    evaluated = False
    evaluation_commands = []

    class Sampler:
        samples = ["60, 1024", "80, 2048"]

        def __init__(self, executable: str | None) -> None:
            pass

        def start(self) -> None:
            nonlocal active
            active = True

        def stop(self) -> None:
            nonlocal active
            active = False

    monkeypatch.setattr(worker, "GpuSampler", Sampler)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    real_run = subprocess.run
    real_execute = executor.execute

    def execute(*args: Any, **kwargs: Any) -> Any:
        assert active
        if fail_training:
            raise RuntimeError("trainer failed")
        return real_execute(*args, **kwargs)

    def run_command(argv: list[str], **kwargs: Any) -> Any:
        nonlocal evaluated
        if len(argv) > 1 and Path(argv[1]).name == "evaluate.py":
            assert not active
            assert argv[argv.index("--renders") + 1] == "3"
            before_gate = Path(argv[argv.index("--out") + 1]).name == "test-trained.json"
            assert argv[argv.index("--render-dir") + 1] == str(
                run / ("renders-trained" if before_gate else "renders")
            )
            assert Path(argv[argv.index("--raw-ply") + 1]).is_file()
            assert (run / "full_sh.ply").is_file()
            write_json(Path(argv[argv.index("--out") + 1]), {"psnr": 21 if before_gate else 20})
            evaluation_commands.append(argv)
            evaluated = True
            return subprocess.CompletedProcess(argv, 0)
        if len(argv) > 1 and Path(argv[1]).name == "topdown.py":
            assert evaluated and not active
            assert 0 < kwargs["timeout"] <= 60
            if fail_topdown:
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            from PIL import Image

            Image.new("RGB", (2, 2)).save(Path(argv[argv.index("--out") + 1]))
            return subprocess.CompletedProcess(argv, 0)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(executor, "execute", execute)
    monkeypatch.setattr(subprocess, "run", run_command)
    plan = {
        "runId": "one",
        "scene": "spool",
        "seed": 42,
        "schedule": "screen",
        "variant": {"name": "baseline", "overrides": {}},
        "tier": "l4",
        "benchCommit": "test",
        "timeoutSeconds": 1800,
        "trainParams": params,
        "renders": 3,
        "topdown": True,
        "keepSplat": True,
    }
    result = worker.run_training(scene, run, plan)
    assert not active
    assert result["status"] == ("failed" if fail_training else "complete"), result["error"]
    assert evaluated == (not fail_training)
    assert result["gpuUtilMean"] == 70 and result["gpuMemMaxMiB"] == 2048
    assert result["artifacts"]["gpu_samples.json"]["samples"] == Sampler.samples
    if not fail_training:
        assert len(evaluation_commands) == (2 if fade else 1)
        assert result["test"]["psnr"] == 20
        assert result["testTrained"]["psnr"] == (21 if fade else 20)
        if fade:
            assert result["testFaded"]["psnr"] == 20
            assert result["headlinePly"] == "faded.ply"
            assert (run / "trained.ply").is_file() and (run / "faded.ply").is_file()
            neutral = []
            for argv in evaluation_commands:
                copied = argv.copy()
                for flag in ("--ply", "--out", "--render-dir"):
                    copied[copied.index(flag) + 1] = "different-output"
                neutral.append(copied)
            assert neutral[0] == neutral[1]  # same data, split, degree and renderer
        assert result["topdown"]["status"] == ("failed" if fail_topdown else "complete")
        if not fail_topdown:
            assert result["topdown"]["pngBase64"]
            assert (run / "renders/topdown.png").is_file()
