"""Paid-trainer failures must retain the diagnostic, on CPU as well as on Modal."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from contracts import StageContext
from errors import StageFailedError
from experiments.bench import __main__ as cli
from experiments.bench import worker
from experiments.bench.rehearsal import synthetic_scene
from workdir import Workdir


def failed_command(work: Workdir) -> None:
    work.prepare_stage("train")
    context = StageContext(
        recipe="test",
        run_id="test",
        stage_id="train",
        impl="gsplat",
        params={},
        attempt=1,
        gpu_tier=None,
        out_dir=work.out_dir("train"),
        work_dir=work.work_dir("train"),
        checkpoint_dir=work.checkpoint_dir("train"),
        log_path=work.log_path("train"),
        checkpoint_key="test/train",
        attempts_path=work.attempts_path("train"),
        _inputs={},
        _produces={},
    )
    script = work.work_dir("train") / "failed.py"
    script.write_text(
        "import sys\n"
        "for i in range(70): print(f'output line {i}', flush=True)\n"
        "print('CUDA out of memory: diagnostic sentinel', file=sys.stderr, flush=True)\n"
        "sys.exit(1)\n"
    )
    try:
        context.run([sys.executable, str(script)])
    except subprocess.CalledProcessError as exc:
        raise StageFailedError("test", "train", "gsplat", exc) from exc


def test_failed_command_keeps_last_forty_lines_and_full_stage_log(tmp_path: Path) -> None:
    work = Workdir.create(tmp_path / "work")
    with pytest.raises(StageFailedError) as failure:
        failed_command(work)
    message = str(failure.value)
    assert "CUDA out of memory: diagnostic sentinel" in message
    assert "output line 31\n" in message
    assert "output line 30\n" not in message
    assert "output line 0\n" in work.log_path("train").read_text()


def test_bench_failure_copies_whole_log_and_memory_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import executor

    def execute(recipe: Any, work: Workdir, runners: Any) -> None:
        work.prepare_stage("train")
        (work.work_dir("train") / "train_memory.json").write_text(
            json.dumps({"samples": [{"step": 1000, "peakAllocatedGb": 6.5}]})
        )
        failed_command(work)

    monkeypatch.setattr(executor, "execute", execute)
    monkeypatch.setattr(cli, "git_commit", lambda: "test")
    plan = cli.plans_for(cli.parser().parse_args(["run", "--scene", "beach", "--rehearse"]))[0]
    plan["timeoutSeconds"] = 1800
    scene, destination = tmp_path / "scene", tmp_path / "run"
    synthetic_scene(scene, "beach", "test")
    result = worker.run_training(scene, destination, plan)
    assert result["status"] == "failed"
    assert "CUDA out of memory: diagnostic sentinel" in result["error"]
    saved = json.loads((destination / "result.json").read_text())
    assert saved["error"] == result["error"]
    assert "output line 0\n" in (destination / "trainer.log").read_text()
    assert "diagnostic sentinel" in (destination / "trainer.log").read_text()
    assert result["artifacts"]["train_memory.json"]["samples"][0]["step"] == 1000


def test_ut_only_is_refused_before_paid_work() -> None:
    from experiments.bench.recipe import validate_renderer

    with pytest.raises(ValueError, match=r"projection.*gradient"):
        validate_renderer({"strategy": "mcmc", "with_ut": True, "with_eval3d": False})


def test_training_step_bound_preserves_camera_rays_and_evaluation_size() -> None:
    import numpy as np

    import prior_maths
    from experiments.bench.recipe import train_params

    # Pixel-centre convention: shrinking an image by two maps source 0.5 to 0.25.
    h, w, sx, sy = prior_maths.resize_geometry(900, 1600, 800)
    assert (h, w) == (450, 800)
    intrinsic = np.array([[1200.0, 0, 800], [0, 1200, 450], [0, 0, 1]])
    resized = intrinsic * np.array([sx, sy, 1.0])[:, None]
    source = np.array([1100.5, 320.5, 1])
    np.testing.assert_allclose(
        np.linalg.solve(resized, source * [sx, sy, 1]), np.linalg.solve(intrinsic, source)
    )
    assert prior_maths.resize_geometry(900, 1600, 0) == (900, 1600, 1.0, 1.0)
    assert prior_maths.resize_geometry(499, 887, 400)[:2] == (225, 400)
    params = train_params("full", {"train": {"step_max_side": 800}})
    assert params["step_max_side"] == 800 and params.get("train_max_side") is None


def test_memory_samples_are_persistent_scalar_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextlib
    from types import SimpleNamespace

    import quality_trainer

    cuda = SimpleNamespace(
        max_memory_allocated=lambda: 12_000_000_000,
        max_memory_reserved=lambda: 14_000_000_000,
        memory_allocated=lambda: 7_000_000_000,
        memory_reserved=lambda: 13_000_000_000,
    )
    torch = SimpleNamespace(cuda=cuda, no_grad=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)
    cfg = {"loss_report": str(tmp_path / "loss_terms.json"), "emit_metrics": False}
    runner = SimpleNamespace(
        hex_settings=cfg, cfg=SimpleNamespace(max_steps=2100), splats={"means": [1, 2]}
    )
    info = {"width": 800, "height": 450, "isect_ids": SimpleNamespace(numel=lambda: 5000)}
    quality_trainer.record_memory(runner, 999, info)
    quality_trainer.record_memory(runner, 1000, info)  # no per-step history growth
    quality_trainer.record_memory(runner, 1999, info)
    report = json.loads((tmp_path / "train_memory.json").read_text())
    assert [x["step"] for x in report["samples"]] == [1000, 2000]
    assert report["samples"][-1]["peakAllocatedGb"] == 12
    assert report["samples"][-1]["allocatedGb"] == 7
    assert report["samples"][-1]["tileIntersections"] == 5000
    assert report["samples"][-1]["splatCount"] == 2
    assert not hasattr(runner, "hex_memory_samples")  # report lives on disk, not GPU


def test_elongation_metric_is_scale_ratio_not_variance_ratio(tmp_path: Path) -> None:
    import numpy as np

    import quality_stack

    columns = {
        "x": np.zeros(4),
        "scale_0": np.log([1.0, 10.0, 11.0, 100.0]),
        "scale_1": np.zeros(4),
        "scale_2": np.zeros(4),
    }
    result = quality_stack.metrics(columns, [], tmp_path, {})
    assert result["elongationShare"] == 0.5  # strict >10, not squared variances


def test_previous_intersections_are_released_before_the_next_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import weakref
    from types import SimpleNamespace

    import quality_trainer

    class Info:
        pass

    previous = None
    calls = 0

    class Stage:
        def render(self, **kwargs: Any) -> tuple[None, None, Info]:
            nonlocal previous, calls
            assert previous is None or previous() is None, "previous intersections are still alive"
            info = Info()
            previous = weakref.ref(info)
            calls += 1
            return None, None, info

    monkeypatch.setattr(quality_trainer, "prepare", lambda *a: None)
    monkeypatch.setattr(quality_trainer, "step_inputs", lambda runner, p, k, m: (p, k, m))
    monkeypatch.setattr(quality_trainer, "loss", lambda *a: a[-1])
    monkeypatch.setattr(quality_trainer, "record_memory", lambda *a: None)
    monkeypatch.setattr(quality_trainer, "guard_scales", lambda *a: None)
    path = tmp_path / "settings.json"
    path.write_text('{"priors": false}')
    source = """class Runner:
    def __init__(self):
        self.stage = Stage()
    def train(self):
        pixels = source_pixels
        Ks = masks = None
        for step in range(2):
            height, width = pixels.shape[1:3]
            # forward
            renders, alphas, info = self.stage.render(
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            )
            loss = total
            loss.backward()
            # eval the full set
"""
    namespace: dict[str, Any] = {
        "Stage": Stage,
        "cfg": SimpleNamespace(depth_loss=False),
        "source_pixels": SimpleNamespace(shape=(1, 4, 4, 3)),
        "Ks": None,
        "masks": None,
        "data": None,
        "depths": None,
        "total": SimpleNamespace(backward=lambda: None),
    }
    patched = quality_trainer.patch_source(source, path, tmp_path / "trainer.py")
    # The smoke seam test executes a complete two-step loop, not a string assertion.
    exec(compile(patched, "test-trainer", "exec"), namespace)
    namespace["Runner"]().train()
    assert calls == 2


def test_step_bound_off_is_identity_and_sparse_pixel_loss_is_refused() -> None:
    from types import SimpleNamespace

    import numpy as np

    import quality_stack
    import quality_trainer

    pixels, intrinsic, masks = np.zeros((1, 90, 160, 3)), np.eye(3)[None], np.ones((1, 90, 160))
    actual = quality_trainer.step_inputs(SimpleNamespace(hex_settings={}), pixels, intrinsic, masks)
    assert all(a is b for a, b in zip(actual, (pixels, intrinsic, masks), strict=True))
    for options in (
        {"depth_loss": True, "step_max_side": 800},
        {"step_max_side": 0.5},
        {"priors": {"max_side": 16}},
    ):
        with pytest.raises(ValueError):
            quality_stack.options({"strategy": "mcmc", **options})


def test_diagnostic_step_and_prior_bounds_are_explicit() -> None:
    import yaml

    from experiments.bench.recipe import PIPELINE, train_params

    arms = yaml.safe_load(
        (PIPELINE / "experiments/variants/quality-stack-diagnostic.yaml").read_text()
    )
    params = train_params("full", arms["all-on-no-3dgut"])
    assert params["step_max_side"] == 800
    assert params["priors"]["max_side"] == 400
    assert params["budget_max"] == 3456214
    assert params.get("train_max_side") is None  # test/holdout remain native 900x1600
    assert params["with_ut"] is False and params["post_processing"] == "ppisp"
