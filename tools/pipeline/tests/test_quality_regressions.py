"""Regressions from the pre-GPU cross review; local tests require no torch."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import prior_maths
import quality_stack
import quality_trainer

PIPELINE = Path(__file__).resolve().parents[1]
UPSTREAM = Path.home() / ".cache/hexapod-upstream/gsplat/examples"


def test_patched_bootstrap_imports_real_upstream_exif(tmp_path: Path) -> None:
    if not UPSTREAM.exists():
        pytest.skip("pinned upstream clone is unavailable")
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    source = """from exif import compute_exposure_from_exif
class Runner:
    def __init__(self):
        self.stage = Stage()
    def train(self):
        render_mode="RGB+ED" if cfg.depth_loss else "RGB",
        if True:
            loss.backward()
"""
    patched = tmp_path / "patched.py"
    patched.write_text(
        quality_trainer.patch_source(source, settings, UPSTREAM / "simple_trainer.py")
    )
    # Only the optional dependency is stubbed; execute the real upstream exif.py,
    # our real bootstrap and its real helper imports in a fresh interpreter.
    script = (
        "import sys, types, runpy; "
        "sys.modules['piexif'] = types.ModuleType('piexif'); "
        f"runpy.run_path({str(patched)!r}, run_name='seam_check')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_image_check_executes_patch_not_only_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    trainer = tmp_path / "simple_trainer.py"
    trainer.write_text(
        "raise ImportError('image dependency is missing')\n"
        "class Runner:\n"
        "    def __init__(self):\n"
        "        self.stage = Stage()\n"
        "    def train(self):\n"
        '        render_mode="RGB+ED" if cfg.depth_loss else "RGB",\n'
        "        if True:\n"
        "            loss.backward()\n"
    )
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    # A compile-only check would pass this source. The real build seam must execute it.
    with pytest.raises(ImportError, match="image dependency is missing"):
        quality_trainer.import_patched_trainer(trainer, settings)


@pytest.mark.parametrize("scale", [0.001, 1000.0])
def test_dense_alignment_accepts_arbitrary_colmap_units(scale: float) -> None:
    points = np.random.default_rng(42).normal(size=(20, 3))
    mapped, report = prior_maths.aligned_chunk(points, points, scale * points + 2, 5 * scale)
    np.testing.assert_allclose(mapped, scale * points + 2, atol=1e-12)
    assert report["inliers"] == 20


def test_priors_refuse_sparse_depth_before_prepass() -> None:
    with pytest.raises(ValueError, match="depth_loss"):
        quality_stack.options(
            {"strategy": "mcmc", "depth_loss": True, "priors": {"normal_weight": 0.1}}
        )


def test_b_does_not_freeze_gaussians_for_unused_controller() -> None:
    variants = yaml.safe_load((PIPELINE / "experiments/variants/quality-stack.yaml").read_text())
    assert variants["all-on"]["train"]["ppisp_use_controller"] is False


def test_depth_alignment_avoids_tall_gpu_least_squares() -> None:
    tree = ast.parse((PIPELINE / "quality_trainer.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "aligned_depth"
    )
    assert not any(isinstance(n, ast.Attribute) and n.attr == "lstsq" for n in ast.walk(function))


def test_full_timeout_override_controls_reservation_and_worker_budget() -> None:
    script = """import json
from experiments.bench import ledger
plan = {"runId": "timeout", "scene": "beach", "schedule": "full",
        "tier": "l4", "tag": "quality-stack"}
ledger.price_plan([], [plan], 3, 30)
assert plan["timeoutSeconds"] == 10800
seconds = 10800 + ledger.OVERHEAD_SECONDS + ledger.RESTART_SECONDS
assert plan["reservedUsd"] == ledger.hourly_rate("l4") * seconds / 3600
print(json.dumps(plan))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PIPELINE,
        env={**os.environ, "HEXAPOD_BENCH_FULL_TIMEOUT": "10800"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout)["reservedUsd"] > 2


def test_dense_keeps_good_chunk_when_next_chunk_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import quality_prepass
    import sfm

    path = tmp_path / "points3D.bin"
    sparse = sfm.Points3D(
        ids=np.arange(1, 9, dtype=np.uint64),
        xyz=np.random.default_rng(1).normal(size=(8, 3)),
        rgb=np.full((8, 3), 100, dtype=np.uint8),
        error=np.zeros(8),
        track=np.empty((0, 2), dtype=np.uint32),
        track_offsets=np.zeros(9, dtype=np.int64),
    )
    sfm.write_points3d(path, sparse)
    model = SimpleNamespace()
    model.to = lambda *a: model
    model.eval = lambda: model
    monkeypatch.setenv("HEXAPOD_MAP_MODEL", "unused")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "mapanything.models",
        SimpleNamespace(MapAnything=SimpleNamespace(from_pretrained=lambda *a: model)),
    )
    monkeypatch.setitem(
        sys.modules, "mapanything.utils.image", SimpleNamespace(preprocess_inputs=None)
    )
    calls = []

    def chunk(*args: Any) -> tuple[Any, Any, dict[str, Any]]:
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("bad chunk geometry")
        return np.array([[10.0, 10, 10]]), np.array([[255, 0, 0]], dtype=np.uint8), {"scale": 100.0}

    monkeypatch.setattr(quality_prepass, "infer_dense_chunk", chunk, raising=False)
    report = quality_prepass.dense(list(range(5)), None, path, {"dense_chunk": 3, "dense_max": 20})
    assert report["denseInitAddedPoints"] == 1 and len(sfm.read_points3d(path)) == 9
    assert len(report["skippedChunks"]) == 1
    assert "bad chunk geometry" in report["skippedChunks"][0]["reason"]


def test_bench_dense_cannot_silently_become_sfm() -> None:
    from experiments.bench.recipe import make_recipe, train_params

    params = train_params("full", {"train": {"init": "sfm+dense"}})
    recipe = make_recipe(params, "l4")
    assert recipe.stages[0].params["_bench_require_dense"] is True


def test_prior_storage_is_mappable_and_versioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior_cache

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    key = prior_maths.cache_key(frame, "model", {})
    monkeypatch.setattr(prior_maths, "CACHE_FORMAT", "future-format")
    assert prior_maths.cache_key(frame, "model", {}) != key
    values = {
        "depth": np.arange(20, dtype=np.float32).reshape(4, 5),
        "normal": np.zeros((4, 5, 3), dtype=np.float32),
        "mask": np.ones((4, 5), dtype=bool),
    }
    target = prior_cache.write(tmp_path, key, values)
    assert prior_cache.complete(target)
    loaded = quality_trainer.prior(str(target))
    for name, value in values.items():
        assert isinstance(loaded[name], np.memmap)
        np.testing.assert_array_equal(loaded[name], value)
    # Partial entries must be rebuilt; a marker alone is not a cache hit.
    (tmp_path / f"{key}.normal.npy").unlink()
    assert not prior_cache.complete(target)


def test_shared_cache_publishes_mmap_arrays_before_markers(tmp_path: Path) -> None:
    from experiments.bench.worker import publish_cache

    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    np.save(source / "key.depth.npy", np.ones((3, 4)))
    (source / "key.json").write_text('{"format": "npy-mmap-v2"}')
    publish_cache(source, target)
    assert (target / "key.depth.npy").is_file()


def test_ut_refuses_packed_instead_of_dying_in_cuda() -> None:
    import training

    with pytest.raises(ValueError, match="packed"):
        training.gsplat_argv(
            "python",
            Path("trainer"),
            Path("data"),
            Path("out"),
            strategy="mcmc",
            with_ut=True,
            packed=True,
        )


def test_ut_budget_does_not_claim_classic_memory_calibration() -> None:
    import gaussian_budget

    kwargs: dict[str, Any] = {
        "model": Path("missing"),
        "train_size": (1600, 900),
        "pixel_scale": 1.0,
        "with_ut": True,
        "with_eval3d": True,
    }
    with pytest.raises(ValueError, match="budget_max"):
        gaussian_budget.plan("auto", **kwargs)
    explicit = gaussian_budget.plan(500000, **kwargs)
    assert explicit is not None and explicit.memory_ceiling is None
    assert "uncalibrated" in explicit.reason
    bounded = gaussian_budget.plan("auto", budget_max=200000, **kwargs)
    assert bounded is not None and bounded.cap <= 200000


def test_weighted_affine_reductions_match_least_squares() -> None:
    x = np.linspace(0.1, 30, 501)
    y = 1.7 * x + 4 + 0.2 * np.sin(x)
    weights = np.linspace(0.01, 1, len(x))
    slope, offset, _ = prior_maths.weighted_affine(x, y, weights, np.finfo(float).eps)
    expected = np.linalg.lstsq(
        np.stack([x, np.ones_like(x)], 1) * np.sqrt(weights[:, None]),
        y * np.sqrt(weights),
        rcond=None,
    )[0]
    np.testing.assert_allclose([slope, offset], expected, rtol=1e-12, atol=1e-12)


def test_zero_dense_bench_fails_before_trainer_and_keeps_report(tmp_path: Path) -> None:
    from types import SimpleNamespace

    settings = quality_stack.options({"strategy": "mcmc", "init": "sfm+dense"})
    ctx = SimpleNamespace(
        work_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoint",
        output=lambda name: tmp_path / "output" / name,
        has_input=lambda name: False,
        param=lambda name, default=None: True if name == "_bench_require_dense" else default,
    )

    def run(argv: Any) -> None:
        (tmp_path / "prepass.json").write_text(
            json.dumps({"denseInitAddedPoints": 0, "warnings": ["all chunks failed"]})
        )

    ctx.run = run
    with pytest.raises(ValueError, match="zero points"):
        quality_stack.prepare(ctx, tmp_path, tmp_path / "real.py", 100, settings, 1)
    assert not (tmp_path / "quality_simple_trainer.py").exists()
    from experiments.bench.worker import ARTIFACTS

    assert "prepass.json" in ARTIFACTS
