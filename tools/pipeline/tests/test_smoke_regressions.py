"""Failures exposed by the first Beach GPU smoke; no GPU needed locally."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_quality_support import _run, _seed

import splat_io
from workdir import Workdir


def test_fade_stage_with_real_ply_far_field_column(tmp_path: Path) -> None:
    work = Workdir.create(tmp_path / "work")
    seeded = _seed(work, held=True)
    columns = seeded["columns"]
    columns["far_field"] = np.zeros(len(columns["x"]), dtype=np.float32)
    columns["far_field"][-1] = 1
    path = work.input_path("trained.ply")
    with splat_io.PlyWriter(path, tuple(columns)) as writer:
        writer.append(columns)
    out = _run(work, seeded["inputs"], {"bar": "fade", "context_fade": 0.5})
    gated = splat_io.SplatReader(out / "gated.ply")
    assert gated.count == len(columns["x"])
    assert "far_field" in gated.properties
    exported = gated.read(0, gated.count)
    np.testing.assert_array_equal(exported["far_field"], columns["far_field"])
    assert exported["opacity"][-1] == columns["opacity"][-1]
    assert json.loads((out / "quality.json").read_text())["barApplied"] == "fade"


def test_percentile_interpolation_uses_array_output_on_numpy_22(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import outofcore

    add = np.add

    def old_add(*args: Any, **kwargs: Any) -> Any:
        # The image pins NumPy 2.2.6: Ellipsis is not a valid ufunc output there.
        if kwargs.get("out") is Ellipsis:
            raise TypeError("return arrays must be of ArrayType")
        return add(*args, **kwargs)

    monkeypatch.setattr(np, "add", old_add)
    assert outofcore._lerp(np.float32(1), np.float32(3), 0.75, np.dtype("float32")) == 2.5


def test_ppisp_display_reference_is_explicit_and_requires_no_controller() -> None:
    import pytest

    import quality_stack

    params = {"strategy": "mcmc", "post_processing": "ppisp", "ppisp_display_referred": True}
    with pytest.raises(ValueError, match="controller"):
        quality_stack.options(params)
    result = quality_stack.options({**params, "ppisp_use_controller": False})
    assert result["enabled"] and result["ppisp_display_referred"]


def test_export_gap_warns_only_when_measured_and_over_three_db() -> None:
    import quality_stack

    measured = quality_stack.export_check(23.098206, {"status": "ok", "meanPsnr": 13.677})
    assert measured["exportMismatchDb"] > 9.4
    assert "WARNING" in measured["exportWarning"]
    assert quality_stack.export_check(23, {"status": "ok", "meanPsnr": 20})["exportWarning"] is None
    assert quality_stack.export_check(23, {"status": "failed"})["exportMismatchDb"] is None


def test_display_identity_parameters_and_setup() -> None:
    from types import SimpleNamespace

    import photometric_maths
    import quality_trainer

    raw = photometric_maths.identity_crf()
    np.testing.assert_allclose(np.logaddexp(0, raw[:3]) + np.array([0.3, 0.3, 0.1]), 1, atol=1e-12)

    class Parameter:
        def __init__(self) -> None:
            self.frozen = False
            self.zeroed = False

        def requires_grad_(self, enabled: bool) -> None:
            self.frozen = not enabled

        def zero_(self) -> None:
            self.zeroed = True

    # Real tensor copies/identity checks are also exercised by the image's CPU seam check.
    module = SimpleNamespace(crf_params=Parameter(), vignetting_params=Parameter())
    quality_trainer.freeze_display_reference(module)
    assert module.crf_params.frozen and module.vignetting_params.frozen


def test_dense_chunk_sixteen_is_an_explicit_candidate() -> None:
    import yaml

    import quality_stack

    variants = Path(__file__).resolve().parents[1] / "experiments/variants/quality-stack.yaml"
    params = yaml.safe_load(variants.read_text())["all-on"]["train"]
    assert params["budget_max"] == 3456214
    assert params["dense_chunk"] == 16
    assert quality_stack.options({"strategy": "mcmc", **params})["dense_chunk"] == 16


def test_diagnostic_differs_only_in_ppisp() -> None:
    import yaml

    directory = Path(__file__).resolve().parents[1] / "experiments/variants"
    main = yaml.safe_load((directory / "quality-stack.yaml").read_text())["all-on"]
    diagnostic = yaml.safe_load((directory / "quality-stack-diagnostic.yaml").read_text())
    expected = {
        **main,
        "train": {
            k: v
            for k, v in main["train"].items()
            if k not in {"post_processing", "ppisp_use_controller", "ppisp_display_referred"}
        },
    }
    assert diagnostic["all-on-no-ppisp"] == expected
    assert set(diagnostic) == {
        "all-on-no-ppisp",
        "upgraded-plain",
        "all-on-no-3dgut",
        "all-on-no-3dgut-native",
    }


def test_ppisp_setup_seam_is_after_module_construction() -> None:
    import quality_trainer

    trainer = Path.home() / ".cache/hexapod-upstream/gsplat/examples/simple_trainer.py"
    if not trainer.exists():
        import pytest

        pytest.skip("pinned upstream clone unavailable")
    source = quality_trainer.patch_source(
        trainer.read_text(), Path("unused.json"), trainer, display_referred=True
    )
    setup = source.index("_hex_quality.configure_photometry(self)")
    assert source.index("self.post_processing_module = PPISP(") < setup
    assert setup < source.index("        # Losses & Metrics.")


def test_bench_report_shouts_about_export_gap(tmp_path: Path) -> None:
    from experiments.bench.results import bakeoff_markdown, markdown, parse_result

    (tmp_path / "train_metrics.json").write_text(json.dumps({"psnr": 23.1}))
    (tmp_path / "holdout.json").write_text(json.dumps({"status": "ok", "meanPsnr": 13.68}))
    plan = {
        "runId": "smoke",
        "scene": "beach",
        "variant": {"name": "all-on"},
        "seed": 42,
        "schedule": "screen",
        "tier": "l4",
        "benchCommit": "test",
        "timeoutSeconds": 1800,
    }
    result = parse_result(
        plan,
        tmp_path,
        params={},
        split_digest=None,
        manifest_digest=None,
        wall=1,
        train_seconds=1,
        gpu_name="test",
    )
    assert result["exportMismatchDb"] > 9.4
    assert "WARNING" in bakeoff_markdown([result]) and "WARNING" in markdown([result])


def test_dense_oom_retries_smaller_without_losing_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    import quality_prepass
    import sfm

    path = tmp_path / "points3D.bin"
    sfm.write_points3d(
        path,
        sfm.Points3D(
            ids=np.arange(1, 9, dtype=np.uint64),
            xyz=np.random.default_rng(1).normal(size=(8, 3)),
            rgb=np.full((8, 3), 100, dtype=np.uint8),
            error=np.zeros(8),
            track=np.empty((0, 2), dtype=np.uint32),
            track_offsets=np.zeros(9, dtype=np.int64),
        ),
    )
    model = SimpleNamespace()
    model.to = lambda *a: model
    model.eval = lambda: model
    monkeypatch.setenv("HEXAPOD_MAP_MODEL", "unused")
    monkeypatch.setitem(
        sys.modules,
        "mapanything.models",
        SimpleNamespace(MapAnything=SimpleNamespace(from_pretrained=lambda *a: model)),
    )
    monkeypatch.setitem(
        sys.modules, "mapanything.utils.image", SimpleNamespace(preprocess_inputs=None)
    )

    class SyntheticOOMError(Exception):
        pass

    cleared = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            OutOfMemoryError=SyntheticOOMError,
            cuda=SimpleNamespace(empty_cache=lambda: cleared.append(True)),
        ),
    )
    seen = set()

    def infer(items: list[int], *args: Any) -> tuple[Any, Any, dict[str, Any]]:
        if len(items) > 8:
            raise SyntheticOOMError("synthetic memory limit")
        seen.update(items)
        return np.array([[100 + items[0], 100, 100]]), np.array([[255, 0, 0]]), {}

    monkeypatch.setattr(quality_prepass, "infer_dense_chunk", infer)
    report = quality_prepass.dense(
        list(range(25)), None, path, {"dense_chunk": 16, "dense_max": 100}
    )
    assert seen == set(range(25)) and cleared
    assert report["memoryRetries"] and not report["skippedChunks"]
    assert report["denseInitAddedPoints"] > 0


def test_dense_peak_is_recorded_even_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    import quality_prepass
    import quality_stack
    import sfm

    model = tmp_path / "dataset/sparse/0"
    model.mkdir(parents=True)
    sfm.write_points3d(
        model / "points3D.bin",
        sfm.Points3D(
            ids=np.arange(1, 9, dtype=np.uint64),
            xyz=np.random.default_rng(1).normal(size=(8, 3)),
            rgb=np.full((8, 3), 100, dtype=np.uint8),
            error=np.zeros(8),
            track=np.empty((0, 2), dtype=np.uint32),
            track_offsets=np.zeros(9, dtype=np.int64),
        ),
    )
    events = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                empty_cache=lambda: events.append("empty"),
                reset_peak_memory_stats=lambda: events.append("reset"),
                synchronize=lambda: events.append("sync"),
                max_memory_allocated=lambda: 8e9,
                max_memory_reserved=lambda: 9e9,
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets.colmap",
        SimpleNamespace(Parser=lambda **kw: None, Dataset=lambda *a, **kw: []),
    )

    def fail(*args: Any) -> None:
        events.append("infer")
        raise ValueError("model failed")

    monkeypatch.setattr(quality_prepass, "dense", fail)
    report = quality_prepass.run(
        {
            **quality_stack.options({"strategy": "mcmc", "init": "sfm+dense"}),
            "dataset": str(tmp_path / "dataset"),
            "cache": str(tmp_path / "cache"),
            "trainer": str(tmp_path / "trainer.py"),
            "data_factor": 1,
        }
    )
    assert events == ["empty", "reset", "infer", "sync"]
    assert report["densePeakMemoryGb"] == 8 and report["densePeakReservedGb"] == 9
    assert report["denseInitAddedPoints"] == 0 and report["warnings"]
