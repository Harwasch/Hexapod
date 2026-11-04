"""Validated opt-in train settings and stable metric names; no GPU imports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

MAP_CODE = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
MAP_MODEL = "facebook/map-anything-apache"
MAP_REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
MOGE_CODE = "74fbce054ebed49800de42d0ad0e83495065719a"
MOGE_MODEL = "Ruicheng/moge-2-vitl-normal"
MOGE_REVISION = "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"
METRIC_NAMES = (
    "splatCount",
    "farFieldShare",
    "elongationShare",
    "lossTerms",
    "valPsnrCurve",
    "denseInitSeconds",
    "denseInitPoints",
    "denseInitAddedPoints",
)
RENDER_KEYS = (
    "with_ut",
    "with_eval3d",
    "rolling_shutter",
    "post_processing",
    "ppisp_use_controller",
    "ppisp_controller_distillation",
    "ppisp_controller_activation_num_steps",
)


def finite(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return float(value)


def options(params: Mapping[str, Any]) -> dict[str, Any]:
    init = params.get("init", "sfm")
    if init not in ("sfm", "sfm+dense"):
        raise ValueError("init must be sfm or sfm+dense")
    priors = params.get("priors") or {}
    aniso = params.get("aniso_cap") or {}
    if not isinstance(priors, dict) or not isinstance(aniso, dict):
        raise ValueError("priors and aniso_cap must be mappings")
    if set(priors) - {"depth_weight", "normal_weight", "gradient_weight", "start_step", "model"}:
        raise ValueError("unknown prior setting")
    if set(aniso) - {"ratio", "weight"}:
        raise ValueError("unknown anisotropy setting")
    if priors.get("model", "moge-2-vitl-normal") != "moge-2-vitl-normal":
        raise ValueError("only pinned moge-2-vitl-normal priors are supported")
    far = params.get("far_field", False)
    if not (isinstance(far, bool) or far == "sphere"):
        raise ValueError("far_field must be false or sphere")
    result = {
        "init": init,
        "depth_weight": finite(priors.get("depth_weight", 0), "depth_weight", 0, 10),
        "normal_weight": finite(priors.get("normal_weight", 0), "normal_weight", 0, 10),
        "gradient_weight": finite(priors.get("gradient_weight", 0), "gradient_weight", 0, 10),
        "start_step": finite(priors.get("start_step", 1000), "start_step", 0, 1_000_000),
        "aniso_weight": finite(aniso.get("weight", 0), "aniso weight", 0, 10),
        "aniso_ratio": finite(aniso.get("ratio", 10), "aniso ratio", 1, 1000),
        "far_field": bool(far),
        "far_field_share": finite(params.get("far_field_share", 0.1), "far_field_share", 0, 0.15),
        "far_field_count": finite(params.get("far_field_count", 256), "far_field_count", 4, 4096),
        "dense_chunk": finite(params.get("dense_chunk", 4), "dense_chunk", 3, 8),
    }
    for name in ("start_step", "far_field_count", "dense_chunk"):
        if not result[name].is_integer():
            raise ValueError(f"{name} must be an integer")
        result[name] = int(result[name])
    result["priors"] = any(
        result[k] > 0 for k in ("depth_weight", "normal_weight", "gradient_weight")
    )
    result["enabled"] = init != "sfm" or result["priors"] or result["aniso_weight"] > 0 or bool(far)
    if result["priors"] and params.get("depth_loss"):
        raise ValueError(
            "mono priors and depth_loss are incompatible: "
            "upstream replaces the depth image with sparse samples"
        )
    if result["enabled"]:
        if params.get("batch_size", 1) != 1:
            raise ValueError("quality-stack extensions require batch_size=1")
        if params.get("strategy", "default") != "mcmc":
            raise ValueError("quality-stack extensions require strategy=mcmc")
        if params.get("app_opt") or params.get("sparse_grad") or params.get("visible_adam"):
            raise ValueError("quality-stack extensions require SH colours and standard dense Adam")
        if params.get("init_from", "sfm") != "sfm" or params.get("patch_size"):
            raise ValueError(
                "quality-stack extensions do not accept external init or random patches"
            )
    return result


def prepare(
    ctx: Any,
    dataset: Any,
    trainer: Any,
    cap: int | None,
    settings: dict[str, Any],
    data_factor: int,
) -> tuple[Any, int | None, dict[str, Any]]:
    """Stage the prepass and guarded source patch; disabled returns the original path."""
    import json
    import shutil
    from pathlib import Path

    import gaussian_budget
    import quality_trainer
    import training

    output_cache = ctx.output("prior_cache")
    output_cache.mkdir(parents=True, exist_ok=True)
    (output_cache / "status.json").write_text(
        json.dumps({"enabled": settings["priors"], "revision": MOGE_REVISION})
    )
    if not settings["enabled"]:
        return trainer, cap, {}
    cache = ctx.checkpoint_dir / "prior_cache"
    cache.mkdir(parents=True, exist_ok=True)
    if ctx.has_input("prior_cache"):
        shutil.copytree(ctx.input("prior_cache"), cache, dirs_exist_ok=True)
    foreground, count = (
        gaussian_budget.far_allocation(
            cap, settings["far_field_share"], settings["far_field_count"]
        )
        if settings["far_field"]
        else (cap, 0)
    )
    config = {
        **settings,
        "dataset": str(dataset),
        "trainer": str(trainer),
        "cache": str(cache),
        "data_factor": data_factor,
        "emit_metrics": ctx.param("live") is not False,
        "ring": str(ctx.work_dir / "far.npz"),
        "far_count": count,
        "report": str(ctx.work_dir / "prepass.json"),
        "loss_report": str(ctx.work_dir / "loss_terms.json"),
        "dense_max": max(0, int(foreground or 100_000) // 4),
        "synthetic": trainer.name == "gsplat_stand_in.py",
        "require_dense": bool(ctx.param("_bench_require_dense", False)),
    }
    path = ctx.work_dir / "quality_settings.json"
    path.write_text(json.dumps(config, indent=2))
    ctx.run(
        [
            training.trainer_python(ctx.param("python")),
            str(Path(__file__).with_name("quality_prepass.py")),
            str(path),
        ]
    )
    report = dict(json.loads(Path(config["report"]).read_text()))
    if (
        config["require_dense"]
        and not config["synthetic"]
        and report.get("denseInitAddedPoints", 0) <= 0
    ):
        raise ValueError("bench dense init added zero points; refusing to score an SfM fallback")
    config.update(center=report["center"], radius=report["radius"])
    path.write_text(json.dumps(config, indent=2))
    shutil.copytree(cache, output_cache, dirs_exist_ok=True)
    if config["synthetic"]:
        # The stand-in accepts an explicit settings argument; never pretend its numbers
        # are gradients or real model inference. It still exercises export and budgets.
        return trainer, foreground, report
    patched = ctx.work_dir / "quality_simple_trainer.py"
    patched.write_text(quality_trainer.patch_source(trainer.read_text(), path, trainer))
    return patched, foreground, report


def metrics(columns: Any, curve: Any, workdir: Any, prepass: dict[str, Any]) -> dict[str, Any]:
    import json

    import numpy as np

    count = len(columns["x"])
    axes = np.stack([columns[f"scale_{i}"] for i in range(3)], 1)
    loss_path = workdir / "loss_terms.json"
    losses = json.loads(loss_path.read_text()).get("lossTerms", {}) if loss_path.exists() else {}
    return {
        "splatCount": count,
        "farFieldShare": float(np.asarray(columns.get("far_field", np.zeros(count))).mean())
        if count
        else 0.0,
        "elongationShare": float((np.ptp(axes, axis=1) > np.log(10)).mean()) if count else 0.0,
        "lossTerms": losses,
        "valPsnrCurve": curve,
        **{
            key: prepass.get(key, 0)
            for key in ("denseInitSeconds", "denseInitPoints", "denseInitAddedPoints")
        },
    }


def ply_metrics(path: Any) -> dict[str, Any]:
    """Merged block outputs use the same vocabulary, without loading a large PLY whole."""
    import numpy as np

    import splat_io

    count = elongated = distant = 0
    for _, columns in splat_io.SplatReader(path).chunks():
        count += len(columns["x"])
        axes = np.stack([columns[f"scale_{i}"] for i in range(3)], 1)
        elongated += int((np.ptp(axes, axis=1) > np.log(10)).sum())
        if "far_field" in columns:
            distant += int((columns["far_field"] >= 0.5).sum())
    return {
        "splatCount": count,
        "farFieldShare": distant / max(1, count),
        "elongationShare": elongated / max(1, count),
        "lossTerms": {},
        "valPsnrCurve": [],
        "denseInitSeconds": 0.0,
        "denseInitPoints": 0,
        "denseInitAddedPoints": 0,
    }
