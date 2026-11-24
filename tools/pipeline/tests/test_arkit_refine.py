"""Refinement must not turn an ARKit provenance tag into an unchecked metric claim."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import arkit_refine
import capture_bundle
import sfm
import stages
from contracts import StageContext


def test_entity_placement_rejects_unverified_refinement(tmp_path: Path) -> None:
    (tmp_path / "poses.json").write_text(
        json.dumps(
            {
                "source": "hex-capture-v1",
                "scale": {"source": "arkit"},
                "refinement": {"status": "accepted", "sim3": {"scale": 1.04}},
            }
        )
    )
    ctx = SimpleNamespace(has_input=lambda name: name == "poses", input=lambda _: tmp_path)
    assert (
        stages.entity_placement(cast(StageContext, ctx), {})["arkitTrainingFrameVerified"] is False
    )


def evidence() -> dict[str, Any]:
    return {
        "status": "accepted",
        "reprojectionBeforePx": 1.5,
        "reprojectionAfterPx": 0.9,
        "sim3": {"scale": 1.001},
        "translationCm": {"median": 1, "p95": 2, "max": 3},
        "rotationDeg": {"median": 0.5, "p95": 0.8, "max": 1},
    }


@pytest.mark.parametrize(
    "key,value,reason",
    [
        ("reprojectionAfterPx", 1.5, "did not improve"),
        ("sim3", {"scale": 1.01}, "scale"),
        ("gauge", {"scaleChecks": [{"scale": 1.01}]}, "scale"),
        ("gauge", {"scaleChecks": [{"scale": float("nan")}]}, "non-finite"),
        ("translationCm", {"max": 15.01}, "correction"),
        ("rotationDeg", {"max": 5.01}, "correction"),
        ("reprojectionAfterPx", float("nan"), "non-finite"),
    ],
)
def test_guards_reject_worse_or_unbounded_refinement(key: str, value: Any, reason: str) -> None:
    report = evidence()
    report[key] = value
    assert reason in (arkit_refine.guard(report) or "")


def test_fixed_pose_check_remains_strict_after_fallback() -> None:
    r = np.diag([1.0, -1.0, -1.0])
    image = sfm.Image(
        1, "frame_00000000.png", tuple(capture_bundle.quaternion(r)), (0.01, 0, 0), 1, 0, 0
    )
    model = sfm.Model((), (image,), 1, 1)
    records = [{"index": 0, "transform": np.eye(4).ravel().tolist()}]
    for report in (None, {"status": "disabled"}, {"status": "fallback", "fallbackReason": "test"}):
        with pytest.raises(ValueError, match="changed a fixed ARKit pose"):
            arkit_refine.verify(model, records, report)


def test_recorded_sim3_accounts_for_changed_pose() -> None:
    r = np.diag([1.0, -1.0, -1.0])
    image = sfm.Image(
        1, "frame_00000000.png", tuple(capture_bundle.quaternion(r)), (0.01, 0, 0), 1, 0, 0
    )
    model = sfm.Model((), (image,), 1, 1)
    records = [{"index": 0, "transform": np.eye(4).ravel().tolist()}]
    report = evidence()
    report["sim3"] = {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0, 0, 0]}
    report["corrections"] = {
        image.name: {
            "rawCentre": image.centre.tolist(),
            "rawRotation": r.tolist(),
            "arkitCentre": [0, 0, 0],
            "arkitRotation": r.tolist(),
        }
    }
    arkit_refine.verify(model, records, report)
    bad = copy.deepcopy(report)
    bad["corrections"][image.name]["rawCentre"][0] += 0.001
    with pytest.raises(ValueError, match="recorded Sim3"):
        arkit_refine.verify(model, records, bad)


def test_degenerate_trajectory_cannot_claim_gravity() -> None:
    line = np.column_stack([np.arange(4), np.zeros((4, 2))]).astype(np.float64)
    with pytest.raises(ValueError, match="degenerate"):
        arkit_refine.align(line, line)


def test_synthetic_bundle_cpu_ba_and_real_gsplat_parser(tmp_path: Path) -> None:
    interpreter = os.environ.get("ARKIT_TEST_PYTHON") or os.environ.get("COLMAP_GLOBAL_PYTHON")
    examples = os.environ.get("ARKIT_TEST_GSPLAT_EXAMPLES")
    if not interpreter or not examples:
        pytest.skip("needs pycolmap 4.2 CPU interpreter and pinned gsplat examples")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).parents[1]),
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "HEAVY_CPUS": "4",
    }
    completed = subprocess.run(
        [
            "heavy",
            interpreter,
            str(Path(__file__).with_name("arkit_refine_scene.py")),
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    # Bench preparation runs in pipeline Python 3.12, distinct from trainer Python 3.10.
    import distortion
    from experiments.bench.scene import prepare_pool, seal

    frozen = tmp_path / "frozen"
    seal(frozen, "test-arkit", "local", "root", "fixture")
    split = json.loads((frozen / "split.json").read_text())
    pool = tmp_path / "pool"
    prepare_pool(frozen, pool, split)
    assert distortion.enabled(pool / "poses")
    summary = json.loads((pool / "poses" / "poses.json").read_text())
    assert arkit_refine.verified_summary(summary)
    assert set(summary["refinement"]["corrections"]) == set(split["pool"])


def test_recipe_defaults_are_bundle_only() -> None:
    from recipe import load_recipe

    recipe = load_recipe(Path(__file__).parents[1] / "recipes/photo-reconstruct.yaml")
    pose = next(s for s in recipe.stages if s.id == "pose")
    assert pose.params["arkit_refine"] is True
    assert pose.params["arkit_shared_lens"] is True


@pytest.mark.parametrize("lens,seed", [(False, 120), (True, 121)])
def test_short_baseline_gauge_cpu(tmp_path: Path, lens: bool, seed: int) -> None:
    interpreter = os.environ.get("ARKIT_TEST_PYTHON") or os.environ.get("COLMAP_GLOBAL_PYTHON")
    if not interpreter:
        pytest.skip("needs pycolmap 4.2 CPU interpreter")
    completed = subprocess.run(
        [
            "heavy",
            interpreter,
            str(Path(__file__).with_name("arkit_gauge_scene.py")),
            str(tmp_path),
            "lens" if lens else "pinhole",
            str(seed),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[1]),
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "HEAVY_CPUS": "4",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
