from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import arkit_depth
import depth_trainer
import resume_trainer


def test_split_is_parser_lexical_order() -> None:
    assert arkit_depth.training_names(["d", "a", "c", "b"], 2) == ["b", "d"]
    assert arkit_depth.training_names(["d", "a"], 0) == ["a", "d"]
    with pytest.raises(ValueError):
        arkit_depth.training_names(["a"], 1)


def test_nonbundle_options_are_inert(tmp_path: Path) -> None:
    assert arkit_depth.options(tmp_path, {"arkit_depth_loss": True}) == (False, False)
    assert arkit_depth.loss_args(tmp_path, {"arkit_depth_loss": True}) == []
    (tmp_path / arkit_depth.DIRECTORY).mkdir()
    (tmp_path / arkit_depth.DIRECTORY / "manifest.json").write_text("{}")
    assert arkit_depth.options(tmp_path, {}) == (True, False)
    assert arkit_depth.options(tmp_path, {"arkit_depth_init": False}) == (False, False)
    assert arkit_depth.loss_args(tmp_path, {"arkit_depth_loss": True}) == [
        "--arkit-depth-loss",
        "--arkit-depth-weight",
        "0.01",
        "--arkit-depth-warmup",
        "1000",
    ]
    with pytest.raises(ValueError):
        arkit_depth.loss_args(tmp_path, {"arkit_depth_loss": True, "arkit_depth_weight": -1})


def test_real_cpu_geometry_and_loss(tmp_path: Path) -> None:
    python = os.environ.get("ARKIT_TEST_PYTHON")
    examples = os.environ.get("ARKIT_TEST_GSPLAT_EXAMPLES")
    if not python or not examples:
        pytest.skip("set ARKIT_TEST_PYTHON and ARKIT_TEST_GSPLAT_EXAMPLES")
    source = Path(examples, "simple_trainer.py").read_text()
    patched = depth_trainer.patch_source(resume_trainer.patch_source(source))
    compile(patched, "depth-trainer", "exec")
    assert patched.count("_hex_resume.save(self") == 1
    with pytest.raises(ValueError, match="already patched"):
        depth_trainer.patch_source(patched)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    completed = subprocess.run(
        [python, str(Path(__file__).with_name("arkit_depth_scene.py")), str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    from experiments.bench.scene import prepare_pool

    prepare_pool(
        tmp_path / "scene", tmp_path / "pool", {"pool": [f"frame_{i:08d}.png" for i in (1, 3, 5)]}
    )
    assert sorted(
        p.name for p in (tmp_path / "pool" / "frames" / arkit_depth.DIRECTORY).glob("*.f16")
    ) == ["smooth_1.f16", "smooth_3.f16", "smooth_5.f16"]


def test_bench_arm_selection_without_loading_modal() -> None:
    import ast
    from typing import Any

    path = Path(__file__).resolve().parents[3] / "infra" / "modal" / "arkit_bench.py"
    tree = ast.parse(path.read_text())
    definitions: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"select_arms", "arm_params"}
    ]
    namespace: dict[str, Any] = {
        "Any": Any,
        "ARMS": ("v17", "pr1", "pr1+init", "pr1+init+depthloss"),
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["select_arms"]("pr1+init") == ("pr1+init",)
    with pytest.raises(ValueError):
        namespace["select_arms"]("pr1,pr1")
    with pytest.raises(ValueError):
        namespace["select_arms"]("unknown")
    for arm, init, loss in [
        ("v17", False, False),
        ("pr1", False, False),
        ("pr1+init", True, False),
        ("pr1+init+depthloss", True, True),
    ]:
        assert namespace["arm_params"](arm) == {
            "test_every": 8,
            "block_parallel": 1,
            "arkit_depth_init": init,
            "arkit_depth_loss": loss,
        }


def test_stage_dispatches_depth_settings_and_records_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from typing import Any

    from executor import execute
    from runners import LocalRunner, RunnerSet
    from test_train_gsplat import seed_inputs, stand_in_params, train_recipe
    from workdir import Workdir

    work = Workdir.create(tmp_path / "run")
    seed_inputs(work)
    payload = work.input_path("frames") / arkit_depth.DIRECTORY
    payload.mkdir()
    (payload / "manifest.json").write_text("{}")
    expected = {
        "seedPoints": 7,
        "priorFrames": 3,
        "confidence2Samples": 14,
        "initEnabled": True,
        "lossEnabled": True,
    }

    def prepare(
        frames: Path, poses: Path, dataset: Path, params: Any, *, cap_max: Any
    ) -> dict[str, Any]:
        assert frames == work.input_path("frames") and poses == work.input_path("poses")
        assert (dataset / "sparse" / "0" / "points3D.bin").read_bytes() == b"colmap"
        assert params["arkit_depth_loss"] is True
        return expected

    monkeypatch.setattr(arkit_depth, "prepare", prepare)
    execute(
        train_recipe(stand_in_params(arkit_depth_loss=True, live=False)),
        work,
        RunnerSet(cpu=LocalRunner()),
    )
    document = json.loads((work.out_dir("train") / "train_metrics.json").read_text())
    assert document["arkitDepth"] == expected
    assert (
        "--arkit-depth-loss --arkit-depth-weight 0.01 --arkit-depth-warmup 1000"
        in work.log_path("train").read_text()
    )
