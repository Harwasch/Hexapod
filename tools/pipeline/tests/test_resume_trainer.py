"""Full checkpoint state and pinned source guards without torch, CUDA or network."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import resume_trainer

STAND_IN = Path(__file__).with_name("gsplat_stand_in.py")


def equal(left: Any, right: Any) -> None:
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, strict=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, list | tuple):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


def invoke(root: Path, *, strategy: str, die: int | None = None, resume: Path | None = None) -> Any:
    data = root.parent / "dataset"
    (data / "images").mkdir(parents=True, exist_ok=True)
    (data / "images/a.png").write_bytes(b"image")
    (data / "sparse/0").mkdir(parents=True, exist_ok=True)
    (data / "sparse/0/cameras.bin").write_bytes(b"poses")
    command = [
        sys.executable,
        str(STAND_IN),
        strategy,
        "--data_dir",
        str(data),
        "--result_dir",
        str(root / "out"),
        "--no-normalize-world-space",
        "--save_ply",
        "--max_steps",
        "19",
        "--checkpoint-every",
        "5",
        "--checkpoint-dir",
        str(root / "ckpt"),
    ]
    if die is not None:
        command += ["--die-at", str(die), "--sync-grace-s", "0"]
    if resume is not None:
        command += ["--resume", str(resume)]
    return subprocess.run(command, capture_output=True, text=True, timeout=15)


@pytest.mark.parametrize("strategy", ["mcmc", "default"])
@pytest.mark.parametrize("interrupted", [5, 8, 10, 14, 18])
def test_interrupt_resume_is_identical_including_adam_adc_rng_and_modules(
    tmp_path: Path, strategy: str, interrupted: int
) -> None:
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    first = invoke(whole, strategy=strategy)
    assert first.returncode == 0, first.stderr
    killed = invoke(resumed, strategy=strategy, die=interrupted)
    assert killed.returncode < 0, killed.stderr
    checkpoint = resume_trainer.newest(resumed / "ckpt")
    assert checkpoint is not None
    continued = invoke(resumed, strategy=strategy, resume=checkpoint)
    assert continued.returncode == 0, continued.stderr
    assert f"starting at step {interrupted // 5 * 5}" in continued.stdout
    whole_ckpt = resume_trainer.newest(whole / "ckpt")
    resumed_ckpt = resume_trainer.newest(resumed / "ckpt")
    assert whole_ckpt is not None and resumed_ckpt is not None
    left = pickle.loads(whole_ckpt.read_bytes())
    right = pickle.loads(resumed_ckpt.read_bytes())
    equal(left, right)
    assert left["step"] == 19
    assert len(left["splats"]["means"]) == 3  # Refined Parameter shape, not initialization.
    assert len(left["optimizers"]["app_optimizers/0"]["param_groups"]) == 2
    assert left["strategy"]["count"].sum() != 0
    assert set(left["modules"]) == set(resume_trainer.MODULES)
    assert len(resume_trainer.checkpoints(resumed / "ckpt")) == 2


def test_atomic_failure_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    from argparse import Namespace

    from resume_maths_stand_in import Model

    previous = sys.modules.get("torch")
    try:
        model = Model(
            Namespace(
                max_steps=20,
                checkpoint_every=5,
                checkpoint_dir=tmp_path,
                checkpoint_token="",
                resume=None,
                strategy="mcmc",
                result_dir=tmp_path / "out",
            )
        )
        for step in range(1, 6):
            model.tick(step)
        model.save(5)
        original = (tmp_path / "resume_000000005.pt").read_bytes()

        def broken(state: Any, handle: Any) -> Any:
            handle.write(b"partial")
            raise OSError("disk full")

        monkeypatch.setattr(model.torch, "save", broken)
        assert model.save(10) is None
        model.tick(11)  # A checkpoint failure must never escape into the training loop.
        output = capsys.readouterr()
        assert "disk full" in output.err and "continuing training" in output.err
        assert "live-metrics: " in output.out and "checkpointErrors" in output.out
        errors = json.loads((tmp_path / "out/checkpoint_errors.json").read_text())
        assert errors["checkpointErrors"][0]["step"] == 10
        assert "disk full" in errors["checkpointErrors"][0]["error"]
        assert resume_trainer.newest(tmp_path) == tmp_path / "resume_000000005.pt"
        assert (tmp_path / "resume_000000005.pt").read_bytes() == original
        assert not list(tmp_path.glob("*.part"))
        assert not (tmp_path / "resume_000000010.pt").exists()
    finally:
        if previous is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous


def test_budget_converge_telemetry_patch_stack_saves_only_data(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from argparse import Namespace

    from resume_maths_stand_in import Model

    import converge_trainer
    import convergence

    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch"))
    model = Model(
        Namespace(
            max_steps=20,
            checkpoint_every=5,
            checkpoint_dir=tmp_path,
            checkpoint_token="",
            resume=None,
            strategy="mcmc",
            result_dir=tmp_path / "out",
        )
    )
    strategy = model.runner.cfg.strategy
    # A bound method on the instance, as left by probe/telemetry restoration too.
    monkeypatch.setattr(type(strategy), "step_post_backward", lambda self: None, raising=False)
    cfg = model.runner.cfg
    cfg.eval_steps, cfg.save_steps, cfg.ply_steps = [20], [20], [20]

    class Runner:
        def train(self) -> Any:
            strategy.step_post_backward()
            # The real probe has installed its functools.wraps closure at this point.
            assert "step_post_backward" in vars(strategy)
            return model.save(5)

        def eval(self, step: int, stage: str = "val") -> None:
            pass

    strategy.step_post_backward = lambda: None
    state = converge_trainer._State(convergence.Rule(), tmp_path)
    converge_trainer.install(Runner, cfg, state)
    converge_trainer.instrument({}, Runner, converge_trainer.Timing(None), cfg)
    runner = Runner()
    runner.__dict__.update(vars(model.runner))
    # Save the same runner whose methods and cfg the wrappers are observing.
    model.runner = runner
    runner.train()
    target = tmp_path / "resume_000000005.pt"
    assert target.exists()
    saved = model.torch.load(target, weights_only=True)
    assert "step_post_backward" not in saved["identity"]["strategySettings"]
    assert isinstance(saved["strategy"], dict)
    cfg.resume = str(target)
    assert resume_trainer.restore(runner, model.schedulers, model.loader) == 5


def test_real_torch_checkpoint_is_weights_only_loadable(tmp_path: Path) -> None:
    interpreter = os.environ.get("GSPLAT_RESUME_PYTHON")
    if interpreter is None:
        pytest.skip("set GSPLAT_RESUME_PYTHON to a local CPU-capable torch interpreter")
    check = Path(__file__).with_name("resume_torch_check.py")
    result = subprocess.run(
        [interpreter, str(check), str(tmp_path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("location", ["strategy", "module", "scheduler", "loader"])
def test_capture_rejects_non_data_in_every_state_path(
    tmp_path: Path, monkeypatch: Any, location: str
) -> None:
    from argparse import Namespace

    from resume_maths_stand_in import Model

    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch"))
    model = Model(
        Namespace(
            max_steps=20,
            checkpoint_every=5,
            checkpoint_dir=tmp_path,
            checkpoint_token="",
            resume=None,
            strategy="default",
            result_dir=tmp_path / "out",
        )
    )
    bad = {"callback": lambda: None}
    if location == "strategy":
        model.runner.strategy_state = bad
    else:
        owner = {
            "module": model.runner.pose_adjust,
            "scheduler": model.schedulers[0],
            "loader": model.loader,
        }[location]
        monkeypatch.setattr(owner, "state_dict", lambda: bad)
    with pytest.raises(TypeError, match="unsupported checkpoint value function"):
        resume_trainer.capture(model.runner, model.schedulers, model.loader, 5)
    assert model.save(5) is None
    assert not resume_trainer.checkpoints(tmp_path)
    model.tick(6) if location != "strategy" else model.optimizers[0].step(1.0)


@pytest.mark.parametrize("failure", ["rotate", "final_outputs", "progress"])
def test_checkpoint_io_failures_outside_torch_save_are_nonfatal(
    tmp_path: Path, monkeypatch: Any, failure: str
) -> None:
    from argparse import Namespace

    from resume_maths_stand_in import Model

    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch"))
    model = Model(
        Namespace(
            max_steps=20,
            checkpoint_every=5,
            checkpoint_dir=tmp_path,
            checkpoint_token="",
            resume=None,
            strategy="mcmc",
            result_dir=tmp_path / "out",
        )
    )
    previous = model.save(5)

    def broken(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected IO failure")

    if failure == "final_outputs":
        (tmp_path / "out/stats").mkdir(parents=True)
        monkeypatch.setattr("resume_trainer.shutil.copytree", broken)
    elif failure == "progress":
        monkeypatch.setattr(Path, "write_text", broken)
    else:
        monkeypatch.setattr(resume_trainer, failure, broken)
    assert model.save(20) is None
    model.tick(6)
    if failure == "rotate":
        newest = resume_trainer.newest(tmp_path)
        assert newest is not None and newest.name == "resume_000000020.pt"
        assert previous is not None and previous.is_file()
    else:
        assert resume_trainer.newest(tmp_path) == previous
    assert model.runner._hex_checkpoint_errors[-1]["step"] == 20


def test_completed_checkpoint_recovers_final_outputs_without_more_updates(tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "run"
    first = invoke(root, strategy="mcmc")
    assert first.returncode == 0, first.stderr
    checkpoint = resume_trainer.newest(root / "ckpt")
    assert checkpoint is not None
    state = checkpoint.read_bytes()
    output = root / "out/ply/point_cloud_18.ply"
    expected = output.read_bytes()
    shutil.rmtree(root / "out")
    continued = invoke(root, strategy="mcmc", resume=checkpoint)
    assert continued.returncode == 0, continued.stderr
    assert "starting at step 19" in continued.stdout
    assert output.read_bytes() == expected
    equal(pickle.loads(state), pickle.loads(checkpoint.read_bytes()))


def test_gpu_module_parses_under_python_310() -> None:
    ast.parse(Path(resume_trainer.__file__).read_text(), feature_version=(3, 10))


def test_pinned_source_patch(tmp_path: Path) -> None:
    path = os.environ.get("GSPLAT_RESUME_SOURCE")
    if path is None:
        pytest.skip("set GSPLAT_RESUME_SOURCE to the locally cached pinned simple_trainer.py")
    source = Path(path).read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == (
        "56496f684cfc1ea737064ff228ef2acdb99c913041f99aa6fa23155487245764"
    )
    patched = resume_trainer.patch_source(source)
    ast.parse(patched, feature_version=(3, 10))
    tree = ast.parse(patched)
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runner")
    train = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    loop = next(n for n in train.body if isinstance(n, ast.For))
    assert ast.unparse(loop.body[-1]) == "_hex_resume.save(self, schedulers, trainloader, step + 1)"
    with pytest.raises(ValueError, match=r"already patched|seam moved"):
        resume_trainer.patch_source(patched)
    with pytest.raises(ValueError, match="seam moved"):
        resume_trainer.patch_source(source.replace("init_step = 0", "init_step = 1"))


def test_prefetched_random_samples_restore_consumed_cursor_and_keep_four_workers(
    monkeypatch: Any,
) -> None:
    import random

    from resume_maths_stand_in import install_torch

    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch"))
    torch = install_torch()

    class RandomDataset:
        def __len__(self) -> int:
            return 13

        def __getitem__(self, index: int) -> Any:
            return [index, random.random(), np.random.random(), torch.rng.rng.random()]

    random.seed(4)
    np.random.seed(5)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    loader = resume_trainer.Loader(RandomDataset(), 2)
    iterator = iter(loader)  # Stand-in prefetches the entire epoch here.
    equal(before, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
    next(iterator)
    next(iterator)
    saved = loader.state_dict()
    assert saved["cursor"] == 4
    expected = list(iterator)
    continued = resume_trainer.Loader(RandomDataset(), 2)
    continued.load_state_dict(saved)
    equal(expected, list(continued))
    equal(list(loader), list(continued))  # Also identical across the next shuffle.


def test_only_complete_checkpoints_with_matching_receipts_are_selected(tmp_path: Path) -> None:
    old = tmp_path / "resume_000000005.pt"
    old.write_bytes(b"complete")
    old.with_suffix(".committed").write_text(json.dumps({"bytes": 8}))
    new = tmp_path / "resume_000000010.pt"
    new.with_suffix(".committed").write_text(json.dumps({"bytes": 20}))
    assert resume_trainer.newest(tmp_path) == old
    new.write_bytes(b"partial")
    assert resume_trainer.newest(tmp_path) == old
    new.write_bytes(b"x" * 20)
    assert resume_trainer.newest(tmp_path) == new


def test_convergence_history_and_pending_stop_survive_checkpoint(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from argparse import Namespace
    from types import SimpleNamespace

    from resume_maths_stand_in import Model

    import convergence

    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch"))
    args = Namespace(
        max_steps=20,
        checkpoint_every=5,
        checkpoint_dir=tmp_path,
        checkpoint_token="",
        resume=None,
        strategy="mcmc",
        result_dir=tmp_path / "out",
    )
    first = Model(args)
    first.runner._hex_converge = SimpleNamespace(
        rule=convergence.Rule(),
        history=[(1, 20.0), (3, 20.01)],
        decided_at=3,
        stopped_at=None,
        gain_db=0.01,
        last="flat",
        lpips_skipped=2,
    )
    first.tick(1)
    checkpoint = first.save(5)
    continued = Model(args)
    continued.runner._hex_converge = SimpleNamespace(rule=convergence.Rule())
    cfg = continued.runner.cfg
    cfg.resume = str(checkpoint)
    cfg.ply_steps, cfg.save_steps, cfg.eval_steps = [], [], []
    assert resume_trainer.restore(continued.runner, continued.schedulers, continued.loader) == 5
    equal(
        resume_trainer.converge_state(first.runner), resume_trainer.converge_state(continued.runner)
    )
    assert cfg.ply_steps == cfg.save_steps == cfg.eval_steps == [5]


@pytest.mark.parametrize("changed", ["token", "version", "schedule"])
def test_stale_plan_discards_only_its_lane_and_restarts_at_zero(
    tmp_path: Path, monkeypatch: Any, capsys: Any, changed: str
) -> None:
    from argparse import Namespace

    from resume_maths_stand_in import Model

    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch"))
    args = Namespace(
        max_steps=20,
        checkpoint_every=5,
        checkpoint_dir=tmp_path / "resume",
        checkpoint_token="old-quality-token",
        resume="latest",
        strategy="mcmc",
        result_dir=tmp_path / "work" / "blocks" / "block_000" / "gsplat",
    )
    first = Model(args)
    first.tick(1)
    checkpoint = first.save(5)
    first.save(10)
    lane = Path(first.runner.cfg.checkpoint_dir)
    other = lane.parent / "other-lane"
    other.mkdir()
    (other / "resume_000000005.pt").write_bytes(b"another block")
    if changed == "token":
        args.checkpoint_token = "new-quality-token"
    elif changed == "schedule":
        args.max_steps = 30
    else:
        saved = first.torch.load(checkpoint)
        saved["version"] = -1
        # Change both committed files so configure still selects a stale checkpoint.
        for path in resume_trainer.checkpoints(lane):
            with path.open("wb") as handle:
                first.torch.save(saved, handle)
            path.with_suffix(".committed").write_text(json.dumps({"bytes": path.stat().st_size}))
    fresh = Model(args)
    assert fresh.start == 0 and fresh.runner.cfg.resume is None
    assert fresh.runner._hex_resume_step == 0
    assert not resume_trainer.checkpoints(lane)
    assert not list(lane.glob("*.committed"))
    assert (other / "resume_000000005.pt").exists()
    assert "incompatible resume checkpoint" in capsys.readouterr().err
    # A new compatible checkpoint can be selected on the very next retry.
    fresh.save(5)
    assert Model(args).start == 5
