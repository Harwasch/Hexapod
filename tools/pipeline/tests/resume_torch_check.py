"""Real CPU torch regression, also executable with the GPU image's Python 3.10."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import converge_trainer
import convergence
import resume_trainer


@dataclass
class MCMCStrategy:
    cap_max: int = 100
    refine_stop_iter: int = 10

    def step_post_backward(self) -> None:
        pass


class Loader:
    def state_dict(self) -> Any:
        return {"cursor": 5}

    def load_state_dict(self, state: Any) -> None:
        assert state == {"cursor": 5}


def check_metadata(module: Any, state: Any, prefix: str, metadata: Any, *args: Any) -> None:
    assert metadata["version"] == module._version == 4


def check(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    cfg = SimpleNamespace(
        max_steps=20,
        checkpoint_every=5,
        checkpoint_dir=str(root / "ckpt"),
        checkpoint_token="",
        resume=None,
        strategy=MCMCStrategy(),
        ckpt=None,
        result_dir=str(root),
        eval_steps=[20],
        save_steps=[20],
        ply_steps=[20],
    )
    loader = Loader()
    paths = []

    class Runner:
        pose_adjust: Any
        app_module: Any
        post_processing_module: Any

        def __init__(self) -> None:
            self.cfg = cfg
            self.device = "cpu"
            self.splats = torch.nn.ParameterDict({"means": torch.nn.Parameter(torch.ones(2, 3))})
            self.optimizers = {"means": torch.optim.Adam(self.splats.parameters())}
            self.schedulers = [
                torch.optim.lr_scheduler.ExponentialLR(self.optimizers["means"], 0.99)
            ]
            self.strategy_state = {
                "grad2d": torch.ones(2),
                "count": torch.ones(2),
                "radii": torch.zeros(2),
                "bins": torch.arange(3),
            }
            for name in resume_trainer.MODULES:
                module = torch.nn.Linear(3, 2)
                module._version = 4
                module.register_load_state_dict_pre_hook(check_metadata)
                setattr(self, name, module)
            self.pose_optimizers = [torch.optim.Adam(self.pose_adjust.parameters())]
            self.app_optimizers = [torch.optim.Adam(self.app_module.parameters())]
            self.post_processing_optimizers = [
                torch.optim.Adam(self.post_processing_module.parameters())
            ]

        def train(self) -> None:
            for step in range(1, 11):
                self.splats["means"].sum().backward()
                self.optimizers["means"].step()
                self.optimizers["means"].zero_grad()
                self.schedulers[0].step()
                cfg.strategy.step_post_backward()
                if step == 5:
                    assert "step_post_backward" in vars(cfg.strategy)
                    paths.append(resume_trainer.save(self, self.schedulers, loader, step))
                if step == 10:
                    real_save = torch.save

                    def broken(*args: Any, **kwargs: Any) -> None:
                        raise OSError("injected checkpoint failure")

                    torch.save = broken
                    try:
                        assert resume_trainer.save(self, self.schedulers, loader, step) is None
                    finally:
                        torch.save = real_save
            self.finished = True

        def eval(self, step: int, stage: str = "val") -> None:
            pass

    cfg.strategy.step_post_backward = lambda: None
    state = converge_trainer._State(convergence.Rule(), root)
    converge_trainer.install(Runner, cfg, state)
    converge_trainer.instrument({}, Runner, converge_trainer.Timing(None), cfg)
    runner = Runner()
    runner.train()
    assert runner.finished
    target = paths[0]
    assert target is not None and target.is_file()
    saved = torch.load(target, weights_only=True)
    assert saved["identity"]["strategySettings"] == {"cap_max": 100, "refine_stop_iter": 10}
    assert saved["optimizers"]["optimizers/means"]["state"][0]["step"].item() == 5
    assert set(saved["modules"]) == set(resume_trainer.MODULES)
    assert all(value[""]["version"] == 4 for value in saved["moduleMetadata"].values())
    assert resume_trainer.newest(root / "ckpt") == target
    errors = json.loads((root / "checkpoint_errors.json").read_text())["checkpointErrors"]
    assert errors[0]["step"] == 10
    cfg.resume = str(target)
    expected = saved["splats"]["means"].clone()
    python_rng, numpy_rng = saved["rng"]["python"], saved["rng"]["numpy"]
    assert resume_trainer.restore(runner, runner.schedulers, loader) == 5
    assert torch.equal(runner.splats["means"], expected)
    assert random.getstate() == python_rng
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_rng[0]
    np.testing.assert_array_equal(restored_numpy[1], numpy_rng[1])
    assert restored_numpy[2:] == tuple(numpy_rng[2:])
    assert torch.equal(torch.get_rng_state(), saved["rng"]["torch"])
    assert torch.equal(runner.strategy_state["grad2d"], saved["strategy"]["grad2d"])
    # A changed data setting still rejects resume, even after probe restores methods.
    cfg.strategy.cap_max += 1
    try:
        resume_trainer.restore(runner, runner.schedulers, loader)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("changed strategy settings accepted")


if __name__ == "__main__":
    check(Path(sys.argv[1]))
