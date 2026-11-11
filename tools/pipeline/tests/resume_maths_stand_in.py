"""Deterministic NumPy Adam stand-in, not a renderer or evidence of CUDA exactness.

Used by gsplat_stand_in's opt-in checkpoint path. Implements the small torch state
interface resume_trainer uses, so the *real* capture/restore/save code runs offline.
"""

import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


class Tensor(np.ndarray[Any, Any]):
    def detach(self) -> Any:
        return self

    def to(self, device: Any) -> Any:
        return self

    def cpu(self) -> Any:
        return self


def tensor(value: Any) -> Any:
    return np.asarray(value).view(Tensor)


class Generator:
    def __init__(self) -> None:
        self.rng = np.random.RandomState(77)

    def manual_seed(self, seed: Any) -> Any:
        self.rng.seed(seed % 2**32)
        return self

    def get_state(self) -> Any:
        return tensor(np.frombuffer(pickle.dumps(self.rng.get_state()), dtype=np.uint8))

    def set_state(self, state: Any) -> Any:
        self.rng.set_state(pickle.loads(bytes(state)))


class DataLoader:
    """Prefetch the epoch so tests detect cursor changes caused by queueing."""

    def __init__(self, dataset: Any, batch_sampler: Any, **options: Any) -> None:
        assert options["num_workers"] == 4 and options["persistent_workers"]
        self.dataset, self.sampler = dataset, batch_sampler

    def __iter__(self) -> Any:
        return iter([np.stack([self.dataset[key] for key in batch]) for batch in self.sampler])


def install_torch() -> Any:
    rng = Generator()
    torch: Any = SimpleNamespace(
        Generator=Generator,
        initial_seed=lambda: 77,
        randperm=lambda count, generator: tensor(generator.rng.permutation(count)),
        get_rng_state=rng.get_state,
        set_rng_state=rng.set_state,
        cuda=SimpleNamespace(is_available=lambda: False),
        utils=SimpleNamespace(
            data=SimpleNamespace(default_collate=np.stack, DataLoader=DataLoader)
        ),
        is_tensor=lambda value: isinstance(value, Tensor),
        from_numpy=tensor,
        save=lambda state, handle: pickle.dump(state, handle),
        load=lambda path, **kwargs: pickle.loads(Path(path).read_bytes()),
        rng=rng,
    )
    sys.modules["torch"] = torch
    return torch


class Parameter:
    def __init__(self, value: Any) -> None:
        self.data = tensor(value)
        self.requires_grad = True

    def requires_grad_(self, value: Any) -> Any:
        self.requires_grad = value


class Parameters(dict[str, Any]):
    def state_dict(self) -> Any:
        return {key: value.data for key, value in self.items()}


class Module:
    def __init__(self) -> None:
        self.params = [Parameter([0.2, 0.3]), Parameter([0.4])]

    def state_dict(self) -> Any:
        return {str(i): p.data for i, p in enumerate(self.params)}

    def load_state_dict(self, state: Any) -> Any:
        for i, param in enumerate(self.params):
            param.data = state[str(i)]

    def named_parameters(self) -> Any:
        return [(str(i), p) for i, p in enumerate(self.params)]


class Scene:
    def __init__(self, splats: Any) -> None:
        self.splats = splats
        self.id = "scene"
        self.component_index = tensor([0, 0])
        self.component_names = ["scene"]
        self.signal = {"history": tensor([0.0, 0.0])}

    def validate(self) -> Any:
        assert len(self.component_index) == len(self.splats["means"].data)
        assert len(self.signal["history"]) == len(self.component_index)


class Adam:
    def __init__(self, params: Any) -> None:
        self.params = params
        self.state = [
            {
                "step": 0,
                "exp_avg": tensor(np.zeros_like(p.data)),
                "exp_avg_sq": tensor(np.zeros_like(p.data)),
            }
            for p in params
        ]
        self.param_groups = [
            {"lr": 0.03 * (i + 1), "betas": (0.9, 0.999)} for i in range(len(params))
        ]

    def state_dict(self) -> Any:
        return {"state": self.state, "param_groups": self.param_groups}

    def load_state_dict(self, state: Any) -> Any:
        self.state, self.param_groups = state["state"], state["param_groups"]

    def step(self, noise: Any) -> Any:
        for param, state, group in zip(self.params, self.state, self.param_groups, strict=True):
            if not param.requires_grad:
                continue
            state["step"] += 1
            grad = param.data * 0.5 + noise
            state["exp_avg"] = 0.9 * state["exp_avg"] + 0.1 * grad
            state["exp_avg_sq"] = 0.999 * state["exp_avg_sq"] + 0.001 * grad**2
            first = state["exp_avg"] / (1 - 0.9 ** state["step"])
            second = state["exp_avg_sq"] / (1 - 0.999 ** state["step"])
            param.data -= group["lr"] * first / (np.sqrt(second) + 1e-8)


class Scheduler:
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        self.epoch = 0

    def state_dict(self) -> Any:
        return {"last_epoch": self.epoch, "lrs": [g["lr"] for g in self.optimizer.param_groups]}

    def load_state_dict(self, state: Any) -> Any:
        self.epoch = state["last_epoch"]

    def step(self) -> Any:
        self.epoch += 1
        for group in self.optimizer.param_groups:
            group["lr"] *= 0.98


class MCMCStrategy:
    pass


class DefaultStrategy:
    pass


class Model:
    def __init__(self, args: Any) -> None:
        import resume_trainer

        self.torch = install_torch()
        random.seed(22)
        np.random.seed(33)
        self.runner: Any = SimpleNamespace(
            cfg=SimpleNamespace(
                max_steps=args.max_steps,
                checkpoint_every=args.checkpoint_every,
                checkpoint_final=not getattr(args, "no_checkpoint_final", False),
                checkpoint_dir=str(args.checkpoint_dir),
                checkpoint_token=args.checkpoint_token,
                resume=args.resume,
                strategy=(MCMCStrategy() if args.strategy == "mcmc" else DefaultStrategy()),
                ckpt=None,
                result_dir=str(args.result_dir),
            ),
            splats=Parameters(
                {
                    k: Parameter([0.1, 0.4])
                    for k in ("means", "scales", "quats", "opacities", "sh0", "shN")
                }
            ),
            strategy_state={
                "grad2d": tensor([0.0, 0.0]),
                "count": tensor([0.0, 0.0]),
                "radii": tensor([0.0, 0.0]),
                "bins": tensor([1.0, 2.0]),
            },
            world_size=1,
            device="cpu",
            _gaussians_frozen=False,
        )
        self.runner.optimizers = {k: Adam([p]) for k, p in self.runner.splats.items()}
        self.runner.scene = Scene(self.runner.splats)
        for name, opt_name in (
            ("pose_adjust", "pose_optimizers"),
            ("app_module", "app_optimizers"),
            ("post_processing_module", "post_processing_optimizers"),
        ):
            module = Module()
            setattr(self.runner, name, module)
            setattr(self.runner, opt_name, [Adam(module.params)])
        self.runner.pose_perturb = Module()
        self.optimizers = list(resume_trainer._optimizer_items(self.runner).values())
        self.schedulers = [Scheduler(o) for o in self.optimizers]
        self.loader = resume_trainer.Loader(list(range(7)), 2)
        resume_trainer.configure(self.runner)
        self.start = resume_trainer.restore(self.runner, self.schedulers, self.loader)
        self.iterator = iter(self.loader)

    def tick(self, step: Any) -> Any:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        noise = random.random() + np.random.random() + self.torch.rng.rng.random() + batch.mean()
        if step == 12:
            self.runner.post_processing_module.params[0].requires_grad = False
        for optimizer in self.optimizers:
            optimizer.step(noise)
        for scheduler in self.schedulers:
            scheduler.step()
        # Stand in for replacement of every Parameter and its Adam moments by MCMC.
        if step == 3:
            for key, old in list(self.runner.splats.items()):
                param = Parameter(np.r_[old.data, 0.6])
                self.runner.splats[key] = param
                opt = self.runner.optimizers[key]
                opt.params[0] = param
                for field in ("exp_avg", "exp_avg_sq"):
                    opt.state[0][field] = tensor(np.r_[opt.state[0][field], 0.0])
            for key, value in self.runner.strategy_state.items():
                self.runner.strategy_state[key] = tensor(np.r_[value, 0.0])
            self.runner.scene.component_index = tensor([0, 0, 0])
            self.runner.scene.signal["history"] = tensor([0.0, 0.0, 0.0])
        for key in self.runner.strategy_state:
            self.runner.strategy_state[key] += noise * step
        self.runner.scene.signal["history"] += noise

    def save(self, step: Any) -> Any:
        import resume_trainer

        return resume_trainer.save(self.runner, self.schedulers, self.loader, step)
