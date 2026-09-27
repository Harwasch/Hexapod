"""Does the installed trainer accept the argv the pipeline will give it? Run at image build.

`infra/modal/app.py` runs this with the training venv's interpreter as the last step of
building the image, so an image whose trainer would refuse `training.gsplat_argv` -- a
renamed flag, a missing dependency, a gsplat that is not the one the pipeline was
written against -- fails to *build*, rather than failing a GPU run an hour later.

It is the same check that was run by hand on a CPU venv while this was written, now
repeated against the real image: import gsplat v1.5.3's `examples/simple_trainer.py` up to
(not including) its `__main__` block, build the same `configs` it builds, and parse the
pipeline's own argv with the same `tyro.extras.overridable_config_cli`. Then the fields
the pipeline depends on are asserted. Nothing here needs a GPU: `Runner`, which is the
part that does, is never constructed.

Three switches go further than parsing, because what they need is only imported once the
trainer is running: `--use_bilateral_grid` imports `lib_bilagrid` (and through it
`tensorly`) in `__main__`, after the CLI has accepted the flag; `--app_opt` reaches into
`gsplat.cuda._torch_impl` on its first forward; `--pose_opt` builds `CameraOptModule`.
Each is constructed here and run one forward and backward pass on CPU tensors, with the
shapes and arguments `simple_trainer.py` uses, so an image in which one of them cannot
run fails to build.

Usage (in the image):  $GSPLAT_PYTHON check_trainer.py <simple_trainer.py> <pipeline dir>
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any


def main() -> int:
    trainer = Path(sys.argv[1]).resolve()
    pipeline = Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(trainer.parent))
    sys.path.insert(0, str(pipeline))

    import convergence  # the stopping rule's schedule; plain Python, 3.10-compatible
    import training  # the pipeline's own argv builder; plain Python, 3.10-compatible

    try:
        import fused_ssim  # noqa: F401 - a CUDA extension; present in the real image
    except ImportError as error:
        # Only tolerated when explicitly asked for, for a CPU replica. In the image this
        # import failing is itself the defect worth failing the build on.
        if "--allow-missing-cuda-extensions" not in sys.argv:
            raise SystemExit(f"fused_ssim does not import in the trainer venv: {error}") from error
        stub = types.ModuleType("fused_ssim")
        stub.fused_ssim = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        sys.modules["fused_ssim"] = stub

    source = trainer.read_text(encoding="utf-8")
    head = source.split('if __name__ == "__main__":')[0]
    # A real module object, registered, because `dataclasses` looks its class's module up
    # in `sys.modules`; and `dont_inherit`, so this file's own `__future__` import does not
    # turn the trainer's annotations into strings it was never written to have.
    module = types.ModuleType("simple_trainer_check")
    module.__file__ = str(trainer)
    sys.modules[module.__name__] = module
    code = compile(head, str(trainer), "exec", dont_inherit=True)
    exec(code, module.__dict__)  # noqa: S102 - the trainer this image ships
    namespace = module.__dict__

    import tyro

    config = namespace["Config"]
    default_strategy = namespace["DefaultStrategy"]
    mcmc_strategy = namespace["MCMCStrategy"]
    # The presets exactly as the trainer's own `__main__` builds them, so a flag that
    # overrides a preset value (`--opacity_reg` over mcmc's 0.01) is checked against it.
    configs = {
        "default": ("default", config(strategy=default_strategy(verbose=True))),  # type: ignore[operator]
        "mcmc": (
            "mcmc",
            config(  # type: ignore[operator]
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=mcmc_strategy(verbose=True),  # type: ignore[operator]
            ),
        ),
    }

    def parse(argv: list[str]) -> object:
        sys.argv = [str(trainer), *argv[2:]]
        cfg = tyro.extras.overridable_config_cli(configs)
        cfg.adjust_steps(cfg.steps_scaler)
        return cfg

    cfg: Any = parse(
        training.gsplat_argv(sys.executable, trainer, Path("/data"), Path("/result"), max_steps=500)
    )
    # Every optional switch the `train` stage can pass, at once: the refine-quality shape.
    tuned: Any = parse(
        training.gsplat_argv(
            sys.executable,
            trainer,
            Path("/data"),
            Path("/result"),
            strategy="mcmc",
            max_steps=30_000,
            steps_scaler=0.1,
            cap_max=200_000,
            antialiased=True,
            opacity_reg=0.001,
            depth_loss=True,
            pose_opt=True,
            app_opt=True,
            bilateral_grid=True,
        )
    )
    # With `converge`: the held-out evaluations the stopping rule reads, as extra
    # `--eval_steps` after the final one. What `converge_trainer.py` hands the trainer is
    # this argv from the strategy on, unchanged.
    evals = convergence.eval_steps(30_000, convergence.REFINE_STOP_ITER["mcmc"])
    converging: Any = parse(
        training.gsplat_argv(
            sys.executable,
            trainer,
            Path("/data"),
            Path("/result"),
            strategy="mcmc",
            max_steps=30_000,
            steps_scaler=0.5,
            cap_max=1_500_000,
            eval_steps=evals,
        )
    )
    # The benchmark's argv (tools/pipeline/experiments/benchmark.py): gsplat's 3DGS
    # reproduction, the full schedule, LPIPS on VGG as the paper measured it.
    bench: Any = parse(
        training.gsplat_argv(
            sys.executable,
            trainer,
            Path("/data"),
            Path("/result"),
            max_steps=30_000,
            extra=["--lpips_net", "vgg"],
        )
    )

    checks = {
        "mcmc strategy": type(tuned.strategy).__name__ == "MCMCStrategy",
        "cap_max": tuned.strategy.cap_max == 200_000,
        "steps_scaler": tuned.max_steps == 3_000,
        "antialiased": tuned.antialiased is True and cfg.antialiased is False,
        "opacity_reg": tuned.opacity_reg == 0.001 and tuned.scale_reg == 0.01,
        "depth_loss": tuned.depth_loss is True and cfg.depth_loss is False,
        "pose_opt": tuned.pose_opt is True and cfg.pose_opt is False,
        "app_opt": tuned.app_opt is True and cfg.app_opt is False,
        "use_bilateral_grid": (
            tuned.use_bilateral_grid is True
            and cfg.use_bilateral_grid is False
            # The fused CUDA grid is a different package the image does not install; the
            # flag the pipeline passes must select the plain-torch one.
            and tuned.use_fused_bilagrid is False
        ),
        "benchmark": (
            bench.lpips_net == "vgg"
            and type(bench.strategy).__name__ == "DefaultStrategy"
            and bench.max_steps == 30_000
        ),
        "data_dir": cfg.data_dir == "/data",
        "result_dir": cfg.result_dir == "/result",
        "max_steps": cfg.max_steps == 500,
        "disable_viewer": cfg.disable_viewer is True,
        "disable_video": cfg.disable_video is True,
        "normalize_world_space": cfg.normalize_world_space is False,
        "save_ply": cfg.save_ply is True,
        "ply_steps": list(cfg.ply_steps) == [500],
        "eval_steps": list(cfg.eval_steps) == [500],
        # Scaled by the trainer as the rule assumes, and densification ending where
        # `convergence.REFINE_STOP_ITER` says each strategy's does.
        "converge eval_steps": (
            list(converging.eval_steps) == [int(i * 0.5) for i in [30_000, *evals]]
        ),
        "refine_stop_iter": (
            converging.strategy.refine_stop_iter
            == int(convergence.REFINE_STOP_ITER["mcmc"] * 0.5)
            and cfg.strategy.refine_stop_iter == convergence.REFINE_STOP_ITER["default"]
        ),
        "ckpt": cfg.ckpt is None,
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise SystemExit(f"simple_trainer.py parsed the pipeline's argv wrongly: {failed}")
    ran = _exercise_switches(namespace)
    sys.stdout.write(
        f"trainer ok: {trainer} accepts training.gsplat_argv (tyro {tyro.__version__}); "
        f"checked {', '.join(sorted(checks))}; ran {', '.join(ran)} on CPU\n"
    )
    return 0


def _exercise_switches(namespace: dict[str, Any]) -> list[str]:
    """One forward and backward pass of what `--use_bilateral_grid`, `--app_opt` and
    `--pose_opt` build, on CPU tensors, called the way v1.5.3's `Runner` calls them.

    `lib_bilagrid` is imported from beside the trainer, as its `__main__` imports it; the
    other two are the classes the trainer module itself imported from `utils`.
    """
    import torch
    from lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss

    ran: list[str] = []
    height, width = 12, 16
    # Runner.__init__: BilateralGrid(len(trainset), grid_X=16, grid_Y=16, grid_W=8).
    grids = BilateralGrid(2, grid_X=16, grid_Y=16, grid_W=8)
    colors = torch.rand(1, height, width, 3, requires_grad=True)
    grid_y, grid_x = torch.meshgrid(
        (torch.arange(height) + 0.5) / height,
        (torch.arange(width) + 0.5) / width,
        indexing="ij",
    )
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    image_ids = torch.tensor([1])
    out = slice(grids, grid_xy.expand(1, -1, -1, -1), colors, image_ids.unsqueeze(-1))["rgb"]
    (out.mean() + 10 * total_variation_loss(grids.grids)).backward()
    corrected = color_correct(out.detach(), torch.rand(1, height, width, 3))
    if grids.grids.grad is None or tuple(corrected.shape) != (1, height, width, 3):
        raise SystemExit("lib_bilagrid ran but produced no gradient or the wrong shape")
    ran.append("bilateral grid")

    # Runner: AppearanceOptModule(len(trainset), 32, app_embed_dim, sh_degree), then the
    # PLY export's bake with embed_ids=None and zero view directions.
    app = namespace["AppearanceOptModule"](2, 32, 16, 3)
    features = torch.rand(10, 32, requires_grad=True)
    app(
        features=features, embed_ids=image_ids, dirs=torch.rand(1, 10, 3), sh_degree=3
    ).mean().backward()
    baked = app(features=features, embed_ids=None, dirs=torch.zeros(1, 10, 3), sh_degree=3)
    if features.grad is None or not bool(torch.isfinite(baked).all()):
        raise SystemExit("AppearanceOptModule ran but gave no gradient or a non-finite bake")
    ran.append("appearance embedding")

    # Runner: CameraOptModule(len(trainset)).zero_init(); a zero delta is the identity.
    pose = namespace["CameraOptModule"](2)
    pose.zero_init()
    adjusted = pose(torch.eye(4)[None], image_ids)
    if not bool(torch.allclose(adjusted, torch.eye(4)[None])):
        raise SystemExit("CameraOptModule's zero-initialised correction is not the identity")
    ran.append("pose correction")
    return ran


if __name__ == "__main__":
    raise SystemExit(main())
