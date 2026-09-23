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

Usage (in the image):  $GSPLAT_PYTHON check_trainer.py <simple_trainer.py> <pipeline dir>
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def main() -> int:
    trainer = Path(sys.argv[1]).resolve()
    pipeline = Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(trainer.parent))
    sys.path.insert(0, str(pipeline))

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
    configs = {
        "default": ("default", config(strategy=default_strategy(verbose=True))),  # type: ignore[operator]
        "mcmc": ("mcmc", config(strategy=mcmc_strategy(verbose=True))),  # type: ignore[operator]
    }
    argv = training.gsplat_argv(
        sys.executable, trainer, Path("/data"), Path("/result"), max_steps=500
    )
    sys.argv = [str(trainer), *argv[2:]]
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    checks = {
        "data_dir": cfg.data_dir == "/data",
        "result_dir": cfg.result_dir == "/result",
        "max_steps": cfg.max_steps == 500,
        "disable_viewer": cfg.disable_viewer is True,
        "disable_video": cfg.disable_video is True,
        "normalize_world_space": cfg.normalize_world_space is False,
        "save_ply": cfg.save_ply is True,
        "ply_steps": list(cfg.ply_steps) == [500],
        "eval_steps": list(cfg.eval_steps) == [500],
        "ckpt": cfg.ckpt is None,
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise SystemExit(f"simple_trainer.py parsed the pipeline's argv wrongly: {failed}")
    sys.stdout.write(
        f"trainer ok: {trainer} accepts training.gsplat_argv (tyro {tyro.__version__}); "
        f"checked {', '.join(sorted(checks))}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
