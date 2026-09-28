"""Train one block with gsplat's `simple_trainer.py` unchanged: a frozen ring of the
prior's gaussians drawn round it, and the convergence stop, as a wrapper.

**Not executed on a GPU in this repository yet.** The mechanism is `converge_trainer.py`'s,
and so is the reason for it (read that docstring): the trainer runs as `python
simple_trainer.py` would, and the one call its `__main__` block ends with -- `cli(main,
cfg)` -- is replaced by one that patches the trainer's module before handing on. Two
patches, each optional:

* **The convergence stop** (`--converge`): `converge_trainer.install`, as it is.
* **The frozen ring** (`--ring ring.npz`): Hierarchical 3DGS loads a "scaffold" of the
  coarse model round each chunk, 0.5-1.5 chunk sizes out, and excludes it from
  densification (`scene/gaussian_model.py`, `densify_and_split`: "No densification of the
  scaffold"), so a chunk's edges are occluded and lit as in the whole scene. Here the ring
  is not optimised at all -- it is discarded after training, and holding it as constants
  costs its parameters only, not their gradients and Adam's two moments (944 of the 1,616
  bytes a trained gaussian costs, `gaussian_budget.py`). v1.5.3's `Runner.rasterize_splats`
  calls the module-global `rasterization(means=..., quats=..., scales=..., opacities=...,
  colors=..., ...)` with activated tensors; the patch replaces that global with one that
  appends the ring's constants to those five and calls the real one. Everything that owns
  or exports gaussians -- `self.splats`, the optimisers, `MCMCStrategy` (relocation and
  growth read `params`, never `info`), `export_splats` -- sees only the block's own, so
  the ring is never densified, relocated, regularised or written to the PLY. `eval()`
  renders through the same method, so the block's held-out PSNR (and the convergence stop
  it drives) is of the block *in its surroundings*. The only per-gaussian output read back
  from the render is `info["radii"]` (by `visible_adam` and the viewer), which is trimmed
  to the block's own rows. Colours: with SH (`sh_degree` given) the ring's DC term, higher
  bands zero; with `app_opt` (post-sigmoid RGB, per camera) its DC colour as RGB.

`--ply-to-ckpt PLY CKPT --step N` is the other use: a merged block splat written as the
trainer's own checkpoint (`{"step", "splats": {means, scales, quats, opacities, sh0,
shN}}`, v1.5.3's `Runner.train` save), so `simple_trainer.py --ckpt CKPT` -- "run eval
only" in its `main` -- scores it on the held-out frames exactly as a single run is scored.
`gsplat.exporter.export_splats` writes `f_rest` as `shN.permute(0, 2, 1).reshape(N, -1)`
(channel-major), which this inverts.

Usage: `$GSPLAT_PYTHON block_trainer.py --trainer <simple_trainer.py> [--converge --every
N --window N --min-gain-db X] [--ring ring.npz] -- <the trainer's own argv>`.
"""

from __future__ import annotations

import argparse
import importlib
import os
import runpy
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import converge_trainer
import convergence

#: What the ring patch needs of the trainer's source, besides what converge_trainer does.
REQUIRED_SOURCE = (
    "from gsplat.rendering import rasterization",
    "render_colors, render_alphas, info = rasterization(",
    "means=means,",
    "colors=colors,",
)

SH_C0 = 0.28209479177387814


def load_ring(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path) as data:
        return {name: np.asarray(data[name], dtype=np.float32) for name in data.files}


def ring_rasterization(real: Any, ring: dict[str, Any]) -> Any:
    """`real` (gsplat's `rasterization`), with the ring's gaussians appended to every call."""
    import torch

    cache: dict[Any, dict[str, Any]] = {}

    def constants(device: Any) -> dict[str, Any]:
        if device not in cache:
            as_tensor = {name: torch.from_numpy(values).to(device) for name, values in ring.items()}
            cache[device] = {
                "means": as_tensor["means"],
                "quats": as_tensor["quats"],
                "scales": torch.exp(as_tensor["log_scales"]),
                "opacities": torch.sigmoid(as_tensor["logit_opacity"]),
                "sh0": as_tensor["f_dc"][:, None, :],
                "rgb": (SH_C0 * as_tensor["f_dc"] + 0.5).clamp(0.0, 1.0),
            }
        return cache[device]

    def rasterization(*args: Any, **kwargs: Any) -> Any:
        names = ("means", "quats", "scales", "opacities", "colors")
        if args or not all(name in kwargs for name in names):
            return real(*args, **kwargs)
        means = kwargs["means"]
        count = int(means.shape[-2])
        ring_ = constants(means.device)
        colors = kwargs["colors"]
        if kwargs.get("sh_degree") is not None and colors.dim() == 3 and colors.shape[0] == count:
            # [N, K, 3] SH coefficients: the ring's DC, the higher bands zero.
            extra = torch.zeros(
                (ring_["sh0"].shape[0], colors.shape[1] - 1, 3),
                dtype=colors.dtype,
                device=colors.device,
            )
            ring_colors = torch.cat([ring_["sh0"].to(colors.dtype), extra], dim=1)
            colors = torch.cat([colors, ring_colors], dim=0)
        elif colors.dim() == 2 and colors.shape[0] == count:
            colors = torch.cat([colors, ring_["rgb"].to(colors.dtype)], dim=0)
        elif colors.dim() == 3 and colors.shape[1] == count:
            # app_opt: [C, N, 3] post-sigmoid RGB, one per camera.
            rgb = ring_["rgb"].to(colors.dtype)[None].expand(colors.shape[0], -1, -1)
            colors = torch.cat([colors, rgb], dim=1)
        else:
            return real(*args, **kwargs)
        patched = dict(kwargs)
        for name in ("means", "quats", "scales", "opacities"):
            patched[name] = torch.cat([kwargs[name], ring_[name].to(kwargs[name].dtype)], dim=0)
        patched["colors"] = colors
        render_colors, render_alphas, info = real(**patched)
        radii = info.get("radii") if isinstance(info, dict) else None
        if radii is not None and not kwargs.get("packed") and radii.dim() >= 2:
            info["radii"] = radii[:, :count] if radii.dim() == 3 else radii[:count]
        return render_colors, render_alphas, info

    return rasterization


def ply_to_checkpoint(ply: Path, target: Path, step: int) -> int:
    """Write `ply` as v1.5.3's checkpoint. Returns the gaussian count."""
    import numpy as np
    import torch

    from holdout_error import read_ply, sh_rest

    columns = read_ply(ply)

    def stack(names: list[str]) -> Any:
        return torch.from_numpy(np.stack([columns[n] for n in names], 1).astype(np.float32))

    count = int(columns["x"].shape[0])
    rest = sh_rest(columns)
    sh_n = (
        torch.zeros((count, 15, 3), dtype=torch.float32)
        if rest is None
        else torch.from_numpy(np.ascontiguousarray(rest, dtype=np.float32))
    )
    splats = {
        "means": stack(["x", "y", "z"]),
        "scales": stack(["scale_0", "scale_1", "scale_2"]),
        "quats": stack(["rot_0", "rot_1", "rot_2", "rot_3"]),
        "opacities": torch.from_numpy(columns["opacity"].astype(np.float32)),
        "sh0": stack(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None, :],
        "shN": sh_n,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": int(step), "splats": splats}, str(target))
    return count


def self_check(trainer: Path) -> int:
    source = trainer.read_text(encoding="utf-8")
    missing = [text for text in REQUIRED_SOURCE if text not in source]
    if missing:
        sys.stderr.write(f"block_trainer: {trainer} no longer has {missing}\n")
        return 1
    return converge_trainer.self_check(trainer)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    own, rest = (
        (args[: args.index("--")], args[args.index("--") + 1 :]) if "--" in args else (args, [])
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--trainer", type=Path, default=None)
    parser.add_argument("--converge", action="store_true")
    parser.add_argument("--every", type=int, default=convergence.DEFAULT_EVERY)
    parser.add_argument("--window", type=int, default=convergence.DEFAULT_WINDOW)
    parser.add_argument(
        "--min-gain-db", dest="min_gain_db", type=float, default=convergence.DEFAULT_MIN_GAIN_DB
    )
    parser.add_argument("--ring", type=Path, default=None)
    parser.add_argument("--ply-to-ckpt", dest="ply_to_ckpt", nargs=2, type=Path, default=None)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--self-check", dest="self_check", action="store_true")
    options = parser.parse_args(own)
    if options.ply_to_ckpt is not None:
        count = ply_to_checkpoint(options.ply_to_ckpt[0], options.ply_to_ckpt[1], options.step)
        sys.stdout.write(f"block_trainer: {count} gaussians -> {options.ply_to_ckpt[1]}\n")
        return 0
    trainer = options.trainer or (
        Path(os.environ["GSPLAT_TRAINER"]) if os.environ.get("GSPLAT_TRAINER") else None
    )
    if trainer is None:
        sys.stderr.write("block_trainer: no --trainer and no $GSPLAT_TRAINER\n")
        return 2
    trainer = trainer.resolve()
    sys.path.insert(0, str(trainer.parent))
    if options.self_check:
        return self_check(trainer)
    state = converge_trainer._State(
        convergence.Rule(
            every=options.every, window=options.window, min_gain_db=options.min_gain_db
        ),
        converge_trainer._result_dir(rest) if options.converge else None,
    )
    ring = None if options.ring is None else load_ring(options.ring)
    timing = converge_trainer.Timing(converge_trainer._result_dir(rest))
    try:
        distributed: Any = importlib.import_module("gsplat.distributed")
    except ImportError as error:
        distributed = None
        state.reason = f"gsplat.distributed does not import ({error}); trained unpatched"
        sys.stdout.write(f"block_trainer: {state.reason}\n")
    if distributed is not None:
        real_cli = distributed.cli

        def cli(fn: Any, cfg: Any, verbose: bool = False) -> Any:
            module = getattr(fn, "__globals__", {})
            runner = module.get("Runner")
            devices = converge_trainer._device_count()
            if devices > 1:
                state.reason = f"{devices} GPUs: gsplat spawns a process each; trained unpatched"
                sys.stdout.write(f"block_trainer: {state.reason}\n")
                return real_cli(fn, cfg, verbose=verbose)
            if ring is not None:
                if "rasterization" in module:
                    module["rasterization"] = ring_rasterization(module["rasterization"], ring)
                    sys.stdout.write(
                        f"block_trainer: {ring['means'].shape[0]} frozen ring gaussians drawn "
                        f"with every render\n"
                    )
                else:
                    sys.stdout.write("block_trainer: no rasterization global; no ring\n")
            if options.converge:
                if runner is None:
                    state.reason = "the trainer's main has no Runner beside it"
                else:
                    converge_trainer.install(runner, cfg, state)
                    state.hooked = True
                    state.reason = "hooked"
            # The timing, and no eval renders (converge_trainer.py says why), whether or
            # not the rule is on: the merged splat's `--ckpt` evaluation comes here too.
            converge_trainer.instrument(module, runner, timing)
            return real_cli(fn, cfg, verbose=verbose)

        distributed.cli = cli
    sys.argv = [str(trainer), *rest]
    try:
        runpy.run_path(str(trainer), run_name="__main__")
    except SystemExit as exit_:
        code = exit_.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    finally:
        if options.converge:
            state.write()
        timing.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
