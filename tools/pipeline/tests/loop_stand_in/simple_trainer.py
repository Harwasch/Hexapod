"""A stand-in for gsplat v1.5.3's `examples/simple_trainer.py` with its loop's *shape*.
**It trains nothing**, and every number it writes is made up.

`gsplat_stand_in.py` imitates the trainer's command line and output files; this one also
imitates the structure `converge_trainer.py` hooks, read from the file at the tag:

* a `Config` whose `adjust_steps(factor)` scales `max_steps`, the three step lists and the
  strategy's `refine_stop_iter` together;
* a `Runner` whose `train()` loop re-reads `cfg.save_steps`, `cfg.ply_steps` and
  `cfg.eval_steps` at every step -- `step in [i - 1 for i in cfg.ply_steps]` -- saving,
  exporting and then evaluating in that order, and whose `eval(step)` writes
  `stats/val_step<step:04d>.json` (and a render per val frame under `renders/`);
* the module globals v1.5.3's loop calls and `converge_trainer.instrument` wraps: `imageio`
  (the stand-in beside this file), whose `imwrite` writes each val frame's render, and
  `export_splats`, which writes the PLYs; and the runner's `self.lpips`, called per frame;
* a `__main__` block that parses the argv, calls `cfg.adjust_steps(cfg.steps_scaler)`, and
  hands `main` to `cli(main, cfg, verbose=True)` imported with
  `from gsplat.distributed import cli` -- here the stand-in `gsplat/` beside this file.

Its held-out PSNR is scripted: it climbs linearly until `--plateau-at` (a fraction of the
scaled `max_steps`; 2.0, the default, is never), then holds within a hundredth of a dB --
a converged tail, which the rule should stop, or a rising one, which it should not.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio
from gsplat.distributed import cli

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsplat_stand_in import write_ply


def export_splats(path: str, count: int, seed: int) -> None:
    """v1.5.3's `from gsplat import export_splats`, a module global the loop calls."""
    write_ply(Path(path), count, seed=seed)


class Lpips:
    """The runner's `self.lpips` (torchmetrics' LPIPS in v1.5.3): called once a frame."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, colors: Any, pixels: Any) -> float:
        self.calls += 1
        return 0.2


@dataclass
class MCMCStrategy:
    cap_max: int = 1_000_000
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    refine_every: int = 100


@dataclass
class Config:
    data_dir: str = ""
    result_dir: str = ""
    max_steps: int = 30_000
    steps_scaler: float = 1.0
    eval_steps: list[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: list[int] = field(default_factory=lambda: [7_000, 30_000])
    ply_steps: list[int] = field(default_factory=lambda: [7_000, 30_000])
    save_ply: bool = False
    strategy: MCMCStrategy = field(default_factory=MCMCStrategy)
    plateau_at: float = 2.0
    gaussians: int = 64
    val_frames: int = 3

    def adjust_steps(self, factor: float) -> None:
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.strategy.refine_start_iter = int(self.strategy.refine_start_iter * factor)
        self.strategy.refine_stop_iter = int(self.strategy.refine_stop_iter * factor)
        self.strategy.refine_every = int(self.strategy.refine_every * factor)


class Runner:
    def __init__(self, local_rank: int, world_rank: int, world_size: int, cfg: Config) -> None:
        self.cfg = cfg
        self.world_rank = world_rank
        self.stats_dir = f"{cfg.result_dir}/stats"
        self.render_dir = f"{cfg.result_dir}/renders"
        self.ply_dir = f"{cfg.result_dir}/ply"
        for directory in (self.stats_dir, self.render_dir, self.ply_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)
        self.lpips = Lpips()

    def train(self) -> None:
        cfg = self.cfg
        max_steps = cfg.max_steps
        for step in range(0, max_steps):
            count = min(cfg.gaussians, cfg.strategy.cap_max)
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                stats = {"mem": 0.1, "ellipse_time": 1.0 + step, "num_GS": count}
                print("Step: ", step, stats)  # noqa: T201 - the trainer's own line
                path = Path(f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json")
                path.write_text(json.dumps(stats), encoding="utf-8")
            if (step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1) and cfg.save_ply:
                export_splats(f"{self.ply_dir}/point_cloud_{step}.ply", count, step)
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)

    def eval(self, step: int, stage: str = "val") -> None:
        cfg = self.cfg
        rise = min(step, int(cfg.plateau_at * cfg.max_steps))
        psnr = 20.0 + 5.0 * rise / max(1, cfg.max_steps) + 0.00001 * step
        scores = []
        for index in range(cfg.val_frames):
            imageio.imwrite(f"{self.render_dir}/{stage}_step{step}_{index:04d}.png", b"png")
            scores.append(float(self.lpips(None, None)))
        lpips = sum(scores) / len(scores)
        stats = {"psnr": psnr, "ssim": 0.8, "lpips": lpips, "ellipse_time": 0.01, "num_GS": 64}
        print(f"PSNR: {psnr:.3f}, SSIM: 0.8000, LPIPS: {lpips:.3f} Time: 0.010s/image")  # noqa: T201
        with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as handle:
            json.dump(stats, handle)


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config) -> None:
    runner = Runner(local_rank, world_rank, world_size, cfg)
    runner.train()


def _parse(argv: list[str]) -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=30_000)
    parser.add_argument("--steps_scaler", type=float, default=1.0)
    parser.add_argument("--save_ply", action="store_true")
    for name in ("--ply_steps", "--save_steps", "--eval_steps"):
        parser.add_argument(name, type=int, nargs="+", default=[])
    parser.add_argument("--strategy.cap-max", dest="cap_max", type=int, default=1_000_000)
    parser.add_argument("--plateau-at", dest="plateau_at", type=float, default=2.0)
    parser.add_argument("--gaussians", type=int, default=64)
    args, _unknown = parser.parse_known_args(argv)
    return Config(
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        max_steps=args.max_steps,
        steps_scaler=args.steps_scaler,
        eval_steps=list(args.eval_steps),
        save_steps=list(args.save_steps),
        ply_steps=list(args.ply_steps),
        save_ply=args.save_ply,
        strategy=MCMCStrategy(cap_max=args.cap_max),
        plateau_at=args.plateau_at,
        gaussians=args.gaussians,
    )


if __name__ == "__main__":
    cfg: Any = _parse(sys.argv[1:])
    cfg.adjust_steps(cfg.steps_scaler)
    cli(main, cfg, verbose=True)
