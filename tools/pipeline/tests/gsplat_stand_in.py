"""A stand-in for gsplat's trainer. **It does not train anything.**

It exists so the `train` stage's *dispatch* can be tested on a machine with no GPU: it
accepts the argv `training.gsplat_argv` builds and writes what gsplat v1.5.3's
`examples/simple_trainer.py` writes, where it writes it -- read from that file at that
tag, not remembered: `ckpts/ckpt_<i>_rank0.pt`, `stats/val_step<i:04d>.json`, and
`ply/point_cloud_<i>.ply` **only when `--save_ply` is passed**, where `<i>` is the
zero-based index of the last step; and it prints the trainer's own progress line,
`Step:  <i> {...}`. It can SIGTERM itself mid-run so a preemption is a real signal
rather than a mock. Every number it writes is made up, and no test in this repository
reads one of them as if it were a measurement.

Like the real trainer it cannot resume: a second attempt starts again at step 0. What it
is genuinely evidence for: that the stage builds a COLMAP dataset the trainer can find,
passes the switches without which v1.5.3 writes no PLY or writes it in another frame,
and reads whatever the trainer writes back into `train_metrics.json` and `trained.ply`.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

PLY_PROPERTIES = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def write_ply(path: Path, count: int, seed: int) -> None:
    """A valid 3DGS PLY with the fourteen properties `gaussians.read_splat` reads."""
    rng = np.random.default_rng(seed)
    rows = np.zeros(count, dtype=np.dtype([(name, "<f4") for name in PLY_PROPERTIES]))
    for axis in ("x", "y", "z"):
        rows[axis] = rng.normal(scale=1.5, size=count).astype(np.float32)
    for index in range(3):
        rows[f"f_dc_{index}"] = rng.normal(scale=0.4, size=count).astype(np.float32)
        rows[f"scale_{index}"] = np.full(count, -3.0, dtype=np.float32)
    rows["opacity"] = np.full(count, 2.0, dtype=np.float32)
    rows["rot_0"] = np.ones(count, dtype=np.float32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment written by tests/gsplat_stand_in.py -- not a trained splat\n"
        f"element vertex {count}\n"
        + "".join(f"property float {name}\n" for name in PLY_PROPERTIES)
        + "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode("ascii") + rows.tobytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--data_factor", type=int, default=1)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--disable_viewer", action="store_true")
    # v1.5.3's own switches, spelled as `training.gsplat_argv` spells them.
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--steps_scaler", type=float, default=1.0)
    parser.add_argument("--strategy.cap-max", dest="cap_max", type=int, default=None)
    parser.add_argument("--no-normalize-world-space", action="store_true")
    # Not gsplat's: how this stand-in is told to behave like a reclaimed machine.
    parser.add_argument("--ckpt-every", type=int, default=100)
    parser.add_argument("--die-at", type=int, default=None)
    parser.add_argument("--die-marker", type=Path, default=None)
    # A provider reclaiming a box kills the container's main process, not just the
    # trainer inside it -- which for `SubprocessAdapter` is `run_stage.py`, this
    # process's parent. Opt-in, and passed by exactly one test, because under
    # `LocalRunner` the parent is whatever is running the suite.
    parser.add_argument("--kill-parent", action="store_true")
    parser.add_argument("--sync-grace-s", type=float, default=1.0)
    parser.add_argument("--gaussians", type=int, default=128)
    # Stands in for the case `training.py` names out loud: gsplat's output file
    # names are transcribed from its repository, not observed here, so "the trainer
    # put its splat somewhere this project does not look" is a real possibility.
    parser.add_argument("--no-ply", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    # What v1.5.3's `Config.adjust_steps` does to the one number this stand-in uses.
    args.max_steps = int(args.max_steps * args.steps_scaler)
    if args.cap_max is not None:
        if args.strategy != "mcmc":
            sys.stderr.write("stand-in: only mcmc has a cap_max\n")
            return 2
        args.gaussians = min(args.gaussians, args.cap_max)

    images = args.data_dir / "images"
    sparse = args.data_dir / "sparse" / "0"
    if not (images.is_dir() and any(images.iterdir())):
        sys.stderr.write(f"no images under {images}\n")
        return 2
    if not (sparse.is_dir() and any(sparse.iterdir())):
        sys.stderr.write(f"no COLMAP model under {sparse}\n")
        return 2

    if not args.no_normalize_world_space:
        sys.stderr.write("stand-in: refusing to train in a normalised world frame\n")
        return 2
    start = 0
    sys.stdout.write(f"stand-in: strategy={args.strategy} starting at step {start}\n")
    ckpts = args.result_dir / "ckpts"
    ckpts.mkdir(parents=True, exist_ok=True)
    # Outside `--result_dir`, which the stage wipes on every attempt: the stand-in has to
    # remember it was already killed once, or the retry would be killed too.
    died = args.die_marker or (args.result_dir / "died-once")

    step = start
    while step < args.max_steps:
        step += 1
        # tqdm's redraw, carriage return and all, which is what the real trainer's loop
        # prints to stderr and what `progress.parse` reads back.
        sys.stderr.write(
            f"\rloss=0.1| sh degree=3| : {100 * step // args.max_steps:3d}%|#| "
            f"{step}/{args.max_steps} [00:0{min(step, 9)}<00:01, 99.00it/s]"
        )
        if step % args.ckpt_every == 0 or step == args.max_steps:
            index = step - 1
            (ckpts / f"ckpt_{index}_rank0.pt").write_text(
                json.dumps({"step": index}), encoding="utf-8"
            )
            stats = {"mem": 0.1, "ellipse_time": 0.5, "num_GS": args.gaussians + step}
            sys.stdout.write(f"Step:  {index} {stats}" + "\n")
        if args.die_at is not None and step == args.die_at and not died.exists():
            died.write_text("1", encoding="utf-8")
            sys.stdout.write(f"stand-in: the machine is being taken back at step {step}" + "\n")
            sys.stdout.flush()
            # Long enough that the checkpoint syncer beside this process has copied
            # `checkpoint/` out; a margin, not a race.
            time.sleep(args.sync_grace_s)
            if args.kill_parent:
                os.kill(os.getppid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5.0)

    stats = args.result_dir / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    (stats / f"val_step{step - 1:04d}.json").write_text(
        json.dumps(
            {
                "psnr": 27.5,
                "ssim": 0.8712,
                "lpips": 0.1431,
                "ellipse_time": 0.0123,
                "num_GS": args.gaussians,
            }
        ),
        encoding="utf-8",
    )
    if args.save_ply and not args.no_ply:
        write_ply(
            args.result_dir / "ply" / f"point_cloud_{step - 1}.ply", args.gaussians, seed=step
        )
    sys.stdout.write(f"stand-in: finished at step {step}" + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
