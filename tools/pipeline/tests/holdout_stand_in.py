"""A stand-in for `holdout_error.py`. **It renders nothing and measures nothing.**

It takes the argv `holdout.measure` gives the real script, reads the PLY the trainer (or
`gsplat_stand_in.py`) exported, and writes the files the real one writes, where it writes
them, with made-up values that are a known function of each gaussian's position -- so a
test can check that the stage keeps them aligned with `trained.ply`'s rows through its
crop, which is what this is evidence for, and nothing about accuracy.

`HOLDOUT_STAND_IN` picks how it behaves: `ok` (default), `fail` (exits 1 after printing),
`short` (arrays one row short: a length the stage must refuse).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import holdout_maths
from holdout_error import read_ply


def made_up_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The "error" each gaussian gets: a known function of its position."""
    return (0.05 + 0.01 * np.abs(x) + 0.001 * np.abs(y)).astype(np.float32)


def seen(x: np.ndarray) -> np.ndarray:
    """Which gaussians a held-out frame "saw": everything with x above -2."""
    return x > -2.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trainer", type=Path)
    parser.add_argument("--data_factor", type=int, default=1)
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--budget-s", dest="budget_s", type=float, default=900.0)
    parser.add_argument("--antialiased", action="store_true")
    # The degree trained.ply ships, which the real script renders at; recorded, so a test
    # can see the stage asked for what it shipped.
    parser.add_argument("--sh-degree", dest="sh_degree", type=int, default=0)
    args = parser.parse_args()
    mode = os.environ.get("HOLDOUT_STAND_IN", "ok")
    if not (args.data_dir / "images").is_dir():
        sys.stderr.write(f"stand-in: no images under {args.data_dir}\n")
        return 2
    if mode == "fail":
        sys.stdout.write("stand-in: failing as asked\n")
        return 1
    columns = read_ply(args.ply)
    x, y = columns["x"], columns["y"]
    visible = seen(x)
    error = np.where(visible, made_up_error(x, y), np.float32(np.nan)).astype(np.float32)
    weight = np.where(visible, np.float32(10.0), np.float32(0.0)).astype(np.float32)
    views = visible.astype(np.uint16) * 2
    if mode == "short":
        error, weight, views = error[:-1], weight[:-1], views[:-1]
    summary = {
        "status": "ok",
        "gaussians": int(x.shape[0]),
        "gaussiansMeasured": int(visible.sum()),
        "views": 2,
        "perView": [
            {"name": "frame_0000.jpg", "psnr": 24.0, "bias": 0.01, "sharpness": 1.0},
            {"name": "frame_0008.jpg", "psnr": 22.0, "bias": -0.01, "sharpness": 0.95},
        ],
        "meanPsnr": 23.0,
        "seconds": 0.01,
        "antialiased": args.antialiased,
        "shDegree": args.sh_degree,
        "note": "written by tests/holdout_stand_in.py -- not a measurement",
    }
    holdout_maths.write(args.out, error, weight, views, summary)
    sys.stdout.write(f"stand-in: wrote held-out arrays for {x.shape[0]} gaussians\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
