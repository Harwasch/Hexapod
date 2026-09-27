"""A stand-in for `lod_optimise.py`. **It optimises nothing and renders nothing.**

It takes the argv `lod_parents.h3dgs_parents` gives the real script, builds the tree with
the packer's own code -- the part that must be real for the hand-over to be tested -- and
writes "optimised" parents that are the merged ones moved 1 cm up, keyed and fingerprinted
exactly as the real script keys them, so a test can see `package` draw them.

`LOD_STAND_IN` picks how it behaves: `ok` (default), `fail` (exits 1 after printing),
`rejected` (the real script's verdict when the held-out loss did not fall: a summary and
no parents).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import captures_bridge  # noqa: E402, F401 - puts tools/captures on the path
import splat_tiles  # noqa: E402

LIFT = 0.01


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("--ply", "--data_dir", "--placement", "--out", "--trainer"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--opacity_min", type=float, required=True)
    parser.add_argument("--tile_gaussians", type=int, required=True)
    parser.add_argument("--budget-s", dest="budget_s", type=float, required=True)
    for name in ("--tau_min", "--tau_max", "--iterations", "--max_iterations", "--seed"):
        parser.add_argument(name, type=float)
    args = parser.parse_args()
    mode = os.environ.get("LOD_STAND_IN", "ok")
    if not (args.data_dir / "images").is_dir() or not (args.data_dir / "sparse" / "0").is_dir():
        sys.stderr.write(f"stand-in: no images/ or sparse/0/ under {args.data_dir}\n")
        return 2
    placement = json.loads(args.placement.read_text(encoding="utf-8"))
    assert set(placement) >= {"scale", "rotation", "translation"}
    if mode == "fail":
        sys.stdout.write("stand-in: failing as asked\n")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    emitted: dict[str, tuple[Any, Any]] = {}

    def emit(tile: Any, gaussians: Any, keys: Any) -> None:
        emitted[tile.uri] = (gaussians, keys)

    budget = args.tile_gaussians or None
    root = splat_tiles.hierarchy(args.ply, args.opacity_min, budget, args.out, emit)
    parents = [tile for tile in root.walk() if tile.children]
    summary: dict[str, Any] = {
        "status": "rejected" if mode == "rejected" else "ok",
        "reason": "written by tests/lod_parents_stand_in.py -- not an optimisation",
        "parentTiles": len(parents),
        "parentGaussians": sum(tile.count for tile in parents),
        "iterationsRun": 7,
        "before": {"loss": 0.1, "psnr": 20.0, "lpips": 0.3},
        "after": {"loss": 0.08, "psnr": 21.0, "lpips": 0.25},
        "switchDistance": {"mean": {"merged": {"lpips": 0.3}, "optimised": {"lpips": 0.2}}},
    }
    if mode != "rejected" and parents:
        parts = [emitted[tile.uri] for tile in parents]
        offsets = np.cumsum([0] + [int(keys.size) for _, keys in parts])

        def cat(name: str) -> np.ndarray:
            return np.concatenate([getattr(g, name) for g, _ in parts])

        splat_tiles.ParentOverrides(
            source_sha256=splat_tiles.file_sha256(args.ply),
            opacity_min=args.opacity_min,
            tile_gaussians=budget,
            uris=[tile.uri for tile in parents],
            offsets=offsets.astype(np.int64),
            keys=np.concatenate([keys for _, keys in parts]),
            gaussians=splat_tiles.Gaussians(
                xyz=cat("xyz") + np.array([0.0, 0.0, LIFT], dtype=np.float32),
                sh0=cat("sh0"),
                opacity_logit=cat("opacity_logit"),
                log_scales=cat("log_scales"),
                quat_xyzw=cat("quat_xyzw"),
            ),
        ).save(args.out / "lod_parents.npz")
    (args.out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    sys.stdout.write(f"stand-in: {summary['status']}, {len(parents)} parent tiles\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
