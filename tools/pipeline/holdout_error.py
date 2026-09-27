"""Per-gaussian error on the frames the trainer never saw. Runs on the GPU, after training.

**Not executed on a GPU in this repository yet.** It needs CUDA (gsplat's rasteriser has
no CPU path) and no machine this was written on has one. Its numpy half
(`holdout_maths.py`) is unit-tested; this file is the torch half, written against gsplat
v1.5.3's source (`gsplat/rendering.py`, `examples/datasets/colmap.py`, `exporter.py`,
read at the tag) and run by the `train` stage with the trainer's own interpreter
(`$GSPLAT_PYTHON`, CPython 3.10), exactly as the trainer is.

What it does, per held-out (`val`) frame:

1. **The same frames, the same cameras.** The trainer's own dataset classes
   (`examples/datasets/colmap.py`: `Parser(data_dir, factor, normalize, test_every)` then
   `Dataset(parser, split="val")`) over the same `data_dir` the trainer read -- the
   downscaled `images/` (`train_max_side`) and the cropped `points3D.bin` included -- so
   the split (`index % test_every == 0` of the names sorted), the intrinsics rescaling
   and the undistortion are gsplat's, not a copy of them. `normalize` is False, as the
   stage trains with `--no-normalize-world-space`.
2. **The splat that was exported, not the checkpoint.** The PLY the trainer wrote, read
   here: gsplat's `export_splats` drops any gaussian with a non-finite value, so the
   checkpoint's rows and the PLY's differ by exactly those, and it bakes an appearance
   module (`app_opt`) into the DC colour. Reading the PLY keeps every array below in the
   PLY's row order by construction -- the order `trained.ply` is written in, before the
   stage's own ROI or support-mask crop (which the stage applies to these arrays too).
3. **Rendered as it will be shown.** `trained.ply` keeps only the DC colour
   (`gaussians.CANONICAL_PROPERTIES`), so the error map is of a degree-0 render: the
   accuracy of what is published. The full-SH render's PSNR is recorded beside it
   (`psnrFullSh`) so the cost of dropping the higher bands is visible.
4. **Error map**: `holdout_maths.error_map`, the trainer's loss per pixel.
5. **Attribution**: one more render with a zero two-channel "colour" per gaussian and one
   backward pass: the gradient is `sum_p e(p) w_i(p)` and `sum_p w_i(p)`, with `w_i(p)`
   gaussian `i`'s blending weight (`alpha * transmittance`) at pixel `p`. gsplat 1.5.3
   exposes no per-gaussian contribution output of its own (`rasterization`'s `meta` has
   `radii`, `means2d`, tile intersections -- visibility of the footprint, not its
   occluded contribution), so the gradient is the direct way to get the latter.

Output, in `--out`: `holdout_error.npy` (float32, per gaussian mean error, NaN where no
held-out frame gave it weight), `holdout_weight.npy` (float32, pixels of held-out
evidence), `holdout_views.npy` (uint16, held-out frames it painted at least half a pixel
of) and `holdout.json` (per frame PSNR, bias, sharpness; frames used; seconds).

A watchdog ends the process after `--budget-s` whatever it is doing, so a hang here can
never hold the stage -- and the trained splat already written -- until the tier's timeout.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import holdout_maths

_PLY_TYPES = {
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "ushort": "u2",
    "uint16": "u2",
    "short": "i2",
    "int16": "i2",
    "uint": "u4",
    "uint32": "u4",
    "int": "i4",
    "int32": "i4",
}


def read_ply(path: Path) -> dict[str, np.ndarray]:
    """Every vertex property of a binary little-endian PLY, in file row order."""
    raw = path.read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        raise ValueError(f"{path.name} is not a PLY: no end_header")
    fields: list[tuple[str, str]] = []
    count = None
    in_vertex = False
    for line in raw[:end].decode("ascii", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and parts[1] != "binary_little_endian":
            raise ValueError(f"{path.name} is {parts[1]}, not binary_little_endian")
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
            elif count is not None:
                break
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise ValueError(f"{path.name} has a list property in its vertices")
            fields.append((parts[2], "<" + _PLY_TYPES[parts[1]]))
    if count is None:
        raise ValueError(f"{path.name} declares no vertex element")
    record = np.frombuffer(raw, dtype=np.dtype(fields), count=count, offset=end + len(marker))
    return {name: np.asarray(record[name]) for name, _ in fields}


def sh_rest(columns: dict[str, np.ndarray]) -> np.ndarray | None:
    """`f_rest_*` as (N, K, 3), undoing `export_splats`' channel-major flattening."""
    names = sorted(
        (name for name in columns if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if not names or len(names) % 3:
        return None
    flat = np.stack([columns[name] for name in names], axis=1).astype(np.float32)
    bands = len(names) // 3
    # exporter.py: shN (N, K, 3) -> permute(0, 2, 1) -> reshape(N, K * 3).
    return np.ascontiguousarray(flat.reshape(-1, 3, bands).transpose(0, 2, 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data_dir", type=Path, help="the dataset the trainer read")
    parser.add_argument("--ply", type=Path, help="the PLY the trainer exported")
    parser.add_argument("--out", type=Path, help="where to write the arrays and holdout.json")
    parser.add_argument("--trainer", type=Path, help="gsplat's examples/simple_trainer.py")
    parser.add_argument("--data_factor", type=int, default=1)
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--antialiased", action="store_true")
    parser.add_argument("--ssim_weight", type=float, default=holdout_maths.SSIM_WEIGHT)
    parser.add_argument("--budget-s", dest="budget_s", type=float, default=900.0)
    # Not for the pipeline: gsplat has no CPU rasteriser. For checking this file's
    # plumbing on a CPU against a stand-in `gsplat.rasterization`.
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    parser.add_argument(
        "--self-check",
        dest="self_check",
        action="store_true",
        help="import torch, gsplat and the trainer's dataset classes, then exit",
    )
    args = parser.parse_args(argv)

    watchdog = threading.Timer(args.budget_s, _out_of_time, args=(args.budget_s,))
    watchdog.daemon = True
    watchdog.start()
    if args.trainer is None:
        trainer = os.environ.get("GSPLAT_TRAINER")
        args.trainer = Path(trainer) if trainer else None
    if args.trainer is None:
        sys.stderr.write("holdout: no --trainer and no $GSPLAT_TRAINER\n")
        return 2
    sys.path.insert(0, str(args.trainer.resolve().parent))
    import torch
    from datasets.colmap import Dataset, Parser
    from gsplat import rasterization

    if args.self_check:
        sys.stdout.write(f"holdout: torch {torch.__version__}, rasterization and datasets ok\n")
        return 0
    for name in ("data_dir", "ply", "out"):
        if getattr(args, name) is None:
            sys.stderr.write(f"holdout: --{name} is required\n")
            return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.stderr.write("holdout: no CUDA device; gsplat's rasteriser has no CPU path\n")
        return 2
    started = time.perf_counter()
    device = torch.device(args.device)
    columns = read_ply(args.ply)
    count = int(columns["x"].shape[0])

    def tensor(names: list[str]) -> Any:
        stacked = np.stack([columns[name] for name in names], axis=1).astype(np.float32)
        return torch.from_numpy(stacked).to(device)

    means = tensor(["x", "y", "z"])
    quats = tensor(["rot_0", "rot_1", "rot_2", "rot_3"])
    scales = torch.exp(tensor(["scale_0", "scale_1", "scale_2"]))
    opacities = torch.sigmoid(tensor(["opacity"])[:, 0])
    sh0 = tensor(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None, :]
    rest = sh_rest(columns)
    full = None if rest is None else torch.cat([sh0, torch.from_numpy(rest).to(device)], dim=1)
    full_degree = None if full is None else round(math.sqrt(full.shape[1])) - 1
    mode = "antialiased" if args.antialiased else "classic"

    def render(colors: Any, viewmat: Any, k: Any, width: int, height: int, degree: Any) -> Any:
        return rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=k,
            width=width,
            height=height,
            sh_degree=degree,
            packed=False,
            rasterize_mode=mode,
        )

    colmap = Parser(
        data_dir=str(args.data_dir),
        factor=args.data_factor,
        normalize=False,
        test_every=args.test_every,
    )
    valset = Dataset(colmap, split="val")
    accumulator = holdout_maths.Accumulator(count)
    views: list[dict[str, Any]] = []
    for item in range(len(valset)):
        data = valset[item]
        name = colmap.image_names[int(valset.indices[item])]
        k = data["K"][None].to(device)
        viewmat = torch.linalg.inv(data["camtoworld"][None].to(device))
        real = data["image"].numpy().astype(np.float32) / 255.0
        height, width = int(real.shape[0]), int(real.shape[1])
        mask = data["mask"].numpy().astype(bool) if "mask" in data else None
        with torch.no_grad():
            rgb, alpha, _ = render(sh0, viewmat, k, width, height, 0)
            shown = rgb[0].clamp(0.0, 1.0).cpu().numpy()
            coverage = alpha[0, ..., 0].cpu().numpy()
            if mask is not None:
                coverage = np.where(mask, coverage, np.float32(0.0))
            psnr_full = None
            if full is not None:
                # Over the same pixels as `view_stats`' psnr: the ones the splat covers.
                rgb_full, _, _ = render(full, viewmat, k, width, height, full_degree)
                psnr_full = holdout_maths.psnr(rgb_full[0].cpu().numpy(), real, coverage > 0.5)
        error = holdout_maths.error_map(shown, real, ssim_weight=args.ssim_weight)
        if mask is not None:
            error = np.where(mask, error, np.float32(0.0)).astype(np.float32)
        stats = holdout_maths.view_stats(shown, real, error, coverage)
        stats["name"] = name
        stats["psnrFullSh"] = None if psnr_full is None else round(psnr_full, 3)
        views.append(stats)

        # The attribution pass: colours are two probes, the error map and a ones map.
        probe = torch.zeros((count, 3), device=device, requires_grad=True)
        painted, _, _ = render(probe, viewmat, k, width, height, None)
        weight_map = torch.ones((height, width), device=device)
        if mask is not None:
            weight_map = torch.from_numpy(mask.astype(np.float32)).to(device)
        error_map = torch.from_numpy(error).to(device)
        objective = (painted[0, ..., 0] * error_map).sum() + (painted[0, ..., 1] * weight_map).sum()
        objective.backward()
        # None only if nothing reached the probe (no gaussian in this frame at all).
        grad = torch.zeros_like(probe) if probe.grad is None else probe.grad.detach()
        accumulator.add(grad[:, 0].cpu().numpy(), grad[:, 1].cpu().numpy())
        del probe, painted, objective, grad
        sys.stdout.write(
            f"holdout: frame {item + 1} of {len(valset)} {name}: psnr={stats['psnr']} "
            f"error={stats['meanError']} bias={stats['bias']} sharpness={stats['sharpness']}\n"
        )

    error = accumulator.mean_error()
    measured = np.isfinite(error)
    psnrs = [view["psnr"] for view in views if view["psnr"] is not None]
    summary: dict[str, Any] = {
        "status": "ok",
        "gaussians": count,
        "gaussiansMeasured": int(measured.sum()),
        "views": len(views),
        "perView": views,
        "meanPsnr": round(float(np.mean(psnrs)), 3) if psnrs else None,
        "medianError": (round(float(np.median(error[measured])), 5) if measured.any() else None),
        "rendered": "dc-only (as trained.ply ships it)",
        "rasterizeMode": mode,
        "ssimWeight": args.ssim_weight,
        "testEvery": args.test_every,
        "seconds": round(time.perf_counter() - started, 2),
        "method": (
            "per-pixel error (1-w)*L1 + w*(1-SSIM) of each held-out frame, shared among "
            "gaussians by blending weight alpha*T (the gradient of a linear colour probe); "
            "mean error = sum(e*w)/sum(w) over every held-out frame"
        ),
    }
    holdout_maths.write(
        args.out, error, accumulator.weight_sum.astype(np.float32), accumulator.views, summary
    )
    sys.stdout.write(
        f"holdout: {len(views)} held-out frames, {int(measured.sum())} of {count} gaussians "
        f"measured, mean psnr={summary['meanPsnr']}, {summary['seconds']} s\n"
    )
    return 0


def _out_of_time(budget: float) -> None:
    sys.stderr.write(f"holdout: out of time after {budget:g} s; stopping\n")
    sys.stderr.flush()
    os._exit(3)


if __name__ == "__main__":
    raise SystemExit(main())
