"""Optimise a tileset's parent gaussians against the photos. Runs on the GPU, after `place`.

**Not executed on a GPU in this repository yet.** gsplat's rasteriser is CUDA only and no
machine this was written on has one. The numpy half (`lod_maths.py`) is unit-tested; this
torch half is exercised on a CPU by tests/test_lod_optimise.py against a torch reference
rasteriser standing in for `gsplat.rasterization` (tests/lod_stand_in/), and the image
build imports it with the trainer's own interpreter (`--self-check`, infra/modal/app.py).

This is Hierarchical 3DGS's hierarchy optimisation (Kerbl et al. 2024, 2406.12080,
Sec. 5.1 and 7.3; code: train_post.py, scripts/full_train.py) applied to the tree
`tools/captures/splat_tiles.py` packs:

1. **The same tree, built by the same code.** `splat_tiles.hierarchy` on the very
   `canonical.ply` `package` will pack, with `package`'s `opacity_min` and
   `tile_gaussians`: the packer is deterministic given those three, so every tile, box,
   error and merged cell here is the one it will write. The result goes back as a
   `splat_tiles.ParentOverrides` file keyed by tile and cell and fingerprinted by the PLY's
   sha256, which the packer refuses to apply to anything else.
2. **Leaves frozen** (Sec. 5.1: "we do not change the leaf nodes during optimization"):
   the leaves are plain tensors; only parent means, log-scales, rotations, opacities and
   DC colours are parameters.
3. **Each iteration** a random training view and a random target granularity tau,
   log-uniform in [tau_min, tau_max] (Sec. 5.1, tau = tau_max^xi tau_min^(1 - xi);
   train_post.py `limit = 2^(sample * (log2 limmax - log2 limmin) + log2 limmin)`), the
   cut of the tree at tau -- the viewer's cut, `lod_maths.cut` -- rendered **at full
   resolution** (Sec. 5.1: "we always render at full resolution during optimization, and
   simply choose random target granularities"), against the photo, with the trainer's
   loss, (1 - 0.2) L1 + 0.2 (1 - SSIM). A (view, tau) whose cut has no parent in view is
   drawn again: it would render only frozen leaves and move nothing.
4. **No interpolation** between levels (Sec. 4.2, Eq. 10). H3DGS trains parents and
   children blended because its renderer blends them; ours -- CesiumJS and Spark -- swap
   a tile for its children outright, so the parents are trained as they will be drawn.
5. **Learning rates** are train_post.py's (`lod_maths`: full_train.py's
   `--feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001`, the defaults' rotation
   0.001 and positions 2e-5 -> 2e-7 times the scene radius), with Adam at eps 1e-15.
6. **Opacity at most 0.99**, because SPZ and KHR_gaussian_splatting store an opacity in
   [0, 1] (the packer cuts H3DGS's merged "falloff" there too, `MERGED_OPACITY_MAX`).
   H3DGS optimises parents with an absolute-value opacity activation and, because its
   rasteriser clamps each fragment's alpha at 0.99, zeroes a gaussian's opacity gradient
   wherever that clamp is hit (Sec. 7.3). Both are kept: the activation is `abs`, the
   per-fragment rule is gsplat's own (RasterizeToPixels3DGSBwd.cu: the alpha and opacity
   gradients are skipped when `opac * vis > 0.999`), and on top of it the stored opacity
   is clamped to 0.99 -- whose gradient is zero past the clamp -- and projected back
   into [0, 0.99] after every step, so no parent ever needs more than SPZ can store.
7. **Kept inside its box.** Each parent's mean is projected into its tile's bounding box
   after every step, and log-scales into what SPZ can store: the box Cesium culls and
   measures distance by, and the error it refines on, are the merge's, unchanged
   (`splat_tiles.write_tiles`), so the viewer's cut is the one trained on.
8. **The target is what the published splat covers.** `quality` crops what the capture
   did not support, so a photo holds content the leaves do not; H3DGS's chunks are whole
   (a skybox included), ours are not. Each photo is weighted by the leaves' own alpha in
   that view (rendered once): parents learn to look like the photo *where the published
   splat is*, and are not paid to grow into what `quality` removed.

How long: `lod_maths.plan_iterations` -- as many iterations as give each parent about as
many optimisation steps as H3DGS's 15,000 give each of its nodes, measured on this tree and
these cameras, never more than 15,000.

**Checked, not assumed.** Before and after, every held-out frame (every 8th, never trained
on -- the trainer's own split) is rendered at fixed taus with the cut, and the parents are
kept only if the held-out loss over the cuts that contain one went down; otherwise the
merge's parents stand (status `rejected`). The Phase 1 measure is repeated too: each
parent tile alone, from where its error projects to 16 px, against its own subtree's
leaves -- PSNR, SSIM and (when torchmetrics' AlexNet is there) LPIPS, merged and optimised.

Output, in `--out`: `summary.json` always, and `lod_parents.npz` when the optimised
parents were kept. A watchdog ends the process after `--budget-s` whatever it is doing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "captures"))

import lod_maths  # noqa: E402
import splat_tiles  # noqa: E402

#: The files this writes in `--out`.
SUMMARY_FILE = "summary.json"
PARENTS_FILE = "lod_parents.npz"

#: Where the viewer's own `maximumScreenSpaceError` lives, in device pixels: the splat floor
#: per preset (4 / 8 / 12, apps/web PerformanceManager `splatMinimumScreenSpaceError`)
#: times the Detail scale (x0.71 .. x1.41, lib/detail.ts) times the device pixel ratio
#: (1..3). 3 px is also H3DGS's tau_min (Sec. 5.2: "the minimum extent of a 3DGS primitive
#: due to low-pass filtering"), and 64 keeps H3DGS's ratio of about 20 (train_post.py:
#: 0.005 .. 0.1).
TAU_MIN = 3.0
TAU_MAX = 64.0

#: Held-out taus: around Cesium's default of 16, where Phase 1 found the merge blurry.
EVAL_TAUS = (4.0, 8.0, 16.0, 32.0)

#: The Phase 1 switch-distance protocol (scratchpad lodcmp/compare.py): 480 x 360, a 50
#: degree vertical field of view, four azimuths 25 degrees up, the parent drawn from
#: where its error projects to 16 px.
SWITCH_TAU = 16.0
SWITCH_SIZE = (480, 360)
SWITCH_FOV_DEG = 50.0
SWITCH_AZIMUTHS_DEG = (20.0, 110.0, 200.0, 290.0)
SWITCH_ELEVATION_DEG = 25.0
#: The largest parent tiles measured that way; a root and its first levels in practice.
SWITCH_TILES = 12

#: How many draws of (view, tau) may come up with no parent in view before an iteration is
#: given up (and counted).
DRAWS = 64

OPACITY_MAX = splat_tiles.MERGED_OPACITY_MAX
LOG_SCALE_RANGE = splat_tiles.SPZ_LOG_SCALE_RANGE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--ply", type=Path, help="canonical.ply: what package packs")
    parser.add_argument("--data_dir", type=Path, help="images/ and sparse/0/, as trained")
    parser.add_argument("--placement", type=Path, help="place's placement.json")
    parser.add_argument("--out", type=Path, help="where summary.json and the parents go")
    parser.add_argument("--trainer", type=Path, help="gsplat's examples/simple_trainer.py")
    parser.add_argument("--opacity_min", type=float, default=0.02)
    parser.add_argument("--tile_gaussians", type=int, default=splat_tiles.TILE_GAUSSIANS)
    parser.add_argument("--tau_min", type=float, default=TAU_MIN)
    parser.add_argument("--tau_max", type=float, default=TAU_MAX)
    parser.add_argument("--iterations", type=int, default=0, help="0: planned from the tree")
    parser.add_argument("--max_iterations", type=int, default=lod_maths.H3DGS_ITERATIONS)
    parser.add_argument("--data_factor", type=int, default=1)
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-s", dest="budget_s", type=float, default=3600.0)
    parser.add_argument("--no-lpips", dest="lpips", action="store_false")
    # Not for the pipeline: gsplat has no CPU rasteriser. For checking this file on a CPU
    # against a stand-in `gsplat.rasterization`.
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    parser.add_argument(
        "--switch-size",
        dest="switch_size",
        type=int,
        nargs=2,
        default=list(SWITCH_SIZE),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--self-check",
        dest="self_check",
        action="store_true",
        help="import torch, gsplat, the trainer's datasets and the packer, then exit",
    )
    args = parser.parse_args(argv)

    started = time.monotonic()
    # The hard stop, past the soft one the loop keeps (`budget_s`): evaluation and writing
    # come after the loop, and a hang in any of it must still end.
    watchdog = threading.Timer(args.budget_s + 900.0, _out_of_time, args=(args.budget_s,))
    watchdog.daemon = True
    watchdog.start()
    if args.trainer is None:
        trainer = os.environ.get("GSPLAT_TRAINER")
        args.trainer = Path(trainer) if trainer else None
    if args.trainer is None:
        sys.stderr.write("lod: no --trainer and no $GSPLAT_TRAINER\n")
        return 2
    sys.path.insert(0, str(args.trainer.resolve().parent))
    import torch
    from datasets.colmap import Dataset, Parser
    from gsplat import rasterization

    if args.self_check:
        sys.stdout.write(
            f"lod: torch {torch.__version__}, rasterization, datasets and splat_tiles ok\n"
        )
        return 0
    for name in ("ply", "data_dir", "placement", "out"):
        if getattr(args, name) is None:
            sys.stderr.write(f"lod: --{name} is required\n")
            return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.stderr.write("lod: no CUDA device; gsplat's rasteriser has no CPU path\n")
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / PARENTS_FILE).unlink(missing_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # --- the tree, exactly as package will build it --------------------------------
    emitted: dict[str, tuple[Any, np.ndarray | None]] = {}

    def emit(tile: Any, gaussians: Any, keys: np.ndarray | None) -> None:
        emitted[tile.uri] = (gaussians, keys)

    budget = args.tile_gaussians or None
    # The packer's sorted working copy goes in a temporary directory under `--out`, and is
    # gone again before `hierarchy` returns.
    root = splat_tiles.hierarchy(args.ply, args.opacity_min, budget, args.out, emit)
    tiles = root.walk()
    tree = lod_maths.tree_from(tiles, [int(emitted[t.uri][0].xyz.shape[0]) for t in tiles])
    sha = splat_tiles.file_sha256(args.ply)
    summary: dict[str, Any] = {
        "sourceSha256": sha,
        "opacityMin": args.opacity_min,
        "tileGaussians": budget,
        "tiles": len(tiles),
        "parentTiles": len(tree.parents),
        "parentGaussians": int(sum(tree.rows[t, 1] - tree.rows[t, 0] for t in tree.parents)),
        "leafGaussians": int(sum(tree.rows[t, 1] - tree.rows[t, 0] for t in tree.leaves)),
        "tauRange": [args.tau_min, args.tau_max],
    }
    if not tree.parents:
        return _finish(args.out, summary, "skipped", "the tileset is one tile: no parents")

    # --- cameras, in the placed frame ----------------------------------------------
    placement = json.loads(args.placement.read_text(encoding="utf-8"))
    rotation = np.asarray(placement["rotation"], dtype=np.float64)
    translation = np.asarray(placement["translation"], dtype=np.float64)
    scale = float(placement["scale"])
    colmap = Parser(
        data_dir=str(args.data_dir),
        factor=args.data_factor,
        normalize=False,
        test_every=args.test_every,
    )
    trainset = Dataset(colmap, split="train")
    valset = Dataset(colmap, split="val")

    def view_of(data: dict[str, Any]) -> tuple[lod_maths.Camera, Any, Any]:
        c2w = lod_maths.placed_camtoworld(
            data["camtoworld"].numpy().astype(np.float64), rotation, translation, scale
        )
        image = data["image"]
        height, width = int(image.shape[0]), int(image.shape[1])
        camera = lod_maths.Camera(
            np.linalg.inv(c2w), data["K"].numpy().astype(np.float64), width, height
        )
        mask = data.get("mask")
        return camera, image.to(device), None if mask is None else mask.to(device)

    loaded = [view_of(trainset[i]) for i in range(len(trainset))]
    if not loaded:
        return _finish(args.out, summary, "skipped", "no training frames")
    cameras = [camera for camera, _, _ in loaded]
    masks = [mask for _, _, mask in loaded]
    plan = lod_maths.plan_iterations(
        tree, cameras, args.tau_min, args.tau_max, most=args.max_iterations
    )
    iterations = args.iterations or plan.iterations
    summary["plan"] = {
        "iterations": plan.iterations,
        "selectionRate": round(plan.selection_rate, 4),
        "parentTilesSeen": plan.parents_seen,
        "reason": plan.reason,
    }
    if iterations <= 0:
        return _finish(args.out, summary, "skipped", plan.reason)

    # --- the gaussians: leaves frozen, parents trainable ---------------------------
    leaf_tiles = tree.leaves
    parent_tiles = tree.parents

    def stacked(uris: list[str]) -> dict[str, np.ndarray]:
        parts = [emitted[uri][0] for uri in uris]
        return {
            name: np.concatenate([np.asarray(getattr(part, name)) for part in parts])
            for name in ("xyz", "sh0", "opacity_logit", "log_scales", "quat_xyzw")
        }

    leaves = stacked([tree.uris[t] for t in leaf_tiles])
    merged = stacked([tree.uris[t] for t in parent_tiles])
    leaf_count = int(leaves["xyz"].shape[0])

    def tensor(values: np.ndarray, **kw: Any) -> Any:
        return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).to(device, **kw)

    frozen = {
        "means": tensor(leaves["xyz"]),
        "quats": tensor(leaves["quat_xyzw"][:, [3, 0, 1, 2]]),
        "scales": torch.exp(tensor(np.clip(leaves["log_scales"], *LOG_SCALE_RANGE))),
        "opacities": torch.sigmoid(tensor(leaves["opacity_logit"])),
        "colors": tensor(leaves["sh0"])[:, None, :],
    }
    initial = {
        "means": merged["xyz"],
        "log_scales": np.clip(merged["log_scales"], *LOG_SCALE_RANGE),
        "quats": merged["quat_xyzw"][:, [3, 0, 1, 2]],
        "opacity": 1.0 / (1.0 + np.exp(-merged["opacity_logit"].astype(np.float64))),
        "sh0": merged["sh0"],
    }
    params = {
        name: torch.nn.Parameter(tensor(np.asarray(value))) for name, value in initial.items()
    }
    # Each parent's box: its tile's, which it must stay inside (point 7 of the docstring).
    box_low = np.concatenate(
        [np.repeat(tree.low[t][None], tree.rows[t, 1] - tree.rows[t, 0], 0) for t in parent_tiles]
    )
    box_high = np.concatenate(
        [np.repeat(tree.high[t][None], tree.rows[t, 1] - tree.rows[t, 0], 0) for t in parent_tiles]
    )
    low_t, high_t = tensor(box_low), tensor(box_high)
    rows = {
        t: torch.arange(
            int(tree.rows[t, 0]) - (0 if not tree.children[t] else leaf_count),
            int(tree.rows[t, 1]) - (0 if not tree.children[t] else leaf_count),
            device=device,
        )
        for t in range(len(tree.uris))
    }

    def gaussians_of(chosen: list[int], state: dict[str, Any]) -> dict[str, Any]:
        leaf_rows = [rows[t] for t in chosen if not tree.children[t]]
        parent_rows = [rows[t] for t in chosen if tree.children[t]]
        leaf_index = torch.cat(leaf_rows) if leaf_rows else None
        parent_index = torch.cat(parent_rows) if parent_rows else None
        parts: dict[str, list[Any]] = {k: [] for k in frozen}
        if leaf_index is not None:
            for key in frozen:
                parts[key].append(frozen[key][leaf_index])
        if parent_index is not None:
            parts["means"].append(state["means"][parent_index])
            parts["quats"].append(state["quats"][parent_index])
            parts["scales"].append(torch.exp(state["log_scales"][parent_index]))
            # H3DGS's absolute-value activation (gaussian_model.py create_from_hier), cut
            # at what SPZ can store; the clamp's gradient is zero past it (Sec. 7.3).
            parts["opacities"].append(state["opacity"][parent_index].abs().clamp(max=OPACITY_MAX))
            parts["colors"].append(state["sh0"][parent_index][:, None, :])
        return {key: torch.cat(value) for key, value in parts.items()}

    def render(chosen: list[int], camera: lod_maths.Camera, state: dict[str, Any]) -> Any:
        g = gaussians_of(chosen, state)
        colors, alphas, _ = rasterization(
            means=g["means"],
            quats=g["quats"],
            scales=g["scales"],
            opacities=g["opacities"],
            colors=g["colors"],
            viewmats=tensor(camera.viewmat)[None],
            Ks=tensor(camera.k)[None],
            width=camera.width,
            height=camera.height,
            sh_degree=0,
            packed=False,
            rasterize_mode="classic",
        )
        return colors[0], alphas[0, ..., 0]

    ssim = _ssim_function(torch)
    lpips = _lpips_function(torch, device) if args.lpips else None
    every_leaf = list(leaf_tiles)

    # The target: the photo, weighted by what the published splat covers (point 8).
    def target_of(camera: lod_maths.Camera, image: Any, mask: Any) -> Any:
        with torch.no_grad():
            _, coverage = render(every_leaf, camera, params)
        pixels = image.float() / 255.0 * coverage[..., None].clamp(0.0, 1.0)
        if mask is not None:
            pixels = pixels * mask[..., None]
        return pixels

    # Kept on the device as bytes (a 1600 x 1200 frame is 5.8 MB): the photos themselves
    # are not needed once their targets are made.
    targets = [
        (target_of(camera, image, mask) * 255.0).round().to(torch.uint8)
        for camera, image, mask in loaded
    ]
    del loaded

    def loss_of(colors: Any, pixels: Any, mask: Any) -> tuple[Any, Any, Any]:
        if mask is not None:
            colors = colors * mask[..., None]
        l1 = (colors - pixels).abs().mean()
        s = ssim(colors, pixels)
        return (1.0 - lod_maths.SSIM_LAMBDA) * l1 + lod_maths.SSIM_LAMBDA * (1.0 - s), l1, s

    snapshot = {name: value.detach().clone() for name, value in params.items()}

    def evaluate(state: dict[str, Any], views: list[Any]) -> dict[str, Any]:
        rows_out: list[dict[str, Any]] = []
        with torch.no_grad():
            for index, (camera, pixels, mask) in enumerate(views):
                for tau in EVAL_TAUS:
                    chosen = lod_maths.cut(tree, camera.eye, camera.focal, tau)
                    seen = lod_maths.visible_parents(
                        tree, chosen, camera.viewmat, camera.k, camera.width, camera.height
                    )
                    if not seen:
                        continue
                    colors, _ = render(chosen, camera, state)
                    loss, l1, s = loss_of(colors, pixels, mask)
                    mse = float(((colors.clamp(0, 1) - pixels) ** 2).mean())
                    row = {
                        "view": index,
                        "tau": tau,
                        "loss": float(loss),
                        "l1": float(l1),
                        "ssim": float(s),
                        "psnr": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
                    }
                    if lpips is not None:
                        row["lpips"] = lpips(colors, pixels)
                    rows_out.append(row)
        return _aggregate(rows_out)

    val_views = []
    for i in range(len(valset)):
        camera, image, mask = view_of(valset[i])
        val_views.append((camera, target_of(camera, image, mask), mask))
    held_out = bool(val_views)
    probe = val_views
    before = evaluate(snapshot, probe)
    if before["pairs"] == 0:
        # No held-out frame sees a parent in any of the taus: judge on training frames
        # (what H3DGS itself reports nothing beyond), and say so.
        held_out = False
        picks = rng.choice(len(cameras), size=min(16, len(cameras)), replace=False)
        probe = [(cameras[int(i)], targets[int(i)].float() / 255.0, masks[int(i)]) for i in picks]
        before = evaluate(snapshot, probe)

    # --- the optimisation ------------------------------------------------------------
    radius = lod_maths.spatial_scale(np.array([camera.eye for camera in cameras]))
    optimiser = torch.optim.Adam(
        [
            {"params": [params["means"]], "lr": lod_maths.position_lr(0, iterations, radius)},
            {"params": [params["sh0"]], "lr": lod_maths.FEATURE_LR},
            {"params": [params["opacity"]], "lr": lod_maths.OPACITY_LR},
            {"params": [params["log_scales"]], "lr": lod_maths.SCALING_LR},
            {"params": [params["quats"]], "lr": lod_maths.ROTATION_LR},
        ],
        lr=0.0,
        eps=lod_maths.ADAM_EPS,
    )
    chosen_count = np.zeros(len(tree.uris), dtype=np.int64)
    skipped = 0
    ema = None
    done = 0
    loop_started = time.monotonic()
    stopped = "finished"
    for step in range(iterations):
        if time.monotonic() - started > args.budget_s:
            stopped = f"out of time after {done} iterations"
            break
        pick = None
        for _ in range(DRAWS):
            view = int(rng.integers(len(cameras)))
            camera = cameras[view]
            tau = lod_maths.sample_tau(rng, args.tau_min, args.tau_max)
            chosen = lod_maths.cut(tree, camera.eye, camera.focal, tau)
            seen = lod_maths.visible_parents(
                tree, chosen, camera.viewmat, camera.k, camera.width, camera.height
            )
            if seen:
                pick = (view, chosen, seen)
                break
        if pick is None:
            skipped += 1
            continue
        view, chosen, seen = pick
        chosen_count[seen] += 1
        optimiser.param_groups[0]["lr"] = lod_maths.position_lr(step, iterations, radius)
        colors, _ = render(chosen, cameras[view], params)
        loss, _, _ = loss_of(colors, targets[view].float() / 255.0, masks[view])
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        with torch.no_grad():
            params["opacity"].clamp_(0.0, OPACITY_MAX)
            params["log_scales"].clamp_(*LOG_SCALE_RANGE)
            params["means"].copy_(torch.maximum(torch.minimum(params["means"], high_t), low_t))
        value = float(loss.detach())
        ema = value if ema is None else 0.4 * value + 0.6 * ema
        done += 1
        if done % 500 == 0 or done == 1:
            rate = done / max(time.monotonic() - loop_started, 1e-9)
            sys.stdout.write(f"lod: step {done} of {iterations} loss={ema:.4f} ({rate:.1f} it/s)\n")
            sys.stdout.flush()
    loop_seconds = time.monotonic() - loop_started
    state = {name: value.detach() for name, value in params.items()}
    after = evaluate(state, probe)

    switch = _switch_distance(
        tree, render, snapshot, state, lpips, ssim, torch, emitted, tuple(args.switch_size)
    )
    accepted = after["pairs"] > 0 and after["loss"] < before["loss"]
    summary.update(
        {
            "iterations": iterations,
            "iterationsRun": done,
            "iterationsSkipped": skipped,
            "stopped": stopped,
            "loopSeconds": round(loop_seconds, 2),
            "itPerSecond": round(done / max(loop_seconds, 1e-9), 2),
            "finalLossEma": None if ema is None else round(ema, 5),
            "sceneRadius": round(radius, 4),
            "timesChosen": {tree.uris[t]: int(chosen_count[t]) for t in parent_tiles},
            "judgedOn": "held-out frames" if held_out else "training frames (no held-out cut)",
            "before": before,
            "after": after,
            "switchDistance": switch,
            "accepted": accepted,
            "learningRates": {
                "positionInit": lod_maths.POSITION_LR_INIT * radius,
                "positionFinal": lod_maths.POSITION_LR_FINAL * radius,
                "feature": lod_maths.FEATURE_LR,
                "opacity": lod_maths.OPACITY_LR,
                "scaling": lod_maths.SCALING_LR,
                "rotation": lod_maths.ROTATION_LR,
            },
        }
    )
    if not accepted:
        return _finish(
            args.out,
            summary,
            "rejected",
            f"the held-out loss did not fall ({before['loss']} -> {after['loss']}); the "
            f"merged parents stand",
        )
    _write_parents(args.out / PARENTS_FILE, tree, emitted, state, sha, args, budget)
    return _finish(
        args.out,
        summary,
        "ok",
        f"{summary['parentGaussians']} parents in {len(parent_tiles)} tiles optimised over "
        f"{done} iterations; loss {before['loss']} -> {after['loss']}",
    )


def _write_parents(
    path: Path,
    tree: lod_maths.Tree,
    emitted: dict[str, tuple[Any, np.ndarray | None]],
    state: dict[str, Any],
    sha: str,
    args: argparse.Namespace,
    budget: int | None,
) -> None:
    """The optimised parents as the packer reads them, tile by tile in the table's order."""
    means = state["means"].cpu().numpy().astype(np.float32)
    quats = state["quats"].cpu().numpy().astype(np.float64)
    quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    opacity = np.clip(np.abs(state["opacity"].cpu().numpy().astype(np.float64)), 1e-6, OPACITY_MAX)
    uris: list[str] = []
    offsets = [0]
    keys: list[np.ndarray] = []
    for t in tree.parents:
        cells = emitted[tree.uris[t]][1]
        assert cells is not None
        uris.append(tree.uris[t])
        offsets.append(offsets[-1] + int(cells.size))
        keys.append(cells)
    overrides = splat_tiles.ParentOverrides(
        source_sha256=sha,
        opacity_min=args.opacity_min,
        tile_gaussians=budget,
        uris=uris,
        offsets=np.asarray(offsets, dtype=np.int64),
        keys=np.concatenate(keys).astype(np.int64),
        gaussians=splat_tiles.Gaussians(
            xyz=means,
            sh0=state["sh0"].cpu().numpy().astype(np.float32),
            opacity_logit=np.log(opacity / (1.0 - opacity)).astype(np.float32),
            log_scales=state["log_scales"].cpu().numpy().astype(np.float32),
            quat_xyzw=quats[:, [1, 2, 3, 0]].astype(np.float32),
        ),
    )
    overrides.save(path)


def _switch_distance(
    tree: lod_maths.Tree,
    render: Any,
    merged: dict[str, Any],
    optimised: dict[str, Any],
    lpips: Any,
    ssim: Any,
    torch: Any,
    emitted: dict[str, tuple[Any, np.ndarray | None]],
    size: tuple[int, ...],
) -> dict[str, Any]:
    """Phase 1's measure, per parent tile: the tile alone from where its error projects to
    `SWITCH_TAU`, against its subtree's leaves drawn from there -- merged and optimised."""
    width, height = int(size[0]), int(size[1])
    focal = 0.5 * height / math.tan(math.radians(SWITCH_FOV_DEG) / 2.0)
    k = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
    biggest = sorted(tree.parents, key=lambda t: -(tree.rows[t, 1] - tree.rows[t, 0]))
    rows: list[dict[str, Any]] = []

    def subtree_leaves(tile: int) -> list[int]:
        out, stack = [], [tile]
        while stack:
            t = stack.pop()
            if tree.children[t]:
                stack.extend(tree.children[t])
            else:
                out.append(t)
        return out

    with torch.no_grad():
        for tile in biggest[:SWITCH_TILES]:
            under = subtree_leaves(tile)
            xyz = np.concatenate([np.asarray(emitted[tree.uris[t]][0].xyz) for t in under])
            target = np.median(xyz.astype(np.float64), axis=0)
            distance = float(tree.error[tile]) * focal / SWITCH_TAU
            elevation = math.radians(SWITCH_ELEVATION_DEG)
            for azimuth_deg in SWITCH_AZIMUTHS_DEG:
                azimuth = math.radians(azimuth_deg)
                direction = np.array(
                    [
                        math.cos(elevation) * math.cos(azimuth),
                        math.cos(elevation) * math.sin(azimuth),
                        math.sin(elevation),
                    ]
                )
                eye = lod_maths.orbit_eye(
                    target, direction, tree.low[tile], tree.high[tile], distance
                )
                camera = lod_maths.Camera(lod_maths.look_at(eye, target), k, width, height)
                reference, reference_alpha = render(under, camera, optimised)
                row: dict[str, Any] = {"tile": tree.uris[tile], "azimuthDeg": azimuth_deg}
                for name, state in (("merged", merged), ("optimised", optimised)):
                    colors, alpha = render([tile], camera, state)
                    mse = float(((colors - reference) ** 2).mean())
                    covered = reference_alpha > 0.5
                    holes = float(((alpha < 0.5) & covered).sum()) / max(float(covered.sum()), 1.0)
                    row[name] = {
                        "psnr": round(10.0 * math.log10(1.0 / max(mse, 1e-12)), 3),
                        "ssim": round(float(ssim(colors, reference)), 4),
                        "holes": round(holes, 4),
                        "lpips": None if lpips is None else lpips(colors, reference),
                    }
                rows.append(row)
    means: dict[str, Any] = {}
    for name in ("merged", "optimised"):
        values: dict[str, Any] = {}
        for metric in ("psnr", "ssim", "holes", "lpips"):
            got = [row[name][metric] for row in rows if row[name][metric] is not None]
            values[metric] = round(float(np.mean(got)), 4) if got else None
        means[name] = values
    return {"tau": SWITCH_TAU, "size": [width, height], "mean": means, "views": rows}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Means over (view, tau) pairs, overall and per tau."""

    def mean(members: list[dict[str, Any]], key: str) -> float | None:
        got = [row[key] for row in members if row.get(key) is not None]
        return round(float(np.mean(got)), 5) if got else None

    per_tau = {}
    for tau in EVAL_TAUS:
        members = [row for row in rows if row["tau"] == tau]
        if members:
            per_tau[f"{tau:g}"] = {
                "pairs": len(members),
                **{key: mean(members, key) for key in ("loss", "psnr", "ssim", "lpips")},
            }
    return {
        "pairs": len(rows),
        **{key: mean(rows, key) for key in ("loss", "psnr", "ssim", "lpips")},
        "perTau": per_tau,
    }


def _ssim_function(torch: Any) -> Any:
    """gsplat's trainer's SSIM (`fused_ssim(..., padding="valid")`) when it is installed,
    else the same windowed SSIM in plain torch (11 x 11 gaussian, sigma 1.5, valid)."""
    try:
        from fused_ssim import fused_ssim

        def fused(a: Any, b: Any) -> Any:
            return fused_ssim(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None], padding="valid")

        return fused
    except ImportError:
        pass
    import torch.nn.functional as functional

    coords = torch.arange(11, dtype=torch.float32) - 5.0
    window = torch.exp(-(coords**2) / (2 * 1.5**2))
    window = window / window.sum()
    kernel = (window[:, None] * window[None, :])[None, None].repeat(3, 1, 1, 1)

    def plain(a: Any, b: Any) -> Any:
        x = a.permute(2, 0, 1)[None]
        y = b.permute(2, 0, 1)[None]
        w = kernel.to(x.device, x.dtype)

        def blur(z: Any) -> Any:
            return functional.conv2d(z, w, groups=3)

        mx, my = blur(x), blur(y)
        sxx = blur(x * x) - mx * mx
        syy = blur(y * y) - my * my
        sxy = blur(x * y) - mx * my
        c1, c2 = 0.01**2, 0.03**2
        top = (2 * mx * my + c1) * (2 * sxy + c2)
        return (top / ((mx * mx + my * my + c1) * (sxx + syy + c2))).mean()

    return plain


def _lpips_function(torch: Any, device: Any) -> Any:
    """LPIPS (AlexNet, torchmetrics) -- what gsplat's trainer and Phase 1 report -- or None
    when it cannot be built (no torchmetrics, or no weights and no network)."""
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    except Exception as problem:  # any failure here only costs the LPIPS column
        sys.stdout.write(f"lod: no LPIPS ({type(problem).__name__}: {problem})\n")
        return None

    def measure(a: Any, b: Any) -> float:
        with torch.no_grad():
            x = a.clamp(0, 1).permute(2, 0, 1)[None]
            y = b.clamp(0, 1).permute(2, 0, 1)[None]
            return round(float(metric(x, y)), 5)

    return measure


def _finish(out: Path, summary: dict[str, Any], status: str, reason: str) -> int:
    summary["status"] = status
    summary["reason"] = reason
    (out / SUMMARY_FILE).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    sys.stdout.write(f"lod: {status}: {reason}\n")
    return 0


def _out_of_time(budget: float) -> None:
    sys.stderr.write(f"lod: out of time past {budget:g} s; stopping\n")
    sys.stderr.flush()
    os._exit(3)


if __name__ == "__main__":
    raise SystemExit(main())
