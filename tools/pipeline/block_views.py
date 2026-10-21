"""The camera test's renders: the prior with and without each block, 1 - SSIM per frame.

**Not executed on a GPU in this repository yet.** It needs CUDA (gsplat's rasteriser has
no CPU path); `tests/block_views_stand_in.py` takes the same arguments and writes the same
file, so the stage around it is tested end to end. Run by `blocks.py` with the trainer's
own interpreter (`$GSPLAT_PYTHON`, CPython 3.10).

The rule (`block_maths.contribution`, `block_maths.ssim`): a camera trains a block it does
not stand in when removing the block's gaussians from the coarse model's render changes
the view by 1 - SSIM > epsilon -- the idea of testing a camera by removing a block from
the coarse model's render is CityGaussian's (arXiv 2404.01133). SSIM is Wang et al. 2004
over an 11-tap gaussian window renormalised to the in-frame pixels, so a block seen only
at the frame border counts as the sliver it is.

Per frame: the full render first; a block none of whose gaussians are visible from the
frame (every projected radius 0), or that the camera stands in, is not rendered and
scores 0. Frames are rendered at most `--max-side` pixels on their longer side.

Output, `--out`: `{"names": the dataset's image names in order, "loss": (frames, blocks)
1 - SSIM rounded to 6 decimals, "seconds": wall time}`. A watchdog ends the process with
code 3 after `--budget-s`, and the stage falls back to the CPU visibility test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import block_maths


def _out_of_time(budget: float) -> None:
    sys.stderr.write(f"block_views: out of time after {budget:g} s\n")
    sys.stderr.flush()
    os._exit(3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--data_dir", type=Path, required=True, help="the trainer's dataset")
    parser.add_argument(
        "--prior",
        type=Path,
        required=True,
        help="npz of xyz, quats (w first), log_scales, logit_opacity, f_dc",
    )
    parser.add_argument(
        "--partition", type=Path, required=True, help="block_maths.Partition.to_dict JSON"
    )
    parser.add_argument("--out", type=Path, required=True, help="where to write the JSON")
    parser.add_argument(
        "--trainer", type=Path, required=True, help="gsplat's examples/simple_trainer.py"
    )
    parser.add_argument("--max-side", dest="max_side", type=int, default=800)
    parser.add_argument("--budget-s", dest="budget_s", type=float, default=1800.0)
    # Not for the pipeline: gsplat has no CPU rasteriser.
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    watchdog = threading.Timer(args.budget_s, _out_of_time, args=(args.budget_s,))
    watchdog.daemon = True
    watchdog.start()
    started = time.perf_counter()

    sys.path.insert(0, str(args.trainer.resolve().parent))
    import torch
    from datasets.colmap import Parser
    from gsplat import rasterization

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.stderr.write("block_views: no CUDA device; gsplat's rasteriser has no CPU path\n")
        return 2
    device = torch.device(args.device)

    part = block_maths.Partition.from_dict(json.loads(args.partition.read_text("utf-8")))
    prior = np.load(args.prior)
    xyz = np.asarray(prior["xyz"], dtype=np.float32)
    owner = part.block_of(part.project(xyz))
    members = [owner == b for b in range(part.count)]

    dataset = Parser(data_dir=str(args.data_dir), factor=1, normalize=False, test_every=8)
    names = [str(name) for name in dataset.image_names]
    camtoworlds = np.asarray(dataset.camtoworlds, dtype=np.float64)
    inside = block_maths.cameras_inside(part, camtoworlds[:, :3, 3])

    def tensor(values: Any) -> Any:
        """`values` as a contiguous float32 torch tensor on `device` (what gsplat takes)."""
        return torch.as_tensor(np.ascontiguousarray(values, dtype=np.float32), device=device)

    means = tensor(xyz)
    quats = tensor(prior["quats"])
    scales = torch.exp(tensor(prior["log_scales"]))
    opacities = torch.sigmoid(tensor(prior["logit_opacity"]))
    colours = tensor(prior["f_dc"])[:, None, :]
    masks = {id(member): torch.from_numpy(member).to(device) for member in members}

    frames: list[tuple[Any, Any, int, int]] = []
    for index in range(len(names)):
        camera_id = dataset.camera_ids[index]
        width, height = dataset.imsize_dict[camera_id]
        scale = min(1.0, args.max_side / max(width, height))
        k = np.array(dataset.Ks_dict[camera_id], dtype=np.float64).copy()
        k[:2] *= scale
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        viewmat = np.linalg.inv(camtoworlds[index])
        frames.append((tensor(viewmat)[None], tensor(k)[None], size[0], size[1]))

    full_cache: dict[int, tuple[np.ndarray, Any]] = {}

    def draw(camera: int, drop: Any) -> tuple[np.ndarray, Any]:
        viewmat, k, width, height = frames[camera]
        alpha = opacities
        if drop is not None:
            mask = masks.get(id(drop))
            if mask is None:
                mask = torch.from_numpy(np.asarray(drop, dtype=bool)).to(device)
            alpha = torch.where(mask, torch.zeros_like(opacities), opacities)
        with torch.no_grad():
            image, _alpha, meta = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=alpha,
                colors=colours,
                viewmats=viewmat,
                Ks=k,
                width=width,
                height=height,
                sh_degree=0,
                packed=False,
            )
        rgb = image[0].clamp(0.0, 1.0).cpu().numpy().astype(np.float64)
        return rgb, meta

    def render(camera: int, drop: Any) -> np.ndarray:
        if drop is None:
            if camera not in full_cache:
                full_cache[camera] = draw(camera, None)
            return full_cache[camera][0]
        return draw(camera, drop)[0]

    loss = np.zeros((len(names), part.count), dtype=np.float64)
    for c in range(len(names)):
        _image, meta = full_cache.setdefault(c, draw(c, None))
        radii = meta["radii"][0]
        radii = radii.reshape(radii.shape[0], -1) if radii.ndim > 1 else radii[:, None]
        visible = (radii > 0).any(dim=1).cpu().numpy()
        skip = np.array(
            [bool(inside[c, b]) or not bool(visible[members[b]].any()) for b in range(part.count)]
        )

        def this_frame(_camera: int, drop: Any, c: int = c) -> np.ndarray:
            return render(c, drop)

        row = block_maths.contribution(this_frame, 1, members, skip=skip[None, :])[0]
        loss[c] = row
        full_cache.pop(c, None)
        if (c + 1) % 25 == 0 or c + 1 == len(names):
            sys.stdout.write(f"block_views: {c + 1} of {len(names)} frames\n")
            sys.stdout.flush()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "names": names,
        "loss": np.round(loss, 6).tolist(),
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.out.write_text(json.dumps(document), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
