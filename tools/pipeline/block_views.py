"""Which cameras each block needs: render the prior with and without the block. GPU.

**Not executed on a GPU in this repository yet**: gsplat's rasteriser has no CPU path.
The decision it feeds is `block_maths.assign`, and the score is `block_maths.contribution`
with `block_maths.ssim` -- the same code the pipeline's tests drive with a stub renderer.
Run by the `train` stage with the trainer's interpreter, as `holdout_error.py` is.

CityGaussian's rule (V1 `data_partition.py`, V2 `projection_based_partition_assignment`):
for each camera not already standing in a block, render the coarse model, render it again
with the block's gaussians' opacity set to zero, and assign the camera to the block when
`1 - SSIM` of the two exceeds epsilon. Here: every registered frame through gsplat's own
parser (`examples/datasets/colmap.py`, the split and undistortion the trainer uses), its
intrinsics scaled so the long side is at most `--max-side` (the renders are compared with
each other, never with a photo, so any size works; the prior's own 800 px is enough to see
a block appear or not), the prior drawn with its DC colour on black. A block none of whose
gaussians lands in the frame (`radii > 0` in the full render) scores 0 without a render.

Output (`--out`): `{"names": [...], "loss": [[1 - SSIM per block] per frame], "seconds"}`,
frames in the parser's (name-sorted) order.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import block_maths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--max-side", dest="max_side", type=int, default=800)
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(args.trainer.resolve().parent))
    import torch
    from datasets.colmap import Parser
    from gsplat import rasterization

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.stderr.write("block_views: no CUDA device; gsplat's rasteriser has no CPU path\n")
        return 2
    started = time.perf_counter()
    device = torch.device(args.device)
    part = block_maths.Partition.from_dict(json.loads(args.partition.read_text("utf-8")))
    with np.load(args.prior) as data:
        prior = {name: np.asarray(data[name], dtype=np.float32) for name in data.files}
    xyz = prior["xyz"]
    member_of = part.block_of(part.project(xyz))
    members = [member_of == b for b in range(part.count)]

    def tensor(values: Any) -> Any:
        return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).to(device)

    means = tensor(xyz)
    quats = tensor(prior["quats"])
    scales = torch.exp(tensor(prior["log_scales"]))
    opacities = torch.sigmoid(tensor(prior["logit_opacity"]))
    colors = tensor(prior["f_dc"])[:, None, :]
    masks = [torch.from_numpy(m).to(device) for m in members]

    colmap = Parser(data_dir=str(args.data_dir), factor=1, normalize=False, test_every=8)
    names = list(colmap.image_names)
    centres = np.asarray(colmap.camtoworlds)[:, :3, 3]
    inside = block_maths.cameras_inside(part, centres)
    views: list[tuple[Any, Any, int, int]] = []
    for index in range(len(names)):
        camera_id = colmap.camera_ids[index]
        k = np.array(colmap.Ks_dict[camera_id], dtype=np.float64)
        width, height = colmap.imsize_dict[camera_id]
        scale = min(1.0, args.max_side / max(width, height))
        k[:2, :] *= scale
        w, h = max(1, round(width * scale)), max(1, round(height * scale))
        viewmat = torch.linalg.inv(tensor(colmap.camtoworlds[index])[None])
        views.append((viewmat, tensor(k)[None], w, h))

    full: dict[int, Any] = {}
    seen: dict[int, Any] = {}

    def render(camera: int, drop: Any) -> Any:
        if drop is None and camera in full:
            return full[camera]
        viewmat, k, w, h = views[camera]
        alpha = opacities
        if drop is not None:
            # `contribution` hands back the very arrays in `members`.
            mask = next((masks[b] for b, m in enumerate(members) if m is drop), None)
            if mask is None:
                mask = torch.from_numpy(np.asarray(drop, dtype=bool)).to(device)
            alpha = torch.where(mask, torch.zeros_like(opacities), opacities)
        with torch.no_grad():
            rgb, _, meta = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=alpha,
                colors=colors,
                viewmats=viewmat,
                Ks=k,
                width=w,
                height=h,
                sh_degree=0,
                packed=False,
            )
        image = rgb[0].clamp(0.0, 1.0).cpu().numpy()
        if drop is None:
            full[camera] = image
            seen[camera] = (meta["radii"][0] > 0).reshape(means.shape[0], -1).any(-1)
        return image

    # The full render first, to learn which blocks the camera sees at all.
    skip = inside.copy()
    loss = np.zeros((len(names), part.count), dtype=np.float64)
    for camera in range(len(names)):
        render(camera, None)
        visible = seen.pop(camera)
        for block in range(part.count):
            if not bool(visible[masks[block]].any()):
                skip[camera, block] = True

        def one(_camera: int, drop: Any, camera: int = camera) -> Any:
            return render(camera, drop)

        row = block_maths.contribution(one, 1, members, skip=skip[camera : camera + 1])
        full.pop(camera, None)
        loss[camera] = row[0]
        if (camera + 1) % 25 == 0 or camera + 1 == len(names):
            sys.stdout.write(f"block_views: {camera + 1} of {len(names)} frames\n")
            sys.stdout.flush()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "names": names,
                "loss": np.round(loss, 6).tolist(),
                "seconds": round(time.perf_counter() - started, 1),
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
