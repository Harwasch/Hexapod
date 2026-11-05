"""Evaluation only: frozen PLYs, classic renderer, test-fitted poses and photometry.

Torch imports stay inside functions so the local bench can rehearse without CUDA.
No optimizer ever receives a splat tensor. A single worker owns the fixed schedule
for every requested export; a timeout fails the batch rather than shortening later rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.bench.evaluate import lookup_frames
from experiments.bench.rescore_maths import (
    MODES,
    footprint,
    needle_scores,
    pose_motion,
    row_base,
)
from holdout_error import read_ply, sh_rest, shipped_sh


def write_json(path: Path, value: object) -> None:
    # scene.py needs Python 3.11, and this script runs on the trainer's 3.10.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def delta_pose(rotation: Any, translation: Any, extent: float) -> Any:
    import torch

    x, y, z = rotation.unbind()
    zero = x * 0
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
    upper = torch.cat([skew, translation[:, None] * extent], dim=1)
    algebra = torch.cat([upper, upper.new_zeros((1, 4))], dim=0)
    return torch.linalg.matrix_exp(algebra)


def photo_objective(prediction: Any, target: Any, valid: Any) -> Any:
    """L1 after detached per-channel gain/offset fit; pose cannot chase mere exposure.

    Fits target = gain*render+offset on valid pixels, constrained to gain [0.5,2]
    and offset [-0.25,0.25]. These bounds are fixed for every run and resolution.
    The separately reported CC metric uses upstream's unbounded inverse-affine fit.
    """
    import torch

    p, t = prediction[valid], target[valid]
    with torch.no_grad():
        pm, tm = p.mean(0), t.mean(0)
        centred = p - pm
        gain = ((centred * (t - tm)).mean(0) / centred.square().mean(0).clamp_min(1e-8)).clamp(
            0.5, 2
        )
        offset = (tm - gain * pm).clamp(-0.25, 0.25)
    return (p * gain + offset - t).abs().mean()


def optimise(
    render: Any,
    original: Any,
    target: Any,
    mask: Any,
    intrinsic: Any,
    extent: float,
    steps: int,
    initial: Any,
    deadline: float,
) -> tuple[Any, dict[str, Any]]:
    """Choose the best actual objective, with raw pose always a candidate."""
    import torch

    if not mask.any():
        raise ValueError("alignment needs valid pixels")
    device = original.device
    rotation = torch.nn.Parameter(torch.zeros(3, device=device))
    translation = torch.nn.Parameter(torch.zeros(3, device=device))
    optimizer = torch.optim.Adam(
        [{"params": [rotation], "lr": 0.001}, {"params": [translation], "lr": 0.0002}]
    )
    # initial is the best half-res twist, in the same original-camera tangent frame.
    with torch.no_grad():
        before = float(
            photo_objective(render(original, intrinsic, target.shape[:2])[0], target, mask)
        )
        if not np.isfinite(before):
            raise ValueError("nonfinite raw alignment objective")
        best_loss, best = before, torch.zeros(6, device=device)
        if initial is not None:
            rotation.copy_(initial[:3])
            translation.copy_(initial[3:])
    for step in range(steps + 1):
        if time.time() >= deadline:
            raise TimeoutError(
                "fixed rescore alignment schedule exceeded its budget; no unequal-step comparison"
            )
        pose = delta_pose(rotation, translation, extent) @ original
        prediction, _ = render(pose, intrinsic, target.shape[:2])
        loss = photo_objective(prediction, target, mask)
        value = float(loss.detach())
        if not np.isfinite(value):
            raise ValueError("nonfinite alignment objective")
        if value < best_loss:
            best_loss = value
            best = torch.cat([rotation.detach(), translation.detach()]).clone()
        if step == steps:
            break
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            rotation.mul_(
                (rotation.new_tensor(np.deg2rad(3)) / rotation.norm().clamp_min(1e-12)).clamp_max(1)
            )
            translation.mul_(
                (translation.new_tensor(0.02) / translation.norm().clamp_min(1e-12)).clamp_max(1)
            )
        del prediction, loss, pose
    aligned = (delta_pose(best[:3], best[3:], extent) @ original).detach()
    movement = pose_motion(original.cpu().numpy(), aligned.cpu().numpy(), extent)
    return best, {
        "viewmat": aligned.cpu().numpy().tolist(),
        "lossBefore": before,
        "lossAfter": best_loss,
        "objective": "valid-pixel L1 with fitted bounded per-channel gain/offset",
        "steps": steps,
        **movement,
    }


def png(path: Path, tensor: Any, *, preview: bool = False) -> None:
    pixels = (tensor.detach().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    image = Image.fromarray(pixels)
    if preview:
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
    image.save(path)


def make_renderer(splats: dict[str, Any], degree: int) -> Any:
    from gsplat import rasterization

    def render(view: Any, intrinsic: Any, size: Any) -> tuple[Any, Any]:
        rgb, alpha, _ = rasterization(
            **splats,
            viewmats=view[None],
            Ks=intrinsic[None],
            height=int(size[0]),
            width=int(size[1]),
            sh_degree=degree,
            packed=False,
            rasterize_mode="classic",
            with_ut=False,
            with_eval3d=False,
            near_plane=0.01,
            far_plane=1e10,
        )
        return rgb[0].clamp(0, 1), alpha[0, ..., 0]

    return render


def score_all(settings: dict[str, Any], output: Path) -> None:
    sys.path.insert(0, str(Path(settings["trainer"]).parent))
    import torch
    import torch.nn.functional as functional
    from datasets.colmap import Dataset, Parser
    from gsplat.color_correct import color_correct_affine
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    torch.manual_seed(42)
    torch.set_num_threads(4)
    parser = Parser(data_dir=settings["data"], factor=1, normalize=False, test_every=1)
    dataset = Dataset(parser, split="val")
    lookup = lookup_frames(parser.image_names, dataset.indices, settings["names"])
    metrics = {
        "psnr": PeakSignalNoiseRatio(data_range=1.0).cuda(),
        "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).cuda(),
        "lpips": LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda(),
    }
    for metric in metrics.values():
        metric.eval().requires_grad_(False)

    def scores(prediction: Any, target: Any) -> dict[str, Any]:
        values = {}
        for name, metric in metrics.items():
            metric.reset()
            value = float(metric(prediction.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]))
            if not np.isfinite(value):
                raise ValueError(f"nonfinite {name}; refusing an invalid score")
            values[name] = value
        return values

    rows = []
    extent = settings["cameras"]["extent"]
    for index, source in enumerate(settings["sources"]):
        columns = read_ply(Path(source["ply"]))
        row = row_base(source, settings["fullSize"])
        row.update(needle=needle_scores(columns, extent), resolutions={}, topdown={"perCamera": []})
        degree = int(source["result"].get("trainParams", {}).get("ship_sh_degree", 0))
        dc = np.stack([columns[f"f_dc_{i}"] for i in range(3)], axis=1)[:, None]
        sh, used = shipped_sh(dc, sh_rest(columns), degree)
        if used != degree:
            raise ValueError("kept PLY lacks its recorded shipped SH degree")
        row["shDegree"] = used

        def tensor(keys: list[str], columns: Any = columns) -> Any:
            return torch.from_numpy(
                np.stack([columns[k] for k in keys], axis=1).astype(np.float32)
            ).cuda()

        splats = {
            "means": tensor(["x", "y", "z"]),
            "quats": tensor([f"rot_{i}" for i in range(4)]),
            "scales": tensor([f"scale_{i}" for i in range(3)]).exp(),
            "opacities": tensor(["opacity"])[:, 0].sigmoid(),
            "colors": torch.from_numpy(sh).cuda(),
        }
        if any(value.requires_grad for value in splats.values()):
            raise ValueError("rescore must never optimize splats")
        far = torch.as_tensor(
            np.asarray(columns.get("far_field", np.zeros(len(columns["x"])))) >= 0.5, device="cuda"
        )
        del columns, dc, sh, tensor

        render = make_renderer(splats, used)

        frame_rows: dict[str, list[Any]] = {"full": [], "half": []}
        for frame_index, name in enumerate(settings["names"]):
            data = dataset[lookup[name]]
            real = data["image"].cuda().float() / 255
            if list(real.shape[:2]) != settings["fullSize"]:
                raise ValueError(
                    "undistorted dataset dimensions differ from frozen full-size contract"
                )
            intrinsic = data["K"].cuda()
            original = torch.linalg.inv(data["camtoworld"].cuda())
            mask = data.get("mask", torch.ones(real.shape[:2], dtype=torch.bool)).cuda().bool()
            half_real = functional.avg_pool2d(real.permute(2, 0, 1)[None], 2)[0].permute(1, 2, 0)
            half_mask = functional.avg_pool2d(mask.float()[None, None], 2)[0, 0] == 1
            half_k = intrinsic.clone()
            half_k[:2] /= 2
            half_twist, half_pose = optimise(
                render,
                original,
                half_real,
                half_mask,
                half_k,
                extent,
                settings["steps"][0],
                None,
                settings["deadline"],
            )
            _, full_pose = optimise(
                render,
                original,
                real,
                mask,
                intrinsic,
                extent,
                settings["steps"][1],
                half_twist,
                settings["deadline"],
            )
            with torch.no_grad():
                for resolution, target, valid, camera_k, pose in (
                    ("full", real, mask, intrinsic, full_pose),
                    ("half", half_real, half_mask, half_k, half_pose),
                ):
                    raw = render(original, camera_k, target.shape[:2])[0]
                    aligned = render(
                        torch.tensor(pose["viewmat"], device="cuda", dtype=torch.float32),
                        camera_k,
                        target.shape[:2],
                    )[0]
                    corrected = torch.zeros_like(raw)
                    aligned_corrected = torch.zeros_like(raw)
                    corrected[valid] = color_correct_affine(raw[valid], target[valid])
                    aligned_corrected[valid] = color_correct_affine(aligned[valid], target[valid])
                    predictions = {
                        "raw": raw * valid[..., None],
                        "aligned": aligned * valid[..., None],
                        "cc": corrected,
                        "aligned+cc": aligned_corrected,
                    }
                    reference = target * valid[..., None]
                    frame_rows[resolution].append(
                        {
                            "name": name,
                            "pose": pose,
                            "metrics": {
                                mode: scores(pred, reference) for mode, pred in predictions.items()
                            },
                        }
                    )
                    if frame_index == 0:
                        png(output / f"row-{index}-{resolution}-gt.png", reference, preview=True)
                        for mode, prediction in predictions.items():
                            png(
                                output / f"row-{index}-{resolution}-{mode}.png",
                                prediction,
                                preview=True,
                            )
            sys.stdout.write(
                f"rescore row {index + 1}: frame {frame_index + 1}/{len(settings['names'])}\n"
            )
            sys.stdout.flush()
        for resolution, frames in frame_rows.items():
            size = (
                settings["fullSize"]
                if resolution == "full"
                else [v // 2 for v in settings["fullSize"]]
            )
            row["resolutions"][resolution] = {
                "scoredAt": {"height": size[0], "width": size[1]},
                "perFrame": frames,
                "metrics": {
                    mode: {
                        metric: float(np.mean([r["metrics"][mode][metric] for r in frames]))
                        for metric in metrics
                    }
                    for mode in MODES
                },
                "poseMoveMedian": {
                    key: float(np.median([r["pose"][key] for r in frames]))
                    for key in ("translationPct", "rotationDeg")
                },
            }
        # Frozen sphere remains in photo scores (it ships in the PLY). It must not
        # fill top-down holes or contribute to the needle denominators.
        with torch.no_grad():
            for camera in settings["cameras"]["cameras"]:
                view = torch.tensor(camera["viewmat"], dtype=torch.float32, device="cuda")
                intrinsic = torch.tensor(camera["K"], dtype=torch.float32, device="cuda")
                inclusive, _ = render(view, intrinsic, (camera["height"], camera["width"]))
                png(output / f"row-{index}-{camera['name']}-inclusive.png", inclusive)
            splats = {key: value[~far] for key, value in splats.items()}
            render = make_renderer(splats, used)
            for camera in settings["cameras"]["cameras"]:
                view = torch.tensor(camera["viewmat"], dtype=torch.float32, device="cuda")
                intrinsic = torch.tensor(camera["K"], dtype=torch.float32, device="cuda")
                rgb, alpha = render(view, intrinsic, (camera["height"], camera["width"]))
                mask_array = footprint(camera, settings["cameras"]["footprint"])
                mask_tensor = torch.as_tensor(mask_array, device="cuda")
                holes = float((alpha[mask_tensor] < 0.5).float().mean())
                row["topdown"]["perCamera"].append(
                    {
                        "name": camera["name"],
                        "holeFraction": holes,
                        "footprintPixels": int(mask_array.sum()),
                    }
                )
                png(output / f"row-{index}-{camera['name']}.png", rgb)
                png(output / f"row-{index}-{camera['name']}-alpha.png", alpha)
        row["topdown"]["holeMean"] = float(
            np.mean([c["holeFraction"] for c in row["topdown"]["perCamera"]])
        )
        rows.append(row)
        write_json(output / "scores.partial.json", {"rows": rows})
        del splats, far, render
        torch.cuda.empty_cache()
    write_json(
        output / "scores.json",
        {
            "protocol": "rescore-v1",
            "rows": rows,
            "pixels": (
                "full image, invalid undistortion pixels black in both; fits use only valid pixels"
            ),
            "lpipsNet": "alex",
            "photoPngs": (
                "first frozen test frame; previews at max 640px, scores use recorded scoredAt"
            ),
        },
    )


def cpu_self_check() -> None:
    """Run by the existing image CPU check, after cached compilation; no CUDA needed."""
    import torch
    from gsplat.color_correct import color_correct_affine

    from experiments.bench.rescore_maths import affine_correct, se3

    rotation = torch.zeros(3, requires_grad=True)
    translation = torch.zeros(3, requires_grad=True)
    pose = delta_pose(rotation, translation, 2)
    pose[:3, 3].sum().backward()
    if not torch.isfinite(rotation.grad).all() or not torch.isfinite(translation.grad).all():
        raise ValueError("SE3 zero-initialization gradient is not finite")
    vector = np.array([0.02, -0.01, 0.03, 0.01, 0.02, -0.01])
    np.testing.assert_allclose(
        delta_pose(torch.tensor(vector[:3]), torch.tensor(vector[3:]), 1).numpy(),
        se3(vector),
        atol=1e-12,
    )
    target = torch.linspace(0.1, 0.8, 90).reshape(30, 3)
    prediction = 0.8 * target + 0.05
    expected, _, _ = affine_correct(prediction.numpy(), target.numpy())
    np.testing.assert_allclose(
        color_correct_affine(prediction, target).numpy(), expected, atol=1e-6
    )

    # Exercise actual best-pose selection without CUDA. A bad coarse initialization
    # must never displace an already perfect raw pose on the full-res objective.
    grid = torch.arange(64.0).reshape(8, 8, 1) / 64
    reference = (0.4 + 0.2 * torch.cos(grid * 6)).expand(-1, -1, 3)

    def toy_render(view: Any, intrinsic: Any, size: Any) -> tuple[Any, Any]:
        prediction = reference + view[0, 3] * torch.sin(grid * 13)
        return prediction, torch.ones((8, 8))

    raw = torch.eye(4)
    _, fitted = optimise(
        toy_render,
        raw,
        reference,
        torch.ones((8, 8), dtype=torch.bool),
        torch.eye(3),
        1,
        3,
        torch.tensor([0.0, 0, 0, 0.02, 0, 0]),
        time.time() + 30,
    )
    if fitted["lossAfter"] > fitted["lossBefore"] or fitted["translationPct"] != 0:
        raise ValueError("alignment lost the best raw-pose candidate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    score_all(json.loads(args.settings.read_text()), args.out)


if __name__ == "__main__":
    main()
