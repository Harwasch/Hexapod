"""Train-pool-only model inference in a separate GPU process; exits before training.

Only the caller's staged COLMAP dataset is accepted. No model or data URL is a runtime
parameter: both model revisions are baked into the image. Dense failure is recoverable;
missing requested priors is not silently treated as a successful regularized run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

import far_field
import prior_cache
import prior_maths
import quality_stack
import sfm


def append_points(path: Path, xyz: Any, rgb: Any) -> None:
    points = sfm.read_points3d(path)
    count = len(xyz)
    updated = sfm.Points3D(
        ids=np.concatenate(
            [
                points.ids,
                np.arange(
                    int(points.ids.max()) + 1, int(points.ids.max()) + 1 + count, dtype=np.uint64
                ),
            ]
        ),
        xyz=np.concatenate([points.xyz, xyz]),
        rgb=np.concatenate([points.rgb, rgb]),
        error=np.concatenate([points.error, np.zeros(count)]),
        track=points.track,
        track_offsets=np.concatenate(
            [points.track_offsets, np.full(count, len(points.track), dtype=np.int64)]
        ),
    )
    temporary = path.with_suffix(".dense.tmp")
    sfm.write_points3d(temporary, updated)
    temporary.replace(path)


def infer_dense_chunk(
    items: list[Any], model: Any, preprocess: Any, sparse: sfm.Points3D, radius: float
) -> tuple[Any, Any, dict[str, Any]]:
    """Infer and align one chunk; its temporary tensors die before the next chunk."""
    torch = importlib.import_module("torch")
    inputs = [
        {
            "img": d["image"].numpy().astype(np.uint8),
            "intrinsics": d["K"].numpy(),
            "camera_poses": d["camtoworld"].numpy(),
            "is_metric_scale": False,
        }
        for d in items
    ]
    views = preprocess(inputs)
    with torch.inference_mode():
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=True,
            confidence_percentile=50,
        )
    points, colors, anchors, matches = [], [], [], []
    for view, output, item in zip(views, outputs, items, strict=True):
        xyz = output["pts3d"][0].detach().cpu().numpy()
        valid = output["mask"][0].detach().cpu().numpy().squeeze(-1).astype(bool)
        valid &= np.isfinite(xyz).all(-1)
        # Recover RGB from the same normalized/cropped model input.
        image = view["img"][0].detach().cpu().numpy().transpose(1, 2, 0)
        rgb = np.clip(
            (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255,
            0,
            255,
        ).astype(np.uint8)
        points.append(xyz[valid][::8])
        colors.append(rgb[valid][::8])
        intrinsic = view["intrinsics"].detach().cpu().numpy().reshape(3, 3)
        world_to_camera = np.linalg.inv(item["camtoworld"].numpy())
        camera = sparse.xyz @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        pixels = camera @ intrinsic.T
        pixels = np.rint(pixels[:, :2] / np.maximum(pixels[:, 2:], 1e-9)).astype(int)
        h, w = valid.shape
        inside = (
            (camera[:, 2] > 0)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < w)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < h)
        )
        ids = np.flatnonzero(inside)[:: max(1, int(inside.sum()) // 256)]
        ids = ids[valid[pixels[ids, 1], pixels[ids, 0]]]
        anchors.extend(xyz[pixels[ids, 1], pixels[ids, 0]])
        matches.extend(sparse.xyz[ids])
    mapped, fit = prior_maths.aligned_chunk(
        np.concatenate(points), np.asarray(anchors), np.asarray(matches), radius
    )
    return mapped, np.concatenate(colors), fit


def dense(dataset: Any, parser: Any, path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    model_type = importlib.import_module("mapanything.models").MapAnything
    preprocess = importlib.import_module("mapanything.utils.image").preprocess_inputs
    model = model_type.from_pretrained(os.environ["HEXAPOD_MAP_MODEL"]).to("cuda").eval()
    sparse = sfm.read_points3d(path)
    _, radius = far_field.frame(sparse.xyz)
    chunk = settings["dense_chunk"]
    all_xyz, all_rgb, fits = [], [], []
    skipped = []
    retries = []
    torch = importlib.import_module("torch")
    pending = deque(
        (start, min(start + chunk, len(dataset))) for start in range(0, len(dataset), chunk - 1)
    )
    # The dense head is minibatched, but the multi-view aggregator still grows with
    # view count. Retry smaller overlapping groups on OOM; never discard other fits.
    while pending:
        start, stop = pending.popleft()
        items = [dataset[i] for i in range(start, stop)]
        if len(items) < 2:
            continue
        try:
            mapped, colors, fit = infer_dense_chunk(items, model, preprocess, sparse, radius)
        except Exception as error:
            oom = isinstance(error, getattr(torch, "OutOfMemoryError", ()))
            if oom and stop - start > 4:
                middle = (start + stop) // 2
                pending.extendleft(reversed([(start, middle + 1), (middle, stop)]))
                retries.append({"start": start, "views": stop - start, "reason": "CUDA OOM"})
                # The failed helper's traceback owns GPU tensors until this except
                # ends. Clear it before releasing the allocator's unused cache.
                error.__traceback__ = None
                torch.cuda.empty_cache()
                continue
            skipped.append(
                {"start": start, "views": len(items), "reason": f"{type(error).__name__}: {error}"}
            )
            continue
        all_xyz.append(mapped)
        all_rgb.append(colors)
        fits.append({**fit, "start": start, "views": len(items)})
    if not all_xyz:
        return {
            "denseInitPoints": len(sparse),
            "denseInitAddedPoints": 0,
            "chunks": [],
            "skippedChunks": skipped,
            "memoryRetries": retries,
        }
    xyz, rgb = np.concatenate(all_xyz), np.concatenate(all_rgb)
    # Joint voxel selection means existing sparse points win duplicate voxels.
    limit = max(0, settings["dense_max"] - len(sparse))
    joint = np.concatenate([sparse.xyz, xyz])
    selected = prior_maths.voxel_select(joint, radius / 512, len(joint))
    selected = selected[selected >= len(sparse)] - len(sparse)
    selected = selected[np.linspace(0, len(selected) - 1, min(limit, len(selected)), dtype=int)]
    if len(selected):
        append_points(path, xyz[selected], rgb[selected])
    return {
        "denseInitPoints": len(sparse) + len(selected),
        "denseInitAddedPoints": len(selected),
        "chunks": fits,
        "skippedChunks": skipped,
        "memoryRetries": retries,
    }


def run(settings: dict[str, Any]) -> dict[str, Any]:
    dataset_path = Path(settings["dataset"])
    points_path = dataset_path / "sparse/0/points3D.bin"
    sparse = sfm.read_points3d(points_path)
    center, radius = far_field.frame(sparse.xyz)
    report: dict[str, Any] = {
        "denseInitSeconds": 0.0,
        "denseInitPoints": len(sparse),
        "denseInitAddedPoints": 0,
        "densePeakMemoryGb": None,
        "densePeakReservedGb": None,
        "warnings": [],
        "center": center.tolist(),
        "radius": radius,
    }
    cache = Path(settings["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    if settings["far_field"]:
        np.savez(
            Path(settings["ring"]),
            **far_field.sphere(sparse.xyz, sparse.rgb, settings["far_count"]),
        )
    if settings.get("synthetic"):
        report["synthetic"] = True
        # Explicitly synthetic rehearsal: model inference requires the GPU image.
        report["warnings"].append("Synthetic prepass: no model predictions or quality measurements")
        return report
    sys.path.insert(0, str(Path(settings["trainer"]).parent))
    from datasets.colmap import Dataset, Parser

    parser = Parser(
        data_dir=str(dataset_path), factor=settings["data_factor"], normalize=False, test_every=8
    )
    dataset = Dataset(parser, split="train")
    if settings["init"] == "sfm+dense":
        start = time.monotonic()
        torch = importlib.import_module("torch")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            report.update(dense(dataset, parser, points_path, settings))
        except Exception as exc:
            report["warnings"].append(f"dense init fell back to SfM: {type(exc).__name__}: {exc}")
        finally:
            torch.cuda.synchronize()
            report["densePeakMemoryGb"] = torch.cuda.max_memory_allocated() / 1e9
            report["densePeakReservedGb"] = torch.cuda.max_memory_reserved() / 1e9
        report["warnings"].extend(
            f"dense chunk at {chunk['start']} skipped: {chunk['reason']}"
            for chunk in report.get("skippedChunks", [])
        )
        report["warnings"].extend(
            f"dense chunk at {chunk['start']} ({chunk['views']} views) "
            "retried smaller after CUDA OOM"
            for chunk in report.get("memoryRetries", [])
        )
        report["denseInitSeconds"] = time.monotonic() - start
        if report["denseInitAddedPoints"] == 0:
            report["warnings"].append("dense init added zero points; using SfM only")
            if settings.get("require_dense"):
                # Let the caller save the report and fail before priors or training.
                return report
    if settings["priors"]:
        torch = importlib.import_module("torch")
        model = None
        manifest = {}
        for index in range(len(dataset)):
            data = dataset[index]
            name = parser.image_names[int(dataset.indices[index])]
            geometry = prior_maths.cache_geometry(data["image"], data["K"], data.get("mask"))
            key = prior_maths.cache_key(
                dataset_path / "images" / name, quality_stack.MOGE_REVISION, geometry
            )
            target = cache / f"{key}.json"
            if not prior_cache.complete(target):
                if model is None:
                    model_type = importlib.import_module("moge.model.v2").MoGeModel
                    model = (
                        model_type.from_pretrained(os.environ["HEXAPOD_MOGE_MODEL"])
                        .to("cuda")
                        .eval()
                    )
                with torch.inference_mode():
                    prediction = model.infer(
                        data["image"].permute(2, 0, 1).float().cuda() / 255,
                        resolution_level=6,
                        fov_x=math.degrees(
                            2 * math.atan(data["image"].shape[1] / (2 * float(data["K"][0, 0])))
                        ),
                    )
                if "mask" in data:
                    prediction["mask"] &= data["mask"].to(prediction["mask"].device).bool()
                prior_cache.write(
                    cache,
                    key,
                    {k: prediction[k].detach().cpu().numpy() for k in ("depth", "normal", "mask")},
                )
            manifest[name] = target.name
        (cache / "manifest.json").write_text(
            json.dumps(
                {
                    "format": prior_maths.CACHE_FORMAT,
                    "revision": quality_stack.MOGE_REVISION,
                    "frames": manifest,
                }
            )
        )
        report["priorFrames"] = len(manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    report = run(settings)
    Path(settings["report"]).write_text(json.dumps(report, indent=2))
    for warning in report["warnings"]:
        sys.stdout.write(f"WARNING: {warning}\n")


if __name__ == "__main__":
    main()
