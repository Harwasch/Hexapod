"""Private real-photo teacher for Living Light; never publish the work directory.

Read inputs in place. Only `bake` writes outside work: material sidecars contain fitted
splat attributes, never images or COLMAP data. Heavy stages must be invoked via `heavy`.
Use `uv run --group light` for teacher only; other stages need the base environment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from light_bake import MODEL, pack_sidecar, srgb_to_linear
from splat_render import Camera, _rotation_matrices, load_tileset, render

MODEL_REVISION = "08c3930bb641abf786ba44ce92547507ebefbc16"


def placed_camera(image: dict, intrinsics: dict, placement: dict, width: int) -> Camera:
    """Exact COLMAP similarity; rectified intrinsics retain centred principal point."""
    params = intrinsics["params"]
    if intrinsics["model"] != "SIMPLE_RADIAL":
        raise ValueError("This prototype supports SIMPLE_RADIAL only")
    if params["cx"] != intrinsics["width"] / 2 or params["cy"] != intrinsics["height"] / 2:
        raise ValueError("CPU renderer requires a centred principal point")
    rotation = np.asarray(placement["rotation"])
    q = _rotation_matrices(np.asarray([image["qvec"]]))[0]
    centre = placement["scale"] * rotation @ np.asarray(image["centre"])
    centre += np.asarray(placement["translation"])
    ratio = width / intrinsics["width"]
    return Camera(
        q @ rotation.T, centre, params["f"] * ratio, width, round(intrinsics["height"] * ratio)
    )


def rectified_photo(path: Path, intrinsics: dict, camera: Camera) -> np.ndarray:
    """Decode private input in memory and remove COLMAP's radial distortion."""
    photo = np.asarray(Image.open(path).convert("RGB"))
    if photo.shape[:2] != (intrinsics["height"], intrinsics["width"]):
        raise ValueError("Photo dimensions disagree with COLMAP")
    p = intrinsics["params"]
    k = np.array([[p["f"], 0, p["cx"]], [0, p["f"], p["cy"]], [0, 0, 1.0]])
    rectified = cv2.undistort(photo, k, np.array([p["k"], 0.0, 0.0, 0.0]))
    return cv2.resize(rectified, (camera.width, camera.height), interpolation=cv2.INTER_AREA)


def prepare(inputs: Path, work: Path, scan: Path, count: int, width: int) -> None:
    start = time.monotonic()
    poses = json.loads((inputs / "pose/poses/poses.json").read_text())
    placement = json.loads((inputs / "place/placement.json").read_text())
    cameras = {c["id"]: c for c in poses["cameras"]}
    images = sorted(poses["images"], key=lambda i: i["name"])
    chosen = np.unique(np.rint(np.linspace(0, len(images) - 1, count)).astype(int))
    work.mkdir(parents=True, exist_ok=True)
    splats = load_tileset(scan / "tileset.json")
    views = []
    for j, index in enumerate(chosen):
        image = images[index]
        intrinsics = cameras[image["cameraId"]]
        camera = placed_camera(image, intrinsics, placement, width)
        photo = rectified_photo(inputs / "normalize/frames" / image["name"], intrinsics, camera)
        frame = render(splats, camera, max_samples=64, sample_budget=32_000_000)
        np.savez_compressed(work / f"depth-{j:03}.npz", depth=frame.depth, alpha=frame.alpha)
        mask = frame.alpha > 0.95
        error = (frame.rgb[mask] - photo[mask] / 255.0) ** 2
        psnr = float(-10 * np.log10(max(float(error.mean()), 1e-10)))
        views.append(
            {
                "name": image["name"],
                "camera": camera.to_json(),
                "intrinsics": intrinsics,
                "opaquePixels": int(mask.sum()),
                "psnrOpaque": psnr,
            }
        )
        print("prepared", j, "PSNR", round(psnr, 2), flush=True)
    (work / "views.json").write_text(json.dumps(views, indent=2))
    (work / "prepare.json").write_text(
        json.dumps(
            {
                "seconds": time.monotonic() - start,
                "count": len(views),
                "width": width,
                "height": views[0]["camera"]["height"],
                "medianOpaquePsnr": float(np.median([v["psnrOpaque"] for v in views])),
            },
            indent=2,
        )
    )


def teacher(inputs: Path, work: Path, limit: int) -> None:
    import torch
    from diffusers import MarigoldIntrinsicsPipeline

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    pipe = MarigoldIntrinsicsPipeline.from_pretrained(
        MODEL, revision=MODEL_REVISION, torch_dtype=torch.float32
    )
    pipe.to("cpu")
    pipe.enable_attention_slicing()
    (work / "teacher-config.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "revision": MODEL_REVISION,
                "torch": torch.__version__,
                "device": "cpu",
                "dtype": "float32",
                "steps": 5,
                "ensemble": 1,
                "seed": 20261025,
                "threads": 8,
                "processingResolution": "selected view width; original aspect ratio",
            },
            indent=2,
        )
    )
    times_path = work / "teacher-times.json"
    times = json.loads(times_path.read_text()) if times_path.exists() else []
    done = 0
    views = json.loads((work / "views.json").read_text())
    for i, view in enumerate(views):
        output = work / f"albedo-{i:03}.npy"
        if output.exists():
            continue
        start = time.monotonic()
        photo = Image.fromarray(
            rectified_photo(
                inputs / "normalize/frames" / view["name"],
                view["intrinsics"],
                Camera.from_json(view["camera"]),
            )
        )
        result = pipe(
            photo,
            num_inference_steps=5,
            ensemble_size=1,
            processing_resolution=max(photo.size),
            generator=torch.Generator().manual_seed(20261025),
        )
        np.save(output, result.prediction[0].astype(np.float32))
        times.append(
            {
                "view": i,
                "seconds": time.monotonic() - start,
                "width": photo.width,
                "height": photo.height,
            }
        )
        times_path.write_text(json.dumps(times, indent=2))
        print(times[-1], flush=True)
        done += 1
        if done >= limit:
            break


def alignment(inputs: Path, work: Path, scan: Path) -> None:
    """Numerical alignment control; no photo or photo-derived image is exported."""
    views = json.loads((work / "views.json").read_text())
    splats = load_tileset(scan / "tileset.json")
    results = []
    for index in (0, len(views) // 2, len(views) - 1):
        view = views[index]
        camera = Camera.from_json(view["camera"])
        photo = (
            rectified_photo(inputs / "normalize/frames" / view["name"], view["intrinsics"], camera)
            / 255.0
        )
        frame = render(splats, camera, max_samples=48, sample_budget=18_000_000)
        margin = 33
        rgb = frame.rgb[margin:-margin, margin:-margin]
        mask = frame.alpha[margin:-margin, margin:-margin] > 0.95
        errors = []
        for dy in (-32, -8, 0, 8, 32):
            for dx in (-32, -8, 0, 8, 32):
                shifted = photo[
                    margin + dy : camera.height - margin + dy,
                    margin + dx : camera.width - margin + dx,
                ]
                mse = float(((rgb[mask] - shifted[mask]) ** 2).mean())
                errors.append({"dx": dx, "dy": dy, "mse": mse})
        results.append(
            {
                "view": index,
                "zero": next(e for e in errors if e["dx"] == e["dy"] == 0),
                "best": min(errors, key=lambda e: e["mse"]),
                "offsetErrors": errors,
            }
        )
    (work / "alignment.json").write_text(json.dumps(results, indent=2))


def robust_colour(samples: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted component median then residual-MAD trimmed mean; zero-weight is unseen.

    Arrays are views × splats × RGB and views × splats. No spatial smoothing or
    contrast-target calibration. Robustness rejects view-specific teacher mistakes.
    """
    total = weights.sum(0)
    median = np.zeros(samples.shape[1:], np.float32)
    for c in range(3):
        order = np.argsort(samples[:, :, c], axis=0)
        w = np.take_along_axis(weights, order, axis=0)
        index = np.argmax(np.cumsum(w, axis=0) >= total[None] * 0.5, axis=0)
        median[:, c] = np.take_along_axis(samples[:, :, c], order, axis=0)[
            index, np.arange(len(index))
        ]
    residual = np.linalg.norm(samples - median[None], axis=2)

    # Weighted median residual and its MAD, so arbitrary zero-weight data has no effect.
    def weighted_median(values):
        order = np.argsort(values, axis=0)
        w = np.take_along_axis(weights, order, axis=0)
        index = np.argmax(np.cumsum(w, axis=0) >= total[None] * 0.5, axis=0)
        return np.take_along_axis(values, order, axis=0)[index, np.arange(len(index))]

    middle = weighted_median(residual)
    mad = weighted_median(np.abs(residual - middle))
    kept = weights * (residual <= middle + 3 * mad + 1e-6)
    denominator = kept.sum(0)
    colour = (samples * kept[:, :, None]).sum(0) / np.maximum(denominator[:, None], 1e-12)
    return colour, (kept > 0).sum(0)


def observations(splats, camera, normal, rotation, albedo, depth, alpha):
    """Depth-tested bilinear albedo samples weighted by footprint and incidence."""
    uv, z = camera.project(splats.positions)
    x, y = np.floor(uv).astype(int).T
    valid = (z > 0) & (x >= 1) & (x < camera.width - 1) & (y >= 1) & (y < camera.height - 1)
    ids = np.flatnonzero(valid)
    x, y = x[ids], y[ids]
    axis = np.einsum("i,nij->nj", camera.rotation[2], rotation[ids])
    sigma = np.linalg.norm(axis * splats.scales[ids], axis=1)
    tolerance = 2 * sigma + 1.5 * z[ids] / camera.focal
    visible = (np.abs(z[ids] - depth[y, x]) < tolerance) & (alpha[y, x] > 0.8)
    ids = ids[visible]
    direction = camera.centre - splats.positions[ids]
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    incidence = np.abs(np.sum(normal[ids] * direction, axis=1))
    scales = np.sort(splats.scales[ids], axis=1)
    footprint = camera.focal**2 * scales[:, 1] * scales[:, 2] / z[ids] ** 2
    weights = np.maximum(incidence, 0.05) * np.sqrt(np.clip(footprint, 0.1, 16))
    # CPU Camera uses pixel centres at i+.5; image arrays index those centres at i.
    coords = (uv[ids] - 0.5).T[::-1]
    colours = np.stack(
        [map_coordinates(albedo[:, :, c], coords, order=1, mode="nearest") for c in range(3)],
        axis=1,
    )
    return ids, colours, weights, direction


def bake(work: Path, scan: Path, previous: Path, output: Path) -> None:
    from rig_tiles import glb_spz
    from splat_render import splats_from_columns
    from splat_tiles import tile_offsets, unpack_spz
    from synthetic_tree import checksum_positions

    start = time.monotonic()
    views = json.loads((work / "views.json").read_text())
    cameras = [Camera.from_json(v["camera"]) for v in views]
    manifest = json.loads((previous / "manifest.json").read_text())
    predictions = [np.load(work / f"albedo-{i:03}.npy", mmap_mode="r") for i in range(len(views))]
    depths = []
    for i in range(len(views)):
        with np.load(work / f"depth-{i:03}.npz") as archive:
            depths.append({key: archive[key] for key in ("depth", "alpha")})
    document = json.loads((scan / "tileset.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    total, seen = 0, 0
    leaves = set()
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("children"):
            stack.extend(tile["children"])
        elif tile.get("content"):
            leaves.add(tile["content"]["uri"])
    leaf_total, leaf_seen = 0, 0
    counts = []
    offered_counts = []
    for uri, offset in tile_offsets(document).items():
        splats = splats_from_columns(unpack_spz(glb_spz(scan / uri)), offset)
        checksum = checksum_positions(splats.positions)
        old = manifest["tiles"][checksum]
        rows = np.frombuffer((previous / old["uri"]).read_bytes(), np.uint8).reshape(-1, 48)
        if len(rows) != len(splats) or not np.all(rows[:, 40:44] == [76, 76, 48, 49]):
            raise ValueError("Previous sidecar order/count/version mismatch")
        rotation = _rotation_matrices(splats.rotations)
        normal = rotation[np.arange(len(splats)), :, np.argmin(splats.scales, axis=1)]
        observer = np.zeros((len(splats), 3))
        samples = np.zeros((len(views), len(splats), 3), np.float32)
        weights = np.zeros((len(views), len(splats)), np.float32)
        for i, camera in enumerate(cameras):
            ids, colours, w, direction = observations(
                splats,
                camera,
                normal,
                rotation,
                predictions[i],
                depths[i]["depth"],
                depths[i]["alpha"],
            )
            samples[i, ids] = colours
            weights[i, ids] = w
            observer[ids] += direction * w[:, None]
        fitted, count = robust_colour(samples, weights)
        observed = count > 0
        albedo = srgb_to_linear(splats.colours)
        albedo[observed] = fitted[observed]
        orientation = np.where(observed[:, None], observer, [0, 0, 1])
        normal[np.sum(normal * orientation, axis=1) < 0] *= -1
        confidence = count / len(views)
        (output / old["uri"]).write_bytes(
            pack_sidecar(albedo, normal, rows[:, 8:40] / 255.0, rows[:, 7] / 255.0, confidence)
        )
        old["observedFraction"] = float(observed.mean())
        old["medianRetainedViews"] = float(np.median(count))
        total += len(splats)
        seen += int(observed.sum())
        if uri in leaves:
            leaf_total += len(splats)
            leaf_seen += int(observed.sum())
        counts.extend(count.tolist())
        offered_counts.extend((weights > 0).sum(0).tolist())
        print("fitted", uri, "coverage", observed.mean(), flush=True)
    # Exact visibility/sky bytes are reused from round 2, isolating the material change.
    manifest.update(
        {
            "bakeSeconds": time.monotonic() - start,
            "teacherViews": len(views),
            "observedFraction": seen / total,
            "leafObservedFraction": leaf_seen / leaf_total,
            "leafCount": leaf_total,
            "medianRetainedViews": float(np.median(counts)),
            "medianDepthTestedViews": float(np.median(offered_counts)),
            "retainedViewHistogram": np.bincount(counts, minlength=len(views) + 1).tolist(),
            "depthTestedViewHistogram": np.bincount(
                offered_counts, minlength=len(views) + 1
            ).tolist(),
            "method": "real photos; footprint/incidence weighted MAD-trimmed mean",
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "alignment", "teacher", "bake"])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.inputs, args.work, args.scan, args.count, args.width)
    elif args.stage == "alignment":
        alignment(args.inputs, args.work, args.scan)
    elif args.stage == "teacher":
        teacher(args.inputs, args.work, args.limit)
    else:
        bake(args.work, args.scan, args.previous, args.output)


if __name__ == "__main__":
    main()
