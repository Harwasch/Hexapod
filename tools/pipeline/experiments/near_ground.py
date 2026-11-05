"""CPU-only frozen-scene coverage, sharpness and reprojection diagnostics.

Run from tools/pipeline: python -m experiments.near_ground SCENE --output DIR.
Lower third is a ground *proxy*, not a semantic ground or occlusion mask.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

import sfm
from video import sharpness


def distribution(values: Any) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "p10": None, "median": None, "p90": None}
    return {
        "count": len(values),
        "mean": float(values.mean()),
        **dict(
            zip(("p10", "median", "p90"), np.percentile(values, [10, 50, 90]).tolist(), strict=True)
        ),
    }


def observations(path: Path) -> dict[int, Any]:
    """Keep the xy and point IDs that sfm.read_model intentionally skips."""
    out = {}
    dtype = np.dtype([("xy", "<f8", (2,)), ("point", "<i8")])
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            image_id = struct.unpack("<IdddddddI", handle.read(64))[0]
            while True:
                char = handle.read(1)
                if char == b"\0":
                    break
                if not char:
                    raise ValueError("truncated COLMAP image name")
            n = struct.unpack("<Q", handle.read(8))[0]
            out[image_id] = np.frombuffer(handle.read(24 * n), dtype=dtype)
    return out


def project(xyz: Any, image: sfm.Image, camera: sfm.Camera) -> tuple[Any, Any]:
    """COLMAP world-to-camera and original (distorted) frame pixel coordinates."""
    local = np.asarray(xyz) @ image.rotation.T + np.asarray(image.tvec)
    depth = local[:, 2]
    xy = local[:, :2] / np.where(depth != 0, depth, np.nan)[:, None]
    p = dict(zip(camera.param_names, camera.params, strict=True))
    r2 = np.square(xy).sum(axis=1)
    if camera.model in ("SIMPLE_RADIAL", "RADIAL", "OPENCV"):
        radial = 1 + p.get("k", p.get("k1", 0)) * r2 + p.get("k2", 0) * r2**2
        distorted = xy * radial[:, None]
        if camera.model == "OPENCV":
            x, y = xy.T
            distorted[:, 0] += 2 * p["p1"] * x * y + p["p2"] * (r2 + 2 * x * x)
            distorted[:, 1] += p["p1"] * (r2 + 2 * y * y) + 2 * p["p2"] * x * y
        xy = distorted
    elif camera.model not in ("SIMPLE_PINHOLE", "PINHOLE"):
        raise ValueError(f"projection for {camera.model} is not implemented")
    focal = [p["f"], p["f"]] if "f" in p else [p["fx"], p["fy"]]
    return xy * focal + [p["cx"], p["cy"]], depth


def pool_support(points: sfm.Points3D, pool_ids: set[int], sharp: dict[int, float]) -> Any:
    """Distinct train-pool images per point; held-out tracks never count as views."""
    counts, means = np.zeros(len(points), dtype=int), np.full(len(points), np.nan)
    for i in range(len(points)):
        ids = set(map(int, points.track_of(i)[:, 0])) & pool_ids
        counts[i] = len(ids)
        if ids:
            means[i] = np.mean([sharp[j] for j in ids])
    return counts, means


def analyse(scene: Path) -> dict[str, Any]:
    split = json.loads((scene / "split.json").read_text())
    pool, test = set(split["pool"]), set(split["test"])
    if pool & test or len(test) != len(split["test"]):
        raise ValueError("split must contain distinct disjoint pool/test names")
    model = sfm.read_model(scene / "poses")
    images = {image.id: image for image in model.images}
    names = {image.name: image.id for image in model.images}
    if not (pool | test) <= names.keys():
        raise ValueError("split references unregistered frames")
    cameras = {camera.id: camera for camera in model.cameras}
    points = sfm.read_points3d(scene / "poses" / "points3D.bin")
    obs = observations(scene / "poses" / "images.bin")
    sharp = {names[name]: sharpness(scene / "frames" / name) for name in sorted(pool | test)}
    pool_ids = {names[name] for name in pool}
    counts, support_sharp = pool_support(points, pool_ids, sharp)
    # Pool-only residuals avoid treating the held-out frame as training evidence.
    # Report one mean per 3D point, so a long track cannot dominate the distribution.
    point_index = {int(pid): i for i, pid in enumerate(points.ids)}
    residual_sum, residual_count = np.zeros(len(points)), np.zeros(len(points), dtype=int)
    observed: dict[int, tuple[Any, Any]] = {}
    for image_id in sorted(pool_ids | {names[name] for name in test}):
        data = obs[image_id]
        valid = data["point"] >= 0
        indices = np.array([point_index[int(pid)] for pid in data["point"][valid]], dtype=int)
        xy, _ = project(points.xyz[indices], images[image_id], cameras[images[image_id].camera_id])
        residual = np.linalg.norm(xy - data["xy"][valid], axis=1)
        observed[image_id] = (indices, residual)
        if image_id in pool_ids:
            np.add.at(residual_sum, indices, residual)
            np.add.at(residual_count, indices, 1)
    pool_error = np.divide(
        residual_sum, residual_count, out=np.full(len(points), np.nan), where=residual_count > 0
    )

    def summary(selection: Any, depth: Any) -> dict[str, Any]:
        n = int(selection.sum())
        return {
            "points": n,
            "trainViews": distribution(counts[selection]),
            "viewsBelow2": int((counts[selection] < 2).sum()),
            "viewsBelow4": int((counts[selection] < 4).sum()),
            "supportSharpness": distribution(support_sharp[selection]),
            "poolReprojectionPx": distribution(pool_error[selection]),
            "colmapAllViewsErrorPx": distribution(points.error[selection]),
            "depth": distribution(depth[selection]),
        }

    frames = []
    for name in split["test"]:
        image = images[names[name]]
        camera = cameras[image.camera_id]
        xy, depth = project(points.xyz, image, camera)
        inside = (
            (depth > 0)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < camera.width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < camera.height)
        )
        bottom = inside & (xy[:, 1] >= 2 * camera.height // 3)
        indices, residual = observed[image.id]
        tracked = np.zeros(len(points), dtype=bool)
        tracked[indices] = True
        regions = {}
        for label, selection in (("bottom", bottom), ("upper", inside & ~bottom)):
            bins = []
            edges = np.quantile(depth[selection], [0, 1 / 3, 2 / 3, 1]) if selection.any() else []
            for i in range(max(0, len(edges) - 1)):
                within = selection & (depth >= edges[i])
                within &= depth <= edges[i + 1] if i == 2 else depth < edges[i + 1]
                bins.append({"range": [edges[i], edges[i + 1]], **summary(within, depth)})
            regions[label] = {
                "projected": summary(selection, depth),
                "trackedInTest": summary(selection & tracked, depth),
                "testReprojectionPx": distribution(residual[selection[indices]]),
                "depthTertiles": bins,
            }
        frames.append({"name": name, "sharpness": sharp[image.id], "regions": regions})
    meta_path = scene / "frames_meta.json"
    luma = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        luma = {f["name"]: f.get("exposure", {}).get("lumaMean") for f in meta.get("frames", [])}
    return {
        "protocol": "near-ground-v1",
        "scene": str(scene.resolve()),
        "splitDigest": split.get("digest"),
        "definitions": {
            "bottom": "original distorted image y >= floor(2*height/3); ground proxy",
            "views": "distinct split.pool frames in each point track, excludes all split.test",
            "supportSharpness": (
                "per-point mean of supporting pool frames' full-resolution PIL L Laplacian "
                "variance, same as keyframes; texture dependent, no blur cutoff"
            ),
            "reprojection": (
                "pool-only mean Euclidean residual per point, in original pixels; "
                "geometry/calibration/feature error proxy, not a pose-only measurement"
            ),
            "depth": "camera Z in scene units; tertiles computed separately per frame/region",
            "visibility": (
                "projected includes potentially occluded points; trackedInTest is the observed "
                "subset, itself biased toward matchable texture"
            ),
        },
        "poolFrames": len(pool),
        "testFrames": len(test),
        "sparsePoints": len(points),
        "sharpness": {
            key: distribution([sharp[names[n]] for n in group])
            for key, group in (("pool", pool), ("test", test))
        },
        "lumaMean": {
            key: distribution([luma[n] for n in group if luma.get(n) is not None])
            for key, group in (("pool", pool), ("test", test))
        },
        "poolFrameSharpness": {name: sharp[names[name]] for name in sorted(pool)},
        "perFrame": frames,
    }


def verdict(result: dict[str, Any]) -> str:
    rows = result["perFrame"]
    compared = {}
    lines = [
        "# Near-ground diagnostic",
        "",
        f"{result['testFrames']} held-out frames; "
        f"only {result['poolFrames']} train-pool frames count as views.",
        "",
        "Medians are across test frames; each frame's distributions and depth bins are in JSON.",
        "",
        "| Region (tracked in test) | Pool views/point | Points with <4 pool views | "
        "Pool reprojection px | Test reprojection px | Supporting-frame sharpness |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for region in ("bottom", "upper"):
        values = [r["regions"][region] for r in rows]

        def median(field: str, values: Any = values) -> float:
            v = [r["trackedInTest"][field]["median"] for r in values]
            return float(np.median([x for x in v if x is not None]))

        weak = np.median(
            [
                r["trackedInTest"]["viewsBelow4"] / max(1, r["trackedInTest"]["points"])
                for r in values
            ]
        )
        error = np.median(
            [
                r["testReprojectionPx"]["median"]
                for r in values
                if r["testReprojectionPx"]["median"] is not None
            ]
        )
        lines.append(
            f"| {region} | {median('trainViews'):.1f} | {100 * weak:.1f}% | "
            f"{median('poolReprojectionPx'):.3f} | {error:.3f} | "
            f"{median('supportSharpness'):.1f} |"
        )
        compared[region] = (median("trainViews"), median("poolReprojectionPx"))
    bottom, upper = compared["bottom"], compared["upper"]
    lines += [
        "",
        (
            "Verdict: limited measured views are the strongest relative evidence here. "
            "The lower third has fewer training views without higher pool reprojection error. "
            "These measurements do not establish blur or worse near-ground poses as the cause."
            if bottom[0] < upper[0] and bottom[1] <= upper[1]
            else "Verdict: these proxies do not isolate a single cause; inspect per-frame "
            "support and depth bins before attributing the limitation to views, blur or poses."
        ),
    ]
    lines += [
        "",
        "Low support is direct evidence of limited measured views. Laplacian variance ranks "
        "texture/blur together and cannot prove motion blur. Reprojection residuals also include "
        "feature and calibration error; they cannot uniquely identify pose error. This sparse "
        "analysis does not measure untracked sand or prove visibility of projected points.",
        "",
        f"Train-pool/test lumaMean: {result['lumaMean']['pool']['mean']} / "
        f"{result['lumaMean']['test']['mean']} (metadata luminance, not mean rendered RGB).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyse(args.scene)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "near-ground.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "near-ground.md").write_text(verdict(result))


if __name__ == "__main__":
    main()
