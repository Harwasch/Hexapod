"""Held-out images -> depth-backed surface patches -> geometry/error table -> hair fit.

Numpy-only CPU path. LPIPS is an injected patch scorer, never silently approximated.
No ground truth is inferred from alpha, tiers, or training-image residuals.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

import holdout_maths

from .geometry import CONES, SHARPNESS, Array, kernel_sum, spread, unit
from .progress import Progress
from .renderers import Backend, Camera, Render, Scene

PatchScorer = Callable[[Array, Array], float]
COLUMNS = (
    "view",
    "patch_i",
    "patch_j",
    "patch_k",
    "pixels",
    "l1",
    "ssim",
    "lpips",
    "error",
    "nearest_deg",
    "within10",
    "within20",
    "within30",
    "best_mm_px",
    "parallax_deg",
    "observing_views",
    "blur",
    "exposure",
    "angles_deg",
)


def world_points(camera: Camera, depth: Array) -> Array:
    yy, xx = np.indices(depth.shape)
    rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(camera.k).T
    local = rays * depth[..., None]
    return np.asarray(local @ camera.c2w[:3, :3].T + camera.centre)


def observing(
    point: Array, cameras: Sequence[Camera], renders: Sequence[Render], tolerance_m: float
) -> Array:
    visible = np.zeros(len(cameras), dtype=bool)
    for i, (camera, render) in enumerate(zip(cameras, renders, strict=True)):
        uv, z = camera.project(point[None])
        x, y = np.rint(uv[0]).astype(int)
        if z[0] > 0 and 0 <= x < camera.width and 0 <= y < camera.height:
            visible[i] = render.alpha[y, x] > 0.5 and abs(
                float(render.depth[y, x]) - float(z[0])
            ) <= max(tolerance_m, float(z[0]) * 0.01)
    return visible


def camera_features(cameras: Sequence[Camera]) -> dict[str, Array]:
    return {
        "centres": np.array([c.centre for c in cameras]),
        "focal": np.array([min(c.k[0, 0], c.k[1, 1]) for c in cameras]),
        "blur": np.array([c.blur for c in cameras]),
        "exposure": np.array([c.exposure for c in cameras]),
    }


def patch_features(
    point: Array,
    normal: Array,
    test: Camera,
    training: Sequence[Camera],
    visible: Array,
    *,
    arrays: dict[str, Array] | None = None,
) -> dict[str, Any]:
    arrays = camera_features(training) if arrays is None else arrays
    delta = arrays["centres"][visible] - point
    directions = unit(delta)
    test_direction = unit(test.centre - point)
    angles = np.degrees(np.arccos(np.clip(directions @ test_direction, -1, 1)))
    distances = np.linalg.norm(delta, axis=1)
    focal = arrays["focal"][visible]
    incidence = np.abs(directions @ normal)
    # pixel=1; distance in metres; focal in pixels; grazing angles have poor resolution.
    mm_px = distances * 1000 / np.maximum(focal * incidence, 1e-9)

    def logged(name: str) -> float | None:
        values = arrays[name][visible]
        return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None

    return {
        "nearest_deg": float(angles.min()) if len(angles) else 180.0,
        **{f"within{angle}": int((angles <= angle).sum()) for angle in (10, 20, 30)},
        "best_mm_px": float(mm_px.min()) if len(mm_px) else None,
        "parallax_deg": spread(directions),
        "observing_views": int(visible.sum()),
        "blur": logged("blur"),
        "exposure": logged("exposure"),
        "angles_deg": angles.tolist(),
    }


def projections(points: Array, cameras: Sequence[Camera]) -> tuple[Array, Array, Array, Array]:
    """Project all P patches into all C cameras, without a P*C Python loop."""
    poses = np.array([c.c2w for c in cameras])
    local = np.einsum("pcj,cjk->pck", points[:, None] - poses[None, :, :3, 3], poses[:, :3, :3])
    z = local[..., 2]
    pixels = np.einsum("pcj,ckj->pck", local, np.array([c.k for c in cameras]))
    uv = pixels[..., :2] / np.where(np.abs(z) > 1e-12, z, 1e-12)[..., None]
    # Avoid invalid integer casts on invalid or extreme projections.
    xy = np.rint(np.clip(np.nan_to_num(uv), -1e8, 1e8)).astype(int)
    x, y = xy[..., 0], xy[..., 1]
    inside = (
        (z > 0)
        & np.isfinite(uv).all(-1)
        & (x >= 0)
        & (y >= 0)
        & (x < np.array([c.width for c in cameras]))
        & (y < np.array([c.height for c in cameras]))
    )
    return x, y, z, inside


def frustum(points: Array, cameras: Sequence[Camera]) -> Array:
    return projections(points, cameras)[3]


def visible_points(
    points: Array, cameras: Sequence[Camera], cache: dict[int, Render], tolerance_m: float
) -> Array:
    x, y, z, inside = projections(points, cameras)
    visible = np.zeros_like(inside)
    # Each depth map is sampled for ALL patches together. Never loop over patches
    # and then cameras, nor repeatedly project an individual point.
    for index, render in cache.items():
        selected = np.flatnonzero(inside[:, index])
        xx, yy = x[selected, index], y[selected, index]
        visible[selected, index] = (render.alpha[yy, xx] > 0.5) & (
            np.abs(render.depth[yy, xx] - z[selected, index])
            <= np.maximum(tolerance_m, 0.01 * z[selected, index])
        )
    return visible


def patch_table(
    scene: Scene,
    training: Sequence[Camera],
    tests: Iterable[tuple[Camera, Array, Array | None]],
    backend: Backend,
    *,
    patch_m: float = 0.1,
    max_patches_per_view: int = 256,
    lpips: PatchScorer | None = None,
    visibility_backend: Backend | None = None,
    visibility_side: int = 160,
    progress: Progress | None = None,
    rows: list[dict[str, Any]] | None = None,
    visibility_batch: int = 4,
    on_view: Callable[[list[dict[str, Any]], str], None] | None = None,
) -> list[dict[str, Any]]:
    """Depth-backed patches, frustum-prefiltered training visibility cached once/run.

    Each same-resolution group renders in GPU batches <=4. CPU backends may render
    sequentially. Counts retain EVERY training camera; no camera subsampling. Only
    cameras outside all sampled test-patch frusta skip depth, an exact exclusion.
    A caller-owned row sink permits derived partial output on a deadline failure.
    """
    if patch_m <= 0 or not training or not tests or visibility_batch < 1:
        raise ValueError("positive patch size/batch and nonempty train/test cameras required")
    visibility_cameras = [resize(c, visibility_side) for c in training]
    feature_arrays = camera_features(training)
    depth_backend = visibility_backend or backend
    cache: dict[int, Render] = {}
    rows = [] if rows is None else rows
    rng = np.random.default_rng(2411)

    def phase(name: str, detail: str = "") -> Any:
        return progress.phase(name, detail) if progress else nullcontext()

    for camera, truth, mask in tests:
        with phase("test_render", camera.name):
            render = backend.render(scene, camera)
        if truth.shape != render.rgb.shape:
            raise ValueError("test image dimensions differ from camera")
        with phase("features", f"surface patches {camera.name}"):
            points = world_points(camera, render.depth)
            dy, dx = np.gradient(points, axis=(0, 1))
            cross = np.cross(dx, dy)
            normals = unit(cross)
            good = (render.alpha > 0.5) & (render.depth > 0) & np.isfinite(points).all(-1)
            if mask is not None:
                good &= mask
            good &= np.linalg.norm(cross, axis=-1) > 1e-12
            good &= (np.linalg.norm(dx, axis=-1) < patch_m) & (
                np.linalg.norm(dy, axis=-1) < patch_m
            )
            y, x = np.nonzero(good)
            if not len(x):
                continue
            keys, inverse = np.unique(
                np.floor(points[y, x] / patch_m).astype(int), axis=0, return_inverse=True
            )
            chosen = rng.choice(len(keys), min(len(keys), max_patches_per_view), replace=False)
            patches = []
            for index in np.sort(chosen):
                members = np.flatnonzero(inverse == index)
                if len(members) < 4:
                    continue
                py, px = y[members], x[members]
                point = np.median(points[py, px], axis=0)
                ns = normals[py, px]
                normal = unit((ns * np.sign(ns @ ns[0])[:, None]).mean(0))
                patches.append((index, py, px, point, normal))
            if not patches:
                continue
            patch_points = np.array([p[3] for p in patches])
            needed = np.flatnonzero(frustum(patch_points, visibility_cameras).any(0))
            pending = [int(i) for i in needed if i not in cache]
            if progress:
                progress.log(
                    f"visibility frustum: {len(patches)} patches, "
                    f"{len(needed)}/{len(training)} candidate cameras, "
                    f"{len(pending)} uncached, {len(cache)} cached"
                )
        with phase("visibility", camera.name):
            began = time.monotonic()
            completed = 0
            warm_count = 0
            warm_end = began
            # Heterogeneous camera sizes cannot share a gsplat raster batch.
            groups: dict[tuple[int, int], list[int]] = {}
            for i in pending:
                c = visibility_cameras[i]
                groups.setdefault((c.width, c.height), []).append(i)
            for group in groups.values():
                for offset in range(0, len(group), visibility_batch):
                    indices = group[offset : offset + visibility_batch]
                    batch = [visibility_cameras[i] for i in indices]
                    render_many = getattr(depth_backend, "render_many", None)
                    results = (
                        render_many(scene, batch)
                        if render_many
                        else [depth_backend.render(scene, c) for c in batch]
                    )
                    for i, depth in zip(indices, results, strict=True):
                        cache[i] = Render(
                            np.empty((0, 0, 3)), depth.alpha.copy(), depth.depth.copy()
                        )
                    completed += len(indices)
                    if completed == len(indices):
                        warm_count, warm_end = completed, time.monotonic()
                        if progress:
                            progress.log(
                                f"visibility warm batch {completed}/{len(pending)} "
                                f"cameras in {warm_end - began:.2f}s"
                            )
                    elif progress and (
                        completed <= 8
                        or completed % 16 < visibility_batch
                        or completed == len(pending)
                    ):
                        progress.log(f"visibility count {completed}/{len(pending)}")
                        progress.estimate(
                            completed - warm_count, len(pending) - warm_count, warm_end, tail=45
                        )
            visible = visible_points(patch_points, visibility_cameras, cache, patch_m * 0.5)
        with phase("features", f"geometry/L1/SSIM {camera.name}"):
            l1 = np.abs(render.rgb - truth).mean(-1)
            ssim = holdout_maths.ssim_map(render.rgb, truth)
            current = []
            regions = []
            for j, (index, py, px, point, normal) in enumerate(patches):
                row = patch_features(
                    point, normal, camera, training, visible[j], arrays=feature_arrays
                )
                yy = min(max(int(np.median(py)) - 16, 0), max(camera.height - 32, 0))
                xx = min(max(int(np.median(px)) - 16, 0), max(camera.width - 32, 0))
                regions.append(np.s_[yy : yy + 32, xx : xx + 32])
                loss, similarity = float(l1[py, px].mean()), float(ssim[py, px].mean())
                row.update(
                    view=camera.name,
                    patch_i=int(keys[index, 0]),
                    patch_j=int(keys[index, 1]),
                    patch_k=int(keys[index, 2]),
                    pixels=len(py),
                    l1=loss,
                    ssim=similarity,
                    lpips=None,
                    error=0.8 * loss + 0.2 * (1 - similarity),
                )
                current.append(row)
            rows.extend(current)  # Keep L1/SSIM rows even if LPIPS later hits its deadline.
        with phase("lpips", f"{camera.name}: {len(current)} patches"):
            if lpips is not None:
                for row, region in zip(current, regions, strict=True):
                    row["lpips"] = lpips(render.rgb[region], truth[region])
        if on_view is not None:
            on_view(rows, camera.name)
        if progress:
            progress.log(
                f"test complete {camera.name}; total rows={len(rows)}; depth cache={len(cache)}"
            )
    return rows


def resize(camera: Camera, side: int) -> Camera:
    factor = min(1, side / max(camera.width, camera.height))
    width, height = max(2, round(camera.width * factor)), max(2, round(camera.height * factor))
    k = camera.k.copy()
    k[0] *= width / camera.width
    k[1] *= height / camera.height
    return Camera(camera.name, camera.c2w, k, width, height, camera.blur, camera.exposure)


def r_squared(actual: Array, predicted: Array) -> float | None:
    variance = float(((actual - actual.mean()) ** 2).sum())
    return None if variance < 1e-15 else float(1 - ((actual - predicted) ** 2).sum() / variance)


def regression(x: Array, y: Array) -> tuple[Array, Array]:
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return coef, x @ coef


def fit_credit(
    rows: Sequence[dict[str, Any]], target: str = "error", *, scene_intercepts: bool = False
) -> dict[str, Any]:
    """Bounded grid fit, equal weight per spatial patch, leave-test-view-out evaluation.

    Geometry-only base form; optional resolution/parallax terms need >= .02 held-out
    R² improvement. Grid selection uses the same CV folds: exploratory, not nested CV.
    The fit returns boundary/identifiability diagnostics instead of claiming calibration.
    """
    rows = [r for r in rows if r.get(target) is not None]
    if len(rows) < 8:
        raise ValueError("at least eight measured patch/view rows required for a fit")
    y = np.array([r[target] for r in rows])
    scene_names = sorted({str(r.get("scene", "scene")) for r in rows})
    scenes = np.array([str(r.get("scene", "scene")) for r in rows])
    groups = np.array([f"{r.get('scene', 'scene')}/{r['view']}" for r in rows])
    patches = [
        (scene_names.index(str(r.get("scene", "scene"))), r["patch_i"], r["patch_j"], r["patch_k"])
        for r in rows
    ]
    _, inverse, counts = np.unique(patches, axis=0, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    root_w = np.sqrt(weights)
    intercepts = np.ones((len(rows), 1))
    if scene_intercepts and len(scene_names) > 1:
        intercepts = np.column_stack([intercepts, *[scenes == s for s in scene_names[1:]]])
    unique_groups = np.unique(groups)

    def fit(x: Array) -> tuple[Array, Array, float | None]:
        coef = regression(x * root_w[:, None], y * root_w)[0]
        prediction = x @ coef
        cv = np.full(len(y), np.nan)
        for group in unique_groups:
            train = groups != group
            if train.sum() >= x.shape[1] + 2:
                c = regression(x[train] * root_w[train, None], y[train] * root_w[train])[0]
                cv[~train] = x[~train] @ c
        score = r_squared(y, cv) if np.isfinite(cv).all() else None
        return coef, prediction, score

    angle_x = np.column_stack([intercepts, [r["nearest_deg"] for r in rows]])
    angle_coef, angle_pred, angle_cv = fit(angle_x)
    best: tuple[float, dict[str, Any], Array] | None = None
    for cone in CONES:
        for sharpness in SHARPNESS:
            totals = np.array([kernel_sum(r, cone, sharpness) for r in rows])
            for fill in (1, 2, 4, 8, 16):
                deficit = 1 - np.minimum(totals / fill, 1)
                x = np.column_stack([intercepts, deficit])
                coef, pred, cv = fit(x)
                if coef[-1] < 0:  # More support must not predict larger error.
                    continue
                score = cv if cv is not None else r_squared(y, pred)
                if score is not None and (best is None or score > best[0]):
                    best = (
                        score,
                        {
                            "cone_deg": cone,
                            "sharpness": sharpness,
                            "views_to_fill": fill,
                            "coefficients": coef.tolist(),
                            "r2": r_squared(y, pred),
                            "leave_view_out_r2": cv,
                        },
                        x,
                    )
    if best is None:
        return {"status": "no_monotone_fit", "rows": len(rows)}
    base_score, result, _ = best
    result.update(resolution_target_mm_px=None, parallax_gate_deg=None)
    totals = np.array([kernel_sum(r, result["cone_deg"], result["sharpness"]) for r in rows])
    multiplier = np.ones(len(rows))
    # Optional factors multiply effective views BEFORE saturation, matching the score
    # exported for the phone. Never accept a gate using in-sample R² alone.
    if result["leave_view_out_r2"] is not None:
        for name, thresholds, values in (
            (
                "resolution_target_mm_px",
                (1, 2, 4, 8),
                np.array([r["best_mm_px"] or 1e9 for r in rows]),
            ),
            ("parallax_gate_deg", (10, 15, 20, 30), np.array([r["parallax_deg"] for r in rows])),
        ):
            choices = []
            for threshold in thresholds:
                factor = (
                    np.minimum((threshold / np.maximum(values, 1e-9)) ** 2, 1)
                    if name.startswith("resolution")
                    else np.minimum(values / threshold, 1)
                )
                weighted = multiplier * factor
                deficit = 1 - np.minimum(totals * weighted / result["views_to_fill"], 1)
                x = np.column_stack([intercepts, deficit])
                coef, pred, cv = fit(x)
                if cv is not None and coef[-1] >= 0 and cv > base_score + 0.02:
                    choices.append((cv, threshold, coef, pred, weighted))
            if choices:
                base_score, threshold, coef, pred, multiplier = max(choices, key=lambda c: c[0])
                result.update(
                    {
                        name: threshold,
                        "coefficients": coef.tolist(),
                        "r2": r_squared(y, pred),
                        "leave_view_out_r2": base_score,
                    }
                )
    result.update(
        status="exploratory",
        rows=len(rows),
        test_views=len(unique_groups),
        spatial_patches=len(counts),
        target=target,
        scene_names=scene_names,
        scene_intercepts=bool(scene_intercepts and len(scene_names) > 1),
        angle_baseline={
            "intercept": float(angle_coef[0]),
            "slope_per_deg": float(angle_coef[-1]),
            "r2": r_squared(y, angle_pred),
            "leave_view_out_r2": angle_cv,
        },
        grid_boundary=bool(
            result["cone_deg"] in (10, 60)
            or result["views_to_fill"] in (1, 16)
            or result["sharpness"] in (0.5, 4)
        ),
        caution="Exploratory grid CV; adjacent patches/views correlated. Validate on new scans.",
    )
    return result


def write_outputs(rows: list[dict[str, Any]], output: Path, *, fit: bool = True) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    table = output / "patches.csv.tmp"
    with table.open("w", newline="") as stream:
        optional = tuple(
            k for k in ("scene", "credit_sums", "cone_counts") if any(k in r for r in rows)
        )
        writer = csv.DictWriter(stream, fieldnames=(*COLUMNS, *optional), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = {**row, "angles_deg": json.dumps(row["angles_deg"])}
            for key in ("credit_sums", "cone_counts"):
                if key in row:
                    encoded[key] = json.dumps(row[key])
            writer.writerow(encoded)
    table.replace(output / "patches.csv")
    if not fit:
        return {}
    fits = {
        name: fit_credit(rows, name)
        for name in ("error", "l1", "lpips")
        if sum(r.get(name) is not None for r in rows) >= 8
    }
    fitted = output / "fit.json.tmp"
    fitted.write_text(json.dumps(fits, indent=2, allow_nan=False) + "\n")
    fitted.replace(output / "fit.json")
    return fits
