"""Held-out images -> depth-backed surface patches -> geometry/error table -> hair fit.

Numpy-only CPU path. LPIPS is an injected patch scorer, never silently approximated.
No ground truth is inferred from alpha, tiers, or training-image residuals.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import holdout_maths

from .geometry import Array, credit, spread, unit
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


def patch_features(
    point: Array, normal: Array, test: Camera, training: Sequence[Camera], visible: Array
) -> dict[str, Any]:
    selected = [camera for camera, seen in zip(training, visible, strict=True) if seen]
    directions = unit(np.asarray([c.centre for c in selected]).reshape(-1, 3) - point)
    test_direction = unit(test.centre - point)
    angles = np.degrees(np.arccos(np.clip(directions @ test_direction, -1, 1)))
    distances = np.array([np.linalg.norm(c.centre - point) for c in selected])
    focal = np.array([min(c.k[0, 0], c.k[1, 1]) for c in selected])
    incidence = np.abs(directions @ normal)
    # pixel=1; distance in metres; focal in pixels; grazing angles have poor resolution.
    mm_px = distances * 1000 / np.maximum(focal * incidence, 1e-9)

    def logged(name: str) -> float | None:
        values = np.array([getattr(c, name) for c in selected])
        return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None

    return {
        "nearest_deg": float(angles.min()) if len(angles) else 180.0,
        **{f"within{angle}": int((angles <= angle).sum()) for angle in (10, 20, 30)},
        "best_mm_px": float(mm_px.min()) if len(mm_px) else None,
        "parallax_deg": spread(directions),
        "observing_views": len(selected),
        "blur": logged("blur"),
        "exposure": logged("exposure"),
        "angles_deg": angles.tolist(),
    }


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
) -> list[dict[str, Any]]:
    """Sample whole 3-D patches; one row per (test view, patch), deterministic seed.

    Training-camera occlusion uses expected depth at bounded resolution; test mask and
    alpha > .5 exclude unobserved pixels. A patch uses a 32x32 image neighborhood for
    LPIPS (may include adjacent surfaces); L1/SSIM use its actual member pixels.
    """
    if patch_m <= 0 or not training or not tests:
        raise ValueError("positive patch size and nonempty train/test cameras required")
    visibility_cameras = [resize(c, visibility_side) for c in training]
    depth_backend = visibility_backend or backend
    training_renders = [depth_backend.render(scene, c) for c in visibility_cameras]
    rows = []
    rng = np.random.default_rng(2411)
    for camera, truth, mask in tests:
        render = backend.render(scene, camera)
        if truth.shape != render.rgb.shape:
            raise ValueError("test image dimensions differ from camera")
        points = world_points(camera, render.depth)
        # Neighboring depths give local surface normals; silhouettes are excluded below.
        dy, dx = np.gradient(points, axis=(0, 1))
        normals = unit(np.cross(dx, dy))
        good = (render.alpha > 0.5) & (render.depth > 0) & np.isfinite(points).all(-1)
        if mask is not None:
            good &= mask
        good &= np.linalg.norm(np.cross(dx, dy), axis=-1) > 1e-12
        good &= (np.linalg.norm(dx, axis=-1) < patch_m) & (np.linalg.norm(dy, axis=-1) < patch_m)
        y, x = np.nonzero(good)
        if not len(x):
            continue
        keys, inverse = np.unique(
            np.floor(points[y, x] / patch_m).astype(int), axis=0, return_inverse=True
        )
        chosen = rng.choice(len(keys), min(len(keys), max_patches_per_view), replace=False)
        l1 = np.abs(render.rgb - truth).mean(-1)
        ssim = holdout_maths.ssim_map(render.rgb, truth)
        for index in np.sort(chosen):
            members = np.flatnonzero(inverse == index)
            if len(members) < 4:
                continue
            py, px = y[members], x[members]
            point = np.median(points[py, px], axis=0)
            # Align signs before averaging, avoiding cancellation on slightly curved patches.
            ns = normals[py, px]
            normal = unit((ns * np.sign(ns @ ns[0])[:, None]).mean(0))
            visible = observing(point, visibility_cameras, training_renders, patch_m * 0.5)
            row = patch_features(point, normal, camera, training, visible)
            yy, xx = int(np.median(py)), int(np.median(px))
            yy = min(max(yy - 16, 0), max(camera.height - 32, 0))
            xx = min(max(xx - 16, 0), max(camera.width - 32, 0))
            region = np.s_[yy : yy + 32, xx : xx + 32]
            loss, similarity = float(l1[py, px].mean()), float(ssim[py, px].mean())
            row.update(
                view=camera.name,
                patch_i=int(keys[index, 0]),
                patch_j=int(keys[index, 1]),
                patch_k=int(keys[index, 2]),
                pixels=len(members),
                l1=loss,
                ssim=similarity,
                lpips=None if lpips is None else lpips(render.rgb[region], truth[region]),
                error=0.8 * loss + 0.2 * (1 - similarity),
            )
            rows.append(row)
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


def fit_credit(rows: Sequence[dict[str, Any]], target: str = "error") -> dict[str, Any]:
    """Bounded grid fit, equal weight per spatial patch, leave-test-view-out evaluation.

    Geometry-only base form; optional resolution/parallax terms need >= .02 held-out
    R² improvement. Grid selection uses the same CV folds: exploratory, not nested CV.
    The fit returns boundary/identifiability diagnostics instead of claiming calibration.
    """
    rows = [r for r in rows if r.get(target) is not None and r["observing_views"] > 0]
    if len(rows) < 8:
        raise ValueError("at least eight observed patch/view rows required for a fit")
    y = np.array([r[target] for r in rows])
    groups = np.array([r["view"] for r in rows])
    patches = [(r["patch_i"], r["patch_j"], r["patch_k"]) for r in rows]
    _, inverse, counts = np.unique(patches, axis=0, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    root_w = np.sqrt(weights)
    cosines = [np.cos(np.radians(r["angles_deg"])) for r in rows]
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

    angle_x = np.column_stack([np.ones(len(y)), [r["nearest_deg"] for r in rows]])
    angle_coef, angle_pred, angle_cv = fit(angle_x)
    best: tuple[float, dict[str, Any], Array] | None = None
    for cone in (10, 20, 25, 30, 45, 60):
        for sharpness in (0.5, 1, 2, 4):
            totals = np.array([credit(c, cone, sharpness).sum() for c in cosines])
            for fill in (1, 2, 4, 8, 16):
                deficit = 1 - np.minimum(totals / fill, 1)
                x = np.column_stack([np.ones(len(y)), deficit])
                coef, pred, cv = fit(x)
                if coef[1] < 0:  # More support must not predict larger error.
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
    totals = np.array([credit(c, result["cone_deg"], result["sharpness"]).sum() for c in cosines])
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
                x = np.column_stack([np.ones(len(y)), deficit])
                coef, pred, cv = fit(x)
                if cv is not None and coef[1] >= 0 and cv > base_score + 0.02:
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
        angle_baseline={
            "intercept": float(angle_coef[0]),
            "slope_per_deg": float(angle_coef[1]),
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


def write_outputs(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "patches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "angles_deg": json.dumps(row["angles_deg"])})
    fits = {
        name: fit_credit(rows, name)
        for name in ("error", "l1", "lpips")
        if sum(r.get(name) is not None for r in rows) >= 8
    }
    (output / "fit.json").write_text(json.dumps(fits, indent=2, allow_nan=False) + "\n")
    return fits
