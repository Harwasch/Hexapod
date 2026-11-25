"""Compact exact grid statistics, scene-aware fitting and honest held-scene transfer."""

from __future__ import annotations

import csv
import importlib
import json
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import fit_credit, r_squared
from .geometry import CONES, KERNELS, Array, kernel_sum


def compact_rows(rows: list[dict[str, Any]], scene: str) -> list[dict[str, Any]]:
    for row in rows:
        row["scene"] = scene
        if "credit_sums" in row:
            continue  # Partial-output recovery may compact an already-fitted table.
        angles = np.asarray(row["angles_deg"])
        row["credit_sums"] = [kernel_sum(row, cone, sharp) for cone, sharp in KERNELS]
        row["cone_counts"] = [int((angles <= cone).sum()) for cone in CONES]
        row["angles_deg"] = []
    return rows


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in ("scene", "view"):
                    row[key] = value
                elif key in ("angles_deg", "credit_sums", "cone_counts"):
                    row[key] = json.loads(value)
                elif value == "":
                    row[key] = None
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def predict(rows: Sequence[dict[str, Any]], fit: dict[str, Any]) -> Array:
    if fit.get("status") != "exploratory":
        raise ValueError("fit has no usable monotone predictor")
    totals = np.array([kernel_sum(r, fit["cone_deg"], fit["sharpness"]) for r in rows])
    target = fit.get("resolution_target_mm_px")
    if target is not None:
        mm = np.array([r["best_mm_px"] or 1e9 for r in rows])
        totals *= np.minimum((target / np.maximum(mm, 1e-9)) ** 2, 1)
    gate = fit.get("parallax_gate_deg")
    if gate is not None:
        totals *= np.minimum(np.array([r["parallax_deg"] for r in rows]) / gate, 1)
    coef = fit["coefficients"]
    result = coef[0] + coef[-1] * (1 - np.minimum(totals / fit["views_to_fill"], 1))
    if fit.get("scene_intercepts"):
        for index, scene in enumerate(fit["scene_names"][1:], 1):
            result += coef[index] * np.array([r["scene"] == scene for r in rows])
    return np.asarray(result)


def pooled_fit(rows: list[dict[str, Any]], target: str = "error") -> dict[str, Any]:
    rows = [r for r in rows if r.get(target) is not None]
    scenes = sorted({r["scene"] for r in rows})
    if len(scenes) < 2:
        raise ValueError("pooled fit requires at least two scenes")
    per_scene = {s: fit_credit([r for r in rows if r["scene"] == s], target) for s in scenes}
    combined = fit_credit(rows, target, scene_intercepts=True)
    transfer: dict[str, Any] = {}
    actuals, predictions = [], []
    for held in scenes:
        train = [r for r in rows if r["scene"] != held]
        test = [r for r in rows if r["scene"] == held]
        # Retune EVERY parameter on training scenes only. No held-scene intercept,
        # mean residual, kernel choice, target or parallax calibration is fitted here.
        model = fit_credit(train, target)
        if model.get("status") != "exploratory":
            transfer[held] = {"r2": None, "status": model["status"]}
            continue
        y = np.array([r[target] for r in test])
        estimate = predict(test, model)
        transfer[held] = {"r2": r_squared(y, estimate), "rows": len(test), "fit": model}
        actuals.append(y)
        predictions.append(estimate)
    combined["leave_one_scene_out"] = transfer
    combined["leave_one_scene_out_r2"] = (
        r_squared(np.concatenate(actuals), np.concatenate(predictions))
        if len(actuals) == len(scenes)
        else None
    )
    combined["transfer_intercept"] = "training-scenes-only; no held-scene intercept adaptation"
    return {"per_scene": per_scene, "pooled": combined}


def plot_fit(rows: list[dict[str, Any]], fits: dict[str, Any], output: Path) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    fit = fits.get("error", {})
    selected = list(rows)
    if fit.get("status") != "exploratory" or not selected:
        return
    predicted = predict(selected, fit)
    cone = fit["cone_deg"]
    for row in selected:
        row["views_in_cone"] = (
            row["cone_counts"][CONES.index(cone)]
            if "cone_counts" in row
            else sum(a <= cone for a in row["angles_deg"])
        )
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, key in zip(
        axes, ("nearest_deg", "views_in_cone", "best_mm_px", "parallax_deg"), strict=True
    ):
        x = np.array([r[key] if r[key] is not None else np.nan for r in selected])
        y = np.array([r["error"] for r in selected])
        ax.scatter(x, y, s=2, alpha=0.15, label="held-out error")
        good = np.isfinite(x)
        bins: Array = (
            np.unique(np.quantile(x[good], np.linspace(0, 1, 17))) if good.any() else np.empty(0)
        )
        bx, by = [], []
        for lower, upper in pairwise(bins):
            members = good & (x >= lower) & (x <= upper)
            if members.any():
                bx.append(float(x[members].mean()))
                by.append(float(predicted[members].mean()))
        ax.plot(bx, by, color="tab:red", label="fitted model: bin mean")
        ax.set(xlabel=key, ylabel="0.8 L1 + 0.2 (1-SSIM)")
    axes[0].legend(fontsize=6)
    fig.suptitle(
        f"{selected[0].get('scene', 'scene')}; R²={fit['r2']:.3f}; "
        f"leave-view-out={fit['leave_view_out_r2']}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output / "fit_error.png", dpi=110)
    plt.close(fig)
    for row in selected:
        row.pop("views_in_cone", None)
