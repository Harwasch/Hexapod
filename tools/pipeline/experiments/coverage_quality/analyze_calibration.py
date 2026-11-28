"""Offline diagnostics for the completed held-out campaign; never import downloaded code.

Run by filename with hw-py -I. Only this repository is added to sys.path. Plotting is
lazy; numerical helpers and tests require only NumPy. Bootstrap units are test views.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "4"

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.coverage_quality.calibration import fit_credit, r_squared  # noqa: E402
from experiments.coverage_quality.fitting import predict, read_rows  # noqa: E402
from experiments.coverage_quality.geometry import Array, hemisphere  # noqa: E402

FEATURES = (
    "nearest_deg",
    "within10",
    "within20",
    "within30",
    "best_mm_px",
    "parallax_deg",
    "observing_views",
)
SCENES = ("c935", "beach")
SEED = 2411


def rank(values: Array) -> Array:
    """Average ranks for ties, without a SciPy dependency in the CPU test path."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    return np.asarray((upper - (counts - 1) / 2)[inverse], dtype=float)


def corr(x: Array, y: Array) -> float | None:
    if len(x) < 3:
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator <= 1e-15 else float(np.dot(x, y) / denominator)


def demean(values: Array, groups: Array) -> Array:
    if not len(values):
        return values.copy()
    _, index = np.unique(groups, return_inverse=True)
    sums = np.zeros((int(index.max()) + 1, values.shape[1]))
    np.add.at(sums, index, values)
    return np.asarray(values - (sums / np.bincount(index)[:, None])[index])


def correlations(x: Array, y: Array, views: Array, pixels: Array) -> list[dict[str, Any]]:
    complete = np.isfinite(x).all(1) & np.isfinite(y) & np.isfinite(pixels)
    ranked = np.column_stack([rank(x[complete, i]) for i in range(x.shape[1])])
    nuisance = rank(pixels[complete])[:, None]
    ry = rank(y[complete])[:, None]
    within = demean(np.column_stack([ranked, nuisance, ry]), views[complete])
    result = []
    for i, name in enumerate(FEATURES):
        good = np.isfinite(x[:, i]) & np.isfinite(y)
        pair = np.column_stack([rank(x[good, i]), rank(y[good])])
        view_adjusted = demean(pair, views[good])
        controls = np.delete(within[:, :-1], i, axis=1)
        residual_x = (
            within[:, i] - controls @ np.linalg.lstsq(controls, within[:, i], rcond=None)[0]
        )
        residual_y = (
            within[:, -1] - controls @ np.linalg.lstsq(controls, within[:, -1], rcond=None)[0]
        )
        result.append(
            {
                "feature": name,
                "n": int(good.sum()),
                "spearman": corr(pair[:, 0], pair[:, 1]),
                "within_view_rank": corr(view_adjusted[:, 0], view_adjusted[:, 1]),
                "partial_rank": corr(residual_x, residual_y),
                "partial_n": int(complete.sum()),
                "control_rank": int(np.linalg.matrix_rank(controls)) if len(controls) else 0,
            }
        )
    return result


def bins(values: Array, feature: str) -> Array:
    if feature == "nearest_deg":
        return np.array([0, 2, 5, 7, 10, 15, 30, 60, 180, 180.00001])
    good = values[np.isfinite(values)]
    if not len(good):
        return np.array([0.0, 1.0])
    edges = np.unique(np.quantile(good, np.linspace(0, 1, 9)))
    if len(edges) < 2:
        edges = np.array([good[0] - 0.5, good[0] + 0.5])
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return np.asarray(edges)


def binned_curve(
    x: Array, y: Array, views: Array, edges: Array, draws: int = 1000
) -> list[dict[str, Any]]:
    """Patch-weighted mean, percentile CI from resampling whole test views.

    Bins are disjoint [low,high); bootstrap-empty bins are omitted from that bin's
    interval, and their frequency is reported. No iid pixel/patch resampling.
    """
    _, group = np.unique(views, return_inverse=True)
    view_count, count_bins = int(group.max()) + 1, len(edges) - 1
    sums, counts = np.zeros((view_count, count_bins)), np.zeros((view_count, count_bins))
    valid = np.isfinite(x) & np.isfinite(y)
    bucket = np.searchsorted(edges, x, side="right") - 1
    valid &= (bucket >= 0) & (bucket < count_bins)
    np.add.at(sums, (group[valid], bucket[valid]), y[valid])
    np.add.at(counts, (group[valid], bucket[valid]), 1)
    rng = np.random.default_rng(SEED)
    multiplicity = rng.multinomial(view_count, np.full(view_count, 1 / view_count), draws)
    boot_count = multiplicity @ counts
    boot_sum = multiplicity @ sums
    boot_mean = np.divide(
        boot_sum, boot_count, out=np.full_like(boot_sum, np.nan), where=boot_count > 0
    )
    result = []
    for i in range(count_bins):
        members = valid & (bucket == i)
        if not members.any():
            continue
        measured = boot_mean[:, i][np.isfinite(boot_mean[:, i])]
        result.append(
            {
                "low": float(edges[i]),
                "high": float(edges[i + 1]),
                "x": float(x[members].mean()),
                "mean": float(y[members].mean()),
                "ci_low": float(np.quantile(measured, 0.025)),
                "ci_high": float(np.quantile(measured, 0.975)),
                "rows": int(members.sum()),
                "views": int((counts[:, i] > 0).sum()),
                "valid_bootstraps": len(measured),
            }
        )
    return result


def variance(y: Array, views: Array, patches: Array) -> dict[str, Any]:
    """Descriptive variance and repeat-cell consistency; never a pure-noise estimate."""
    view_residual = demean(y[:, None], views)[:, 0]
    total = float(np.sum((y - y.mean()) ** 2))
    within = float(np.dot(view_residual, view_residual))
    _, inverse, counts = np.unique(patches, axis=0, return_inverse=True, return_counts=True)
    repeated = np.flatnonzero(counts >= 2)
    index = [np.flatnonzero(inverse == i) for i in repeated]
    result: dict[str, Any] = {
        "total_variance": float(np.var(y)),
        "error_quantiles_p50_p90_p99_max": np.quantile(y, [0.5, 0.9, 0.99, 1]).tolist(),
        "top1_error_share_of_squared_deviation": (
            float(np.sum((y[y >= np.quantile(y, 0.99)] - y.mean()) ** 2) / total) if total else None
        ),
        "between_view_fraction": None if total == 0 else 1 - within / total,
        "within_view_fraction": None if total == 0 else within / total,
        "unique_patches": len(counts),
        "repeated_patches": len(repeated),
        "repeated_rows": int(counts[repeated].sum()),
        "max_repeats": int(counts.max()),
        "noise_ceiling": None,
        "ceiling_reason": "No repeated render/photo measurement at fixed patch AND view; "
        "cross-view differences include real appearance/visibility and cell mismatch.",
    }
    if len(repeated) < 3:
        result["repeat_status"] = "too few repeated cells for split-half or ICC"
        return result
    rng = np.random.default_rng(SEED)
    for label, values in (("raw", y), ("view_demeaned", view_residual)):
        arrays = [values[i] for i in index]
        ns = np.array([len(a) for a in arrays])
        means = np.array([a.mean() for a in arrays])
        n, k = int(ns.sum()), len(ns)
        mean = float(np.dot(ns, means) / n)
        ms_between = float(np.sum(ns * (means - mean) ** 2) / (k - 1))
        ms_within = float(sum(np.sum((a - a.mean()) ** 2) for a in arrays) / (n - k))
        effective = (n - float(np.dot(ns, ns)) / n) / (k - 1)
        component = max(0.0, (ms_between - ms_within) / effective)
        icc = component / (component + ms_within) if component + ms_within else None
        split = []
        first_x, first_y = [], []
        for draw in range(200):
            halves = [np.array_split(rng.permutation(a), 2) for a in arrays]
            left = np.array([a.mean() for a, _ in halves])
            right = np.array([b.mean() for _, b in halves])
            value = corr(left, right)
            if value is not None:
                split.append(value)
            if draw == 0:
                first_x, first_y = left.tolist(), right.tolist()
        result[label] = {
            "icc_assuming_exchangeable_replicates": icc,
            "split_half_pearson_median": float(np.median(split)) if split else None,
            "split_half_allocation_p025_p975": np.quantile(split, [0.025, 0.975]).tolist()
            if split
            else None,
            "first_split_x": first_x,
            "first_split_y": first_y,
            "caution": "Allocation range, NOT a population CI; repeated-cell subset only. "
            "Exchangeable-noise assumption is unverified for different viewpoints.",
        }
    return result


def extend_grid(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact extension only when EVERY observing view's bearing angle survived export."""
    cones, powers = (3, 5, 7, 10), (4, 8, 16)
    observed = [r for r in rows if r["observing_views"] > 0]
    complete = all(len(r["angles_deg"]) == int(r["observing_views"]) for r in observed)
    if not observed or not complete:
        return {
            "status": "not_identifiable_from_saved_statistics",
            "cones": cones,
            "powers": powers,
            "rows_with_raw_angles": sum(bool(r["angles_deg"]) for r in rows),
            "reason": "angles_deg is empty despite observing_views>0. The 24 original "
            "kernel sums and cone counts do not identify narrower-cone/higher-power sums.",
        }
    # Keep production fitting unchanged; patch grid constants only within this CPU call.
    module: Any = importlib.import_module("experiments.coverage_quality.calibration")
    original = module.CONES, module.SHARPNESS
    try:
        module.CONES, module.SHARPNESS = cones, powers
        raw = [{k: v for k, v in row.items() if k != "credit_sums"} for row in rows]
        result = module.fit_credit(raw)
    finally:
        module.CONES, module.SHARPNESS = original
    result["extension_boundary"] = result.get("cone_deg") in (3, 10) or result.get("sharpness") in (
        4,
        16,
    )
    return {"status": "exact_raw_angle_refit", "cones": cones, "powers": powers, "fit": result}


def direction_sampling() -> list[dict[str, Any]]:
    """Geometric discretization only, not an inference of trained-quality hair count."""
    targets = hemisphere(np.array([0.0, 0.0, 1.0]), 10000)
    result = []
    for count in (14, 64, 96, 128, 256):
        directions = hemisphere(np.array([0.0, 0.0, 1.0]), count)
        cosine = np.max(targets @ directions.T, axis=1)
        angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        result.append(
            {
                "directions": count,
                "within10_fraction": float((angles <= 10).mean()),
                "sampled_max_gap_deg": float(angles.max()),
            }
        )
    return result


def copy_table(source: Path, output: Path, scene: str) -> dict[str, Any]:
    raw = source.read_bytes()
    result: dict[str, Any] = {
        "source": str(source),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if len(raw) < 1_000_000:
        target = output / f"{scene}-patches.csv"
        target.write_bytes(raw)
        result.update(copy=target.name, encoding="csv")
    else:
        compressed = gzip.compress(raw, mtime=0)
        result["gzip_bytes"] = len(compressed)
        if len(compressed) < 1_000_000:
            target = output / f"{scene}-patches.csv.gz"
            target.write_bytes(compressed)
            result.update(copy=target.name, encoding="gzip")
        else:
            result.update(
                copy=None, reason="both CSV and gzip exceed 1 MB; source remains read-only"
            )
    return result


def plots(results: dict[str, Any], output: Path) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

    def curve(ax: Any, scene: str, key: str) -> None:
        data = results[scene]["curves"][key]
        missing = data[-1] if key == "nearest_deg" and data[-1]["low"] == 180 else None
        data = data[:-1] if missing else data
        x = np.array([r["x"] for r in data])
        y = np.array([r["mean"] for r in data])
        low = np.array([r["ci_low"] for r in data])
        high = np.array([r["ci_high"] for r in data])
        ax.plot(x, y, "o-", ms=3)
        ax.fill_between(x, low, high, alpha=0.2)
        ax.plot(
            x,
            [r["fitted_mean"] for r in data],
            "--",
            color="tab:orange",
            lw=1,
            label="fitted conditional bin mean",
        )
        if missing:
            ax.errorbar(
                [missing["x"]],
                [missing["mean"]],
                yerr=[
                    [missing["mean"] - missing["ci_low"]],
                    [missing["ci_high"] - missing["mean"]],
                ],
                fmt="s",
                ms=3,
                capsize=2,
                color="tab:blue",
            )
            ax.scatter([missing["x"]], [missing["fitted_mean"]], s=12, color="tab:orange")
        ax.set(xlabel=key, ylabel="held-out combined error", title=scene)
        if key != "parallax_deg":
            ax.set_xscale("symlog", linthresh=2 if key != "best_mm_px" else 0.1)
        if key == "nearest_deg":
            ax.set_xticks([0, 2, 5, 10, 30, 90, 180], ["0", "2", "5", "10", "30", "90", "none"])
        if key == "best_mm_px" and scene == "beach":
            ax.set_title("beach (scale-uncertain)")
        if key == "nearest_deg":
            ax.legend(fontsize=6, loc="upper left")
        ax.grid(alpha=0.15)

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        path = output / name
        fig.savefig(path, dpi=110)
        plt.close(fig)
        if path.stat().st_size >= 300_000:
            raise ValueError(f"plot exceeds 300 KB: {name}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, scene in zip(axes, SCENES, strict=True):
        curve(ax, scene, "nearest_deg")
    save(fig, "nearest-angle.png")
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    for i, scene in enumerate(SCENES):
        for j, feature in enumerate(("within10", "within20", "within30", "observing_views")):
            curve(axes[i, j], scene, feature)
    save(fig, "view-counts.png")
    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    for i, scene in enumerate(SCENES):
        for j, feature in enumerate(("best_mm_px", "parallax_deg")):
            curve(axes[i, j], scene, feature)
    save(fig, "resolution-parallax.png")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, key in zip(axes, ("spearman", "within_view_rank", "partial_rank"), strict=True):
        values = np.array(
            [
                [r[key] if r[key] is not None else np.nan for r in results[scene]["correlations"]]
                for scene in SCENES
            ]
        ).T
        shown = ax.imshow(values, vmin=-0.5, vmax=0.5, cmap="coolwarm", aspect="auto")
        ax.set(
            xticks=[0, 1],
            xticklabels=SCENES,
            yticks=np.arange(len(FEATURES)),
            yticklabels=FEATURES,
            title=key,
        )
        for i in range(len(FEATURES)):
            for j in range(2):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(shown, ax=axes[-1], shrink=0.7)
    save(fig, "rank-correlations.png")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    between = [results[s]["variance"]["between_view_fraction"] for s in SCENES]
    axes[0].bar(SCENES, between, label="between test views")
    axes[0].bar(SCENES, 1 - np.array(between), bottom=between, label="within test views")
    axes[0].set(ylabel="fraction of row error variance", ylim=(0, 1))
    axes[0].legend(fontsize=6)
    for ax, key in zip(axes[1:], ("raw", "view_demeaned"), strict=True):
        repeat = results["c935"]["variance"].get(key)
        if repeat:
            ax.scatter(repeat["first_split_x"], repeat["first_split_y"], s=5, alpha=0.35)
            ax.set(
                xlabel="cell mean: half A",
                ylabel="cell mean: half B",
                title=f"c935 {key}; r={repeat['split_half_pearson_median']:.2f}",
            )
    save(fig, "variance-repeats.png")


def analyze(root: Path, output: Path, draws: int = 1000) -> dict[str, Any]:
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("output must not be inside the read-only input directory")
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    scene_only_actual, scene_only_prediction = [], []
    for scene in SCENES:
        folder = root / f"main-{scene}"
        rows = read_rows(folder / "patches.csv")
        x = np.array(
            [[r[name] if r[name] is not None else np.nan for name in FEATURES] for r in rows]
        )
        y = np.array([r["error"] for r in rows])
        views = np.array([r["view"] for r in rows])
        patches = np.array([[r[k] for k in ("patch_i", "patch_j", "patch_k")] for r in rows])
        _, patch_index, patch_counts = np.unique(
            patches, axis=0, return_inverse=True, return_counts=True
        )
        weights = 1 / patch_counts[patch_index]
        scene_only_actual.extend(y.tolist())
        scene_only_prediction.extend([float(np.average(y, weights=weights))] * len(y))
        saved = json.loads((folder / "fit.json").read_text())
        replay = fit_credit(rows)
        if any(
            not np.isclose(replay[k], saved["error"][k], rtol=1e-10, atol=1e-12)
            for k in ("r2", "leave_view_out_r2")
        ):
            raise ValueError("replayed saved-grid fit differs from campaign")
        observable = np.array([r["observing_views"] > 0 for r in rows])
        results[scene] = {
            "observed_only_nearest_spearman": corr(rank(x[observable, 0]), rank(y[observable])),
            "unobserved_mean_error": float(y[~observable].mean()) if (~observable).any() else None,
            "observed_mean_error": float(y[observable].mean()) if observable.any() else None,
            "rows": len(rows),
            "views": len(np.unique(views)),
            "mean_error": float(y.mean()),
            "unobserved_rows": sum(r["observing_views"] == 0 for r in rows),
            "missing_features": {
                name: int((~np.isfinite(x[:, i])).sum()) for i, name in enumerate(FEATURES)
            },
            "curves": {
                name: binned_curve(x[:, i], y, views, bins(x[:, i], name), draws)
                for i, name in enumerate(FEATURES)
            },
            "correlations": correlations(x, y, views, np.array([r["pixels"] for r in rows])),
            "variance": variance(y, views, patches),
            "saved_grid_replay": replay,
            "boundary_extension": extend_grid(rows),
            "table": copy_table(folder / "patches.csv", output, scene),
        }
        prediction = predict(rows, saved["error"])
        for i, name in enumerate(FEATURES):
            for interval in results[scene]["curves"][name]:
                selected = (
                    np.isfinite(x[:, i])
                    & (x[:, i] >= interval["low"])
                    & (x[:, i] < interval["high"])
                )
                interval["fitted_mean"] = float(prediction[selected].mean())
        for name in ("fit.json", "provenance.json"):
            shutil.copyfile(folder / name, output / f"{scene}-{name}")
    for name in ("fit.json", "provenance.json"):
        shutil.copyfile(root / "pooled" / name, output / f"pooled-{name}")
    plots(results, output)
    for data in results.values():
        for mode in ("raw", "view_demeaned"):
            if mode in data["variance"]:
                for key in ("first_split_x", "first_split_y"):
                    data["variance"][mode].pop(key, None)
    summary = {
        "direction_sampling": direction_sampling(),
        "pooled_scene_only_r2": r_squared(
            np.array(scene_only_actual), np.array(scene_only_prediction)
        ),
        "seed": SEED,
        "bootstrap_draws": draws,
        "bootstrap_unit": "test view",
        "bin_estimand": "patch-row-weighted mean; disjoint bins",
        "partial_controls": "all other geometry ranks + pixel-count rank + test-view fixed effects",
        "scenes": results,
    }
    (output / "diagnostics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    result = analyze(args.results_root, args.out, args.bootstrap)
    for scene, values in result["scenes"].items():
        sys.stdout.write(
            f"{scene}: {values['rows']} rows / {values['views']} views; "
            f"grid extension={values['boundary_extension']['status']}\n"
        )


if __name__ == "__main__":
    main()
