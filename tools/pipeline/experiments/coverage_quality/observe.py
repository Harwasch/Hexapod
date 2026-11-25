"""Reproduce offline observational tables from the supplied read-only artifact snapshot.

Run with hw-py -I and a filename (not -m); only this repository's modules are added to
sys.path. scipy/matplotlib are analysis-only, dynamically imported. Never execute data.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[3] / "captures"))

import cell_grid  # noqa: E402
import holdout_error  # noqa: E402
from experiments.coverage_quality.geometry import Array, features, unit  # noqa: E402
from experiments.lod_compare import decode  # noqa: E402

CANDIDATES = (
    "views",
    "spread",
    "distinct15",
    "fisher",
    "hairs",
    "hairs_resolution",
    "fisher_resolution",
)


def safe(folder: Path, uri: str) -> Path:
    target = (folder / uri).resolve()
    if not target.is_relative_to(folder.resolve()):
        raise ValueError("artifact URI escapes its folder")
    return target


def ranks(values: Array) -> Array:
    return np.asarray(importlib.import_module("scipy.stats").rankdata(values))


def auc(y: Array, score: Array) -> float | None:
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    return float(
        (ranks(score)[y].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def rho(y: Array, score: Array) -> float | None:
    if np.ptp(y) == 0 or np.ptp(score) == 0:
        return None
    return float(np.corrcoef(ranks(y), ranks(score))[0, 1])


def evaluate(
    job: str, outcome: str, y: Array, x: dict[str, Array], *, binary: bool, fov: int
) -> list[dict[str, Any]]:
    rows = []
    for name in CANDIDATES:
        value = auc(y.astype(bool), x[name]) if binary else rho(y, x[name])
        rows.append(
            dict(
                job=job,
                outcome=outcome,
                fov=fov,
                candidate=name,
                n=len(y),
                positive=int(y.sum()) if binary else None,
                value=value,
                metric="auc" if binary else "spearman",
            )
        )
    return rows


def normals_at(points: Array, cloud: Array, centres: Array) -> Array:
    tree = importlib.import_module("scipy.spatial").cKDTree(cloud)
    _, indices = tree.query(points, k=min(16, len(cloud)), workers=4)
    neighbors = cloud[indices]
    delta = neighbors - neighbors.mean(1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", delta, delta)
    normals = np.linalg.eigh(covariance)[1][:, :, 0]
    sign = np.where(np.sum(normals * (centres.mean(0) - points), axis=1) >= 0, 1, -1)
    return normals * sign[:, None]


def geometry(
    points: Array, normals: Array, centres: Array, roi: Array, fov: int, focal: float
) -> dict[str, Array]:
    axes = unit(roi - centres)
    records = []
    for point, normal in zip(points, normals, strict=True):
        toward = unit(point - centres)
        visible = np.sum(toward * axes, axis=1) >= np.cos(np.radians(fov / 2))
        records.append(features(point, normal, centres, visible, focal))
    return {name: np.array([r[name] for r in records]) for name in records[0]}


def leaf_sample(folder: Path, document: dict[str, Any], count: int) -> dict[str, Array]:
    """Uniform priorities over ALL leaf rows; no LOD-parent duplication, one tile in RAM."""
    import splat_tiles

    offsets = splat_tiles.tile_offsets(document)
    leaves: list[str] = []

    def walk(node: dict[str, Any]) -> None:
        if not node.get("children") and node.get("content"):
            leaves.append(node["content"]["uri"])
        for child in node.get("children", []):
            walk(child)

    walk(document["root"])
    rng = np.random.default_rng(2411)
    chosen: dict[str, Array] = {}
    priorities = np.empty(0)
    for uri in sorted(leaves):
        data = decode(safe(folder, uri), offsets.get(uri))
        keys = rng.random(len(data["xyz"]))
        ids = np.argpartition(keys, min(count, len(keys)) - 1)[:count]
        priorities = np.concatenate([priorities, keys[ids]])
        for key in ("xyz", "opacity", "scales"):
            chosen[key] = np.concatenate([chosen.get(key, data[key][:0]), data[key][ids]])
        keep = np.argsort(priorities)[:count]
        priorities = priorities[keep]
        chosen = {k: v[keep] for k, v in chosen.items()}
    return chosen


def grid_points(grid: Any, cloud: Array, maximum: int) -> tuple[Array, Array]:
    plane = grid.plane()
    cells = plane.data
    if grid.kind == "confidence":
        valid = cells[..., 0] > 0
    else:
        valid = (cells[..., grid.channel("support")] & 15) > 0
    j, i = np.nonzero(valid)
    rng = np.random.default_rng(2411)
    selection = np.sort(rng.choice(len(i), min(maximum, len(i)), replace=False))
    i, j = i[selection], j[selection]
    xy = np.column_stack([(i + plane.i0 + 0.5) * plane.cell_m, (j + plane.j0 + 0.5) * plane.cell_m])
    # Match native 2.5D columns to sample height, never pretend they are 10 cm voxels.
    tree = importlib.import_module("scipy.spatial").cKDTree(cloud[:, :2])
    distance, nearest = tree.query(xy, workers=4)
    valid = distance <= plane.cell_m * np.sqrt(2)
    points = np.column_stack([xy, cloud[nearest, 2]])
    return points[valid], cells[j, i][valid]


def run(data_root: Path, out: Path, maximum: int) -> dict[str, Any]:
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    datasets: dict[str, tuple[Array, dict[str, Array]]] = {}
    rng = np.random.default_rng(2411)
    for job_file in sorted((data_root / "jobs").glob("*.json")):
        job = json.loads(job_file.read_text())
        job_id = str(job["id"])
        generations = sorted((data_root / "pub" / "runs" / job_id).glob("*"))
        runs = [p for p in generations if (p / "place/coverage_enu.ply").is_file()]
        if not runs:
            metadata.append(dict(job=job_id[:8], excluded="no coverage/cameras"))
            continue
        if len(runs) != 1:
            raise ValueError("select a single generation per job in snapshot")
        run_root = runs[0]
        coverage = holdout_error.read_ply(run_root / "place/coverage_enu.ply")
        all_xyz = np.column_stack([coverage[k] for k in ("x", "y", "z")]).astype(float)
        tiers = coverage["tier"]
        if not np.isfinite(all_xyz).all():
            raise ValueError("nonfinite coverage centre")
        centres = all_xyz[tiers == 3]
        cloud, tier = all_xyz[tiers < 3], tiers[tiers < 3]
        if len(cloud) < 16 or not len(centres):
            continue
        # Equal weight per occupied 10 cm voxel; outcome = majority sample rows are keep.
        keys, inverse, counts = np.unique(
            np.floor(cloud / 0.1).astype(int), axis=0, return_inverse=True, return_counts=True
        )
        first = np.unique(inverse, return_index=True)[1]
        kept = np.bincount(inverse, weights=tier == 2) / counts
        chosen = np.sort(rng.choice(len(keys), min(maximum, len(keys)), replace=False))
        points, y = cloud[first[chosen]], kept[chosen] >= 0.5
        normals = normals_at(points, cloud, centres)
        roi = np.median(cloud, axis=0)
        steps = {s["stageId"]: s.get("metrics") or {} for s in job["steps"]}
        width = steps.get("normalize", {}).get("width", 1080)
        focal = float(width) / (2 * np.tan(np.radians(35)))
        record: dict[str, Any] = dict(
            job=job_id[:8],
            cameras=len(centres),
            sample_gaussians=len(cloud),
            voxels=len(keys),
            evaluated_voxels=len(y),
            keep_voxels=int(y.sum()),
            train=steps.get("train"),
            quality=steps.get("quality"),
        )
        # Do not copy large job metrics to small summary.
        for key, fields in (
            (
                "train",
                (
                    "gaussians",
                    "trainFrames",
                    "valFrames",
                    "trainSeconds",
                    "metricsSplit",
                    "holdoutError",
                    "shDegree",
                ),
            ),
            ("quality", ("keep", "context", "drop", "keepPct", "demotedByHeldOut", "keepVerified")),
        ):
            record[key] = {k: record[key].get(k) for k in fields}
        total = sum(record["quality"].get(k) or 0 for k in ("keep", "context", "drop"))
        demoted = record["quality"].get("demotedByHeldOut") or 0
        record["structural_rule_tier_agreement"] = 1 - demoted / total
        record["geometry_keep_demotion_rate"] = (
            demoted / (record["quality"]["keep"] + demoted)
            if record["quality"]["keep"] + demoted
            else 0
        )
        for fov in (50, 70, 90):
            x = geometry(points, normals, centres, roi, fov, focal)
            rows += evaluate(job_id[:8], "tier_keep_circular", y, x, binary=True, fov=fov)
            if fov == 70:
                datasets[job_id[:8]] = (y, x)
                record["geometry_medians"] = {k: float(np.median(v)) for k, v in x.items()}
        folder = run_root / "package/splat"
        document = json.loads((folder / "tileset.json").read_text())
        grids = cell_grid.from_tileset(folder / "tileset.json")
        for name, grid in grids.items():
            if name not in ("confidence", "surface"):
                continue
            gp, values = grid_points(grid, cloud, maximum)
            if not len(gp):
                continue
            gx = geometry(gp, normals_at(gp, cloud, centres), centres, roi, 70, focal)
            if name == "confidence":
                reason = values[:, grid.channel("reasons")].astype(int)
                accuracy = (reason & 8) > 0
                record["accuracy_cells"] = {
                    "sampled": len(gp),
                    "flagged": int(accuracy.sum()),
                    "cell_m": grid.plane().cell_m,
                    "all_flagged": int(((grid.plane().data[..., 1] & 8) > 0).sum()),
                }
                if record["train"].get("valFrames"):
                    eligible = (reason & 7) == 0
                    if eligible.any():
                        rows += evaluate(
                            job_id[:8],
                            "accuracy_absent_no_geometry_reason",
                            ~accuracy[eligible],
                            {k: v[eligible] for k, v in gx.items()},
                            binary=True,
                            fov=70,
                        )
                        record["accuracy_cells"]["no_geometry_reason"] = int(eligible.sum())
                        record["accuracy_cells"]["eligible_flagged"] = int(accuracy[eligible].sum())
                    # Higher score should predict absence of accuracy flag (not verified pass).
                    rows += evaluate(
                        job_id[:8], "accuracy_flag_absent", ~accuracy, gx, binary=True, fov=70
                    )
                    for fov in (50, 90):
                        other = geometry(
                            gp, normals_at(gp, cloud, centres), centres, roi, fov, focal
                        )
                        rows += evaluate(
                            job_id[:8],
                            "accuracy_flag_absent",
                            ~accuracy,
                            other,
                            binary=True,
                            fov=fov,
                        )
            else:
                # Surface availability; includes thin (<3 samples) and out-of-range cells.
                thin = values[:, grid.channel("dsm")] == cell_grid.NO_HEIGHT
                rows += evaluate(
                    job_id[:8], "surface_present_proxy", ~thin, gx, binary=True, fov=70
                )
        data = leaf_sample(folder, document, maximum)
        if data:
            gx = geometry(
                data["xyz"], normals_at(data["xyz"], cloud, centres), centres, roi, 70, focal
            )
            rows += evaluate(
                job_id[:8], "published_opacity", data["opacity"], gx, binary=False, fov=70
            )
            anisotropy = np.log(data["scales"].max(1) / data["scales"].min(1))
            rows += evaluate(
                job_id[:8], "published_log_anisotropy", anisotropy, gx, binary=False, fov=70
            )
        metadata.append(record)
        sys.stdout.write(
            f"{job_id[:8]}: {len(cloud)} sample gaussians, {len(centres)} cameras, "
            f"{len(y)} voxels ({int(y.sum())} keep)\n"
        )
        sys.stdout.flush()
    # Leave-one-job-out linear probability baseline, standardized
    # on training jobs only; compare to fixed candidates without tuning their sign.
    lojo = []
    for held, (ytest, xtest) in datasets.items():
        others = [d for j, d in datasets.items() if j != held]
        for names in [(name,) for name in CANDIDATES] + [CANDIDATES]:
            xtrain = np.concatenate([np.column_stack([x[n] for n in names]) for _, x in others])
            ytrain = np.concatenate([y for y, _ in others])
            mean, std = xtrain.mean(0), np.maximum(xtrain.std(0), 1e-8)
            design = np.column_stack([np.ones(len(xtrain)), (xtrain - mean) / std])
            coef = np.linalg.lstsq(design, ytrain, rcond=None)[0]
            test = np.column_stack(
                [np.ones(len(ytest)), (np.column_stack([xtest[n] for n in names]) - mean) / std]
            )
            lojo.append(
                dict(job=held, candidate="+".join(names), n=len(ytest), auc=auc(ytest, test @ coef))
            )
    summary = dict(
        seed=2411,
        maximum=maximum,
        assumptions="ROI-looking; cone FOV; no occlusion",
        jobs=metadata,
        results=rows,
        leave_one_job_out=lojo,
        runtime_seconds=round(time.monotonic() - started, 2),
        peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    )
    out.mkdir(parents=True, exist_ok=True)
    plot(summary, out)
    summary.update(
        runtime_seconds=round(time.monotonic() - started, 2),
        peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    )
    (out / "summary.json").write_text(
        json.dumps(summary, separators=(",", ":"), allow_nan=False) + "\n"
    )
    return summary


def plot(summary: dict[str, Any], out: Path) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    for outcome, metric in (
        ("tier_keep_circular", "AUC (circular tier)"),
        ("accuracy_flag_absent", "AUC (absence of accuracy flag)"),
        ("published_log_anisotropy", "Spearman rho (anisotropy proxy)"),
    ):
        rows = [r for r in summary["results"] if r["outcome"] == outcome and r["fov"] == 70]
        fig, ax = plt.subplots(figsize=(8, 3.3))
        jobs = sorted({r["job"] for r in rows})
        values = np.array(
            [
                [
                    next(
                        (r["value"] for r in rows if r["job"] == job and r["candidate"] == c), None
                    )
                    for c in CANDIDATES
                ]
                for job in jobs
            ],
            dtype=float,
        )
        im = ax.imshow(values, vmin=0 if "AUC" in metric else -1, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(CANDIDATES)), CANDIDATES, rotation=22, ha="right", fontsize=8)
        ax.set_yticks(range(len(jobs)), jobs, fontsize=8)
        for j in range(len(jobs)):
            for i in range(len(CANDIDATES)):
                ax.text(
                    i,
                    j,
                    "NA" if np.isnan(values[j, i]) else f"{values[j, i]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if values[j, i] < 0.6 else "black",
                )
        ax.set_title(metric)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / f"{outcome}.png", dpi=110)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maximum", type=int, default=1000)
    args = parser.parse_args()
    run(args.data_root, args.out, args.maximum)


if __name__ == "__main__":
    main()
