"""Conservative leaf-resolution gate, adapted from the living-structure CPU probe.

One bounded 12-NN graph over a deterministic native-splat sample (not LOD parents).
A piece needs >=8 splats, 3..30 cm 2-sigma support diagonal, and skeleton contact
within 4 cm. At least 65% of the crown must pass every gate. This is a permission
threshold chosen to reject ambiguous crown masses, not a calibrated leaf detector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from living_articulation import tile_data
from splat_tiles import tile_offsets

THRESHOLD = 0.65


def has_segmented_vegetation(instances: list[dict]) -> bool:
    """Conservative permission: known ground/rigid/static classes win over plant scores."""
    classes = {"trees", "shrubs", "grass", "crops", "flowers", "ferns", "moss", "vegetation"}
    for instance in instances:
        props = instance.get("properties", {})
        if (
            instance.get("behaviour") == "static"
            or props.get("rigid", 0) >= 0.5
            or props.get("static", 0) >= 0.8
        ):
            continue
        category = instance.get("category")
        if category in classes or (not category and props.get("vegetation", 0) >= 0.5):
            return True
    return False


def leaf_uris(doc: dict) -> list[str]:
    out: list[str] = []

    def walk(t):
        if t.get("children"):
            for c in t["children"]:
                walk(c)
        elif t.get("content"):
            out.append(t["content"]["uri"])

    walk(doc["root"])
    return out


def nearest_skeleton(points: np.ndarray, nodes: list[dict]) -> np.ndarray:
    if not nodes:
        return np.full(len(points), np.inf)
    rest = np.array([n["position"] for n in nodes])
    samples = []
    for n in nodes:
        if n["parent"] < 0:
            continue
        start, end = rest[n["parent"]], np.array(n["position"])
        t = np.linspace(0, 1, max(2, int(np.ceil(np.linalg.norm(end - start) / 0.02)) + 1))
        samples.append(start + t[:, None] * (end - start))
    if not samples:
        return np.full(len(points), np.inf)
    # At most +1cm overestimate; conservative contact, not botanical attachment.
    return cKDTree(np.concatenate(samples)).query(points, workers=4)[0]


def measure(points, rgb, scales, quat, alpha, nodes: list[dict], vegetation: bool) -> dict:
    n = len(points)
    if n == 0:
        return {
            "vegetation": vegetation,
            "fittedRig": bool(nodes),
            "separableCrownShare": 0.0,
            "attachedCrownShare": 0.0,
            "smallSplatShare": 0.0,
            "leafBindings": False,
            "score": 0.0,
            "tier": "B" if nodes and vegetation else "A" if vegetation else "off",
            "sampledCrownSplats": 0,
        }
    distance, near = cKDTree(points).query(points, k=min(13, n), workers=4)
    if n == 1:
        labels = np.zeros(1, dtype=int)
    else:
        row = np.repeat(np.arange(n), near.shape[1] - 1)
        col, d = near[:, 1:].ravel(), distance[:, 1:].ravel()
        active = (alpha >= 0.15) & (scales.max(1) <= 0.12)
        keep = (d <= 0.025) & active[row] & active[col]
        keep &= np.linalg.norm(rgb[row] - rgb[col], axis=1) <= 0.12
        norm = Rotation.from_quat(quat).apply(np.eye(3)[np.argmin(scales, axis=1)])
        ordered = np.sort(scales, axis=1)
        flat = ordered[:, 0] < ordered[:, 1] * 0.5
        keep &= ~flat[row] | ~flat[col] | (abs(np.sum(norm[row] * norm[col], 1)) >= 0.8)
        graph = coo_matrix((np.ones(keep.sum(), dtype=np.uint8), (row[keep], col[keep])), (n, n))
        labels = connected_components(graph, directed=False)[1]
    count = np.bincount(labels)
    low, high = np.full((len(count), 3), np.inf), np.full((len(count), 3), -np.inf)
    np.minimum.at(low, labels, points - 2 * scales.max(1)[:, None])
    np.maximum.at(high, labels, points + 2 * scales.max(1)[:, None])
    size = np.linalg.norm(high - low, axis=1)
    viable = (count >= 8) & (size >= 0.03) & (size <= 0.30)
    contact = np.full(len(count), np.inf)
    skeleton_distance = nearest_skeleton(points, nodes)
    np.minimum.at(contact, labels, skeleton_distance)
    shares = [
        float(count[viable].sum() / n),
        float(count[viable & (contact <= 0.04)].sum() / n),
        float(np.mean(4 * scales.max(1) <= 0.10)),
    ]
    # 10cm nominal lamina (Tadrist 2018); require splat 4-sigma diameter <= it.
    score = min(shares)
    tier = "B" if nodes else "A"
    if not vegetation:
        tier = "off"
    return {
        "vegetation": vegetation,
        "fittedRig": bool(nodes),
        "separableCrownShare": shares[0],
        "attachedCrownShare": shares[1],
        "smallSplatShare": shares[2],
        "leafBindings": False,
        "score": score,
        "threshold": THRESHOLD,
        "tier": tier,
        "sampledCrownSplats": n,
        "nearSkeletonCrownShare": float(np.mean(skeleton_distance <= 0.10)),
        "pieces": len(count),
        "viablePieces": int(viable.sum()),
        "leafLengthPriorM": 0.10,
    }


def check(source: Path, budget: int = 400000, ground_override: float | None = None) -> dict:
    folder = source / "splat"
    doc = json.loads((folder / "tileset.json").read_text())
    offsets = tile_offsets(doc)
    arrays: dict[str, list] = {k: [] for k in ("points", "rgb", "scales", "quat", "alpha")}
    uris = leaf_uris(doc)
    # Equal per-tile quotas avoid an unbounded all-camp allocation; record the actual sample.
    quota = max(1, budget // max(1, len(uris)))
    total = 0
    for uri in uris:
        c, p = tile_data(folder, uri, offsets)
        total += len(p)
        ix = np.linspace(0, len(p) - 1, min(quota, len(p)), dtype=int)
        arrays["points"].append(p[ix])
        arrays["rgb"].append(
            np.clip(np.stack([c[f"f_dc_{i}"][ix] for i in range(3)], 1) * 0.2820947918 + 0.5, 0, 1)
        )
        arrays["scales"].append(np.exp(np.stack([c[f"scale_{i}"][ix] for i in range(3)], 1)))
        arrays["quat"].append(np.stack([c[f"rot_{i}"][ix] for i in (1, 2, 3, 0)], 1))
        arrays["alpha"].append(1 / (1 + np.exp(-c["opacity"][ix])))
    a = {k: np.concatenate(v) for k, v in arrays.items()}
    # Vegetation class comes from published segmentation when available; a green fallback
    # only nominates candidates and never enables Tier C or calls them actual leaves.
    ground = (
        float(np.quantile(a["points"][:, 2], 0.01)) if ground_override is None else ground_override
    )
    height = a["points"][:, 2] - ground
    rgb = a["rgb"]
    green = (2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]) / np.maximum(rgb.sum(1), 0.03) > 0.06
    instances = source / "splat/instances.json"
    declared_vegetation = False
    if instances.exists():
        table = json.loads(instances.read_text()).get("instances", [])
        declared_vegetation = has_segmented_vegetation(table)
    vegetation = (
        declared_vegetation
        if instances.exists()
        else float(np.mean(green & (height > 0.08))) > 0.001
    )
    crown = (height >= 2) if np.quantile(height, 0.99) >= 3 else (height >= 0.08)
    rig_file = source / "source/rig.json"
    nodes = json.loads(rig_file.read_text()).get("nodes", []) if rig_file.exists() else []
    result = measure(
        *(a[k][crown] for k in ("points", "rgb", "scales", "quat", "alpha")), nodes, vegetation
    )
    result.update(
        {
            "terminalSplats": total,
            "sampledSplats": len(height),
            "terminalTiles": len(uris),
            "vegetationEvidence": "published vegetation with rigid/static/ground exclusions"
            if instances.exists()
            else "green/height candidate mask",
            "sampled": total > len(height),
            "groundEstimateM": ground,
        }
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--budget", type=int, default=400000)
    parser.add_argument("--ground", type=float)
    args = parser.parse_args()
    result = check(args.source, args.budget, args.ground)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
