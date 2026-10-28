"""CPU structure probe, not semantic segmentation. See docs/LIVING_STRUCTURE.md.

Read immutable published SPZ tiles, without counting replacement LOD parents twice.
Graph components test whether appearance/geometry expose natural cuts. Colonisation
tests the benefit and evidential limits of adding skeleton nodes inside the crown.
All thresholds are experimental priors, swept rather than claimed as calibrated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from living_articulation import tile_data
from splat_tiles import tile_offsets


def leaf_uris(doc: dict) -> list[str]:
    out = []

    def walk(tile):
        if tile.get("children"):
            for child in tile["children"]:
                walk(child)
        elif "content" in tile:
            out.append(tile["content"]["uri"])

    walk(doc["root"])
    return out


def load_scan(source: Path, out: Path) -> dict:
    """Preserve canonical order and all covariance/opacity/DC colour attributes."""
    folder = source / "splat"
    doc = json.loads((folder / "tileset.json").read_text())
    offsets = tile_offsets(doc)
    arrays: dict[str, list] = {k: [] for k in ("points", "rgb", "scales", "quat", "alpha")}
    tiles = []
    digest = hashlib.sha256()
    for uri in leaf_uris(doc):
        digest.update(uri.encode())
        digest.update((folder / uri).read_bytes())
        c, p = tile_data(folder, uri, offsets)
        arrays["points"].append(p)
        arrays["rgb"].append(
            np.clip(np.stack([c[f"f_dc_{i}"] for i in range(3)], 1) * 0.2820947918 + 0.5, 0, 1)
        )
        arrays["scales"].append(np.exp(np.stack([c[f"scale_{i}"] for i in range(3)], 1)))
        arrays["quat"].append(np.stack([c[f"rot_{i}"] for i in (1, 2, 3, 0)], 1))
        arrays["alpha"].append(1 / (1 + np.exp(-c["opacity"])))
        tiles.append({"uri": uri, "count": len(p)})
    data = {k: np.concatenate(v).astype(np.float32) for k, v in arrays.items()}
    np.savez(
        out / "scan.npz",
        points=data["points"],
        rgb=data["rgb"],
        scales=data["scales"],
        quat=data["quat"],
        alpha=data["alpha"],
    )
    meta = {"sha256LeafUriAndBytes": digest.hexdigest(), "tiles": tiles}
    (out / "source.json").write_text(json.dumps(meta, indent=2) + "\n")
    return data


def representatives(points: np.ndarray, cell: float, seed: int):
    """One seeded representative per voxel; full-detail memberships remain measurable."""
    rng = np.random.default_rng(seed)
    keys = np.floor(points / cell).astype(np.int32)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = rng.permutation(len(points))
    _, first = np.unique(inverse[order], return_index=True)
    return order[first], inverse


def nearest_segments(points: np.ndarray, nodes: list[dict], step: float = 0.02):
    """Sampled-segment search, then exact projection on selected segment.

    Returned distance overestimates the global minimum by at most step/2. This is
    attachment proximity, never evidence of a real twig between skeleton and foliage.
    """
    rest = np.array([n["position"] for n in nodes])
    parents = np.array([n["parent"] for n in nodes])
    samples, owners = [], []
    for i, parent in enumerate(parents):
        if parent < 0:
            continue
        length = np.linalg.norm(rest[i] - rest[parent])
        t = np.linspace(0, 1, max(2, int(np.ceil(length / step)) + 1))
        samples.append(rest[parent] + t[:, None] * (rest[i] - rest[parent]))
        owners.extend([i] * len(t))
    if not samples:
        raise ValueError("At least one skeleton segment is required")
    _, index = cKDTree(np.concatenate(samples)).query(points, workers=4)
    owner = np.asarray(owners)[index]
    start = rest[parents[owner]]
    vector = rest[owner] - start
    t = np.clip(
        np.sum((points - start) * vector, 1) / np.maximum(np.sum(vector**2, 1), 1e-12), 0, 1
    )
    return np.linalg.norm(points - start - t[:, None] * vector, axis=1), owner


def normals(scales: np.ndarray, quat: np.ndarray):
    small = np.argmin(scales, axis=1)
    axis = np.eye(3)[small]
    return Rotation.from_quat(quat).apply(axis)


def graph_components(
    points, rgb, scales, quat, alpha, radius, colour, normal_cos=0.8, alpha_min=0.15
):
    """Bounded 12-NN graph avoids quadratic crown adjacency memory.

    Normals are unoriented (abs dot); only flat splats have a meaningful normal.
    Opacity and maximum covariance radius reject broad transparent haze as bridges.
    """
    n = len(points)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    distance, near = cKDTree(points).query(points, k=min(13, n), workers=4)
    if n == 1:
        return np.zeros(1, dtype=np.int32)
    row = np.repeat(np.arange(n), near.shape[1] - 1)
    col = near[:, 1:].ravel()
    d = distance[:, 1:].ravel()
    active = (alpha >= alpha_min) & (scales.max(1) <= 0.12)
    keep = (d <= radius) & active[row] & active[col]
    keep &= np.linalg.norm(rgb[row] - rgb[col], axis=1) <= colour
    norm = normals(scales, quat)
    sorted_scale = np.sort(scales, axis=1)
    flat = sorted_scale[:, 0] < sorted_scale[:, 1] * 0.5
    cosine = abs(np.sum(norm[row] * norm[col], axis=1))
    keep &= ~flat[row] | ~flat[col] | (cosine >= normal_cos)
    graph = coo_matrix((np.ones(keep.sum(), dtype=np.uint8), (row[keep], col[keep])), (n, n))
    return connected_components(graph, directed=False)[1]


def piece_table(points, scales, labels, distance):
    """AABB diagonal includes 2 sigma support; not a leaf length estimate."""
    _, labels = np.unique(labels, return_inverse=True)
    n = int(labels.max()) + 1
    count = np.bincount(labels, minlength=n)
    low, high = np.full((n, 3), np.inf), np.full((n, 3), -np.inf)
    np.minimum.at(low, labels, points - 2 * scales.max(1)[:, None])
    np.maximum.at(high, labels, points + 2 * scales.max(1)[:, None])
    contact = np.full(n, np.inf)
    np.minimum.at(contact, labels, distance)
    centre = np.column_stack([np.bincount(labels, weights=points[:, j]) / count for j in range(3)])
    return {
        "count": count,
        "diameter": np.linalg.norm(high - low, axis=1),
        "contact": contact,
        "centre": centre,
    }


def describe(points, scales, labels, distance):
    table = piece_table(points, scales, labels, distance)
    count, size, contact = (table[k] for k in ("count", "diameter", "contact"))
    # Deliberately generous leaf-cluster prior, not a botanical species identification.
    viable = (count >= 8) & (size >= 0.03) & (size <= 0.30)
    attached = viable & (contact <= 0.04)
    return {
        "pieces": len(count),
        "largestShare": float(count.max() / count.sum()),
        "splatsInSingletonComponents": float(count[count == 1].sum() / count.sum()),
        "sizeQuantilesM": quantiles(size),
        "splatsPerPieceQuantiles": quantiles(count),
        "plausiblePieces": int(viable.sum()),
        "plausibleCrownShare": float(count[viable].sum() / count.sum()),
        "contactSupportedPieces": int(attached.sum()),
        "contactSupportedCrownShare": float(count[attached].sum() / count.sum()),
        "mechanicallyBindableCrownShare": 1.0,
        "over30cmCrownShare": float(count[size > 0.30].sum() / count.sum()),
        "under3cmCrownShare": float(count[size < 0.03].sum() / count.sum()),
    }


def quantiles(values):
    return np.quantile(values, [0, 0.1, 0.5, 0.9, 0.99, 1]).tolist()


def adjusted_rand(a, b):
    """Sparse contingency; never allocate number-of-pieces squared."""
    _, na = np.unique(a, return_counts=True)
    _, nb = np.unique(b, return_counts=True)
    _, both = np.unique(np.column_stack([a, b]), axis=0, return_counts=True)
    pairs = lambda x: np.sum(x.astype(float) * (x - 1) / 2)
    total = len(a) * (len(a) - 1) / 2
    if total == 0:
        return 1.0
    expected = pairs(na) * pairs(nb) / total
    upper = (pairs(na) + pairs(nb)) / 2
    return float((pairs(both) - expected) / (upper - expected)) if upper != expected else 1.0


def colonise(points, nodes, seed, limit=1000, step=0.06, attraction=0.35, kill=0.08):
    """Space colonisation from the existing rig, limited to the round-3 player's budget.

    Attractions are measured splat centres, not known wood. Branch connectivity is
    synthetic. Randomise candidate priority so stability is not guaranteed by grid bins.
    """
    nodes = [dict(n) for n in nodes]
    rng = np.random.default_rng(seed)
    active = points.copy()
    for _ in range(120):
        rest = np.array([n["position"] for n in nodes])
        distance, owner = cKDTree(rest).query(active, workers=4)
        active = active[distance > kill]
        owner, distance = owner[distance > kill], distance[distance > kill]
        use = distance < attraction
        if not use.any() or len(nodes) >= limit:
            break
        p, own, d = active[use], owner[use], distance[use]
        direction = np.zeros_like(rest)
        np.add.at(direction, own, (p - rest[own]) / d[:, None])
        length = np.linalg.norm(direction, axis=1)
        candidates = rng.permutation(np.flatnonzero(length > 1e-8))
        added = 0
        for i in candidates:
            pos = rest[i] + direction[i] / length[i] * step
            if cKDTree(rest).query(pos)[0] < step * 0.7:
                continue
            nodes.append(
                {
                    "id": f"inferred-{len(nodes)}",
                    "parent": int(i),
                    "position": pos.tolist(),
                    "band": "branch",
                }
            )
            added += 1
            if len(nodes) >= limit:
                break
        if not added:
            break
    return nodes


def wood_prediction(rgb, scales, distance):
    """Uncalibrated, conservative woody candidate; audit ambiguous dark leaves explicitly."""
    green = (2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]) / np.maximum(rgb.sum(1), 0.03)
    axes = np.sort(scales, axis=1)
    elongated = axes[:, 2] / np.maximum(axes[:, 1], 1e-6) > 2
    return (green < 0.06) & ((distance < 0.10) | elongated)


def run(source: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cached = out / "scan.npz"
    data = dict(np.load(cached)) if cached.exists() else load_scan(source, out)
    points, rgb, scales = (data[k] for k in ("points", "rgb", "scales"))
    nodes = json.loads((source / "source/rig.json").read_text())["nodes"]
    distance, owner = nearest_segments(points, nodes)
    # Fixed geometric crown ROI independent of colour and classifier outputs.
    crown = points[:, 2] >= 2.0
    ci = np.flatnonzero(crown)
    p, s = points[crown], scales[crown]
    summary = {
        "splats": len(points),
        "crownSplats": int(crown.sum()),
        "crownMinZM": 2.0,
        "boundsM": [points.min(0).tolist(), points.max(0).tolist()],
        "axisSigmaQuantilesM": [quantiles(scales[crown, j]) for j in range(3)],
        "maxSigmaQuantilesM": quantiles(scales[crown].max(1)),
        "skeletonDistanceQuantilesM": quantiles(distance[crown]),
        "originalNodes": len(nodes),
        "arms": {},
        "refinement": {},
    }
    all_labels = {}
    for seed in (17, 29):
        index, inverse = representatives(p, 0.015, seed)
        selected = ci[index]
        print(f"seed {seed}: {len(index)} 15mm representatives", flush=True)
        nn = cKDTree(p[index]).query(p[index], k=2, workers=4)[0][:, 1]
        summary["representativeSpacingQuantilesM"] = quantiles(nn)
        for radius, colour, cosine in ((0.025, 0.12, 0.8), (0.05, 0.24, 0.8), (0.10, 0.36, 0.5)):
            key = f"r{radius}-c{colour}-n{cosine}"
            labels = graph_components(
                data["points"][selected],
                data["rgb"][selected],
                data["scales"][selected],
                data["quat"][selected],
                data["alpha"][selected],
                radius,
                colour,
                cosine,
            )[inverse]
            stats = describe(p, s, labels, distance[crown])
            summary["arms"][f"{key}-seed{seed}"] = stats
            all_labels[key, seed] = labels
            np.save(out / f"labels-{key}-seed{seed}.npy", labels)
            print(key, seed, json.dumps(stats), flush=True)
        refined = colonise(p[index], nodes, seed)
        refined_distance, refined_owner = nearest_segments(points, refined)
        np.save(out / f"refined-owner-{seed}.npy", refined_owner)
        (out / f"refined-rig-{seed}.json").write_text(
            json.dumps({"nodes": refined}, indent=2) + "\n"
        )
        new = np.array([n["position"] for n in refined[len(nodes) :]])
        pred = wood_prediction(rgb, scales, distance)
        wood_distance = (
            cKDTree(points[pred]).query(new, workers=4)[0] if len(new) else np.array([0.0])
        )
        summary["refinement"][str(seed)] = {
            "nodes": len(refined),
            "newNodes": len(new),
            "crownWithin10cmBefore": float(np.mean(distance[crown] <= 0.10)),
            "crownWithin10cmAfter": float(np.mean(refined_distance[crown] <= 0.10)),
            "newNodeWoodCandidateDistanceM": quantiles(wood_distance),
            "newNodesWithin3cmWoodCandidate": float(np.mean(wood_distance <= 0.03)),
            "nearestSegmentPieceMetrics": describe(
                p, s, refined_owner[crown], refined_distance[crown]
            ),
        }
    for key in {k for k, _ in all_labels}:
        summary["arms"][key + "-stabilityARI"] = adjusted_rand(
            all_labels[key, 17], all_labels[key, 29]
        )
    a = json.loads((out / "refined-rig-17.json").read_text())["nodes"][len(nodes) :]
    b = json.loads((out / "refined-rig-29.json").read_text())["nodes"][len(nodes) :]
    if a and b:
        d = cKDTree([n["position"] for n in b]).query([n["position"] for n in a])[0]
        summary["refinement"]["seed17NodesWithin5cmSeed29"] = float(np.mean(d <= 0.05))
    pred = wood_prediction(rgb, scales, distance)
    np.savez(out / "derived.npz", crown=crown, skeleton_distance=distance, owner=owner, wood=pred)
    summary["wood"] = {
        "predictedWoodShare": float(pred.mean()),
        "predictedWoodCrownShare": float(pred[crown].mean()),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["refinement"], indent=2), flush=True)


def native_probe(out: Path, renders: Path):
    """Control for representative-grid fragmentation, using every crown splat.

    Seed changes KD-tree input order, including tie ordering on the SPZ grid.
    No artificial jitter is added to measured positions. This control is kept
    separate from the voxel-seeded probe and never replaces its reported results.
    """
    data = dict(np.load(out / "scan.npz"))
    derived = dict(np.load(out / "derived.npz"))
    crown = derived["crown"]
    points, scales = data["points"][crown], data["scales"][crown]
    distance = derived["skeleton_distance"][crown]
    summary, labels_all = {}, {}
    for radius, colour, cosine, seed in (
        (0.025, 0.12, 0.8, 17),
        (0.025, 0.12, 0.8, 29),
        (0.05, 0.24, 0.8, 17),
        (0.10, 0.36, 0.5, 17),
    ):
        order = np.random.default_rng(seed).permutation(len(points))
        selected = np.flatnonzero(crown)[order]
        raw = graph_components(
            data["points"][selected],
            data["rgb"][selected],
            data["scales"][selected],
            data["quat"][selected],
            data["alpha"][selected],
            radius,
            colour,
            cosine,
        )
        labels = np.empty_like(raw)
        labels[order] = raw
        key = f"r{radius}-seed{seed}"
        labels_all[key] = labels
        np.save(out / f"native-labels-{key}.npy", labels)
        stats = describe(points, scales, labels, distance)
        for view in ("tree", "crown", "crown-back"):
            image = np.load(renders / f"{view}-contributions.npz")
            ids, best = image["ids"], image["best"]
            full = np.full(len(crown), -1, dtype=np.int32)
            full[crown] = labels
            use = (ids >= 0) & (best >= 0.05)
            visible = full[ids[use]]
            _, pixels = np.unique(visible[visible >= 0], return_counts=True)
            stats[view + "Visibility"] = {
                "atLeast16Pixels": int(np.sum(pixels >= 16)),
                "atLeast64Pixels": int(np.sum(pixels >= 64)),
            }
        summary[key] = stats
        print(key, json.dumps(stats), flush=True)
    summary["tightOrderSeedARI"] = adjusted_rand(
        labels_all["r0.025-seed17"], labels_all["r0.025-seed29"]
    )
    (out / "native-metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--native-renders", type=Path)
    args = parser.parse_args()
    if args.native_renders:
        native_probe(args.out, args.native_renders)
    else:
        run(args.source, args.out)
