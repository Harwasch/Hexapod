"""Geometric skeleton extraction: a real splat capture to a Living Survey ``MotionRig``.

``synthetic_tree.py`` builds a tree whose skeleton is known because it generated it. This module
does the opposite and much harder thing: given a 3DGS PLY of a *real* scene and a bounding region
that contains one tree, it isolates that tree and infers a skeleton from geometry alone.

The method is height banding plus connectivity clustering, which is the classic approach and the
one the sprint plan names:

1. **Isolate** — keep splats inside the region, above the estimated ground, opaque enough to be
   surface rather than haze, and with enough neighbours to be structure rather than a floater.
2. **Band** — slice the isolated points into horizontal bands. A band cutting a tree crosses the
   trunk once and each branch once, so *within a band* the connected components are the separate
   limbs at that height. This is what turns an unstructured cloud into a graph.
3. **Cluster** — connected components within each band, linked at ``link_radius``. Each component
   above a minimum size becomes one skeleton node at its centroid.
4. **Parent** — each node attaches to the node in a lower band whose points come nearest to its
   own. Branches lean, so nearest *point* beats nearest centroid.
5. **Prune and label** — merge the smallest subtrees away until the rig is within ``max_nodes``,
   then label bands: the trunk is the main chain up to the first real fork, terminal nodes are
   leaves, everything between is branch.

The output is the schema in ``packages/world/src/rig.ts``, so ``parseRig`` accepts it directly.

**What this cannot do.** It recovers the *shape* of the tree, not its biomechanics. Stiffness is
interpolated from the same uncalibrated constants ``syntheticTreeRig()`` uses — a plausible
gradient, not a measurement. Nothing here is validated against how a real tree of this species
actually moves, and the runtime labels the resulting motion Simulated for exactly that reason.

Run it against the synthetic tree, where ground truth exists, to see what it is worth::

    uv run python skeleton.py ../../data/tiles/synthetic-tree/source/splat.ply /tmp/out \\
        --lat 28.0389 --lon -82.6966 \\
        --labels ../../data/tiles/synthetic-tree/source/labels.json \\
        --truth-rig ../../data/tiles/synthetic-tree/source/rig.json

Usage:
    python skeleton.py capture.ply out_dir --lat 28.04 --lon -82.70
        [--cylinder EAST NORTH RADIUS] [--box XMIN YMIN ZMIN XMAX YMAX ZMAX]
        [--z-range LO HI] [--bands 24] [--link-factor 4.5] [--max-nodes 200]
        [--labels labels.json --truth-rig rig.json]

It writes ``<out_dir>/source/splat.ply`` (the isolated tree, positions pre-snapped to the SPZ
grid), ``<out_dir>/source/rig.json``, and ``<out_dir>/splat/`` via ``splat_tiles.convert``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import splat_tiles

# One implementation of the checksum contract, not two. It lives in synthetic_tree.py because
# that is where it was first needed; it is the same digest either way, and a second copy here
# is exactly the drift `source/checksum_vectors.json` exists to prevent.
from synthetic_tree import POSITION_QUANTUM, checksum_positions, write_ply

#: Stiffness at the base and at the top of the tree, per band. These mirror the values
#: `syntheticTreeRig()` uses in packages/world/src/rig.ts. They are a plausible gradient —
#: thicker and lower bends less — and are *not* calibrated against any real tree.
BAND_STIFFNESS: dict[str, tuple[float, float]] = {
    "trunk": (8.0, 6.0),
    "branch": (3.2, 2.0),
    "leaf": (1.0, 1.0),
}

#: A child counts as a real fork (rather than noise off the side of the trunk) only when its
#: subtree holds at least this share of its parent's subtree.
FORK_SHARE = 0.15


# --------------------------------------------------------------------------------- isolation


def region_mask(
    xyz: np.ndarray,
    *,
    cylinder: tuple[float, float, float] | None = None,
    box: tuple[float, float, float, float, float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Which splats lie inside the bounding region, in the PLY's local ENU frame, metres.

    All three constraints are optional and intersect. No region at all means the whole file,
    which is the right default for a capture that is already one tree and nothing else.
    """
    keep = np.ones(xyz.shape[0], dtype=bool)
    if cylinder is not None:
        east, north, radius = cylinder
        offset = xyz[:, :2] - np.asarray([east, north], dtype=np.float64)
        keep &= np.einsum("ij,ij->i", offset, offset) <= radius * radius
    if box is not None:
        lo = np.asarray(box[:3], dtype=np.float64)
        hi = np.asarray(box[3:], dtype=np.float64)
        keep &= np.all((xyz >= lo) & (xyz <= hi), axis=1)
    if z_range is not None:
        keep &= (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])
    return keep


def neighbour_spacing(xyz: np.ndarray) -> float:
    """Median nearest-neighbour distance, metres: this cloud's characteristic point spacing.

    Every radius in this module is expressed as a multiple of it. A 2,000-splat tree and a
    50,000-splat tree of the same size are a factor of 2.5 apart in spacing, and a link radius
    that separates limbs on one welds the whole canopy into a single blob on the other — so
    fixed metre thresholds make the tool work only at the density it was tuned at.
    """
    if xyz.shape[0] < 2:
        return 0.0
    distance, _ = cKDTree(xyz).query(xyz, k=2)
    return float(np.median(distance[:, 1]))


def dense_mask(xyz: np.ndarray, k: int, factor: float) -> np.ndarray:
    """Which splats are not isolated: k-th neighbour no further than ``factor`` × the median.

    Photogrammetric splats come with haze — lone gaussians floating in the space around the
    subject. They are few but they are *far*, so they drag a centroid and stretch a band, and a
    band stretched by one floater slices the tree differently. The threshold is relative to the
    cloud's own k-th-neighbour distances, so it carries across densities without retuning.
    """
    count = xyz.shape[0]
    if count == 0:
        return np.zeros(0, dtype=bool)
    if k <= 0 or count <= k:
        return np.ones(count, dtype=bool)
    distance, _ = cKDTree(xyz).query(xyz, k=k + 1)
    kth = distance[:, -1]
    return kth <= factor * float(np.median(kth))


def isolate_tree(
    xyz: np.ndarray,
    opacity: np.ndarray,
    *,
    cylinder: tuple[float, float, float] | None = None,
    box: tuple[float, float, float, float, float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    opacity_min: float = 0.02,
    ground_percentile: float = 1.0,
    ground_clearance: float = 0.0,
    denoise_k: int = 8,
    denoise_factor: float = 3.0,
) -> np.ndarray:
    """The boolean keep-mask for one tree: region, opacity, ground and density, in that order.

    Ground is the ``ground_percentile`` of height *within the region* rather than an absolute
    number, because a capture's local frame has no guarantee that z=0 is the ground under this
    particular tree. ``ground_clearance`` then lifts the cut, for a capture where the mown grass
    around the trunk reconstructed as a skirt.
    """
    keep = region_mask(xyz, cylinder=cylinder, box=box, z_range=z_range)
    keep &= np.isfinite(xyz).all(axis=1)
    keep &= opacity >= opacity_min
    if not keep.any():
        return keep
    ground = float(np.percentile(xyz[keep, 2], ground_percentile))
    keep &= xyz[:, 2] >= ground + ground_clearance
    if not keep.any():
        return keep
    index = np.flatnonzero(keep)
    dense = dense_mask(xyz[index], denoise_k, denoise_factor)
    keep[index[~dense]] = False
    return keep


# ------------------------------------------------------------------------------- clustering


def cluster_points(xyz: np.ndarray, link_radius: float) -> np.ndarray:
    """Single-linkage connected components at ``link_radius``: one label per point.

    ``sparse_distance_matrix`` + ``connected_components`` rather than ``query_pairs`` because
    the pair list of a dense band is quadratic in its point count, and a canopy band is dense.
    """
    count = xyz.shape[0]
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    if count == 1:
        return np.zeros(1, dtype=np.int64)
    tree = cKDTree(xyz)
    graph = tree.sparse_distance_matrix(tree, link_radius, output_type="coo_matrix")
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64)


def band_of(z: np.ndarray, ground: float, band_height: float, bands: int) -> np.ndarray:
    """Band index per point, clamped into ``[0, bands)``."""
    raw = np.floor((z - ground) / band_height).astype(np.int64)
    return np.clip(raw, 0, bands - 1)


def _rms_radius(points: np.ndarray, centre: np.ndarray) -> float:
    """Horizontal RMS spread about the centroid: the radius of a cylinder shell of these points."""
    offset = points[:, :2] - centre[:2]
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", offset, offset))))


def _build_bands(
    xyz: np.ndarray, bands: int, link_radius: float, min_cluster_points: int
) -> tuple[list[dict], np.ndarray]:
    """One node per connected component per band, plus the owning node index of every point."""
    z = xyz[:, 2]
    ground = float(z.min())
    height = float(z.max()) - ground
    band_height = max(height / max(bands, 1), 1e-6)
    band_index = band_of(z, ground, band_height, bands)

    nodes: list[dict] = []
    owner = np.full(xyz.shape[0], -1, dtype=np.int64)
    for b in range(bands):
        member = np.flatnonzero(band_index == b)
        if member.size < min_cluster_points:
            continue
        labels = cluster_points(xyz[member], link_radius)
        for label in np.unique(labels):
            part = member[labels == label]
            if part.size < min_cluster_points:
                continue
            centre = xyz[part].mean(axis=0)
            owner[part] = len(nodes)
            nodes.append(
                {
                    "position": centre,
                    "radius": _rms_radius(xyz[part], centre),
                    "band_index": int(b),
                    "members": part,
                    "parent": -1,
                }
            )
    return nodes, owner


def _attach_parents(nodes: list[dict], xyz: np.ndarray) -> int:
    """Parent every node to the nearest node in a strictly lower band. Returns the root index.

    Nearest *point* rather than nearest centroid: a branch that leans a metre out from the trunk
    has a centroid far from the trunk's but points that still touch it, and the centroid rule
    would hang that branch off a neighbouring branch instead.
    """
    if not nodes:
        raise SystemExit("no skeleton nodes: the region is empty or every cluster was too small")
    by_band: dict[int, list[int]] = {}
    for i, node in enumerate(nodes):
        by_band.setdefault(node["band_index"], []).append(i)
    lowest = min(by_band)
    # The root is the biggest thing at the bottom of the tree: the trunk, by construction.
    root = max(by_band[lowest], key=lambda i: nodes[i]["members"].size)
    nodes[root]["parent"] = -1

    lower_points: list[np.ndarray] = []
    lower_owner: list[np.ndarray] = []
    for band in sorted(by_band):
        if lower_points:
            stacked = np.concatenate(lower_points, axis=0)
            stacked_owner = np.concatenate(lower_owner, axis=0)
            tree = cKDTree(stacked)
        else:
            tree = None
        for i in by_band[band]:
            if i == root:
                continue
            if tree is None:
                # Same band as the root, nothing below: hang it on the trunk base.
                nodes[i]["parent"] = root
                continue
            # The lower-band point closest to any of this node's points decides the parent.
            distance, nearest = tree.query(xyz[nodes[i]["members"]], k=1)
            nodes[i]["parent"] = int(stacked_owner[nearest[int(np.argmin(distance))]])
        for i in by_band[band]:
            lower_points.append(xyz[nodes[i]["members"]])
            lower_owner.append(np.full(nodes[i]["members"].size, i, dtype=np.int64))
    return root


def _children_of(nodes: list[dict], root: int, alive: list[bool] | None = None) -> list[list[int]]:
    """Child lists, skipping nodes pruning has already merged away.

    ``alive`` is not optional in spirit: without it a merged node still counts as its parent's
    child, so the parent never looks like a twig tip and pruning stalls short of ``max_nodes``.
    """
    children: list[list[int]] = [[] for _ in nodes]
    for i, node in enumerate(nodes):
        if alive is not None and not alive[i]:
            continue
        if i != root and node["parent"] >= 0:
            children[node["parent"]].append(i)
    return children


def _subtree_counts(nodes: list[dict], children: list[list[int]], root: int) -> np.ndarray:
    """Points in each node's subtree, computed bottom-up from a post-order walk."""
    counts = np.asarray([node["members"].size for node in nodes], dtype=np.int64)
    order: list[int] = []
    stack = [root]
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(children[i])
    for i in reversed(order):
        for child in children[i]:
            counts[i] += counts[child]
    return counts


def _prune_to(nodes: list[dict], root: int, max_nodes: int) -> tuple[list[dict], int]:
    """Merge the smallest subtrees into their parents until at most ``max_nodes`` remain.

    Smallest *subtree*, not smallest node: removing a fat node with thin children would orphan
    real structure, while removing the thinnest twig loses the least shape. Members move to the
    parent, so no splat is ever left without a node.
    """
    alive = [True] * len(nodes)
    while sum(alive) > max_nodes:
        children = _children_of(nodes, root, alive)
        counts = _subtree_counts(nodes, children, root)
        candidates = [i for i in range(len(nodes)) if alive[i] and i != root and not children[i]]
        if not candidates:
            break
        # Ties broken by index so the result does not depend on sort stability.
        victim = min(candidates, key=lambda i: (int(counts[i]), i))
        parent = nodes[victim]["parent"]
        nodes[parent]["members"] = np.concatenate(
            [nodes[parent]["members"], nodes[victim]["members"]]
        )
        alive[victim] = False
        for node in nodes:
            if node["parent"] == victim:
                node["parent"] = parent
    kept = [i for i in range(len(nodes)) if alive[i]]
    remap = {old: new for new, old in enumerate(kept)}
    out = []
    for old in kept:
        node = dict(nodes[old])
        node["parent"] = -1 if old == root else remap[node["parent"]]
        out.append(node)
    return out, remap[root]


def _topological(nodes: list[dict], root: int) -> list[dict]:
    """Breadth-first from the root, so every node's parent has a strictly smaller index."""
    children = _children_of(nodes, root)
    order: list[int] = []
    queue = [root]
    while queue:
        i = queue.pop(0)
        order.append(i)
        queue.extend(sorted(children[i]))
    remap = {old: new for new, old in enumerate(order)}
    out = []
    for old in order:
        node = dict(nodes[old])
        node["parent"] = -1 if old == root else remap[node["parent"]]
        out.append(node)
    return out


def _label_bands(nodes: list[dict]) -> list[str]:
    """trunk / branch / leaf, from the shape of the tree rather than from height alone.

    The trunk is the main chain — at each step, the child with the biggest subtree — walked from
    the root and stopped at the first *real* fork, a node with two children each holding at least
    ``FORK_SHARE`` of its subtree. That is where the crown starts, and it is a property of the
    tree rather than of an arbitrary height cut, so it survives a leaning or a short-boled tree.
    """
    children = _children_of(nodes, 0)
    counts = _subtree_counts(nodes, children, 0)
    labels = ["branch"] * len(nodes)
    cursor = 0
    while True:
        labels[cursor] = "trunk"
        kids = children[cursor]
        if not kids:
            break
        ranked = sorted(kids, key=lambda i: (-int(counts[i]), i))
        if len(ranked) > 1 and counts[ranked[1]] >= FORK_SHARE * counts[cursor]:
            break  # A real fork: the trunk ends at this node, the crown starts above it.
        cursor = ranked[0]
    for i, kids in enumerate(children):
        if not kids and labels[i] != "trunk":
            labels[i] = "leaf"
    return labels


# ------------------------------------------------------------------------------ rig assembly


def extract_skeleton(
    xyz: np.ndarray,
    *,
    bands: int = 24,
    link_factor: float = 4.5,
    link_radius: float | None = None,
    min_cluster_points: int = 8,
    max_nodes: int = 200,
) -> list[dict]:
    """The skeleton as an ordered node list: position, radius, band, parent, member indices.

    ``link_radius`` defaults to ``link_factor`` × this cloud's own point spacing. Pass an
    explicit radius only when a capture's spacing is misleading — a scan with one dense side,
    say — and say why in the rig's ``sourceNote``.
    """
    if xyz.shape[0] == 0:
        raise SystemExit("no splats to extract a skeleton from")
    if link_radius is None:
        link_radius = max(link_factor * neighbour_spacing(xyz), 1e-4)
    raw, _owner = _build_bands(xyz, bands, link_radius, min_cluster_points)
    root = _attach_parents(raw, xyz)
    pruned, root = _prune_to(raw, root, max_nodes)
    ordered = _topological(pruned, root)
    for node, band in zip(ordered, _label_bands(ordered), strict=True):
        node["band"] = band
    return ordered


def _stiffness(band: str, height_fraction: float) -> float:
    base, top = BAND_STIFFNESS[band]
    return float(base + (top - base) * min(max(height_fraction, 0.0), 1.0))


def build_rig(nodes: list[dict], canonical_checksum: str, source_note: str) -> dict:
    """The extracted skeleton as MotionRig JSON — the schema packages/world/src/rig.ts parses.

    Node positions stay in the capture's own local ENU frame, un-recentred: the runtime assigns
    splats to nodes by distance in that same frame, so moving the rig would break the mapping it
    exists to describe.
    """
    if not nodes:
        raise SystemExit("cannot build a rig with no nodes")
    root_z = float(nodes[0]["position"][2])
    top_z = max(float(node["position"][2]) for node in nodes)
    span = max(top_z - root_z, 1e-6)
    out = []
    for i, node in enumerate(nodes):
        position = [float(value) for value in node["position"]]
        out.append(
            {
                "id": f"{node['band']}-{i}",
                "parent": int(node["parent"]),
                "position": position,
                "radius": round(float(node["radius"]), 4),
                "stiffness": round(_stiffness(node["band"], (position[2] - root_z) / span), 4),
                "band": node["band"],
            }
        )
    return {
        # Key order matches serializeRig() in packages/world/src/rig.ts.
        "units": "meters",
        "canonicalChecksum": canonical_checksum,
        "sourceNote": source_note,
        "nodes": out,
    }


def rig_issues(rig: dict) -> list[str]:
    """The structural checks validateRig() makes in packages/world/src/rig.ts, in Python.

    Duplicated deliberately: a rig that only fails in the browser fails after it has been
    committed, seeded and served. This makes the tool refuse to write one.
    """
    issues: list[str] = []
    nodes = rig["nodes"]
    if not nodes:
        return ["rig has no nodes"]
    seen: set[str] = set()
    for i, node in enumerate(nodes):
        where = f"node {i} ({node['id']})"
        if not node["id"]:
            issues.append(f"{where}: empty id")
        if node["id"] in seen:
            issues.append(f"{where}: duplicate id")
        seen.add(node["id"])
        if i == 0:
            if node["parent"] != -1:
                issues.append(f"{where}: the first node must be the root (parent -1)")
        elif not 0 <= node["parent"] < i:
            issues.append(f"{where}: parent {node['parent']} must be an earlier index")
        if len(node["position"]) != 3 or not all(np.isfinite(node["position"])):
            issues.append(f"{where}: position must be three finite numbers")
        if not np.isfinite(node["radius"]) or node["radius"] < 0:
            issues.append(f"{where}: radius must be >= 0")
        if not np.isfinite(node["stiffness"]) or node["stiffness"] <= 0:
            issues.append(f"{where}: stiffness must be > 0")
        if node["band"] not in ("trunk", "branch", "leaf"):
            issues.append(f"{where}: band must be trunk, branch or leaf")
    return issues


# ---------------------------------------------------------------------------------- scoring


def _contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Counts of every (label in a, label in b) pair, as a dense table."""
    rows = int(a.max()) + 1 if a.size else 0
    cols = int(b.max()) + 1 if b.size else 0
    table = np.zeros((rows, cols), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    return table


def _adjusted_rand(table: np.ndarray) -> float:
    """Adjusted Rand index from a contingency table: agreement corrected for chance.

    The one metric here that does not care how many nodes each side has. It asks only whether
    two splats that truth puts on the same limb end up on the same recovered limb — which is
    exactly the property the deformer needs, since a node's identity is never shown to anyone.
    """

    def pairs(counts: np.ndarray) -> float:
        return float(np.sum(counts * (counts - 1) / 2))

    total = float(table.sum())
    if total < 2:
        return 1.0
    both = pairs(table)
    rows = pairs(table.sum(axis=1))
    cols = pairs(table.sum(axis=0))
    expected = rows * cols / (total * (total - 1) / 2)
    maximum = (rows + cols) / 2
    if maximum == expected:
        return 1.0
    return (both - expected) / (maximum - expected)


def score_against_truth(
    positions: np.ndarray, true_labels: np.ndarray, truth_rig: dict, rig: dict
) -> dict:
    """How well an extracted rig recovers a known skeleton. Every number is a fraction or metres.

    The extracted rig has its own node count and its own ids, so there is no per-node identity to
    check against. What can be checked is the *partition* (do splats that belong together stay
    together), the *band* each splat ends up in (bands set stiffness, so this one is felt), and
    how close the recovered joints land to the real ones.
    """
    recovered = np.asarray([node["position"] for node in rig["nodes"]], dtype=np.float64)
    truth = np.asarray([node["position"] for node in truth_rig["nodes"]], dtype=np.float64)
    assigned = cKDTree(recovered).query(positions, k=1)[1].astype(np.int64)

    table = _contingency(true_labels.astype(np.int64), assigned)
    majority = table.argmax(axis=0)
    purity = float(np.mean(true_labels == majority[assigned]))

    true_bands = np.asarray([node["band"] for node in truth_rig["nodes"]])[true_labels]
    recovered_bands = np.asarray([node["band"] for node in rig["nodes"]])[assigned]

    offsets = cKDTree(recovered).query(truth, k=1)[0]
    return {
        "splats": int(positions.shape[0]),
        "trueNodes": int(truth.shape[0]),
        "recoveredNodes": int(recovered.shape[0]),
        "adjustedRandIndex": round(_adjusted_rand(table), 4),
        "purity": round(purity, 4),
        "bandAccuracy": round(float(np.mean(true_bands == recovered_bands)), 4),
        "nodeOffsetMeanM": round(float(offsets.mean()), 4),
        "nodeOffsetP90M": round(float(np.percentile(offsets, 90)), 4),
        "trueNodesWithin0_5m": round(float(np.mean(offsets <= 0.5)), 4),
    }


# --------------------------------------------------------------------------------- pipeline


def _snap(xyz: np.ndarray) -> np.ndarray:
    """The SPZ fixed-point grid, applied before anything is written.

    Same reasoning as synthetic_tree.py: SPZ quantises positions to 1/4096 m, so snapping here
    makes the PLY floats and the splats CesiumJS decodes bit-identical — which is the only way
    ``canonicalChecksum`` can mean the same thing on both sides of the tiler. Negative zero is
    normalised because it has different bytes from positive zero and SPZ's integer round trip
    turns it into positive zero.
    """
    snapped = np.round(np.asarray(xyz, dtype=np.float64) / POSITION_QUANTUM) * POSITION_QUANTUM
    snapped = snapped.astype(np.float32)
    snapped[snapped == 0.0] = 0.0
    return snapped


def extract(
    ply: Path,
    out_dir: Path,
    *,
    lat: float,
    lon: float,
    height: float = 0.0,
    cylinder: tuple[float, float, float] | None = None,
    box: tuple[float, float, float, float, float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    opacity_min: float = 0.02,
    ground_percentile: float = 1.0,
    ground_clearance: float = 0.0,
    denoise_k: int = 8,
    denoise_factor: float = 3.0,
    bands: int = 24,
    link_factor: float = 4.5,
    link_radius: float | None = None,
    min_cluster_points: int = 8,
    max_nodes: int = 200,
    geometric_error: float = 0.5,
    source_note: str | None = None,
    tile: bool = True,
    truth: tuple[dict, dict] | None = None,
) -> dict:
    """Isolate one tree from a capture, write its splats, rig and tileset, and report on it.

    ``truth`` is ``(labels.json, rig.json)`` from the synthetic tree, and turns the run into a
    scored one. Scoring happens here rather than from the outside because isolation drops splats
    and the ground-truth labels are in PLY order — only this function still holds the keep mask
    that relates the two, and reconstructing it afterwards is how labels silently shift.
    """
    data = splat_tiles.read_ply(ply)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    opacity = splat_tiles.sigmoid(data["opacity"])
    keep = isolate_tree(
        xyz,
        opacity,
        cylinder=cylinder,
        box=box,
        z_range=z_range,
        opacity_min=opacity_min,
        ground_percentile=ground_percentile,
        ground_clearance=ground_clearance,
        denoise_k=denoise_k,
        denoise_factor=denoise_factor,
    )
    if not keep.any():
        raise SystemExit("the bounding region contains no splats after isolation")

    positions = _snap(xyz[keep])
    # Resolved here rather than inside extract_skeleton so the report can state it: on a real
    # capture the link radius is the number to reach for first when a rig comes out wrong.
    spacing = neighbour_spacing(positions.astype(np.float64))
    if link_radius is None:
        link_radius = max(link_factor * spacing, 1e-4)
    nodes = extract_skeleton(
        positions.astype(np.float64),
        bands=bands,
        link_radius=link_radius,
        min_cluster_points=min_cluster_points,
        max_nodes=max_nodes,
    )
    canonical = np.ascontiguousarray(positions, dtype="<f4")
    note = source_note or (
        f"geometric skeleton extraction from {ply.name}: {len(nodes)} nodes over "
        f"{int(keep.sum()):,} splats (tools/captures/skeleton.py)"
    )
    rig = build_rig(nodes, checksum_positions(canonical), note)
    issues = rig_issues(rig)
    if issues:
        raise SystemExit("extracted rig is invalid:\n  " + "\n  ".join(issues))

    source_dir = out_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    ply_out = source_dir / "splat.ply"
    write_ply(
        ply_out,
        {
            "position": positions,
            # write_ply re-encodes RGB through _rgb_to_sh0; pass the original coefficients back
            # through it so the isolated tree keeps the colours the trainer produced.
            "rgb": np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)[keep]
            * splat_tiles.SH_C0
            + 0.5,
            "opacity_logit": data["opacity"][keep],
            "log_scale": np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)[
                keep
            ],
            "quat_wxyz": np.stack(
                [data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1
            )[keep],
        },
    )
    (source_dir / "rig.json").write_text(
        json.dumps(rig, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (source_dir / "positions.f32").write_bytes(canonical.tobytes())

    report: dict = {
        "splats": int(keep.sum()),
        "dropped_by_isolation": int((~keep).sum()),
        "nodes": len(rig["nodes"]),
        "bands": {
            band: sum(1 for node in rig["nodes"] if node["band"] == band)
            for band in ("trunk", "branch", "leaf")
        },
        "height_m": round(float(positions[:, 2].max() - positions[:, 2].min()), 3),
        "spacing_m": round(spacing, 4),
        "link_radius_m": round(float(link_radius), 4),
        "canonicalChecksum": rig["canonicalChecksum"],
        "out_dir": str(out_dir),
    }
    if tile:
        stats = splat_tiles.convert(
            ply_out,
            out_dir / "splat",
            lat,
            lon,
            height,
            max_gaussians=int(positions.shape[0]) + 1,
            opacity_min=opacity_min,
            geometric_error=geometric_error,
        )
        if stats["dropped"] != 0:
            # The rig's checksum is over the splats written above. If the tiler drops any of
            # them the runtime decodes a different set, the checksum fails and the deformer
            # refuses — better to fail here, where the numbers can still be adjusted.
            raise SystemExit(
                f"the tiler dropped {stats['dropped']} of the isolated splats, so the rig's "
                "canonicalChecksum would not match what the runtime decodes — widen "
                "--opacity-min or tighten the region"
            )
        report["tileset"] = str(out_dir / "splat" / "tileset.json")
        report["extent_m"] = stats["extent_m"]
    if truth is not None:
        labels, truth_rig = truth
        all_labels = np.asarray(labels["nodes"], dtype=np.int64)
        if all_labels.size != xyz.shape[0]:
            raise SystemExit(
                f"{all_labels.size} ground-truth labels for {xyz.shape[0]} splats in the PLY — "
                "the labels do not belong to this capture"
            )
        report["score"] = score_against_truth(
            positions.astype(np.float64), all_labels[keep], truth_rig, rig
        )
        report["score"]["isolationRetained"] = round(float(keep.mean()), 4)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--height", type=float, default=0.0)
    parser.add_argument(
        "--cylinder",
        type=float,
        nargs=3,
        metavar=("EAST", "NORTH", "RADIUS"),
        help="keep splats within RADIUS metres of (EAST, NORTH) in the PLY's local frame",
    )
    parser.add_argument(
        "--box", type=float, nargs=6, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX")
    )
    parser.add_argument("--z-range", type=float, nargs=2, metavar=("LO", "HI"))
    parser.add_argument("--opacity-min", type=float, default=0.02)
    parser.add_argument("--ground-percentile", type=float, default=1.0)
    parser.add_argument("--ground-clearance", type=float, default=0.0)
    parser.add_argument("--denoise-k", type=int, default=8)
    parser.add_argument("--denoise-factor", type=float, default=3.0)
    parser.add_argument("--bands", type=int, default=24)
    parser.add_argument(
        "--link-factor",
        type=float,
        default=4.5,
        help="clustering link radius as a multiple of the cloud's own point spacing",
    )
    parser.add_argument(
        "--link-radius", type=float, default=None, help="override --link-factor, in metres"
    )
    parser.add_argument("--min-cluster-points", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=200)
    parser.add_argument("--geometric-error", type=float, default=0.5)
    parser.add_argument("--source-note", type=str, default=None)
    parser.add_argument("--no-tile", action="store_true", help="skip splat_tiles.convert")
    parser.add_argument("--labels", type=Path, help="ground-truth labels.json, to score against")
    parser.add_argument("--truth-rig", type=Path, help="ground-truth rig.json, to score against")
    args = parser.parse_args()

    truth = None
    if bool(args.labels) != bool(args.truth_rig):
        parser.error("--labels and --truth-rig are only meaningful together")
    if args.labels and args.truth_rig:
        truth = (
            json.loads(args.labels.read_text(encoding="utf-8")),
            json.loads(args.truth_rig.read_text(encoding="utf-8")),
        )

    report = extract(
        args.ply,
        args.out_dir,
        lat=args.lat,
        lon=args.lon,
        height=args.height,
        cylinder=tuple(args.cylinder) if args.cylinder else None,
        box=tuple(args.box) if args.box else None,
        z_range=tuple(args.z_range) if args.z_range else None,
        opacity_min=args.opacity_min,
        ground_percentile=args.ground_percentile,
        ground_clearance=args.ground_clearance,
        denoise_k=args.denoise_k,
        denoise_factor=args.denoise_factor,
        bands=args.bands,
        link_factor=args.link_factor,
        link_radius=args.link_radius,
        min_cluster_points=args.min_cluster_points,
        max_nodes=args.max_nodes,
        geometric_error=args.geometric_error,
        source_note=args.source_note,
        tile=not args.no_tile,
        truth=truth,
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
