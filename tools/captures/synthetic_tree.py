"""A procedural tree with known ground truth: Gaussian splats, a motion rig, and a tileset.

The Living Survey runtime deforms a scanned tree by assigning each splat to the nearest node
of a skeleton rig. On a real capture nobody knows which splat *should* belong to which branch,
so there is nothing to score the assignment against. This module builds a tree where the answer
is known by construction: every splat is generated from one rig node and that node index is
written out beside the splats.

It emits, into ``<out_dir>``::

    source/splat.ply             binary little-endian 3DGS PLY, the layout splat_tiles reads
    source/rig.json              MotionRig JSON, the schema in packages/world/src/rig.ts
    source/labels.json           ground-truth node index per splat, in PLY order
    source/positions.f32         the canonical positions the checksum is over, raw float32
    source/checksum_vectors.json small cases pinning checksum_positions to its TS twin
    splat/tileset.json           3D Tiles, via splat_tiles.convert (loadable with no API)
    splat/splat.glb

The frame is local ENU — +X east, +Y north, +Z up — in metres, which is what ``@twin/world``'s
``deform`` assumes and what ``splat_tiles`` places on the globe with its root transform.

Output is byte-reproducible: the RNG is seeded, positions are snapped to the SPZ fixed-point
grid before anything is written, and ``splat_tiles.pack_spz`` already pins ``gzip mtime=0``.
Two runs with the same arguments produce identical files.

The rig mirrors ``syntheticTreeRig()`` in packages/world/src/rig.ts node for node, so the two
sizes below share one 33-node skeleton and differ only in splat density.

Usage:
    python synthetic_tree.py ../../data/tiles/synthetic-tree --splats 2000
        [--seed 7] [--lat 28.0389] [--lon -82.6966] [--height 0] [--height-m 6]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import splat_tiles
from splat_tiles import SH_C0, SPZ_FRACTIONAL_BITS

# Golden angle, radians: spreads branch azimuths without any two lining up. Mirrors
# GOLDEN_ANGLE in packages/world/src/rig.ts.
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))

#: Fraction of the splat budget spent on bark (trunk and branch cylinders); the rest is foliage.
WOOD_FRACTION = 0.32
#: Of the foliage budget, the share that goes to tip clusters rather than inner canopy.
TIP_FRACTION = 0.72

BARK_RGB = (0.34, 0.235, 0.155)
FOLIAGE_RGB = (0.185, 0.415, 0.145)

#: Splat positions are snapped to this grid so the PLY floats survive SPZ encoding unchanged.
POSITION_QUANTUM = 1.0 / (1 << SPZ_FRACTIONAL_BITS)

#: Properties of the emitted PLY, in order. splat_tiles.convert reads exactly these.
PLY_PROPERTIES = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


# --------------------------------------------------------------------------------------- rig


def synthetic_tree_rig(
    height_m: float = 6.0,
    trunk_segments: int = 6,
    whorls: int = 3,
    branches_per_whorl: int = 3,
    canonical_checksum: str = "fnv1a32:0:00000000",
) -> dict:
    """The same 33-node skeleton ``syntheticTreeRig()`` builds in packages/world/src/rig.ts.

    Kept deliberately in lockstep with the TypeScript: a test in ``@twin/world`` parses the
    emitted rig.json and compares it node for node with its own generator, so a change to
    either side that is not mirrored fails CI rather than quietly producing two different trees.
    """
    trunk_segments = max(2, int(trunk_segments))
    whorls = max(1, int(whorls))
    branches_per_whorl = max(1, int(branches_per_whorl))
    nodes: list[dict] = []
    trunk_indices: list[int] = []

    for i in range(trunk_segments):
        f = i / (trunk_segments - 1)
        trunk_indices.append(len(nodes))
        nodes.append(
            {
                "id": f"trunk-{i}",
                "parent": -1 if i == 0 else trunk_indices[i - 1],
                "position": [0.0, 0.0, f * height_m],
                "radius": 0.35 - 0.23 * f,
                "stiffness": 8 - 2 * f,
                "band": "trunk",
            }
        )

    branch_ordinal = 0
    for w in range(whorls):
        trunk_index = trunk_indices[trunk_segments - 1 - w]
        base = nodes[trunk_index]["position"]
        for _ in range(branches_per_whorl):
            azimuth = GOLDEN_ANGLE * branch_ordinal
            east = math.cos(azimuth)
            north = math.sin(azimuth)
            primary_index = len(nodes)
            nodes.append(
                {
                    "id": f"branch-{branch_ordinal}-0",
                    "parent": trunk_index,
                    "position": [base[0] + east * 1.1, base[1] + north * 1.1, base[2] + 0.35],
                    "radius": 0.12,
                    "stiffness": 3.2,
                    "band": "branch",
                }
            )
            secondary_index = len(nodes)
            primary = nodes[primary_index]["position"]
            nodes.append(
                {
                    "id": f"branch-{branch_ordinal}-1",
                    "parent": primary_index,
                    "position": [
                        primary[0] + east * 0.9,
                        primary[1] + north * 0.9,
                        primary[2] + 0.3,
                    ],
                    "radius": 0.07,
                    "stiffness": 2,
                    "band": "branch",
                }
            )
            secondary = nodes[secondary_index]["position"]
            nodes.append(
                {
                    "id": f"leaf-{branch_ordinal}",
                    "parent": secondary_index,
                    "position": [
                        secondary[0] + east * 0.6,
                        secondary[1] + north * 0.6,
                        secondary[2] + 0.25,
                    ],
                    "radius": 0.04,
                    "stiffness": 1,
                    "band": "leaf",
                }
            )
            branch_ordinal += 1

    return {
        # Key order matches serializeRig() in packages/world/src/rig.ts.
        "units": "meters",
        "canonicalChecksum": canonical_checksum,
        "sourceNote": (
            f"synthetic tree, {height_m} m, {len(nodes)} nodes (tools/captures/synthetic_tree.py)"
        ),
        "nodes": nodes,
    }


# ---------------------------------------------------------------------------------- checksum


def checksum_positions(positions: np.ndarray) -> str:
    """FNV-1a over the raw float32 bytes, plus the splat count.

    A direct transcription of ``checksumPositions`` in packages/world/src/rig.ts. The two are
    pinned to each other by ``source/checksum_vectors.json``, which both languages' tests read:
    this contract is what lets the runtime refuse to deform when it is looking at splats the
    rig was not built from, so it must not be able to drift.
    """
    flat = np.ascontiguousarray(positions, dtype="<f4").reshape(-1)
    digest = 0x811C9DC5
    for byte in flat.tobytes():
        digest = ((digest ^ byte) * 0x01000193) & 0xFFFFFFFF
    return f"fnv1a32:{flat.size // 3}:{digest:08x}"


def _checksum_vectors() -> list[dict]:
    """Edge cases for the cross-language checksum test, as hex-encoded little-endian bytes."""
    cases: dict[str, list[float]] = {
        "empty": [],
        "origin": [0.0, 0.0, 0.0],
        "unit": [1.0, 2.0, 3.0],
        # Negative zero has different bytes from positive zero and must change the digest.
        "negative-zero": [-0.0, 0.0, -0.0],
        "negatives": [-1.5, -0.25, -1024.0],
        "two-splats": [0.001953125, -0.5, 6.0, 1.1000000238418579, -2.25, 0.125],
        "fractional-grid": [k * POSITION_QUANTUM for k in (1, -1, 4095, 7, -8191, 3)],
    }
    out = []
    for name, values in cases.items():
        array = np.asarray(values, dtype="<f4")
        out.append(
            {
                "name": name,
                "positionsHex": array.tobytes().hex(),
                "checksum": checksum_positions(array),
            }
        )
    return out


# ------------------------------------------------------------------------------- splat maths


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


def _orthonormal_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A right-handed frame whose first vector is ``axis``. Deterministic given the axis."""
    e0 = _unit(axis)
    seed = np.array([0.0, 0.0, 1.0]) if abs(e0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(np.cross(seed, e0))
    e2 = np.cross(e0, e1)
    return e0, e1, e2


def _matrix_to_quat_wxyz(columns: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Rotation matrix (given as its three columns) to a w-first quaternion.

    Column ``i`` is the axis that scale component ``i`` stretches along, which is the
    convention the covariance ``R diag(s^2) R^T`` in every 3DGS renderer uses.
    """
    m = np.stack(columns, axis=1)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    quat = np.asarray(q, dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _random_quats_wxyz(rng: np.random.Generator, count: int) -> np.ndarray:
    """Uniformly distributed orientations: a normalised Gaussian 4-vector is exactly that."""
    raw = rng.standard_normal((count, 4))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def _rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
    """Linear colour in [0, 1] to the degree-0 spherical harmonic coefficient 3DGS stores."""
    return (rgb - 0.5) / SH_C0


# ------------------------------------------------------------------------------ tree surface


class _Splats:
    """Accumulates the per-splat arrays, one append per generated group."""

    def __init__(self) -> None:
        self.position: list[np.ndarray] = []
        self.rgb: list[np.ndarray] = []
        self.opacity_logit: list[np.ndarray] = []
        self.log_scale: list[np.ndarray] = []
        self.quat_wxyz: list[np.ndarray] = []
        self.node: list[np.ndarray] = []

    def add(
        self,
        position: np.ndarray,
        rgb: np.ndarray,
        opacity_logit: np.ndarray,
        log_scale: np.ndarray,
        quat_wxyz: np.ndarray,
        node: np.ndarray,
    ) -> None:
        self.position.append(position)
        self.rgb.append(rgb)
        self.opacity_logit.append(opacity_logit)
        self.log_scale.append(log_scale)
        self.quat_wxyz.append(quat_wxyz)
        self.node.append(node)

    def stacked(self) -> dict[str, np.ndarray]:
        return {
            "position": np.concatenate(self.position, axis=0),
            "rgb": np.concatenate(self.rgb, axis=0),
            "opacity_logit": np.concatenate(self.opacity_logit, axis=0),
            "log_scale": np.concatenate(self.log_scale, axis=0),
            "quat_wxyz": np.concatenate(self.quat_wxyz, axis=0),
            "node": np.concatenate(self.node, axis=0),
        }


def _bark_splats(
    rng: np.random.Generator,
    out: _Splats,
    p0: np.ndarray,
    p1: np.ndarray,
    r0: float,
    r1: float,
    count: int,
    node0: int,
    node1: int,
) -> None:
    """A tapering cylinder of bark splats between two rig nodes, on and just under the surface.

    Ground truth is the nearer endpoint, which is also what nearest-node assignment recovers
    for a splat on the axis — the radial offset is what makes the two disagree at all.
    """
    if count <= 0:
        return
    axis, tangent, normal = _orthonormal_frame(p1 - p0)
    length = float(np.linalg.norm(p1 - p0))
    t = rng.random(count)
    radius = (r0 + (r1 - r0) * t) * (0.82 + 0.18 * rng.random(count))
    theta = rng.random(count) * (2.0 * math.pi)
    centre = p0[None, :] + np.outer(t, p1 - p0)
    offset = np.outer(radius * np.cos(theta), tangent) + np.outer(radius * np.sin(theta), normal)
    position = centre + offset

    # Bark splats lie flat on the trunk: long along the axis, thin along the surface normal.
    surface = offset / np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-9)
    along = np.broadcast_to(axis, (count, 3))
    across = np.cross(surface, along)
    quats = np.empty((count, 4))
    for i in range(count):
        quats[i] = _matrix_to_quat_wxyz((along[i], _unit(across[i]), surface[i]))

    span = max(length / max(count / 6.0, 1.0), 0.02)
    scale = np.stack(
        [
            np.full(count, span * 1.6),
            np.maximum(radius, 0.01) * 0.55,
            np.full(count, 0.010),
        ],
        axis=1,
    ) * (0.8 + 0.4 * rng.random((count, 3)))

    shade = 0.78 + 0.44 * rng.random((count, 1))
    rgb = np.clip(np.asarray(BARK_RGB)[None, :] * shade, 0.0, 1.0)

    node = np.where(t < 0.5, node0, node1).astype(np.int32)
    out.add(position, rgb, 1.9 + 0.6 * rng.random(count), np.log(scale), quats, node)


def _foliage_splats(
    rng: np.random.Generator,
    out: _Splats,
    centre: np.ndarray,
    direction: np.ndarray,
    extent: float,
    count: int,
    node: int,
    height_m: float,
) -> None:
    """One roughly ellipsoidal leaf cluster, flattened in Z the way real canopies are."""
    if count <= 0:
        return
    sigma = np.array([extent, extent, extent * 0.72])
    offset = np.clip(rng.standard_normal((count, 3)), -2.1, 2.1) * sigma
    position = centre[None, :] + offset + direction[None, :] * (extent * 0.28)

    # Sunlit at the top of the canopy, darker underneath — enough to read as a tree.
    lit = 0.72 + 0.5 * np.clip(position[:, 2:3] / max(height_m, 1e-6), 0.0, 1.4)
    tint = 0.85 + 0.3 * rng.random((count, 3))
    rgb = np.clip(np.asarray(FOLIAGE_RGB)[None, :] * lit * tint, 0.0, 1.0)

    leaf = extent * (0.16 + 0.12 * rng.random((count, 1)))
    scale = leaf * np.array([1.0, 0.85, 0.42])[None, :] * (0.75 + 0.5 * rng.random((count, 3)))
    out.add(
        position,
        rgb,
        1.4 + 0.8 * rng.random(count),
        np.log(np.maximum(scale, 1e-4)),
        _random_quats_wxyz(rng, count),
        np.full(count, node, dtype=np.int32),
    )


def _allocate(weights: np.ndarray, total: int) -> np.ndarray:
    """Split ``total`` across ``weights`` by largest remainder, so the parts sum exactly."""
    if total <= 0 or weights.sum() <= 0:
        return np.zeros(weights.shape, dtype=np.int64)
    exact = weights / weights.sum() * total
    counts = np.floor(exact).astype(np.int64)
    # Ties broken by index keeps this deterministic regardless of the sort implementation.
    order = np.lexsort((np.arange(weights.size), -(exact - counts)))
    counts[order[: total - int(counts.sum())]] += 1
    return counts


def generate_splats(rig: dict, splats: int, seed: int, height_m: float) -> dict[str, np.ndarray]:
    """Every splat of the tree, with the rig node index each one was generated from."""
    rng = np.random.default_rng(seed)
    nodes = rig["nodes"]
    positions = np.asarray([node["position"] for node in nodes], dtype=np.float64)
    radii = np.asarray([node["radius"] for node in nodes], dtype=np.float64)
    bands = [node["band"] for node in nodes]

    wood_edges = [
        (node["parent"], i) for i, node in enumerate(nodes) if node["parent"] >= 0 and i > 0
    ]
    leaves = [i for i, band in enumerate(bands) if band == "leaf"]
    inner = [i for i, node in enumerate(nodes) if node["band"] == "branch" and node["parent"] > 0]

    wood_total = round(splats * WOOD_FRACTION)
    foliage_total = splats - wood_total
    tip_total = round(foliage_total * TIP_FRACTION)
    inner_total = foliage_total - tip_total

    # Bark budget follows surface area, so the trunk is not out-sampled by nine thin twigs.
    areas = np.asarray(
        [
            float(np.linalg.norm(positions[c] - positions[p])) * (radii[p] + radii[c])
            for p, c in wood_edges
        ]
    )
    wood_counts = _allocate(areas, wood_total)
    tip_counts = _allocate(np.ones(len(leaves)), tip_total)
    inner_counts = _allocate(np.ones(len(inner)), inner_total)

    out = _Splats()
    for (parent, child), count in zip(wood_edges, wood_counts, strict=True):
        _bark_splats(
            rng,
            out,
            positions[parent],
            positions[child],
            float(radii[parent]),
            float(radii[child]),
            int(count),
            parent,
            child,
        )
    for leaf, count in zip(leaves, tip_counts, strict=True):
        parent = nodes[leaf]["parent"]
        _foliage_splats(
            rng,
            out,
            positions[leaf],
            _unit(positions[leaf] - positions[parent]),
            0.44,
            int(count),
            leaf,
            height_m,
        )
    for branch, count in zip(inner, inner_counts, strict=True):
        parent = nodes[branch]["parent"]
        _foliage_splats(
            rng,
            out,
            positions[branch],
            _unit(positions[branch] - positions[parent]),
            0.3,
            int(count),
            branch,
            height_m,
        )

    data = out.stacked()
    # Snap before anything downstream sees the positions: the SPZ encoder quantises to this
    # grid anyway, so snapping here makes the PLY floats and the decoded splats bit-identical
    # and the checksum meaningful on both sides of the tiler.
    snapped = (np.round(data["position"] / POSITION_QUANTUM) * POSITION_QUANTUM).astype(np.float32)
    # Negative zero compares equal to zero but has different bytes, and the checksum is over
    # bytes. SPZ's integer round trip turns -0.0 into +0.0, so a single -0.0 here would make
    # the decoded splats fail the rig's checksum while being numerically identical.
    snapped[snapped == 0.0] = 0.0
    data["position"] = snapped
    return data


# ----------------------------------------------------------------------------------- writing


def write_ply(path: Path, data: dict[str, np.ndarray]) -> None:
    """Binary little-endian 3DGS PLY with exactly the properties splat_tiles.read_ply wants."""
    count = int(data["position"].shape[0])
    dtype = np.dtype([(name, "<f4") for name in PLY_PROPERTIES])
    rows = np.zeros(count, dtype=dtype)
    rows["x"], rows["y"], rows["z"] = data["position"].T
    sh0 = _rgb_to_sh0(data["rgb"])
    rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"] = sh0.T.astype(np.float32)
    rows["opacity"] = data["opacity_logit"].astype(np.float32)
    rows["scale_0"], rows["scale_1"], rows["scale_2"] = data["log_scale"].T.astype(np.float32)
    quat = data["quat_wxyz"].astype(np.float32)
    rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"] = quat.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment generated by tools/captures/synthetic_tree.py\n"
        f"element vertex {count}\n"
        + "".join(f"property float {name}\n" for name in PLY_PROPERTIES)
        + "end_header\n"
    )
    path.write_bytes(header.encode("ascii") + rows.tobytes())


def generate(
    out_dir: Path,
    splats: int,
    seed: int,
    lat: float,
    lon: float,
    height: float,
    height_m: float,
    geometric_error: float,
) -> dict[str, float | int | str]:
    """Write every artefact for one synthetic tree and return a summary of what was written."""
    rig = synthetic_tree_rig(height_m=height_m)
    data = generate_splats(rig, splats, seed, height_m)
    positions = np.ascontiguousarray(data["position"], dtype="<f4")
    rig["canonicalChecksum"] = checksum_positions(positions)

    source_dir = out_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    ply_path = source_dir / "splat.ply"
    write_ply(ply_path, data)
    (source_dir / "rig.json").write_text(
        json.dumps(rig, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (source_dir / "labels.json").write_text(
        json.dumps(
            {
                "note": "ground-truth rig node index per splat, in PLY order",
                "nodeIds": [node["id"] for node in rig["nodes"]],
                "canonicalChecksum": rig["canonicalChecksum"],
                "nodes": [int(value) for value in data["node"]],
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    (source_dir / "positions.f32").write_bytes(positions.tobytes())
    (source_dir / "checksum_vectors.json").write_text(
        json.dumps(
            {
                "note": (
                    "checksumPositions (packages/world/src/rig.ts) must agree with "
                    "checksum_positions (tools/captures/synthetic_tree.py) on every case"
                ),
                "cases": _checksum_vectors(),
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )

    stats = splat_tiles.convert(
        ply_path,
        out_dir / "splat",
        lat,
        lon,
        height,
        max_gaussians=int(positions.shape[0]) + 1,
        opacity_min=0.02,
        geometric_error=geometric_error,
    )
    if stats["dropped"] != 0:
        # The ground-truth labels are in PLY order. splat_tiles preserves order but not
        # contiguity, so a dropped splat would silently shift every label after it. The
        # generator's job is to emit a tree the tiler keeps whole; if it did not, say so.
        raise SystemExit(
            f"the tiler dropped {stats['dropped']} splats, so the ground-truth labels no "
            "longer line up with the tiled splats — adjust opacity or the tree extent"
        )

    return {
        "splats": int(data["position"].shape[0]),
        "nodes": len(rig["nodes"]),
        "canonicalChecksum": rig["canonicalChecksum"],
        "extent_m": stats["extent_m"],
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--splats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    # Somewhere in the Sheffield Park capture, so the tree can be flown to beside real data.
    parser.add_argument("--lat", type=float, default=28.0389)
    parser.add_argument("--lon", type=float, default=-82.6966)
    parser.add_argument("--height", type=float, default=0.0)
    parser.add_argument("--height-m", type=float, default=6.0, help="trunk height, metres")
    parser.add_argument("--geometric-error", type=float, default=0.5)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.out_dir,
                args.splats,
                args.seed,
                args.lat,
                args.lon,
                args.height,
                args.height_m,
                args.geometric_error,
            )
        )
    )


if __name__ == "__main__":
    main()
