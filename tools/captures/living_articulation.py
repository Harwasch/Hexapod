"""Lab-only branch rotation spectra and immutable per-splat bindings (CPU, simulated)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from living_spectra import inverse, spectrum
from rig_tiles import glb_spz, tile_uris
from splat_tiles import build_glb, pack_spz, site_positions, tile_offsets, unpack_spz
from synthetic_tree import checksum_positions


def node_spectra(nodes: list[dict], grass: bool = False):
    """Ten seconds of local joint rotvecs; 120 Hz damped oscillators, five warmup periods.

    Different branch frequencies and phases prevent a single canopy-wide travelling field.
    All forcing bins are multiples of .1 Hz; no forcing is inferred from real footage.
    """
    n = len(nodes)
    parents = np.array([v["parent"] for v in nodes])
    if any(p >= i or p < -1 for i, p in enumerate(parents)):
        raise ValueError("Hierarchy must be parent-first and acyclic")
    rng = np.random.default_rng(251022)
    phase = rng.uniform(0, 2 * np.pi, (n, 3))
    frequency = rng.choice([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7], n)[:, None]
    trunk = np.array([v.get("band") == "trunk" for v in nodes])
    gain = np.where(trunk, 0.0015, 0.025 if not grass else 0.07)[:, None]
    omega = 2 * np.pi * (frequency + 0.15)
    q = np.zeros((n, 3))
    velocity = q.copy()
    samples = []
    for step in range(7200):
        t = step / 120
        target = (
            gain
            * (
                np.sin(2 * np.pi * frequency * t + phase)
                + 0.35 * np.sin(2 * np.pi * 0.3 * t + phase * 0.3)
                + 0.2 * np.sin(2 * np.pi * 2.1 * t + phase * 1.9)
            )
            * [0.45, 1, 0.2]
        )
        target[parents < 0] = 0
        velocity += (omega**2 * (target - q) - 2 * 0.28 * omega * velocity) / 120
        q += velocity / 120
        if step >= 6000 and step % 4 == 0:
            samples.append(q.copy())
    samples = np.array(samples)
    f, c = spectrum(samples, 30)
    error = float(np.sqrt(np.mean((inverse(f, c, np.arange(300) / 30) - samples) ** 2)))
    return f, c, error


def forward_kinematics(nodes: list[dict], local: np.ndarray):
    """Reference FK: each incoming branch rotates about its proximal (parent) joint."""
    rest = np.array([v["position"] for v in nodes], dtype=float)
    parents = np.array([v["parent"] for v in nodes])
    joints = rest[np.maximum(parents, 0)].copy()
    joints[parents < 0] = rest[parents < 0]
    rotations = Rotation.from_rotvec(local).as_matrix()
    posed = joints.copy()
    for i, p in enumerate(parents):
        if p >= 0:
            posed[i] = posed[p] + rotations[p] @ (joints[i] - joints[p])
            rotations[i] = rotations[p] @ rotations[i]
    shifts = posed - np.einsum("nij,nj->ni", rotations, joints)
    return rotations, shifts


def segment_binding(points: np.ndarray, nodes: list[dict]) -> np.ndarray:
    """One branch except within 6 cm of its joint; blend only with its parent branch.

    Never average unrelated nearby branches. Entries: node+1, parent+1, weight/65535, leaf+1.
    Node zero is a real pinned root; encoded zero means entirely static/unbound.
    """
    rest = np.array([v["position"] for v in nodes])
    parent = np.array([v["parent"] for v in nodes])
    distance = np.full(len(points), np.inf)
    owner = np.zeros(len(points), dtype=int)
    for i, p in enumerate(parent):
        if p < 0:
            continue
        start, direction = rest[p], rest[i] - rest[p]
        length2 = max(float(direction @ direction), 1e-12)
        t = np.clip((points - start) @ direction / length2, 0, 1)
        d = np.sum((points - start - t[:, None] * direction) ** 2, axis=1)
        better = d < distance
        owner[better], distance[better] = i, d[better]
    second = np.maximum(parent[owner], 0)
    joint_distance = np.linalg.norm(points - rest[second], axis=1)
    blend = np.clip(joint_distance / 0.06, 0, 1)
    weight = 0.5 + 0.5 * blend * blend * (3 - 2 * blend)
    out = np.zeros((len(points), 4), dtype="<u4")
    out[:, 0] = owner + 1
    out[:, 1] = second + 1
    out[:, 2] = np.rint(weight * 65535).astype(np.uint32)
    return out


def vegetation(columns: dict, points: np.ndarray, ground: float | None = None) -> np.ndarray:
    rgb = np.stack([columns[f"f_dc_{i}"] for i in range(3)], axis=1) * 0.2820947918 + 0.5
    green = (2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]) / np.maximum(rgb.sum(axis=1), 0.03)
    mask = (green > 0.06) & (columns["opacity"] > -1.5)
    if ground is not None:
        mask &= points[:, 2] > ground + 0.035
    return mask


def tuft_hierarchy(points: np.ndarray, mask: np.ndarray, ground: float) -> list[dict]:
    """Measured vegetation clusters in 22 cm xy patches, each with three articulated bones."""
    selected = points[mask]
    groups = np.floor(selected[:, :2] / 0.22).astype(int)
    keys, inverse_id, count = np.unique(groups, axis=0, return_inverse=True, return_counts=True)
    nodes = []
    for group in np.flatnonzero(count >= 40):
        own = selected[inverse_id == group]
        top = float(np.quantile(own[:, 2], 0.95))
        if top - ground < 0.10:
            continue
        xy = np.median(own[:, :2], axis=0)
        root = len(nodes)
        for j in range(4):
            nodes.append(
                {
                    "id": f"tuft-{keys[group, 0]}-{keys[group, 1]}-{j}",
                    "position": [float(xy[0]), float(xy[1]), ground + (top - ground) * j / 3],
                    "parent": -1 if j == 0 else root + j - 1,
                    "band": "root" if j == 0 else "branch",
                }
            )
    if not nodes:
        raise ValueError("No vegetation tufts in the selected crop")
    return nodes


def tile_data(folder: Path, uri: str, offsets: dict):
    c = unpack_spz(glb_spz(folder / uri))
    points = site_positions(np.stack([c[k] for k in ("x", "y", "z")], axis=1), offsets.get(uri))
    points[points == 0] = 0
    return c, points.astype("<f4")


def spool_crop(source: Path, out: Path) -> Path:
    """Explicit Minnetonka-independent Spool vegetation ROI, retaining low ground.

    Canonical source bytes stay untouched. The derivative crop is re-encoded SPZ/DC colour,
    shared by both recording arms. Green filtering excludes chair fragments above the soil.
    """
    folder = source / "splat"
    doc = json.loads((folder / "tileset.json").read_text())
    offsets = tile_offsets(doc)
    uris = []

    def walk(tile):
        if tile.get("children"):
            for child in tile["children"]:
                walk(child)
        else:
            uris.append(tile["content"]["uri"])

    walk(doc["root"])
    columns = {}
    for uri in uris:
        c, points = tile_data(folder, uri, offsets)
        for j, key in enumerate(("x", "y", "z")):
            c[key] = points[:, j]
        for key, values in c.items():
            columns.setdefault(key, []).append(values)
    c = {key: np.concatenate(values) for key, values in columns.items()}
    points = np.stack([c[key] for key in ("x", "y", "z")], axis=1)
    scales = np.stack([c[f"scale_{i}"] for i in range(3)], axis=1)
    mask = ((points > [0.55, 0.15, 0.23]) & (points < [1.75, 1.3, 0.72])).all(axis=1)
    mask &= (points[:, 2] < 0.355) | vegetation(c, points)
    mask &= np.exp(scales).max(axis=1) < 0.12
    points = points[mask]
    quat = np.stack([c[f"rot_{i}"][mask] for i in (1, 2, 3, 0)], axis=1)
    sh0 = np.stack([c[f"f_dc_{i}"][mask] for i in range(3)], axis=1)
    spz = pack_spz(points, sh0, c["opacity"][mask], scales[mask], quat)
    destination = out / "scene" / "splat"
    destination.mkdir(parents=True, exist_ok=True)
    low, high = points.min(axis=0), points.max(axis=0)
    (destination / "splat.glb").write_bytes(
        build_glb(len(points), low.tolist(), high.tolist(), spz)
    )
    center, half = (low + high) / 2, (high - low) / 2
    root = {
        "transform": doc["root"]["transform"],
        "refine": "REPLACE",
        "geometricError": 0,
        "boundingVolume": {
            "box": [
                *center.tolist(),
                float(half[0]),
                0,
                0,
                0,
                float(half[1]),
                0,
                0,
                0,
                float(half[2]),
            ]
        },
        "content": {"uri": "splat.glb"},
        "extras": {"gaussians": len(points), "nativeLod": False},
    }
    (destination / "tileset.json").write_text(
        json.dumps({"asset": {"version": "1.1"}, "geometricError": 0, "root": root}, indent=2)
        + "\n"
    )
    (out / "crop.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "bounds": [[0.55, 0.15, 0.23], [1.75, 1.3, 0.72]],
                "ground": 0.29,
                "soilRetainedBelow": 0.355,
                "vegetationExcessGreen": 0.06,
                "maximumSplatAxisM": 0.12,
                "splats": len(points),
                "encoding": "SPZ re-encode, DC colour only",
            },
            indent=2,
        )
        + "\n"
    )
    return destination.parent


def bake(source: Path, out: Path, grass: bool = False, ground: float = -0.35) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    folder = source / "splat"
    tileset = json.loads((folder / "tileset.json").read_text())
    offsets = tile_offsets(tileset)
    uris = tile_uris(tileset)
    if grass:
        # A crop written by the lab contains just one leaf tile; never duplicate parent LODs.
        if len(uris) != 1:
            raise ValueError("Grass baker expects a one-tile crop")
        columns, points = tile_data(folder, uris[0], offsets)
        nodes = tuft_hierarchy(points, vegetation(columns, points, ground), ground)
        source_note = "Spool scan vegetation crop; inferred tuft chains, simulated motion"
    else:
        rig = json.loads((source / "source/rig.json").read_text())
        if rig.get("bindingPath"):
            raise ValueError("This nearest-segment baker requires a rig without a bindingPath")
        nodes, source_note = rig["nodes"], rig.get("sourceNote", "scan rig")
    frequencies, coefficients, error = node_spectra(nodes, grass)
    np.stack([coefficients.real, coefficients.imag], axis=-1).astype("<f4").tofile(
        out / "motion.bin"
    )
    leaves = []
    cluster_ids = {}
    tiles = {}
    offset = 0
    moving_count = 0
    with (out / "bindings.bin").open("wb") as stream:
        for uri in uris:
            columns, points = tile_data(folder, uri, offsets)
            binding = segment_binding(points, nodes)
            green = vegetation(columns, points, ground if grass else None)
            if grass:
                binding[~green] = 0
            scales = np.exp(np.stack([columns[f"scale_{i}"] for i in range(3)], axis=1))
            small = (
                green
                & (scales.max(axis=1) < 0.065)
                & (points[:, 2] > (ground + 0.08 if grass else 0.8))
            )
            # Tiny spatial packets approximate a leaf, not a semantic leaf segmentation.
            # Membership includes branch ID; neighbouring branches never share flutter.
            cells = np.floor(points / 0.035).astype(int)
            keys = np.column_stack([binding[:, 0], cells])
            unique, reverse = np.unique(keys[small], axis=0, return_inverse=True)
            local_ids = []
            for key in unique:
                key_tuple = tuple(int(v) for v in key)
                if key_tuple not in cluster_ids:
                    index = len(leaves)
                    cluster_ids[key_tuple] = index + 1
                    pivot = (key[1:] + 0.5) * 0.035
                    rng = np.random.default_rng(
                        sum((int(v) & 0xFFFF) * p for v, p in zip(key, [3, 5, 7, 11], strict=True))
                    )
                    axis = rng.normal(size=3)
                    axis /= np.linalg.norm(axis)
                    leaves.append(
                        [
                            *pivot,
                            *axis,
                            float(rng.integers(20, 61)) / 10,
                            float(rng.uniform(0, 2 * np.pi)),
                            float(rng.uniform(0.035, 0.10)),
                        ]
                    )
                local_ids.append(cluster_ids[key_tuple])
            binding[small, 3] = np.array(local_ids, dtype=np.uint32)[reverse]
            # Each leaf packet is rigid on one branch. Blend only woody joint splats.
            binding[small, 2] = 65535
            data = binding.tobytes()
            checksum = checksum_positions(points)
            tiles[checksum] = {"offset": offset, "count": len(points)}
            stream.write(data)
            offset += len(data)
            moving_count += int(np.count_nonzero(binding[:, 0]))
            print(
                f"{uri}: {len(points)} splats, {np.count_nonzero(small)} flutter splats", flush=True
            )
    np.asarray(leaves, dtype="<f4").reshape(-1, 9).tofile(out / "leaves.bin")
    meta = {
        "format": "hexapod.living-articulation",
        "version": 1,
        "motionEvidence": "simulated",
        "nodes": nodes,
        "frequenciesHz": frequencies.tolist(),
        "referenceBearingDeg": 90,
        "motionBinary": "motion.bin",
        "leafBinary": "leaves.bin",
        "bindingBinary": "bindings.bin",
        "leafCount": len(leaves),
        "tiles": tiles,
        "durationS": 10,
        "sampleRateHz": 30,
        "sourceNote": source_note,
        "metrics": {
            "rotationRmsErrorRad": error,
            "movingSplatsAcrossLods": moving_count,
            "leafPacketM": 0.035,
        },
    }
    (out / "articulation.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        json.dumps({"nodes": len(nodes), "leafClusters": len(leaves), "rmsRad": error}), flush=True
    )
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--grass", action="store_true")
    parser.add_argument("--spool-crop", action="store_true")
    parser.add_argument("--ground", type=float, default=-0.35)
    args = parser.parse_args()
    if args.spool_crop:
        args.source = spool_crop(args.source, args.out)
        args.grass, args.ground = True, 0.29
    bake(args.source, args.out, args.grass, args.ground)
