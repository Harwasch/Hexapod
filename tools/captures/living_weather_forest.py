"""Deterministic, DC-only coarse Minnetonka stand, baked once; canonical bytes untouched.

80 copies, crown-spaced grove, 30k splats/tree, +/-10% scale. Twelve connected
rig nodes/tree keep the unchanged articulation texture below its 1024-node limit.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from living_articulation import pin_static, segment_binding, tile_data
from rig_tiles import glb_spz
from splat_tiles import build_glb, pack_spz, tile_offsets, unpack_spz
from synthetic_tree import checksum_positions


def crown_measurement(points: np.ndarray) -> dict:
    """Central 98% crown-centre extent above 1.5 m; excludes sparse reconstruction tails."""
    crown = points[points[:, 2] > 1.5]
    low, high = np.quantile(crown, [0.01, 0.99], axis=0)
    axes = high[:2] - low[:2]
    return {
        "diameterM": float(np.mean(axes)),
        "axisDiametersM": axes.tolist(),
        "crownBaseM": 1.5,
        "quantiles": [0.01, 0.99],
        "splatsMeasured": len(points),
    }


def layout(diameter: float = 3.889033203125, seed: int = 281026) -> list[dict]:
    """Jittered hex grove; nominal spacing .9 crown diameter, no coincident trunks."""
    rng = np.random.default_rng(seed)
    spacing = 0.9 * diameter
    return [
        {
            "x": float(
                (x - 4.5 + 0.5 * (y % 2) - 0.25) * spacing + rng.uniform(-0.035, 0.035) * diameter
            ),
            "y": float(
                (y - 3.5) * spacing * np.sqrt(3) / 2 + rng.uniform(-0.035, 0.035) * diameter
            ),
            "yaw": float(rng.uniform(0, 2 * np.pi)),
            "scale": float(rng.uniform(0.9, 1.1)),
            "mirror": bool(rng.integers(2)),
            "brightness": float(rng.uniform(0.95, 1.05)),
        }
        for y in range(8)
        for x in range(10)
    ]


def coarse_nodes(nodes: list[dict], budget: int = 12) -> tuple[list[dict], np.ndarray]:
    """Keep trunk and spatially spread crown joints, reconnect to nearest retained ancestor."""
    chosen = {i for i, n in enumerate(nodes) if n["parent"] < 0 or n.get("band") == "trunk"}
    points = np.array([n["position"] for n in nodes])
    while len(chosen) < min(budget, len(nodes)):
        distance = np.min(np.sum((points[:, None] - points[sorted(chosen)][None]) ** 2, 2), 1)
        distance[list(chosen)] = -1
        chosen.add(int(np.argmax(distance)))
    keep = sorted(chosen)
    nearest = np.zeros(len(nodes), dtype=int)
    for i, n in enumerate(nodes):
        nearest[i] = keep.index(i) if i in chosen else nearest[max(0, n["parent"])]
    out = []
    for i in keep:
        n = dict(nodes[i])
        n["parent"] = int(nearest[n["parent"]]) if n["parent"] >= 0 else -1
        out.append(n)
    return out, nearest


def coarse_binding(points: np.ndarray, original: list[dict], nodes: list[dict]) -> np.ndarray:
    # Preserve measured static pins, but reskin foliage to the retained segments.
    # Ancestor remapping alone attached most discarded crown joints to the trunk.
    original_binding = segment_binding(points, original)
    pin_static(original_binding, points, original, 0)
    binding = segment_binding(points, nodes)
    binding[original_binding[:, 0] == 0] = 0
    return binding


def bake(source: Path, out: Path, budget: int = 30000) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    doc = json.loads((source / "splat/tileset.json").read_text())
    offsets = tile_offsets(doc)
    terminal_points = []

    def visit(tile: dict) -> None:
        if tile.get("children"):
            for child in tile["children"]:
                visit(child)
        elif tile.get("content"):
            terminal_points.append(tile_data(source / "splat", tile["content"]["uri"], offsets)[1])

    visit(doc["root"])
    measurement = crown_measurement(np.concatenate(terminal_points))
    terminal_points.clear()
    uris = [c["content"]["uri"] for c in doc["root"]["children"]]
    columns: dict[str, list] = {}
    for uri in uris:
        c, p = tile_data(source / "splat", uri, offsets)
        for j, k in enumerate(("x", "y", "z")):
            c[k] = p[:, j]
        for k, v in c.items():
            columns.setdefault(k, []).append(v)
    data = {k: np.concatenate(v) for k, v in columns.items()}
    take = np.linspace(0, len(data["x"]) - 1, min(budget, len(data["x"])), dtype=int)
    data = {k: v[take] for k, v in data.items()}
    points = np.stack([data[k] for k in ("x", "y", "z")], 1)
    scales = np.stack([data[f"scale_{i}"] for i in range(3)], 1)
    quat = np.stack([data[f"rot_{i}"] for i in (1, 2, 3, 0)], 1)
    dc = np.stack([data[f"f_dc_{i}"] for i in range(3)], 1)
    original = json.loads((source / "source/rig.json").read_text())["nodes"]
    nodes, _ = coarse_nodes(original)
    binding = coarse_binding(points, original, nodes)
    moving = binding[:, 0] > 0
    children, rig, tiles = [], [], {}
    placements = layout(measurement["diameterM"])
    with (out / "bindings.bin").open("wb") as stream:
        for i, at in enumerate(placements):
            rot = Rotation.from_rotvec([0, 0, at["yaw"]])
            reflection = np.diag([-1 if at["mirror"] else 1, 1, 1])
            reflected = points @ reflection
            p = rot.apply(reflected * at["scale"]) + [at["x"], at["y"], 0]
            local = Rotation.from_matrix(
                reflection @ Rotation.from_quat(quat).as_matrix() @ reflection
            )
            q = (rot * local).as_quat()
            colour = (
                (dc * 0.28209479177387814 + 0.5) * at["brightness"] - 0.5
            ) / 0.28209479177387814
            spz = pack_spz(p, colour, data["opacity"], scales + np.log(at["scale"]), q)
            uri = f"tree-{i:02}.glb"
            (out / uri).write_bytes(build_glb(len(p), p.min(0).tolist(), p.max(0).tolist(), spz))
            # Bind the decoded derivative, not pre-quantized input coordinates.
            c = unpack_spz(glb_spz(out / uri))
            encoded = np.stack([c[k] for k in ("x", "y", "z")], 1).astype("<f4")
            checksum = checksum_positions(encoded)
            b = binding.copy()
            b[moving, :2] += i * len(nodes)
            tiles[checksum] = {"offset": stream.tell(), "count": len(p)}
            stream.write(b.astype("<u4").tobytes())
            for n in nodes:
                n = dict(n)
                n["position"] = (
                    rot.apply((np.array(n["position"]) @ reflection) * at["scale"])
                    + [at["x"], at["y"], 0]
                ).tolist()
                if n["parent"] >= 0:
                    n["parent"] += i * len(nodes)
                rig.append(n)
            low, high = encoded.min(0), encoded.max(0)
            center, half = (low + high) / 2, (high - low) / 2
            children.append(
                {
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
                    "geometricError": 0,
                    "content": {"uri": uri},
                    "extras": {"gaussians": len(p)},
                }
            )
    np.zeros((len(rig), 1, 6), dtype="<f4").tofile(out / "motion.bin")
    (out / "leaves.bin").write_bytes(b"")
    detail = {
        "vegetation": True,
        "fittedRig": True,
        "separableCrownShare": 0,
        "attachedCrownShare": 0,
        "smallSplatShare": 0,
        "leafBindings": False,
    }
    header = {
        "format": "hexapod.living-articulation",
        "version": 1,
        "motionEvidence": "simulated",
        "nodes": rig,
        "weatherResponseScale": 0.31,
        "stand": {"diameterM": measurement["diameterM"], "copies": placements},
        "frequenciesHz": [0],
        "referenceBearingDeg": 90,
        "motionBinary": "motion.bin",
        "leafBinary": "leaves.bin",
        "bindingBinary": "bindings.bin",
        "leafCount": 0,
        "tiles": tiles,
        "detail": detail,
    }
    (out / "articulation.json").write_text(json.dumps(header) + "\n")
    shutil.copyfile(out / "tree-00.glb", out / "overview.glb")
    root = {
        "content": {"uri": "overview.glb"},
        "transform": doc["root"]["transform"],
        "boundingVolume": {"box": [0, 0, 3, 23, 0, 0, 0, 18, 0, 0, 0, 4]},
        "geometricError": 1000,
        "refine": "REPLACE",
        "children": children,
        "extras": {"nativeLod": False, "gaussians": len(points)},
    }
    (out / "tileset.json").write_text(
        json.dumps({"asset": {"version": "1.1"}, "geometricError": 1000, "root": root}) + "\n"
    )
    stats = {
        "crownMeasurement": measurement,
        "nominalSpacingM": 0.9 * measurement["diameterM"],
        "copies": len(placements),
        "splatsPerCopy": len(points),
        "totalSplats": len(points) * len(placements),
        "nodesPerCopy": len(nodes),
        "movingSplatsPerCopy": int(
            np.sum(
                (binding[:, 0] > 0)
                & np.array([n.get("band") != "trunk" for n in nodes])[
                    np.maximum(binding[:, 0], 1) - 1
                ]
            )
        ),
        "weatherResponseScale": 0.31,
        "nodes": len(rig),
        "placements": placements,
        "tier": "B",
        "colour": "DC only",
        "seed": 281026,
    }
    (out / "forest.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = bake(args.source, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "placements"}), flush=True)
