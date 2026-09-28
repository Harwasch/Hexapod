"""Living Mode motion sidecar: per-branch oscillators for a rig, from its skeleton alone.

A rig (``rig.json``) says where a tree's joints are. The sidecar this writes (``motion.json``
beside it) says how each joint moves under wind — which branch oscillator it belongs to, that
branch's natural frequency and damping, the joint's share of the branch's bend, how hard its
leaves flutter — plus wind defaults, a seed, and ``motionEvidence: "allometric"``: the lowest
rung of the evidence ladder, parameters from tree size and branch lengths alone. The rig gains a
``"motion": "motion.json"`` pointer so the runtime reads the sidecar without probing for it.

This is the Python half of ``deriveMotionSidecar`` in ``packages/world/src/motionParams.ts``;
the TypeScript test ``motionParams.test.ts`` re-derives the committed synthetic-tree sidecar and
compares it field by field, so the two cannot drift. The rules and their sources:

- branch frequency ``f = 2.55 * L**-0.59`` Hz (Coder 2000, via Habel, Kusternig & Wimmer,
  "Physically Guided Animation of Trees", EG 2009, eq. 17), ``L`` the branch length in metres;
- whole-tree frequency ``f0 = 2.4 / sqrt(H)`` (simple pendulum, Jackson et al. 2021,
  Biogeosciences 18, 4059; the constant is an estimate read off their Fig. 2a);
- whole-tree damping 0.086 in leaf, 0.039 leafless (Jackson et al. 2019, J. R. Soc. Interface);
- limb damping rising to 0.15 with leaf load (UNVERIFIED prior);
- every branch bends the same angle (elastic similarity), spread over its joints by length, so
  no radius is read — ``skeleton.py``'s woody radius is a resolution limit on most crown nodes.

Usage::

    uv run python motion_params.py ../../data/tiles/synthetic-tree/source/rig.json \\
        --ply ../../data/tiles/synthetic-tree/source/splat.ply

writes ``motion.json`` beside the rig and adds the pointer to ``rig.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

MOTION_SIDECAR_FORMAT = "hexapod.motion"
MOTION_SIDECAR_VERSION = 1

CODER_COEFFICIENT_HZ = 2.55
CODER_EXPONENT = -0.59
LEAFLESS_FREQUENCY_SCALE = 2.5
PENDULUM_COEFFICIENT = 2.4
TREE_DAMPING_SUMMER = 0.086
TREE_DAMPING_WINTER = 0.039
LIMB_DAMPING_MAX = 0.15
DAMPING_LEVELS = 3
MIN_BRANCH_LENGTH_M = 0.05
CONTINUATION_MAX_ANGLE_RAD = 35 * math.pi / 180
CONTINUATION_MIN_SHARE = 0.5
REFERENCE_SPEED_MPS = 10
TREE_BEND_REF_RAD = 0.02
BRANCH_BEND_REF_RAD = 0.05
FLUTTER_REF_M = 0.012
FLUTTER_REACH_M = 0.5
FLUTTER_CUT = 0.05
DEFAULT_LEAF_SIZE_M = 0.06

DEFAULT_SIDECAR_WIND = {
    "meanSpeedMps": 5,
    "bearingDeg": 0,
    "gust": {"strength": 0.35, "variance": 0.3, "frequencyPerMin": 3, "durationS": 5},
    "turbulence": {"along": 0.8, "across": 0.6},
    "canopyAdvection": 0.3,
}

#: Where every number came from. Must equal ALLOMETRIC_PROVENANCE in motionParams.ts.
ALLOMETRIC_PROVENANCE = {
    "branchFrequency": {
        "rule": "f = 2.55 * L^-0.59 Hz, L = branch length in metres (unit not stated in the source)",
        "source": "Coder 2000, via Habel, Kusternig & Wimmer, Physically Guided Animation of "
        "Trees, EG 2009, eq. 17",
        "status": "cited",
    },
    "leaflessFrequency": {
        "rule": "leafless branches ~2.5x eq. 17",
        "source": "Habel et al. EG 2009, sec. 5.4",
        "status": "cited",
    },
    "treeFrequency": {
        "rule": "f0 = 2.4 / sqrt(H) Hz (simple pendulum); C = 2.4 read by eye from Fig. 2a, +-30%",
        "source": "Jackson et al. 2021, Biogeosciences 18, 4059, Table 2 (pendulum best for "
        "open-grown broadleaves) and Fig. 2a",
        "status": "estimate",
    },
    "treeDamping": {
        "rule": "zeta = 0.086 summer, 0.039 winter",
        "source": "Jackson et al. 2019, J. R. Soc. Interface 16: 20190116 (pull-and-release, "
        "4 broadleaves)",
        "status": "cited",
    },
    "limbDamping": {
        "rule": "zeta rises from 0.086 to 0.15 with sqrt(limb leaf tips / largest limb's), "
        "in 3 steps",
        "source": "Habel et al. EG 2009 sec. 5.4 citing Moore & Maguire 2004: large leafy "
        "branches near critical damping",
        "status": "unverified",
    },
    "windSpectrum": {
        "rule": "P(f) shape (1 + f/U)^(-5/3), applied at each branch's own frequency",
        "source": "Simiu & Scanlan 1986 as used by Habel et al. EG 2009, eq. 14",
        "status": "cited",
    },
    "dragScaling": {
        "rule": "deflection proportional to U^2",
        "source": "Jackson et al. 2021 (deflection fitted linear in squared wind speed)",
        "status": "cited",
    },
    "bendGains": {
        "rule": "tree lean 0.02 rad, every branch 0.05 rad at 10 m/s, spread over joints by length",
        "source": "elastic similarity (McMahon & Kronauer 1976) for the spread; magnitudes chosen",
        "status": "estimate",
    },
    "turbulence": {
        "rule": "sway RMS 0.8 along / 0.6 across the mean lean; gusts +35% every ~20 s",
        "source": "chosen; SpeedTree-style gust parameters",
        "status": "estimate",
    },
    "leafFlutter": {
        "rule": "advected field, wavelengths 4-10 leaf sizes, carried at 0.3 U; 12 mm at 10 m/s "
        "at the tips",
        "source": "Habel et al. EG 2009 sec. 7.2 (min wavelength >= 4x leaf size, advected by "
        "-W t)",
        "status": "estimate",
    },
}


def round_to(value: float, digits: int) -> float:
    """``Math.round(value * 10**digits) / 10**digits`` — JavaScript's half-up, not banker's."""
    factor = 10**digits
    return math.floor(value * factor + 0.5) / factor


def _distance(a: list[float], b: list[float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2])


def _unit(a: list[float], b: list[float]) -> list[float] | None:
    d = _distance(a, b)
    if not d > 0:
        return None
    return [(b[0] - a[0]) / d, (b[1] - a[1]) / d, (b[2] - a[2]) / d]


def branch_structure(rig: dict) -> dict:
    """Which joint continues which limb. Mirrors ``branchStructure`` in motionParams.ts."""
    nodes = rig["nodes"]
    count = len(nodes)
    children: list[list[int]] = [[] for _ in nodes]
    segment = [0.0] * count
    for i, node in enumerate(nodes):
        if node["parent"] >= 0:
            children[node["parent"]].append(i)
            segment[i] = _distance(node["position"], nodes[node["parent"]]["position"])
    reach = [0.0] * count
    tips = [0] * count
    for i in range(count - 1, -1, -1):
        if not children[i]:
            tips[i] = 1
        for kid in children[i]:
            tips[i] += tips[kid]
            through = segment[kid] + reach[kid]
            reach[i] = max(reach[i], through)
    cos_limit = math.cos(CONTINUATION_MAX_ANGLE_RAD)
    continuation = [-1] * count
    for i, node in enumerate(nodes):
        kids = children[i]
        if not kids:
            continue
        parent = nodes[node["parent"]] if node["parent"] >= 0 else None
        incoming = (
            _unit(parent["position"], node["position"]) if parent is not None else None
        ) or [0.0, 0.0, 1.0]
        longest = max(segment[kid] + reach[kid] for kid in kids)
        best = -1
        best_cos = cos_limit
        for kid in kids:
            out = _unit(node["position"], nodes[kid]["position"])
            if out is None:
                continue
            if segment[kid] + reach[kid] < CONTINUATION_MIN_SHARE * longest:
                continue
            c = out[0] * incoming[0] + out[1] * incoming[1] + out[2] * incoming[2]
            if c > best_cos:
                best_cos = c
                best = kid
        continuation[i] = best
    branch = [0] * count
    branch_length: dict[int, float] = {}
    for i in range(1, count):
        p = nodes[i]["parent"]
        base = branch[p] if p > 0 and continuation[p] == i else i
        branch[i] = base
        branch_length[base] = branch_length.get(base, 0.0) + segment[i]
    tree_branch = continuation[0] if count else -1
    if tree_branch < 0 and count:
        longest = -1.0
        for kid in children[0]:
            length = segment[kid] + reach[kid]
            if length > longest:
                longest = length
                tree_branch = kid
    return {
        "continuation": continuation,
        "branch": branch,
        "segment": segment,
        "branch_length": branch_length,
        "tips": tips,
        "reach": reach,
        "tree_branch": tree_branch,
    }


def branch_frequency_hz(length_m: float) -> float:
    return CODER_COEFFICIENT_HZ * math.pow(max(length_m, MIN_BRANCH_LENGTH_M), CODER_EXPONENT)


def tree_frequency_hz(height_m: float) -> float:
    return PENDULUM_COEFFICIENT / math.sqrt(max(height_m, 0.5))


def limb_damping(tips: int, max_tips: int) -> float:
    share = math.sqrt(min(1.0, max(0.0, tips / max_tips))) if max_tips > 0 else 0.0
    level = math.floor(share * DAMPING_LEVELS + 0.5) / DAMPING_LEVELS
    return TREE_DAMPING_SUMMER + (LIMB_DAMPING_MAX - TREE_DAMPING_SUMMER) * level


def estimate_leaf_size(
    positions: np.ndarray, log_scales: np.ndarray, rig: dict, flutter_m: list[float]
) -> float | None:
    """Median foliage splat diameter, metres: twice the largest axis of every splat whose
    nearest rig node flutters. On a capture a splat, not a leaf, is the primitive that must move
    as one, so it is the size Habel's "shortest wavelength >= 4 leaf sizes" rule protects."""
    from scipy.spatial import cKDTree

    if positions.size == 0:
        return None
    centres = np.asarray([node["position"] for node in rig["nodes"]], dtype=np.float64)
    _, nearest = cKDTree(centres).query(positions.astype(np.float64))
    fluttering = np.asarray(flutter_m, dtype=np.float64)[nearest] > 0
    if not fluttering.any():
        return None
    diameters = 2.0 * np.exp(log_scales[fluttering].max(axis=1))
    return float(np.median(diameters))


def derive_sidecar(
    rig: dict,
    *,
    tree_height_m: float | None = None,
    leaf_size_m: float = DEFAULT_LEAF_SIZE_M,
    seed: int = 1,
    generator: str = "tools/captures/motion_params.py",
) -> dict:
    """An ``allometric`` sidecar for ``rig``. Mirrors ``deriveMotionSidecar``."""
    structure = branch_structure(rig)
    nodes = rig["nodes"]
    root_z = nodes[0]["position"][2]
    top = max(node["position"][2] - root_z for node in nodes)
    height = top if tree_height_m is None else tree_height_m
    tree = structure["tree_branch"]
    max_tips = max(
        (structure["tips"][base] for base in structure["branch_length"] if base != tree), default=0
    )
    columns: dict[str, list[float]] = {
        key: []
        for key in ("branch", "mode", "share", "frequencyHz", "damping", "gainRad", "flutterM")
    }
    for i in range(len(nodes)):
        if i == 0:
            values = (0, 0, 0, round_to(tree_frequency_hz(height), 4), TREE_DAMPING_SUMMER, 0, 0)
        else:
            base = structure["branch"][i]
            is_tree = base == tree
            length = structure["branch_length"].get(base, 0.0)
            share = structure["segment"][i] / length if length > 0 else 0.0
            weight = math.exp(-structure["reach"][i] / FLUTTER_REACH_M)
            values = (
                base,
                0 if is_tree else 1,
                round_to(share, 6),
                round_to(tree_frequency_hz(height) if is_tree else branch_frequency_hz(length), 4),
                round_to(
                    TREE_DAMPING_SUMMER
                    if is_tree
                    else limb_damping(structure["tips"][base], max_tips),
                    4,
                ),
                round_to((TREE_BEND_REF_RAD if is_tree else BRANCH_BEND_REF_RAD) * share, 7),
                0 if weight < FLUTTER_CUT else round_to(FLUTTER_REF_M * weight, 6),
            )
        for key, value in zip(columns, values, strict=True):
            columns[key].append(value)
    return {
        "format": MOTION_SIDECAR_FORMAT,
        "version": MOTION_SIDECAR_VERSION,
        "motionEvidence": "allometric",
        "rigChecksum": rig["canonicalChecksum"],
        "nodeCount": len(nodes),
        "seed": seed,
        "treeHeightM": round_to(height, 4),
        "leafSizeM": leaf_size_m,
        "referenceSpeedMps": REFERENCE_SPEED_MPS,
        "wind": DEFAULT_SIDECAR_WIND,
        "seasons": {
            "winter": {
                "dampingScale": round_to(TREE_DAMPING_WINTER / TREE_DAMPING_SUMMER, 4),
                "branchFrequencyScale": LEAFLESS_FREQUENCY_SCALE,
                "flutterScale": 0,
            }
        },
        "nodes": columns,
        "provenance": ALLOMETRIC_PROVENANCE,
        "generator": generator,
    }


def sidecar_for(
    rig: dict,
    positions: np.ndarray | None = None,
    log_scales: np.ndarray | None = None,
    *,
    leaf_size_m: float | None = None,
    tree_height_m: float | None = None,
    seed: int = 1,
) -> dict:
    """The sidecar, with the tree's height taken from its splats and its leaf size from its
    foliage splats, when given.

    Height is the splats' own vertical extent, top to bottom: the joints stop short of the
    crown's top, and an extracted skeleton's root is the centroid of its lowest band, not the
    ground (on the synthetic tree, 0.7 m up — 10 % of the height, 5 % of the frequency)."""
    height = tree_height_m
    if height is None and positions is not None and positions.size:
        height = float(positions[:, 2].max() - positions[:, 2].min())
    draft = derive_sidecar(rig, tree_height_m=height, seed=seed)
    size = leaf_size_m
    if size is None and positions is not None and log_scales is not None:
        estimate = estimate_leaf_size(positions, log_scales, rig, draft["nodes"]["flutterM"])
        size = None if estimate is None else round_to(estimate, 4)
    return derive_sidecar(
        rig,
        tree_height_m=height,
        leaf_size_m=DEFAULT_LEAF_SIZE_M if size is None else size,
        seed=seed,
    )


def write_sidecar(source_dir: Path, rig: dict, sidecar: dict, name: str = "motion.json") -> dict:
    """Writes the sidecar beside the rig and points the rig at it. Returns the updated rig."""
    (source_dir / name).write_text(
        json.dumps(sidecar, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    pointed = {**rig, "motion": name}
    (source_dir / "rig.json").write_text(
        json.dumps(pointed, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return pointed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rig", type=Path, help="rig.json; motion.json is written beside it")
    parser.add_argument("--ply", type=Path, help="the rig's splat.ply, for height and leaf size")
    parser.add_argument("--leaf-size", type=float, default=None, help="metres; overrides the PLY")
    parser.add_argument(
        "--tree-height", type=float, default=None, help="measured height, metres; overrides the PLY"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rig = json.loads(args.rig.read_text(encoding="utf-8"))
    positions = log_scales = None
    if args.ply is not None:
        import splat_tiles

        data = splat_tiles.read_ply(args.ply)
        positions = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
        log_scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)
    sidecar = sidecar_for(
        rig,
        positions,
        log_scales,
        leaf_size_m=args.leaf_size,
        tree_height_m=args.tree_height,
        seed=args.seed,
    )
    write_sidecar(args.rig.parent, rig, sidecar)
    frequencies = [f for i, f in enumerate(sidecar["nodes"]["frequencyHz"]) if i > 0]
    print(
        json.dumps(
            {
                "nodes": sidecar["nodeCount"],
                "branches": len(set(sidecar["nodes"]["branch"][1:])),
                "treeHeightM": sidecar["treeHeightM"],
                "leafSizeM": sidecar["leafSizeM"],
                "frequencyHz": [min(frequencies), max(frequencies)] if frequencies else [],
                "motionEvidence": sidecar["motionEvidence"],
            }
        )
    )


if __name__ == "__main__":
    main()
