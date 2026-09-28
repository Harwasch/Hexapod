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

- limbs: at every joint the child carrying the most tips continues the axis (straightest on a
  tie, no angle limit — an extracted skeleton zigzags); a limb whose frequency would exceed
  ``MODE_BAND * f0`` (3 f0, Rodriguez et al. 2008) or is deeper than ``MAX_BRANCH_ORDER`` (3) is
  a twig that rides the oscillator of the limb it hangs from, unbent;
- branch frequency ``f = 2.55 * L**-0.59`` Hz (Coder 2000, via Habel, Kusternig & Wimmer,
  "Physically Guided Animation of Trees", EG 2009, eq. 17), ``L`` the limb's chord in metres
  (Habel states no unit and the primary, UGA FOR00-24, was not retrievable; only metres agrees
  with measured whole-tree frequencies);
- whole-tree frequency ``f0 = 2.4 / sqrt(H)`` (simple pendulum, Jackson et al. 2021,
  Biogeosciences 18, 4059; the constant is an estimate read off their Fig. 2a);
- whole-tree damping 0.086 in leaf, 0.039 leafless (Jackson et al. 2019, J. R. Soc. Interface);
- limb damping rising to 0.15 with leaf load (UNVERIFIED prior);
- every limb bends the same angle (elastic similarity) times ``min(1, f0/f)^0.305``, so its tip
  deflection falls as 1/f^2 like a sub-resonant oscillator's (estimate), spread over its joints
  by length, so no
  radius is read — ``skeleton.py``'s woody radius is a resolution limit on most crown nodes;
- leaf flutter 6 mm at 10 m/s at the tips, on wavelengths of 4-10 leaf sizes, the leaf size no
  smaller than 5 cm however fine the capture's splats (estimates).

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
MODE_BAND = 3.0
MAX_BRANCH_ORDER = 3
RESPONSE_EXPONENT = 2 + 1 / CODER_EXPONENT
REFERENCE_SPEED_MPS = 10
TREE_BEND_REF_RAD = 0.02
BRANCH_BEND_REF_RAD = 0.05
FLUTTER_REF_M = 0.006
FLUTTER_REACH_M = 0.5
FLUTTER_CUT = 0.05
DEFAULT_LEAF_SIZE_M = 0.06
MIN_LEAF_SIZE_M = 0.05

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
        "rule": "f = 2.55 * L^-0.59 Hz, L = limb span (farthest reach of its subtree from its "
        "attachment) in metres; unit not stated by Habel, primary not retrieved; metres because "
        "only then does it agree with whole-tree data (6 m: 0.89 Hz vs 2.4/sqrt(H) = 0.98; in "
        "feet 0.44)",
        "source": "Coder 2000, Sway frequency in tree stems, UGA FOR00-24, via Habel, "
        "Kusternig & Wimmer, Physically Guided Animation of Trees, EG 2009, eq. 17",
        "status": "cited",
    },
    "limbGrouping": {
        "rule": "each joint continues into its child with the most tips (straightest on a tie), "
        "no angle limit; a limb is a mode when eq. 17 puts it within 3 f0 and it is at most "
        "third-order, otherwise a twig riding the limb it hangs from with no bend of its own; a "
        "mode rings no lower than f0",
        "source": "da Vinci's rule for the main axis; band from Rodriguez, de Langre & Moulia "
        "2008, AJB 95:1523 (walnut: first 25 modes in 1.4-2.6 Hz; branch modes 2.5-3 Hz over "
        "1-1.5 Hz fundamentals); order cap within game practice (SpeedTree 1-2 branch levels, "
        "Pivot Painter 2 up to 4)",
        "status": "estimate",
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
        "rule": "tree lean 0.02 rad, a limb 0.05 rad * min(1, f0/f)^(2 - 1/0.59) at 10 m/s, "
        "spread over its joints by segment length",
        "source": "elastic similarity (McMahon & Kronauer 1976) for the spread; tip deflection "
        "(F/m)/(2 pi f)^2 below resonance (Habel et al. EG 2009 eq. 15) with L from eq. 17 "
        "gives the angle's f^-0.305, at equal drag per unit mass (assumed); magnitudes chosen",
        "status": "estimate",
    },
    "turbulence": {
        "rule": "sway RMS 0.8 along / 0.6 across the mean lean; gusts +35% every ~20 s",
        "source": "chosen; SpeedTree-style gust parameters",
        "status": "estimate",
    },
    "leafFlutter": {
        "rule": "advected field, wavelengths 4-10 leaf sizes (leaf size >= 5 cm), carried at "
        "0.3 U; 6 mm at 10 m/s at the tips",
        "source": "Habel et al. EG 2009 sec. 7.2 (min wavelength >= 4x leaf size, advected by "
        "-W t); Tadrist et al. 2018, J. R. Soc. Interface 15: 20180010 (flutter dominates only "
        "at low wind, branch buffeting above); amplitudes and the 5 cm floor chosen",
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
    # The axis continues into the child that carries the most of the crown — the most tips, since
    # by da Vinci's rule a limb's cross-section follows what it carries — and the straightest of
    # those on a tie. There is no angle limit: an extracted skeleton zigzags (a median 56 deg
    # turn per joint on the Minnetonka rig), and an unbranched chain is one limb however much it
    # wanders. The v1 rule (35 deg, half the longest sibling) cut that rig into 150 "branches".
    continuation = [-1] * count
    for i, node in enumerate(nodes):
        kids = children[i]
        if not kids:
            continue
        parent = nodes[node["parent"]] if node["parent"] >= 0 else None
        incoming = (
            _unit(parent["position"], node["position"]) if parent is not None else None
        ) or [0.0, 0.0, 1.0]
        best = -1
        best_tips = -1
        best_cos = -2.0
        for kid in kids:
            out = _unit(node["position"], nodes[kid]["position"]) or incoming
            c = out[0] * incoming[0] + out[1] * incoming[1] + out[2] * incoming[2]
            if tips[kid] > best_tips or (tips[kid] == best_tips and c > best_cos):
                best_tips = tips[kid]
                best_cos = c
                best = kid
        continuation[i] = best
    # Limbs: maximal chains of continuations, each named by its first joint.
    limb = [0] * count
    path_length: dict[int, float] = {}
    last: dict[int, int] = {}
    for i in range(1, count):
        p = nodes[i]["parent"]
        base = limb[p] if p > 0 and continuation[p] == i else i
        limb[i] = base
        path_length[base] = path_length.get(base, 0.0) + segment[i]
        last[base] = i  # topological order: a chain's far end is its highest index
    tree_branch = continuation[0] if count else -1
    # Branching order: the trunk is 0, a limb on it 1, a limb on that 2, ...
    order = [0] * count
    for i in range(1, count):
        base = limb[i]
        if i == base and base != tree_branch:
            attach = nodes[base]["parent"]
            order[i] = order[limb[attach]] + 1 if attach > 0 else 1
        else:
            order[i] = order[base]
    # A limb's length is its span: the farthest any joint of its subtree (its axis and all that
    # hangs from it) reaches from the joint it hangs from. Beam scaling (f ~ D/L^2, D ~ L^1.4)
    # holds for a whole branch, not a segment; and a straight-line reach ignores the zigzag of
    # an extracted chain, which summing segments would count as length.
    span: dict[int, float] = {base: 0.0 for base in last}
    for i in range(1, count):
        cursor = i
        while cursor > 0:
            base = limb[cursor]
            attach = nodes[base]["parent"]
            reach_m = _distance(nodes[attach]["position"], nodes[i]["position"])
            span[base] = max(span[base], reach_m)
            cursor = attach
    return {
        "continuation": continuation,
        "limb": limb,
        "order": order,
        "segment": segment,
        "path_length": path_length,
        "branch_length": span,
        "tips": tips,
        "reach": reach,
        "tree_branch": tree_branch,
    }


def limb_modes(rig: dict, structure: dict, tree_hz: float) -> tuple[list[int], list[bool]]:
    """Which oscillator every joint follows, and whether its limb is a twig. Mirrors
    ``limbModes`` in motionParams.ts.

    A limb is a twig — it rides the oscillator of the limb it hangs from, with no bend of its
    own, and moves only by leaf flutter — when it is too short to ring inside the tree's band
    (eq. 17 above ``MODE_BAND * f0``) or deeper than ``MAX_BRANCH_ORDER``. A twig on the root
    joins the trunk's mode."""
    nodes = rig["nodes"]
    limb = structure["limb"]
    tree = structure["tree_branch"]
    ceiling = MODE_BAND * tree_hz
    branch = [0] * len(nodes)
    twig = [False] * len(nodes)
    for i in range(1, len(nodes)):
        base = limb[i]
        if base != tree and (
            structure["order"][base] > MAX_BRANCH_ORDER
            or branch_frequency_hz(structure["branch_length"][base]) > ceiling
        ):
            twig[i] = True
            attach = nodes[base]["parent"]
            branch[i] = branch[attach] if attach > 0 else tree
        else:
            branch[i] = base
    return branch, twig


def branch_frequency_hz(length_m: float) -> float:
    return CODER_COEFFICIENT_HZ * math.pow(max(length_m, MIN_BRANCH_LENGTH_M), CODER_EXPONENT)


def tree_frequency_hz(height_m: float) -> float:
    return PENDULUM_COEFFICIENT / math.sqrt(max(height_m, 0.5))


def limb_bend_rad(frequency_hz: float, tree_frequency: float) -> float:
    """A limb's total bend at the reference speed: ``BRANCH_BEND_REF_RAD`` times
    ``min(1, f0 / f)^(2 + 1/CODER_EXPONENT)``. Below resonance a limb's tip deflects by
    ``(F/m) / (2 pi f)^2`` (Habel eq. 15 at f << f_h), and by eq. 17 its length goes as
    ``f^(1/CODER_EXPONENT)``, so its bend angle, deflection over length, goes as
    ``f^(-2 - 1/CODER_EXPONENT)`` = ``f^-0.31``."""
    ratio = min(1.0, tree_frequency / frequency_hz) if frequency_hz > 0 else 1.0
    return BRANCH_BEND_REF_RAD * math.pow(ratio, RESPONSE_EXPONENT)


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
    tree_hz = tree_frequency_hz(height)
    branch, twig = limb_modes(rig, structure, tree_hz)
    modes = set(branch[1:])
    max_tips = max((structure["tips"][base] for base in modes if base != tree), default=0)
    columns: dict[str, list[float]] = {
        key: []
        for key in ("branch", "mode", "share", "frequencyHz", "damping", "gainRad", "flutterM")
    }
    for i in range(len(nodes)):
        if i == 0:
            values = (0, 0, 0, round_to(tree_hz, 4), TREE_DAMPING_SUMMER, 0, 0)
        else:
            base = branch[i]
            is_tree = base == tree
            # A mode rings no lower than the tree it hangs from (Rodriguez et al. 2008).
            frequency = (
                tree_hz
                if is_tree
                else max(tree_hz, branch_frequency_hz(structure["branch_length"][base]))
            )
            # A twig rides its limb: same oscillator, no bend of its own.
            path = structure["path_length"][base]
            share = 0.0 if twig[i] or not path > 0 else structure["segment"][i] / path
            bend = TREE_BEND_REF_RAD if is_tree else limb_bend_rad(frequency, tree_hz)
            weight = math.exp(-structure["reach"][i] / FLUTTER_REACH_M)
            values = (
                base,
                0 if is_tree else 1,
                round_to(share, 6),
                round_to(frequency, 4),
                round_to(
                    TREE_DAMPING_SUMMER
                    if is_tree
                    else limb_damping(structure["tips"][base], max_tips),
                    4,
                ),
                round_to(bend * share, 7),
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
        # A splat is a piece of a leaf, not a leaf: a fine capture's foliage splats are 2 cm
        # (Minnetonka), and wavelengths of 4-10 of those, advected at 0.3 U, flicker at 10-20 Hz.
        size = None if estimate is None else max(round_to(estimate, 4), MIN_LEAF_SIZE_M)
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
