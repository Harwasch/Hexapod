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

The rig mirrors ``syntheticTreeRig()`` in packages/world/src/rig.ts node for node, so any two
sizes share one 212-node skeleton and differ only in splat density.

Usage:
    python synthetic_tree.py ../../data/tiles/synthetic-tree --splats 12000
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

#: Fraction of the splat budget spent on bark (trunk and limb cylinders); the rest is foliage.
WOOD_FRACTION = 0.34
#: Of the foliage budget, the share that goes to tip tufts rather than inner canopy.
TIP_FRACTION = 0.78

BARK_RGB = (0.34, 0.235, 0.155)
FOLIAGE_RGB = (0.185, 0.415, 0.145)

#: Bark splats spiral along a limb rather than scattering: turns of the spiral over one segment,
#: how far the azimuth is jittered off it, and how much the surface radius ripples with azimuth.
BARK_TURNS = 2.5
BARK_THETA_JITTER = 1.6
BARK_RIPPLE = 0.12
#: Angular width of a bark splat, as a multiple of the local radius. Above ~1 the spiral closes.
BARK_GIRTH_SPAN = 1.25

#: The leaf sleeve runs from this fraction of the twig's length to this one — starting clear of
#: the fork, so the splats near the parent still belong to the parent, and running past the tip.
FOLIAGE_START = 0.50
FOLIAGE_END = 1.18
#: Exponent on the along-twig coordinate. Below 1 the leaves crowd towards the tip.
FOLIAGE_TAPER = 0.75
#: A sleeve is squashed in Z, the way a sprig of foliage hangs.
FOLIAGE_FLATTEN = 0.72
#: Leaf-disc size as a fraction of the sleeve's girth. Small: the tuft must have visible grain.
FOLIAGE_LEAF_MIN = 0.22
FOLIAGE_LEAF_SPAN = 0.20
#: Smallest share of the bark budget any one limb gets, as a multiple of an equal split. Pure
#: surface-area weighting starves a 9 mm twig; a limb drawn with three splats is a dotted line.
BARK_MIN_SHARE = 0.55

#: Sleeve girth as a fraction of the limb's own length, at a twig and on the inner canopy.
TIP_FOLIAGE_GIRTH = 0.26
INNER_FOLIAGE_GIRTH = 0.24

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

#: The trunk ends at this fraction of the tree's overall height; the crown carries the rest.
TRUNK_TOP_FRACTION = 0.7

#: Radius of a terminal twig node, metres. Every other radius follows from it by da Vinci's
#: rule, so this one number sets the whole tree's thickness. 8.8 mm over a ~0.7 m twig is a
#: slenderness near 40; it also fixes the trunk at ~0.15 m radius, a slenderness near 14.
TWIG_RADIUS_M = 0.0088

#: Radius growth per segment walking *towards* a limb's base, on top of da Vinci's rule. A real
#: limb tapers along its length and not only at its forks; without this a chain of single-child
#: nodes would be a constant-radius tube.
TRUNK_TAPER = 1.03
LIMB_TAPER = 1.06

#: Tree height the limb lengths below are quoted at, metres; they scale with ``height_m``.
HEIGHT_REFERENCE_M = 6.0
#: Length of a primary branch at the lowest whorl and at the highest, metres at that height.
PRIMARY_LENGTH_M = (1.6, 0.95)
#: Elevation of a primary branch above horizontal at the lowest and highest whorl, radians.
PRIMARY_ELEVATION_RAD = (0.45, 0.82)
#: A secondary is this fraction of its primary's length; a twig this fraction of its secondary's.
SECONDARY_LENGTH_FRACTION = 0.56
TWIG_LENGTH_FRACTION = 0.85
#: Elevation each generation adds to its parent's, radians. Small, and negative at the twigs:
#: elevation accumulates down the chain, and four generations of generous gains put the tips
#: past vertical, where the three twigs of one fork end up 2 cm apart and nearest-node
#: assignment can no longer tell them apart.
SECONDARY_ELEVATION_GAIN_RAD = 0.10
TWIG_ELEVATION_GAIN_RAD = -0.05
#: Half-width of the fan a limb's children are spread through, in azimuth and in elevation.
CHILD_SPREAD_RAD = 1.15
CHILD_ELEVATION_SPREAD_RAD = 0.40
#: Elevation a limb gains along its own length: limbs curve up rather than running straight.
LIMB_CURL_RAD = 0.16

#: Stiffness runs between these, linearly in radius/trunk-base-radius. Invented, like the old
#: constants it replaces: "thicker bends less" is the only claim it makes.
STIFFNESS_MIN = 1.0
STIFFNESS_MAX = 8.0


#: Deterministic wobble applied to limb lengths and elevations so no two limbs of a whorl are
#: identical. Without it the three primaries of a whorl have the same length and the same
#: elevation, hence the same natural frequency, and they differ only in a hashed phase — which
#: is not enough to stop them moving as a unit. Real trees do not have congruent branches.
#: The multipliers are irrational, so the sequence never repeats over any limb count.
LIMB_LENGTH_JITTER = 0.22
LIMB_ELEVATION_JITTER = 0.12
LENGTH_PHASE = 0.618033988749895
ELEVATION_PHASE = 0.7548776662466927


def _wobble(index: int, phase: float) -> float:
    """A deterministic value in ``[-1, 1)`` from an integer. No RNG: the rig has no seed."""
    return 2.0 * (((index + 1) * phase) % 1.0) - 1.0


def _limb_chain(
    start: tuple[float, float, float],
    azimuth: float,
    elevation: float,
    length: float,
    segments: int,
    curl: float,
) -> list[tuple[float, float, float]]:
    """Positions of a chain of ``segments`` nodes walking out from ``start``.

    The elevation rises by ``curl`` over the whole chain, so a limb arcs upward instead of
    being a straight spoke. Pure arithmetic in the same order as the TypeScript twin.
    """
    out: list[tuple[float, float, float]] = []
    x, y, z = start
    step = length / segments
    for j in range(segments):
        el = elevation + curl * (j + 1) / segments
        horizontal = math.cos(el) * step
        x += horizontal * math.cos(azimuth)
        y += horizontal * math.sin(azimuth)
        z += math.sin(el) * step
        out.append((x, y, z))
    return out


def _child_fan(index: int, count: int) -> float:
    """Where child ``index`` of ``count`` sits in its parent's fan, in ``[-1, 1]``."""
    return (2 * index + 1) / count - 1


def _da_vinci_radii(nodes: list[dict]) -> list[float]:
    """Radii from the tips inward: a node's cross-section is the sum of its children's.

    This is da Vinci's rule, applied exactly. It is what makes the fixture's proportions
    physical rather than decorative: the old fixture gave a 1.15 m branch a 12 cm radius — a
    slenderness of 4.8 where a real branch is 20–60 — so ``radius / length²`` correctly called
    it stiff, put it at 7–10 Hz outside the forcing band, and the whole crown rode rigid while
    the trunk did all the visible work.

    One honest deviation, and it is a consequence of sampling: a real 6 m tree carries thousands
    of twigs, this rig carries ~100, and `r_trunk = r_twig · √(tips)` means the two cannot both
    be realistic at this node count. The tip radius is therefore chosen so the *trunk* lands
    where a real trunk is (slenderness ~14, fundamental ~0.5 Hz) and the twigs come out around
    slenderness 40 rather than the 50–200 of a real twig. Frequencies, which are what the motion
    model reads, land in the right band at every level.
    """
    children: list[list[int]] = [[] for _ in nodes]
    for index, node in enumerate(nodes):
        if node["parent"] >= 0:
            children[node["parent"]].append(index)
    radii = [0.0] * len(nodes)
    for index in range(len(nodes) - 1, -1, -1):
        kids = children[index]
        if not kids:
            radii[index] = TWIG_RADIUS_M
            continue
        area = 0.0
        for kid in kids:
            area += radii[kid] ** 2
        taper = TRUNK_TAPER if nodes[index]["band"] == "trunk" else LIMB_TAPER
        radii[index] = math.sqrt(area) * taper
    return radii


def synthetic_tree_rig(
    height_m: float = 6.0,
    trunk_segments: int = 10,
    whorls: int = 4,
    branches_per_whorl: int = 3,
    secondaries_per_branch: int = 3,
    twigs_per_secondary: int = 3,
    canonical_checksum: str = "fnv1a32:0:00000000",
) -> dict:
    """The same 214-node skeleton ``syntheticTreeRig()`` builds in packages/world/src/rig.ts.

    Kept deliberately in lockstep with the TypeScript: a test in ``@twin/world`` parses the
    emitted rig.json and compares it node for node with its own generator, so a change to
    either side that is not mirrored fails CI rather than quietly producing two different trees.

    Four generations — trunk, primary, secondary, twig — because nine leaf clusters cannot
    rustle. 108 of them can.
    """
    trunk_segments = max(2, int(trunk_segments))
    whorls = max(1, int(whorls))
    branches_per_whorl = max(1, int(branches_per_whorl))
    secondaries_per_branch = max(1, int(secondaries_per_branch))
    twigs_per_secondary = max(1, int(twigs_per_secondary))
    trunk_top_m = TRUNK_TOP_FRACTION * height_m

    nodes: list[dict] = []
    trunk_indices: list[int] = []
    for i in range(trunk_segments):
        f = i / (trunk_segments - 1)
        trunk_indices.append(len(nodes))
        nodes.append(
            {
                "id": f"trunk-{i}",
                "parent": -1 if i == 0 else trunk_indices[i - 1],
                "position": [0.0, 0.0, f * trunk_top_m],
                "band": "trunk",
            }
        )

    ordinal = 0
    for w in range(whorls):
        trunk_index = trunk_indices[trunk_segments - 1 - w]
        base = tuple(nodes[trunk_index]["position"])
        # w counts down from the apex, so f is 0 at the top of the crown and 1 at its skirt:
        # long, shallow branches at the bottom and short, steep ones at the top.
        f = 1.0 if whorls == 1 else w / (whorls - 1)
        scale = height_m / HEIGHT_REFERENCE_M
        whorl_len = (PRIMARY_LENGTH_M[1] + (PRIMARY_LENGTH_M[0] - PRIMARY_LENGTH_M[1]) * f) * scale
        whorl_el = (
            PRIMARY_ELEVATION_RAD[1] + (PRIMARY_ELEVATION_RAD[0] - PRIMARY_ELEVATION_RAD[1]) * f
        )
        for _ in range(branches_per_whorl):
            azimuth = GOLDEN_ANGLE * ordinal
            primary_len = whorl_len * (1.0 + LIMB_LENGTH_JITTER * _wobble(ordinal, LENGTH_PHASE))
            primary_el = whorl_el + LIMB_ELEVATION_JITTER * _wobble(ordinal, ELEVATION_PHASE)
            first = len(nodes)
            for j, point in enumerate(
                _limb_chain(base, azimuth, primary_el, primary_len, 2, LIMB_CURL_RAD)
            ):
                nodes.append(
                    {
                        "id": f"branch-{ordinal}-{j}",
                        "parent": trunk_index if j == 0 else first + j - 1,
                        "position": list(point),
                        "band": "branch",
                    }
                )
            primary_tip = first + 1
            secondary_len = primary_len * SECONDARY_LENGTH_FRACTION
            secondary_el = primary_el + LIMB_CURL_RAD + SECONDARY_ELEVATION_GAIN_RAD
            for s in range(secondaries_per_branch):
                fan = _child_fan(s, secondaries_per_branch)
                wobble_index = ordinal * 7 + s
                secondary_len_s = secondary_len * (
                    1.0 + LIMB_LENGTH_JITTER * _wobble(wobble_index, LENGTH_PHASE)
                )
                secondary_az = azimuth + CHILD_SPREAD_RAD * fan
                secondary_el_s = (
                    secondary_el
                    + CHILD_ELEVATION_SPREAD_RAD * fan
                    + LIMB_ELEVATION_JITTER * _wobble(wobble_index, ELEVATION_PHASE)
                )
                start = tuple(nodes[primary_tip]["position"])
                base_index = len(nodes)
                for j, point in enumerate(
                    _limb_chain(
                        start, secondary_az, secondary_el_s, secondary_len_s, 2, LIMB_CURL_RAD
                    )
                ):
                    nodes.append(
                        {
                            "id": f"twig-{ordinal}-{s}-{j}",
                            "parent": primary_tip if j == 0 else base_index + j - 1,
                            "position": list(point),
                            "band": "branch",
                        }
                    )
                secondary_tip = base_index + 1
                twig_len = secondary_len_s * TWIG_LENGTH_FRACTION
                twig_el = secondary_el_s + LIMB_CURL_RAD + TWIG_ELEVATION_GAIN_RAD
                for k in range(twigs_per_secondary):
                    twig_fan = _child_fan(k, twigs_per_secondary)
                    twig_index = ordinal * 13 + s * 3 + k
                    point = _limb_chain(
                        tuple(nodes[secondary_tip]["position"]),
                        secondary_az + CHILD_SPREAD_RAD * twig_fan,
                        twig_el
                        + CHILD_ELEVATION_SPREAD_RAD * twig_fan
                        + LIMB_ELEVATION_JITTER * _wobble(twig_index, ELEVATION_PHASE),
                        twig_len * (1.0 + LIMB_LENGTH_JITTER * _wobble(twig_index, LENGTH_PHASE)),
                        1,
                        LIMB_CURL_RAD,
                    )[0]
                    nodes.append(
                        {
                            "id": f"leaf-{ordinal}-{s}-{k}",
                            "parent": secondary_tip,
                            "position": list(point),
                            "band": "leaf",
                        }
                    )
            ordinal += 1

    radii = _da_vinci_radii(nodes)
    root_radius = radii[0] if radii else TWIG_RADIUS_M
    # Linear in thickness between the twigs and the trunk base, so the thinnest node in the rig
    # is exactly STIFFNESS_MIN whatever the tree's absolute scale.
    span = max(root_radius - TWIG_RADIUS_M, 1e-9)
    for node, radius in zip(nodes, radii, strict=True):
        node["radius"] = radius
        node["stiffness"] = STIFFNESS_MIN + (STIFFNESS_MAX - STIFFNESS_MIN) * (
            (radius - TWIG_RADIUS_M) / span
        )

    ordered = [
        {
            "id": node["id"],
            "parent": node["parent"],
            "position": node["position"],
            "radius": node["radius"],
            "stiffness": node["stiffness"],
            "band": node["band"],
        }
        for node in nodes
    ]
    return {
        # Key order matches serializeRig() in packages/world/src/rig.ts.
        "units": "meters",
        "canonicalChecksum": canonical_checksum,
        "sourceNote": (
            f"synthetic tree, {height_m} m, {len(ordered)} nodes (tools/captures/synthetic_tree.py)"
        ),
        "nodes": ordered,
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

    The splats are stratified along the limb rather than scattered uniformly, so a 9 mm twig
    reads as a continuous stick at ten splats instead of a dotted line, and the surface radius
    carries a low-order azimuthal ripple so a limb is not a machined tube.
    """
    if count <= 0:
        return
    axis, tangent, normal = _orthonormal_frame(p1 - p0)
    length = float(np.linalg.norm(p1 - p0))
    # Stratified: one splat per equal slice of the limb, jittered inside its own slice.
    t = (np.arange(count) + rng.random(count)) / count
    theta = t * (2.0 * math.pi * BARK_TURNS) + rng.random(count) * BARK_THETA_JITTER
    ripple = 1.0 + BARK_RIPPLE * np.cos(theta * 3.0 + t * 7.0)
    radius = (r0 + (r1 - r0) * t) * ripple * (0.9 + 0.2 * rng.random(count))
    centre = p0[None, :] + np.outer(t, p1 - p0)
    offset = np.outer(radius * np.cos(theta), tangent) + np.outer(radius * np.sin(theta), normal)
    position = centre + offset

    # Bark splats lie flat on the limb: long along the axis, thin along the surface normal.
    surface = offset / np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-9)
    along = np.broadcast_to(axis, (count, 3))
    across = np.cross(surface, along)
    quats = np.empty((count, 4))
    for i in range(count):
        quats[i] = _matrix_to_quat_wxyz((along[i], _unit(across[i]), surface[i]))

    # Long enough along the limb to close the gaps between consecutive slices, and wide enough
    # around it to close the gaps between turns of the spiral.
    span = length / count
    girth = np.maximum(radius, 0.004)
    scale = np.stack(
        [
            np.full(count, max(span * 1.15, 0.012)),
            girth * BARK_GIRTH_SPAN,
            girth * 0.35,
        ],
        axis=1,
    ) * (0.85 + 0.3 * rng.random((count, 3)))

    # Darker in the crevices the ripple makes, lighter on the ridges: enough shading for the
    # eye to read a round limb rather than a flat stripe.
    shade = (0.72 + 0.42 * rng.random((count, 1))) * (
        1.0 + 0.22 * np.cos(theta * 3.0 + t * 7.0)[:, None]
    )
    rgb = np.clip(np.asarray(BARK_RGB)[None, :] * shade, 0.0, 1.0)

    node = np.where(t < 0.5, node0, node1).astype(np.int32)
    out.add(position, rgb, 1.9 + 0.6 * rng.random(count), np.log(scale), quats, node)


def _foliage_splats(
    rng: np.random.Generator,
    out: _Splats,
    anchor: np.ndarray,
    tip: np.ndarray,
    spread: float,
    count: int,
    node: int,
    height_m: float,
) -> None:
    """Leaves along one twig: a tapered sleeve about the twig's own axis, not a ball at its end.

    The old fixture spent its whole foliage budget on eighteen large ellipsoids of randomly
    oriented splats centred past the tip of a limb, which is why it read as green fog attached
    to a post, and why — once the rig was dense enough for tufts to touch — nearest-node
    assignment could no longer tell one tuft from its neighbour.

    A sleeve fixes both. Leaves grow *along* a twig, so the splats are distributed over the
    twig's length with a bulge past its tip, which is where they are genuinely nearest to their
    own node; and each splat is a flat disc whose normal points out of the sleeve, so the
    silhouette has leaf-sized grain instead of being a smooth gaussian gradient.
    """
    if count <= 0:
        return
    axis, tangent, normal = _orthonormal_frame(tip - anchor)
    length = float(np.linalg.norm(tip - anchor))
    # Along the twig, weighted towards the tip, running a little past it.
    t = FOLIAGE_START + (FOLIAGE_END - FOLIAGE_START) * np.power(rng.random(count), FOLIAGE_TAPER)
    # Girth swells to the tip and falls away past it, so the sleeve ends in a tuft.
    girth = spread * np.sin(
        np.clip((t - FOLIAGE_START) / (FOLIAGE_END - FOLIAGE_START), 0, 1) * math.pi * 0.9 + 0.25
    )
    theta = rng.random(count) * (2.0 * math.pi)
    radial = girth * np.sqrt(rng.random(count))
    offset = np.outer(radial * np.cos(theta), tangent) + np.outer(radial * np.sin(theta), normal)
    offset[:, 2] *= FOLIAGE_FLATTEN
    position = anchor[None, :] + np.outer(t * length, axis) + offset

    # Sunlit at the top of the canopy, darker underneath — enough to read as a tree.
    lit = 0.66 + 0.56 * np.clip(position[:, 2:3] / max(height_m, 1e-6), 0.0, 1.4)
    # Hue wander as well as brightness: a canopy is not one colour with a gain on it.
    tint = 0.84 + 0.32 * rng.random((count, 3))
    rgb = np.clip(np.asarray(FOLIAGE_RGB)[None, :] * lit * tint, 0.0, 1.0)

    # Flat discs facing out of the sleeve: two broad axes spanning it, one thin along the normal.
    facing = offset / np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-9)
    leaf = spread * (FOLIAGE_LEAF_MIN + FOLIAGE_LEAF_SPAN * rng.random((count, 1)))
    quats = np.empty((count, 4))
    for i in range(count):
        e0, e1, e2 = _orthonormal_frame(facing[i])
        quats[i] = _matrix_to_quat_wxyz((e1, e2, e0))
    scale = leaf * np.array([1.0, 0.7, 0.16])[None, :] * (0.7 + 0.6 * rng.random((count, 3)))
    out.add(
        position,
        rgb,
        1.4 + 0.8 * rng.random(count),
        np.log(np.maximum(scale, 1e-4)),
        quats,
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
    # Inner canopy hangs where the leaf-bearing limbs fork, which is a topological fact about
    # any rig rather than a property of this generator's naming. A rig from skeleton.py, whose
    # bands are inferred and whose ids mean nothing, picks out the same nodes.
    has_leaf_child = {node["parent"] for i, node in enumerate(nodes) if bands[i] == "leaf"}
    inner = [i for i in sorted(has_leaf_child) if i >= 0 and bands[i] != "leaf"]

    wood_total = round(splats * WOOD_FRACTION)
    foliage_total = splats - wood_total
    tip_total = round(foliage_total * TIP_FRACTION)
    inner_total = foliage_total - tip_total

    # Bark budget follows surface area with a floor under it: area alone would spend almost
    # everything on the trunk now that a twig is 9 mm rather than 8 cm across, and a limb drawn
    # with three splats is a dotted line. The floor is the price of a slender tree.
    areas = np.asarray(
        [
            float(np.linalg.norm(positions[c] - positions[p])) * (radii[p] + radii[c])
            for p, c in wood_edges
        ]
    )
    weights = areas / max(float(areas.sum()), 1e-12)
    weights = np.maximum(weights, BARK_MIN_SHARE / max(len(wood_edges), 1))
    wood_counts = _allocate(weights, wood_total)
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
        segment = float(np.linalg.norm(positions[leaf] - positions[parent]))
        _foliage_splats(
            rng,
            out,
            positions[parent],
            positions[leaf],
            segment * TIP_FOLIAGE_GIRTH,
            int(count),
            leaf,
            height_m,
        )
    for branch, count in zip(inner, inner_counts, strict=True):
        parent = nodes[branch]["parent"]
        segment = float(np.linalg.norm(positions[branch] - positions[parent]))
        _foliage_splats(
            rng,
            out,
            positions[parent],
            positions[branch],
            segment * INNER_FOLIAGE_GIRTH,
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
    parser.add_argument("--splats", type=int, default=12000)
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
