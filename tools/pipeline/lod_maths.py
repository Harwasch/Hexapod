"""The numpy half of optimising a tileset's parent gaussians (`lod_optimise.py`).

Everything here is decided on the CPU and is deterministic given its inputs, so it is
unit-tested without a GPU (tests/test_lod_optimise.py) and imported unchanged by the torch
half, which runs under the trainer's CPython 3.10 -- so nothing here may need more than
numpy and 3.10's syntax.

**The cut is the viewer's.** Hierarchical 3DGS (2406.12080, Sec. 4.2) chooses, for a
target granularity tau, the nodes whose projected size is under tau and whose parent's is
not. Our tree is 3D Tiles, and the viewer that draws it -- CesiumJS 1.145, REPLACE
refinement, `skipLevelOfDetail` off -- refines a tile when its screen-space error,
`geometricError * f / distance` with `distance` from the camera to the tile's bounding box
(Cesium3DTile.getScreenSpaceError; the same rule scratchpad lodcmp/compare.py measured
Phase 1 by), exceeds `maximumScreenSpaceError`. So `cut` is that traversal, with tau as
`maximumScreenSpaceError` in the training frame's pixels: a parent is optimised in exactly
the views and at exactly the errors the viewer will draw it in. H3DGS's own granularity
(the node's AABB size over its distance, gaussian-hierarchy runtime_switching.cu
`computeSizeGPU`) is a node-size rule for a binary BVH; ours is the rule of our viewer.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

#: H3DGS's post-optimisation length: `train_post.py --iterations 15000`
#: (scripts/full_train.py, `post_opt_chunk_args`).
H3DGS_ITERATIONS = 15_000

#: The range of target granularities H3DGS samples from in train_post.py: `limmin = 0.005`,
#: `limmax = 0.1` -- a ratio of 20, sampled log-uniformly (Sec. 5.1).
H3DGS_TAU_RATIO = 0.1 / 0.005

#: How many times H3DGS optimises each interior node, in expectation, before visibility:
#: its compacted hierarchy (Sec. 5.2) keeps nodes at granularities a factor of 2 apart
#: ("we then raise the target granularity by 2x and repeat"), so a node is the cut's
#: choice for a band of tau one factor of 2 wide: ln 2 / ln 20 of the log-uniform samples,
#: 23%, of 15,000 iterations -- about 3,470. `plan_iterations` gives each of our parents
#: the same number.
H3DGS_SELECTIONS = H3DGS_ITERATIONS * math.log(2.0) / math.log(H3DGS_TAU_RATIO)

#: H3DGS's learning rates for the post-optimisation, and where each is set. The defaults
#: are arguments/__init__.py `OptimizationParams`; full_train.py overrides three of them
#: for train_post.py (`--feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001`).
POSITION_LR_INIT = 0.00002  # position_lr_init, times the scene radius
POSITION_LR_FINAL = 0.0000002  # position_lr_final, times the scene radius
FEATURE_LR = 0.0005  # full_train.py --feature_lr (DC colour; f_rest would be /20)
OPACITY_LR = 0.01  # full_train.py --opacity_lr, on the raw opacity (abs activation)
SCALING_LR = 0.001  # full_train.py --scaling_lr, on log-scales
ROTATION_LR = 0.001  # rotation_lr, not overridden
ADAM_EPS = 1e-15  # gaussian_model.py training_setup: Adam(..., eps=1e-15)
#: The loss is the trainer's: (1 - 0.2) L1 + 0.2 (1 - SSIM) (arguments `lambda_dssim`,
#: and gsplat's simple_trainer `ssim_lambda`).
SSIM_LAMBDA = 0.2


@dataclass(frozen=True)
class Tree:
    """A tileset's tiles as arrays, in `walk` order (parents before children).

    `rows[t]` is the half-open run of tile `t`'s gaussians in one table where every leaf's
    gaussians come first, then every parent's -- the layout the optimiser keeps them in
    (leaves frozen, parents trainable).
    """

    uris: tuple[str, ...]
    low: F64
    high: F64
    error: F64
    children: tuple[tuple[int, ...], ...]
    rows: I64

    @property
    def parents(self) -> tuple[int, ...]:
        return tuple(t for t, kids in enumerate(self.children) if kids)

    @property
    def leaves(self) -> tuple[int, ...]:
        return tuple(t for t, kids in enumerate(self.children) if not kids)


def tree_from(tiles: Sequence[Any], counts: Sequence[int]) -> Tree:
    """`splat_tiles.Tile`s (root first, `Tile.walk()` order) as a `Tree`.

    `counts[t]` is how many gaussians tile `t` draws. Leaves are laid out first in walk
    order, then parents in walk order.
    """
    index = {id(tile): t for t, tile in enumerate(tiles)}
    children = tuple(tuple(index[id(child)] for child in tile.children) for tile in tiles)
    rows = np.zeros((len(tiles), 2), dtype=np.int64)
    at = 0
    for leaf_pass in (True, False):
        for t, kids in enumerate(children):
            if (not kids) == leaf_pass:
                rows[t] = (at, at + int(counts[t]))
                at += int(counts[t])
    return Tree(
        uris=tuple(tile.uri for tile in tiles),
        low=np.array([np.asarray(tile.low, dtype=np.float64) for tile in tiles]).reshape(-1, 3),
        high=np.array([np.asarray(tile.high, dtype=np.float64) for tile in tiles]).reshape(-1, 3),
        error=np.array([float(tile.geometric_error) for tile in tiles], dtype=np.float64),
        children=children,
        rows=rows,
    )


def box_distance(eye: F64, low: F64, high: F64) -> float:
    """Distance from `eye` to the axis-aligned box, 0 inside (Cesium's box distance)."""
    outside = np.maximum(np.maximum(low - eye, eye - high), 0.0)
    return float(np.linalg.norm(outside))


def screen_space_error(tree: Tree, tile: int, eye: F64, focal: float) -> float:
    """Cesium3DTile.getScreenSpaceError for a perspective camera: `geometricError * f / d`
    with `d` the distance to the tile's box (a camera inside the box is infinite error)."""
    distance = box_distance(eye, tree.low[tile], tree.high[tile])
    if distance <= 0.0:
        return math.inf
    return float(tree.error[tile]) * focal / distance


def cut(tree: Tree, eye: F64, focal: float, tau: float) -> list[int]:
    """The tiles CesiumJS draws at `maximumScreenSpaceError` tau, all loaded: REPLACE
    refinement (Cesium3DTilesetBaseTraversal) -- a tile whose error projects over tau is
    replaced by all its children, each of which decides for itself."""
    chosen: list[int] = []
    stack = [0]
    while stack:
        tile = stack.pop()
        kids = tree.children[tile]
        if kids and screen_space_error(tree, tile, eye, focal) > tau:
            stack.extend(reversed(kids))
        else:
            chosen.append(tile)
    return chosen


def camera_centre(viewmat: F64) -> F64:
    """The eye of a world-to-camera 4x4."""
    rotation, translation = viewmat[:3, :3], viewmat[:3, 3]
    return np.asarray(-rotation.T @ translation, dtype=np.float64)


def in_view(low: F64, high: F64, viewmat: F64, k: F64, width: int, height: int) -> bool:
    """Whether the box can put anything on screen: not wholly behind the camera and not
    wholly beyond one side of the image (the usual all-corners-outside-one-plane test,
    conservative -- a box it keeps may still miss the frustum at a corner)."""
    corners = np.array(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
    )
    cam = corners @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = cam[:, 2]
    if np.all(z <= 1e-6):
        return False
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    # Each side plane in camera space: u = fx x / z + cx in [0, width] <=> the signs below.
    tests = (
        fx * cam[:, 0] + (cx - 0.0) * z,  # u >= 0
        (width - cx) * z - fx * cam[:, 0],  # u <= width
        fy * cam[:, 1] + (cy - 0.0) * z,  # v >= 0
        (height - cy) * z - fy * cam[:, 1],  # v <= height
    )
    return not any(bool(np.all(test < 0)) for test in tests)


def visible_parents(
    tree: Tree, chosen: Iterable[int], viewmat: F64, k: F64, width: int, height: int
) -> list[int]:
    """The parent tiles of a cut whose boxes are in view: the ones a render can move."""
    return [
        t
        for t in chosen
        if tree.children[t] and in_view(tree.low[t], tree.high[t], viewmat, k, width, height)
    ]


def sample_tau(rng: np.random.Generator, tau_min: float, tau_max: float) -> float:
    """H3DGS Sec. 5.1: tau = tau_max^xi * tau_min^(1 - xi), xi uniform in [0, 1)."""
    xi = float(rng.random())
    return float(tau_max**xi * tau_min ** (1.0 - xi))


@dataclass(frozen=True)
class Camera:
    """One frame as the optimiser sees it: world-to-camera in the placed frame, K, size."""

    viewmat: F64
    k: F64
    width: int
    height: int

    @property
    def focal(self) -> float:
        # Cesium's sseDenominator is 2 tan(fovy / 2), so its pixels are the vertical focal.
        return float(self.k[1, 1])

    @property
    def eye(self) -> F64:
        return camera_centre(self.viewmat)


@dataclass(frozen=True)
class Plan:
    """How long to optimise, and the evidence for it."""

    iterations: int
    selection_rate: float
    parents_seen: int
    reason: str


def plan_iterations(
    tree: Tree,
    cameras: Sequence[Camera],
    tau_min: float,
    tau_max: float,
    *,
    most: int = H3DGS_ITERATIONS,
    least: int = 1000,
    taus: int = 24,
) -> Plan:
    """Iterations enough that each parent is optimised about as often as H3DGS's nodes are.

    H3DGS runs 15,000 iterations over a compacted hierarchy whose nodes sit a factor of 2
    apart in granularity, so each node is chosen in ln 2 / ln 20 of the samples
    (`H3DGS_SELECTIONS`, ~3,470 times). Our parents sit at least an octree level -- a
    factor of 2 -- apart too, but the root's band is open-ended above and a tree of a few
    tiles has few levels, so a parent here is usually chosen far more often than 23% of
    the time and needs fewer iterations to be chosen as often. The rate is measured, not
    assumed: over every training camera and a log-spaced grid of tau, the share of
    (camera, tau) pairs the optimiser would actually train on -- a cut with a parent in
    view, since the others are skipped -- in which each parent is in the cut and in view,
    averaged over parents weighted by how many gaussians each holds. Clamped to
    [`least`, `most`]: never more than H3DGS's own 15,000.
    """
    parents = tree.parents
    if not parents:
        return Plan(0, 0.0, 0, "the tileset has no parent tiles")
    grid = np.exp(np.linspace(math.log(tau_min), math.log(tau_max), taus))
    hits = np.zeros(len(tree.uris), dtype=np.float64)
    useful = 0
    for camera in cameras:
        eye = camera.eye
        for tau in grid:
            seen = visible_parents(
                tree, cut(tree, eye, camera.focal, float(tau)), camera.viewmat, camera.k,
                camera.width, camera.height,
            )  # fmt: skip
            if seen:
                useful += 1
                hits[seen] += 1.0
    if useful == 0:
        return Plan(0, 0.0, 0, "no training camera sees a parent tile in any cut")
    weights = np.array([tree.rows[t, 1] - tree.rows[t, 0] for t in parents], dtype=np.float64)
    rates = hits[list(parents)] / useful
    rate = float(np.sum(rates * weights) / max(float(weights.sum()), 1.0))
    seen_count = int(np.count_nonzero(rates))
    wanted = math.ceil(H3DGS_SELECTIONS / max(rate, 1e-9))
    iterations = int(min(max(wanted, least), most))
    return Plan(
        iterations,
        rate,
        seen_count,
        f"each parent is chosen in {rate:.0%} of trained samples; {H3DGS_SELECTIONS:.0f} "
        f"choices (H3DGS: 15,000 x ln 2 / ln 20) need {wanted}, clamped to "
        f"[{least}, {most}]",
    )


def spatial_scale(eyes: F64) -> float:
    """H3DGS's `spatial_lr_scale` (scene/dataset_readers.py getNerfppNorm): 1.1 times the
    90th percentile of the cameras' distances from their mean."""
    eyes = np.asarray(eyes, dtype=np.float64).reshape(-1, 3)
    distances = np.linalg.norm(eyes - eyes.mean(axis=0), axis=1)
    return max(1.1 * float(np.quantile(distances, 0.9)), 1e-6)


def position_lr(step: int, iterations: int, scale: float) -> float:
    """H3DGS's position schedule (utils/general_utils.py get_expon_lr_func): log-linear
    from `POSITION_LR_INIT` to `POSITION_LR_FINAL`, times the scene radius, over
    `position_lr_max_steps` -- 30,000, of which train_post.py runs 15,000, so it ends at
    the schedule's geometric middle. Kept as that proportion for any run length: the
    schedule spans twice the iterations run."""
    t = min(max(step / max(2 * iterations, 1), 0.0), 1.0)
    return float(
        math.exp(math.log(POSITION_LR_INIT) * (1 - t) + math.log(POSITION_LR_FINAL) * t) * scale
    )


def placed_camtoworld(camtoworld: F64, rotation: F64, translation: F64, scale: float) -> F64:
    """A COLMAP-frame camera-to-world into the placed (canonical.ply) frame.

    `place` moves every point by `x' = scale * R x + t` (`placement.json`). A camera moves
    with it: its centre like any point, its axes by R. The 1/scale that relates camera
    coordinates before and after does not change a projection, so the placed camera sees
    the placed splat exactly as the original saw the original.
    """
    out = np.eye(4)
    out[:3, :3] = rotation @ camtoworld[:3, :3]
    out[:3, 3] = scale * (rotation @ camtoworld[:3, 3]) + translation
    return out


def orbit_eye(target: F64, direction: F64, low: F64, high: F64, distance: float) -> F64:
    """The point along `direction` from `target` whose distance to the box is `distance`
    (the Phase 1 switch-distance protocol, scratchpad lodcmp/compare.py): bisection, since
    box distance grows monotonically along a ray leaving the box."""
    direction = direction / np.linalg.norm(direction)
    lo, hi = 0.0, max(1.0, distance * 4 + float(np.linalg.norm(high - low)) * 4)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if box_distance(target + mid * direction, low, high) < distance:
            lo = mid
        else:
            hi = mid
    return np.asarray(target + hi * direction, dtype=np.float64)


def look_at(eye: F64, target: F64, up: Sequence[float] = (0.0, 0.0, 1.0)) -> F64:
    """World-to-camera 4x4 looking from `eye` at `target`, OpenCV axes (x right, y down)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = -rotation @ eye
    return out
