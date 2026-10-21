"""The viewer's cut through a tileset, in numpy: CesiumJS's REPLACE traversal on
screen-space error, as `experiments/lod_compare.py` draws it.

A tile is refined -- replaced by all of its children, each of which decides again -- when
its geometric error projected to the screen, error * focal / distance (pixels), exceeds the
viewer's maximumScreenSpaceError tau; distance is from the eye to the tile's bounding box,
so an eye inside the box sees infinite error and always refines. This is the rule
CesiumJS's 3D Tiles traversal applies (Apache-2.0), written from its documented behaviour.

CPython 3.10 and numpy only, so it runs under a trainer's interpreter as well as here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Tree:
    """A tileset's tree, tile 0 the root, tiles in depth-first (walk) order.

    `low` / `high` are each tile's axis-aligned box corners, `error` its geometric error,
    `children` its child indices, and `rows` its half-open run [start, stop) in one table
    holding every tile's gaussians: all leaves first, then all parents, each in walk order.
    """

    uris: tuple[str, ...]
    low: F64
    high: F64
    error: F64
    children: tuple[tuple[int, ...], ...]
    rows: I64

    @property
    def parents(self) -> tuple[int, ...]:
        """The tiles that have children, in walk order."""
        return tuple(t for t, kids in enumerate(self.children) if kids)

    @property
    def leaves(self) -> tuple[int, ...]:
        """The tiles that have none, in walk order."""
        return tuple(t for t, kids in enumerate(self.children) if not kids)


def tree_from(tiles: Sequence[Any], counts: Sequence[int]) -> Tree:
    """The `Tree` of `tiles`, given in walk order (root first; each has `.uri`, `.low`,
    `.high`, `.geometric_error` and `.children`), tile t holding `counts[t]` gaussians."""
    index = {id(tile): t for t, tile in enumerate(tiles)}
    children = tuple(tuple(index[id(child)] for child in tile.children) for tile in tiles)
    n = len(tiles)
    low = np.array([np.asarray(tile.low, dtype=np.float64).reshape(3) for tile in tiles])
    high = np.array([np.asarray(tile.high, dtype=np.float64).reshape(3) for tile in tiles])
    rows = np.zeros((n, 2), dtype=np.int64)
    start = 0
    leaves = [t for t in range(n) if not children[t]]
    parents = [t for t in range(n) if children[t]]
    for t in leaves + parents:
        rows[t] = (start, start + int(counts[t]))
        start += int(counts[t])
    return Tree(
        uris=tuple(str(tile.uri) for tile in tiles),
        low=low.reshape(n, 3),
        high=high.reshape(n, 3),
        error=np.array([float(tile.geometric_error) for tile in tiles], dtype=np.float64),
        children=children,
        rows=rows,
    )


def box_distance(eye: npt.ArrayLike, low: npt.ArrayLike, high: npt.ArrayLike) -> float:
    """Euclidean distance from `eye` to the box [low, high]; 0 inside it."""
    e = np.asarray(eye, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    gap = np.maximum(np.maximum(lo - e, e - hi), 0.0)
    return float(np.linalg.norm(gap))


def screen_space_error(tree: Tree, t: int, eye: npt.ArrayLike, focal: float) -> float:
    """Tile t's geometric error in pixels from `eye`: error * focal / distance, or +inf when
    the eye is in (or on) its box."""
    d = box_distance(eye, tree.low[t], tree.high[t])
    if d <= 0.0:
        return float("inf")
    return float(tree.error[t]) * float(focal) / d


def cut(tree: Tree, eye: npt.ArrayLike, focal: float, tau: float) -> list[int]:
    """The tiles a REPLACE traversal draws from `eye`, in visit order (depth first from the
    root): a tile with children whose screen-space error exceeds `tau` is replaced by its
    children; every other tile is drawn."""
    chosen: list[int] = []
    stack = [0]
    while stack:
        t = stack.pop()
        kids = tree.children[t]
        if kids and screen_space_error(tree, t, eye, focal) > tau:
            stack.extend(reversed(kids))  # so the first child is visited first
        else:
            chosen.append(t)
    return chosen


def orbit_eye(
    target: npt.ArrayLike,
    direction: npt.ArrayLike,
    low: npt.ArrayLike,
    high: npt.ArrayLike,
    distance: float,
) -> F64:
    """The point along `direction` from `target` whose distance to the box [low, high] is
    `distance`, by bisection (80 halvings) on the ray parameter."""
    origin = np.asarray(target, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    lo_box = np.asarray(low, dtype=np.float64)
    hi_box = np.asarray(high, dtype=np.float64)
    t_lo = 0.0
    t_hi = max(1.0, 4.0 * float(distance) + 4.0 * float(np.linalg.norm(hi_box - lo_box)))
    for _ in range(80):
        mid = 0.5 * (t_lo + t_hi)
        if box_distance(origin + mid * d, lo_box, hi_box) < distance:
            t_lo = mid
        else:
            t_hi = mid
    out: F64 = origin + t_hi * d
    return out


def look_at(eye: npt.ArrayLike, target: npt.ArrayLike, up: npt.ArrayLike = (0.0, 0.0, 1.0)) -> F64:
    """The 4x4 world-to-camera matrix of a camera at `eye` looking at `target`, in OpenCV's
    axes (x right, y down, z forward)."""
    e = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - e
    forward = forward / np.linalg.norm(forward)
    u = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, u)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    view: F64 = np.eye(4)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ e
    return view
