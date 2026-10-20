"""Which cells a re-scan left weak: what an inferred fill may be asked to fill (M4, F4).

Policy `rescan-gate-v1` (ADR 0016 § API). The targets are cells of epoch B's **finest**
level -- the finest cells the re-scan measured -- and a cell is one when:

* it is weak in B (weak-v1: scored under 128, or unseen), so B covers it, and
* the newest earlier measured epoch whose footprint covers it was weak there too.

An earlier epoch counts for a cell only when it covers that cell (scored, or expected and
unseen): one that never reached it says nothing, and the next older one speaks instead (C7:
the re-scan of a cell is a later epoch that covers it). One that covered it well fixed it,
so B's weakness there is new, not a re-scan's, and the cell is not a target. Each target
records which earlier epoch it re-scanned (`satisfied`).

Levels nest (the site lattice, C1), so an earlier grid is read at B's level without
resampling. A coarser one: the cell of it that holds B's cell. A finer one: the B cell is
covered when any of its children is, and weak only when none of the children it saw was
strong -- part of the cell seen well is not a weak cell, and the gate never widens one weak
child to its parent. Generation is never asked to cover what nobody tried to capture again,
nor anything the re-scan itself saw well.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.analysis.cell_grid import Grid, Plane, covered_mask, weak_mask, window

POLICY = "rescan-gate-v1"


@dataclass(frozen=True)
class StillWeak:
    """The targets over B's finest window at `level`; per cell, the index of the earlier
    grid each target re-scanned (`satisfied`, -1 off the targets); which grids those are."""

    level: int
    i0: int
    j0: int
    mask: NDArray[np.bool_]
    satisfied: NDArray[np.int8]
    contributed: list[int]

    @property
    def ni(self) -> int:
        return int(self.mask.shape[1])

    @property
    def nj(self) -> int:
        return int(self.mask.shape[0])

    @property
    def cells(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def row_bytes(self) -> int:
        return (self.ni + 7) // 8

    def packed(self) -> str:
        """Base64 of the rows, south to north, each `row_bytes` bytes, least significant
        bit first: bit `k` of byte `b` in row `r` is cell `(i0 + 8b + k, j0 + r)`."""
        if self.mask.size == 0:
            return ""
        bits = np.packbits(self.mask.astype(np.uint8), axis=1, bitorder="little")
        return base64.b64encode(bits.tobytes()).decode("ascii")

    def satisfied_packed(self) -> str:
        """Base64 of one byte per target, in the mask's order (rows south to north, `i`
        ascending): the index, into the earlier epochs compared with, of the one whose weak
        cell it re-scanned."""
        return base64.b64encode(self.satisfied[self.mask].astype(np.uint8).tobytes()).decode(
            "ascii"
        )


def _any_up(
    mask: NDArray[np.bool_], i0: int, j0: int, shift: int
) -> tuple[NDArray[np.bool_], int, int]:
    """`mask` (a window at `i0, j0`) read `shift` levels coarser: a cell is True when any
    of its children is. Its window's origin with it."""
    if mask.size == 0:
        return mask, i0 >> shift, j0 >> shift
    nj, ni = mask.shape
    pi0, pj0 = i0 >> shift, j0 >> shift
    pi1, pj1 = (i0 + ni - 1) >> shift, (j0 + nj - 1) >> shift
    side = 1 << shift
    padded = np.zeros(((pj1 - pj0 + 1) * side, (pi1 - pi0 + 1) * side), dtype=bool)
    di, dj = i0 - pi0 * side, j0 - pj0 * side
    padded[dj : dj + nj, di : di + ni] = mask
    out: NDArray[np.bool_] = padded.reshape(pj1 - pj0 + 1, side, pi1 - pi0 + 1, side).any(
        axis=(1, 3)
    )
    return out, pi0, pj0


def seen_at(grid: Grid, target: Plane) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Over `target`'s window and level, `(covered, weak)` as `grid` saw each cell (module
    docstring): from its finest level, held or aggregated, never resampled."""
    plane = grid.finest
    covered, weak = covered_mask(plane), weak_mask(plane)
    if plane.level >= target.level:
        shift = plane.level - target.level
        ii = (np.arange(target.i0, target.i0 + target.ni, dtype=np.int64) >> shift) - plane.i0
        jj = (np.arange(target.j0, target.j0 + target.nj, dtype=np.int64) >> shift) - plane.j0
        inside = np.outer((jj >= 0) & (jj < plane.nj), (ii >= 0) & (ii < plane.ni))
        rows = np.clip(jj, 0, max(plane.nj - 1, 0))
        cols = np.clip(ii, 0, max(plane.ni - 1, 0))
        if plane.cells == 0:
            empty = np.zeros((target.nj, target.ni), dtype=bool)
            return empty, empty.copy()
        held_covered: NDArray[np.bool_] = covered[np.ix_(rows, cols)] & inside
        held_weak: NDArray[np.bool_] = weak[np.ix_(rows, cols)] & inside
        return held_covered, held_weak
    shift = target.level - plane.level
    strong = covered & ~weak
    any_covered, i0, j0 = _any_up(covered, plane.i0, plane.j0, shift)
    any_strong, _, _ = _any_up(strong, plane.i0, plane.j0, shift)
    up = Plane(target.level, i0, j0, any_covered.shape[1], any_covered.shape[0], {})
    placed_covered = window(up, target.i0, target.j0, target.ni, target.nj, any_covered)
    placed_strong = window(up, target.i0, target.j0, target.ni, target.nj, any_strong)
    return placed_covered, placed_covered & ~placed_strong


def still_weak(b: Grid, earlier: Sequence[Grid]) -> StillWeak:
    """B's finest cells that are weak and whose newest covering earlier grid was weak too
    (module docstring). `earlier` is newest first; `satisfied` and `contributed` index it."""
    plane: Plane = b.finest
    undecided = weak_mask(plane) & covered_mask(plane)
    hits = np.zeros(undecided.shape, dtype=bool)
    satisfied = np.full(undecided.shape, -1, dtype=np.int8)
    contributed: list[int] = []
    for index, grid in enumerate(earlier):
        if not undecided.any():
            break
        covered, weak = seen_at(grid, plane)
        found = undecided & covered & weak
        if found.any():
            contributed.append(index)
            hits |= found
            satisfied[found] = index
        undecided &= ~covered
    return StillWeak(plane.level, plane.i0, plane.j0, hits, satisfied, contributed)


def unpack(encoded: str, ni: int, nj: int) -> NDArray[np.bool_]:
    """`StillWeak.packed` read back: what a consumer of the targets does."""
    row_bytes = (ni + 7) // 8
    data = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).reshape(nj, row_bytes)
    bits = np.unpackbits(data, axis=1, bitorder="little")[:, :ni]
    out: NDArray[np.bool_] = bits.astype(bool)
    return out


def unpack_satisfied(encoded: str, cells: int) -> list[int]:
    """`StillWeak.satisfied_packed` read back: one index per target, in mask order."""
    data = base64.b64decode(encoded)
    if len(data) != cells:
        raise ValueError(f"{len(data)} indices for {cells} targets")
    return list(data)
