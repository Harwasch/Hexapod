"""Which cells a re-scan left weak: what an inferred fill may be asked to fill (M4, F4).

Policy `rescan-gate-v1` (ADR 0016 § API). A cell is a fill target in epoch B when it is:

* weak in B (weak-v1: scored under 128, or unseen),
* covered by B (scored, or expected and unseen -- B speaks for it), and
* weak in at least one earlier measured epoch A of the site,

compared at the coarsest of their levels, where every grid's cells nest (the site lattice,
C1): each grid is read there through its own min-of-children pyramid, so a coarse cell is
weak when any of its children was. A cell weak only in B has not been re-scanned yet, and a
cell weak in A that B saw well was fixed by B: neither is a target. Generation is never
asked to cover what nobody has tried to capture again.
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
    """The targets over B's window at `level`, and which earlier grids contributed."""

    level: int
    i0: int
    j0: int
    mask: NDArray[np.bool_]
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


def common_level(grids: Sequence[Grid]) -> int:
    """The coarsest finest level among `grids`: where all of them can be read."""
    return max(grid.finest.level for grid in grids)


def still_weak(b: Grid, earlier: Sequence[Grid]) -> StillWeak:
    """B's cells that are weak, covered and weak in at least one of `earlier` (module
    docstring). `contributed` indexes the earlier grids with at least one such cell."""
    level = common_level([b, *earlier])
    plane: Plane = b.at(level)
    base = weak_mask(plane) & covered_mask(plane)
    hits = np.zeros(base.shape, dtype=bool)
    contributed: list[int] = []
    for index, grid in enumerate(earlier):
        other = grid.at(level)
        placed = window(other, plane.i0, plane.j0, plane.ni, plane.nj, weak_mask(other))
        overlap = base & placed
        if overlap.any():
            contributed.append(index)
            hits |= overlap
    return StillWeak(level, plane.i0, plane.j0, hits, contributed)


def unpack(encoded: str, ni: int, nj: int) -> NDArray[np.bool_]:
    """`StillWeak.packed` read back: what a consumer of the targets does."""
    row_bytes = (ni + 7) // 8
    data = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).reshape(nj, row_bytes)
    bits = np.unpackbits(data, axis=1, bitorder="little")[:, :ni]
    out: NDArray[np.bool_] = bits.astype(bool)
    return out
