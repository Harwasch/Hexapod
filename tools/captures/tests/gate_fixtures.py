"""Fill gates for tests, built from confidence planes the way the API builds them.

`still_weak_payload` is a test oracle for `rescan-gate-v1` (apps/api/app/analysis/
rescan_gate.py `still_weak`, and its endpoint's `FillTargetsRead`): B's own finest cells that
are weak and covered by B, where the newest earlier grid that covers the cell was weak there
too, packed rows south to north, least significant bit first, with the earlier grid each
target re-scanned (`satisfiedBy`). It is written cell by cell, not from the API's code. tests/test_rescan_gate.py holds it to the
API's own committed answer over the yard fixtures, so the tests that use it are gated the
way a real fill is.
"""

from __future__ import annotations

import base64
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import cell_grid

SITE = "00000000-0000-4000-8000-000000005173"
A = "00000000-0000-4000-8000-00000000000a"
B = "00000000-0000-4000-8000-00000000000b"


def confidence_grid(level: int, i0: int, j0: int, score: Any, reasons: Any) -> cell_grid.CellGrid:
    """A confidence grid whose finest plane is `score`/`reasons` (rows south to north),
    relief 0, with its min-of-children pyramid."""
    s = np.asarray(score, dtype=np.uint8)
    data = np.zeros(s.shape + (3,), dtype=np.uint8)
    data[..., 0] = s
    data[..., 1] = np.asarray(reasons, dtype=np.uint8)
    plane = cell_grid.CellPlane(level, i0, j0, s.shape[1], s.shape[0], data)
    return cell_grid.CellGrid(
        "confidence",
        cell_grid.CONFIDENCE_CHANNELS,
        tuple(cell_grid.min_pyramid(plane, 8)),
        {"policy": cell_grid.WEAK_POLICY, "weakBelow": cell_grid.WEAK_BELOW},
    )


def _seen(grid: cell_grid.CellGrid, level: int, i: int, j: int) -> tuple[bool, bool]:
    """`(covered, weak)` as `grid` saw cell `(i, j)` at `level`, from its finest plane: the
    cell holding it when that plane is no finer; else covered when any child is, and weak
    only when none of the children it saw was strong."""
    plane = grid.plane()
    weak, covered = cell_grid.weak_mask(plane), cell_grid.covered_mask(plane)

    def at(ci: int, cj: int) -> tuple[bool, bool]:
        di, dj = ci - plane.i0, cj - plane.j0
        if 0 <= di < plane.ni and 0 <= dj < plane.nj:
            return bool(covered[dj, di]), bool(weak[dj, di])
        return False, False

    if plane.level >= level:
        shift = plane.level - level
        return at(i >> shift, j >> shift)
    side = 1 << (level - plane.level)
    children = [
        at(ci, cj)
        for cj in range(j * side, (j + 1) * side)
        for ci in range(i * side, (i + 1) * side)
    ]
    seen = any(c for c, _ in children)
    strong = any(c and not w for c, w in children)
    return seen, seen and not strong


def still_weak_payload(
    b: cell_grid.CellGrid,
    earlier: Sequence[cell_grid.CellGrid],
    *,
    site: str = SITE,
    epoch: str = B,
    earlier_ids: Sequence[str] = (A,),
) -> dict[str, Any]:
    """The `FillTargetsRead` the API would answer for B over `earlier` (same order as
    `earlier_ids`, newest first as the API lists them)."""
    plane = b.plane()
    level = plane.level
    base = cell_grid.weak_mask(plane) & cell_grid.covered_mask(plane)
    hits = np.zeros(base.shape, dtype=bool)
    satisfied: list[int] = []
    for dj in range(plane.nj):
        for di in range(plane.ni):
            if not base[dj, di]:
                continue
            for index, grid in enumerate(earlier):
                covered, weak = _seen(grid, level, plane.i0 + di, plane.j0 + dj)
                if covered:  # the newest earlier epoch that reached the cell speaks for it
                    if weak:
                        hits[dj, di] = True
                        satisfied.append(index)
                    break
    if len(earlier) != len(earlier_ids):
        raise ValueError("one id per earlier grid")
    packed = (
        np.packbits(hits.astype(np.uint8), axis=1, bitorder="little").tobytes()
        if hits.size
        else b""
    )
    cell_m = cell_grid.Lattice(level).cell_m
    return {
        "policy": "rescan-gate-v1",
        "siteId": site,
        "epochId": epoch,
        "weakIn": [earlier_ids[k] for k in sorted(set(satisfied))],
        "comparedWith": list(earlier_ids),
        # The API compares with at most its eight most recent earlier epochs.
        "capped": False,
        "frame": "tileset",
        "cellM": cell_m,
        "level": level,
        "i0": plane.i0,
        "j0": plane.j0,
        "ni": plane.ni,
        "nj": plane.nj,
        "rowBytes": (plane.ni + 7) // 8,
        "bitOrder": "little",
        "mask": base64.b64encode(packed).decode("ascii"),
        "cells": int(hits.sum()),
        "satisfiedBy": base64.b64encode(bytes(satisfied)).decode("ascii"),
        "reason": (
            "weak in this epoch, and weak in the newest earlier measured epoch that covers "
            "them: a re-scan that did not fix them (rescan-gate-v1, at this epoch's own "
            f"{cell_m:g} m cells)"
        ),
    }


def write_payload(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def grid_json(folder: Path, grid: cell_grid.CellGrid) -> Path:
    """`confidence.json` and its `confidence.bin` in `folder`, as `splat_tiles.convert`'s
    `confidence=` takes them."""
    descriptor = cell_grid.encode(grid, folder, "confidence.bin")
    (folder / "confidence.json").write_text(json.dumps(descriptor, indent=1), encoding="utf-8")
    return folder / "confidence.json"


def bare_tileset(folder: Path, grid: cell_grid.CellGrid | None) -> Path:
    """A `tileset.json` that declares only `grid` (or nothing) on its root: all
    `RescanGate.check_tileset` reads."""
    folder.mkdir(parents=True, exist_ok=True)
    extras: dict[str, Any] = {}
    if grid is not None:
        extras["confidence"] = cell_grid.encode(grid, folder, "confidence.bin")
    path = folder / "tileset.json"
    path.write_text(json.dumps({"root": {"extras": extras}}), encoding="utf-8")
    return path


def with_confidence(tileset: Path, descriptor: dict[str, Any], blob: bytes, out: Path) -> Path:
    """A copy of `tileset`'s folder with `descriptor` appended last on its root extras and
    its file beside it -- what a confidence backfill writes."""
    shutil.copytree(tileset.parent, out)
    document = json.loads((out / tileset.name).read_text(encoding="utf-8"))
    document["root"].setdefault("extras", {})["confidence"] = descriptor
    (out / str(descriptor["uri"])).write_bytes(blob)
    (out / tileset.name).write_text(json.dumps(document, indent=1), encoding="utf-8")
    return out / tileset.name
