"""Fill gates for tests, built from confidence planes the way the API builds them.

`still_weak_payload` is a test oracle for `rescan-gate-v1` (apps/api/app/analysis/
rescan_gate.py `still_weak`, and its endpoint's `FillTargetsRead`): B's cells that are weak,
covered by B and weak in at least one earlier grid, at the coarsest of their levels, packed
rows south to north, least significant bit first. tests/test_rescan_gate.py holds it to the
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


def _placed(plane: cell_grid.CellPlane, window: cell_grid.CellPlane) -> np.ndarray:
    out = np.zeros((window.nj, window.ni), dtype=bool)
    weak = cell_grid.weak_mask(plane)
    for dj in range(window.nj):
        for di in range(window.ni):
            i, j = window.i0 + di, window.j0 + dj
            if plane.i0 <= i < plane.i0 + plane.ni and plane.j0 <= j < plane.j0 + plane.nj:
                out[dj, di] = weak[j - plane.j0, i - plane.i0]
    return out


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
    level = max(grid.level for grid in (b, *earlier))
    plane = cell_grid.plane_at(b, level)
    base = cell_grid.weak_mask(plane) & cell_grid.covered_mask(plane)
    hits = np.zeros(base.shape, dtype=bool)
    weak_in: list[str] = []
    for grid, ident in zip(earlier, earlier_ids, strict=True):
        overlap = base & _placed(cell_grid.plane_at(grid, level), plane)
        if overlap.any():
            weak_in.append(ident)
            hits |= overlap
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
        "weakIn": weak_in,
        "comparedWith": list(earlier_ids),
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
        "reason": (
            "weak in this epoch, which covers them, and weak in at least one earlier "
            "measured epoch: a re-scan that did not fix them (rescan-gate-v1, compared "
            f"at {cell_m:g} m cells)"
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
