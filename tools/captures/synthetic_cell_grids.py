"""Two epochs' confidence grids over the synthetic yard: the cross-language cell-grid fixture.

`data/tiles/synthetic-yard-epochs/` holds what every reader of `hexapod.cellgrid` (this
project's `cell_grid.py`, the API's decoder, the web's `lib/cellGrid.ts`) is checked
against: `e1/confidence.bin` and `e2/confidence.bin`, and `grids.json` -- each epoch's
descriptor (the value its tileset's `extras.confidence` would hold), the sha256 of each
level's decoded plane and of the file. The planes are made by integer arithmetic alone
over the cells of the committed yard tileset's root box (`data/tiles/synthetic-yard/
splat/tileset.json`, its tileset frame), so any machine regenerates the same bytes:

* **e1**, level 2 (1 m cells): a first scan from the south -- strong in the south half,
  weak in the north half for views or spread, a few cells there unseen;
* **e2**, level 1 (0.5 m cells, so the two nest at level 2): the re-scan, strong
  everywhere except a patch in the north-east still weak for GSD, aligned at +/-0.062 m.

Both carry the min-of-children pyramid (`cell_grid.min_pyramid`) and a summary. They are
fixtures, not measurements: no score here was measured from a capture.

    cd tools/captures && uv run python synthetic_cell_grids.py [--out DIR]

The default `--out` is the committed directory; `tests/test_synthetic_cell_grids.py`
regenerates into a temporary one and compares every byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import cell_grid

REPO = Path(__file__).resolve().parents[2]
YARD = REPO / "data" / "tiles" / "synthetic-yard" / "splat" / "tileset.json"
OUT = REPO / "data" / "tiles" / "synthetic-yard-epochs"
BITS = cell_grid.REASON_BITS


def yard_window(level: int, tileset: Path = YARD) -> tuple[int, int, int, int]:
    """`(i0, j0, ni, nj)`: the cells at `level` under the yard root box's footprint."""
    box = json.loads(tileset.read_text(encoding="utf-8"))["root"]["boundingVolume"]["box"]
    cx, cy, hx, hy = float(box[0]), float(box[1]), float(box[3]), float(box[7])
    corners = np.array([[cx - hx, cy - hy], [cx + hx, cy + hy]])
    i, j = cell_grid.Lattice(level).index(corners)
    return int(i[0]), int(j[0]), int(i[1] - i[0]) + 1, int(j[1] - j[0]) + 1


def _e1(i: np.ndarray, j: np.ndarray, middle: int) -> np.ndarray:
    data = np.zeros(i.shape + (3,), dtype=np.int64)
    north = j >= middle
    data[..., 0] = np.where(north, 40 + (i * 7 + j * 13) % 80, 140 + (i * 11 + j * 5) % 116)
    shy = np.where((i + j) % 2 == 0, BITS["views"], BITS["spread"])
    data[..., 1] = np.where(north, shy | np.where((i * 3 + j) % 5 == 0, BITS["gsd"], 0), 0)
    unseen = north & (i % 9 == 4) & (j % 7 == 3)
    data[..., 0] = np.where(unseen, 0, data[..., 0])
    data[..., 1] = np.where(unseen, BITS["unseen"], data[..., 1])
    data[..., 2] = (i * 3 + j) % 20
    return data


def _e2(i: np.ndarray, j: np.ndarray, corner: tuple[int, int]) -> np.ndarray:
    data = np.zeros(i.shape + (3,), dtype=np.int64)
    weak = (i >= corner[0]) & (j >= corner[1])
    data[..., 0] = np.where(weak, 30 + (i * 5 + j * 3) % 90, 150 + (i * 13 + j * 7) % 106)
    data[..., 1] = np.where(weak, BITS["gsd"], 0)
    data[..., 2] = (i + 2 * j) % 24
    return data


def _summary(plane: cell_grid.CellPlane) -> dict[str, Any]:
    score = plane.data[..., 0].astype(np.int64)
    reasons = plane.data[..., 1].astype(np.int64)
    weak = cell_grid.weak_mask(plane)
    unseen = (reasons & BITS["unseen"]) > 0
    scored = int((score > 0).sum())
    expected = scored + int(unseen.sum())
    return {
        "cells": plane.cells,
        "scored": scored,
        "weak": int(weak.sum()),
        "unseen": int(unseen.sum()),
        "weakShare": round(int(weak.sum()) / expected, 4) if expected else 0.0,
        "reasons": {name: int((weak & ((reasons & bit) > 0)).sum()) for name, bit in BITS.items()},
        "gaussians": 0,
        "outside": 0,
        "medianGsdMm": None,
        "metric": False,
    }


def grids(tileset: Path = YARD) -> dict[str, cell_grid.CellGrid]:
    """Both epochs' grids, from the yard's root box."""
    out: dict[str, cell_grid.CellGrid] = {}
    caps = {
        "e1": {"uncertaintyM": 0.0, "priorEpochs": 0, "factor": 1.0, "source": "site_context.json"},
        "e2": {
            "uncertaintyM": 0.062,
            "priorEpochs": 1,
            "factor": 1.0,
            "source": "epoch_alignment.json",
        },
    }
    for name, level in (("e1", 2), ("e2", 1)):
        i0, j0, ni, nj = yard_window(level, tileset)
        jj, ii = np.meshgrid(np.arange(j0, j0 + nj), np.arange(i0, i0 + ni), indexing="ij")
        if name == "e1":
            data = _e1(ii, jj, j0 + nj // 2)
        else:
            data = _e2(ii, jj, (i0 + (3 * ni) // 4, j0 + (3 * nj) // 4))
        plane = cell_grid.CellPlane(level, i0, j0, ni, nj, data.astype(np.uint8))
        meta: dict[str, Any] = {
            "reasonBits": dict(BITS),
            "policy": cell_grid.WEAK_POLICY,
            "weakBelow": cell_grid.WEAK_BELOW,
            "alignmentCap": caps[name],
            "roiDisc": None,
            "summary": _summary(plane),
            "fixture": "synthetic: integer arithmetic over the synthetic-yard root box, "
            "not a measurement (tools/captures/synthetic_cell_grids.py)",
        }
        out[name] = cell_grid.CellGrid(
            "confidence",
            cell_grid.CONFIDENCE_CHANNELS,
            tuple(cell_grid.min_pyramid(plane, 8)),
            meta,
        )
    return out


def _plane_sha256(plane: cell_grid.CellPlane, channels: tuple[cell_grid.Channel, ...]) -> str:
    record = np.dtype([(c.name, cell_grid.TYPES[c.type]) for c in channels])
    rows = np.zeros((plane.nj, plane.ni), dtype=record)
    for index, channel in enumerate(channels):
        rows[channel.name] = plane.data[..., index]
    return hashlib.sha256(rows.tobytes()).hexdigest()


def write(out: Path, tileset: Path = YARD) -> dict[str, Any]:
    """`e1/confidence.bin`, `e2/confidence.bin` and `grids.json` in `out`."""
    document: dict[str, Any] = {
        "note": "cross-language fixture for hexapod.cellgrid v1 (tools/captures/"
        "synthetic_cell_grids.py); synthetic, not measured",
        "frame": "data/tiles/synthetic-yard/splat/tileset.json (tileset frame)",
        "epochs": {},
    }
    for name, grid in grids(tileset).items():
        folder = out / name
        descriptor = cell_grid.encode(grid, folder, "confidence.bin")
        blob = (folder / "confidence.bin").read_bytes()
        document["epochs"][name] = {
            "descriptor": descriptor,
            "file": f"{name}/confidence.bin",
            "fileSha256": hashlib.sha256(blob).hexdigest(),
            "planes": [
                {"level": p.level, "sha256": _plane_sha256(p, grid.channels)} for p in grid.planes
            ],
        }
    (out / "grids.json").write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    document = write(args.out)
    for name, epoch in document["epochs"].items():
        summary = epoch["descriptor"]["summary"]
        print(f"{name}: level {epoch['descriptor']['lattice']['level']}, {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
