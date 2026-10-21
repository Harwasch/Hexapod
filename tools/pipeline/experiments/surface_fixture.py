"""Two epochs' height surfaces over the synthetic yard: the cross-language surface fixture.

`data/tiles/synthetic-yard-surfaces/` holds what the API's reader of `surface` cell grids
(apps/api `app/analysis/cell_grid.py`, `app/analysis/change.py`) is checked against: P5's
own encoder's bytes (`cell_grid.pack`, tools/captures) for `e1/surface.bin` and
`e2/surface.bin`, and `surfaces.json` -- each epoch's descriptor (the value its tileset's
`extras.surface` would hold), the sha256 of its plane and of its file, e2's plane brought
to e1's level by P5's demo `coarsen_surface` (its sha256), and the change between them as
P5's demo `surface_change` measures it (counts and every changed cell), so the API's own
resampling and classification are held to the reference the change map was specified by.

Each surface is on the very plane of the matching confidence grid of
`data/tiles/synthetic-yard-epochs` (e1 level 2, 1 m cells; e2 level 1, 0.5 m cells), as
the `surface` stage writes it on its epoch's confidence plane. Heights are whole
centimetres from integer arithmetic over the cells, so any machine writes the same bytes:

* a ground sloping gently east and north, the same in both epochs;
* a hedge 1.8 m tall standing in both (sigma 30 cm: foliage), which must read unchanged;
* in e2 a shed 2.5 m tall in the south-west, where both epochs saw well: raised;
* in e1 a spoil heap 1.8 m tall in the south-east that e2 no longer has: lowered;
* in e2 a 1.6 m stack in the north, where e1's confidence is weak: raised, low confidence;
* in e2 a pond dug 1 m deep over 10 x 10 m in the east, where both saw well, ground and
  top alike: the ground lowered (wider than P5's ground window, so its envelope follows);
* the shed, spoil and stack stand on the ground, which does not move under them: they
  are change of the top surface only (the change map's vegetation layer, whose own
  threshold is about 1.1 m here: why they are taller than a metre);
* a few centimetres of height "noise" each, different per epoch: unchanged;
* the ground's own sigma (`groundSigma`) 5 cm in e1 and 6 cm in e2;
* e1 has no surface where its confidence grid says unseen and e2 none in a strip along
  the west edge (insufficient where the other side has one), and one e2 cell's sigma is
  saturated (255 cm).

Fixtures, not measurements: no height here was measured from a capture.

    cd tools/pipeline && uv run --no-sync python -m experiments.surface_fixture [--out DIR]

The default `--out` is the committed directory; `tests/test_surface_fixture.py` regenerates
into a temporary one and compares every byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import surface as p5  # noqa: E402
from captures_bridge import cell_grid  # noqa: E402
from experiments.confidence_demo import coarsen_surface, surface_change  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
YARD = REPO / "data" / "tiles" / "synthetic-yard" / "splat" / "tileset.json"
EPOCHS = REPO / "data" / "tiles" / "synthetic-yard-epochs"
OUT = REPO / "data" / "tiles" / "synthetic-yard-surfaces"
NO = cell_grid.NO_HEIGHT
Box = tuple[int, int, int, int]
#: (x0, y0, x1, y1) metres in the yard's tileset frame, and the height above the ground (cm).
HEDGE: tuple[Box, int] = ((12, 0, 14, 10), 180)
SHED: tuple[Box, int] = ((2, 2, 8, 6), 250)
SPOIL: tuple[Box, int] = ((20, 3, 26, 7), 180)
STACK: tuple[Box, int] = ((4, 16, 10, 20), 160)
#: e2's pond: the ground itself, and the top with it, this much lower (cm).
POND: tuple[Box, int] = ((26, 1, 36, 11), 100)
#: P5's ground window at each epoch's level (cells a side): 5 m wide, centres within 2.5 m.
GROUND_WINDOW_CELLS = {2: 5, 1: 11}
#: e2 has no surface west of this (metres east).
E2_WEST_EDGE_M = -2
#: The e2 cell whose sigma saturates: the south-west corner of it, metres east and north.
SATURATED = (30.0, 2.0)
#: Both epochs' support: SfM points and splat centres, 2^4 samples.
SUPPORT = 3 | (4 << 4)
#: Each epoch's ground sigma (cm): a lower envelope's, larger than its top's.
GROUND_SIGMA = {"e1": 5, "e2": 6}


def origin_height(tileset: Path = YARD) -> float:
    """The ellipsoidal height of the yard's tileset frame origin (its root transform's
    translation), metres to the millimetre: Bowring's closed form on WGS84."""
    t = json.loads(tileset.read_text(encoding="utf-8"))["root"]["transform"]
    x, y, z = float(t[12]), float(t[13]), float(t[14])
    a, f = 6378137.0, 1 / 298.257223563
    b = a * (1 - f)
    e2, ep2 = f * (2 - f), (a * a - b * b) / (b * b)
    p = math.hypot(x, y)
    theta = math.atan2(z * a, p * b)
    lat = math.atan2(z + ep2 * b * math.sin(theta) ** 3, p - e2 * a * math.cos(theta) ** 3)
    n = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return round(p / math.cos(lat) - n, 3)


def confidence(epoch: str) -> tuple[dict[str, Any], Any]:
    """The epoch's committed confidence grid (synthetic-yard-epochs): descriptor, grid."""
    document = json.loads((EPOCHS / "grids.json").read_text(encoding="utf-8"))
    descriptor: dict[str, Any] = document["epochs"][epoch]["descriptor"]
    blob = (EPOCHS / epoch / "confidence.bin").read_bytes()
    return descriptor, cell_grid.decode(descriptor, blob)


def _inside(x: np.ndarray, y: np.ndarray, box: Box) -> np.ndarray:
    x0, y0, x1, y1 = box
    inside: np.ndarray = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    return inside


def surface_plane(epoch: str, graded: Any) -> Any:
    """The epoch's surface on its confidence grid's finest plane (module docstring)."""
    plane = graded.planes[0]
    cell_cm = 25 << plane.level
    jj, ii = np.meshgrid(
        np.arange(plane.j0, plane.j0 + plane.nj, dtype=np.int64),
        np.arange(plane.i0, plane.i0 + plane.ni, dtype=np.int64),
        indexing="ij",
    )
    # Cell centres in whole centimetres (cells are 50 cm or more here); corners in metres.
    xc, yc = (2 * ii + 1) * cell_cm // 2, (2 * jj + 1) * cell_cm // 2
    x, y = ii * cell_cm / 100.0, jj * cell_cm / 100.0
    ground = 1000 + xc // 20 + yc // 40
    if epoch == "e2":
        box, deep = POND
        ground = np.where(_inside(x, y, box), ground - deep, ground)
    if epoch == "e1":
        noise = (ii * 7 + jj * 3) % 5 - 2
        sigma = 2 + (ii + jj) % 3
    else:
        noise = (ii * 5 + jj * 11) % 5 - 2
        sigma = 2 + (2 * ii + jj) % 3
    box, tall = HEDGE
    hedge = _inside(x, y, box)
    dsm = np.where(hedge, ground + tall, ground) + noise
    sigma = np.where(hedge, 30, sigma)
    if epoch == "e2":
        for box, tall in (SHED, STACK):
            dsm = np.where(_inside(x, y, box), ground + tall + noise, dsm)
        sigma = np.where((x == SATURATED[0]) & (y == SATURATED[1]), 255, sigma)
        none = x < E2_WEST_EDGE_M
    else:
        box, tall = SPOIL
        dsm = np.where(_inside(x, y, box), ground + tall + noise, dsm)
        none = (plane.data[..., 1].astype(np.int64) & cell_grid.REASON_BITS["unseen"]) > 0
    data = np.zeros((plane.nj, plane.ni, len(cell_grid.SURFACE_CHANNELS)), dtype=np.int64)
    data[..., 0] = np.where(none, NO, dsm)
    data[..., 1] = np.where(none, NO, ground)
    data[..., 2] = np.where(none, 0, sigma)
    data[..., 3] = np.where(none, 0, SUPPORT)
    data[..., 4] = np.where(none, 0, GROUND_SIGMA[epoch])
    return cell_grid.CellPlane(plane.level, plane.i0, plane.j0, plane.ni, plane.nj, data)


def _meta(plane: Any, graded_descriptor: dict[str, Any], height: float) -> dict[str, Any]:
    has = plane.data[..., 2] > 0
    sigma = plane.data[..., 2][has]
    return {
        "policy": "surface-v1",
        "status": "uncalibrated",
        "estimators": {
            "fixture": "synthetic: integer arithmetic over the synthetic-yard-epochs planes, "
            "not a measurement (tools/pipeline/experiments/surface_fixture.py)",
            # The window P5's ground would have been taken the least over at this level.
            "groundWindowM": 5.0,
            "groundWindowCells": GROUND_WINDOW_CELLS[plane.level],
        },
        "noData": NO,
        "sourceBits": dict(cell_grid.SURFACE_SOURCE_BITS),
        "sigmaSaturatesAt": 255,
        "vertical": {
            "zero": "the tileset frame's origin",
            "originHeightM": height,
            "surface": "the tangent plane at the origin, not a height above a datum",
        },
        "alignedTo": {
            "kind": "confidence",
            "level": plane.level,
            "sha256": graded_descriptor["sha256"],
        },
        "summary": {
            "cells": plane.cells,
            "surfaced": int(has.sum()),
            "thin": int((~has).sum()),
            "samples": {"sfm": 0, "splat": 0},
            "outside": 0,
            "nonFinite": 0,
            "outOfRange": 0,
            "sigmaSaturated": int((sigma >= 255).sum()),
            "medianSigmaCm": float(np.median(sigma)) if sigma.size else None,
            # Built in metres, not reconstructed: its scale is the generator's own. The
            # change map refuses a surface whose heights are not metres (review B5a).
            "metric": True,
        },
    }


def plane_sha256(plane: Any) -> str:
    """sha256 of `plane` laid out as the container lays a surface plane (each channel's
    own type, interleaved): what a decoder's plane must hash to."""
    record = np.dtype([(c.name, cell_grid.TYPES[c.type]) for c in cell_grid.SURFACE_CHANNELS])
    rows = np.zeros((plane.nj, plane.ni), dtype=record)
    for index, channel in enumerate(cell_grid.SURFACE_CHANNELS):
        rows[channel.name] = plane.data[..., index]
    return hashlib.sha256(rows.tobytes()).hexdigest()


# --- the no-change control (review B5a) ---------------------------------------------------

#: The control's scans of one unchanged scene: (samples a cell, height noise in metres).
#: Pairs with very different counts (a, b) and noise (c, d); review B5a's own cases.
CONTROL: dict[str, tuple[int, float]] = {
    "a": (20, 0.10),
    "b": (2000, 0.10),
    "c": (100, 0.01),
    "d": (100, 0.10),
}
#: The control's plane: 32 x 32 cells of 0.25 m, a 2 x 2 m box 1.5 m tall in the middle.
CONTROL_CELLS = 32
CONTROL_BOX: tuple[Box, int] = ((3, 3, 5, 5), 150)
_MASK64 = (1 << 64) - 1


def _stream(seed: int, n: int) -> np.ndarray:
    """`n` 64-bit words of splitmix64 from `seed`: integer arithmetic alone, so every
    machine draws the same ones (numpy's generators promise no such thing across
    versions)."""
    with np.errstate(over="ignore"):
        x = (np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)) + np.uint64(
            seed & _MASK64
        )
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        out: np.ndarray = x ^ (x >> np.uint64(31))
    return out


def _normal(seed: int, n: int) -> np.ndarray:
    """`n` near-normal draws, mean 0 and standard deviation 1: the sum of twelve uniform
    16-bit draws less their mean (Irwin-Hall), whole multiples of 2^-16, so exact."""
    words = _stream(seed, 3 * n).reshape(n, 3)
    total = np.zeros(n, dtype=np.int64)
    for column in range(3):
        for shift in (0, 16, 32, 48):
            total += ((words[:, column] >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(np.int64)
    out: np.ndarray = (total - 6 * 65536 + 6) / 65536.0  # (12 * 65535 / 2 rounded)
    return out


def control_scan(name: str) -> Any:
    """One scan of the control's unchanged scene by P5's own estimator (`surface_plane`):
    its samples uniform in each cell, its heights the scene's plus its noise."""
    per_cell, noise = CONTROL[name]
    seed = sum(ord(c) << (8 * k) for k, c in enumerate(f"control-{name}"))
    cells = CONTROL_CELLS * CONTROL_CELLS
    n = cells * per_cell
    words = _stream(seed ^ 0x5DEECE66D, 2 * n).reshape(n, 2)
    unit = ((words >> np.uint64(11)).astype(np.float64)) / float(1 << 53)  # [0, 1), exact
    i, j = np.divmod(np.repeat(np.arange(cells, dtype=np.int64), per_cell), CONTROL_CELLS)
    cell = 0.25
    x, y = (i + unit[:, 0]) * cell, (j + unit[:, 1]) * cell
    (x0, y0, x1, y1), tall = CONTROL_BOX
    box = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    z = np.where(box, tall / 100.0, 0.0) + noise * _normal(seed, n)
    xyz = np.column_stack([x, y, z])
    window = p5.Window(level=0, i0=0, j0=0, ni=CONTROL_CELLS, nj=CONTROL_CELLS)
    return p5.surface_plane(lambda: iter([(xyz, p5.SPLAT)]), window)


def control(out: Path, height: float) -> dict[str, Any]:
    """`control/<name>/surface.bin` for every scan of `CONTROL`, and their descriptors."""
    scans: dict[str, Any] = {}
    for name, (per_cell, noise) in CONTROL.items():
        found = control_scan(name)
        plane = found.plane
        has = plane.data[..., 2] > 0
        meta = {
            "policy": "surface-v1",
            "status": "uncalibrated",
            "estimators": {
                "control": f"P5's surface_plane over {per_cell} samples a cell with {noise:g} m "
                "of near-normal height noise, one unchanged scene (review B5a's no-change "
                "control; tools/pipeline/experiments/surface_fixture.py)",
                "groundWindowM": p5.GROUND_WINDOW_M,
                "groundWindowCells": 2 * found.ground_half + 1,
            },
            "noData": NO,
            "sourceBits": dict(cell_grid.SURFACE_SOURCE_BITS),
            "sigmaSaturatesAt": 255,
            "vertical": {
                "zero": "the tileset frame's origin",
                "originHeightM": height,
                "surface": "the tangent plane at the origin, not a height above a datum",
            },
            "summary": {
                "cells": plane.cells,
                "surfaced": int(has.sum()),
                "groundUnobserved": found.ground_unobserved,
                "metric": True,
            },
        }
        grid = cell_grid.CellGrid("surface", cell_grid.SURFACE_CHANNELS, (plane,), meta)
        descriptor = cell_grid.encode(grid, out / "control" / name, "surface.bin")
        blob = (out / "control" / name / "surface.bin").read_bytes()
        scans[name] = {
            "samplesPerCell": per_cell,
            "noiseM": noise,
            "descriptor": descriptor,
            "file": f"control/{name}/surface.bin",
            "fileSha256": hashlib.sha256(blob).hexdigest(),
        }
    return {
        "note": "review B5a's no-change control: one unchanged scene (flat ground, a 2 x 2 m "
        "box 1.5 m tall) scanned with very different sample counts (a, b) and noise (c, d), "
        "each made by P5's own estimator; no change of either layer may be read between them",
        "pairs": [["a", "b"], ["c", "d"]],
        "scans": scans,
    }


def write(out: Path) -> dict[str, Any]:
    """`e1/surface.bin`, `e2/surface.bin` and `surfaces.json` in `out`."""
    height = origin_height()
    document: dict[str, Any] = {
        "note": "cross-language fixture for hexapod.cellgrid v1 surface grids "
        "(tools/pipeline/experiments/surface_fixture.py); synthetic, not measured",
        "frame": "data/tiles/synthetic-yard/splat/tileset.json (tileset frame), on the "
        "planes of data/tiles/synthetic-yard-epochs",
        "epochs": {},
    }
    surfaces: dict[str, Any] = {}
    graded: dict[str, Any] = {}
    for name in ("e1", "e2"):
        graded_descriptor, graded[name] = confidence(name)
        plane = surface_plane(name, graded[name])
        meta = _meta(plane, graded_descriptor, height)
        grid = cell_grid.CellGrid("surface", cell_grid.SURFACE_CHANNELS, (plane,), meta)
        descriptor = cell_grid.encode(grid, out / name, "surface.bin")
        blob = (out / name / "surface.bin").read_bytes()
        surfaces[name] = cell_grid.decode(descriptor, blob)
        document["epochs"][name] = {
            "descriptor": descriptor,
            "file": f"{name}/surface.bin",
            "fileSha256": hashlib.sha256(blob).hexdigest(),
            "plane": {"level": plane.level, "sha256": plane_sha256(plane)},
            "uncertaintyM": graded_descriptor["alignmentCap"]["uncertaintyM"],
        }
    level = max(surfaces["e1"].level, surfaces["e2"].level)
    coarse = coarsen_surface(surfaces["e2"].planes[0], level)
    document["coarsened"] = {
        "epoch": "e2",
        "level": level,
        "window": [coarse.i0, coarse.j0, coarse.ni, coarse.nj],
        "sha256": plane_sha256(coarse),
    }
    ua = float(document["epochs"]["e1"]["uncertaintyM"])
    ub = float(document["epochs"]["e2"]["uncertaintyM"])
    change = surface_change(surfaces["e1"], surfaces["e2"], graded["e1"], graded["e2"], ua, ub)
    document["change"] = {
        "reference": "experiments.confidence_demo.surface_change (P5): dsm by max, ground by "
        "min, sigma by max at the coarser level; 3 sqrt(sa^2 + sb^2 + ua^2 + ub^2)",
        "from": "e1",
        "to": "e2",
        "level": change["level"],
        "cellM": change["cellM"],
        "counts": change["counts"],
        "weakChanged": change["weakChanged"],
        "changed": [
            {key: row[key] for key in ("i", "j", "deltaM", "thresholdM", "class", "weak")}
            for row in change["changed"]
        ],
    }
    document["control"] = control(out, height)
    (out / "surfaces.json").write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    document = write(args.out)
    for name, epoch in document["epochs"].items():
        summary = epoch["descriptor"]["summary"]
        sys.stdout.write(f"{name}: level {epoch['plane']['level']}, {summary}\n")
    change = document["change"]
    sys.stdout.write(f"change e1 -> e2 at level {change['level']}: {change['counts']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
