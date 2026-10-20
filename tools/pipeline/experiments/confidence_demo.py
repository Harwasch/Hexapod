"""The milestone-4 pipeline demo: an epoch scanned from one side, re-scanned all round.

Builds `tests/confidence_scene.py`'s field -- ground with a pond in it and a 4 m "tree"
box -- captured twice: **e1**, a 160 degree arc of 24 cameras on the south side, and
**e2**, the re-scan, a full orbit of 24 plus 8 steep obliques, aligned visually to e1 at
+/-0.062 m (a synthetic `epoch_alignment.json`: `align` needs COLMAP, which this demo does
not run). Each goes through the shipped `quality -> place -> confidence -> surface ->
package` under LocalRunner, CPU only. Then e2's grid is made again as if its alignment had
failed (unaligned, +/-10 m) to show the cap. It prints one row per epoch -- level, cells,
weak share, the reasons, the cap, the root tile's `confidenceMin` -- and what the re-scan
did to e1's weak cells at the coarser of the two levels, with a map of each. Two more rows
show the cap is the frame's: e2 aligned as well to an e1 consumer GPS put +/-5 m from the
globe (not capped), and e2 as a later epoch placed by its GPS alone (capped, weak
everywhere). Both tilesets are written under `--out` for a viewer, and `report.json` beside
them.

**`--change`** (milestone 5, ADR 0018): e2 has a 1.5 m crate on open ground north-west of
the tree, where both epochs saw well, that e1 did not, and each epoch is its own scan of
the field -- its own sample of it, 1 cm of height noise, and a sparse SfM cloud in its
pose model. The demo then prints the change between their height surfaces (the `surface`
stage's) per cell, the way the change map (A5) measures it: both surfaces at the coarser
level (dsm by max, ground by min, sigma by max), and `delta = dsm_e2 - dsm_e1` against
`3 sqrt(sigma_e1^2 + sigma_e2^2 + u_e1^2 + u_e2^2)`, where `u` is each epoch's uncertainty
from the site frame (its confidence grid's `alignmentCap.uncertaintyM`: 0 for e1, which
made the frame; 0.062 m for e2). A cell is `insufficient` when either epoch has no surface
there. The report also counts the changed cells either epoch's confidence grid calls weak
-- whether those may be called changed is the change map's decision (A5), not this
demo's -- and the changed cells that do not touch the crate: what the noise did.

**Every capture is synthetic.** It shows the stage working against a scene whose answers
are known; it is not a measurement of a real capture, and no threshold here was
calibrated on it (docs/DECISIONS/0016-confidence-map.md).

    cd tools/pipeline && uv run --no-sync python -m experiments.confidence_demo \\
        --out <dir> [--change] | tee <dir>/report.txt
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
# The scene builder is a test helper; the demo runs the very same code the tests do.
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tests"))

import stages  # noqa: E402,F401 - registers the shipped stage implementations
from captures_bridge import cell_grid  # noqa: E402

__all__ = ["coarsen_surface", "main", "rescan", "rows", "surface_change"]

#: Gaussians in the field, and the field's own seed: the same place, scanned twice.
GAUSSIANS = 150_000
SEED = 7


#: `--change`: e2's crate (centre east/north, side and height, metres), each scan's height
#: noise, and the share of each scan's samples its sparse SfM cloud holds.
CRATE = ((-3.1, 3.4), 1.5, 1.5)
NOISE_M = 0.01
SFM_SHARE = 0.03


def _field(seed: int = SEED) -> dict[str, np.ndarray]:
    import confidence_scene

    return confidence_scene.field(GAUSSIANS, tree_m=(2.0, 4.0), ground_m=20.0, seed=seed)


def _run(
    root: Path,
    cameras: Sequence[Any],
    columns: Mapping[str, np.ndarray] | None = None,
    points: np.ndarray | None = None,
    **context: Any,
) -> Path:
    import confidence_scene
    from conftest import make_recipe
    from executor import execute
    from runners import RunnerSet
    from workdir import Workdir

    workdir = Workdir.create(root)
    splat = dict(columns) if columns is not None else _field()
    names = confidence_scene.write_epoch(
        workdir.root / "inputs", splat, list(cameras), points=points, **context
    )
    execute(
        make_recipe(
            [
                {"id": "quality", "impl": "support_gate", "params": {"bar": "strict"}},
                {"id": "place", "impl": "place_splat"},
                {"id": "confidence", "impl": "confidence_grid"},
                {"id": "surface", "impl": "height_surface"},
                {"id": "package", "impl": "splat_tiles", "params": {"tile_gaussians": 20_000}},
            ],
            inputs=names,
        ),
        workdir,
        RunnerSet.local(),
    )
    return workdir.root


def _confidence_again(root: Path, epoch: Path, alignment: Mapping[str, Any]) -> Path:
    """e2's confidence stage alone, over its own support, with another alignment."""
    from conftest import make_recipe
    from executor import execute
    from runners import RunnerSet
    from workdir import Workdir

    workdir = Workdir.create(root)
    inputs = workdir.root / "inputs"
    shutil.copytree(epoch / "stages" / "quality" / "out" / "support", inputs / "support")
    shutil.copy(epoch / "stages" / "place" / "out" / "placement.json", inputs / "placement.json")
    shutil.copytree(epoch / "inputs" / "site_context", inputs / "site_context")
    shutil.copy(epoch / "inputs" / "georef.json", inputs / "georef.json")
    (inputs / "epoch_alignment.json").write_text(json.dumps(alignment), encoding="utf-8")
    names = ["support", "placement.json", "site_context", "georef.json", "epoch_alignment.json"]
    execute(
        make_recipe([{"id": "confidence", "impl": "confidence_grid"}], inputs=names),
        workdir,
        RunnerSet.local(),
    )
    return workdir.root


def _grid(run: Path) -> tuple[dict[str, Any], cell_grid.CellGrid]:
    folder = run / "stages" / "confidence" / "out" / "confidence"
    descriptor = json.loads((folder / "confidence.json").read_text(encoding="utf-8"))
    return descriptor, cell_grid.decode(descriptor, (folder / "confidence.bin").read_bytes())


def rows(epochs: Mapping[str, Path]) -> list[dict[str, object]]:
    """One summary per epoch, from its grid and its tileset."""
    out: list[dict[str, object]] = []
    for name, run in epochs.items():
        descriptor, grid = _grid(run)
        summary, cap = descriptor["summary"], descriptor["alignmentCap"]
        tileset_path = run / "stages" / "package" / "out" / "splat" / "tileset.json"
        root_min: object = "-"
        if tileset_path.is_file():
            root = json.loads(tileset_path.read_text(encoding="utf-8"))["root"]
            children = [c["extras"].get("confidenceMin") for c in root.get("children", [])]
            root_min = f"{root['extras'].get('confidenceMin')} (children {children})"
        reasons = {k: v for k, v in summary["reasons"].items() if v}
        out.append(
            {
                "epoch": name,
                "level": grid.level,
                "cellM": grid.planes[0].cell_m,
                "cells": summary["cells"],
                "scored": summary["scored"],
                "weak": summary["weak"],
                "unseen": summary["unseen"],
                "weakShare": summary["weakShare"],
                "reasons": reasons,
                "cap": f"u={cap['uncertaintyM']} m ({cap['source']}, prior {cap['priorEpochs']}) "
                f"-> f={cap['factor']}",
                "rootConfidenceMin": root_min,
            }
        )
    return out


def _common(a: cell_grid.CellGrid, b: cell_grid.CellGrid) -> tuple[Any, Any, Any, Any]:
    """Both grids at the coarser of their finest levels, on one window of cells."""
    level = max(a.level, b.level)
    pa, pb = a.plane(level), b.plane(level)
    i0, j0 = min(pa.i0, pb.i0), min(pa.j0, pb.j0)
    i1, j1 = max(pa.i0 + pa.ni, pb.i0 + pb.ni), max(pa.j0 + pa.nj, pb.j0 + pb.nj)

    def window(plane: cell_grid.CellPlane) -> np.ndarray:
        out = np.zeros((j1 - j0, i1 - i0, 3), dtype=np.uint8)
        dj, di = plane.j0 - j0, plane.i0 - i0
        out[dj : dj + plane.nj, di : di + plane.ni] = plane.data
        return out

    wa = cell_grid.CellPlane(level, i0, j0, i1 - i0, j1 - j0, window(pa))
    wb = cell_grid.CellPlane(level, i0, j0, i1 - i0, j1 - j0, window(pb))
    return level, wa, wb, (i0, j0)


def rescan(e1: cell_grid.CellGrid, e2: cell_grid.CellGrid) -> dict[str, Any]:
    """What the re-scan did to the first epoch's weak cells, at the common level."""
    level, a, b, _ = _common(e1, e2)
    weak_a, weak_b = cell_grid.weak_mask(a), cell_grid.weak_mask(b)
    covered_b = (b.data[..., 0] > 0) | ((b.data[..., 1] & cell_grid.REASON_BITS["unseen"]) > 0)
    still = weak_a & weak_b & covered_b
    return {
        "level": level,
        "cellM": cell_grid.Lattice(level).cell_m,
        "weakInE1": int(weak_a.sum()),
        "coveredByE2": int((weak_a & covered_b).sum()),
        "strongAfterRescan": int((weak_a & covered_b & ~weak_b).sum()),
        "stillWeak": int(still.sum()),
        "newlyWeakInE2": int((~weak_a & (a.data[..., 0] > 0) & weak_b).sum()),
        "map": _map(a, b),
    }


def _map(a: cell_grid.CellPlane, b: cell_grid.CellPlane) -> list[str]:
    """North up: `#` weak in both, `+` weak in e1 and strong after, `-` weak only in e2,
    `.` strong in both, `u` unseen in e2, blank no evidence in either."""
    weak_a, weak_b = cell_grid.weak_mask(a), cell_grid.weak_mask(b)
    seen_a, seen_b = a.data[..., 0] > 0, b.data[..., 0] > 0
    unseen_b = (b.data[..., 1] & cell_grid.REASON_BITS["unseen"]) > 0
    lines = []
    for j in reversed(range(a.nj)):
        line = ""
        for i in range(a.ni):
            if weak_a[j, i] and weak_b[j, i]:
                line += "u" if unseen_b[j, i] else "#"
            elif weak_a[j, i] and seen_b[j, i]:
                line += "+"
            elif weak_b[j, i]:
                line += "-"
            elif seen_a[j, i] or seen_b[j, i]:
                line += "."
            else:
                line += " "
        lines.append(line.rstrip())
    return lines


# --- --change: the change between e1 and e2, from their height surfaces ------------------


def _surface(run: Path) -> tuple[dict[str, Any], cell_grid.CellGrid]:
    folder = run / "stages" / "surface" / "out" / "surface"
    descriptor = json.loads((folder / "surface.json").read_text(encoding="utf-8"))
    return descriptor, cell_grid.decode(descriptor, (folder / "surface.bin").read_bytes())


def coarsen_surface(plane: cell_grid.CellPlane, level: int) -> cell_grid.CellPlane:
    """`plane` at `level` (>= its own): dsm the highest, ground the lowest and sigma the
    largest of the children that have a surface; support's bits OR'd, its count the
    largest child's (a floor). A cell with no child surfaced has none."""
    no = cell_grid.NO_HEIGHT
    while plane.level < level:
        pi0, pj0 = plane.i0 >> 1, plane.j0 >> 1
        ni = ((plane.i0 + plane.ni - 1) >> 1) - pi0 + 1
        nj = ((plane.j0 + plane.nj - 1) >> 1) - pj0 + 1
        padded = np.zeros((2 * nj, 2 * ni, 4), dtype=np.int64)
        padded[..., 0] = padded[..., 1] = no
        di, dj = plane.i0 - 2 * pi0, plane.j0 - 2 * pj0
        padded[dj : dj + plane.nj, di : di + plane.ni] = plane.data
        quads = padded.reshape(nj, 2, ni, 2, 4)
        has = quads[..., 2] > 0
        out = np.zeros((nj, ni, 4), dtype=np.int64)
        out[..., 0] = np.where(has, quads[..., 0], no - 1).max(axis=(1, 3))
        out[..., 1] = np.where(has, quads[..., 1], 1 << 20).min(axis=(1, 3))
        out[..., 2] = np.where(has, quads[..., 2], 0).max(axis=(1, 3))
        any_has = has.any(axis=(1, 3))
        out[..., 0] = np.where(any_has, out[..., 0], no)
        out[..., 1] = np.where(any_has, out[..., 1], no)
        support = quads[..., 3]
        bits = np.bitwise_or.reduce(np.bitwise_or.reduce(support & 0x0F, axis=3), axis=1)
        out[..., 3] = bits | ((support >> 4).max(axis=(1, 3)) << 4)
        plane = cell_grid.CellPlane(plane.level + 1, pi0, pj0, ni, nj, out)
    return plane


def _window(
    plane: cell_grid.CellPlane, i0: int, j0: int, ni: int, nj: int, fill: Sequence[int]
) -> np.ndarray:
    out = np.empty((nj, ni, plane.data.shape[2]), dtype=np.int64)
    out[...] = np.asarray(fill, dtype=np.int64)
    dj, di = plane.j0 - j0, plane.i0 - i0
    out[dj : dj + plane.nj, di : di + plane.ni] = plane.data
    return out


#: The change classes A5 emits (`appearance-only` is reserved and never emitted here).
CLASSES = ("raised", "lowered", "unchanged", "insufficient")


def surface_change(
    a: cell_grid.CellGrid,
    b: cell_grid.CellGrid,
    graded_a: cell_grid.CellGrid,
    graded_b: cell_grid.CellGrid,
    ua: float,
    ub: float,
    *,
    sigmas: float = 3.0,
) -> dict[str, Any]:
    """Per cell of the coarser of the two surfaces' levels: the change in dsm, the
    threshold `sigmas * sqrt(sa^2 + sb^2 + ua^2 + ub^2)` and the class (module docstring);
    and, per changed cell, whether either confidence grid calls it weak."""
    level = max(a.level, b.level)
    pa, pb = coarsen_surface(a.planes[0], level), coarsen_surface(b.planes[0], level)
    i0, j0 = min(pa.i0, pb.i0), min(pa.j0, pb.j0)
    ni = max(pa.i0 + pa.ni, pb.i0 + pb.ni) - i0
    nj = max(pa.j0 + pa.nj, pb.j0 + pb.nj) - j0
    no = cell_grid.NO_HEIGHT
    wa, wb = (_window(p, i0, j0, ni, nj, (no, no, 0, 0)) for p in (pa, pb))
    weak = np.zeros((nj, ni), dtype=bool)
    for graded in (graded_a, graded_b):
        plane = graded.plane(level)
        window = cell_grid.CellPlane(
            level, i0, j0, ni, nj, _window(plane, i0, j0, ni, nj, (0, 0, 0))
        )
        weak |= cell_grid.weak_mask(window)
    both = (wa[..., 2] > 0) & (wb[..., 2] > 0)
    delta = (wb[..., 0] - wa[..., 0]) * cell_grid.HEIGHT_UNIT_M
    sa, sb = wa[..., 2] * cell_grid.HEIGHT_UNIT_M, wb[..., 2] * cell_grid.HEIGHT_UNIT_M
    threshold = sigmas * np.sqrt(sa**2 + sb**2 + ua**2 + ub**2)
    measured = np.where(delta > threshold, 0, np.where(delta < -threshold, 1, 2))
    classes = np.where(both, measured, 3)
    cell = cell_grid.Lattice(level).cell_m
    rows = []
    for j, i in np.argwhere(classes < 2):
        rows.append(
            {
                "i": int(i0 + i),
                "j": int(j0 + j),
                "eastM": round((i0 + i + 0.5) * cell, 3),
                "northM": round((j0 + j + 0.5) * cell, 3),
                "dsmE1M": round(float(wa[j, i, 0]) * cell_grid.HEIGHT_UNIT_M, 2),
                "dsmE2M": round(float(wb[j, i, 0]) * cell_grid.HEIGHT_UNIT_M, 2),
                "deltaM": round(float(delta[j, i]), 2),
                "thresholdM": round(float(threshold[j, i]), 3),
                "class": CLASSES[int(classes[j, i])],
                "weak": bool(weak[j, i]),
            }
        )
    unchanged = classes == 2
    ratio = np.abs(delta[unchanged]) / threshold[unchanged] if unchanged.any() else np.zeros(1)
    return {
        "level": level,
        "cellM": cell,
        "uE1M": ua,
        "uE2M": ub,
        "counts": {name: int((classes == k).sum()) for k, name in enumerate(CLASSES)},
        "minDetectableM": round(float(sigmas * math.sqrt(0.01**2 * 2 + ua**2 + ub**2)), 3),
        "unchangedMaxAbsDeltaM": round(float(np.abs(delta[unchanged]).max()), 3)
        if unchanged.any()
        else None,
        "unchangedMaxDeltaOverThreshold": round(float(ratio.max()), 3),
        # Changed cells either epoch's confidence grid calls weak (A5 decides about them).
        "weakChanged": int((weak & (classes < 2)).sum()),
        "changed": rows,
        "map": _change_map(classes),
    }


def _crate_tally(found: Mapping[str, Any]) -> dict[str, Any]:
    """How much of the crate was called raised, and what was called changed elsewhere: a
    cell is on the crate when its centre is on the crate's footprint, and touches it when
    any of it is (the cells its sides stand in)."""
    (east, north), side, height = CRATE
    cell = float(found["cellM"])
    i0, i1 = math.floor((east - side / 2) / cell), math.floor((east + side / 2) / cell)
    j0, j1 = math.floor((north - side / 2) / cell), math.floor((north + side / 2) / cell)

    def on(row: Mapping[str, Any]) -> bool:
        return abs(row["eastM"] - east) <= side / 2 and abs(row["northM"] - north) <= side / 2

    def touches(row: Mapping[str, Any]) -> bool:
        reach = side / 2 + cell / 2
        return abs(row["eastM"] - east) < reach and abs(row["northM"] - north) < reach

    footprint = sum(
        1
        for i in range(i0, i1 + 1)
        for j in range(j0, j1 + 1)
        if abs((i + 0.5) * cell - east) <= side / 2 and abs((j + 0.5) * cell - north) <= side / 2
    )
    rows = found["changed"]
    return {
        "centreM": [east, north],
        "sideM": side,
        "heightM": height,
        "cellsInside": footprint,
        "raisedInside": sum(1 for r in rows if on(r) and r["class"] == "raised"),
        "raisedTouching": sum(1 for r in rows if touches(r) and not on(r)),
        "changedOutside": sum(1 for r in rows if not touches(r)),
    }


def _change_map(classes: np.ndarray) -> list[str]:
    """North up: `R` raised, `L` lowered, `.` unchanged, ` ` insufficient."""
    marks = "RL. "
    return [
        "".join(marks[int(c)] for c in classes[j]).rstrip()
        for j in reversed(range(classes.shape[0]))
    ]


def change_report(found: Mapping[str, Any]) -> str:
    lines = [
        f"Change e1 -> e2 at level {found['level']} ({found['cellM']} m cells): "
        f"u_e1 = {found['uE1M']} m, u_e2 = {found['uE2M']} m (from the site frame); threshold "
        f"3 sqrt(s1^2 + s2^2 + u1^2 + u2^2), at least {found['minDetectableM']} m here.",
        "| class | cells |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {found['counts'][name]} |" for name in CLASSES)
    lines.append("")
    lines.append(
        f"Unchanged cells: |delta| at most {found['unchangedMaxAbsDeltaM']} m, at most "
        f"{found['unchangedMaxDeltaOverThreshold']} of their threshold."
    )
    lines.append(
        f"Changed cells weak in either confidence grid: {found['weakChanged']} (whether they may "
        "be called changed is the change map's decision)."
    )
    if "crate" in found:
        crate = found["crate"]
        lines.append(
            f"The crate ({crate['sideM']} m, {crate['heightM']} m tall, at "
            f"{crate['centreM']}): {crate['raisedInside']} of its {crate['cellsInside']} cells "
            f"raised, and {crate['raisedTouching']} cells its sides stand in; changed cells "
            f"that do not touch it (what the noise did): {crate['changedOutside']}."
        )
    lines.append("")
    lines.append(
        "| i | j | east m | north m | dsm e1 m | dsm e2 m | delta m | 3 sigma m | class | weak |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in found["changed"]:
        lines.append(
            f"| {row['i']} | {row['j']} | {row['eastM']} | {row['northM']} | {row['dsmE1M']} | "
            f"{row['dsmE2M']} | {row['deltaM']} | {row['thresholdM']} | {row['class']} | "
            f"{'weak' if row['weak'] else ''} |"
        )
    lines.append("")
    lines.append("Map (north up; R raised, L lowered, . unchanged, blank insufficient):")
    lines.extend("  " + line for line in found["map"])
    lines.append("Derived from height surfaces, not observed. Every capture here is synthetic.")
    return "\n".join(lines)


def report(table: Sequence[Mapping[str, object]], change: Mapping[str, Any], seconds: float) -> str:
    keys = ["epoch", "level", "cellM", "cells", "scored", "weak", "unseen", "weakShare"]
    lines = ["| " + " | ".join(keys) + " | reasons (weak cells) | cap | root confidenceMin |"]
    lines.append("|" + "---|" * (len(keys) + 3))
    for row in table:
        cells = [str(row[k]) for k in keys]
        lines.append(
            "| "
            + " | ".join(cells)
            + f" | {row['reasons']} | {row['cap']} | {row['rootConfidenceMin']} |"
        )
    lines.append("")
    lines.append(
        f"Re-scan, at level {change['level']} ({change['cellM']} m): {change['weakInE1']} cells "
        f"weak in e1; {change['coveredByE2']} of them covered by e2; "
        f"{change['strongAfterRescan']} strong after it; {change['stillWeak']} still weak; "
        f"{change['newlyWeakInE2']} weak only in e2."
    )
    lines.append("Map (north up; # still weak, + fixed by the re-scan, - weak only in e2,")
    lines.append(" . strong, u unseen in both):")
    lines.extend("  " + line for line in change["map"])
    lines.append("")
    lines.append(f"Ran in {seconds:.1f} s, CPU only. Every capture here is synthetic.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import confidence_scene

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--change",
        action="store_true",
        help="put a 1.5 m crate in e2, scan each epoch with its own noise and SfM cloud, "
        "and print the change between their height surfaces",
    )
    args = parser.parse_args(argv)
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    started = time.perf_counter()
    scans: dict[str, dict[str, Any]] = {"e1": {}, "e2": {}}
    if args.change:
        (east, north), side, height = CRATE
        for name, seed in (("e1", SEED), ("e2", SEED + 1)):
            columns = _field(seed)
            if name == "e2":
                columns = confidence_scene.with_crate(
                    columns, centre=(east, north), side_m=side, height_m=height, seed=seed
                )
            columns = confidence_scene.jitter(columns, NOISE_M, seed=seed + 100)
            rng = np.random.default_rng(seed + 200)
            cloud = np.column_stack([columns["x"], columns["y"], columns["z"]])
            scans[name] = {"columns": columns, "points": cloud[rng.random(len(cloud)) < SFM_SHARE]}
    e1 = _run(
        out / "e1", confidence_scene.arc(24, degrees=160.0, radius=7.0, height=2.5), **scans["e1"]
    )
    e2 = _run(
        out / "e2",
        confidence_scene.orbit(24) + confidence_scene.obliques(8),
        prior_epochs=1,
        alignment=confidence_scene.alignment_of(0.062),
        **scans["e2"],
    )
    unaligned = _confidence_again(
        out / "e2-unaligned", e2, confidence_scene.alignment_of(10.0, "unaligned")
    )
    # Aligned just as well, but to an epoch 1 that consumer GPS put +/-5 m from the globe:
    # +/-5 m on the globe, +/-0.062 m from the site frame -- and the cap is the frame's.
    gps_framed = _confidence_again(
        out / "e2-gps-framed",
        e2,
        confidence_scene.alignment_of(5.0004, frame_relative_m=0.062),
    )
    # A later epoch nobody could align, placed by its EXIF GPS alone (+/-5 m).
    gps_only = _confidence_again(
        out / "e2-gps-only", e2, confidence_scene.alignment_of(5.0, "unaligned")
    )
    seconds = time.perf_counter() - started
    table = rows(
        {
            "e1 (south arc)": e1,
            "e2 (re-scan)": e2,
            "e2 if unaligned": unaligned,
            "e2 aligned to a GPS-framed e1": gps_framed,
            "e2 if GPS only": gps_only,
        }
    )
    change = rescan(_grid(e1)[1], _grid(e2)[1])
    text = report(table, change, seconds)
    sys.stdout.write(text + "\n")
    surfaces: dict[str, Any] | None = None
    if args.change:
        (ga, gb) = (_grid(run) for run in (e1, e2))
        (sa, sb) = (_surface(run) for run in (e1, e2))
        surfaces = surface_change(
            sa[1],
            sb[1],
            ga[1],
            gb[1],
            float(ga[0]["alignmentCap"]["uncertaintyM"]),
            float(gb[0]["alignmentCap"]["uncertaintyM"]),
        )
        surfaces["crate"] = _crate_tally(surfaces)
        sys.stdout.write("\n" + change_report(surfaces) + "\n")
    for name, run in (("e1", e1), ("e2", e2)):
        shutil.copytree(run / "stages" / "package" / "out" / "splat", out / f"{name}-splat")
    (out / "report.json").write_text(
        json.dumps(
            {
                "epochs": table,
                "rescan": change,
                "change": surfaces,
                "seconds": round(seconds, 1),
                "descriptors": {
                    "e1": _grid(e1)[0],
                    "e2": _grid(e2)[0],
                    "e2-unaligned": _grid(unaligned)[0],
                    "e2-gps-framed": _grid(gps_framed)[0],
                    "e2-gps-only": _grid(gps_only)[0],
                },
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
