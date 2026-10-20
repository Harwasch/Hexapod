"""The milestone-4 pipeline demo: an epoch scanned from one side, re-scanned all round.

Builds `tests/confidence_scene.py`'s field -- ground with a pond in it and a 4 m "tree"
box -- captured twice: **e1**, a 160 degree arc of 24 cameras on the south side, and
**e2**, the re-scan, a full orbit of 24 plus 8 steep obliques, aligned visually to e1 at
+/-0.062 m (a synthetic `epoch_alignment.json`: `align` needs COLMAP, which this demo does
not run). Each goes through the shipped `quality -> place -> confidence -> package` under
LocalRunner, CPU only. Then e2's grid is made again as if its alignment had failed
(unaligned, +/-10 m) to show the cap. It prints one row per epoch -- level, cells, weak
share, the reasons, the cap, the root tile's `confidenceMin` -- and what the re-scan did
to e1's weak cells at the coarser of the two levels, with a map of each. Both tilesets
are written under `--out` for a viewer, and `report.json` beside them.

**Every capture is synthetic.** It shows the stage working against a scene whose answers
are known; it is not a measurement of a real capture, and no threshold here was
calibrated on it (docs/DECISIONS/0016).

    cd tools/pipeline && uv run --no-sync python -m experiments.confidence_demo \\
        --out <dir> | tee <dir>/report.txt
"""

from __future__ import annotations

import argparse
import json
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

__all__ = ["main", "rescan", "rows"]

#: Gaussians in the field, and the field's own seed: the same place, scanned twice.
GAUSSIANS = 150_000
SEED = 7


def _run(root: Path, cameras: Sequence[Any], **context: Any) -> Path:
    import confidence_scene
    from conftest import make_recipe
    from executor import execute
    from runners import RunnerSet
    from workdir import Workdir

    workdir = Workdir.create(root)
    columns = confidence_scene.field(GAUSSIANS, tree_m=(2.0, 4.0), ground_m=20.0, seed=SEED)
    names = confidence_scene.write_epoch(workdir.root / "inputs", columns, list(cameras), **context)
    execute(
        make_recipe(
            [
                {"id": "quality", "impl": "support_gate", "params": {"bar": "strict"}},
                {"id": "place", "impl": "place_splat"},
                {"id": "confidence", "impl": "confidence_grid"},
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
    args = parser.parse_args(argv)
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    started = time.perf_counter()
    e1 = _run(out / "e1", confidence_scene.arc(24, degrees=160.0, radius=7.0, height=2.5))
    e2 = _run(
        out / "e2",
        confidence_scene.orbit(24) + confidence_scene.obliques(8),
        prior_epochs=1,
        alignment=confidence_scene.alignment_of(0.062),
    )
    unaligned = _confidence_again(
        out / "e2-unaligned", e2, confidence_scene.alignment_of(10.0, "unaligned")
    )
    seconds = time.perf_counter() - started
    table = rows({"e1 (south arc)": e1, "e2 (re-scan)": e2, "e2 if unaligned": unaligned})
    change = rescan(_grid(e1)[1], _grid(e2)[1])
    text = report(table, change, seconds)
    sys.stdout.write(text + "\n")
    for name, run in (("e1", e1), ("e2", e2)):
        shutil.copytree(run / "stages" / "package" / "out" / "splat", out / f"{name}-splat")
    (out / "report.json").write_text(
        json.dumps(
            {
                "epochs": table,
                "rescan": change,
                "seconds": round(seconds, 1),
                "descriptors": {
                    "e1": _grid(e1)[0],
                    "e2": _grid(e2)[0],
                    "e2-unaligned": _grid(unaligned)[0],
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
