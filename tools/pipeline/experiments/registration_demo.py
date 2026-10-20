"""The milestone-3 demo: two epochs of the synthetic tree align, with a residual report.

Builds `tests/registration_scene.py`'s epochs in `--out` -- epoch 1, an RTK drone survey
with eight check targets; a second RTK drone flight on another orbit; an iPhone-style
`.mov` with one ISO 6709 point metres off and no per-frame GPS (run twice: on its own,
and told epoch 1's origin as its site frame, as M2's worker runs it); and an unrelated
scene placed by hand -- runs the shipped normalize, pose, georeference, align and place
on each under LocalRunner with pycolmap 4.2, aligns every later epoch to epoch 1, and
prints one row per epoch: the method, verdict, label, confidence, registered frames,
inliers, the alignment's residual (RMS / median / p90 / max), the placed cameras' RMSE
against the truth, the check targets' RMSE against their tolerance, and the uncertainty
claimed. `report.json` beside it holds every epoch's whole `epoch_alignment.json`.

**Every capture is synthetic** (the scene module says how each was built). It shows the
registration working against a known truth; it is not a measurement of a real capture.

    cd tools/pipeline && uv run --no-sync python -m experiments.registration_demo \\
        --out <dir> | tee <dir>/report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# The scene builder is a test helper (it renders the committed tree with tests/
# tree_frames.py and registers a stand-in stage); the demo runs the very same code.
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tests"))

import stages  # noqa: E402,F401 - registers the shipped stage implementations

__all__ = ["main", "report", "rows"]


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def rows(epochs: Mapping[str, Any]) -> list[dict[str, object]]:
    """One summary per epoch, from its `epoch_alignment.json` and the truth."""
    import registration_scene

    out: list[dict[str, object]] = []
    for name, epoch in epochs.items():
        a = epoch.alignment
        residual = a.get("residualM") or {}
        checks = a.get("checkTargets") or {}
        rmse = checks.get("rmseM") or {}
        tolerance = checks.get("tolerance") or {}
        registration = a.get("registration") or {}
        out.append(
            {
                "epoch": name,
                "platform": a.get("platform"),
                "method": a.get("method"),
                "verdict": a.get("verdict"),
                "label": a.get("label"),
                "confidence": a.get("confidence"),
                "uncertaintyM": a.get("uncertaintyM"),
                "registered": None
                if not registration
                else f"{registration.get('registered')}/{registration.get('newFrames')}",
                "inliers": f"{a.get('inliers')}/{a.get('correspondences')}",
                "rmsM": residual.get("rms"),
                "medianM": residual.get("median"),
                "p90M": residual.get("p90"),
                "maxM": residual.get("max"),
                "truthRmseM": registration_scene.truth_centre_rmse(epoch),
                "checkStatus": checks.get("status"),
                "checkTargets": checks.get("evaluated"),
                "checkRmseHM": rmse.get("horizontal"),
                "checkRmseVM": rmse.get("vertical"),
                "toleranceHM": tolerance.get("horizontalM"),
                "toleranceVM": tolerance.get("verticalM"),
                "gnssOffsetM": (a.get("gnssCheck") or {}).get("offsetM"),
                "frameSource": a.get("frameSource"),
                "reasons": a.get("reasons"),
            }
        )
    return out


COLUMNS = (
    ("epoch", "epoch"),
    ("platform", "platform"),
    ("method", "method"),
    ("verdict", "verdict"),
    ("registered", "reg"),
    ("inliers", "inliers"),
    ("rmsM", "rms m"),
    ("medianM", "med m"),
    ("p90M", "p90 m"),
    ("maxM", "max m"),
    ("truthRmseM", "truth m"),
    ("checkTargets", "chk"),
    ("checkRmseHM", "chk h"),
    ("checkRmseVM", "chk v"),
    ("toleranceHM", "tol h"),
    ("toleranceVM", "tol v"),
    ("gnssOffsetM", "gnss m"),
    ("uncertaintyM", "+/- m"),
    ("confidence", "conf"),
)


def report(epochs: Mapping[str, Any]) -> str:
    """The table, then each epoch's label, frame source and reasons."""
    table = [[title for _, title in COLUMNS]]
    summaries = rows(epochs)
    for row in summaries:
        table.append([_fmt(row[key]) for key, _ in COLUMNS])
    widths = [max(len(line[i]) for line in table) for i in range(len(COLUMNS))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True))
             for line in table]  # fmt: skip
    lines.append("")
    for row in summaries:
        lines.append(f"{row['epoch']}: {row['label']}  (frame from {row['frameSource']})")
        for reason in row["reasons"] or []:  # type: ignore[attr-defined]
            lines.append(f"    - {reason}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    import registration_scene

    began = time.monotonic()
    epochs = registration_scene.build(args.out)
    seconds = time.monotonic() - began
    text = report(epochs)
    sys.stdout.write(text)
    sys.stdout.write(f"\nbuilt and registered {len(epochs)} epochs in {seconds:.1f} s\n")
    document = {
        "seconds": round(seconds, 1),
        "rows": rows(epochs),
        "alignments": {name: epoch.alignment for name, epoch in epochs.items()},
    }
    (args.out / "report.json").write_text(json.dumps(document, indent=1) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
