"""The committed confidence grids (data/tiles/synthetic-yard-epochs), as API tests use them.

Two epochs of the synthetic yard written by `tools/captures/synthetic_cell_grids.py` from
integer arithmetic: e1 at level 2 (1 m cells), weak in its north half; e2 at level 1
(0.5 m), the re-scan, weak only in a north-east patch for GSD. Fixtures, not measurements.
"""

from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models import Capture
from app.storage import S3Storage
from app.worker import registration
from app.worker.carry import CarryPlan
from tests.test_asset_scale import _document, _job, _registration

FIXTURE = Path(__file__).resolve().parents[3] / "data" / "tiles" / "synthetic-yard-epochs"
GRIDS: dict[str, Any] = json.loads((FIXTURE / "grids.json").read_text(encoding="utf-8"))


def descriptor(epoch: str) -> dict[str, Any]:
    """The epoch's descriptor, as its tileset's `extras.confidence` would hold it."""
    found: dict[str, Any] = copy.deepcopy(GRIDS["epochs"][epoch]["descriptor"])
    return found


def blob(epoch: str) -> bytes:
    return (FIXTURE / epoch / "confidence.bin").read_bytes()


def measured(epoch: str | None, document: dict[str, Any] | None = None) -> dict[str, Any]:
    """`registration.json` (`_document()` unless given) with the epoch's grid, as `catalog`
    copies it up, or with none (`epoch` None: a run that measured nothing)."""
    out = copy.deepcopy(document) if document is not None else _document()
    if epoch is not None:
        out["confidence"] = descriptor(epoch)
    return out


def tiles_url(job_id: uuid.UUID) -> str:
    return f"https://cdn.example.com/twin-worker-test/runs/{job_id}/package/splat/tileset.json"


def over_its_grid(document: dict[str, Any]) -> dict[str, Any]:
    """`document` with its splat's box grown to hold its confidence grid's finest level, as
    a run's grid lies over the splat it measured: the site boundary registration grows from
    that box is then the ground the grid's weak areas are clipped to (ADR 0016). With a
    metre to spare: registration places the footprint by its own degree arithmetic, a few
    centimetres off the frame's tangent plane over these grids, and these tests are about
    the grid, not that edge (test_weak_areas_api clips to a boundary on purpose)."""
    grid = document.get("confidence")
    box = document.get("bboxLocalM")
    if not isinstance(grid, dict) or not isinstance(box, dict):
        return document
    level = grid["levels"][0]
    cell = float(grid["cellM"])
    west, south = level["i0"] * cell - 1.0, level["j0"] * cell - 1.0
    east, north = west + level["ni"] * cell + 2.0, south + level["nj"] * cell + 2.0
    out = copy.deepcopy(document)
    out["bboxLocalM"] = {
        "min": [min(box["min"][0], west), min(box["min"][1], south), box["min"][2]],
        "max": [max(box["max"][0], east), max(box["max"][1], north), box["max"][2]],
    }
    return out


def register_run(
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    capture: Capture,
    document: dict[str, Any],
    *,
    url: str | None = None,
    grid: bytes | None = None,
    carried: tuple[str, ...] = (),
    content_addressed: bool = False,
) -> uuid.UUID:
    """One run registered as the worker registers it (published to `url`, by default the
    run's own key in the one test bucket), with `grid` written beside the tiles; its job."""
    document = over_its_grid(document)
    job = _job(db, capture)
    tileset = url or tiles_url(job.id)
    if grid is not None:
        directory = tileset.removeprefix("https://cdn.example.com/twin-worker-test/").rsplit(
            "/", 1
        )[0]
        uri = str(document.get("confidence", {}).get("uri") or "confidence.bin")
        storage.put_object(f"{directory}/{uri}", grid, "application/octet-stream")
    live = registration.live_tileset_url(db, capture)
    registration.register(
        db,
        storage,
        capture=capture,
        job_id=job.id,
        registration=_registration(tmp_path, document),
        tiles_stage_id="package",
        published=registration.Published(
            tileset=tileset,
            thumbnail=None,
            coverage=None,
            carry=CarryPlan(based_on=live, carried=carried),
            content_addressed=content_addressed,
        ),
    )
    return job.id


def encode_grid(
    level: int,
    i0: int,
    j0: int,
    score: Any,
    reasons: Any,
    relief: Any | None = None,
    *,
    cap: dict[str, Any] | None = None,
    median_gsd_mm: float | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """A one-level confidence grid in the container's own layout (ADR 0016, C2) and its
    descriptor with a weak-v1 summary: for cases the committed fixtures do not cover."""
    import gzip
    import hashlib
    import math

    import numpy as np

    score = np.asarray(score, dtype=np.uint8)
    reasons = np.asarray(reasons, dtype=np.uint8)
    relief = np.zeros_like(score) if relief is None else np.asarray(relief, dtype=np.uint8)
    nj, ni = score.shape
    raw = np.stack([score, reasons, relief], axis=-1).tobytes()
    weak = ((score > 0) & (score < 128)) | ((reasons & 16) > 0)
    unseen = (reasons & 16) > 0
    scored = score > 0
    bits = {"views": 1, "spread": 2, "gsd": 4, "accuracy": 8, "unseen": 16, "alignment": 32}
    expected = int(scored.sum() + unseen.sum())
    descriptor: dict[str, Any] = {
        "format": "hexapod.cellgrid",
        "version": 1,
        "kind": "confidence",
        "uri": "confidence.bin",
        "frame": "tileset",
        "lattice": {"baseM": 0.25, "level": level},
        "cellM": math.ldexp(0.25, level),
        "channels": [
            {"name": "score", "type": "u1"},
            {"name": "reasons", "type": "u1"},
            {"name": "relief", "type": "u1", "unitM": 0.25},
        ],
        "levels": [
            {
                "level": level,
                "cellM": math.ldexp(0.25, level),
                "i0": i0,
                "j0": j0,
                "ni": ni,
                "nj": nj,
                "offset": 0,
                "bytes": len(raw),
            }
        ],
        "reasonBits": bits,
        "policy": "weak-v1",
        "weakBelow": 128,
        "alignmentCap": cap
        or {"uncertaintyM": 0.0, "priorEpochs": 0, "factor": 1.0, "source": "site_context.json"},
        "roiDisc": None,
        "summary": {
            "cells": ni * nj,
            "scored": int(scored.sum()),
            "weak": int(weak.sum()),
            "unseen": int(unseen.sum()),
            "weakShare": round(int(weak.sum()) / expected, 4) if expected else 0.0,
            "reasons": {n: int((weak & ((reasons & b) > 0)).sum()) for n, b in bits.items()},
            "gaussians": 0,
            "outside": 0,
            "medianGsdMm": median_gsd_mm,
            "metric": median_gsd_mm is not None,
        },
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return gzip.compress(raw, compresslevel=9, mtime=0), descriptor
