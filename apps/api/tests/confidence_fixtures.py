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
