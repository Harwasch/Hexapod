"""What the site knows before a run: the `site_context` input (M3b; decision D4).

`tools/pipeline/alignment.py` (`align_epoch`) registers a new epoch's frames into an
earlier epoch of its site -- the **reference** -- and scores it against the site's
surveyed **check targets**, using this capture's **clicks** on them. The worker is what
knows all three, so it seeds them into `inputs/site_context/` before the first stage:

* the site is the one the run belongs to (`services/epochs.site_for_run`, the same answer
  the run's frame comes from), or none for a capture that will make a new site;
* the reference is `services/epochs.reference_epoch`: its kept frames, its pose model,
  its `placement.json` and its `georef.json`, downloaded from the bucket file by file
  under `reference/`, with its verdict, platform and uncertainty beside them;
* the targets are the site's active `site_check_targets` rows;
* the observations are the capture's own uploaded `check_observations.json`
  (`{"version": 1, "observations": [...]}`), when it has a complete one.

`site_context.json` is written **last**, through a temporary file, and a context that
already has one is never seeded again: a resumed or retried run keeps the reference and
targets it started with, so it is never measured against a different scaffold halfway.
A seed interrupted before the JSON was written starts over -- with the reference it now
chooses, after clearing the files of another one (`.reference.json` names whose they are).

The database is read first, as plain values, and the transaction ended before a byte is
downloaded (as `runner._seed` does for the upload): a reference's frames are minutes on a
slow link, and a session idle in a transaction that long is killed by the database.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError
from sqlalchemy.orm import Session

from app.models import Capture
from app.models.enums import UploadStatus
from app.services import check_targets, epochs
from app.storage import ObjectStorage
from app.storage.base import ObjectSummary
from app.worker.pipeline_bridge import (
    SITE_CONTEXT_FILE,
    empty_site_context,
    image_names,
    sample_evenly,
)

log = logging.getLogger("app.worker")

#: The capture file a person's (or a detector's) clicks on the targets arrive in.
OBSERVATIONS_FILE = "check_observations.json"
#: Where the reference epoch's files go, relative to `site_context/`.
REFERENCE_DIR = "reference"
#: Which epoch's files, of which revision, `reference/` holds, so an interrupted seed
#: never mixes two references, nor two revisions of one.
REFERENCE_MARKER = ".reference.json"
#: The most a capture's clicks file may be and still be read: a click is ~100 bytes, so
#: 5 MB is tens of thousands of them, and anything larger is not clicks. Checked before a
#: byte is downloaded -- the file is read whole, on a worker with 2 GB.
MAX_OBSERVATIONS_BYTES = 5 * 1024 * 1024
#: The reference's placement as it is shown, when it was resized at runtime (`_shown`).
SHOWN_PLACEMENT = "placement.shown.json"
#: How close the reference's georef origin must be to the site frame's to be the pivot a
#: runtime scale is about: the same numbers, to float round-off (not a tolerance).
SAME_ORIGIN_DEG, SAME_ORIGIN_M = 1e-9, 1e-6
#: The two reference artifacts that are directories (the others are single files).
DIRECTORIES = frozenset({"frames", "poses"})


@dataclass(frozen=True)
class SeedPlan:
    """Everything the seed needs from the database, read in one transaction."""

    site_id: uuid.UUID | None = None
    reference: epochs.ReferenceChoice | None = None
    targets: list[dict[str, Any]] = field(default_factory=list)
    observations_key: str | None = None
    #: The site's published epochs before this run's (`epochs.prior_epoch_count`).
    prior_epochs: int = 0


def _target(row: Any) -> dict[str, Any]:
    """A `site_check_targets` row in `site_context.json`'s shape (design section 4.1)."""
    return {
        "label": row.label,
        "role": row.role,
        "lat": row.lat,
        "lon": row.lon,
        "height": row.height,
        "heightDatum": row.height_datum,
        "sigmaHM": row.sigma_h_m,
        "sigmaVM": row.sigma_v_m,
    }


def plan_seed(
    db: Session, capture: Capture | None, located: tuple[float, float] | None = None
) -> SeedPlan:
    """What a run of `capture` is seeded with, read now (no I/O but the database)."""
    if capture is None:
        return SeedPlan()
    observations = next(
        (
            source.storage_key
            for source in capture.files
            if source.status is UploadStatus.COMPLETE
            and Path(source.filename).name == OBSERVATIONS_FILE
        ),
        None,
    )
    site_id = epochs.site_for_run(db, capture, located)
    if site_id is None:
        return SeedPlan(observations_key=observations)
    return SeedPlan(
        site_id=site_id,
        reference=epochs.reference_epoch(db, site_id, capture_id=capture.id),
        targets=[_target(row) for row in check_targets.list_active(db, site_id)],
        observations_key=observations,
        prior_epochs=epochs.prior_epoch_count(db, site_id, capture_id=capture.id),
    )


def _end_transaction(db: Session) -> None:
    if db.in_transaction():
        db.commit()


class Vanished(Exception):  # noqa: N818 - an event, as `LiveMoved` is
    """An object that was there a moment ago is not: listed, then deleted before its
    download (a lifecycle rule, a person tidying the bucket)."""


def _download(storage: ObjectStorage, key: str, path: Path) -> None:
    """`download_file`, with an object that is gone by the time it is fetched raised as
    `Vanished`; any other failure is what it is (a transient one dead-letters the run as
    a failed fetch of the capture's own files does)."""
    try:
        storage.download_file(key, path)
    except ClientError:
        if storage.head_object(key) is None:
            raise Vanished(key) from None
        raise


def _download_tree(
    storage: ObjectStorage,
    prefix: str,
    target: Path,
    *,
    names: set[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Every object under `prefix/` into `target`, file by file, skipping what is there
    (a download writes beside its target and renames, so a present file is whole)."""
    items: list[ObjectSummary] = []
    token: str | None = None
    while True:
        page = storage.list_objects(f"{prefix}/", continuation_token=token)
        items.extend(page.objects)
        token = page.next_continuation_token
        if token is None:
            break
    selected = []
    for item in items:
        relative = item.key[len(prefix) + 1 :]
        path = (target / relative).resolve()
        if (
            relative
            and path.is_relative_to(target.resolve())
            and (names is None or relative in names)
        ):
            selected.append((item, path))
    if names is not None and len(selected) != len(names):
        raise Vanished(prefix)
    total = sum(item.size for item, _ in selected)
    began = time.monotonic()
    log.info("site context: preparing %s (%d files, %d bytes)", target, len(selected), total)
    done = 0
    if progress is not None:
        progress(done, total)
    for item, path in selected:
        if not path.is_file():
            _download(storage, item.key, path)
        done += item.size
        if progress is not None:
            progress(done, total)
    log.info("site context: ready %s (%d bytes), %.1f s", target, total, time.monotonic() - began)
    return len(selected)


def _fetch_reference(
    storage: ObjectStorage,
    reference: epochs.ReferenceChoice,
    directory: Path,
    *,
    frame_limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any] | None:
    """The reference's files under `reference/` and its block of `site_context.json`, or
    None (logged) when the bucket no longer has one of them: the run then aligns by its
    own georeference rather than failing on a scaffold that is gone."""
    root = directory / REFERENCE_DIR
    marker = root / REFERENCE_MARKER
    # By revision, not only by epoch and job: a retry of the reference's run rewrites the
    # same keys under the same ids, and files of two revisions must never be mixed.
    whose = {
        "epochId": str(reference.epoch_id),
        "jobId": str(reference.job_id),
        "revision": reference.revision,
        "runtimeScale": reference.runtime_scale,
    }
    try:
        held = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else None
    except ValueError:
        held = None
    if root.exists() and held != whose:
        # Another reference's files, from a seed interrupted before it chose this one.
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(whose), encoding="utf-8")
    names = epochs.REFERENCE_ARTIFACTS
    # Poses first: only registered names participate in the aligner's sample.
    for role, key in sorted(reference.keys.items(), key=lambda pair: pair[0] == "frames"):
        path = root / names[role]
        try:
            if role in DIRECTORIES:
                selected = None
                if role == "frames" and frame_limit is not None:
                    selected = set(
                        sample_evenly(
                            image_names(root / names["poses"] / "images.bin"), frame_limit
                        )
                    )
                    # A seed interrupted on the previous worker version may have fetched
                    # the entire capture. Keep only what alignment will actually read.
                    if path.is_dir():
                        for held_frame in path.rglob("*"):
                            if (
                                held_frame.is_file()
                                and held_frame.relative_to(path).as_posix() not in selected
                            ):
                                held_frame.unlink()
                if _download_tree(storage, key, path, names=selected, progress=progress) == 0:
                    raise Vanished(key)
            elif not path.is_file():
                if storage.head_object(key) is None:
                    raise Vanished(key)
                _download(storage, key, path)
        except Vanished as gone:
            log.warning(
                "site context: reference epoch %s has no %s in the bucket (%s is gone); "
                "the run is aligned by its own georeference",
                reference.epoch_id,
                role,
                gone,
            )
            return None
    placement = f"{REFERENCE_DIR}/{names['placement']}"
    if reference.runtime_scale != 1.0:
        problem = _shown(root, reference)
        if problem is not None:
            log.warning(
                "site context: reference epoch %s is drawn at %g its registered size, but %s; "
                "not used, the run is aligned by its own georeference",
                reference.epoch_id,
                reference.runtime_scale,
                problem,
            )
            return None
        placement = f"{REFERENCE_DIR}/{SHOWN_PLACEMENT}"
    return {
        "epochId": str(reference.epoch_id),
        "jobId": str(reference.job_id),
        "observedAt": reference.observed_at.isoformat(),
        "platform": reference.platform,
        "verdict": reference.verdict,
        "uncertaintyM": reference.uncertainty_m,
        "revision": reference.revision,
        "heightDatum": reference.height_datum,
        "runtimeScale": reference.runtime_scale,
        "frameRelativeUncertaintyM": reference.frame_relative_uncertainty_m,
        **{role: f"{REFERENCE_DIR}/{names[role]}" for role in epochs.REFERENCE_ARTIFACTS},
        "placement": placement,
        # Internal seed metadata: a retry may change align.max_frames. Fetch its new
        # sample from this frozen reference, without choosing a different epoch.
        **(
            {"frameSource": reference.keys["frames"], "frameLimit": frame_limit}
            if frame_limit is not None
            else {}
        ),
    }


def _shown(root: Path, reference: epochs.ReferenceChoice) -> str | None:
    """Write `SHOWN_PLACEMENT`: the reference's placement as the viewer draws it, or say
    why it cannot be (R2 review of #52: "Set real size" resizes the shown tiles while the
    files stay as registered, and a later epoch aligned to the files came out at the
    uncorrected size beside the corrected reference).

    `PUT /assets/{id}/scale` resizes an epoch's splat by `s` about its site frame's origin
    (`assets._placed_epoch`). The placement takes the model into east/north/up about the
    reference's georef origin; when that is the frame's origin, the shown placement is
    exactly `s * placement`: scale and translation times `s`, rotation unchanged. When it is
    not, the pivot is elsewhere in that frame and nothing here is guessed: refused. Always
    computed from the downloaded `placement.json`, which is kept as is, so a resumed seed
    never scales twice."""
    scale = reference.runtime_scale
    if not math.isfinite(scale) or scale <= 0:
        return f"its runtime scale {scale!r} is not a positive number"
    try:
        georef = json.loads((root / epochs.REFERENCE_ARTIFACTS["georef"]).read_text("utf-8"))
        placement = json.loads((root / epochs.REFERENCE_ARTIFACTS["placement"]).read_text("utf-8"))
    except (OSError, ValueError) as error:
        return f"its placement or georef cannot be read ({error})"
    origin = reference.frame_origin
    raw = [georef.get(k) for k in ("lat", "lon", "height")] if isinstance(georef, dict) else []
    here = [float(v) for v in raw if isinstance(v, int | float) and not isinstance(v, bool)]
    if (
        origin is None
        or len(here) != 3
        or abs(here[0] - origin[0]) > SAME_ORIGIN_DEG
        or abs(here[1] - origin[1]) > SAME_ORIGIN_DEG
        or abs(here[2] - origin[2]) > SAME_ORIGIN_M
    ):
        return "its georef origin is not the site frame's, about which it is resized"
    if not isinstance(placement, dict) or not isinstance(placement.get("scale"), int | float):
        return "its placement has no scale"
    shown = dict(placement)
    shown["scale"] = float(placement["scale"]) * scale
    for key in ("translationM", "translation"):
        value = placement.get(key)
        if isinstance(value, list):
            shown[key] = [float(v) * scale for v in value]
    shown["runtimeScale"] = scale
    (root / SHOWN_PLACEMENT).write_text(json.dumps(shown, indent=1) + "\n", encoding="utf-8")
    return None


def _observations(storage: ObjectStorage, key: str, directory: Path) -> list[Any]:
    """The clicks in the capture's `check_observations.json`; a file that is not that
    shape is logged and contributes none (a stray file must not fail a capture -- the
    pipeline is as forgiving of a stray click)."""
    path = directory / OBSERVATIONS_FILE
    if not path.is_file():
        found = storage.head_object(key)
        if found is None:
            log.warning(
                "site context: %s is not in the bucket (%s); no observations",
                OBSERVATIONS_FILE,
                key,
            )
            return []
        if found.size > MAX_OBSERVATIONS_BYTES:
            log.warning(
                "site context: %s is %d bytes, over the %d a clicks file may be; not read, "
                "no observations",
                OBSERVATIONS_FILE,
                found.size,
                MAX_OBSERVATIONS_BYTES,
            )
            return []
        try:
            _download(storage, key, path)
        except Vanished:
            log.warning(
                "site context: %s vanished before its download; no observations", OBSERVATIONS_FILE
            )
            return []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        log.warning("site context: %s is not JSON (%s); no observations", OBSERVATIONS_FILE, error)
        return []
    rows = document.get("observations") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        log.warning("site context: %s has no observations list; none used", OBSERVATIONS_FILE)
        return []
    return rows


def seed_site_context(
    db: Session,
    storage: ObjectStorage,
    capture: Capture | None,
    directory: Path,
    *,
    located: tuple[float, float] | None = None,
    reference_frame_limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Write `directory/site_context.json` (and the files it names) unless it exists."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SITE_CONTEXT_FILE
    if path.exists():
        document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        reference = document.get("reference")
        if (
            reference_frame_limit is not None
            and isinstance(reference, dict)
            and isinstance(reference.get("frameSource"), str)
            and reference.get("frameLimit") != reference_frame_limit
        ):
            # The context and pose model stay frozen even when the site's current
            # reference has changed. Only hydrate the new aligner's required JPEGs.
            selected = set(
                sample_evenly(
                    image_names(directory / reference["poses"] / "images.bin"),
                    reference_frame_limit,
                )
            )
            _end_transaction(db)
            try:
                _download_tree(
                    storage,
                    reference["frameSource"],
                    directory / reference["frames"],
                    names=selected,
                    progress=progress,
                )
                reference["frameLimit"] = reference_frame_limit
            except Vanished:
                log.warning("site context: frozen reference's new frame sample vanished")
                document["reference"] = None
            staging = path.with_suffix(".json.tmp")
            staging.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
            staging.replace(path)
        return
    plan = plan_seed(db, capture, located)
    _end_transaction(db)
    document = empty_site_context(str(plan.site_id) if plan.site_id is not None else None)
    if plan.reference is not None:
        document["reference"] = _fetch_reference(
            storage, plan.reference, directory, frame_limit=reference_frame_limit, progress=progress
        )
    document["targets"] = plan.targets
    # Whether this epoch makes the site's frame (0) or is placed relative to one: what
    # caps its confidence grid when it could not be aligned (ADR 0016).
    document["priorEpochs"] = plan.prior_epochs
    if plan.observations_key is not None:
        document["observations"] = _observations(storage, plan.observations_key, directory)
    staging = path.with_suffix(".json.tmp")
    staging.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    staging.replace(path)
