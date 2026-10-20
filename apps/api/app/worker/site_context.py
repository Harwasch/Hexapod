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
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError
from sqlalchemy.orm import Session

from app.models import Capture
from app.models.enums import UploadStatus
from app.services import check_targets, epochs
from app.storage import ObjectStorage
from app.worker.pipeline_bridge import SITE_CONTEXT_FILE, empty_site_context

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
#: The two reference artifacts that are directories (the others are single files).
DIRECTORIES = frozenset({"frames", "poses"})


@dataclass(frozen=True)
class SeedPlan:
    """Everything the seed needs from the database, read in one transaction."""

    site_id: uuid.UUID | None = None
    reference: epochs.ReferenceChoice | None = None
    targets: list[dict[str, Any]] = field(default_factory=list)
    observations_key: str | None = None


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


def _download_tree(storage: ObjectStorage, prefix: str, target: Path) -> int:
    """Every object under `prefix/` into `target`, file by file, skipping what is there
    (a download writes beside its target and renames, so a present file is whole)."""
    count = 0
    token: str | None = None
    while True:
        page = storage.list_objects(f"{prefix}/", continuation_token=token)
        for item in page.objects:
            relative = item.key[len(prefix) + 1 :]
            path = (target / relative).resolve()
            if not relative or not path.is_relative_to(target.resolve()):
                continue
            count += 1
            if not path.is_file():
                _download(storage, item.key, path)
        token = page.next_continuation_token
        if token is None:
            return count


def _fetch_reference(
    storage: ObjectStorage, reference: epochs.ReferenceChoice, directory: Path
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
    for role, key in reference.keys.items():
        path = root / names[role]
        try:
            if role in DIRECTORIES:
                if _download_tree(storage, key, path) == 0:
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
    return {
        "epochId": str(reference.epoch_id),
        "jobId": str(reference.job_id),
        "observedAt": reference.observed_at.isoformat(),
        "platform": reference.platform,
        "verdict": reference.verdict,
        "uncertaintyM": reference.uncertainty_m,
        "revision": reference.revision,
        **{role: f"{REFERENCE_DIR}/{names[role]}" for role in epochs.REFERENCE_ARTIFACTS},
    }


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
) -> None:
    """Write `directory/site_context.json` (and the files it names) unless it exists."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SITE_CONTEXT_FILE
    if path.exists():
        return
    plan = plan_seed(db, capture, located)
    _end_transaction(db)
    document: dict[str, Any] = empty_site_context(
        str(plan.site_id) if plan.site_id is not None else None
    )
    if plan.reference is not None:
        document["reference"] = _fetch_reference(storage, plan.reference, directory)
    document["targets"] = plan.targets
    if plan.observations_key is not None:
        document["observations"] = _observations(storage, plan.observations_key, directory)
    staging = path.with_suffix(".json.tmp")
    staging.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    staging.replace(path)
