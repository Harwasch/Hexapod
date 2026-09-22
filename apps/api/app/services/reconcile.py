"""Storage against the database, in both directions.

A1 made `ObjectStorage.list_objects` paginated *specifically* for this, and this is the
call that uses it: a bucket is unbounded, so the walk is a loop over pages with a ceiling
on it rather than a listing that assumes it fits in memory.

The two directions are answered by two different mechanisms on purpose:

* **orphans** come from walking the bucket and asking, per object, whether any row claims
  it. That needs the walk, and a truncated walk means a sampled answer, which the response
  says.
* **missing** comes from taking each row and asking storage directly whether its object
  is there. That is exact whether or not the walk was truncated, which matters because
  "this site's tileset is gone" is the half a person acts on immediately.

What claims an object is deliberately wider than the `artifacts` table. A run's footprint
under `runs/<job id>/` is its artifacts *plus* its step logs and its checkpoints, and a
checkpoint is not an artifact (A2: logs and checkpoints are `job_steps.log_key` and
`job_steps.checkpoint_key`). Treating them as orphans would report every healthy run as
garbage, which is the fastest way to make a reconciliation view ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Artifact, CaptureFile, Job, JobStep
from app.models.enums import UploadStatus
from app.schemas.storage import MissingObject, OrphanObject, StorageReconciliation
from app.storage import ObjectStorage

#: The prefixes this system writes. `sites/` is deliberately not one of them: A9 publishes
#: the migrated captures there out of `data/tiles`, and those objects are described by no
#: row at all — reporting every one of them as an orphan would be reporting the migration.
RECONCILED_PREFIXES: tuple[str, ...] = ("captures/", "runs/")

#: How many objects one pass will walk before it stops and says it stopped.
DEFAULT_MAX_OBJECTS = 5000

#: Page size for the walk. The S3 maximum; nothing is gained by asking for less.
_PAGE = 1000


@dataclass
class _Claims:
    """Every key the database says should exist, and who says so.

    `exact` are keys an object must equal. `prefixes` are keys an object may live *under*:
    a directory artifact is one row whose `storage_key` is the prefix its members sit in
    (A7's `upload_artifact`), and a checkpoint is a prefix for the same reason.

    `checkable` is the subset whose object is supposed to be in the bucket *right now*, and
    it is the one the missing half iterates. Two things are claimed but not checked: a
    capture file that is still uploading, was aborted or failed has no completed object
    yet, and a checkpoint is deleted once the stage it was resuming finishes, so its
    absence is the normal end state rather than a fault.
    """

    exact: dict[str, MissingObject] = field(default_factory=dict)
    prefixes: dict[str, MissingObject] = field(default_factory=dict)
    checkable: list[MissingObject] = field(default_factory=list)

    def claims(self, key: str) -> bool:
        if key in self.exact:
            return True
        # Walk the key's ancestors rather than scanning every prefix: a key has a handful
        # of path segments and a bucket has an unbounded number of prefixes.
        cut = key.rfind("/")
        while cut > 0:
            if key[:cut] in self.prefixes:
                return True
            cut = key.rfind("/", 0, cut)
        return False


def collect_claims(db: Session) -> _Claims:
    claims = _Claims()

    for file in db.scalars(select(CaptureFile)).all():
        row = MissingObject(
            kind="capture-file",
            id=file.id,
            key=file.storage_key,
            label=file.filename,
            capture_id=file.capture_id,
            job_id=None,
        )
        claims.exact[file.storage_key] = row
        if file.status is UploadStatus.COMPLETE:
            claims.checkable.append(row)

    artifacts = db.execute(
        select(Artifact, JobStep, Job)
        .join(JobStep, Artifact.job_step_id == JobStep.id)
        .join(Job, JobStep.job_id == Job.id)
    ).all()
    for artifact, step, job in artifacts:
        row = MissingObject(
            kind="artifact",
            id=artifact.id,
            key=artifact.storage_key,
            label=f"{step.stage_id} · {artifact.kind.value}",
            capture_id=job.capture_id,
            job_id=job.id,
        )
        # Both maps: an artifact is a file *or* a directory and the row does not record
        # which. Claiming the key and everything under it is right either way, and cheaper
        # than asking storage what shape it is.
        claims.exact[artifact.storage_key] = row
        claims.prefixes[artifact.storage_key] = row
        claims.checkable.append(row)

    for step, job in db.execute(select(JobStep, Job).join(Job, JobStep.job_id == Job.id)).all():
        if step.log_key:
            log = MissingObject(
                kind="step-log",
                id=step.id,
                key=step.log_key,
                label=f"{step.stage_id} log",
                capture_id=job.capture_id,
                job_id=job.id,
            )
            claims.exact[step.log_key] = log
            claims.checkable.append(log)
        if step.checkpoint_key:
            checkpoint = MissingObject(
                kind="checkpoint",
                id=step.id,
                key=step.checkpoint_key,
                label=f"{step.stage_id} checkpoint",
                capture_id=job.capture_id,
                job_id=job.id,
            )
            claims.exact[step.checkpoint_key] = checkpoint
            claims.prefixes[step.checkpoint_key] = checkpoint
    return claims


def _reason(key: str) -> str:
    """Why nothing claims this object, said in terms of the key's own shape."""
    parts = key.split("/")
    if key.startswith("runs/") and len(parts) > 2:
        return f"under runs/{parts[1]}: no artifact, log or checkpoint row claims it"
    if key.startswith("captures/") and len(parts) > 2:
        return f"under captures/{parts[1]}: no capture_files row claims it"
    return "no row claims this key"


def _present(storage: ObjectStorage, key: str) -> bool:
    """Whether the bucket has this key — as an object, or as a directory of them.

    Two calls at worst, because one row can be either: `upload_artifact` writes a file
    artifact at the key and a directory artifact's members *under* it. Asking storage is
    what makes this half exact even when the walk was cut short.
    """
    if storage.head_object(key) is not None:
        return True
    return bool(storage.list_objects(f"{key}/", max_keys=1).objects)


def _walk(
    storage: ObjectStorage, claims: _Claims, prefix: str, budget: int
) -> tuple[list[OrphanObject], int, int, int, bool]:
    """One prefix, page by page, until it is exhausted or `budget` objects were seen."""
    orphans: list[OrphanObject] = []
    scanned = matched = total_bytes = 0
    token: str | None = None
    while True:
        page = storage.list_objects(prefix, continuation_token=token, max_keys=_PAGE)
        for item in page.objects:
            if scanned >= budget:
                return orphans, scanned, matched, total_bytes, True
            scanned += 1
            total_bytes += item.size
            if claims.claims(item.key):
                matched += 1
                continue
            orphans.append(
                OrphanObject(
                    key=item.key,
                    bytes=item.size,
                    last_modified=item.last_modified,
                    etag=item.etag,
                    reason=_reason(item.key),
                )
            )
        token = page.next_continuation_token
        if token is None:
            return orphans, scanned, matched, total_bytes, False


def reconcile(
    db: Session,
    storage: ObjectStorage,
    *,
    prefixes: tuple[str, ...] = RECONCILED_PREFIXES,
    max_objects: int = DEFAULT_MAX_OBJECTS,
) -> StorageReconciliation:
    claims = collect_claims(db)
    orphans: list[OrphanObject] = []
    scanned = matched = bytes_scanned = 0
    truncated = False

    for prefix in prefixes:
        if scanned >= max_objects:
            truncated = True
            break
        found, seen, hit, size, cut = _walk(storage, claims, prefix, max_objects - scanned)
        orphans.extend(found)
        scanned += seen
        matched += hit
        bytes_scanned += size
        truncated = truncated or cut

    missing = [row for row in claims.checkable if not _present(storage, row.key)]
    return StorageReconciliation(
        prefixes=list(prefixes),
        scanned=scanned,
        truncated=truncated,
        matched=matched,
        bytes_scanned=bytes_scanned,
        bytes_orphaned=sum(orphan.bytes for orphan in orphans),
        orphans=sorted(orphans, key=lambda orphan: orphan.key),
        rows_checked=len(claims.checkable),
        missing=sorted(missing, key=lambda row: row.key),
        checked_at=datetime.now(tz=UTC),
    )
