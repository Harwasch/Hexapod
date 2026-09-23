"""Getting a finished stage out of the workdir: its log, and its artifacts.

Two rules, both from the plan:

* **logs go to object storage, not the database.** `job_steps.log_key` is a key, and a
  log drawer reads the object. A run's logs are unbounded and a database row is the wrong
  place for them.
* **an artifact in the bucket has a row.** Without one there is no Outputs view, no
  reconciliation in either direction, and "retry from this stage" cannot say what it is
  replacing (A2's note on the `artifacts` table).

Keys are `runs/<job id>/<stage id>/…`, which is the same shape as the `checkpoint_key`
A6 already puts in the StepResult, so a run's whole footprint is one prefix.
"""

from __future__ import annotations

import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.models.enums import ArtifactKind
from app.storage import ObjectStorage
from app.storage.null import StorageUnavailableError
from app.worker.pipeline_bridge import ArtifactRef

#: Pipeline artifact name -> the kind A2's `artifacts.kind` records.
#:
#: Anything not named here is `metadata`: the JSON sidecars a stage writes beside its real
#: output (`georef.json`, `source_meta.json`, `train_metrics.json`, `registration.json`).
#: A new artifact in B2 therefore gets a row rather than crashing the worker, and
#: gets a better kind by being added here.
ARTIFACT_KINDS: dict[str, ArtifactKind] = {
    "frames": ArtifactKind.FRAMES,
    "poses": ArtifactKind.POSES,
    "masks": ArtifactKind.MASKS,
    # The splat itself, before packaging: both lanes converge on it.
    "canonical.ply": ArtifactKind.SPLAT,
    # The packaged tileset the console renders.
    "splat": ArtifactKind.TILES_3D,
    "mesh": ArtifactKind.MESH,
    "pointcloud": ArtifactKind.POINT_CLOUD,
    "thumbnail.jpg": ArtifactKind.THUMBNAIL,
    "ground_samples.json": ArtifactKind.GROUND_SAMPLES,
    "manifest.json": ArtifactKind.MANIFEST,
    "motion/clip.f32": ArtifactKind.CLIP,
    "motion/field.npz": ArtifactKind.DEFORMATION_FIELD,
}


def kind_for(name: str) -> ArtifactKind:
    return ARTIFACT_KINDS.get(name, ArtifactKind.METADATA)


def run_prefix(job_id: uuid.UUID) -> str:
    return f"runs/{job_id}"


def stage_prefix(job_id: uuid.UUID, stage_id: str) -> str:
    return f"{run_prefix(job_id)}/{stage_id}"


def log_key(job_id: uuid.UUID, stage_id: str) -> str:
    return f"{stage_prefix(job_id, stage_id)}/log.txt"


def artifact_key(job_id: uuid.UUID, stage_id: str, name: str) -> str:
    return f"{stage_prefix(job_id, stage_id)}/{name}"


def checkpoint_key(job_id: uuid.UUID, stage_id: str) -> str:
    """Where a dispatched stage's checkpoint lives, stated once on this side too.

    It is the same string `BaseRunner` puts in `StageContext.checkpoint_key`, built from
    the same two facts (the run id is the workdir's directory name, which is the job id).
    The supervisor needs it for a step that did *not* finish -- a preempted attempt has
    a checkpoint and no StepResult to read it out of.
    """
    return f"{stage_prefix(job_id, stage_id)}/checkpoint"


@dataclass(frozen=True)
class UploadedArtifact:
    """One row's worth of `artifacts`, ready to insert."""

    kind: ArtifactKind
    storage_key: str
    bytes: int
    checksum: str
    content_type: str


def _content_type(path: Path, fallback: str) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or fallback


def upload_log(
    storage: ObjectStorage, workdir_root: Path, job_id: uuid.UUID, stage_id: str
) -> str | None:
    """Put the stage's log in the bucket and return its key, or None if there is nowhere
    to put it. A deployment with no bucket still runs; it just has no log drawer."""
    path = workdir_root / "stages" / stage_id / "log.txt"
    if not path.is_file():
        return None
    key = log_key(job_id, stage_id)
    try:
        storage.put_object(key, path.read_bytes(), "text/plain; charset=utf-8")
    except StorageUnavailableError:
        return None
    return key


def upload_artifact(
    storage: ObjectStorage, workdir_root: Path, job_id: uuid.UUID, ref: ArtifactRef
) -> UploadedArtifact | None:
    """Upload one artifact and describe the row it becomes.

    A directory artifact is uploaded member by member under one prefix, and still becomes
    **one** row — the artifact is the tileset, not each of its tiles. `storage_key` is
    that prefix, `bytes` and `checksum` are the pipeline's own figures for the whole
    directory, so the row and `artifacts.json` agree without recomputing anything.
    """
    source = workdir_root / ref.path
    key = artifact_key(job_id, ref.stage_id, ref.name)
    try:
        if ref.kind == "dir":
            if not source.is_dir():
                return None
            for member in sorted(p for p in source.rglob("*") if p.is_file()):
                relative = member.relative_to(source).as_posix()
                storage.put_object(
                    f"{key}/{relative}",
                    member.read_bytes(),
                    _content_type(member, "application/octet-stream"),
                )
        else:
            if not source.is_file():
                return None
            storage.put_object(key, source.read_bytes(), ref.content_type)
    except StorageUnavailableError:
        return None
    return UploadedArtifact(
        kind=kind_for(ref.name),
        storage_key=key,
        bytes=ref.bytes,
        checksum=ref.checksum,
        content_type=ref.content_type,
    )
