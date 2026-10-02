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

import json
import logging
import mimetypes
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.models.enums import ArtifactKind
from app.storage import ObjectStorage
from app.storage.null import StorageUnavailableError
from app.worker.parallel import each
from app.worker.pipeline_bridge import ArtifactRef, Workdir
from app.worker.pipeline_bridge import checkpoint_key as stage_checkpoint_key

log = logging.getLogger("app.worker")

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


def transfer_outputs_key(job_id: uuid.UUID, stage_id: str, attempt: int = 1) -> str:
    """Where attempt `attempt` of a stage dispatched to a provider had its `out/` put.

    The pipeline's `StageKeys.outputs`, stated once on this side as `checkpoint_key` below
    is: `runs/<job>/<stage>/transfer/out`, under `transfer/` so it cannot collide with
    the per-artifact keys uploaded here, with the attempt's suffix after the first
    (`runners.per_attempt`: `out-a2`, ...) so an attempt nobody stopped cannot land its
    outputs where the next one's are read. `tests/test_worker_outputs.py` holds the two
    to the same string.
    """
    name = "out" if attempt == 1 else f"out-a{attempt}"
    return f"{stage_prefix(job_id, stage_id)}/transfer/{name}"


def checkpoint_key(job_id: uuid.UUID, stage_id: str, attempt: int = 1) -> str:
    """Where attempt `attempt` of a dispatched stage synced its checkpoint.

    It is the string `BaseRunner` puts in `StageContext.checkpoint_key` -- the pipeline's
    own `runners.checkpoint_key`, one key per attempt so a call nobody stopped cannot
    write over the next attempt's -- built from the same facts (the run id is the
    workdir's directory name, which is the job id). The supervisor needs it for a step
    that did *not* finish -- a preempted attempt has a checkpoint and no StepResult to
    read it out of.
    """
    return stage_checkpoint_key(str(job_id), stage_id, attempt)


@dataclass(frozen=True)
class UploadedArtifact:
    """One row's worth of `artifacts`, ready to insert."""

    kind: ArtifactKind
    storage_key: str
    bytes: int
    checksum: str
    content_type: str


#: Member types stated rather than guessed. `mimetypes` reads the host's own tables
#: (/etc/mime.types and friends) on top of Python's, so a guess can differ between the
#: worker image and a developer's machine. `collision.bin` is the packer's gzip-compressed
#: collision grid (tools/captures splat_tiles `COLLISION_FORMAT`): the web clients fetch
#: it and inflate it themselves, so it goes out as opaque bytes.
_MEMBER_TYPES = {".bin": "application/octet-stream"}


def _content_type(path: Path, fallback: str) -> str:
    stated = _MEMBER_TYPES.get(path.suffix.lower())
    if stated is not None:
        return stated
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


#: `Cache-Control` for a run output that is written once: the packaged tiles, the
#: thumbnail, the splat. A browser that has one never asks again.
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
#: For everything else: five minutes, then served stale for up to a week while it is
#: revalidated. Tileset JSON is here because a backfill rewrites it in place
#: (collision-backfill.yml adds `extras.collision` to a published `tileset.json`).
SHORT_CACHE = "public, max-age=300, stale-while-revalidate=604800"
_JSON_KEY = re.compile(r"\.json(\?|$)")


def cache_control_for(key: str) -> str:
    """The `Cache-Control` an object under `key` is written with.

    The same rule, word for word, as `immutable()` in `functions/r2/[[path]].js`, the
    Pages Function that serves the public bucket on the web app's origin: a key under
    `runs/` that is not JSON is immutable, and anything else gets the short lifetime. Set
    on the object itself, a browser reading the bucket's own public URL is told the same
    as one going through the proxy, and the published copy carries it (CopyObject keeps
    the source's metadata). If one of the two rules changes, the other must.

    "Immutable" is a promise that a key is never rewritten with different bytes. A run's
    keys carry its job id, which is what makes it true for a fresh run -- but a Refine
    re-runs the *same* job from `train`, and `package` writes the same tile names again
    with new geometry. That was already the proxy's promise before it was the object's.
    """
    immutable = key.startswith("runs/") and not _JSON_KEY.search(key)
    return IMMUTABLE_CACHE if immutable else SHORT_CACHE


#: The runner a stage dispatched to a provider records in its `step.json`
#: (`tools/pipeline/cloud.py`, `CloudRunner.name`).
DISPATCHED_RUNNER = "cloud"
#: One `CopyObject`'s ceiling on S3 and on R2. Bigger needs `UploadPartCopy`; an object
#: over it is uploaded instead (a trained splat is ~2 GB at the L4's 8M gaussians).
MAX_COPY_BYTES = 5 * 1024**3


def upload_artifact(
    storage: ObjectStorage, workdir_root: Path, job_id: uuid.UUID, ref: ArtifactRef
) -> UploadedArtifact | None:
    """Upload one artifact and describe the row it becomes.

    A directory artifact is uploaded member by member under one prefix, eight at a time
    (`app.worker.parallel`: normalize's frames and package's tiles are hundreds of small
    objects whose cost is round trips), and still becomes **one** row — the artifact is
    the tileset, not each of its tiles. `storage_key` is that prefix, `bytes` and
    `checksum` are the pipeline's own figures for the whole directory, so the row and
    `artifacts.json` agree without recomputing anything. Every object is written with the
    `Cache-Control` a browser should get for it (`cache_control_for`).

    A stage that ran on a provider is not uploaded at all when its bytes are already in
    the bucket: see `_copy_from_transfer`.
    """
    source = workdir_root / ref.path
    key = artifact_key(job_id, ref.stage_id, ref.name)
    try:
        if ref.kind == "dir":
            if not source.is_dir():
                return None
            if not _copy_from_transfer(storage, workdir_root, job_id, ref, source, key):

                def one(member: Path) -> object:
                    member_key = f"{key}/{member.relative_to(source).as_posix()}"
                    return storage.upload_file(
                        member_key,
                        member,
                        _content_type(member, "application/octet-stream"),
                        cache_control=cache_control_for(member_key),
                    )

                each(one, _members(source))
        else:
            if not source.is_file():
                return None
            if not _copy_from_transfer(storage, workdir_root, job_id, ref, source, key):
                # Streamed from disk: a trained splat can be larger than the worker's memory.
                storage.upload_file(
                    key, source, ref.content_type, cache_control=cache_control_for(key)
                )
    except StorageUnavailableError:
        return None
    return UploadedArtifact(
        kind=kind_for(ref.name),
        storage_key=key,
        bytes=ref.bytes,
        checksum=ref.checksum,
        content_type=ref.content_type,
    )


def _members(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _copy_from_transfer(
    storage: ObjectStorage,
    workdir_root: Path,
    job_id: uuid.UUID,
    ref: ArtifactRef,
    source: Path,
    key: str,
) -> bool:
    """Copy a dispatched stage's artifact into place inside the bucket, if it can be.

    A stage that ran on Modal handed its `out/` back through the bucket: the provider put
    it under `transfer_outputs_key`, and the recipe process downloaded it into the
    workdir. Uploading those bytes again from the worker -- a trained splat is up to 2 GB,
    from a shared-CPU machine -- moves them a third time to put them where they already
    are. So the artifact is copied server side from the provider's copy instead.

    **The download is not skipped**, and cannot be from here: the stages after `train`
    that run on this machine (`place`, `package`, `thumbnail`, ...) read `canonical.ply`
    from the workdir, the runner checksums `out/` for `artifacts.json`, and "retry from
    this stage" reads the workdir. Only the upload goes.

    Done only when it is safe to say the bucket's copy *is* the workdir's: the stage's
    `step.json` says it was dispatched, and the provider's objects match the workdir's
    files name for name and byte count for byte count. Anything else -- a stage that ran
    here, a transfer through a shared directory rather than the bucket, an object too big
    for one `CopyObject`, any error from the copy -- returns False and the caller uploads,
    as it always did. Only "there is no bucket" propagates.
    """
    attempt = _dispatched_attempt(workdir_root, ref.stage_id)
    if attempt is None:
        return False
    remote = f"{transfer_outputs_key(job_id, ref.stage_id, attempt)}/{ref.name}"
    try:
        if ref.kind == "dir":
            local = {
                member.relative_to(source).as_posix(): member.stat().st_size
                for member in _members(source)
            }
            if _sizes_under(storage, f"{remote}/") != local or any(
                size > MAX_COPY_BYTES for size in local.values()
            ):
                return False

            def one(relative: str) -> object:
                member_key = f"{key}/{relative}"
                return storage.copy_object(
                    storage.bucket,
                    f"{remote}/{relative}",
                    member_key,
                    content_type=_content_type(Path(relative), "application/octet-stream"),
                    cache_control=cache_control_for(member_key),
                )

            each(one, sorted(local))
        else:
            size = source.stat().st_size
            there = storage.head_object(remote)
            if there is None or there.size != size or size > MAX_COPY_BYTES:
                return False
            storage.copy_object(
                storage.bucket,
                remote,
                key,
                content_type=ref.content_type,
                cache_control=cache_control_for(key),
            )
    except StorageUnavailableError:
        raise
    except Exception:
        log.warning(
            "worker: could not copy %s within the bucket from %s; uploading it instead",
            key,
            remote,
            exc_info=True,
        )
        return False
    log.info("worker: %s copied within the bucket from the provider's %s", key, remote)
    return True


def _dispatched_attempt(workdir_root: Path, stage_id: str) -> int | None:
    """The attempt that ran this stage on a provider, or None if it ran here.

    Its `step.json` -- written before the stage is reported finished -- names the runner
    that ran it and the attempt it was. The attempt matters: every attempt after the first
    hands its `out/` back under a key of its own (`transfer_outputs_key`), and a file of
    the same name and size left there by an earlier attempt must never be copied as this
    one's.
    """
    try:
        step = json.loads(Workdir(workdir_root).step_path(stage_id).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(step, dict) or step.get("runner") != DISPATCHED_RUNNER:
        return None
    attempt = step.get("attempt", 1)
    return attempt if isinstance(attempt, int) and attempt >= 1 else None


def _sizes_under(storage: ObjectStorage, prefix: str) -> dict[str, int]:
    """Every object under `prefix`, by its key relative to it, with its size. Paged to
    the end: a tileset or a frames directory can be more than one page."""
    sizes: dict[str, int] = {}
    token: str | None = None
    while True:
        page = storage.list_objects(prefix, continuation_token=token)
        sizes.update({item.key[len(prefix) :]: item.size for item in page.objects})
        token = page.next_continuation_token
        if token is None:
            return sizes
