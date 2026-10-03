"""A finished stage's artifacts into the bucket: fast, with the right cache headers, and
not uploaded at all when the bytes are already there.

Against moto through the real `S3Storage`, so content types, Cache-Control and copies are
what boto3 actually sends, plus a slowed-down wrapper where the question is timing.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import boto3
import pytest

from app.config import REPO_ROOT
from app.services.published import PUBLISHED_KEY, published_key
from app.storage import S3Storage
from app.worker import outputs
from app.worker.outputs import (
    IMMUTABLE_CACHE,
    SHORT_CACHE,
    cache_control_for,
    transfer_outputs_key,
    upload_artifact,
)
from app.worker.pipeline_bridge import ArtifactRef
from tests.conftest import WORKER_BUCKET

JOB = uuid.UUID("00000000-0000-0000-0000-0000000000cd")


def raw_head(key: str) -> dict[str, Any]:
    return dict(
        boto3.client("s3", region_name="us-east-1").head_object(Bucket=WORKER_BUCKET, Key=key)
    )


def tiles_ref(stage: str = "package") -> ArtifactRef:
    return ArtifactRef(
        name="splat",
        stage_id=stage,
        path=f"stages/{stage}/out/splat",
        kind="dir",
        content_type="application/octet-stream",
        bytes=0,
        checksum="sha256:x",
    )


def splat_ref() -> ArtifactRef:
    return ArtifactRef(
        name="canonical.ply",
        stage_id="train",
        path="stages/train/out/canonical.ply",
        kind="file",
        content_type="application/octet-stream",
        bytes=7,
        checksum="sha256:y",
    )


def a_package(root: Path, tiles: int = 3) -> Path:
    splat = root / "stages" / "package" / "out" / "splat"
    splat.mkdir(parents=True)
    (splat / "tileset.json").write_text("{}")
    for index in range(tiles):
        (splat / f"splat_{index}.glb").write_bytes(b"glb" * (index + 1))
    (splat / "collision.bin").write_bytes(b"\x1f\x8b")
    return splat


#: `step_json`'s default: the key a call submitted by the step's own attempt wrote to.
OWN = "own"


def step_json(
    root: Path,
    stage: str,
    runner: str,
    attempt: int | None = None,
    outputs_key: str | None = OWN,
) -> None:
    """A stage's `step.json`, as `CloudRunner` writes it: with the key its outputs came
    home from (`metrics.outputsKey`) -- by default its own attempt's, None for none."""
    path = root / "stages" / stage / "step.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    step: dict[str, object] = {"stageId": stage, "runner": runner, "metrics": {}}
    if attempt is not None:
        step["attempt"] = attempt
    if outputs_key == OWN:
        outputs_key = transfer_outputs_key(JOB, stage, attempt or 1)
    if outputs_key is not None and runner == "cloud":
        step["metrics"] = {"outputsKey": outputs_key}
    path.write_text(json.dumps(step))


# --- Cache-Control ------------------------------------------------------------------------


def test_the_cache_rule_is_the_tile_proxys_rule() -> None:
    """`functions/r2/[[path]].js` decides the same thing for the proxy. The two are kept
    in step by hand, so this reads the function and checks the parts that decide."""
    source = (REPO_ROOT / "functions" / "r2" / "[[path]].js").read_text()
    assert "PUBLISHED_KEY.test(name) && !/\\.json$/i.test(name)" in source
    spelled = re.search(r"const PUBLISHED_KEY = /(.+)/;", source)
    assert spelled and spelled[1].replace("\\/", "/") == PUBLISHED_KEY.pattern
    year = re.search(r"const YEAR_S = (\d+);", source)
    short = re.search(r"const SHORT_S = (\d+);", source)
    assert year and short
    assert f"public, max-age={year[1]}, immutable" == IMMUTABLE_CACHE
    assert f"public, max-age={short[1]}, stale-while-revalidate=604800" == SHORT_CACHE
    assert "stale-while-revalidate=604800" in source

    published = published_key(f"runs/{JOB}/package/splat", "0123456789abcdef")
    for key in (
        f"{published}/splat_3-5.glb",
        f"{published}/collision.bin",
        f"{published}/x.spz",
        f"{published}/a.webp",
    ):
        assert cache_control_for(key) == IMMUTABLE_CACHE, key
    run = f"runs/{JOB}/package/splat"
    for key in (
        f"{published}/tileset.json",
        f"{published}/tileset.JSON",
        # A run's own keys: a Refine or a retry writes them again with new bytes.
        f"{run}/splat_3-5.glb",
        f"runs/{JOB}/train/canonical.ply",
        f"runs/{JOB}/register/registration.json",
        "sites/a/b.glb",
    ):
        assert cache_control_for(key) == SHORT_CACHE, key


def test_a_run_s_own_keys_are_not_promised_to_anyone_for_a_year(
    storage: S3Storage, tmp_path: Path
) -> None:
    """A Refine re-runs the same job from `train` and `package` rewrites the same tile
    names with new geometry. Uploaded `immutable`, a browser reading the bucket's own URL
    (one bucket, no proxy) would keep the preview's tiles under the new tileset.json."""
    a_package(tmp_path)
    ref = tiles_ref()
    assert upload_artifact(storage, tmp_path, JOB, ref) is not None
    (tmp_path / "stages" / "package" / "out" / "splat" / "splat_0.glb").write_bytes(b"new")
    assert upload_artifact(storage, tmp_path, JOB, ref) is not None

    tile = raw_head(f"runs/{JOB}/package/splat/splat_0.glb")
    assert tile["CacheControl"] == SHORT_CACHE
    assert storage.get_object(f"runs/{JOB}/package/splat/splat_0.glb") == b"new"


def test_artifacts_are_uploaded_with_their_cache_control(
    storage: S3Storage, tmp_path: Path
) -> None:
    a_package(tmp_path)
    ply = tmp_path / "stages" / "train" / "out" / "canonical.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b"ply\x00\x01\x02\x03")

    assert upload_artifact(storage, tmp_path, JOB, tiles_ref()) is not None
    assert upload_artifact(storage, tmp_path, JOB, splat_ref()) is not None

    run = f"runs/{JOB}"
    assert raw_head(f"{run}/package/splat/splat_0.glb")["CacheControl"] == SHORT_CACHE
    assert (
        raw_head(f"{run}/package/splat/collision.bin")["ContentType"] == "application/octet-stream"
    )
    root = raw_head(f"{run}/package/splat/tileset.json")
    assert (root["CacheControl"], root["ContentType"]) == (SHORT_CACHE, "application/json")
    assert raw_head(f"{run}/train/canonical.ply")["CacheControl"] == SHORT_CACHE


def test_a_large_file_is_still_streamed_and_still_gets_its_headers(
    storage: S3Storage, tmp_path: Path
) -> None:
    """Over the multipart threshold, the managed transfer -- and its ETag read back."""
    big = tmp_path / "big.ply"
    big.write_bytes(b"x" * (9 * 1024 * 1024))
    stored = storage.upload_file(
        "runs/j/train/big.ply", big, "application/octet-stream", cache_control=IMMUTABLE_CACHE
    )
    head = raw_head("runs/j/train/big.ply")
    assert head["CacheControl"] == IMMUTABLE_CACHE
    assert stored.etag == head["ETag"].strip('"')
    # And a small one is one PUT whose own response gave the ETag.
    small = tmp_path / "small.json"
    small.write_text("{}")
    stored = storage.upload_file("runs/j/x/small.json", small, "application/json")
    assert stored.etag == raw_head("runs/j/x/small.json")["ETag"].strip('"')
    assert "CacheControl" not in raw_head("runs/j/x/small.json")


# --- in parallel ------------------------------------------------------------------------


class Slow:
    """`upload_file`, slowed and counted, delegating to the real storage."""

    def __init__(self, inner: S3Storage, *, delay_s: float = 0.05, fail: str = "") -> None:
        self.inner = inner
        self.delay_s = delay_s
        self.fail = fail
        self.active = 0
        self.peak = 0
        self.uploads: list[str] = []
        self.lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def upload_file(self, key: str, source: Path, content_type: str, **kwargs: Any) -> Any:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.uploads.append(key)
        try:
            time.sleep(self.delay_s)
            if key.endswith(self.fail) and self.fail:
                raise RuntimeError(f"R2 said no to {key}")
            return self.inner.upload_file(key, source, content_type, **kwargs)
        finally:
            with self.lock:
                self.active -= 1


def test_a_directory_goes_up_eight_objects_at_a_time(storage: S3Storage, tmp_path: Path) -> None:
    a_package(tmp_path, tiles=30)
    slow = Slow(storage)

    started = time.monotonic()
    assert upload_artifact(slow, tmp_path, JOB, tiles_ref()) is not None
    took = time.monotonic() - started

    assert slow.peak == 8
    assert len(slow.uploads) == 32
    assert took < 1.0  # 32 x 0.05 s is 1.6 s one at a time


def test_a_failed_upload_fails_the_artifact(storage: S3Storage, tmp_path: Path) -> None:
    """As before: an upload error is the supervisor's to turn into the job's error. What
    changed is that nothing not yet started is started after it."""
    a_package(tmp_path, tiles=30)
    slow = Slow(storage, fail="splat_0.glb")

    with pytest.raises(RuntimeError, match="R2 said no"):
        upload_artifact(slow, tmp_path, JOB, tiles_ref())
    assert len(slow.uploads) < 32


def test_a_directory_uploaded_again_keeps_nothing_the_earlier_attempt_left(
    storage: S3Storage, tmp_path: Path
) -> None:
    """A Refine re-runs the same job into the same prefix. A grid or a tile the first
    package wrote and the second did not would otherwise sit beside the new ones -- and be
    published with them, a stale `collision.bin` passing for the new run's own."""
    first = a_package(tmp_path / "first", tiles=4)
    assert upload_artifact(storage, tmp_path / "first", JOB, tiles_ref()) is not None
    second = a_package(tmp_path / "second", tiles=2)
    (second / "collision.bin").unlink()
    (second / "splat_0.glb").write_bytes(b"refined")

    assert upload_artifact(storage, tmp_path / "second", JOB, tiles_ref()) is not None

    prefix = f"runs/{JOB}/package/splat/"
    listed = {item.key[len(prefix) :] for item in storage.list_objects(prefix).objects}
    assert listed == {"tileset.json", "splat_0.glb", "splat_1.glb"}
    assert storage.get_object(f"{prefix}splat_0.glb") == b"refined"
    assert {p.name for p in first.iterdir()} > listed


# --- dispatched stages: copied within the bucket, not uploaded again --------------------


class NoUploads:
    def __init__(self, inner: S3Storage) -> None:
        self.inner = inner
        self.uploads: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def upload_file(self, key: str, source: Path, content_type: str, **kwargs: Any) -> Any:
        self.uploads.append(key)
        return self.inner.upload_file(key, source, content_type, **kwargs)


def provider_left(
    storage: S3Storage, stage: str, files: dict[str, bytes], attempt: int = 1
) -> None:
    """What a provider puts under `transfer/out` (`out-a<N>` after the first attempt):
    plain octet-stream, no cache header."""
    for name, data in files.items():
        storage.put_object(
            f"{transfer_outputs_key(JOB, stage, attempt)}/{name}",
            data,
            "application/octet-stream",
        )


def test_the_transfer_key_is_the_pipelines() -> None:
    from cloud import StageKeys

    keys = StageKeys(
        root=f"runs/{JOB}", stage=f"runs/{JOB}/train", checkpoint=f"runs/{JOB}/train/checkpoint"
    )
    assert transfer_outputs_key(JOB, "train") == keys.outputs
    assert outputs.DISPATCHED_RUNNER == "cloud"


def test_a_dispatched_stages_file_is_copied_not_uploaded(
    storage: S3Storage, tmp_path: Path
) -> None:
    ply = tmp_path / "stages" / "train" / "out" / "canonical.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b"ply\x00abc")
    step_json(tmp_path, "train", "cloud")
    provider_left(storage, "train", {"canonical.ply": b"ply\x00abc"})
    counting = NoUploads(storage)

    uploaded = upload_artifact(counting, tmp_path, JOB, splat_ref())

    assert uploaded is not None and uploaded.storage_key == f"runs/{JOB}/train/canonical.ply"
    assert counting.uploads == []
    assert storage.get_object(uploaded.storage_key) == b"ply\x00abc"
    head = raw_head(uploaded.storage_key)
    assert (head["ContentType"], head["CacheControl"]) == (
        "application/octet-stream",
        SHORT_CACHE,
    )


def test_the_transfer_key_is_the_pipelines_for_a_later_attempt_too() -> None:
    from cloud import StageKeys

    keys = StageKeys(
        root=f"runs/{JOB}",
        stage=f"runs/{JOB}/train",
        checkpoint=f"runs/{JOB}/train/checkpoint-a2",
        attempt=2,
    )
    assert transfer_outputs_key(JOB, "train", 2) == keys.outputs


def test_a_later_attempt_is_copied_from_its_own_transfer_key(
    storage: S3Storage, tmp_path: Path
) -> None:
    ply = tmp_path / "stages" / "train" / "out" / "canonical.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b"ply\x00new")
    step_json(tmp_path, "train", "cloud", attempt=2)
    provider_left(storage, "train", {"canonical.ply": b"ply\x00new"}, attempt=2)
    counting = NoUploads(storage)

    uploaded = upload_artifact(counting, tmp_path, JOB, splat_ref())

    assert uploaded is not None
    assert counting.uploads == []
    assert storage.get_object(uploaded.storage_key) == b"ply\x00new"


def test_an_earlier_attempts_leftover_is_never_copied_as_a_later_ones(
    storage: S3Storage, tmp_path: Path
) -> None:
    # Attempt 1 left a file of the same name and size; attempt 2 handed nothing back
    # through the bucket under its own key. The leftover must not be copied as attempt
    # 2's: the workdir's bytes are uploaded instead.
    ply = tmp_path / "stages" / "train" / "out" / "canonical.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b"ply\x00new")
    step_json(tmp_path, "train", "cloud", attempt=2)
    provider_left(storage, "train", {"canonical.ply": b"ply\x00old"}, attempt=1)
    counting = NoUploads(storage)

    uploaded = upload_artifact(counting, tmp_path, JOB, splat_ref())

    assert uploaded is not None
    assert counting.uploads == [uploaded.storage_key]
    assert storage.get_object(uploaded.storage_key) == b"ply\x00new"


def test_an_adopted_calls_outputs_are_copied_from_the_key_that_call_wrote(
    storage: S3Storage, tmp_path: Path
) -> None:
    """Attempt 2 re-attached to attempt 1's call, so its outputs came home from `out`, and
    `out-a2` holds what an earlier run's attempt 2 left there before a Retry reset the
    attempts -- a file of the same name and size. The step's own record of the key is
    what is copied from; the attempt number would have copied the other run's bytes."""
    ply = tmp_path / "stages" / "train" / "out" / "canonical.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b"ply\x00new")
    step_json(tmp_path, "train", "cloud", attempt=2, outputs_key=transfer_outputs_key(JOB, "train"))
    provider_left(storage, "train", {"canonical.ply": b"ply\x00new"}, attempt=1)
    provider_left(storage, "train", {"canonical.ply": b"ply\x00old"}, attempt=2)
    counting = NoUploads(storage)

    uploaded = upload_artifact(counting, tmp_path, JOB, splat_ref())

    assert uploaded is not None and counting.uploads == []
    assert storage.get_object(uploaded.storage_key) == b"ply\x00new"


@pytest.mark.parametrize(
    "recorded",
    [None, f"runs/{uuid.UUID(int=7)}/train/transfer/out", f"runs/{JOB}/train/checkpoint"],
)
def test_without_a_recorded_key_of_its_own_a_dispatched_stage_is_uploaded(
    storage: S3Storage, tmp_path: Path, recorded: str | None
) -> None:
    """No guessing: a step with no `outputsKey` (written before it was recorded), or one
    naming anything but this stage's own transfer outputs, is uploaded from the workdir
    even when the attempt's key holds a file that matches."""
    ply = tmp_path / "stages" / "train" / "out" / "canonical.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b"ply\x00abc")
    step_json(tmp_path, "train", "cloud", outputs_key=recorded)
    provider_left(storage, "train", {"canonical.ply": b"ply\x00abc"})
    counting = NoUploads(storage)

    assert upload_artifact(counting, tmp_path, JOB, splat_ref()) is not None
    assert counting.uploads == [f"runs/{JOB}/train/canonical.ply"]


def test_a_dispatched_stages_directory_is_copied_member_by_member(
    storage: S3Storage, tmp_path: Path
) -> None:
    splat = a_package(tmp_path)
    step_json(tmp_path, "package", "cloud")
    provider_left(
        storage,
        "package",
        {f"splat/{p.relative_to(splat).as_posix()}": p.read_bytes() for p in splat.rglob("*")},
    )
    counting = NoUploads(storage)

    assert upload_artifact(counting, tmp_path, JOB, tiles_ref()) is not None

    assert counting.uploads == []
    root = raw_head(f"runs/{JOB}/package/splat/tileset.json")
    assert (root["ContentType"], root["CacheControl"]) == ("application/json", SHORT_CACHE)
    assert raw_head(f"runs/{JOB}/package/splat/splat_2.glb")["ContentType"] == "model/gltf-binary"


@pytest.mark.parametrize(
    ("runner", "left", "why"),
    [
        ("local", {"canonical.ply": b"ply\x00abc"}, "the stage ran here"),
        ("cloud", {"canonical.ply": b"ply\x00abcd"}, "the sizes differ"),
        ("cloud", {}, "the provider left nothing in the bucket (a shared transfer dir)"),
    ],
)
def test_anything_short_of_a_match_is_uploaded_as_before(
    storage: S3Storage, tmp_path: Path, runner: str, left: dict[str, bytes], why: str
) -> None:
    ply = tmp_path / "stages" / "train" / "out" / "canonical.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b"ply\x00abc")
    step_json(tmp_path, "train", runner)
    provider_left(storage, "train", left)
    counting = NoUploads(storage)

    assert upload_artifact(counting, tmp_path, JOB, splat_ref()) is not None

    assert counting.uploads == [f"runs/{JOB}/train/canonical.ply"], why
    assert storage.get_object(f"runs/{JOB}/train/canonical.ply") == b"ply\x00abc"


def test_a_copy_that_fails_falls_back_to_the_upload(
    storage: S3Storage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ply = tmp_path / "stages" / "train" / "out" / "canonical.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b"ply\x00abc")
    step_json(tmp_path, "train", "cloud")
    provider_left(storage, "train", {"canonical.ply": b"ply\x00abc"})
    counting = NoUploads(storage)

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("NotImplemented: x-amz-metadata-directive")

    monkeypatch.setattr(storage, "copy_object", refuse)

    assert upload_artifact(counting, tmp_path, JOB, splat_ref()) is not None
    assert counting.uploads == [f"runs/{JOB}/train/canonical.ply"]
