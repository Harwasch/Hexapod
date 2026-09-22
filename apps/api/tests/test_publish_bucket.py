"""The bucket split: what reaches the public bucket, and what must never.

The behaviour under test is one sentence long. A run's outputs all land in the private
bucket; exactly the tileset and the thumbnail are copied out of it; and a site's URLs
point at the copies. Everything else a run produced -- the raw upload, the frames, the
logs, the checkpoints -- stays where a browser cannot reach it.

It is worth a file of its own because the failure is silent. A deployment that publishes
its whole bucket works perfectly: the globe renders, every test passes, and the only
symptom is that somebody who has a key can also read every scan anyone uploaded. There is
nothing to notice, so there has to be something to fail.

`moto` is used rather than a fake so the copy is a real `CopyObject` between two real
buckets. It does not enforce authorisation (tests/test_captures.py says so at more
length), so this proves what ends up where and never who may read it.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import boto3
import pytest
from moto import mock_aws

from app.config import Settings
from app.storage import ObjectStorage, S3Storage
from app.storage.factory import build_public_storage, build_publish_storage, build_storage
from app.storage.null import StorageUnavailableError
from app.worker.publish import Publisher, PublishError

PRIVATE = "twin-assets"
PUBLIC = "twin-public"
PUBLIC_URL = "https://tiles.example.com"

JOB = uuid.UUID("00000000-0000-0000-0000-0000000000ab")
TILES = f"runs/{JOB}/package/splat"


def settings_with(**overrides: object) -> Settings:
    """Every storage field explicit, for the reason tests/test_captures.py gives: the
    repository's own .env points at MinIO, and a Settings built without them is
    configured on a developer's machine and unconfigured in CI."""
    configured: dict[str, object] = {
        "object_storage_endpoint_url": "https://s3.example.com",
        "object_storage_bucket": PRIVATE,
        "object_storage_access_key": "key",
        "object_storage_secret_key": "secret",
        "object_storage_public_bucket": None,
        "object_storage_public_url": None,
    }
    return Settings(**{**configured, **overrides})  # type: ignore[arg-type]


def storage_over(bucket: str, *, public_base_url: str | None = None) -> S3Storage:
    """`endpoint_url=None` because moto intercepts the AWS hostnames; a custom endpoint
    host makes botocore open a real connection instead."""
    return S3Storage(
        bucket=bucket,
        endpoint_url=None,
        access_key="key",
        secret_key="secret",
        region="us-east-1",
        public_base_url=public_base_url,
    )


@pytest.fixture
def buckets() -> Iterator[Publisher]:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=PRIVATE)
        client.create_bucket(Bucket=PUBLIC)
        yield Publisher(
            private=storage_over(PRIVATE),
            public=storage_over(PUBLIC, public_base_url=PUBLIC_URL),
        )


def seed_a_run(publisher: Publisher) -> None:
    """One run's footprint in the private bucket: publishable and not."""
    private = publisher.private
    private.put_object(f"{TILES}/tileset.json", b"{}", "application/json")
    private.put_object(f"{TILES}/0/0.glb", b"glb", "model/gltf-binary")
    private.put_object(f"runs/{JOB}/thumb/thumbnail.jpg", b"jpg", "image/jpeg")
    # Everything a browser must never be handed.
    private.put_object("captures/abc/source/1/scan.ply", b"raw", "application/octet-stream")
    private.put_object(f"runs/{JOB}/package/log.txt", b"log", "text/plain")
    private.put_object(f"runs/{JOB}/train/checkpoint/step.pt", b"ckpt", "application/octet-stream")
    private.put_object(f"runs/{JOB}/normalize/frames/0001.jpg", b"frame", "image/jpeg")


def keys_in(storage: ObjectStorage, prefix: str = "") -> set[str]:
    return {item.key for item in storage.list_objects(prefix).objects}


def test_a_published_tileset_is_the_whole_directory(buckets: Publisher) -> None:
    """The root file names the tiles, so publishing it alone renders nothing."""
    seed_a_run(buckets)
    url = buckets.publish_tree(TILES, f"{TILES}/tileset.json")
    assert url == f"{PUBLIC_URL}/{TILES}/tileset.json"
    assert keys_in(buckets.public) == {f"{TILES}/tileset.json", f"{TILES}/0/0.glb"}


def test_nothing_else_the_run_produced_is_published(buckets: Publisher) -> None:
    """The whole point, stated as the thing that must not happen.

    A raw upload, a log, a checkpoint and a frame are all in the private bucket under
    keys a viewer's URL sits right next to. After publishing a site, none of them is
    reachable from the bucket a browser reads.
    """
    seed_a_run(buckets)
    buckets.publish_tree(TILES, f"{TILES}/tileset.json")
    buckets.publish_object(f"runs/{JOB}/thumb/thumbnail.jpg")

    published = keys_in(buckets.public)
    assert not any(key.startswith("captures/") for key in published)
    assert not any(key.endswith(("log.txt", "step.pt", "0001.jpg")) for key in published)
    # And the private bucket still has everything, including what was copied out of it.
    assert len(keys_in(buckets.private)) == 7


def test_the_copy_leaves_the_private_bucket_intact(buckets: Publisher) -> None:
    """A copy, not a move: the run's own outputs are what reconciliation reads."""
    seed_a_run(buckets)
    buckets.publish_object(f"{TILES}/tileset.json")
    assert buckets.private.get_object(f"{TILES}/tileset.json") == b"{}"
    assert buckets.public.get_object(f"{TILES}/tileset.json") == b"{}"


def test_publishing_an_empty_prefix_refuses_rather_than_returning_a_dead_url(
    buckets: Publisher,
) -> None:
    with pytest.raises(PublishError, match="nothing to publish"):
        buckets.publish_tree(TILES, f"{TILES}/tileset.json")


def test_a_tree_without_its_entry_file_refuses(buckets: Publisher) -> None:
    """Tiles but no `tileset.json` is a site that looks fine until somebody opens it."""
    buckets.private.put_object(f"{TILES}/0/0.glb", b"glb", "model/gltf-binary")
    with pytest.raises(PublishError, match="but not"):
        buckets.publish_tree(TILES, f"{TILES}/tileset.json")


# --- one bucket: the behaviour this replaced, unchanged -----------------------------


def test_with_one_bucket_publishing_copies_nothing_and_still_returns_a_url() -> None:
    """A fresh checkout and the MinIO dev loop have one bucket and must keep working."""
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=PRIVATE)
        only = storage_over(PRIVATE)
        publisher = Publisher(private=only, public=only)
        assert publisher.splits_buckets is False
        only.put_object(f"{TILES}/tileset.json", b"{}", "application/json")
        assert publisher.publish_tree(TILES, f"{TILES}/tileset.json") == only.public_url(
            f"{TILES}/tileset.json"
        )


def test_with_no_storage_at_all_a_publish_is_a_missing_url_rather_than_a_crash() -> None:
    """`register` asks for a URL on every run, including runs with no bucket configured."""
    settings = settings_with(
        object_storage_endpoint_url=None,
        object_storage_bucket=None,
        object_storage_access_key=None,
        object_storage_secret_key=None,
    )
    private = build_storage(settings)
    publisher = Publisher(private=private, public=build_publish_storage(settings))
    assert publisher.splits_buckets is False
    assert publisher.publish_object("anything") is None
    with pytest.raises(StorageUnavailableError):
        private.public_url("anything")


# --- which storage the seed paths write to ------------------------------------------


def test_the_seed_paths_write_to_the_public_bucket_when_there_is_one() -> None:
    """`sites/` and `catalog.json` are public by their whole purpose, so they go straight
    there rather than being written privately and copied."""
    split = settings_with(object_storage_public_bucket=PUBLIC, object_storage_public_url=PUBLIC_URL)
    assert build_public_storage(split).bucket == PUBLIC
    assert build_storage(split).bucket == PRIVATE


def test_the_seed_paths_fall_back_to_the_only_bucket() -> None:
    assert build_public_storage(settings_with()).bucket == PRIVATE


def test_a_public_bucket_named_the_same_as_the_private_one_is_not_a_split() -> None:
    same = settings_with(object_storage_public_bucket=PRIVATE)
    assert same.publish_bucket_configured is False
    assert build_publish_storage(same).available is False
