"""The bucket split: what reaches the public bucket, and what must never.

The behaviour under test is one sentence long. A run's outputs all land in the private
bucket; exactly the tileset and the thumbnail are copied out of it; and a site's URLs
point at the copies. Everything else a run produced -- the raw upload, the frames, the
logs, the checkpoints -- stays where a browser cannot reach it.

And one more, since a Refine rewrites a run's own keys: every publish copies into a
generation of keys of its own (`runs/<job>/p<generation>/...`), so a key a browser may
hold as `immutable` is never written again, and a republish that fails leaves the live
site exactly as it was.

It is worth a file of its own because the failure is silent. A deployment that publishes
its whole bucket works perfectly: the globe renders, every test passes, and the only
symptom is that somebody who has a key can also read every scan anyone uploaded. There is
nothing to notice, so there has to be something to fail.

`moto` is used rather than a fake so the copy is a real `CopyObject` between two real
buckets. It does not enforce authorisation (tests/test_captures.py says so at more
length), so this proves what ends up where and never who may read it.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import pytest
from moto import mock_aws
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Capture, Site
from app.models.enums import CaptureKind, GeorefMethod, ScaleSource
from app.services.published import is_published, published_key, unpublished
from app.storage import ObjectStorage, S3Storage
from app.storage.base import ObjectPage, ObjectSummary
from app.storage.factory import build_public_storage, build_publish_storage, build_storage
from app.storage.null import StorageUnavailableError
from app.worker.outputs import IMMUTABLE_CACHE, upload_artifact
from app.worker.pipeline_bridge import ArtifactRef
from app.worker.publish import PUBLISH_WORKERS, Publisher, PublishError
from app.worker.registration import Registration, publish_outputs, register

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


def generation_of(url: str | None) -> str:
    """The generation a published URL is in: `.../runs/<job>/p<generation>/...`."""
    assert url is not None
    segment = url.removeprefix(f"{PUBLIC_URL}/").split("/")[2]
    assert segment.startswith("p"), url
    return segment[1:]


def test_a_published_tileset_is_the_whole_directory(buckets: Publisher) -> None:
    """The root file names the tiles, so publishing it alone renders nothing."""
    seed_a_run(buckets)
    url = buckets.publish_tree(TILES, f"{TILES}/tileset.json")
    generation = generation_of(url)
    assert url == f"{PUBLIC_URL}/{published_key(TILES, generation)}/tileset.json"
    assert keys_in(buckets.public) == {
        published_key(f"{TILES}/tileset.json", generation),
        published_key(f"{TILES}/0/0.glb", generation),
    }


def test_the_collision_grid_beside_the_tiles_is_published_as_it_was_stored(
    buckets: Publisher,
) -> None:
    """A splat tileset's `collision.bin` (the web clients' collision grid) sits beside
    `tileset.json`, so publishing the directory publishes it, content type and all."""
    seed_a_run(buckets)
    buckets.private.put_object(f"{TILES}/collision.bin", b"\x1f\x8b", "application/octet-stream")
    generation = generation_of(buckets.publish_tree(TILES, f"{TILES}/tileset.json"))
    key = published_key(f"{TILES}/collision.bin", generation)
    head = buckets.public.head_object(key)
    assert head is not None and head.content_type == "application/octet-stream"
    assert buckets.public.get_object(key) == b"\x1f\x8b"


def test_a_directory_is_not_every_key_that_begins_like_it(buckets: Publisher) -> None:
    """`splat` is a directory: `splat_old/` beside it is not part of the tileset."""
    seed_a_run(buckets)
    buckets.private.put_object(f"{TILES}_old/x.glb", b"old", "model/gltf-binary")
    buckets.publish_tree(TILES, f"{TILES}/tileset.json")
    assert not any("splat_old" in key for key in keys_in(buckets.public))


# --- generations: a republish never writes a key a viewer may hold --------------------


def test_the_generation_layout_and_its_way_back() -> None:
    own = f"runs/{JOB}/package/splat/0.glb"
    key = published_key(own, "0123456789abcdef")
    assert key == f"runs/{JOB}/p0123456789abcdef/package/splat/0.glb"
    assert is_published(key) and not is_published(own)
    assert not is_published(f"runs/{JOB}/p0123/package/splat/0.glb")
    assert unpublished(f"{PUBLIC_URL}/{key}") == f"{PUBLIC_URL}/{own}"
    assert unpublished(own) == own
    for bad in ("sites/x/0.glb", f"runs/{JOB}", "runs//a.glb"):
        with pytest.raises(ValueError):
            published_key(bad, "0123456789abcdef")
    with pytest.raises(ValueError):
        published_key(own, "not-a-generation")


def test_a_republish_of_new_bytes_lands_in_a_new_generation_beside_the_live_one(
    buckets: Publisher,
) -> None:
    """A Refine re-runs the same job and rewrites its tiles under the same names. The
    publish after it must not write over the live tiles a browser holds as immutable."""
    seed_a_run(buckets)
    live = buckets.publish_tree(TILES, f"{TILES}/tileset.json")
    buckets.private.put_object(f"{TILES}/0/0.glb", b"refined", "model/gltf-binary")
    buckets.private.put_object(f"{TILES}/tileset.json", b'{"refined":1}', "application/json")

    refined = buckets.publish_tree(TILES, f"{TILES}/tileset.json")

    assert refined != live
    before, after = generation_of(live), generation_of(refined)
    assert buckets.public.get_object(published_key(f"{TILES}/0/0.glb", before)) == b"glb"
    assert buckets.public.get_object(published_key(f"{TILES}/tileset.json", before)) == b"{}"
    assert buckets.public.get_object(published_key(f"{TILES}/0/0.glb", after)) == b"refined"
    # Nothing was published outside a generation: every key there is written once.
    assert all(is_published(key) for key in keys_in(buckets.public))


def test_the_same_bytes_published_again_land_on_the_same_keys(buckets: Publisher) -> None:
    """A retried `register`, or a worker that lost its lease after publishing: the same
    generation, so the URL a browser has cached against does not change."""
    seed_a_run(buckets)
    first = buckets.publish_tree(TILES, f"{TILES}/tileset.json")
    assert buckets.publish_tree(TILES, f"{TILES}/tileset.json") == first
    assert len(keys_in(buckets.public)) == 2


def test_without_etags_to_vouch_for_the_bytes_every_publish_is_a_new_generation(
    buckets: Publisher,
) -> None:
    seed_a_run(buckets)
    blind = Publisher(private=NoEtags(buckets.private), public=buckets.public)
    first = blind.generation(prefixes=[TILES])
    assert first is not None and first != blind.generation(prefixes=[TILES])


def test_a_published_copy_is_immutable_and_labelled_as_the_original(
    buckets: Publisher,
) -> None:
    """The copy states its metadata: the content type the original was stored with and
    the lifetime of the key it lands on -- a year, because nothing writes it again."""
    seed_a_run(buckets)
    generation = generation_of(buckets.publish_tree(TILES, f"{TILES}/tileset.json"))
    url = buckets.publish_object(f"runs/{JOB}/thumb/thumbnail.jpg", generation=generation)
    assert url == f"{PUBLIC_URL}/runs/{JOB}/p{generation}/thumb/thumbnail.jpg"
    raw = boto3.client("s3", region_name="us-east-1")
    thumbnail = raw.head_object(
        Bucket=PUBLIC, Key=published_key(f"runs/{JOB}/thumb/thumbnail.jpg", generation)
    )
    assert (thumbnail["ContentType"], thumbnail["CacheControl"]) == ("image/jpeg", IMMUTABLE_CACHE)


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
    generation = generation_of(buckets.publish_object(f"{TILES}/tileset.json"))
    assert buckets.private.get_object(f"{TILES}/tileset.json") == b"{}"
    assert buckets.public.get_object(published_key(f"{TILES}/tileset.json", generation)) == b"{}"


def test_publishing_an_empty_prefix_refuses_rather_than_returning_a_dead_url(
    buckets: Publisher,
) -> None:
    with pytest.raises(PublishError, match="nothing to publish"):
        buckets.publish_tree(TILES, f"{TILES}/tileset.json")


def test_a_tree_without_its_entry_file_refuses(buckets: Publisher) -> None:
    """Tiles but no `tileset.json` is a site that looks fine until somebody opens it --
    refused before a single tile is copied."""
    buckets.private.put_object(f"{TILES}/0/0.glb", b"glb", "model/gltf-binary")
    with pytest.raises(PublishError, match="but not"):
        buckets.publish_tree(TILES, f"{TILES}/tileset.json")
    assert keys_in(buckets.public) == set()


# --- many objects: in parallel, the entry last, and all or no site --------------------


class Recorded:
    """The public bucket, with every copy slowed down and its start and end recorded."""

    def __init__(self, inner: S3Storage, *, delay_s: float = 0.05, fail: str = "") -> None:
        self.inner = inner
        self.delay_s = delay_s
        self.fail = fail
        self.spans: dict[str, tuple[float, float]] = {}
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def available(self) -> bool:
        return True

    @property
    def bucket(self) -> str:
        return self.inner.bucket

    def copy_object(self, source_bucket: str, source_key: str, key: str, **kwargs: Any) -> Any:
        started = time.monotonic()
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay_s)
            if key == self.fail:
                raise RuntimeError(f"R2 said no to {key}")
            return self.inner.copy_object(source_bucket, source_key, key, **kwargs)
        finally:
            with self.lock:
                self.active -= 1
                self.spans[key] = (started, time.monotonic())


def a_large_tileset(private: ObjectStorage, tiles: int = 24) -> None:
    """Keys chosen so the listing puts `tileset.json` *before* the files after it in
    sort order, which is what made the entry public first when copies went in order."""
    private.put_object(f"{TILES}/tileset.json", b"{}", "application/json")
    for index in range(tiles):
        private.put_object(f"{TILES}/splat_{index}.glb", b"glb", "model/gltf-binary")
    private.put_object(f"{TILES}/viewcones.bin", b"vc", "application/octet-stream")


class NoEtags:
    """The private bucket as a store that reports no ETags in its listings."""

    def __init__(self, inner: ObjectStorage) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def available(self) -> bool:
        return True

    @property
    def bucket(self) -> str:
        return self.inner.bucket

    def list_objects(self, prefix: str, **kwargs: Any) -> ObjectPage:
        page = self.inner.list_objects(prefix, **kwargs)
        return ObjectPage(
            objects=tuple(
                ObjectSummary(key=o.key, size=o.size, etag="", last_modified=o.last_modified)
                for o in page.objects
            ),
            next_continuation_token=page.next_continuation_token,
        )


def recorded(buckets: Publisher, **kwargs: Any) -> tuple[Publisher, Recorded]:
    assert isinstance(buckets.public, S3Storage)
    slow = Recorded(buckets.public, **kwargs)
    return Publisher(private=buckets.private, public=slow), slow


def test_the_entry_is_copied_last_after_every_other_copy_has_returned(
    buckets: Publisher,
) -> None:
    a_large_tileset(buckets.private)
    publisher, slow = recorded(buckets)

    url = publisher.publish_tree(TILES, f"{TILES}/tileset.json")

    generation = generation_of(url)
    assert url == f"{PUBLIC_URL}/{published_key(TILES, generation)}/tileset.json"
    entry_started, _ = slow.spans.pop(published_key(f"{TILES}/tileset.json", generation))
    assert len(slow.spans) == 25
    assert all(ended <= entry_started for _, ended in slow.spans.values())
    assert keys_in(buckets.public) == {
        published_key(key, generation) for key in keys_in(buckets.private, TILES)
    }


def test_the_copies_overlap(buckets: Publisher) -> None:
    a_large_tileset(buckets.private)
    publisher, slow = recorded(buckets, delay_s=0.1)

    started = time.monotonic()
    publisher.publish_tree(TILES, f"{TILES}/tileset.json")
    took = time.monotonic() - started

    assert slow.peak == PUBLISH_WORKERS
    # 26 copies of 0.1 s: 2.6 s one at a time, four rounds of eight and the entry here.
    assert took < 1.2


def test_a_failed_copy_leaves_no_entry_and_no_site(buckets: Publisher) -> None:
    """The run's outputs are safe in the private bucket; the public one may hold a few
    stray tiles, but never the root that would point a viewer at a partial tileset --
    and `publish_outputs` turns the failure into no tileset URL, so no site."""
    a_large_tileset(buckets.private)
    generation = buckets.generation(prefixes=[TILES])
    assert generation is not None
    entry = published_key(f"{TILES}/tileset.json", generation)
    publisher, slow = recorded(buckets, fail=published_key(f"{TILES}/splat_3.glb", generation))

    with pytest.raises(PublishError, match="R2 said no"):
        publisher.publish_tree(TILES, f"{TILES}/tileset.json")
    assert entry not in keys_in(buckets.public)
    assert entry not in slow.spans

    published = publish_outputs(
        buckets.private,
        publish=publisher,
        job_id=JOB,
        registration=a_registration(),
        tiles_stage_id="package",
    )
    assert published.tileset is None
    assert published.withheld


def test_a_published_copy_has_the_cache_control_of_the_key_it_lands_on(
    buckets: Publisher, tmp_path: Path
) -> None:
    """Uploaded with the short lifetime -- a Refine rewrites the run's own keys -- and
    copied with a year's, because the generation it lands in is never written again."""
    package = tmp_path / "stages" / "package" / "out" / "splat"
    package.mkdir(parents=True)
    (package / "tileset.json").write_text("{}")
    (package / "splat_0.glb").write_bytes(b"glb")
    ref = ArtifactRef(
        name="splat",
        stage_id="package",
        path="stages/package/out/splat",
        kind="dir",
        content_type="application/octet-stream",
        bytes=5,
        checksum="sha256:x",
    )
    assert upload_artifact(buckets.private, tmp_path, JOB, ref) is not None

    generation = generation_of(buckets.publish_tree(TILES, f"{TILES}/tileset.json"))

    raw = boto3.client("s3", region_name="us-east-1")
    original = raw.head_object(Bucket=PRIVATE, Key=f"{TILES}/splat_0.glb")
    assert original["CacheControl"] == "public, max-age=300, stale-while-revalidate=604800"
    tile = raw.head_object(Bucket=PUBLIC, Key=published_key(f"{TILES}/splat_0.glb", generation))
    root = raw.head_object(Bucket=PUBLIC, Key=published_key(f"{TILES}/tileset.json", generation))
    assert tile["CacheControl"] == "public, max-age=31536000, immutable"
    assert tile["ContentType"] == "model/gltf-binary"
    assert root["CacheControl"] == "public, max-age=300, stale-while-revalidate=604800"
    assert root["ContentType"] == "application/json"


# --- registering a republish: the site moves whole, or not at all --------------------


def a_registration(**overrides: Any) -> Registration:
    values: dict[str, Any] = {
        "slug": "orchard",
        "title": "Orchard",
        "recipe": "splat-ingest",
        "lat": 0.0,
        "lon": 0.0,
        "height": 0.0,
        "georef_method": GeorefMethod.NONE,
        "scale_source": ScaleSource.UNRESOLVED,
        "uncertainty_m": 0.0,
        "document": {},
    }
    values.update(overrides)
    return Registration(**values)


def register_run(db: Session, publisher: Publisher, capture: Capture) -> uuid.UUID | None:
    return register(
        db,
        publisher.private,
        publish=publisher,
        capture=capture,
        job_id=JOB,
        registration=a_registration(thumbnail="thumbnail.jpg", coverage="coverage_enu.ply"),
        tiles_stage_id="package",
        thumbnail_stage_id="thumb",
        coverage_stage_id="place",
    )


def a_published_site(db: Session, buckets: Publisher) -> tuple[Capture, Site]:
    seed_a_run(buckets)
    buckets.private.put_object(f"runs/{JOB}/place/coverage_enu.ply", b"ply", "application/x")
    capture = Capture(slug="orchard", name="Orchard", kind=CaptureKind.VIDEO)
    db.add(capture)
    db.commit()
    site_id = register_run(db, buckets, capture)
    site = db.get(Site, site_id)
    assert site is not None
    return capture, site


def splat_of(site: Site) -> dict[str, Any]:
    (asset,) = site.assets
    return {"url": asset.source["url"], "render": dict(asset.render_config)}


def test_a_republish_that_fails_leaves_the_live_site_exactly_as_it_was(
    db: Session, buckets: Publisher
) -> None:
    """Its tiles, its thumbnail, its coverage overlay: nothing of the run that could not be
    published goes on the site, and no key of the live generation is written."""
    capture, site = a_published_site(db, buckets)
    live = splat_of(site)
    thumbnail, coverage = site.thumbnail_url, site.metadata_["coverageUrl"]
    assert is_published(str(live["url"]).removeprefix(f"{PUBLIC_URL}/"))
    before = {key: buckets.public.get_object(key) for key in keys_in(buckets.public)}

    # A Refine: new tiles, a new thumbnail and overlay -- and a copy that fails.
    buckets.private.put_object(f"{TILES}/0/0.glb", b"refined", "model/gltf-binary")
    buckets.private.put_object(f"runs/{JOB}/thumb/thumbnail.jpg", b"new jpg", "image/jpeg")
    generation = buckets.generation(
        prefixes=[TILES],
        keys=[f"runs/{JOB}/thumb/thumbnail.jpg", f"runs/{JOB}/place/coverage_enu.ply"],
    )
    assert generation is not None
    failing, _ = recorded(buckets, delay_s=0, fail=published_key(f"{TILES}/0/0.glb", generation))
    register_run(db, failing, capture)

    db.expire_all()
    site = db.get(Site, site.id)  # type: ignore[assignment]
    assert splat_of(site) == live
    assert (site.thumbnail_url, site.metadata_.get("coverageUrl")) == (thumbnail, coverage)
    for key, data in before.items():
        assert buckets.public.get_object(key) == data, key


def test_a_republish_moves_the_site_and_drops_a_rig_built_on_the_old_tiles(
    db: Session, buckets: Publisher
) -> None:
    """A rig sits beside the tiles it was stamped on (`rigUrl` is relative to the tileset)
    and binds them by checksum, so it cannot come along to a new generation."""
    capture, site = a_published_site(db, buckets)
    (asset,) = site.assets
    asset.render_config = {**dict(asset.render_config), "rigUrl": "rig.json"}
    db.commit()

    # The same bytes again: the same generation, and the rig still describes them.
    register_run(db, buckets, capture)
    db.expire_all()
    assert splat_of(db.get(Site, site.id))["render"]["rigUrl"] == "rig.json"  # type: ignore[arg-type]

    live = splat_of(db.get(Site, site.id))["url"]  # type: ignore[arg-type]
    buckets.private.put_object(f"{TILES}/0/0.glb", b"refined", "model/gltf-binary")
    register_run(db, buckets, capture)
    db.expire_all()
    moved = splat_of(db.get(Site, site.id))  # type: ignore[arg-type]
    assert moved["url"] != live
    assert "rigUrl" not in moved["render"]
    assert (
        buckets.public.get_object(
            str(moved["url"]).removeprefix(f"{PUBLIC_URL}/").replace("tileset.json", "0/0.glb")
        )
        == b"refined"
    )


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
