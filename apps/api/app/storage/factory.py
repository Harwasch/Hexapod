from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.storage.base import ObjectStorage
from app.storage.null import NullStorage
from app.storage.s3 import S3Storage


def build_storage(settings: Settings) -> ObjectStorage:
    if not settings.object_storage_configured:
        return NullStorage()
    assert settings.object_storage_bucket
    assert settings.object_storage_access_key
    assert settings.object_storage_secret_key
    return S3Storage(
        bucket=settings.object_storage_bucket,
        endpoint_url=settings.object_storage_endpoint_url,
        access_key=settings.object_storage_access_key,
        secret_key=settings.object_storage_secret_key,
        region=settings.object_storage_region,
        public_base_url=settings.object_storage_public_url,
    )


def build_publish_storage(settings: Settings) -> ObjectStorage:
    """The bucket published tiles are copied into, or `NullStorage` where there is none.

    Same endpoint, same credentials, different bucket -- which is what lets one
    `CopyObject` move an object across without the bytes coming back here. The public
    base URL belongs to *this* bucket: it is the host a browser fetches a tileset from.
    """
    if not (settings.object_storage_configured and settings.publish_bucket_configured):
        return NullStorage()
    assert settings.object_storage_public_bucket
    assert settings.object_storage_access_key
    assert settings.object_storage_secret_key
    return S3Storage(
        bucket=settings.object_storage_public_bucket,
        endpoint_url=settings.object_storage_endpoint_url,
        access_key=settings.object_storage_access_key,
        secret_key=settings.object_storage_secret_key,
        region=settings.object_storage_region,
        public_base_url=settings.object_storage_public_url,
    )


def build_public_storage(settings: Settings) -> ObjectStorage:
    """Where an object a browser fetches lives: the public bucket, or the only bucket.

    The one place that encodes "one bucket or two" for everything that writes something
    world-readable up front rather than copying it later -- `app/seed/publish.py` puts
    migrated capture tiles and the offline catalog straight here, because `sites/` and
    `catalog.json` are public by their whole purpose and never hold anything else. The
    run outputs take the other route, through `app/worker/publish.py`, because they are
    produced into the private bucket and only some of them are ever published.
    """
    split = build_publish_storage(settings)
    return split if split.available else build_storage(settings)


@lru_cache
def get_storage() -> ObjectStorage:
    return build_storage(get_settings())


@lru_cache
def get_publish_storage() -> ObjectStorage:
    return build_publish_storage(get_settings())
