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


@lru_cache
def get_storage() -> ObjectStorage:
    return build_storage(get_settings())
