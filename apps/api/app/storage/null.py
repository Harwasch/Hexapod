from __future__ import annotations

from collections.abc import Sequence

from app.storage.base import (
    DEFAULT_EXPIRES_IN,
    MultipartPart,
    ObjectPage,
    StoredObject,
)

_UNAVAILABLE = "Object storage is not configured. Set OBJECT_STORAGE_* in .env (see .env.example)."


class StorageUnavailableError(RuntimeError):
    pass


class NullStorage:
    """Used when no object store is configured; every call raises a clear error."""

    name = "none"

    @property
    def available(self) -> bool:
        return False

    @property
    def bucket(self) -> str:
        """There is no bucket. Empty rather than raising: `Publisher` asks this while
        deciding whether it has anywhere to publish to, which must not be an error."""
        return ""

    def put_object(self, key: str, data: bytes, content_type: str) -> StoredObject:
        raise StorageUnavailableError(_UNAVAILABLE)

    def get_object(self, key: str) -> bytes:
        raise StorageUnavailableError(_UNAVAILABLE)

    def head_object(self, key: str) -> StoredObject | None:
        raise StorageUnavailableError(_UNAVAILABLE)

    def list_objects(
        self,
        prefix: str,
        *,
        continuation_token: str | None = None,
        max_keys: int = 1000,
    ) -> ObjectPage:
        raise StorageUnavailableError(_UNAVAILABLE)

    def delete_object(self, key: str) -> None:
        raise StorageUnavailableError(_UNAVAILABLE)

    def copy_object(self, source_bucket: str, source_key: str, key: str) -> StoredObject:
        raise StorageUnavailableError(_UNAVAILABLE)

    def public_url(self, key: str) -> str:
        raise StorageUnavailableError(_UNAVAILABLE)

    def create_multipart(self, key: str, content_type: str) -> str:
        raise StorageUnavailableError(_UNAVAILABLE)

    def presign_part(
        self,
        key: str,
        upload_id: str,
        part_number: int,
        *,
        expires_in: int = DEFAULT_EXPIRES_IN,
    ) -> str:
        raise StorageUnavailableError(_UNAVAILABLE)

    def complete_multipart(
        self, key: str, upload_id: str, parts: Sequence[MultipartPart]
    ) -> StoredObject:
        raise StorageUnavailableError(_UNAVAILABLE)

    def abort_multipart(self, key: str, upload_id: str) -> None:
        raise StorageUnavailableError(_UNAVAILABLE)

    def presign_get(self, key: str, *, expires_in: int = DEFAULT_EXPIRES_IN) -> str:
        raise StorageUnavailableError(_UNAVAILABLE)

    def presign_put(
        self, key: str, content_type: str, *, expires_in: int = DEFAULT_EXPIRES_IN
    ) -> str:
        raise StorageUnavailableError(_UNAVAILABLE)
