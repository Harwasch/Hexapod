from __future__ import annotations

from app.storage.base import StoredObject


class StorageUnavailableError(RuntimeError):
    pass


class NullStorage:
    """Used when no object store is configured; every write raises a clear error."""

    name = "none"

    @property
    def available(self) -> bool:
        return False

    def put_object(self, key: str, data: bytes, content_type: str) -> StoredObject:
        raise StorageUnavailableError(
            "Object storage is not configured. Set OBJECT_STORAGE_* in .env (see .env.example)."
        )

    def delete_object(self, key: str) -> None:
        raise StorageUnavailableError("Object storage is not configured.")

    def public_url(self, key: str) -> str:
        raise StorageUnavailableError("Object storage is not configured.")
