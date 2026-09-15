"""Object-storage abstraction.

Large 3D assets are never proxied through the API; storage is used for small
derived objects (site thumbnails today; later canonical captures, COGs, COPC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StoredObject:
    key: str
    url: str
    content_type: str
    size: int


class ObjectStorage(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    def put_object(self, key: str, data: bytes, content_type: str) -> StoredObject: ...

    def delete_object(self, key: str) -> None: ...

    def public_url(self, key: str) -> str: ...
