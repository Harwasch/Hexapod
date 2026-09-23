"""Object-storage abstraction.

Large assets are never proxied through the API. Small derived objects (site
thumbnails today) are written directly; big uploads go browser -> storage over
presigned S3 multipart URLs, and the API only ever hands out the URLs.

This layer is deliberately generic: keys are opaque strings, and nothing here
knows about captures, jobs or any key-naming convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

#: Default lifetime for a presigned URL. An hour is long enough for one part of
#: a slow mobile upload and short enough that a leaked URL expires on its own.
DEFAULT_EXPIRES_IN = 3600

#: S3 refuses a non-final multipart part smaller than this.
MIN_MULTIPART_PART_SIZE = 5 * 1024 * 1024


@dataclass(frozen=True)
class StoredObject:
    """One object that exists in the bucket.

    ``etag`` is normalised without S3's surrounding double quotes. For an object
    written by a multipart upload it carries S3's ``-N`` part-count suffix, which
    is why it is not a plain MD5 and must not be treated as one.
    """

    key: str
    url: str
    content_type: str
    size: int
    etag: str | None = None


@dataclass(frozen=True)
class ObjectSummary:
    """A listing entry. Listings carry no content type; S3 does not return one."""

    key: str
    size: int
    etag: str
    last_modified: datetime


@dataclass(frozen=True)
class ObjectPage:
    """One page of a listing.

    ``next_continuation_token`` is None on the last page. Callers walking a whole
    bucket loop while it is not None; nothing here fetches every page for you,
    because a bucket is unbounded.
    """

    objects: tuple[ObjectSummary, ...]
    next_continuation_token: str | None = None

    @property
    def is_truncated(self) -> bool:
        return self.next_continuation_token is not None


@dataclass(frozen=True)
class MultipartPart:
    """One finished part, as reported by the uploader.

    ``etag`` may be given with or without S3's surrounding quotes; the backend
    normalises it. A browser reads this off the PUT response's ``ETag`` header,
    which requires ``ExposeHeaders: ["ETag"]`` in the bucket's CORS rule --
    without it the header is invisible to JavaScript and a multipart upload can
    never be completed. See infra/cors/upload.json.
    """

    part_number: int
    etag: str


class ObjectStorage(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    @property
    def bucket(self) -> str:
        """The bucket these keys live in.

        Exposed because publishing copies an object from one bucket to another and the
        copy has to name its source. Nothing else in the API asks which bucket it is
        talking to, and nothing else should: keys stay opaque.
        """
        ...

    def put_object(self, key: str, data: bytes, content_type: str) -> StoredObject: ...

    def get_object(self, key: str) -> bytes: ...

    def download_file(self, key: str, target: Path) -> int:
        """Stream an object to a file and return its size, never holding it in memory.

        `get_object` returns the whole object as one `bytes`, which is right for a
        manifest and wrong for a 4 GB iPhone video on a worker with 2 GB of RAM: the
        process is killed before the first stage runs, and the job looks like a crash.
        """
        ...

    def head_object(self, key: str) -> StoredObject | None: ...

    def list_objects(
        self,
        prefix: str,
        *,
        continuation_token: str | None = None,
        max_keys: int = 1000,
    ) -> ObjectPage: ...

    def delete_object(self, key: str) -> None: ...

    def copy_object(self, source_bucket: str, source_key: str, key: str) -> StoredObject:
        """Copy an object into this storage from another bucket, server side.

        The bytes never come back to this process, which is the entire point: a packaged
        tileset is hundreds of megabytes and `get_object` returns them all at once. Both
        buckets are reached with the same credentials and the same endpoint, which is
        what makes one `CopyObject` legal across them.
        """
        ...

    def public_url(self, key: str) -> str: ...

    def create_multipart(self, key: str, content_type: str) -> str: ...

    def presign_part(
        self,
        key: str,
        upload_id: str,
        part_number: int,
        *,
        expires_in: int = DEFAULT_EXPIRES_IN,
    ) -> str: ...

    def complete_multipart(
        self, key: str, upload_id: str, parts: Sequence[MultipartPart]
    ) -> StoredObject: ...

    def abort_multipart(self, key: str, upload_id: str) -> None: ...

    def presign_get(self, key: str, *, expires_in: int = DEFAULT_EXPIRES_IN) -> str: ...

    def presign_put(
        self, key: str, content_type: str, *, expires_in: int = DEFAULT_EXPIRES_IN
    ) -> str: ...


def normalise_etag(etag: str) -> str:
    """Strip S3's surrounding double quotes from an ETag."""
    return etag.strip('"')


def quote_etag(etag: str) -> str:
    """Put S3's surrounding double quotes back, for sending in a request."""
    return f'"{normalise_etag(etag)}"'
