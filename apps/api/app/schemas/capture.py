from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from app.models.enums import CaptureKind, CaptureStatus, GeorefMethod, ScaleSource, UploadStatus
from app.schemas.base import CamelModel
from app.schemas.common import Attribution, LicenseMetadata, Provenance, TemporalExtent
from app.schemas.job import JobRead

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class CaptureBase(CamelModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    kind: CaptureKind
    device: str | None = Field(default=None, max_length=200)
    sensor: str | None = Field(default=None, max_length=120)
    captured_at: datetime | None = None
    temporal_extent: TemporalExtent | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    attribution: list[Attribution] = Field(default_factory=list)
    license: LicenseMetadata | None = None
    provenance: Provenance | None = None


class CaptureCreate(CaptureBase):
    slug: str | None = Field(default=None, max_length=120)

    @field_validator("slug")
    @classmethod
    def _slug(cls, value: str | None) -> str | None:
        if value is not None and not SLUG_RE.match(value):
            raise ValueError("slug must be lowercase letters, digits and single hyphens")
        return value


class CaptureFileCreate(CamelModel):
    filename: str = Field(min_length=1, max_length=500)
    content_type: str | None = Field(default=None, max_length=200)
    bytes: int | None = Field(default=None, ge=0)


class CaptureFileRead(CamelModel):
    """Read model: every field is explicit (no defaults) so the OpenAPI contract marks it
    required."""

    id: uuid.UUID
    capture_id: uuid.UUID
    filename: str
    content_type: str | None
    bytes: int | None
    checksum: str | None
    storage_key: str
    status: UploadStatus
    upload_id: str | None
    parts_total: int | None
    parts_completed: int
    created_at: datetime
    updated_at: datetime


class CaptureRead(CamelModel):
    id: uuid.UUID
    slug: str
    name: str
    description: str | None
    status: CaptureStatus
    kind: CaptureKind
    device: str | None
    sensor: str | None
    captured_at: datetime | None
    temporal_extent: TemporalExtent | None
    # How it was placed on the globe and how well, per the plan's B4. Null until a
    # georeference stage has run.
    georef_method: GeorefMethod | None
    scale_source: ScaleSource | None
    uncertainty_m: float | None
    metadata: dict[str, Any]
    attribution: list[Attribution]
    license: LicenseMetadata | None
    provenance: Provenance | None
    site_id: uuid.UUID | None
    files: list[CaptureFileRead]
    created_at: datetime
    updated_at: datetime


class PresignedPart(CamelModel):
    """One part of a multipart upload, and the URL to PUT it to.

    The uploader must read the `ETag` response header off that PUT and send it back to
    `.../complete` — which needs `ExposeHeaders: ["ETag"]` in the bucket's CORS rule
    (infra/cors/upload.json), or the header is invisible to JavaScript.
    """

    part_number: int
    url: str


class UploadWindow(CamelModel):
    """A bounded run of presigned part URLs, not the whole upload.

    A 12 GB video at 8 MiB parts is 1536 parts, about 590 KB of URLs if they were all
    presigned at once (A0's sizing), and they would start expiring long before the
    upload reached them. So the client uploads this window, then asks
    `POST /captures/{id}/files/{fileId}/parts` for the next one starting at
    `nextPartNumber`. Resuming an interrupted upload is the same call with the first
    part number the client has not confirmed.
    """

    upload_id: str
    storage_key: str
    #: Exact size of every part but the last. Slicing anywhere else fails the upload:
    #: S3 rejects a non-final part under 5 MiB.
    part_size: int
    #: Null when the client did not declare a size, in which case it uploads until its
    #: bytes run out and the part count is whatever it turned out to be.
    parts_total: int | None
    parts: list[PresignedPart]
    #: Seconds until these URLs stop working. Asking for the window again re-presigns it.
    expires_in: int
    #: Where the next window starts; null when this window reaches the last part.
    next_part_number: int | None


class CaptureFileUpload(CamelModel):
    """The registered row plus the first window of presigned parts."""

    file: CaptureFileRead
    upload: UploadWindow


class CaptureFilePartsRequest(CamelModel):
    """Ask for the window of parts starting at `firstPartNumber`.

    `count` is clamped to the server's window size; asking for 1536 parts does not get
    you 1536 parts.
    """

    first_part_number: int = Field(default=1, ge=1)
    count: int | None = Field(default=None, ge=1)


class CaptureFilePart(CamelModel):
    """A part the client has finished uploading, as S3 reported it."""

    part_number: int = Field(ge=1)
    etag: str = Field(min_length=1, max_length=128)


class CaptureFileComplete(CamelModel):
    """Finish the multipart upload. Parts may arrive in any order; they are sorted."""

    parts: list[CaptureFilePart] = Field(min_length=1)
    #: A checksum the client computed over the bytes it sent. Without one the stored
    #: checksum is S3's ETag, which for a multipart object is a digest of digests with a
    #: `-N` suffix — useful for change detection, not an MD5 of the content.
    checksum: str | None = Field(default=None, max_length=128)


class CaptureDetail(CaptureRead):
    """One capture with everything hanging off it: its files and every run over it."""

    jobs: list[JobRead]


class CaptureHandoff(CamelModel):
    """A short-lived way to hand one capture's upload to a phone.

    `qrSvg` is rendered here rather than in the browser on purpose: the console would
    otherwise grow a QR library to draw a picture of a string the server already has.
    It is a complete `<svg>` element, ready to drop into the DOM.

    The token is *not* the write token. It authorises the upload endpoints of this one
    capture and expires; see `app/services/handoff.py` for exactly what it can do.
    """

    capture_id: uuid.UUID
    #: `{publicWebBase}/upload.html#<token>` — the token is in the fragment, so it is
    #: never sent to a server, never lands in an access log, and never leaks in a Referer.
    url: str
    token: str
    expires_at: datetime
    #: Seconds until `token` stops working, for a countdown that needs no clock skew fix.
    expires_in: int
    #: When the chain of renewals ends, however often the phone refreshes its token.
    renewable_until: datetime
    qr_svg: str
