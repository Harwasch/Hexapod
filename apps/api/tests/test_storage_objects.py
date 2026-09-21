"""Storage primitives against moto.

moto is trustworthy for the S3 semantics asserted here: ETag values including the
multipart ``-N`` suffix, EntityTooSmall on an undersized non-final part, and
NoSuchUpload after an abort.

It is NOT trustworthy for two things, so nothing below asserts on them:

* **Auth.** A PUT to moto with no signature at all returns 200, and a tampered or
  expired signature returns 500 rather than 403. A test here that "proved" a bad
  signature is rejected would be testing the mock. The real proof is
  tests/test_storage_minio.py, which runs against a MinIO service container in CI.
* **CORS.** moto ignores bucket CORS rules entirely and invents response headers.
  The rules live in infra/cors/ and are verified at deploy time, not here.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, cast

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from moto import mock_aws

from app.storage import (
    MIN_MULTIPART_PART_SIZE,
    MultipartPart,
    NullStorage,
    S3Storage,
    StorageUnavailableError,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

BUCKET = "twin-test"
# us-east-1 on purpose: it is one of the regions where botocore's presign default
# silently drops to SigV2 (A0 #1), so these tests exercise the case that breaks.
REGION = "us-east-1"


@pytest.fixture
def storage() -> Iterator[S3Storage]:
    with mock_aws():
        boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
        # endpoint_url is None so the requests go to the AWS hostnames moto
        # intercepts; a custom endpoint host is not recognised by the mock and
        # botocore tries to open a real connection.
        yield S3Storage(
            bucket=BUCKET,
            endpoint_url=None,
            access_key="key",
            secret_key="secret",
            region=REGION,
            public_base_url="https://cdn.example.com/twin-test",
        )


@pytest.fixture
def raw_client(storage: S3Storage) -> S3Client:
    return boto3.client("s3", region_name=REGION)


# --- the SigV2 regression (A0 #1) ------------------------------------------


@pytest.mark.parametrize("scheme", ["get", "put", "part"])
def test_presigned_urls_are_sigv4(storage: S3Storage, scheme: str) -> None:
    """Assert on the emitted URL, never on client.meta.config.

    client.meta.config.signature_version reads 's3v4' even while botocore is
    emitting SigV2, so a test that inspects the config object passes with the bug
    live. The query string is the only place the difference is visible.
    """
    if scheme == "get":
        url = storage.presign_get("a/b.bin", expires_in=60)
    elif scheme == "put":
        url = storage.presign_put("a/b.bin", "application/octet-stream", expires_in=60)
    else:
        url = storage.presign_part("a/b.bin", "upload-id", 1, expires_in=60)

    assert "X-Amz-Signature=" in url
    assert "X-Amz-Algorithm=AWS4-HMAC-SHA256" in url
    assert "X-Amz-Credential=" in url
    assert "X-Amz-Expires=60" in url
    # SigV2 markers. Their absence is the whole point of the explicit
    # signature_version in S3Storage.__init__. ("Expires" alone is not a marker:
    # SigV4 spells it X-Amz-Expires.)
    assert "AWSAccessKeyId=" not in url
    assert "&Signature=" not in url


def test_botocore_default_still_presigns_sigv2() -> None:
    """Pins the finding that makes the explicit signature_version load-bearing.

    This constructs the client exactly as app/storage/s3.py did before A1 -- path
    addressing, no explicit signature_version -- and asserts it emits SigV2 while
    reporting s3v4. If botocore ever fixes _default_s3_presign_to_sigv2 this test
    fails, which is the signal to re-read the comment in S3Storage.__init__ and
    decide whether the explicit setting is still needed. It is not a reason to
    delete the explicit setting without checking every supported region.
    """
    client = boto3.client(
        "s3",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
        region_name=REGION,
        config=Config(s3={"addressing_style": "path"}),
    )
    url = client.generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": "k"}, ExpiresIn=60
    )
    # cast because botocore's Config is untyped here; the point is what it reports.
    assert cast(Any, client.meta.config).signature_version == "s3v4"  # the config lies
    assert "AWSAccessKeyId=" in url  # ...the URL tells the truth
    assert "X-Amz-Signature=" not in url


# --- reads ------------------------------------------------------------------


def test_put_get_head_round_trip(storage: S3Storage) -> None:
    stored = storage.put_object("a/b.txt", b"hello", "text/plain")
    assert stored.size == 5 and stored.etag and '"' not in stored.etag

    assert storage.get_object("a/b.txt") == b"hello"

    head = storage.head_object("a/b.txt")
    assert head is not None
    assert head.size == 5
    assert head.content_type == "text/plain"
    assert head.etag == stored.etag
    assert head.url == storage.public_url("a/b.txt")


def test_head_object_returns_none_when_absent(storage: S3Storage) -> None:
    assert storage.head_object("nope/missing.txt") is None


def test_get_object_still_raises_when_absent(storage: S3Storage) -> None:
    with pytest.raises(ClientError):
        storage.get_object("nope/missing.txt")


def test_list_objects_paginates(storage: S3Storage) -> None:
    for index in range(5):
        storage.put_object(f"p/{index}.txt", b"x", "text/plain")
    storage.put_object("other/x.txt", b"x", "text/plain")

    seen: list[str] = []
    token: str | None = None
    pages = 0
    while True:
        page = storage.list_objects("p/", continuation_token=token, max_keys=2)
        pages += 1
        seen.extend(item.key for item in page.objects)
        token = page.next_continuation_token
        if token is None:
            break
        assert page.is_truncated
        assert pages < 10, "pagination did not terminate"

    assert pages == 3
    assert seen == [f"p/{index}.txt" for index in range(5)]


def test_list_objects_entries_carry_metadata(storage: S3Storage) -> None:
    storage.put_object("p/one.txt", b"abcd", "text/plain")
    page = storage.list_objects("p/")
    assert not page.is_truncated
    (entry,) = page.objects
    assert entry.key == "p/one.txt"
    assert entry.size == 4
    assert '"' not in entry.etag
    assert entry.last_modified is not None


def test_delete_object(storage: S3Storage) -> None:
    storage.put_object("gone.txt", b"x", "text/plain")
    storage.delete_object("gone.txt")
    assert storage.head_object("gone.txt") is None


# --- multipart --------------------------------------------------------------


def _upload_part(raw_client: S3Client, key: str, upload_id: str, number: int, body: bytes) -> str:
    response = raw_client.upload_part(
        Bucket=BUCKET, Key=key, UploadId=upload_id, PartNumber=number, Body=body
    )
    return str(response["ETag"])


def test_multipart_round_trip(storage: S3Storage, raw_client: S3Client) -> None:
    key = "big/file.bin"
    upload_id = storage.create_multipart(key, "application/octet-stream")

    first = os.urandom(MIN_MULTIPART_PART_SIZE)
    last = b"tail"
    parts = [
        MultipartPart(1, _upload_part(raw_client, key, upload_id, 1, first)),
        MultipartPart(2, _upload_part(raw_client, key, upload_id, 2, last)),
    ]

    # Out of order on purpose: the backend sorts before sending.
    stored = storage.complete_multipart(key, upload_id, list(reversed(parts)))

    assert stored.size == len(first) + len(last)
    assert stored.etag is not None and stored.etag.endswith("-2")
    assert storage.get_object(key) == first + last


def test_multipart_accepts_quoted_or_bare_etags(storage: S3Storage, raw_client: S3Client) -> None:
    key = "big/quoted.bin"
    upload_id = storage.create_multipart(key, "application/octet-stream")
    first = _upload_part(raw_client, key, upload_id, 1, os.urandom(MIN_MULTIPART_PART_SIZE))
    last = _upload_part(raw_client, key, upload_id, 2, b"tail")
    stored = storage.complete_multipart(
        key,
        upload_id,
        [MultipartPart(1, first.strip('"')), MultipartPart(2, last)],
    )
    assert stored.etag is not None and stored.etag.endswith("-2")


def test_undersized_non_final_part_is_rejected(storage: S3Storage, raw_client: S3Client) -> None:
    key = "big/short.bin"
    upload_id = storage.create_multipart(key, "application/octet-stream")
    parts = [
        MultipartPart(1, _upload_part(raw_client, key, upload_id, 1, b"too small")),
        MultipartPart(2, _upload_part(raw_client, key, upload_id, 2, b"tail")),
    ]
    with pytest.raises(ClientError) as excinfo:
        storage.complete_multipart(key, upload_id, parts)
    assert excinfo.value.response["Error"]["Code"] == "EntityTooSmall"


def test_abort_multipart_invalidates_the_upload(storage: S3Storage, raw_client: S3Client) -> None:
    key = "big/aborted.bin"
    upload_id = storage.create_multipart(key, "application/octet-stream")
    _upload_part(raw_client, key, upload_id, 1, os.urandom(MIN_MULTIPART_PART_SIZE))
    storage.abort_multipart(key, upload_id)

    assert storage.head_object(key) is None
    # moto 5.2.3 raises a bare KeyError from its own backend here rather than the
    # NoSuchUpload ClientError real S3 returns, so this only asserts that the
    # upload id is dead. The error *code* is asserted against real MinIO in
    # tests/test_storage_minio.py.
    with pytest.raises((ClientError, KeyError)):
        _upload_part(raw_client, key, upload_id, 2, b"tail")


# --- the unconfigured backend ----------------------------------------------


def test_null_storage_raises_from_every_method() -> None:
    null = NullStorage()
    assert null.available is False
    calls: list[Callable[[], object]] = [
        lambda: null.put_object("k", b"x", "text/plain"),
        lambda: null.get_object("k"),
        lambda: null.head_object("k"),
        lambda: null.list_objects("p/"),
        lambda: null.delete_object("k"),
        lambda: null.public_url("k"),
        lambda: null.create_multipart("k", "text/plain"),
        lambda: null.presign_part("k", "u", 1),
        lambda: null.complete_multipart("k", "u", [MultipartPart(1, "e")]),
        lambda: null.abort_multipart("k", "u"),
        lambda: null.presign_get("k"),
        lambda: null.presign_put("k", "text/plain"),
    ]
    for call in calls:
        with pytest.raises(StorageUnavailableError) as excinfo:
            call()
        assert "OBJECT_STORAGE_" in str(excinfo.value)
