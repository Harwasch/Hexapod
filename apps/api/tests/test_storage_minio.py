"""Storage primitives against a real S3 implementation (MinIO).

This is the verification for A1, not tests/test_storage_objects.py. moto proved
unable to speak for the two things that matter most here -- it returns 200 for a
PUT with no signature at all, and it ignores bucket CORS rules -- so the
presigned-URL paths are exercised against a real server over real HTTP.

CI runs a MinIO container and sets TEST_S3_ENDPOINT_URL. On a machine without one
these tests skip, and test_real_storage_is_exercised_in_ci fails the suite if CI
ever starts skipping them silently, which would leave the job green while proving
nothing.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import boto3
import httpx
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError

from app.storage import MIN_MULTIPART_PART_SIZE, MultipartPart, S3Storage

ENDPOINT_URL = os.environ.get("TEST_S3_ENDPOINT_URL")
ACCESS_KEY = os.environ.get("TEST_S3_ACCESS_KEY", "twin")
SECRET_KEY = os.environ.get("TEST_S3_SECRET_KEY", "twin-secret")
BUCKET = os.environ.get("TEST_S3_BUCKET", "twin-test")
# us-east-1 on purpose: it is a region where botocore's presign default drops to
# SigV2 (A0 #1). A real server rejecting SigV2-signed requests is what turns the
# explicit signature_version from a claim into a fact.
REGION = os.environ.get("TEST_S3_REGION", "us-east-1")

requires_minio = pytest.mark.skipif(
    not ENDPOINT_URL,
    reason="set TEST_S3_ENDPOINT_URL to run the real-S3 tests (CI does)",
)


def test_real_storage_is_exercised_in_ci() -> None:
    """The one test in this file that never skips.

    A skipped suite is indistinguishable from a passing one in a CI summary. If
    the MinIO service disappears from the workflow, this fails instead of the job
    going green while testing nothing.
    """
    if os.environ.get("CI", "").lower() == "true":
        assert ENDPOINT_URL, (
            "TEST_S3_ENDPOINT_URL is unset under CI: the MinIO service container "
            "is missing from .github/workflows/ci.yml, so the presigned-URL tests "
            "would skip and A1 would be unverified"
        )


@pytest.fixture
def storage() -> Iterator[S3Storage]:
    assert ENDPOINT_URL
    client = boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
    )
    try:
        client.create_bucket(Bucket=BUCKET)
    except ClientError as error:
        if error.response["Error"]["Code"] not in {
            "BucketAlreadyOwnedByYou",
            "BucketAlreadyExists",
        }:
            raise

    instance = S3Storage(
        bucket=BUCKET,
        endpoint_url=ENDPOINT_URL,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        region=REGION,
        public_base_url=None,
    )
    yield instance

    for upload in client.list_multipart_uploads(Bucket=BUCKET).get("Uploads", []):
        client.abort_multipart_upload(Bucket=BUCKET, Key=upload["Key"], UploadId=upload["UploadId"])
    keys = [item["Key"] for item in client.list_objects_v2(Bucket=BUCKET).get("Contents", [])]
    if keys:
        client.delete_objects(Bucket=BUCKET, Delete={"Objects": [{"Key": k} for k in keys]})


@pytest.fixture
def prefix() -> str:
    return f"t/{uuid.uuid4().hex}"


# --- presigned single-shot --------------------------------------------------


@requires_minio
def test_presigned_put_is_accepted_by_a_real_server(storage: S3Storage, prefix: str) -> None:
    """The SigV4 proof. A SigV2 URL is what this rejects."""
    key = f"{prefix}/small.bin"
    url = storage.presign_put(key, "application/octet-stream", expires_in=120)
    assert "X-Amz-Signature=" in url and "AWSAccessKeyId=" not in url

    response = httpx.put(
        url, content=b"hello", headers={"Content-Type": "application/octet-stream"}
    )
    assert response.status_code == 200, response.text

    assert storage.get_object(key) == b"hello"
    head = storage.head_object(key)
    assert head is not None and head.size == 5


@requires_minio
def test_presigned_get_round_trip(storage: S3Storage, prefix: str) -> None:
    key = f"{prefix}/get.txt"
    storage.put_object(key, b"body bytes", "text/plain")
    response = httpx.get(storage.presign_get(key, expires_in=120))
    assert response.status_code == 200
    assert response.content == b"body bytes"


@requires_minio
def test_ranged_get_returns_206(storage: S3Storage, prefix: str) -> None:
    key = f"{prefix}/range.bin"
    storage.put_object(key, b"0123456789", "application/octet-stream")
    response = httpx.get(storage.presign_get(key, expires_in=120), headers={"Range": "bytes=2-5"})
    assert response.status_code == 206
    assert response.content == b"2345"


@requires_minio
def test_unsigned_and_tampered_requests_are_rejected(storage: S3Storage, prefix: str) -> None:
    """The assertion moto cannot make: it returns 200 for an unsigned PUT."""
    key = f"{prefix}/denied.bin"
    assert ENDPOINT_URL
    unsigned = httpx.put(f"{ENDPOINT_URL.rstrip('/')}/{BUCKET}/{key}", content=b"x")
    assert unsigned.status_code == 403, unsigned.text

    url = storage.presign_put(key, "application/octet-stream", expires_in=120)
    signature = url.split("X-Amz-Signature=")[1]
    tampered = url.replace(signature, ("0" if signature[0] != "0" else "1") + signature[1:])
    response = httpx.put(
        tampered, content=b"x", headers={"Content-Type": "application/octet-stream"}
    )
    assert response.status_code == 403, response.text
    assert storage.head_object(key) is None


@requires_minio
def test_presigned_put_binds_the_content_type(storage: S3Storage, prefix: str) -> None:
    """Content-Type is signed, so the uploader has to send the one it asked for."""
    key = f"{prefix}/typed.bin"
    url = storage.presign_put(key, "image/png", expires_in=120)
    mismatched = httpx.put(url, content=b"x", headers={"Content-Type": "text/plain"})
    assert mismatched.status_code == 403
    matched = httpx.put(url, content=b"x", headers={"Content-Type": "image/png"})
    assert matched.status_code == 200
    head = storage.head_object(key)
    assert head is not None and head.content_type == "image/png"


# --- presigned multipart, the way the browser will do it --------------------


@requires_minio
def test_multipart_through_presigned_part_urls(storage: S3Storage, prefix: str) -> None:
    key = f"{prefix}/big.bin"
    upload_id = storage.create_multipart(key, "application/octet-stream")

    first = os.urandom(MIN_MULTIPART_PART_SIZE)
    last = os.urandom(1024)
    parts: list[MultipartPart] = []
    for number, chunk in ((1, first), (2, last)):
        url = storage.presign_part(key, upload_id, number, expires_in=300)
        response = httpx.put(url, content=chunk, timeout=120)
        assert response.status_code == 200, response.text
        # A browser only sees this header if the bucket's CORS rule carries
        # ExposeHeaders: ["ETag"] (infra/cors/upload.json). Server-side it is
        # always present, so this asserts the value, not the exposure.
        etag = response.headers["etag"]
        assert etag
        parts.append(MultipartPart(number, etag))

    stored = storage.complete_multipart(key, upload_id, parts)
    assert stored.size == len(first) + len(last)
    assert stored.etag is not None and stored.etag.endswith("-2")
    assert storage.get_object(key) == first + last


@requires_minio
def test_undersized_non_final_part_is_rejected(storage: S3Storage, prefix: str) -> None:
    key = f"{prefix}/short.bin"
    upload_id = storage.create_multipart(key, "application/octet-stream")
    parts: list[MultipartPart] = []
    for number, chunk in ((1, b"too small"), (2, b"tail")):
        url = storage.presign_part(key, upload_id, number, expires_in=300)
        response = httpx.put(url, content=chunk)
        assert response.status_code == 200
        parts.append(MultipartPart(number, response.headers["etag"]))

    with pytest.raises(ClientError) as excinfo:
        storage.complete_multipart(key, upload_id, parts)
    assert excinfo.value.response["Error"]["Code"] == "EntityTooSmall"


@requires_minio
def test_abort_multipart_returns_no_such_upload(storage: S3Storage, prefix: str) -> None:
    """Real S3 semantics. moto raises a bare KeyError from its own backend here."""
    key = f"{prefix}/aborted.bin"
    upload_id = storage.create_multipart(key, "application/octet-stream")
    url = storage.presign_part(key, upload_id, 1, expires_in=300)
    assert (
        httpx.put(url, content=os.urandom(MIN_MULTIPART_PART_SIZE), timeout=120).status_code == 200
    )

    storage.abort_multipart(key, upload_id)
    assert storage.head_object(key) is None

    with pytest.raises(ClientError) as excinfo:
        storage.complete_multipart(key, upload_id, [MultipartPart(1, "whatever")])
    assert excinfo.value.response["Error"]["Code"] == "NoSuchUpload"


# --- listing ----------------------------------------------------------------


@requires_minio
def test_list_objects_paginates(storage: S3Storage, prefix: str) -> None:
    for index in range(5):
        storage.put_object(f"{prefix}/{index}.txt", b"x", "text/plain")

    seen: list[str] = []
    token: str | None = None
    pages = 0
    while True:
        page = storage.list_objects(f"{prefix}/", continuation_token=token, max_keys=2)
        pages += 1
        seen.extend(item.key for item in page.objects)
        token = page.next_continuation_token
        if token is None:
            break
        assert pages < 10, "pagination did not terminate"

    assert pages == 3
    assert seen == [f"{prefix}/{index}.txt" for index in range(5)]
    assert all(entry.size == 1 and entry.etag for entry in page.objects)
