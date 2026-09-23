from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.storage.base import (
    DEFAULT_EXPIRES_IN,
    MultipartPart,
    ObjectPage,
    ObjectSummary,
    StoredObject,
    normalise_etag,
    quote_etag,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.type_defs import CompletedPartTypeDef

_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class S3Storage:
    """S3-compatible storage (AWS S3, MinIO, Cloudflare R2...)."""

    name = "s3"

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None,
        access_key: str,
        secret_key: str,
        region: str,
        public_base_url: str | None,
    ) -> None:
        self._bucket = bucket
        self._public_base_url = public_base_url.rstrip("/") if public_base_url else None
        self._client: S3Client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            # signature_version is explicit on purpose -- do not remove it as
            # redundant. Without it botocore's _default_s3_presign_to_sigv2 makes
            # generate_presigned_url emit SigV2 (?AWSAccessKeyId&Signature&Expires)
            # for every region except "auto", while client.meta.config.signature_version
            # still reads "s3v4". So MinIO in dev (us-east-1) would presign SigV2 and
            # R2 in production (region "auto") SigV4 -- a dev/prod split that no config
            # inspection can see. tests/test_storage_objects.py::test_presigned_urls_are_sigv4
            # asserts on the URL string itself -- the only place the difference shows up --
            # and tests/test_storage_minio.py puts the URL to a real server in CI.
            config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
        )
        self._endpoint_url = endpoint_url

    @property
    def available(self) -> bool:
        return True

    @property
    def bucket(self) -> str:
        return self._bucket

    # --- reads -------------------------------------------------------------

    def put_object(self, key: str, data: bytes, content_type: str) -> StoredObject:
        response = self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
        )
        return StoredObject(
            key=key,
            url=self.public_url(key),
            content_type=content_type,
            size=len(data),
            etag=normalise_etag(response.get("ETag", "")) or None,
        )

    def get_object(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def download_file(self, key: str, target: Path) -> int:
        # boto3's managed transfer: ranged GETs written to a temporary file beside the
        # target and renamed into place, so memory stays at a few chunks whatever the
        # object's size, and a half-written download never looks like a finished one.
        target.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self._bucket, key, str(target))
        return target.stat().st_size

    def head_object(self, key: str) -> StoredObject | None:
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as error:
            if _is_missing(error):
                return None
            raise
        return StoredObject(
            key=key,
            url=self.public_url(key),
            content_type=response.get("ContentType", "application/octet-stream"),
            size=int(response.get("ContentLength", 0)),
            etag=normalise_etag(response.get("ETag", "")) or None,
        )

    def list_objects(
        self,
        prefix: str,
        *,
        continuation_token: str | None = None,
        max_keys: int = 1000,
    ) -> ObjectPage:
        """One page of a listing. Loop on the returned token to walk a bucket."""
        response = (
            self._client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=prefix,
                MaxKeys=max_keys,
                ContinuationToken=continuation_token,
            )
            if continuation_token
            else self._client.list_objects_v2(Bucket=self._bucket, Prefix=prefix, MaxKeys=max_keys)
        )
        objects = tuple(
            ObjectSummary(
                key=item["Key"],
                size=int(item["Size"]),
                etag=normalise_etag(item.get("ETag", "")),
                last_modified=item["LastModified"],
            )
            for item in response.get("Contents", [])
        )
        token = response.get("NextContinuationToken") if response.get("IsTruncated") else None
        return ObjectPage(objects=objects, next_continuation_token=token)

    def delete_object(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def copy_object(self, source_bucket: str, source_key: str, key: str) -> StoredObject:
        self._client.copy_object(
            Bucket=self._bucket,
            Key=key,
            CopySource={"Bucket": source_bucket, "Key": source_key},
        )
        copied = self.head_object(key)
        if copied is None:  # pragma: no cover - a copy that succeeded and then vanished
            raise RuntimeError(f"copied {source_bucket}/{source_key} to {key}, and it is not there")
        return copied

    def public_url(self, key: str) -> str:
        if self._public_base_url:
            return f"{self._public_base_url}/{key}"
        if self._endpoint_url:
            return f"{self._endpoint_url.rstrip('/')}/{self._bucket}/{key}"
        return f"https://{self._bucket}.s3.amazonaws.com/{key}"

    # --- multipart ---------------------------------------------------------

    def create_multipart(self, key: str, content_type: str) -> str:
        response = self._client.create_multipart_upload(
            Bucket=self._bucket, Key=key, ContentType=content_type
        )
        return response["UploadId"]

    def presign_part(
        self,
        key: str,
        upload_id: str,
        part_number: int,
        *,
        expires_in: int = DEFAULT_EXPIRES_IN,
    ) -> str:
        """Presign exactly one part.

        One call per part on purpose: a 12 GB upload at 8 MiB parts is 1536 parts,
        and presigning all of them at once is ~590 KB of URLs in one response.
        Callers presign a window ahead of the uploader instead.
        """
        return self._client.generate_presigned_url(
            "upload_part",
            Params={
                "Bucket": self._bucket,
                "Key": key,
                "UploadId": upload_id,
                "PartNumber": part_number,
            },
            ExpiresIn=expires_in,
        )

    def complete_multipart(
        self, key: str, upload_id: str, parts: Sequence[MultipartPart]
    ) -> StoredObject:
        completed: list[CompletedPartTypeDef] = [
            {"PartNumber": part.part_number, "ETag": quote_etag(part.etag)}
            for part in sorted(parts, key=lambda part: part.part_number)
        ]
        self._client.complete_multipart_upload(
            Bucket=self._bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": completed},
        )
        stored = self.head_object(key)
        if stored is None:  # pragma: no cover - S3 just told us it exists
            raise RuntimeError(f"completed multipart upload but {key!r} is not readable")
        return stored

    def abort_multipart(self, key: str, upload_id: str) -> None:
        self._client.abort_multipart_upload(Bucket=self._bucket, Key=key, UploadId=upload_id)

    # --- single-shot presigns ----------------------------------------------

    def presign_get(self, key: str, *, expires_in: int = DEFAULT_EXPIRES_IN) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def presign_put(
        self, key: str, content_type: str, *, expires_in: int = DEFAULT_EXPIRES_IN
    ) -> str:
        """Single-shot upload. The client must send exactly this Content-Type:
        it is part of what was signed, so a mismatch fails the signature check.
        """
        return self._client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self._bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=expires_in,
        )


def _is_missing(error: ClientError) -> bool:
    return str(error.response.get("Error", {}).get("Code", "")) in _MISSING_CODES
