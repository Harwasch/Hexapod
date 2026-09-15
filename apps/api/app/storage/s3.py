from __future__ import annotations

from typing import TYPE_CHECKING

import boto3
from botocore.config import Config

from app.storage.base import StoredObject

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


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
            config=Config(s3={"addressing_style": "path"}),
        )
        self._endpoint_url = endpoint_url

    @property
    def available(self) -> bool:
        return True

    def put_object(self, key: str, data: bytes, content_type: str) -> StoredObject:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)
        return StoredObject(
            key=key, url=self.public_url(key), content_type=content_type, size=len(data)
        )

    def delete_object(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def public_url(self, key: str) -> str:
        if self._public_base_url:
            return f"{self._public_base_url}/{key}"
        if self._endpoint_url:
            return f"{self._endpoint_url.rstrip('/')}/{self._bucket}/{key}"
        return f"https://{self._bucket}.s3.amazonaws.com/{key}"
