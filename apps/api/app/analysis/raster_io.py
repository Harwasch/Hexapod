"""Bounded, version-checked COG reads through the application HTTP transport.

GDAL sees a read-only Python file and never gets an arbitrary network URL to fetch.
Only explicitly registered public dataset hosts are accepted; auxiliary files and
redirects are not followed. Shared budgets apply across every source in an analysis.
"""

from __future__ import annotations

import io
import re
import time
from collections import OrderedDict
from collections.abc import Buffer
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

HOSTS = frozenset(
    {
        "copernicus-dem-30m.s3.eu-central-1.amazonaws.com",
        "copernicus-dem-90m.s3.eu-central-1.amazonaws.com",
        "esa-worldcover.s3.eu-central-1.amazonaws.com",
        "sentinel-cogs.s3.us-west-2.amazonaws.com",
        "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com",
    }
)
BLOCK_SIZE = 64 * 1024


@dataclass
class ReadBudget:
    max_bytes: int = 64 * 1024 * 1024
    max_requests: int = 256
    max_seconds: float = 60
    downloaded_bytes: int = 0
    requests: int = 0
    started: float = field(default_factory=time.monotonic)

    def remaining(self) -> float:
        remaining = self.max_seconds - (time.monotonic() - self.started)
        if (
            remaining <= 0
            or self.requests >= self.max_requests
            or self.downloaded_bytes >= self.max_bytes
        ):
            raise OSError("The raster read budget was exhausted.")
        return min(20, remaining)


class RasterHttpFile(io.RawIOBase):
    def __init__(self, client: httpx.Client, url: str, budget: ReadBudget) -> None:
        super().__init__()
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in HOSTS
            or parsed.port not in (None, 443)
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not parsed.path.endswith(".tif")
            or "/../" in parsed.path
        ):
            raise OSError("This raster URL is not an approved public dataset asset.")
        self.client, self.url, self.budget = client, url, budget
        self.position = 0
        self.size = 0
        self.etag: str | None = None
        self.last_modified: str | None = None
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self._range(0, 0, probe=True)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if self.closed:
            raise ValueError("I/O operation on a closed raster")
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError("Invalid seek mode")
        position = offset + (
            self.position if whence == io.SEEK_CUR else self.size if whence == io.SEEK_END else 0
        )
        if position < 0:
            raise ValueError("Negative raster offset")
        self.position = position
        return position

    def _range(self, start: int, end: int, *, probe: bool = False) -> bytes:
        timeout = self.budget.remaining()
        expected = end - start + 1
        if expected > self.budget.max_bytes - self.budget.downloaded_bytes:
            raise OSError("The raster download exceeds the remaining byte budget.")
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self.etag and not self.etag.startswith("W/"):
            headers["If-Match"] = self.etag
        self.budget.requests += 1
        with self.client.stream(
            "GET", self.url, headers=headers, timeout=timeout, follow_redirects=False
        ) as response:
            if response.status_code != 206:
                raise OSError(
                    f"The raster source must support byte ranges and remain unchanged (HTTP {response.status_code})."
                )
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if match is None or int(match[1]) != start or int(match[2]) != end:
                raise OSError("The raster source returned an unexpected byte range.")
            size = int(match[3])
            if size <= end:
                raise OSError("The raster source returned an invalid file size.")
            etag, modified = response.headers.get("ETag"), response.headers.get("Last-Modified")
            if probe:
                if not etag and not modified:
                    raise OSError("The raster source does not expose a stable version validator.")
                self.size, self.etag, self.last_modified = size, etag, modified
            elif (self.size, self.etag, self.last_modified) != (size, etag, modified):
                raise OSError(
                    "The source raster changed during analysis. Retry against a stable version."
                )
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise OSError("Encoded raster ranges are unsupported.")
            data = bytearray()
            for chunk in response.iter_raw():
                if time.monotonic() - self.budget.started > self.budget.max_seconds:
                    raise OSError("The raster read exceeded its time budget.")
                self.budget.downloaded_bytes += len(chunk)
                if (
                    len(data) + len(chunk) > expected
                    or self.budget.downloaded_bytes > self.budget.max_bytes
                ):
                    raise OSError("The raster source exceeded the requested byte range.")
                data.extend(chunk)
            if len(data) != expected:
                raise OSError("The raster range was incomplete.")
            return bytes(data)

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on a closed raster")
        size = (
            max(0, self.size - self.position)
            if size < 0
            else min(size, max(0, self.size - self.position))
        )
        if size > self.budget.max_bytes:
            raise OSError("This raster read exceeds the analysis memory limit.")
        result = bytearray()
        while len(result) < size:
            block = self.position // BLOCK_SIZE
            if block not in self.cache:
                start = block * BLOCK_SIZE
                self.cache[block] = self._range(start, min(self.size - 1, start + BLOCK_SIZE - 1))
                if len(self.cache) > 64:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(block)
            data = self.cache[block]
            offset = self.position % BLOCK_SIZE
            chunk = data[offset : offset + size - len(result)]
            result.extend(chunk)
            self.position += len(chunk)
        return bytes(result)

    def readinto(self, buffer: Buffer) -> int:
        view = memoryview(buffer).cast("B")
        data = self.read(len(view))
        view[: len(data)] = data
        return len(data)

    def close(self) -> None:
        self.cache.clear()
        super().close()


class RasterOpener:
    def __init__(self, client: httpx.Client, url: str, budget: ReadBudget) -> None:
        self.client, self.url, self.budget = client, url, budget
        self.files: list[RasterHttpFile] = []

    def __call__(self, path: str, mode: str = "r") -> RasterHttpFile:
        if path != self.url or mode not in ("r", "rb"):
            raise OSError("Only the registered raster can be opened, in read-only mode.")
        file = RasterHttpFile(self.client, self.url, self.budget)
        if self.files:
            previous = self.files[0]
            if (file.size, file.etag, file.last_modified) != (
                previous.size,
                previous.etag,
                previous.last_modified,
            ):
                file.close()
                raise OSError(
                    "The source raster changed between reads. Retry against a stable version."
                )
        self.files.append(file)
        return file
