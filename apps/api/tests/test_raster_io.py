from __future__ import annotations

import io
import re

import httpx
import pytest

from app.analysis.raster_io import BLOCK_SIZE, RasterHttpFile, RasterOpener, ReadBudget

URL = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/tile/tile.tif"
DATA = bytes(range(256)) * 1000


def response(
    request: httpx.Request, *, etag: str = '"version-1"', data: bytes = DATA
) -> httpx.Response:
    match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"])
    assert match is not None
    start, end = int(match[1]), min(int(match[2]), len(data) - 1)
    return httpx.Response(
        206,
        headers={"Content-Range": f"bytes {start}-{end}/{len(data)}", "ETag": etag},
        stream=httpx.ByteStream(data[start : end + 1]),
    )


def test_seek_range_cache_and_readinto_are_bounded() -> None:
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        budget = ReadBudget()
        with RasterHttpFile(client, URL, budget) as file:
            assert file.read(8) == DATA[:8]
            requests = budget.requests
            file.seek(2)
            assert file.read(4) == DATA[2:6] and budget.requests == requests
            file.seek(BLOCK_SIZE - 2)
            target = bytearray(6)
            assert file.readinto(target) == 6
            assert bytes(target) == DATA[BLOCK_SIZE - 2 : BLOCK_SIZE + 4]
            file.seek(-3, io.SEEK_END)
            assert file.read(20) == DATA[-3:]
            assert file.read() == b""
            assert budget.downloaded_bytes < len(DATA)
        assert file.closed and not file.cache


def test_changed_versions_and_non_range_responses_are_rejected() -> None:
    calls = 0

    def changing(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response(request, etag=f'"version-{calls}"')

    with (
        httpx.Client(transport=httpx.MockTransport(changing)) as client,
        RasterHttpFile(client, URL, ReadBudget()) as file,
        pytest.raises(OSError, match="changed"),
    ):
        file.read(10)
    with httpx.Client(transport=httpx.MockTransport(changing)) as client:
        opener = RasterOpener(client, URL, ReadBudget())
        opener(URL).close()
        with pytest.raises(OSError, match="changed between reads"):
            opener(URL)
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=DATA))
        ) as client,
        pytest.raises(OSError, match="byte ranges"),
    ):
        RasterHttpFile(client, URL, ReadBudget())


def test_unapproved_urls_auxiliary_files_and_budget_excess_are_rejected() -> None:
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        for url in ["file:///etc/passwd", "https://127.0.0.1/tile.tif", URL + "?redirect=1"]:
            with pytest.raises(OSError, match="approved"):
                RasterHttpFile(client, url, ReadBudget())
        opener = RasterOpener(client, URL, ReadBudget())
        with pytest.raises(OSError, match="registered"):
            opener(URL + ".aux.xml")
        with pytest.raises(OSError, match="registered"):
            opener(URL, "w")
        with (
            RasterHttpFile(client, URL, ReadBudget(max_bytes=1024)) as file,
            pytest.raises(OSError, match="byte budget"),
        ):
            file.read(8)
