"""`services/video_location.py`: the location a phone wrote into its video's container.

Read from the box tree alone -- `moov/meta` (the iOS camera's `mdta` keys) and
`moov/udta/(c)xyz` (everything else) -- through ranged reads, so a 12 GB clip costs a few
kilobytes and its `mdat` is never touched. The fixtures are synthetic boxes
(`tests/mov_fixtures.py`), not committed video.
"""

from __future__ import annotations

import io
import struct
from types import SimpleNamespace
from typing import Any

import pytest

from app.services import video_location
from app.storage import S3Storage
from app.storage.base import RangeNotHonouredError
from tests.mov_fixtures import FLOAT32, FLOAT64, box, mov, mp4_with_loci, mp4_with_xyz

SF = "+37.7749-122.4194+010.000/"


def read(data: bytes) -> video_location.VideoLocation | None:
    return video_location.location_of_bytes(data)


def test_an_iphone_clip_with_its_moov_at_the_end() -> None:
    found = read(mov(iso6709=SF, accuracy="4.723285"))

    assert found == video_location.VideoLocation(
        lat=37.7749, lon=-122.4194, accuracy_m=4.723285, key="iso6709"
    )
    assert found.to_metadata() == {
        "lat": 37.7749,
        "lon": -122.4194,
        "accuracyM": 4.723285,
        "key": "iso6709",
    }


def test_moov_before_mdat_and_a_64_bit_mdat() -> None:
    assert read(mov(iso6709=SF, moov_first=True)) is not None
    found = read(mov(iso6709=SF, accuracy="12", large_mdat=True))
    assert found is not None and found.accuracy_m == 12.0


def test_a_meta_written_as_a_full_box_is_read_too() -> None:
    found = read(mov(iso6709=SF, accuracy="3", full_box_meta=True))
    assert found is not None and (found.lat, found.accuracy_m) == (37.7749, 3.0)


def test_the_xyz_atom_alone_as_an_android_phone_writes_it() -> None:
    found = read(mp4_with_xyz("+51.5007-000.1246/"))
    assert found == video_location.VideoLocation(
        lat=51.5007, lon=-0.1246, accuracy_m=None, key="xyz"
    )


def test_the_3gpp_loci_box_as_ffmpeg_writes_an_mp4s_location() -> None:
    """16.16 fixed point: good to 1/65536 of a degree, under two metres."""
    found = read(mp4_with_loci(51.5007, -0.1246))
    assert found is not None and found.key == "loci" and found.accuracy_m is None
    assert abs(found.lat - 51.5007) < 2e-5 and abs(found.lon + 0.1246) < 2e-5


def test_the_mdta_key_wins_over_the_xyz_atom() -> None:
    found = read(mov(iso6709=SF, xyz="+51.5007-000.1246/"))
    assert found is not None and found.key == "iso6709" and found.lat == 37.7749


def test_a_clip_without_location_says_none() -> None:
    """iOS strips both atoms when a video is shared with location off, or picked through
    some Safari and Photos sharing paths: the clip still plays, and says nowhere."""
    assert read(mov()) is None
    assert read(mov(accuracy="5")) is None  # an accuracy is not a location


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x1a\x45\xdf\xa3" + b"\0" * 64,  # Matroska / WebM: not a QuickTime container
        b"not a video at all, but long enough to look at",
        box(b"ftyp", b"qt  ") + struct.pack(">I", 5) + b"moov",  # a size under its header
        box(b"ftyp", b"qt  ") + struct.pack(">I", 9999) + b"moov" + b"\0" * 10,  # truncated
    ],
)
def test_anything_that_is_not_a_readable_container_says_none(data: bytes) -> None:
    assert read(data) is None


@pytest.mark.parametrize(
    "text", ["+95.0000-122.4194/", "+37.7749-190.0000/", "37.7749,-122.4194", "+nan+nan/", ""]
)
def test_a_location_that_is_not_a_point_on_the_globe_is_none(text: str) -> None:
    assert read(mov(iso6709=text)) is None


@pytest.mark.parametrize(
    ("accuracy", "kind", "expected"),
    [
        ("4.5", 1, 4.5),
        (6.25, FLOAT32, 6.25),
        (7.5, FLOAT64, 7.5),
        ("0", 1, None),
        ("-3", 1, None),
        ("inf", 1, None),
        ("about five metres", 1, None),
    ],
)
def test_the_horizontal_accuracy_is_a_positive_number_or_none(
    accuracy: str | float, kind: int, expected: float | None
) -> None:
    found = read(mov(iso6709=SF, accuracy=accuracy, accuracy_type=kind))
    assert found is not None and found.accuracy_m == expected


def test_only_box_headers_and_the_metadata_boxes_are_read() -> None:
    """A 12 GB clip is a few small reads: the `mdat` and `trak` payloads are skipped by
    their sizes, never fetched."""
    data = mov(iso6709=SF, accuracy="5", mdat_bytes=200_000, trak_bytes=100_000)
    mdat_at = data.index(b"mdat") - 4
    mdat_end = mdat_at + struct.unpack(">I", data[mdat_at : mdat_at + 4])[0]
    reads: list[tuple[int, int]] = []

    def read_at(offset: int, length: int) -> bytes:
        reads.append((offset, length))
        return data[offset : offset + length]

    found = video_location.read_location(read_at, len(data), chunk=4096)

    assert found is not None and found.lat == 37.7749
    assert len(reads) <= 8, reads
    assert sum(length for _, length in reads) <= 8 * 4096
    # Not one byte of the mdat's payload was asked for (the cross review of #72: the
    # first read ran 64 KiB ahead of the top-level headers, into the mdat).
    overlap = sum(max(0, min(off + n, mdat_end) - max(off, mdat_at + 8)) for off, n in reads)
    assert overlap == 0, reads


def test_read_from_the_bucket_by_ranged_gets(storage: S3Storage) -> None:
    storage.put_object("captures/c/clip.MOV", mov(iso6709=SF, accuracy="5"), "video/quicktime")
    storage.put_object("captures/c/bare.MOV", mov(), "video/quicktime")

    found = video_location.location_in_storage(storage, "captures/c/clip.MOV")

    assert found is not None and (found.lat, found.lon, found.accuracy_m) == (
        37.7749,
        -122.4194,
        5.0,
    )
    assert video_location.location_in_storage(storage, "captures/c/bare.MOV") is None
    assert storage.get_object_range("captures/c/bare.MOV", 4, 4) == b"ftyp"


@pytest.mark.parametrize(
    "text",
    [SF, "+51.5007-000.1246/", "-33.8688+151.2093", "+95.0-122.0/", "37.7,-122.4", "", "+1+2+3+4"],
)
def test_the_api_and_the_pipeline_read_the_same_places_the_same_way(text: str) -> None:
    """The join (here) and the georeference (`tools/pipeline/video.py`) must agree on what
    a clip says, or a capture is matched by one point and placed by another."""
    from importlib import import_module

    import app.worker.pipeline_bridge  # noqa: F401 - puts tools/pipeline on the path

    video = import_module("video")
    assert video_location.VIDEO_SUFFIXES == video.VIDEO_SUFFIXES
    assert video_location.ACCURACY_KEY == video.ACCURACY_KEY
    assert video.LOCATION_KEYS[0] == video_location.ISO6709_KEY
    theirs = video.parse_iso6709(text)
    ours = read(mov(iso6709=text))
    on_globe = theirs is not None and -90 <= theirs[0] <= 90 and -180 <= theirs[1] <= 180
    assert (None if not on_globe else theirs[:2]) == (
        None if ours is None else (ours.lat, ours.lon)
    )


def test_a_crafted_moov_of_many_large_metadata_boxes_costs_a_bounded_read() -> None:
    """Each `udta` is under the 1 MiB a metadata box may be, and there are many: the bytes
    fetched are capped as well as the reads, so the file costs what an honest one could."""
    filler = box(b"udta", box(b"free", b"\0" * 900_000))
    data = box(b"ftyp", b"qt  ") + box(b"moov", filler * 40)
    fetched: list[int] = []

    def read_at(offset: int, length: int) -> bytes:
        fetched.append(length)
        return data[offset : offset + length]

    assert video_location.read_location(read_at, len(data)) is None
    assert sum(fetched) <= video_location.MAX_READ_BYTES


class _Body(io.BytesIO):
    """A response body that counts what was taken out of it."""

    served = 0

    def read(self, size: int | None = -1) -> bytes:
        data = super().read(size)
        self.served += len(data)
        return data


def _offline_s3(response: dict[str, Any]) -> S3Storage:
    storage = object.__new__(S3Storage)
    storage._bucket = "offline"
    storage._client = SimpleNamespace(get_object=lambda **_: response)  # type: ignore[assignment]
    return storage


def test_a_server_that_ignores_the_range_is_refused_after_a_bounded_read() -> None:
    """From the cross review: a 64 KiB read of an 8 MiB object whose server ignored Range
    pulled all 8 MiB through `Body.read()` before anything was checked. The body is now
    read no further than the bytes asked for, and an answer that is not the range asked
    for is refused, whatever the server did."""
    body = _Body(mov(iso6709=SF, mdat_bytes=8 * 1024 * 1024))

    with pytest.raises(RangeNotHonouredError):
        _offline_s3({"Body": body}).get_object_range("clip.mov", 0, 65536)

    assert body.served <= 65536 + 1
    assert body.closed


def test_a_range_answered_from_elsewhere_is_refused() -> None:
    body = _Body(b"\0" * 64)
    response = {"Body": body, "ContentRange": "bytes 0-63/1000"}

    with pytest.raises(RangeNotHonouredError):
        _offline_s3(response).get_object_range("clip.mov", 100, 64)

    honest = {"Body": _Body(b"x" * 64), "ContentRange": "bytes 100-163/1000"}
    assert _offline_s3(honest).get_object_range("clip.mov", 100, 64) == b"x" * 64
    # A body longer than its range is cut at the range, never read on.
    longer = _Body(b"y" * 10_000)
    assert (
        _offline_s3({"Body": longer, "ContentRange": "bytes 0-63/10000"}).get_object_range(
            "clip.mov", 0, 64
        )
        == b"y" * 64
    )
    assert longer.served <= 65
