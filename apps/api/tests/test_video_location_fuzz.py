"""Adversarial and fuzz tests of `services/video_location.py`, from the cross-model review
of #72 (seed 720013, 20,000 mutations): bit flips, insertions, deletions, truncation,
corrupt sizes and types, nesting, short reads and lying HEAD sizes. Every case must end
within a second, inside the read and byte budgets, with no exception and only points on
the globe. Random cases have a one-second SIGALRM deadline.
"""

from __future__ import annotations

import json
import math
import random
import signal
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.services import video_location as v
from tests.mov_fixtures import (
    ACCURACY_KEY,
    ISO6709_KEY,
    UTF8,
    box,
    large_box,
    loci_atom,
    mdta_meta,
    mov,
    mp4_with_loci,
    mp4_with_xyz,
    xyz_atom,
)

SF = "+37.7749-122.4194+010.000/"


class DeadlineExceeded(Exception):  # noqa: N818 - a deadline, as `_Exhausted` is a budget
    pass


@contextmanager
def deadline() -> Iterator[None]:
    def alarm(*_: object) -> None:
        raise DeadlineExceeded("case exceeded one second")

    old = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class CountingStorage:
    def __init__(self, data: bytes, short: int | None = None, size: int | None = None) -> None:
        self.data = data
        self.short = short
        self.size = len(data) if size is None else size
        self.served = 0
        self.requested = 0
        self.reads: list[tuple[int, int]] = []

    def head_object(self, key: str) -> Any:
        return SimpleNamespace(size=self.size)

    def get_object_range(self, key: str, start: int, length: int) -> bytes:
        assert start >= 0 and length > 0
        self.requested += length
        count = length if self.short is None else min(length, self.short)
        result = self.data[start : start + count]
        self.served += len(result)
        self.reads.append((start, len(result)))
        return result


def checked(
    data: bytes, short: int | None = None, size: int | None = None
) -> tuple[v.VideoLocation | None, CountingStorage]:
    storage = CountingStorage(data, short, size)
    with deadline():
        found = v.location_in_storage(storage, "clip.mov")  # type: ignore[arg-type]
    assert storage.served <= v.MAX_READ_BYTES, ("served cap", storage.served)
    assert storage.requested <= v.MAX_READ_BYTES, ("requested cap", storage.requested)
    assert len(storage.reads) <= v.MAX_READS
    if found:
        assert math.isfinite(found.lat) and -90 <= found.lat <= 90
        assert math.isfinite(found.lon) and -180 <= found.lon <= 180
        assert found.accuracy_m is None or (
            math.isfinite(found.accuracy_m) and found.accuracy_m > 0
        )
    return found, storage


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("+90+180/", (90.0, 180.0)),
        ("+90.0000+180.0000", (90.0, 180.0)),
        ("-90-180/", (-90.0, -180.0)),
        ("-0-0/", (-0.0, -0.0)),
        ("+37.7749-122.4194/", (37.7749, -122.4194)),
        ("+90.0001+180/", None),
        ("+90+180.0001/", None),
        ("90+180/", None),
        ("+90 180/", None),
        ("+1+2//", None),
        ("+1+2/junk", None),
        ("+1+2+3+4/", None),
        ("+nan+inf/", None),
        ("+1+2/\x00junk", None),
        ("++1+2/", None),
    ],
)
def test_iso_edges(text: str, expected: tuple[float, float] | None) -> None:
    found, _ = checked(mov(iso6709=text))
    assert (None if found is None else (found.lat, found.lon)) == expected
    if text == "-0-0/":
        assert found is not None
        assert math.copysign(1, found.lat) == -1 and math.copysign(1, found.lon) == -1


@pytest.mark.parametrize(
    "accuracy", ["1000000", "-1000000", "-1", "0", "nan", "inf", "-inf", "1e308"]
)
def test_accuracy_and_buffer(accuracy: str) -> None:
    from app.services.epochs import video_join_buffer_m

    found, _ = checked(mov(iso6709=SF, accuracy=accuracy))
    positive = math.isfinite(float(accuracy)) and float(accuracy) > 0
    assert found is not None
    assert found.accuracy_m == (float(accuracy) if positive else None)
    assert video_join_buffer_m(found.accuracy_m) == (50 if positive else 25)


@pytest.mark.parametrize("kind", [UTF8, 23, 24, 21, 22])
def test_accuracy_types(kind: int) -> None:
    raw = {1: b"5", 23: struct.pack(">f", 5), 24: struct.pack(">d", 5), 21: b"\x05", 22: b"\x05"}[
        kind
    ]
    found, _ = checked(
        box(b"moov", mdta_meta([(ISO6709_KEY, SF.encode(), UTF8), (ACCURACY_KEY, raw, kind)]))
    )
    assert found is not None and found.accuracy_m == 5


def structural_cases() -> list[bytes]:
    meta = mdta_meta([(ISO6709_KEY, SF.encode(), UTF8)])
    nested = meta
    for _ in range(5000):
        nested = box(b"moov", nested)
    return [
        b"",
        b"bad",
        struct.pack(">I4s", 0, b"moov") + meta,
        large_box(b"moov", meta),
        struct.pack(">I4sQ", 1, b"moov", 15),
        struct.pack(">I4sQ", 1, b"moov", 2**64 - 1),
        struct.pack(">I4s", 4, b"moov"),
        struct.pack(">I4s", 999999, b"moov"),
        struct.pack(">I4s", 1, b"moov") + b"\0" * 7,
        nested,
        box(b"moov", box(b"udta", struct.pack(">I4s", 0, b"udta"))),
        box(b"moov", box(b"meta", box(b"keys", b"\0" * 4 + b"\xff" * 4))),
        box(b"moov", box(b"meta", box(b"ilst", box(b"free") * 2000))),
        box(b"free") * 2000,
        box(b"moov", mdta_meta([(ISO6709_KEY, b"\xff\xfe", UTF8)])),
        box(b"moov", box(b"udta", box(v.XYZ, b"\0\x02\0\0\xff\xfe"))),
        box(b"moov", meta.replace(b"mdta", b"\xff\xff\xff\xff")),
        mov(iso6709="+" + "9" * 100000 + "+2/"),
        mov(iso6709="+" + "0" * 100000 + "+2/junk"),
        box(b"moov", box(b"udta", box(b"free", b"\0" * 900000)) * 6),
    ]


@pytest.mark.parametrize("index", range(20))
def test_adversarial_structure(index: int) -> None:
    checked(structural_cases()[index])


@pytest.mark.parametrize("short", [0, 1, 7, 8, 9, 15, 16, 31, 64, 256])
def test_short_ranges(short: int) -> None:
    checked(mov(iso6709=SF), short=short)
    checked(mov(iso6709=SF), short=short, size=2**63)


def test_api_rejects_video_location() -> None:
    from pydantic import ValidationError

    from app.schemas.capture import CaptureCreate

    for value in [None, {}, {"lat": 1, "lon": 2}, "fake"]:
        with pytest.raises(ValidationError):
            CaptureCreate(name="review", kind="video", metadata={"videoLocation": value})


def test_seeded_mutations_20000(tmp_path: Path) -> None:
    rng = random.Random(720013)
    seeds = [
        mov(iso6709=SF, accuracy="4.5"),
        mov(),
        mov(iso6709=SF, large_mdat=True),
        mov(iso6709=SF, moov_first=True, full_box_meta=True),
        mp4_with_xyz(SF),
        mp4_with_loci(37.7749, -122.4194),
        mov(iso6709="+90+180/", xyz="-1-2/"),
    ]
    seeds += structural_cases()[:17]
    stats: dict[str, Any] = {
        "seed": 720013,
        "cases": 0,
        "failures": {},
        "examples": [],
        "max_served": 0,
        "max_reads": 0,
        "locations": 0,
    }
    for i in range(20000):
        data = bytearray(rng.choice(seeds))
        for _ in range(rng.randrange(1, 7)):
            pos = rng.randrange(len(data) + 1)
            op = rng.randrange(7)
            if op == 0 and pos < len(data):
                data[pos] ^= rng.randrange(1, 256)
            elif op == 1:
                data[pos : pos + rng.randrange(1, 17)] = rng.randbytes(rng.randrange(17))
            elif op == 2:
                del data[pos : pos + rng.randrange(1, 65)]
            elif op == 3:
                data[pos:pos] = struct.pack(">I", rng.choice([0, 1, 7, 8, 15, 16, 2**32 - 1]))
            elif op == 4:
                del data[pos:]
            elif op == 5:
                data[pos:pos] = rng.choice(
                    [b"moov", b"meta", b"udta", b"keys", b"ilst", b"data", b"\xff" * 16]
                )
            else:
                data = bytearray(box(rng.choice([b"moov", b"udta", b"meta"]), bytes(data)))
        short = rng.choice([None] * 8 + [0, 7, 8, 15, 16, 64, 256])
        size = len(data) + rng.choice([0] * 8 + [1, 100000, 2**32])
        try:
            found, storage = checked(bytes(data), short=short, size=size)
            stats["max_served"] = max(stats["max_served"], storage.served)
            stats["max_reads"] = max(stats["max_reads"], len(storage.reads))
            stats["locations"] += found is not None
        except Exception as exc:
            kind = type(exc).__name__
            stats["failures"][kind] = stats["failures"].get(kind, 0) + 1
            if len(stats["examples"]) < 20:
                path = tmp_path / f"failure_{i}.bin"
                path.write_bytes(data)
                stats["examples"].append(
                    {"case": i, "kind": kind, "message": str(exc), "short": short, "size": size}
                )
        stats["cases"] += 1
    (tmp_path / "mutation_results.json").write_text(json.dumps(stats, indent=2) + "\n")
    assert not stats["failures"], stats


def test_no_mdat_payload_is_fetched() -> None:
    data = mov(iso6709=SF, mdat_bytes=200000)
    start = data.index(b"mdat") - 4
    end = start + struct.unpack(">I", data[start : start + 4])[0]
    _, storage = checked(data)
    overlap = sum(max(0, min(off + n, end) - max(off, start + 8)) for off, n in storage.reads)
    assert overlap == 0, f"{overlap} mdat payload bytes fetched: {storage.reads}"


def test_declared_xyz_text_must_not_be_truncated() -> None:
    data = box(b"moov", box(b"udta", box(v.XYZ, struct.pack(">HH", 65535, 0) + b"+1+2/")))
    found, _ = checked(data)
    assert found is None


@pytest.mark.parametrize("text", ["+0030.0+00030.0/", "+000030.0+0000030.0/"])
def test_sexagesimal_is_converted_or_rejected_never_read_as_decimal(text: str) -> None:
    found, _ = checked(mov(iso6709=text))
    expected = 0.5 if text == "+0030.0+00030.0/" else 30 / 3600
    assert found is None or (found.lat == expected and found.lon == expected)


def test_conflicting_locations_are_no_location() -> None:
    """The cross review's precedence cases, under the ruling that followed it: points that
    disagree past their accuracy are no location, in whichever order the atoms come."""
    meta = mdta_meta([(ISO6709_KEY, b"+1+2/", UTF8)])
    for fallback in [xyz_atom("+3+4/") + loci_atom(5, 6), loci_atom(5, 6) + xyz_atom("+3+4/")]:
        for data in [
            box(b"moov", meta + box(b"udta", fallback)),
            box(b"moov", box(b"udta", fallback) + meta),
            box(b"moov", box(b"udta", fallback)),
        ]:
            found, _ = checked(data)
            assert found is None


@pytest.mark.parametrize(
    "variant", ["xyz_then_loci", "loci_then_xyz", "duplicate_meta", "duplicate_agreeing"]
)
def test_the_reader_never_vouches_for_a_point_ffmpeg_reads_otherwise(
    variant: str, tmp_path: Path
) -> None:
    """The cross review's ffmpeg comparison: on conflicting atoms ffmpeg 7.0.2 (and so the
    pipeline's own scrape) reads the last. The pipeline is now handed the worker's reading
    instead (`worker/params.VIDEO_LOCATION_PARAM`), and this holds the reader to never
    vouching for a point more than the agreement tolerance from what ffmpeg would read."""
    import subprocess
    from importlib import import_module

    import imageio_ffmpeg

    import app.worker.pipeline_bridge  # noqa: F401 - puts tools/pipeline on the path

    video = import_module("video")
    if variant == "duplicate_meta":
        children = mdta_meta([(ISO6709_KEY, b"+1+2/", UTF8)]) + mdta_meta(
            [(ISO6709_KEY, b"+3+4/", UTF8)]
        )
    elif variant == "duplicate_agreeing":
        children = mdta_meta([(ISO6709_KEY, SF.encode(), UTF8)]) + box(b"udta", xyz_atom(SF))
    else:
        atoms = [xyz_atom("+3+4/"), loci_atom(5, 6)]
        if variant == "loci_then_xyz":
            atoms.reverse()
        children = box(b"udta", b"".join(atoms))
    data = box(b"ftyp", b"qt  " + b"\0" * 4 + b"qt  ") + box(b"moov", children)
    path = tmp_path / f"{variant}.mov"
    path.write_bytes(data)
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    pipeline = video.parse_probe(result.stderr).location
    reader, _ = checked(data)
    assert pipeline is not None
    if variant == "duplicate_agreeing":
        assert reader is not None and (reader.lat, reader.lon) == (pipeline.lat, pipeline.lon)
    else:
        assert reader is None
