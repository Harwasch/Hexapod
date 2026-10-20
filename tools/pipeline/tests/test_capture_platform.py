"""`capture_platform.detect`: the platform table, one row per rule (and the order of them).

Pure: records and video metadata are built by hand in the shapes `frame_meta` and
`video.parse_probe` produce. The same table is printed by `experiments/ingest_demo.py`.
"""

from __future__ import annotations

import pytest

import capture_platform
from frame_meta import DjiXmp, FrameRecord
from video import VideoMeta


def _dji(flag: int | None, *, flight: bool = True) -> FrameRecord:
    dji = DjiXmp(
        rtk_flag=flag,
        relative_altitude_m=60.0 if flight else None,
        flight_yaw_deg=-12.0 if flight else None,
    )
    return FrameRecord(name="f.jpg", original="DJI.JPG", make="DJI", model="M3E", dji=dji)


def _photo(make: str | None) -> FrameRecord:
    return FrameRecord(name="f.jpg", original="IMG.JPG", make=make)


APPLE_VIDEO = VideoMeta(
    make="Apple",
    tags={"com.apple.quicktime.make": "Apple", "com.apple.quicktime.model": "iPhone 15 Pro"},
)
ANDROID_VIDEO = VideoMeta(tags={"com.android.version": "14", "com.android.manufacturer": "Google"})
DJI_VIDEO = VideoMeta(make="DJI", tags={"encoder": "DJI AVC encoder"})
BARE_VIDEO = VideoMeta(tags={"encoder": "Lavf61.1.100"})

#: (case, fmt, records, video, requested, hint, value, source, evidence fragment)
Row = tuple[str, str, list[FrameRecord], VideoMeta | None, str | None, str | None, str, str, str]
TABLE: list[Row] = [
    ("rtk fixed throughout", "images", [_dji(50)] * 40, None, None, None,
     "rtk-drone", "dji-rtk-fixed", "40/40 frames RtkFlag 50"),
    ("rtk mostly single", "images", [_dji(50)] * 10 + [_dji(16)] * 30, None, None, None,
     "drone", "drone-xmp", "10/40 fixed"),
    ("half fixed is rtk", "images", [_dji(50)] * 20 + [_dji(34)] * 20, None, None, None,
     "rtk-drone", "dji-rtk-fixed", "20/40"),
    ("mavic 2: no rtk fields", "images", [_dji(None)] * 12, None, None, None,
     "drone", "drone-xmp", "12/12 frames carry DJI flight XMP"),
    ("dji xmp without flight fields", "images", [_dji(None, flight=False)] * 4, None, None, None,
     "unknown", "none", "no rule"),
    ("apple photos", "images", [_photo("Apple")] * 5, None, None, None,
     "phone-photo", "phone-make", "5/5 frames by Apple"),
    ("samsung photos", "images", [_photo("samsung")] * 3, None, None, None,
     "phone-photo", "phone-make", "samsung"),
    ("quicktime apple video", "video", [], APPLE_VIDEO, None, None,
     "phone-video", "phone-quicktime", "QuickTime keys"),
    ("android video", "video", [], ANDROID_VIDEO, None, None,
     "phone-video", "phone-android", "com.android.version"),
    ("dji-make video", "video", [], DJI_VIDEO, None, None,
     "drone", "drone-make", "UNVERIFIED"),
    ("bare video, phone key", "video", [], BARE_VIDEO, None, "phone",
     "phone-video", "phone-key", "phone key"),
    ("bare images, phone key", "images", [_photo(None)] * 3, None, None, "phone",
     "phone-photo", "phone-key", "phone key"),
    ("canon photos", "images", [_photo("Canon")] * 6, None, None, None,
     "unknown", "none", "no rule"),
    ("requested over apple", "images", [_photo("Apple")] * 5, None, "rtk-drone", None,
     "rtk-drone", "param", "asked for"),
    ("drone frames beat a phone key", "images", [_dji(50)] * 3, None, None, "phone",
     "rtk-drone", "dji-rtk-fixed", "3/3"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("case", "fmt", "records", "video", "requested", "hint", "value", "source", "evidence"),
    TABLE,
    ids=[row[0] for row in TABLE],
)
def test_the_platform_table(
    case: str,
    fmt: str,
    records: list[FrameRecord],
    video: VideoMeta | None,
    requested: str | None,
    hint: str | None,
    value: str,
    source: str,
    evidence: str,
) -> None:
    platform = capture_platform.detect(
        fmt=fmt, records=records, video=video, requested=requested, hint=hint
    )

    assert (platform.value, platform.source) == (value, source), case
    assert evidence in platform.evidence
    assert platform.value in capture_platform.PLATFORMS


def test_a_platform_not_in_the_vocabulary_is_refused() -> None:
    with pytest.raises(ValueError, match="rtk-drone, drone, phone-video, phone-photo, unknown"):
        capture_platform.detect(fmt="images", records=[], requested="satellite")


def test_an_empty_capture_is_unknown_not_an_error() -> None:
    platform = capture_platform.detect(fmt="images", records=[])

    assert platform.to_dict() == {
        "value": "unknown",
        "source": "none",
        "evidence": "no rule matched what the frames say",
    }
