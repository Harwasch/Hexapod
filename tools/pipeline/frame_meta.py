"""What each kept frame's original said about itself, read before it was shrunk.

`normalize` shrinks a full-size photo to the frame size training reads (`video._shrink`),
keeping only the EXIF block. Everything else a camera wrote -- the DJI XMP packet with its
gimbal, flight and RTK fields, above all -- is gone from the frame by the time `pose` or
`georeference` sees it. So this module reads the **original**, once, into one record per
kept frame (`frames_meta.json`), keyed by the renumbered frame name.

What it reads, and how far each field can be trusted:

* **EXIF** (Pillow's reader; tag numbers spelled out below so the set is visible): capture
  time with its offset, exposure, focal length and its 35 mm equivalent, make, model and
  lens, and the GPS fix with its horizontal error (`GPSHPositioningError`, tag 31).
  **Serial numbers are never read** (`BodySerialNumber`, DJI's `CameraSerialNumber` and
  `DroneSerialNumber`): they identify a person's device, and nothing here needs them.
* **DJI XMP** (`drone-dji:` properties), parsed as text the way every DJI metadata tool
  does: the packet is ASCII inside an APP1 segment. Field names are checked against
  ExifTool's DJI tag table (https://exiftool.org/TagNames/DJI.html) except those in
  `UNVERIFIED_DJI_FIELDS`, which come from forum posts and vendor articles only; the RTK
  flag codes come from Sitemark's article; the datum of `AbsoluteAltitude` is a forum
  claim. **No real Mavic 3 Enterprise file has been read by this code** -- the test
  fixtures are synthetic. Each record lists the unverified fields it actually carried.

The altitude datum of an EXIF fix is written in the site frame's height-datum vocabulary
(`site_frame.HEIGHT_DATUMS`): `msl-unverified`, with the words in `altitudeNote`. A later
stage that compares a fix's datum with its frame's (M3b's pose priors) can then compare
two equal strings rather than parse prose.

Stdlib and PIL only at module scope, so the Modal image needs nothing new.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import PIL.Image

import exif
import site_frame

__all__ = [
    "DJI_ALTITUDE_DATUM",
    "EXPOSURE_RANGES",
    "FOCAL_FIELD_OF_VIEW_DEG",
    "FOCAL_SPREAD_LIMIT",
    "GPS_ALTITUDE_DATUM",
    "META_BYTES",
    "RTK_FIXED",
    "RTK_FLAG_MEANING",
    "UNVERIFIED_DJI_FIELDS",
    "DjiXmp",
    "Exposure",
    "Focal",
    "FrameRecord",
    "FrameTime",
    "calibrated_focal_px",
    "capture_focal",
    "ev100",
    "exif_time",
    "focal_px_from_35mm",
    "implausible_exposure",
    "implausible_focal",
    "offset_text",
    "read_dji_xmp",
    "read_photo",
    "rtk_flag_meaning",
    "utc_text",
    "xmp_packet",
    "xmp_value",
]

#: The EXIF sub-IFDs.
IFD_EXIF = 0x8769
IFD_GPS = 0x8825
#: IFD0 tags.
MAKE = 0x010F
MODEL = 0x0110
#: Exif IFD tags.
EXPOSURE_TIME = 0x829A
F_NUMBER = 0x829D
ISO = 0x8827
DATETIME_ORIGINAL = 0x9003
OFFSET_ORIGINAL = 0x9011
EXPOSURE_BIAS = 0x9204
FOCAL_LENGTH = 0x920A
SUBSEC_ORIGINAL = 0x9291
FOCAL_35MM = 0xA405
LENS_MODEL = 0xA434
#: GPS IFD tags.
GPS_TIME = 7
GPS_DATE = 29
GPS_H_ERROR = 31

#: How much of the start of a JPEG to read for its XMP when Pillow did not surface it.
#: The DJI packet sits after the EXIF block and a thumbnail; in the Minnetonka set it
#: starts as late as ~49 KiB.
META_BYTES = 192 * 1024

#: An EXIF `GPSAltitude` is "metres above sea level" by the standard, so above whichever
#: geoid the receiver applied -- not the WGS84 ellipsoid. In the site frame's vocabulary.
GPS_ALTITUDE_DATUM = "msl-unverified"
GPS_ALTITUDE_NOTE = (
    "EXIF GPSAltitude: metres above mean sea level by the EXIF standard, against whatever "
    "geoid the receiver applied (unknown); not the WGS84 ellipsoid"
)
if GPS_ALTITUDE_DATUM not in site_frame.HEIGHT_DATUMS:  # pragma: no cover - import-time guard
    raise RuntimeError(f"{GPS_ALTITUDE_DATUM!r} is not one of {site_frame.HEIGHT_DATUMS}")

#: The datum of DJI's `AbsoluteAltitude`, which no primary source states: deliberately
#: outside `site_frame.HEIGHT_DATUMS`, so it never compares equal to a frame's datum.
DJI_ALTITUDE_DATUM = "unverified"
DJI_ALTITUDE_NOTE = (
    "DJI AbsoluteAltitude: forum reports say ellipsoidal when RTK-fixed / XMP >= 1.5 and "
    "above mean sea level otherwise; not confirmed by DJI or ExifTool"
)
if DJI_ALTITUDE_DATUM in site_frame.HEIGHT_DATUMS:  # pragma: no cover - import-time guard
    raise RuntimeError("the DJI altitude datum must not pass for a verified one")

#: DJI XMP fields whose spelling or meaning no primary source confirms: `RtkDiffAge`
#: (inferred from Sitemark's `rtk_diff_age`), `GpsStatus` and `AltitudeType` (an Emlid
#: forum sample from a P1/M300), `UTCAtExposure` (unconfirmed). Read when present and
#: named in every record that carried one, so nothing downstream mistakes them for fact.
UNVERIFIED_DJI_FIELDS: tuple[str, ...] = (
    "RtkDiffAge",
    "GpsStatus",
    "AltitudeType",
    "UTCAtExposure",
)

#: DJI's RTK solution codes, from Sitemark's "How to read DJI RTK/NTRIP metadata"; 34-49
#: are all a float solution. Not in ExifTool's table.
RTK_FLAG_MEANING: dict[int, str] = {0: "none", 16: "single", 34: "float", 50: "fixed"}
#: The flag of a fixed RTK solution: centimetre-level, the survey tier.
RTK_FIXED = 50

#: Above this relative spread of per-frame focal lengths, the frames are not one camera
#: and no single focal seeds pose. First principles, not a measurement: the smallest step
#: an EXIF 35 mm value can take is 1 mm, which at 24 mm is ~4%; a DJI calibrated focal is
#: a per-unit constant. Any spread above float noise therefore means mixed lenses or zoom
#: settings, and 0.5% sits far below the smallest real step and far above rounding.
FOCAL_SPREAD_LIMIT = 0.005

_XMP_START = b"<x:xmpmeta"
_XMP_END = b"</x:xmpmeta>"

TimeSource = Literal[
    "exif-offset",
    "gps-time",
    "exif-local",
    "xmp-createdate",
    "xmp-createdate-local",
    "quicktime-creationdate",
    "mvhd-creation-time",
]
FocalFrom = Literal["dji-calibrated", "exif-35mm", "quicktime-35mm"]


# --- XMP -----------------------------------------------------------------------------


def xmp_packet(data: bytes) -> str | None:
    """The `<x:xmpmeta ...</x:xmpmeta>` text of a file's head, or None if it has none.

    An unterminated packet (the head was cut short) runs to the end of what was read.
    """
    start = data.find(_XMP_START)
    if start < 0:
        return None
    end = data.find(_XMP_END, start + 1)
    stop = end + len(_XMP_END) if end > 0 else len(data)
    return data[start:stop].decode("utf-8", errors="replace")


def xmp_value(text: str, prefix: str, name: str) -> str | None:
    """`prefix:name="v"` (attribute form, what the drone writes) or
    `<prefix:name>v</prefix:name>` (element form, what re-saved files carry).

    Case-insensitive, as ExifTool matches DJI's property names: firmware has spelled some
    of them in more than one case. Single-quoted attributes are valid XML and accepted.
    The name must end where the match does, so `RtkStdLon` never reads `RtkStdLonX`.
    """
    p, n = re.escape(prefix), re.escape(name)
    found = (
        re.search(rf"(?<![\w-]){p}:{n}\s*=\s*\"([^\"]*)\"", text, re.I)
        or re.search(rf"(?<![\w-]){p}:{n}\s*=\s*'([^']*)'", text, re.I)
        or re.search(rf"<{p}:{n}(?:\s[^<>]*)?>([^<]*)</{p}:{n}\s*>", text, re.I)
    )
    return found.group(1).strip() if found else None


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _integer(value: str | None) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number == int(number) else None


def rtk_flag_meaning(flag: int | None) -> str | None:
    """`RtkFlag` in words (Sitemark's codes), `unknown` for a code none of them is."""
    if flag is None:
        return None
    if flag in RTK_FLAG_MEANING:
        return RTK_FLAG_MEANING[flag]
    return "float" if 34 <= flag <= 49 else "unknown"


@dataclass(frozen=True)
class DjiXmp:
    """The `drone-dji:` properties of one photo. Every field optional."""

    relative_altitude_m: float | None = None
    absolute_altitude_m: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    gimbal_pitch_deg: float | None = None
    gimbal_yaw_deg: float | None = None
    gimbal_roll_deg: float | None = None
    flight_pitch_deg: float | None = None
    flight_yaw_deg: float | None = None
    flight_roll_deg: float | None = None
    rtk_flag: int | None = None
    rtk_std_lon_m: float | None = None
    rtk_std_lat_m: float | None = None
    rtk_std_hgt_m: float | None = None
    rtk_diff_age_s: float | None = None
    gps_status: str | None = None
    altitude_type: str | None = None
    calibrated_focal_px: float | None = None
    calibrated_center_px: tuple[float, float] | None = None
    dewarp_flag: int | None = None
    create_date: str | None = None
    utc_at_exposure: str | None = None
    #: Which of `UNVERIFIED_DJI_FIELDS` this photo actually carried.
    unverified_fields: tuple[str, ...] = ()

    @property
    def is_flight(self) -> bool:
        """Says it flew: a height above take-off or an airframe heading."""
        return self.relative_altitude_m is not None or self.flight_yaw_deg is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "relativeAltitudeM": self.relative_altitude_m,
            "absoluteAltitudeM": self.absolute_altitude_m,
            # Not one of the site frame's datums: nobody has confirmed which this is, so
            # a stage comparing datums sees a mismatch and leaves the vertical free.
            "absoluteAltitudeDatum": DJI_ALTITUDE_DATUM,
            "absoluteAltitudeNote": DJI_ALTITUDE_NOTE,
            "altitudeType": self.altitude_type,
            "gpsStatus": self.gps_status,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "gimbal": {
                "pitchDeg": self.gimbal_pitch_deg,
                "yawDeg": self.gimbal_yaw_deg,
                "rollDeg": self.gimbal_roll_deg,
            },
            "flight": {
                "pitchDeg": self.flight_pitch_deg,
                "yawDeg": self.flight_yaw_deg,
                "rollDeg": self.flight_roll_deg,
            },
            "rtk": {
                "flag": self.rtk_flag,
                "flagMeaning": rtk_flag_meaning(self.rtk_flag),
                "stdLonM": self.rtk_std_lon_m,
                "stdLatM": self.rtk_std_lat_m,
                "stdHgtM": self.rtk_std_hgt_m,
                "diffAgeS": self.rtk_diff_age_s,
            },
            "calibratedFocalLengthPx": self.calibrated_focal_px,
            "calibratedOpticalCenterPx": (
                None if self.calibrated_center_px is None else list(self.calibrated_center_px)
            ),
            "dewarpFlag": self.dewarp_flag,
            "createDate": self.create_date,
            "utcAtExposure": self.utc_at_exposure,
            "unverifiedFields": list(self.unverified_fields),
        }


def read_dji_xmp(text: str | None) -> DjiXmp | None:
    """The DJI fields of an XMP packet, or None when it has no `drone-dji:` property."""
    if not text or re.search(r"drone-dji:[A-Za-z]", text, re.I) is None:
        return None

    def dji(name: str) -> str | None:
        return xmp_value(text, "drone-dji", name)

    def num(name: str) -> float | None:
        return _number(dji(name))

    longitude = num("GpsLongitude")
    if longitude is None:
        # ExifTool lists the misspelling too: some firmware writes it.
        longitude = num("GpsLongtitude")
    cx, cy = num("CalibratedOpticalCenterX"), num("CalibratedOpticalCenterY")
    return DjiXmp(
        relative_altitude_m=num("RelativeAltitude"),
        absolute_altitude_m=num("AbsoluteAltitude"),
        latitude=num("GpsLatitude"),
        longitude=longitude,
        gimbal_pitch_deg=num("GimbalPitchDegree"),
        gimbal_yaw_deg=num("GimbalYawDegree"),
        gimbal_roll_deg=num("GimbalRollDegree"),
        flight_pitch_deg=num("FlightPitchDegree"),
        flight_yaw_deg=num("FlightYawDegree"),
        flight_roll_deg=num("FlightRollDegree"),
        rtk_flag=_integer(dji("RtkFlag")),
        rtk_std_lon_m=num("RtkStdLon"),
        rtk_std_lat_m=num("RtkStdLat"),
        rtk_std_hgt_m=num("RtkStdHgt"),
        rtk_diff_age_s=num("RtkDiffAge"),
        gps_status=dji("GpsStatus"),
        altitude_type=dji("AltitudeType"),
        calibrated_focal_px=_positive(num("CalibratedFocalLength")),
        calibrated_center_px=None if cx is None or cy is None else (cx, cy),
        dewarp_flag=_integer(dji("DewarpFlag")),
        create_date=xmp_value(text, "xmp", "CreateDate"),
        utc_at_exposure=dji("UTCAtExposure"),
        unverified_fields=tuple(name for name in UNVERIFIED_DJI_FIELDS if dji(name) is not None),
    )


def _positive(value: float | None) -> float | None:
    return value if value is not None and value > 0 else None


# --- time ----------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameTime:
    """When a frame was taken: the camera's local time, its offset, UTC when derivable,
    and which tag said so. `utc` is None rather than a guess when no zone is known."""

    local: str | None = None
    offset: str | None = None
    utc: str | None = None
    source: TimeSource | None = None

    def to_dict(self) -> dict[str, object]:
        return {"local": self.local, "offset": self.offset, "utc": self.utc, "source": self.source}


def utc_text(moment: datetime) -> str | None:
    """ISO 8601 in UTC with a `Z` and milliseconds: `2026-10-01T15:20:30.123Z`; None when
    the moment is outside datetime's range once in UTC (year 1 at +01:00 -- a dead camera
    clock), which costs the UTC time and nothing else."""
    try:
        utc = moment.astimezone(UTC)
    except (OverflowError, ValueError):
        return None
    return utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _exif_text(value: object) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    if not isinstance(value, str):
        return None
    text = value.strip("\x00 \t\r\n")
    return text or None


def _exif_local(stamp: str | None, subsec: str | None) -> datetime | None:
    """`YYYY:MM:DD HH:MM:SS` plus `SubSecTime*` digits, as a naive datetime."""
    if stamp is None:
        return None
    try:
        moment = datetime.strptime(stamp[:19], "%Y:%m:%d %H:%M:%S")  # a local time, by definition
    except ValueError:
        return None
    digits = re.sub(r"\D", "", subsec or "")[:6]
    if digits:
        moment = moment.replace(microsecond=int(digits.ljust(6, "0")))
    return moment


def _local_text(moment: datetime) -> str:
    """A camera clock as ISO 8601 with no zone: to the millisecond when it has a fraction,
    as a video frame's offset time is, else to the second."""
    return moment.isoformat(timespec="milliseconds" if moment.microsecond else "seconds")


_OFFSET_RE = re.compile(r"^([+-])(\d{2}):?(\d{2})$")


def _offset(text: str | None) -> timezone | None:
    """`+01:00` / `-0500` as a timezone; None for anything else (including `   :  `)."""
    if text is None:
        return None
    match = _OFFSET_RE.match(text.strip())
    if match is None:
        return None
    sign = -1 if match.group(1) == "-" else 1
    hours, minutes = int(match.group(2)), int(match.group(3))
    if hours > 14 or minutes >= 60:
        return None
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def offset_text(delta: timedelta) -> str:
    """A UTC offset as `+HH:MM` / `-HH:MM`."""
    total = int(delta.total_seconds() // 60)
    sign = "-" if total < 0 else "+"
    total = abs(total)
    return f"{sign}{total // 60:02d}:{total % 60:02d}"


def _gps_utc(gps: Mapping[int, object]) -> datetime | None:
    """`GPSDateStamp` + `GPSTimeStamp` as an aware UTC datetime, or None."""
    date = _exif_text(gps.get(GPS_DATE))
    stamp = gps.get(GPS_TIME)
    if date is None or not isinstance(stamp, tuple | list) or len(stamp) != 3:
        return None
    parts = [_rational(part) for part in stamp]
    hours, minutes, seconds = (-1.0 if part is None else part for part in parts)
    if not (0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 61):
        return None
    try:
        day = datetime.strptime(date[:10], "%Y:%m:%d").replace(tzinfo=UTC)
        return day + timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except (ValueError, OverflowError):
        return None


def exif_time(
    exif_ifd: Mapping[int, object],
    gps_ifd: Mapping[int, object] | None = None,
    *,
    xmp_create_date: str | None = None,
) -> FrameTime:
    """When a photo was taken, first that applies:

    1. `DateTimeOriginal` + `SubSecTimeOriginal` + `OffsetTimeOriginal`: `exif-offset`,
       with UTC;
    2. else the GPS date and time: `gps-time`. UTC, and the local time kept beside it
       but not reconciled with it -- the GPS stamp is the fix's time, which a camera may
       write seconds stale, so no offset is derived from the difference;
    3. else `DateTimeOriginal` alone: `exif-local`, UTC unknown;
    4. else the XMP `xmp:CreateDate` (a DJI writes one): `xmp-createdate` when it carries
       a zone, `xmp-createdate-local` when it does not.
    """
    local = _exif_local(
        _exif_text(exif_ifd.get(DATETIME_ORIGINAL)), _exif_text(exif_ifd.get(SUBSEC_ORIGINAL))
    )
    zone = _offset(_exif_text(exif_ifd.get(OFFSET_ORIGINAL)))
    local_text = None if local is None else _local_text(local)
    if local is not None and zone is not None:
        aware = local.replace(tzinfo=zone)
        utc = utc_text(aware)
        if utc is not None:
            return FrameTime(
                local=local_text,
                offset=offset_text(aware.utcoffset() or timedelta(0)),
                utc=utc,
                source="exif-offset",
            )
    gps = _gps_utc(gps_ifd or {})
    gps_text = None if gps is None else utc_text(gps)
    if gps_text is not None:
        return FrameTime(local=local_text, offset=None, utc=gps_text, source="gps-time")
    if local is not None:
        return FrameTime(local=local_text, source="exif-local")
    if xmp_create_date:
        try:
            parsed = datetime.fromisoformat(xmp_create_date.strip())
        except ValueError:
            return FrameTime()
        utc = None if parsed.tzinfo is None else utc_text(parsed)
        if utc is None:
            naive = parsed.replace(tzinfo=None).isoformat()
            return FrameTime(local=naive, source="xmp-createdate-local")
        return FrameTime(
            local=parsed.replace(tzinfo=None).isoformat(),
            offset=offset_text(parsed.utcoffset() or timedelta(0)),
            utc=utc,
            source="xmp-createdate",
        )
    return FrameTime()


# --- exposure and focal --------------------------------------------------------------


def _rational(value: object) -> float | None:
    """An EXIF rational (Pillow's `IFDRational`), int or float, as a finite float."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) else None


def _iso(value: object) -> float | None:
    """`ISOSpeedRatings` is a SHORT, and some writers make it a list of them."""
    if isinstance(value, tuple | list):
        value = value[0] if value else None
    return _positive(_rational(value))


#: The exposure settings a camera can physically record, `(low, high)`, inclusive. Wide on
#: purpose: they keep out what no camera can write, not what is merely unusual.
#:
#: * f-number: N = 1 / (2 NA), and a lens in air has a numerical aperture below 1, so no
#:   lens is faster than f/0.5. The top, f/1024 (APEX Av 20), is past the f/200-f/500 of
#:   pinhole cameras.
#: * exposure time: 1 us is far shorter than the fastest electronic shutters (1/64000 s);
#:   a day (86 400 s) is longer than any long exposure a capture would carry.
#: * ISO: 1 is below every sensor's base sensitivity, 10 000 000 above the highest
#:   (4 560 000) any camera has sold with.
EXPOSURE_RANGES: dict[str, tuple[float, float]] = {
    "exposureTimeS": (1e-6, 86_400.0),
    "fNumber": (0.5, 1024.0),
    "iso": (1.0, 10_000_000.0),
}


def implausible_exposure(name: str, value: float | None) -> str | None:
    """Why `value` cannot be the `name` setting (an `EXPOSURE_RANGES` key), or None."""
    if value is None:
        return None
    low, high = EXPOSURE_RANGES[name]
    if math.isfinite(value) and low <= value <= high:
        return None
    return f"{name} {value:.3g} is outside {low:g}-{high:g}, which no camera records"


def ev100(exposure_time_s: float | None, f_number: float | None, iso: float | None) -> float | None:
    """The exposure value at ISO 100: `log2(N^2 / t) - log2(ISO / 100)`.

    Brightness of the scene the camera metered, in stops: one EV is a doubling. Two
    frames of the same surface under the same light should agree whatever settings the
    camera chose to record them with. None when a setting is missing or outside what a
    camera can record (`EXPOSURE_RANGES`): inside them the logarithms are always finite.
    """
    settings = {"exposureTimeS": exposure_time_s, "fNumber": f_number, "iso": iso}
    for name, value in settings.items():
        if value is None or implausible_exposure(name, value) is not None:
            return None
    assert exposure_time_s is not None and f_number is not None and iso is not None
    return math.log2(f_number * f_number / exposure_time_s) - math.log2(iso / 100.0)


@dataclass(frozen=True)
class Exposure:
    """A frame's exposure: what the camera recorded, and what the kept pixels measure
    (`lumaMean` and the clipped percentages are filled by `normalize` from the frame
    itself, after shrinking: they describe what training reads)."""

    exposure_time_s: float | None = None
    f_number: float | None = None
    iso: float | None = None
    bias_ev: float | None = None
    luma_mean: float | None = None
    clipped_high_pct: float | None = None
    clipped_low_pct: float | None = None

    @property
    def ev100(self) -> float | None:
        return ev100(self.exposure_time_s, self.f_number, self.iso)

    def to_dict(self) -> dict[str, object]:
        value = self.ev100
        return {
            "exposureTimeS": self.exposure_time_s,
            "fNumber": self.f_number,
            "iso": self.iso,
            "biasEv": self.bias_ev,
            "ev100": None if value is None else round(value, 4),
            "lumaMean": self.luma_mean,
            "clippedHighPct": self.clipped_high_pct,
            "clippedLowPct": self.clipped_low_pct,
        }


#: The diagonal of a 36 x 24 mm frame as COLMAP writes it: 43.27 mm, rounded from
#: hypot(36, 24) = 43.2666. Read off COLMAP 4.2 (pycolmap gave 443.72544 px for 24 mm at
#: 640x480, which is 24 * 800 / 43.27 exactly; the unrounded diagonal gives 443.7602).
#: The seed uses COLMAP's constant so it is the very value COLMAP would seed a photo with.
DIAGONAL_35MM = 43.27


def focal_px_from_35mm(f35: float, width: int, height: int) -> float:
    """A 35 mm-equivalent focal length in pixels of a `width` x `height` frame.

    By the **diagonal**: f_px = f35 * hypot(W, H) / 43.27 (`DIAGONAL_35MM`). That is the
    definition of the 35 mm equivalent (the same angle of view across the diagonal), and
    it is what COLMAP 4.2 itself does with EXIF `FocalLengthIn35mmFilm`; "long side over
    36", which the subsystem map said, would give 426.7 px for 24 mm at 640x480 where
    COLMAP reads 443.725. `tests/test_frame_meta.py::test_the_35mm_rule_is_colmaps_own`
    re-checks it against pycolmap wherever pycolmap is installed.
    """
    return f35 * math.hypot(width, height) / DIAGONAL_35MM


#: The field of view across a frame's long side a focal must give, in degrees,
#: `(low, high)`. A pinhole camera -- the model COLMAP seeds -- sees less than 180
#: degrees, so 179 is the widest; 0.1 degrees is a 20 m full-frame-equivalent focal, past
#: any camera lens. Wide on purpose: they keep out what no lens is, not what is rare.
FOCAL_FIELD_OF_VIEW_DEG = (0.1, 179.0)


def implausible_focal(px: float | None, frame_size: tuple[int, int]) -> str | None:
    """Why `px` cannot be a focal of a `frame_size` frame, or None (also for no focal)."""
    if px is None:
        return None
    low, high = FOCAL_FIELD_OF_VIEW_DEG
    if math.isfinite(px) and px > 0:
        fov = 2.0 * math.degrees(math.atan(max(frame_size) / (2.0 * px)))
        if low <= fov <= high:
            return None
        return (
            f"{px:.6g} px gives a field of view of {fov:.3g} degrees across {max(frame_size)} "
            f"px, outside the {low:g}-{high:g} a pinhole camera has"
        )
    return f"{px} px is no focal (field of view undefined)"


def calibrated_focal_px(
    focal_px: float, original_size: tuple[int, int], frame_size: tuple[int, int]
) -> float:
    """DJI's per-unit `CalibratedFocalLength`, in pixels of the original image (as DJI's
    P4 RTK documentation gives it; the unit is UNVERIFIED for the M3E), scaled to the kept
    frame by the long side."""
    return focal_px * max(frame_size) / max(original_size)


@dataclass(frozen=True)
class Focal:
    mm: float | None = None
    mm35: float | None = None
    px: float | None = None
    px_from: FocalFrom | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "mm": self.mm,
            "mm35": self.mm35,
            "px": None if self.px is None else round(self.px, 5),
            "pxFrom": self.px_from,
        }


# --- the record ----------------------------------------------------------------------


def mask_name(frame_name: str) -> str:
    """The frame's mask in COLMAP's `ImageReader.mask_path` convention: `<name>.png`."""
    return f"{frame_name}.png"


@dataclass(frozen=True)
class FrameRecord:
    """One kept frame, described from its original. Serialised by `to_dict`."""

    name: str
    original: str
    original_path: str | None = None
    original_size: tuple[int, int] | None = None
    #: The kept frame's own size, which `focal.px` is in pixels of.
    frame_size: tuple[int, int] | None = None
    candidate_index: int | None = None
    offset_s: float | None = None
    time: FrameTime = field(default_factory=FrameTime)
    gps: exif.Fix | None = None
    make: str | None = None
    model: str | None = None
    lens: str | None = None
    exposure: Exposure = field(default_factory=Exposure)
    focal: Focal = field(default_factory=Focal)
    dji: DjiXmp | None = None
    #: Why the original could not be read, when it could not; the record is then thin.
    unreadable: str | None = None
    #: Values the original carried that no camera can have written, each dropped from
    #: the record, with why (`implausible_exposure`): read, not trusted, not failed on.
    implausible: tuple[str, ...] = ()

    @property
    def mask(self) -> str:
        return mask_name(self.name)

    def with_luma(self, mean: float, high_pct: float, low_pct: float) -> FrameRecord:
        return replace(
            self,
            exposure=replace(
                self.exposure,
                luma_mean=round(mean, 3),
                clipped_high_pct=round(high_pct, 3),
                clipped_low_pct=round(low_pct, 3),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        gps = self.gps
        return {
            "name": self.name,
            "original": self.original,
            "originalPath": self.original_path,
            "originalSize": None if self.original_size is None else list(self.original_size),
            "frameSize": None if self.frame_size is None else list(self.frame_size),
            "candidateIndex": self.candidate_index,
            "offsetS": self.offset_s,
            "time": self.time.to_dict(),
            "gps": None
            if gps is None
            else {
                "lat": gps.lat,
                "lon": gps.lon,
                "altitudeM": gps.alt,
                "altitudeDatum": GPS_ALTITUDE_DATUM,
                "altitudeNote": GPS_ALTITUDE_NOTE,
                "hAccuracyM": gps.h_error_m,
                "dop": gps.dop,
            },
            "camera": {"make": self.make, "model": self.model, "lens": self.lens},
            "exposure": self.exposure.to_dict(),
            "focal": self.focal.to_dict(),
            "dji": None if self.dji is None else self.dji.to_dict(),
            "mask": self.mask,
            "unreadable": self.unreadable,
            "implausible": list(self.implausible),
        }


def read_photo(
    original: Path,
    *,
    frame_name: str,
    upload_root: Path,
    frame_size: tuple[int, int] | None,
) -> FrameRecord:
    """Everything `original` says about itself, for the kept frame `frame_name`.

    Never raises: a photo set can hold anything a phone or a drone put in it, and one
    unreadable original is a thin record (name, original, mask, `unreadable`), not a
    failed run. `frame_size` is the kept frame's, so the focal in pixels is the one pose
    will see; None (the kept frame did not open) leaves the focal in pixels unset.
    """
    try:
        relative = original.relative_to(upload_root).as_posix()
    except ValueError:
        relative = original.name
    try:
        return _read_photo(original, frame_name, relative, frame_size)
    except Exception as exc:  # a metadata reader of arbitrary files: total by design
        return FrameRecord(
            name=frame_name,
            original=original.name,
            original_path=relative,
            frame_size=frame_size,
            unreadable=f"{type(exc).__name__}: {exc}"[:200],
        )


def _read_photo(
    original: Path, frame_name: str, relative: str, frame_size: tuple[int, int] | None
) -> FrameRecord:
    with PIL.Image.open(original) as image:
        size = (int(image.width), int(image.height))
        tags = image.getexif()
        exif_ifd: Mapping[int, object] = tags.get_ifd(IFD_EXIF)
        gps_ifd: Mapping[int, object] = tags.get_ifd(IFD_GPS)
        info_xmp = image.info.get("xmp")
        make = _exif_text(tags.get(MAKE))
        model = _exif_text(tags.get(MODEL))
    packet: str | None = None
    if isinstance(info_xmp, str):
        info_xmp = info_xmp.encode("utf-8", errors="replace")
    if isinstance(info_xmp, bytes) and info_xmp:
        # Pillow's XMP segment, with or without the `x:xmpmeta` wrapper some writers omit.
        packet = xmp_packet(info_xmp) or info_xmp.decode("utf-8", errors="replace")
    if packet is None:
        with original.open("rb") as handle:
            packet = xmp_packet(handle.read(META_BYTES))
    dji = read_dji_xmp(packet)
    focal_mm = _positive(_rational(exif_ifd.get(FOCAL_LENGTH)))
    focal_35 = _positive(_rational(exif_ifd.get(FOCAL_35MM)))
    px: float | None = None
    px_from: FocalFrom | None = None
    implausible: list[str] = []
    if frame_size is not None:
        # The first source that gives a focal a camera can have; one that does not is
        # dropped, with why, and the next is tried.
        candidates: list[tuple[FocalFrom, float | None]] = [
            (
                "dji-calibrated",
                None
                if dji is None or dji.calibrated_focal_px is None
                else calibrated_focal_px(dji.calibrated_focal_px, size, frame_size),
            ),
            (
                "exif-35mm",
                None if focal_35 is None else focal_px_from_35mm(focal_35, *frame_size),
            ),
        ]
        for source, value in candidates:
            if value is None:
                continue
            reason = implausible_focal(value, frame_size)
            if reason is None:
                px, px_from = value, source
                break
            implausible.append(f"{source} focal {reason}")

    def plausible(name: str, value: float | None) -> float | None:
        reason = implausible_exposure(name, value)
        if reason is not None:
            implausible.append(reason)
            return None
        return value

    exposure = Exposure(
        exposure_time_s=plausible(
            "exposureTimeS", _positive(_rational(exif_ifd.get(EXPOSURE_TIME)))
        ),
        f_number=plausible("fNumber", _positive(_rational(exif_ifd.get(F_NUMBER)))),
        iso=plausible("iso", _iso(exif_ifd.get(ISO))),
        bias_ev=_rational(exif_ifd.get(EXPOSURE_BIAS)),
    )
    return FrameRecord(
        name=frame_name,
        original=original.name,
        original_path=relative,
        original_size=size,
        frame_size=frame_size,
        time=exif_time(exif_ifd, gps_ifd, xmp_create_date=None if dji is None else dji.create_date),
        gps=exif.read_fix(original),
        make=make,
        model=model,
        lens=_exif_text(exif_ifd.get(LENS_MODEL)),
        exposure=exposure,
        focal=Focal(mm=focal_mm, mm35=focal_35, px=px, px_from=px_from),
        dji=dji,
        implausible=tuple(implausible),
    )


def _usable_focal(record: FrameRecord) -> bool:
    px = record.focal.px
    if px is None:
        return False
    if record.frame_size is None:
        return math.isfinite(px) and px > 0
    return implausible_focal(px, record.frame_size) is None


def capture_focal(records: Sequence[FrameRecord]) -> dict[str, object]:
    """One focal length for the capture, or None when its frames disagree.

    `px` is the median of the frames' focal in pixels, and None when the relative spread
    `(max - min) / median` exceeds `FOCAL_SPREAD_LIMIT` (mixed lenses or zoom: one
    `SINGLE` camera must not be seeded from them) or when the frames disagree on where it
    came from. `frames` is how many had one, of `of`.
    """
    # Only focals a camera can have: a record built before `implausible_focal`, or by
    # anything else, may say inf, and frames that all say inf do not agree on a focal.
    usable = [r for r in records if _usable_focal(r)]
    values = [r.focal.px for r in usable if r.focal.px is not None]
    sources = sorted({r.focal.px_from for r in usable if r.focal.px_from is not None})
    summary: dict[str, object] = {
        "px": None,
        "from": sources[0] if len(sources) == 1 else ("mixed" if sources else None),
        "frames": len(values),
        "of": len(records),
        "spreadPct": None,
        "limitPct": round(FOCAL_SPREAD_LIMIT * 100.0, 6),
        "note": None,
    }
    if not values:
        summary["note"] = "no frame carried a focal length"
        return summary
    median = statistics.median(values)
    spread = (max(values) - min(values)) / median if median > 0 else math.inf
    summary["spreadPct"] = round(spread * 100.0, 4) if math.isfinite(spread) else None
    if len(sources) > 1:
        summary["note"] = f"mixed focal sources ({', '.join(sources)}): not one camera"
    elif spread > FOCAL_SPREAD_LIMIT:
        summary["note"] = (
            f"mixed focal lengths ({min(values):.1f}-{max(values):.1f} px): not one camera, "
            "so no single focal seeds pose"
        )
    else:
        summary["px"] = round(median, 5)
        summary["note"] = f"{len(values)} of {len(records)} frames agree"
    return summary
