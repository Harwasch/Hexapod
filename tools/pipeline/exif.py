"""EXIF GPS, and the local metric frame a set of fixes defines.

Everything here is pure or a file read, so the `georeference` stage stays a thin layer
that resolves artifacts. Three things are worth naming, because each one is a way this
could be wrong and still look right:

* **the altitude is not an ellipsoid height.** `GPSAltitude` with `GPSAltitudeRef 0` is
  metres above mean sea level, which is the geoid -- and the geoid is tens of metres from
  the WGS84 ellipsoid the globe draws. So the vertical component of an EXIF georeference
  is the least trustworthy thing in it by a wide margin, `georef.json` says so in words,
  and the viewer's clamp (`apps/web/src/cesium/placement.ts`) is what actually puts the
  model on the ground. Nothing here pretends to a geoid model it does not have.
* **degrees, minutes and seconds are three rationals, and cameras disagree about which
  ones they use.** Some write `(d, m, s)`, some write `(d, 0, 0)` with a big denominator,
  some write `(d, m.mmm, 0)`. `_degrees` sums all three regardless, which is correct for
  every one of those spellings and is why no camera-specific branch exists.
* **the reference frame is chosen here, not by COLMAP.** `colmap model_aligner` will
  happily take GPS directly (`--ref_is_gps 1 --alignment_type enu`) and pick its own
  origin, and then nothing downstream knows where that origin is. Instead the origin is
  the median fix, the positions handed to COLMAP are already metres east/north/up about
  it, and what comes back is a similarity into a frame this project defined.

The ellipsoid arithmetic is WGS84 proper -- geodetic to ECEF to ENU -- rather than the
flat degrees-per-metre approximation `gaussians.ground_samples` uses. That approximation
is fine over the tens of metres a ground grid spans and is not fine for a frame whose
whole job is to carry metric scale.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import PIL.Image

__all__ = [
    "Fix",
    "ecef_to_enu",
    "enu_offsets",
    "geodetic_to_ecef",
    "median_fix",
    "read_fix",
    "read_fixes",
]

F64 = npt.NDArray[np.float64]

#: WGS84, the datum the globe draws and the datum EXIF GPS is written in
#: (`GPSMapDatum` says "WGS-84" when a camera bothers to write it at all).
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

#: EXIF GPS IFD tag numbers. Spelled out rather than imported from `PIL.ExifTags` so that
#: the set this reads is visible in one place.
_GPS_IFD = 0x8825
_LAT_REF, _LAT = 1, 2
_LON_REF, _LON = 3, 4
_ALT_REF, _ALT = 5, 6
_DOP = 11


@dataclass(frozen=True)
class Fix:
    """One image's GPS fix, in degrees and metres.

    `alt` is None when the camera wrote no `GPSAltitude` -- common, and not an error. A
    set of fixes with no altitudes still defines a horizontal frame; it just cannot say
    anything about height, which `georeference` then records rather than inventing.
    """

    name: str
    lat: float
    lon: float
    alt: float | None = None
    #: `GPSDOP`, the camera's own dilution of precision, when it wrote one.
    dop: float | None = None


def read_fix(path: Path) -> Fix | None:
    """The GPS fix in one image's EXIF, or None if it has none.

    Never raises on a file that is not a readable image: a frame set can contain anything
    a phone put in it, and one unreadable frame is not a reason to fail a georeference
    that thirty-nine others can carry.
    """
    try:
        with PIL.Image.open(path) as image:
            gps = image.getexif().get_ifd(_GPS_IFD)
    except (OSError, ValueError, SyntaxError):
        return None
    if not gps:
        return None
    lat = _signed(gps.get(_LAT), gps.get(_LAT_REF), negative="S")
    lon = _signed(gps.get(_LON), gps.get(_LON_REF), negative="W")
    if lat is None or lon is None or not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return Fix(
        name=path.name,
        lat=lat,
        lon=lon,
        alt=_altitude(gps.get(_ALT), gps.get(_ALT_REF)),
        dop=_float(gps.get(_DOP)),
    )


def read_fixes(frames: Path) -> tuple[Fix, ...]:
    """Every fix in a directory of frames, in file-name order.

    File-name order because that is the order COLMAP names images in, and the two lists
    are joined by name later anyway -- but a stable order makes the log readable and the
    median deterministic.
    """
    found = [read_fix(path) for path in sorted(p for p in frames.iterdir() if p.is_file())]
    return tuple(fix for fix in found if fix is not None)


def _signed(value: object, ref: object, *, negative: str) -> float | None:
    degrees = _degrees(value)
    if degrees is None:
        return None
    hemisphere = str(ref).strip().upper()[:1] if ref is not None else ""
    return -degrees if hemisphere == negative else degrees


def _degrees(value: object) -> float | None:
    """`(d, m, s)` to decimal degrees, summing all three whatever the camera put where."""
    if not isinstance(value, (tuple, list)) or not value:
        return None
    parts = [_float(part) for part in value[:3]]
    if parts[0] is None:
        return None
    total = 0.0
    for scale, part in zip((1.0, 60.0, 3600.0), parts, strict=False):
        if part is not None:
            total += part / scale
    return total


def _altitude(value: object, ref: object) -> float | None:
    """`GPSAltitude`, negated when `GPSAltitudeRef` says below sea level.

    The reference byte arrives as `0`, `b"\\x00"` or `"0"` depending on how the file was
    written and which PIL read it, so it is compared as an integer after coercion rather
    than against any one of those spellings.
    """
    metres = _float(value)
    if metres is None:
        return None
    return -metres if _ref_byte(ref) == 1 else metres


def _ref_byte(value: object) -> int:
    if isinstance(value, bytes):
        return int(value[0]) if value else 0
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# --- the frame a set of fixes defines ------------------------------------------------


def median_fix(fixes: Sequence[Fix]) -> tuple[float, float, float]:
    """The reference point: the component-wise median of the fixes, in degrees and metres.

    The median rather than the mean because a single wild fix -- a phone that reported a
    cell-tower position for one frame before the GPS locked -- should not move the origin
    of the whole capture. Altitude falls back to zero when no camera wrote one, and the
    stage records that it did so rather than letting a zero read as a measurement.
    """
    if not fixes:
        raise ValueError("no GPS fixes; there is no frame to define")
    lat = float(np.median([fix.lat for fix in fixes]))
    lon = float(np.median([fix.lon for fix in fixes]))
    heights = [fix.alt for fix in fixes if fix.alt is not None]
    alt = float(np.median(heights)) if heights else 0.0
    return lat, lon, alt


def geodetic_to_ecef(lat: float, lon: float, height: float) -> F64:
    """WGS84 geodetic to earth-centred, earth-fixed metres."""
    phi, lam = math.radians(lat), math.radians(lon)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)
    return np.array(
        [
            (n + height) * cos_phi * math.cos(lam),
            (n + height) * cos_phi * math.sin(lam),
            (n * (1.0 - WGS84_E2) + height) * sin_phi,
        ],
        dtype=np.float64,
    )


def ecef_to_enu(point: F64, origin_lat: float, origin_lon: float, origin_height: float) -> F64:
    """ECEF metres to east/north/up metres about a geodetic origin."""
    phi, lam = math.radians(origin_lat), math.radians(origin_lon)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    rotation = np.array(
        [
            [-sin_lam, cos_lam, 0.0],
            [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
            [cos_phi * cos_lam, cos_phi * sin_lam, sin_phi],
        ],
        dtype=np.float64,
    )
    return rotation @ (point - geodetic_to_ecef(origin_lat, origin_lon, origin_height))


def enu_offsets(
    fixes: Iterable[Fix], origin: tuple[float, float, float]
) -> dict[str, tuple[float, float, float]]:
    """Each fix as east/north/up metres about `origin`, keyed by image name.

    A fix with no altitude is placed at the origin's height rather than dropped: its
    horizontal position is a real measurement and is worth keeping, and the vertical
    residual it contributes is zero rather than wrong.
    """
    lat, lon, height = origin
    out: dict[str, tuple[float, float, float]] = {}
    for fix in fixes:
        ecef = geodetic_to_ecef(fix.lat, fix.lon, height if fix.alt is None else fix.alt)
        east, north, up = ecef_to_enu(ecef, lat, lon, height)
        out[fix.name] = (float(east), float(north), float(up))
    return out
