"""EXIF GPS on the rendered orbit, from positions that are known exactly.

`tree_frames.py` renders the committed synthetic tree from poses whose camera centres are
known in metres. This module puts those centres on the globe and writes them into each
frame's EXIF, so `georeference: exif_gps` can be scored against a truth rather than
against "a JSON file appeared".

Two properties this is careful about, because both are ways the fixture could flatter the
implementation:

* **the pixels do not change.** The EXIF is spliced in as an `APP1` segment immediately
  after the JPEG's `SOI` marker rather than by re-saving the image, so the compressed
  scan is the byte-for-byte one `tree_frames` rendered. The COLMAP reconstruction here is
  therefore the same reconstruction `test_pose_colmap.py` measures at 40/40, and a
  regression in one cannot be hidden by re-encoding in the other.
* **the inverse of the projection is exact.** `enu_to_geodetic` is Ferrari's closed-form
  solution, not the flat degrees-per-metre approximation, and it is checked against
  `exif.enu_offsets` -- the forward direction the stage itself uses -- inside the test
  file. A fixture built with the same approximation as the code under test would agree
  with it for the wrong reason.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

import exif

#: Denominator for the seconds field of a GPS coordinate: 1e-4 arcsec, about 3 mm.
#:
#: Real cameras write far coarser than this (1/100 arcsec is common, about 30 cm). The
#: fixture writes finely because the quantity under test is the alignment residual, and a
#: residual dominated by the fixture's own rounding measures the fixture.
_SECONDS_DENOMINATOR = 10_000

#: Metres of altitude per unit of the `GPSAltitude` rational: 1 mm.
_ALTITUDE_DENOMINATOR = 1_000


def enu_to_geodetic(
    east: float, north: float, up: float, lat: float, lon: float, height: float
) -> tuple[float, float, float]:
    """East/north/up metres about a geodetic origin, back to latitude, longitude, height.

    Ferrari's closed-form inverse of the geodetic-to-ECEF map, so there is no iteration to
    converge and no tolerance to tune.
    """
    phi, lam = math.radians(lat), math.radians(lon)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    # The transpose of `exif.ecef_to_enu`'s rotation: east/north/up back into ECEF.
    rotation = np.array(
        [
            [-sin_lam, -sin_phi * cos_lam, cos_phi * cos_lam],
            [cos_lam, -sin_phi * sin_lam, cos_phi * sin_lam],
            [0.0, cos_phi, sin_phi],
        ],
        dtype=np.float64,
    )
    point = exif.geodetic_to_ecef(lat, lon, height) + rotation @ np.array([east, north, up])
    return _ecef_to_geodetic(float(point[0]), float(point[1]), float(point[2]))


def _ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    a = exif.WGS84_A
    e2 = exif.WGS84_E2
    b = a * math.sqrt(1.0 - e2)
    ep2 = (a * a - b * b) / (b * b)
    radius = math.hypot(x, y)
    theta = math.atan2(z * a, radius * b)
    latitude = math.atan2(
        z + ep2 * b * math.sin(theta) ** 3, radius - e2 * a * math.cos(theta) ** 3
    )
    longitude = math.atan2(y, x)
    prime = a / math.sqrt(1.0 - e2 * math.sin(latitude) ** 2)
    height = radius / math.cos(latitude) - prime
    return math.degrees(latitude), math.degrees(longitude), height


def write_gps(path: Path, lat: float, lon: float, alt: float) -> None:
    """Splice a GPS `APP1` segment into an existing JPEG, leaving its scan untouched."""
    tags = Image.Exif()
    tags[0x8825] = {
        1: "N" if lat >= 0 else "S",
        2: _sexagesimal(abs(lat)),
        3: "E" if lon >= 0 else "W",
        4: _sexagesimal(abs(lon)),
        5: 0 if alt >= 0 else 1,
        6: IFDRational(round(abs(alt) * _ALTITUDE_DENOMINATOR), _ALTITUDE_DENOMINATOR),
        18: "WGS-84",
    }
    payload = b"Exif\x00\x00" + tags.tobytes()
    segment = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    raw = path.read_bytes()
    if raw[:2] != b"\xff\xd8":
        raise ValueError(f"{path.name} is not a JPEG, so there is no APP1 slot to fill")
    path.write_bytes(raw[:2] + segment + raw[2:])


def _sexagesimal(degrees: float) -> tuple[IFDRational, IFDRational, IFDRational]:
    whole = int(degrees)
    minutes_total = (degrees - whole) * 60.0
    minutes = int(minutes_total)
    seconds = (minutes_total - minutes) * 60.0
    return (
        IFDRational(whole, 1),
        IFDRational(minutes, 1),
        IFDRational(round(seconds * _SECONDS_DENOMINATOR), _SECONDS_DENOMINATOR),
    )


def tag_orbit(
    frames: Path,
    centres: dict[str, np.ndarray],
    origin: tuple[float, float, float],
    *,
    noise_m: float = 0.0,
    seed: int = 20260922,
) -> dict[str, tuple[float, float, float]]:
    """Write a GPS fix onto every frame, at its true position plus optional noise.

    Returns the geodetic position written for each frame, so a test can assert against
    what it asked for rather than against what it hoped was written. The noise is seeded,
    so a failure reproduces.
    """
    rng = np.random.default_rng(seed)
    written: dict[str, tuple[float, float, float]] = {}
    for name in sorted(centres):
        offset = rng.normal(scale=noise_m, size=3) if noise_m > 0.0 else np.zeros(3)
        east, north, up = np.asarray(centres[name], dtype=np.float64) + offset
        lat, lon, alt = enu_to_geodetic(float(east), float(north), float(up), *origin)
        write_gps(frames / name, lat, lon, alt)
        written[name] = (lat, lon, alt)
    return written
