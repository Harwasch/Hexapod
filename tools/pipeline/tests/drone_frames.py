"""Synthetic drone and phone photos with the metadata real ones carry, written at test time.

**These are not real files.** No Mavic 3 Enterprise photo is in the repository, so the DJI
XMP below is assembled from the field names in ExifTool's DJI tag table
(https://exiftool.org/TagNames/DJI.html), the RTK flag codes in Sitemark's article, and
values of a plausible size (an RTK-fixed standard deviation of about a centimetre, a
calibrated focal of a few thousand pixels at full resolution). A real M3E photo is what
would verify them; until then `frame_meta` names the unverified fields in every record.

The XMP is spliced in by hand as an APP1 segment straight after SOI, the way a camera
writes it (and the way `tests/test_minnetonka.py` does), so these tests do not depend on
Pillow's `xmp=` save option. EXIF goes through Pillow's `exif=`, nested IFDs included.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

#: Where the synthetic flights are: a field in Minnesota (no real site).
LAT, LON = 44.9446257, -93.4259758
#: A plausible calibrated focal for a 24 mm-equivalent 4/3" sensor at 5280 px wide.
CALIBRATED_FOCAL_PX = 3713.29
M3E_SIZE = (5280, 3956)


def dji_xmp(
    *,
    rtk_flag: int | None = 50,
    std: tuple[float, float, float] | None = (0.012, 0.011, 0.02),
    lat: float = LAT,
    lon: float = LON,
    abs_alt: float = 312.45,
    rel_alt: float = 60.1,
    calibrated_focal: float | None = CALIBRATED_FOCAL_PX,
    element_form: bool = False,
    longtitude_typo: bool = False,
    extra: Mapping[str, str] | None = None,
) -> str:
    """A DJI-style XMP packet. `rtk_flag=None` writes no Rtk fields at all (a Mavic 2)."""
    fields: dict[str, str] = {
        "GpsLatitude": f"{lat:+.9f}",
        ("GpsLongtitude" if longtitude_typo else "GpsLongitude"): f"{lon:+.9f}",
        "AbsoluteAltitude": f"{abs_alt:+.3f}",
        "RelativeAltitude": f"{rel_alt:+.3f}",
        "GimbalRollDegree": "+0.00",
        "GimbalYawDegree": "-12.30",
        "GimbalPitchDegree": "-90.00",
        "FlightRollDegree": "+1.20",
        "FlightYawDegree": "-11.80",
        "FlightPitchDegree": "-3.40",
    }
    if calibrated_focal is not None:
        fields["CalibratedFocalLength"] = f"{calibrated_focal:.6f}"
        fields["CalibratedOpticalCenterX"] = "2640.000000"
        fields["CalibratedOpticalCenterY"] = "1978.000000"
        fields["DewarpFlag"] = "0"
    if rtk_flag is not None:
        fields["RtkFlag"] = str(rtk_flag)
        if std is not None:
            fields["RtkStdLon"] = f"{std[0]:.5f}"
            fields["RtkStdLat"] = f"{std[1]:.5f}"
            fields["RtkStdHgt"] = f"{std[2]:.5f}"
    fields.update(extra or {})
    namespaces = (
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        'xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"'
    )
    create = "2026-10-01T10:20:30"
    if element_form:
        body = "".join(f"<drone-dji:{k}>{v}</drone-dji:{k}>" for k, v in fields.items())
        description = (
            f"<rdf:Description {namespaces}><xmp:CreateDate>{create}</xmp:CreateDate>"
            f"{body}</rdf:Description>"
        )
    else:
        attributes = " ".join(f'drone-dji:{k}="{v}"' for k, v in fields.items())
        description = f'<rdf:Description {namespaces} xmp:CreateDate="{create}" {attributes}/>'
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f"{description}</rdf:RDF></x:xmpmeta>"
    )


def _sexagesimal(degrees: float) -> tuple[IFDRational, IFDRational, IFDRational]:
    whole = int(degrees)
    minutes_total = (degrees - whole) * 60.0
    minutes = int(minutes_total)
    seconds = (minutes_total - minutes) * 60.0
    return (
        IFDRational(whole, 1),
        IFDRational(minutes, 1),
        IFDRational(round(seconds * 1e4), 10_000),
    )


def gps_ifd(
    lat: float, lon: float, alt: float, *, h_error: float | None = None
) -> dict[int, object]:
    """A GPS IFD for (lat, lon, alt), with `GPSHPositioningError` when given."""
    ifd: dict[int, object] = {
        1: "N" if lat >= 0 else "S",
        2: _sexagesimal(abs(lat)),
        3: "E" if lon >= 0 else "W",
        4: _sexagesimal(abs(lon)),
        5: 0 if alt >= 0 else 1,
        6: IFDRational(round(abs(alt) * 1000), 1000),
        18: "WGS-84",
    }
    if h_error is not None:
        ifd[31] = IFDRational(round(h_error * 1000), 1000)
    return ifd


def pixels(size: tuple[int, int], *, seed: int = 0, level: int | None = None) -> Image.Image:
    """A blocky seeded pattern (it compresses, unlike per-pixel noise), or a flat grey."""
    width, height = size
    if level is not None:
        return Image.new("RGB", size, (level, level, level))
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(height // 16 + 1, width // 16 + 1, 3), dtype=np.uint8)
    blocks = np.kron(coarse, np.ones((16, 16, 1), dtype=np.uint8))[:height, :width]
    return Image.fromarray(blocks, mode="RGB")


def write_jpeg(
    path: Path,
    *,
    size: tuple[int, int],
    exif: Mapping[int, object] | None = None,
    exif_ifd: Mapping[int, object] | None = None,
    gps: Mapping[int, object] | None = None,
    xmp: str | None = None,
    seed: int = 0,
    level: int | None = None,
) -> Path:
    """Save a JPEG with EXIF (IFD0, Exif IFD, GPS IFD) and, optionally, an XMP APP1."""
    tags = Image.Exif()
    for key, value in (exif or {}).items():
        tags[key] = value
    if exif_ifd:
        tags[0x8769] = dict(exif_ifd)
    if gps:
        tags[0x8825] = dict(gps)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels(size, seed=seed, level=level).save(path, format="JPEG", quality=90, exif=tags)
    if xmp is not None:
        payload = b"http://ns.adobe.com/xap/1.0/\0" + xmp.encode()
        segment = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
        raw = path.read_bytes()
        path.write_bytes(raw[:2] + segment + raw[2:])
    return path


def dji_m3e_photo(
    path: Path,
    index: int,
    *,
    rtk_flag: int | None = 50,
    size: tuple[int, int] = (3000, 2250),
    exposure_s: float = 1 / 1000,
    calibrated_focal: float | None = CALIBRATED_FOCAL_PX,
    h_error: float | None = None,
    level: int | None = None,
) -> Path:
    """A synthetic M3E-style photo: EXIF without an offset, GPS, and DJI XMP with RTK.

    `calibrated_focal` is at the M3E's full 5280 px width; at a smaller synthetic `size`
    it is scaled by the long side, as a real file's would be at that resolution."""
    lat, lon = LAT + index * 1e-5, LON + index * 1e-5
    focal = None if calibrated_focal is None else calibrated_focal * max(size) / max(M3E_SIZE)
    return write_jpeg(
        path,
        size=size,
        exif={0x010F: "DJI", 0x0110: "M3E"},
        exif_ifd={
            0x9003: f"2026:10:01 10:20:{30 + index:02d}",
            0x829A: IFDRational(round(exposure_s * 1_000_000), 1_000_000),
            0x829D: IFDRational(28, 10),
            0x8827: 100,
            0x9204: IFDRational(0, 1),
            0x920A: IFDRational(1229, 100),
            0xA405: 24,
        },
        gps=gps_ifd(lat, lon, 312.45, h_error=h_error),
        xmp=dji_xmp(rtk_flag=rtk_flag, lat=lat, lon=lon, calibrated_focal=focal),
        seed=index,
        level=level,
    )


def iphone_photo(
    path: Path,
    index: int,
    *,
    size: tuple[int, int] = (1600, 1200),
    f35: int = 24,
    h_error: float = 4.7,
    level: int | None = None,
) -> Path:
    """A synthetic iPhone-style photo: EXIF with `OffsetTimeOriginal` and sub-seconds,
    a 35 mm focal, and GPS with `GPSHPositioningError`. No XMP drone fields."""
    return write_jpeg(
        path,
        size=size,
        exif={0x010F: "Apple", 0x0110: "iPhone 15 Pro"},
        exif_ifd={
            0x9003: f"2025:06:14 10:41:{index:02d}",
            0x9291: "123",
            0x9011: "+01:00",
            0x829A: IFDRational(1, 120),
            0x829D: IFDRational(178, 100),
            0x8827: 64,
            0x920A: IFDRational(6765, 1000),
            0xA405: f35,
            0xA434: "iPhone 15 Pro back triple camera 6.765mm f/1.78",
        },
        gps=gps_ifd(51.5074 + index * 1e-5, -0.1278, 31.48, h_error=h_error),
        seed=100 + index,
        level=level,
    )
