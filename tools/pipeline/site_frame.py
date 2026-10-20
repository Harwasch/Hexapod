"""A site's fixed east/north/up frame, and where in it a capture of that site lands.

A site is captured more than once: each capture is an *epoch* of it (decision D3), and
every epoch is expressed in the one frame the site's first epoch defined. The worker
hands that frame to the stages that place a capture -- `ingest_splat`,
`manual_placement`, `exif_gps` -- as the `site_frame` param (`from_param`), and they write
it into `georef.json`, whose top-level `lat`/`lon`/`height` then *are* the frame's
origin. Every stage after them (`place_splat`, `splat_tiles`, `splat_ground`, `catalog`)
already reads its origin from `georef.json`, so the frame reaches them with no second
source of truth.

What a frame does **not** remove is the recentring: an unaligned capture is still centred
on its own footprint and ground, because that is how the pipeline decides which point of
the splat the capture's coordinate refers to (`stages._frame_from_poses`). It is then
moved by `offset_m` -- the east/north/up vector from the frame's origin to the capture's
located point -- so it lands where it was captured, inside the site's frame. An
EXIF-aligned capture never recentred; its fixes are simply expressed about the frame's
origin instead of their own median, and it carries no offset.

The echo (`echo`) is what a stage writes into `georef.json` / `source_meta.json` under
`siteFrame`, so the API can check, when it registers the run, that the tiles really are
about the frame it was given (`FrameMismatch` there).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import exif

#: D1's SPZ range at 12 fractional bits: a position must satisfy |p| < 2047 m in its
#: tile's frame. An offset at or past it cannot be packed in the root tile at all.
MAX_OFFSET_M = 2047.0

#: Bounds on a frame's height, in metres: generous (the Dead Sea to above Everest, and
#: then some), only there so a garbage number is refused rather than used.
_HEIGHT_RANGE = (-1.0e5, 1.0e5)


@dataclass(frozen=True)
class SiteFrame:
    """A site's origin: the WGS84 point its east/north/up frame is about."""

    site_id: str
    lat: float
    lon: float
    height: float


def _number(value: object, key: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"site_frame.{key} must be a number, not {value!r}")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(
            f"site_frame.{key} must be a finite number in [{low:g}, {high:g}], not {value!r}"
        )
    return number


def from_param(value: object) -> SiteFrame | None:
    """The `site_frame` param as a `SiteFrame`; None when the run was given none.

    Anything that is not `{"siteId": str, "lat", "lon", "height"}` with finite numbers in
    range is refused, naming the key, rather than placing a capture in a frame nobody
    meant.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"site_frame must be an object, not {type(value).__name__}")
    site_id = value.get("siteId")
    if not isinstance(site_id, str) or not site_id:
        raise ValueError(f"site_frame.siteId must be a non-empty string, not {site_id!r}")
    return SiteFrame(
        site_id=site_id,
        lat=_number(value.get("lat"), "lat", -90.0, 90.0),
        lon=_number(value.get("lon"), "lon", -180.0, 180.0),
        height=_number(value.get("height"), "height", *_HEIGHT_RANGE),
    )


def offset_m(frame: SiteFrame, lat: float, lon: float) -> tuple[float, float, float]:
    """East/north/up metres from the frame's origin to (lat, lon) at the frame's height.

    The frame's height rather than the capture's: recentring has already put the
    capture's own ground at its z = 0, so the only consistent vertical for an unaligned
    capture is the frame's. What `up` remains is the earth's curvature between the two
    points (about d^2 / 2R: under 8 cm within a kilometre) -- the same rule
    `exif.enu_offsets` applies to a fix with no altitude.

    Refused at or past `MAX_OFFSET_M` horizontally. That is necessary, not sufficient: a
    capture whose own extent reaches past the range from there still cannot be packed, and
    the tiler's refusal to wrap (D1) is the real gate.
    """
    point = exif.geodetic_to_ecef(lat, lon, frame.height)
    enu = exif.ecef_to_enu(point, frame.lat, frame.lon, frame.height)
    east, north, up = float(enu[0]), float(enu[1]), float(enu[2])
    if max(abs(east), abs(north)) >= MAX_OFFSET_M:
        raise ValueError(
            f"the capture at {lat}, {lon} is {east:.1f} m east and {north:.1f} m north of "
            f"site {frame.site_id}'s origin ({frame.lat}, {frame.lon}); a capture "
            f"{MAX_OFFSET_M:g} m or more from its site's origin cannot be packed in that "
            f"site's frame. Capture it as a site of its own"
        )
    return east, north, up


def is_zero(offset: Sequence[float]) -> bool:
    """Whether an offset moves nothing (`-0.0 == 0.0`, so a negative zero is zero too)."""
    return all(v == 0.0 for v in offset)


def is_origin(frame: SiteFrame, lat: float, lon: float) -> bool:
    """Whether (lat, lon) is the frame's own origin, exactly."""
    return lat == frame.lat and lon == frame.lon


def echo(
    frame: SiteFrame,
    *,
    offset: Sequence[float] | None,
    offset_source: str | None,
    located: Sequence[float] | None,
) -> dict[str, object]:
    """The `siteFrame` block a placing stage writes: the frame it used, and how far from
    its origin the capture was put (`offsetM`; null when its fixes were expressed in the
    frame directly, the EXIF-aligned branch)."""
    return {
        "siteId": frame.site_id,
        "lat": frame.lat,
        "lon": frame.lon,
        "height": frame.height,
        "offsetM": None if offset is None else [float(v) for v in offset],
        "offsetSource": offset_source,
        "located": None
        if located is None
        else {"lat": float(located[0]), "lon": float(located[1]), "height": float(located[2])},
    }
