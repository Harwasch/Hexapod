"""GNSS pose priors for bundle adjustment: each frame's fix, in the georeference's frame.

On the COLMAP 4.2 path (`colmap: "4.2"`, `colmap4.py`) a frame's GPS fix can enter
incremental mapping as a `PosePrior` -- a position with a covariance -- so bundle
adjustment solves the cameras *in* the east/north/up frame the fixes define, at metric
scale, instead of in an arbitrary frame that `georeference` must then fit a similarity
to. Decision D4: "GNSS priors enter BA on the pycolmap 4.2 path (PosePrior, CARTESIAN in
site-frame ENU, per-image covariance from RTK std / GPS accuracy)".

**The frame is the georeference's** (`frame_origin`): the site's frame when the run was
given one, else the fixes' median (`exif.median_fix`) -- exactly the origin
`exif_gps`'s aligned branch expresses the same fixes about, so the two stages agree by
construction and the georeference similarity comes out near the identity.

**The standard deviation of each prior** is the best the capture says about itself, then
what the run says, then a consumer figure (`sigma_for`):

* horizontal: a DJI RTK **fixed** solution's own `RtkStdLon` / `RtkStdLat` (the larger),
  else the EXIF `GPSHPositioningError` (`gps.hAccuracyM`), both from `frames_meta.json`
  (M3a); else the `prior_std_m` param; else `DEFAULT_STD_M` (5 m, the same consumer-GNSS
  figure as `stages.DEFAULT_ALIGNMENT_MAX_ERROR_M`);
* vertical: the RTK fixed `RtkStdHgt`; else `prior_std_v_m`; else the horizontal times
  `VERTICAL_FACTOR` (2). First principles, not a measurement: every satellite is above
  the horizon, so the vertical dilution of precision is typically 1.5-2x the horizontal;
* **unconstrained** (`UNCONSTRAINED_STD_M`, 1 km) when the fix has no altitude, or its
  altitude datum is not the frame's (an EXIF `GPSAltitude` is above mean sea level; a
  site frame may be ellipsoidal -- tens of metres apart).

An RTK *float* or consumer fix never borrows an RTK figure: only flag 50 (fixed) does
(`frame_meta.RTK_FIXED`). A wrong declared sigma is the one way these priors overclaim,
so the source of every sigma is recorded (`summary()`).

Stdlib and numpy only; the pycolmap half is `colmap4.py priors`.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import exif
import frame_meta
import site_frame

__all__ = [
    "COORDINATE_SYSTEM",
    "DEFAULT_STD_M",
    "LOSS_SCALE",
    "REQUESTS",
    "UNCONSTRAINED_STD_M",
    "VERTICAL_FACTOR",
    "FramePrior",
    "PriorPlan",
    "frame_origin",
    "plan_priors",
    "sigma_for",
]

#: Consumer GNSS, metres (1 sigma, horizontal).
DEFAULT_STD_M = 5.0
#: Vertical over horizontal GNSS error, when nothing measured the vertical.
VERTICAL_FACTOR = 2.0
#: A sigma large enough that the prior says nothing about that axis.
UNCONSTRAINED_STD_M = 1000.0
#: COLMAP's own `prior_position_loss_scale`: chi-square with 3 dof at 95%. Left as is.
LOSS_SCALE = 7.815
COORDINATE_SYSTEM = "CARTESIAN"
#: `pose_priors`: `off`, or `auto` (used whenever the engine and the fixes allow).
REQUESTS: tuple[str, ...] = ("off", "auto")
#: What the frame's height is when it is the fixes' own median: EXIF altitude's datum.
MEDIAN_FIX_DATUM = frame_meta.GPS_ALTITUDE_DATUM


@dataclass(frozen=True)
class FramePrior:
    """One frame's prior: east/north/up metres about the plan's origin, and its sigmas."""

    name: str
    position: tuple[float, float, float]
    std_h_m: float
    std_v_m: float
    #: Where the horizontal sigma came from: `frames_meta:dji-rtk`,
    #: `frames_meta:gps-h-accuracy`, `param` or `default`.
    source: str
    #: Why the vertical is unconstrained, when it is.
    unconstrained: str | None = None

    def covariance(self) -> list[list[float]]:
        h, v = self.std_h_m**2, self.std_v_m**2
        return [[h, 0.0, 0.0], [0.0, h, 0.0], [0.0, 0.0, v]]


@dataclass(frozen=True)
class PriorPlan:
    """What the pose stage will hand bundle adjustment, or why it will hand it nothing."""

    requested: str
    used: bool
    reason: str
    origin: tuple[float, float, float] | None
    #: `median-fix` or `site-frame`.
    origin_source: str | None
    height_datum: str | None
    priors: tuple[FramePrior, ...] = ()

    @property
    def count(self) -> int:
        return len(self.priors) if self.used else 0

    def vertical(self) -> str:
        if not self.used:
            return "unconstrained: no priors"
        free = [p for p in self.priors if p.unconstrained is not None]
        if not free:
            return "constrained"
        why = sorted({p.unconstrained for p in free if p.unconstrained is not None})
        if len(free) == len(self.priors):
            return f"unconstrained: {'; '.join(why)}"
        return f"unconstrained on {len(free)} of {len(self.priors)}: {'; '.join(why)}"

    def write(self, path: Path) -> None:
        """`work/priors.json`, what `colmap4.py priors` reads (design section 4.2)."""
        document = {
            "coordinateSystem": COORDINATE_SYSTEM,
            "priors": [
                {"name": p.name, "position": list(p.position), "covariance": p.covariance()}
                for p in self.priors
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")

    def summary(self) -> dict[str, object]:
        """`poses.json`'s `priors` block (design section 4.3), less `mapper`."""
        sources = sorted({p.source for p in self.priors})
        horizontal = [p.std_h_m for p in self.priors]
        vertical = [p.std_v_m for p in self.priors]
        return {
            "requested": self.requested,
            "used": self.used,
            "reason": self.reason,
            "count": self.count,
            "coordinateSystem": COORDINATE_SYSTEM,
            "frame": "east/north/up metres about origin",
            "origin": None
            if self.origin is None
            else {
                "lat": self.origin[0],
                "lon": self.origin[1],
                "height": self.origin[2],
                "source": self.origin_source,
                "heightDatum": self.height_datum,
            },
            "sigmaM": {
                "horizontalMedian": statistics.median(horizontal) if horizontal else None,
                "verticalMedian": statistics.median(vertical) if vertical else None,
                "source": sources[0] if len(sources) == 1 else (sources or None),
            },
            "vertical": self.vertical(),
            "robustLoss": True,
            "lossScale": LOSS_SCALE,
        }


def frame_origin(
    fixes: tuple[exif.Fix, ...], frame: site_frame.SiteFrame | None
) -> tuple[tuple[float, float, float], str, str]:
    """The origin the priors (and `exif_gps`'s aligned branch) are about, where it came
    from, and its height datum: the site's frame when given (its own datum, else
    `msl-unverified`), else the fixes' median (EXIF altitude's datum)."""
    if frame is not None:
        return (
            (frame.lat, frame.lon, frame.height),
            "site-frame",
            frame.height_datum or MEDIAN_FIX_DATUM,
        )
    return exif.median_fix(fixes), "median-fix", MEDIAN_FIX_DATUM


def _records(frames_meta: Mapping[str, object] | None) -> dict[str, Mapping[str, object]]:
    """`frames_meta.json`'s records keyed by kept frame name (M3a: a list of records)."""
    if not isinstance(frames_meta, Mapping):
        return {}
    frames = frames_meta.get("frames")
    if not isinstance(frames, list):
        return {}
    out: dict[str, Mapping[str, object]] = {}
    for record in frames:
        if isinstance(record, Mapping) and isinstance(record.get("name"), str):
            out[str(record["name"])] = record
    return out


def _positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def _sigma_from_meta(
    record: Mapping[str, object] | None,
) -> tuple[float | None, float | None, str | None]:
    """The M3a adapter: (horizontal, vertical, source) from one `frames_meta` record.

    An RTK fixed solution (`dji.rtk.flag == 50`) gives both from its own standard
    deviations (horizontal the larger of lon and lat). Otherwise the EXIF horizontal
    error (`gps.hAccuracyM`) gives the horizontal and nothing the vertical."""
    if record is None:
        return None, None, None
    dji = record.get("dji")
    rtk = dji.get("rtk") if isinstance(dji, Mapping) else None
    if isinstance(rtk, Mapping) and rtk.get("flag") == frame_meta.RTK_FIXED:
        lon, lat = _positive(rtk.get("stdLonM")), _positive(rtk.get("stdLatM"))
        horizontal = max(lon, lat) if lon is not None and lat is not None else (lon or lat)
        if horizontal is not None:
            return horizontal, _positive(rtk.get("stdHgtM")), "frames_meta:dji-rtk"
    gps = record.get("gps")
    if isinstance(gps, Mapping):
        accuracy = _positive(gps.get("hAccuracyM"))
        if accuracy is not None:
            return accuracy, None, "frames_meta:gps-h-accuracy"
    return None, None, None


def sigma_for(
    record: Mapping[str, object] | None, std_m: float | None, std_v_m: float | None
) -> tuple[float, float, str]:
    """One frame's (horizontal, vertical, source), by the module's order."""
    horizontal, vertical, source = _sigma_from_meta(record)
    if horizontal is None:
        if std_m is not None:
            horizontal, source = std_m, "param"
        else:
            horizontal, source = DEFAULT_STD_M, "default"
    if vertical is None:
        vertical = std_v_m if std_v_m is not None else horizontal * VERTICAL_FACTOR
    assert source is not None
    return horizontal, vertical, source


def _fix_datum(record: Mapping[str, object] | None) -> str:
    gps = record.get("gps") if record is not None else None
    datum = gps.get("altitudeDatum") if isinstance(gps, Mapping) else None
    return datum if isinstance(datum, str) else MEDIAN_FIX_DATUM


def plan_priors(
    frames: Path,
    *,
    requested: str,
    engine: str,
    frame: site_frame.SiteFrame | None,
    frames_meta: Mapping[str, object] | None,
    std_m: float | None,
    std_v_m: float | None,
    min_fixes: int = 3,
) -> PriorPlan:
    """Which priors pose hands bundle adjustment, or why none (`used` False, `reason`).

    Refused, in this order: `pose_priors: off`; COLMAP 3.9.1 (no pose-prior BA through
    its CLI here); fewer than `min_fixes` EXIF fixes."""
    if requested not in REQUESTS:
        raise ValueError(f"pose_priors must be one of {', '.join(REQUESTS)}, not {requested!r}")
    for name, value in (("prior_std_m", std_m), ("prior_std_v_m", std_v_m)):
        if value is not None and _positive(value) is None:
            raise ValueError(f"{name} must be a positive number of metres, not {value!r}")
    if requested == "off":
        return PriorPlan(requested, False, "pose_priors: off", None, None, None)
    if engine != "4.2":
        return PriorPlan(
            requested,
            False,
            "COLMAP 3.9.1 has no pose-prior bundle adjustment; priors need colmap: 4.2",
            None,
            None,
            None,
        )
    fixes = exif.read_fixes(frames)
    if len(fixes) < min_fixes:
        return PriorPlan(
            requested,
            False,
            f"{len(fixes)} EXIF fixes, fewer than {min_fixes}",
            None,
            None,
            None,
        )
    origin, origin_source, datum = frame_origin(fixes, frame)
    positions = exif.enu_offsets(fixes, origin)
    records = _records(frames_meta)
    priors: list[FramePrior] = []
    for fix in sorted(fixes, key=lambda f: f.name):
        record = records.get(fix.name)
        horizontal, vertical, source = sigma_for(record, std_m, std_v_m)
        unconstrained: str | None = None
        if fix.alt is None:
            unconstrained = "a fix with no altitude"
        elif _fix_datum(record) != datum:
            unconstrained = f"fix altitude is {_fix_datum(record)}, the frame is {datum}"
        priors.append(
            FramePrior(
                name=fix.name,
                position=positions[fix.name],
                std_h_m=horizontal,
                std_v_m=UNCONSTRAINED_STD_M if unconstrained else vertical,
                source=source,
                unconstrained=unconstrained,
            )
        )
    sources = sorted({p.source for p in priors})
    reason = f"{len(priors)} EXIF fixes; sigma from {', '.join(sources)}"
    return PriorPlan(requested, True, reason, origin, origin_source, datum, tuple(priors))
