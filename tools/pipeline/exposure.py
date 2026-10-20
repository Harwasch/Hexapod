"""Exposure across a capture's kept frames: measured, flagged, and failed only when asked.

What is measured, per kept frame (the 8-bit frame training reads, after any tonemap and
shrink): the mean luma, the percentage of pixels at or above 250 (blown highlights) and
at or below 5 (crushed shadows), and the mean of linear light (`(v/255)^2.2`). Per photo,
also the camera's own EV100 from its EXIF (`frame_meta.ev100`).

The flag thresholds are **first principles, not measurements** (`BASIS`), and flags are
information: they never fail a run. The one gate -- the fraction of frames clipped past
`clip_pct` -- is off unless a recipe sets `max_clipped_frames_fraction`, because no run
has measured where a clipped fraction starts to cost a reconstruction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

import video
from frame_meta import FrameRecord

__all__ = ["BASIS", "HIGH", "LOW", "clipped_frames", "linear_mean", "luma_stats", "summarise"]

#: A pixel at or above this 8-bit level is clipped highlight, at or below `LOW` clipped
#: shadow: within 5 levels of the ends, where JPEG ringing still reaches but no texture
#: SIFT can match survives.
HIGH = 250
LOW = 5

BASIS = (
    "first principles, not measured: a frame more than clippedPct% saturated has that "
    "much area where SIFT and the trainer can recover nothing; one stop (EV) is a "
    "doubling of recorded brightness for the same surface, which `compensate: none` "
    "cannot explain except as wrong colour or geometry; a brightness swing can also be "
    "scene content (walking from shade into sun), so it is informational"
)


def luma_stats(grey: npt.NDArray[Any]) -> tuple[float, float, float]:
    """`(mean, % pixels >= HIGH, % pixels <= LOW)` of an 8-bit grey frame."""
    values = np.asarray(grey)
    if values.size == 0:
        return 0.0, 0.0, 0.0
    total = float(values.size)
    return (
        float(values.mean()),
        100.0 * float(np.count_nonzero(values >= HIGH)) / total,
        100.0 * float(np.count_nonzero(values <= LOW)) / total,
    )


def linear_mean(grey: npt.NDArray[Any]) -> float:
    """Mean linear light of an 8-bit grey frame, by a 2.2 gamma: what a stop is about."""
    values = np.asarray(grey, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.mean((values / 255.0) ** 2.2))


def clipped_frames(records: Sequence[FrameRecord], clip_pct: float) -> list[str]:
    """The frames more than `clip_pct`% clipped at either end, in frame order."""
    return [
        r.name
        for r in records
        if any(
            pct is not None and pct > clip_pct
            for pct in (r.exposure.clipped_high_pct, r.exposure.clipped_low_pct)
        )
    ]


def summarise(
    records: Sequence[FrameRecord],
    linear_means: Mapping[str, float],
    *,
    clip_pct: float,
    ev_spread: float,
    luma_spread_stops: float,
) -> dict[str, object]:
    """The capture's exposure: per-frame statistics summarised, and the flags they raise.

    `linear_means` is keyed by frame name, for the frames that could be measured.
    """
    lumas = [r.exposure.luma_mean for r in records if r.exposure.luma_mean is not None]
    highs = [r.exposure.clipped_high_pct for r in records]
    lows = [r.exposure.clipped_low_pct for r in records]
    evs = [value for r in records if (value := r.exposure.ev100) is not None]
    spread = round(max(evs) - min(evs), 4) if len(evs) >= 2 else None
    linear = [v for v in linear_means.values() if v > 0 and math.isfinite(v)]
    stops = round(math.log2(max(linear) / min(linear)), 4) if len(linear) >= 2 else None
    clipped_high = [
        r.name for r, pct in zip(records, highs, strict=True) if pct is not None and pct > clip_pct
    ]
    clipped_low = [
        r.name for r, pct in zip(records, lows, strict=True) if pct is not None and pct > clip_pct
    ]
    return {
        "lumaMean": video.summarise(lumas),
        "clippedHighPct": video.summarise([p for p in highs if p is not None]),
        "clippedLowPct": video.summarise([p for p in lows if p is not None]),
        "ev100": video.summarise(evs) if evs else None,
        "evSpread": spread,
        "lumaSpreadStops": stops,
        "flags": {
            "clippedHighlights": clipped_high,
            "clippedShadows": clipped_low,
            "evSwing": spread is not None and spread > ev_spread,
            "brightnessSwing": stops is not None and stops > luma_spread_stops,
        },
        "flaggedFrames": len(clipped_frames(records, clip_pct)),
        "measuredFrames": len(lumas),
        "thresholds": {
            "clippedPct": clip_pct,
            "evSpread": ev_spread,
            "lumaSpreadStops": luma_spread_stops,
            "basis": BASIS,
        },
    }
