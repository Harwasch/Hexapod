"""The frame size a capture has earned: 1600 px unless it measurably holds more (#6).

`max_side: auto`, the recipe's default. Every frame used to be shrunk to 1600 px on its
long side whatever it came from. For most phone captures that is right -- a 1080p clip
has nothing above it, and a soft or digitally zoomed 4K one has nothing above it either
-- but 4K video of a building across a street, or a folder of 24 MP stills, can carry
real detail past 1600 px that the splat is then trained without. Training at 2400 px
costs about twice the GPU time (2.25x the pixels), so the larger size has to be earned
by measurement, and the rule is written to say no when unsure.

The measurement, on the sharpest half of a handful of frames spread over the capture
(the sharpest by rank, A0's variance-of-Laplacian -- never a blur cutoff), each taken at
the size that would be kept (the source's long side, at most `ceiling`):

* the **band residual**: the frame shrunk to `base` and enlarged back, subtracted from
  itself. What is left is exactly the content between the two sizes' Nyquist limits --
  what training at `base` throws away. An upscaled or blurred source leaves nearly
  nothing, whatever its nominal size;
* its RMS in `TILE_PX` tiles, and a **noise floor**: the 10th percentile of those tiles.
  The quietest tenth of a frame (sky, a wall, a shadow) holds no detail, so what the band
  carries there is sensor noise and compression -- which is also in the band, and which
  a grainy low-light 4K clip has everywhere. The noise is the camera's, so the floor is
  the **capture's**: the lowest of the sampled frames' own (`capture_noise_floor`),
  clipped tiles left out. Per frame, a frame textured edge to edge read its texture as
  noise and a tree filling the picture never earned more than 1600 px;
* a tile **has detail** when its residual is at least `DETAIL_SNR` (3) times the floor --
  9x its energy -- and at least `DETAIL_MIN_RMS` (2.5 grey levels). The absolute part is
  what separates real detail from resampling: an upscaled frame's strong edges still
  leave a residual of 1-2 levels (the upscaler's own overshoot), where fine texture
  leaves 3 and more;
* the larger size is kept when the **median over the sampled frames** of the share of
  such tiles is at least `DETAIL_SHARE` (a quarter of the picture). Less than that is a
  detail here and there -- a sign, a railing -- not a capture that trains better at 2.25x
  the cost.

Measured 2026-09-27 on 8 of the backhoe stills (Sony a6500, 3008x2000, the sharpest 4
measured at 2400): as shot, a median 50% of the picture had detail -> 2400; the same
frames shrunk to 1200 and enlarged back (a 4K file with 1200 px of content) 6% -> 1600;
blurred (Gaussian, sigma 2 px) 0% -> 1600; enlarged from 1200 plus noise of sigma 4
(a grainy low-light clip) 0% -> 1600. With a 2.0-level minimum the enlarged frames read
12% (up to 23% in one frame), too close to the bar, so the minimum is 2.5.

A source whose long side is under `MIN_GAIN` x base (2000 px for 1600) is not measured at
all: 1080p (1920) would gain 20% in width for 44% more training, which is not worth a
measurement that can only say yes to it. The decision and every number it was made from
go in `source_meta.json`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image

__all__ = [
    "BASE",
    "CEILING",
    "DETAIL_SHARE",
    "MIN_GAIN",
    "SAMPLES",
    "DetailMeasure",
    "SizeDecision",
    "band_residual",
    "decide",
    "measure_detail",
    "measured_side",
    "not_measured",
]

Grey = npt.NDArray[np.float32]

#: The size every frame had before this rule, and what pose extracts features at.
BASE = 1600
#: The largest size the rule will choose. 2400 is ~2.25x the pixels and about twice the
#: GPU time of 1600; the phone's explicit "2400 px" is the way past it, not this rule.
CEILING = 2400
#: A source must be this much larger than `base` for the question to be asked.
MIN_GAIN = 1.25
#: Frames sampled across the capture; the sharpest half of them are measured.
SAMPLES = 8
#: Tile edge, in pixels at the measured size, for the residual RMS.
TILE_PX = 32
#: The percentile of tile RMS taken as the noise floor, and the least it may be (a
#: grey level's quantisation step is 1; half of it is the floor of what is measurable).
NOISE_PERCENTILE = 10.0
NOISE_FLOOR_MIN = 0.5
#: A tile whose mean is within this many grey levels of black or white is clipped: its
#: noise went with its signal, so it is no evidence of how quiet the camera is.
CLIPPED_LEVELS = 4.0
#: A tile has detail at this multiple of the floor and at least this RMS (grey levels).
DETAIL_SNR = 3.0
DETAIL_MIN_RMS = 2.5
#: The median share of detailed tiles that earns the larger size.
DETAIL_SHARE = 0.25


@dataclass(frozen=True)
class DetailMeasure:
    """One sampled frame's answer."""

    detail_share: float
    noise_rms: float
    residual_rms: float

    def to_dict(self) -> dict[str, float]:
        return {
            "detailShare": round(self.detail_share, 4),
            "noiseRms": round(self.noise_rms, 3),
            "residualRms": round(self.residual_rms, 3),
        }


@dataclass(frozen=True)
class SizeDecision:
    """The long side chosen, and the measurement (or the reason there was none)."""

    max_side: int
    source_side: int | None
    measured_side: int | None
    reason: str
    detail_share: float | None = None
    frames: tuple[DetailMeasure, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "rule": "auto",
            "maxSide": self.max_side,
            "sourceSide": self.source_side,
            "measuredSide": self.measured_side,
            "reason": self.reason,
            "detailShare": None if self.detail_share is None else round(self.detail_share, 4),
            "detailShareNeeded": DETAIL_SHARE,
            "frames": [frame.to_dict() for frame in self.frames],
        }


def measured_side(
    source_side: int | None, *, base: int = BASE, ceiling: int = CEILING
) -> int | None:
    """The size a capture would be measured (and kept) at, or None if not worth asking."""
    if source_side is None or source_side < MIN_GAIN * base:
        return None
    return min(source_side, ceiling)


def not_measured(source_side: int | None, *, base: int = BASE) -> SizeDecision:
    """The decision for a capture too small to be measured, or of unknown size."""
    if source_side is None:
        reason = f"the source's size is unknown, so {base} px"
    else:
        reason = (
            f"the source's long side ({source_side} px) is under {MIN_GAIN:g}x {base}, so "
            f"there is too little above {base} px to be worth measuring"
        )
    return SizeDecision(base, source_side, None, reason)


def band_residual(grey: Grey, base: int) -> Grey:
    """What shrinking `grey` to `base` on its long side, and back, takes out of it."""
    height, width = grey.shape
    scale = base / max(width, height)
    if scale >= 1.0:
        return np.zeros_like(grey)
    small = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = Image.fromarray(grey.astype(np.float32), mode="F")
    down = image.resize(small, Image.Resampling.LANCZOS)
    up = down.resize((width, height), Image.Resampling.LANCZOS)
    return (grey - np.asarray(up, dtype=np.float32)).astype(np.float32)


def _band_tiles(grey: Grey, base: int) -> tuple[npt.NDArray[np.float64], float]:
    """Each tile's band RMS, and the frame's own noise floor: the `NOISE_PERCENTILE` of
    the tiles that are not clipped. A tile at the ends of the grey scale (a blown-out sky,
    a crushed shadow) has had its noise clipped away with its signal, so it says nothing
    about the noise and is left out of the floor -- it would otherwise read as a camera
    with none."""
    residual = band_residual(grey, base)
    height, width = residual.shape
    rows, cols = height // TILE_PX, width // TILE_PX
    if rows == 0 or cols == 0:
        return np.zeros(0), NOISE_FLOOR_MIN
    shape = (rows, TILE_PX, cols, TILE_PX)
    tiles = residual[: rows * TILE_PX, : cols * TILE_PX].reshape(shape)
    rms = np.sqrt((tiles.astype(np.float64) ** 2).mean(axis=(1, 3))).ravel()
    level = grey[: rows * TILE_PX, : cols * TILE_PX].reshape(shape).mean(axis=(1, 3)).ravel()
    unclipped = rms[(level > CLIPPED_LEVELS) & (level < 255.0 - CLIPPED_LEVELS)]
    pool = unclipped if unclipped.size else rms
    return rms, max(NOISE_FLOOR_MIN, float(np.percentile(pool, NOISE_PERCENTILE)))


def measure_detail(grey: Grey, base: int = BASE, floor: float | None = None) -> DetailMeasure:
    """The share of `grey`'s tiles whose above-`base` band stands out of the noise:
    `floor` when given (the capture's, `decide`), else the frame's own."""
    rms, own = _band_tiles(grey, base)
    if rms.size == 0:
        return DetailMeasure(0.0, 0.0, 0.0)
    noise = own if floor is None else max(NOISE_FLOOR_MIN, floor)
    needed = max(DETAIL_SNR * noise, DETAIL_MIN_RMS)
    share = float((rms >= needed).mean())
    return DetailMeasure(share, noise, float(np.median(rms)))


def capture_noise_floor(greys: Sequence[Grey], base: int = BASE) -> float:
    """The capture's noise floor: the lowest of its sampled frames' own floors.

    Noise is the camera's -- one sensor, one pipeline, one set of settings -- not the
    frame's, but a frame can only show it where it has a quiet region. A frame that is
    textured edge to edge (foliage, a lawn, gravel) has none, and its quietest tenth is
    still texture: measured 2026-09-28 on the Minnetonka tree's photos (a drone orbit,
    5,464 px, the sample `max_side: auto` takes), three of the four sharpest frames read
    floors of 4.1-5.2 grey levels -- all foliage and lawn -- and the fourth, with sky in
    it, 0.55. Per frame, the rule then saw detail in 7% of the picture and kept 1600 px;
    the same band read against the camera's own 0.55 is detail almost everywhere. So the
    floor is taken where the sample shows it best. Clipped tiles are not a quiet region
    (`_band_tiles`), so a blown-out sky cannot stand in for one."""
    floors = [_band_tiles(grey, base)[1] for grey in greys]
    return min(floors) if floors else NOISE_FLOOR_MIN


def decide(
    greys: Sequence[Grey],
    *,
    source_side: int,
    base: int = BASE,
    ceiling: int = CEILING,
) -> SizeDecision:
    """`base`, or the measured size if the sampled frames carry detail past `base`.

    `greys` are the frames to measure, already at the measured size and already the
    sharpest of the sample -- choosing them is the caller's, because the caller knows
    how to get frames out of its source.
    """
    side = measured_side(source_side, base=base, ceiling=ceiling)
    if side is None:
        return not_measured(source_side, base=base)
    if not greys:
        return SizeDecision(
            base, source_side, side, f"no frame could be sampled to measure, so {base} px"
        )
    floor = capture_noise_floor(greys, base)
    frames = tuple(measure_detail(grey, base, floor) for grey in greys)
    share = float(np.median([frame.detail_share for frame in frames]))
    against = f">= {DETAIL_SNR:g}x the capture's noise floor ({floor:.2f} grey levels)"
    if share >= DETAIL_SHARE:
        reason = (
            f"{share:.0%} of the picture (median of {len(frames)} sharp frames) carries detail "
            f"above {base} px at {against}; {DETAIL_SHARE:.0%} earns {side} px"
        )
        return SizeDecision(side, source_side, side, reason, share, frames)
    reason = (
        f"only {share:.0%} of the picture (median of {len(frames)} sharp frames) carries "
        f"detail above {base} px at {against}; {DETAIL_SHARE:.0%} would have earned "
        f"{side} px"
    )
    return SizeDecision(base, source_side, side, reason, share, frames)


def sample_count(available: int) -> int:
    """How many of the sampled frames are measured: the sharpest half, at least one."""
    return max(1, math.ceil(min(available, SAMPLES) / 2))
