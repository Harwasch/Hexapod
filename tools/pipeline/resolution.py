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
  a grainy low-light 4K clip has everywhere. A frame that is textured edge to edge reads
  a floor that is detail rather than noise, and so asks more of its tiles: conservative,
  in the direction the cost says to be;
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


def measure_detail(grey: Grey, base: int = BASE) -> DetailMeasure:
    """The share of `grey`'s tiles whose above-`base` band stands out of the noise."""
    residual = band_residual(grey, base)
    height, width = residual.shape
    rows, cols = height // TILE_PX, width // TILE_PX
    if rows == 0 or cols == 0:
        return DetailMeasure(0.0, 0.0, 0.0)
    tiles = residual[: rows * TILE_PX, : cols * TILE_PX].reshape(rows, TILE_PX, cols, TILE_PX)
    rms = np.sqrt((tiles.astype(np.float64) ** 2).mean(axis=(1, 3))).ravel()
    noise = max(NOISE_FLOOR_MIN, float(np.percentile(rms, NOISE_PERCENTILE)))
    needed = max(DETAIL_SNR * noise, DETAIL_MIN_RMS)
    share = float((rms >= needed).mean())
    return DetailMeasure(share, noise, float(np.median(rms)))


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
    frames = tuple(measure_detail(grey, base) for grey in greys)
    share = float(np.median([frame.detail_share for frame in frames]))
    if share >= DETAIL_SHARE:
        reason = (
            f"{share:.0%} of the picture (median of {len(frames)} sharp frames) carries detail "
            f"above {base} px at >= {DETAIL_SNR:g}x its noise floor; {DETAIL_SHARE:.0%} earns "
            f"{side} px"
        )
        return SizeDecision(side, source_side, side, reason, share, frames)
    reason = (
        f"only {share:.0%} of the picture (median of {len(frames)} sharp frames) carries "
        f"detail above {base} px at >= {DETAIL_SNR:g}x its noise floor; {DETAIL_SHARE:.0%} "
        f"would have earned {side} px"
    )
    return SizeDecision(base, source_side, side, reason, share, frames)


def sample_count(available: int) -> int:
    """How many of the sampled frames are measured: the sharpest half, at least one."""
    return max(1, math.ceil(min(available, SAMPLES) / 2))
