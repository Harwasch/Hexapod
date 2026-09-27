"""Keyframes by how far the camera moved, not by how long the clip ran.

`select: viewpoint`, the recipe's choice for a video. The time-based rule it replaces
took frames at a fixed rate and then cut them to a count, which gets both ends of a
capture wrong: a two-minute walk round a building is thinned to ~1.5 frames a second
while a slow close-up keeps near-duplicates. The one change of nine that beat run-to-run
noise on the spool-table capture (22 s, 2026-09-27) was *more frames* -- 87 -> 173,
LPIPS 0.174 -> 0.148, SSIM 0.773 -> 0.838 -- so the number of frames is the lever, and
it should follow how much new view the video holds.

How motion is measured, cheaply and with numpy alone (normalize runs on the Fly worker,
whose image has numpy and Pillow and no OpenCV -- and a new wheel there is a new wheel in
both Modal images too, since they import every stage):

* each candidate is decoded once, to grey, and shrunk to `ANALYSIS_SIDE` px on its long
  side. Consecutive candidates are compared by **phase correlation** -- first the whole
  frame, for the dominant shift, then a grid of `TILE` px tiles searched about that
  shift. Phase correlation is brightness-invariant and needs no feature detector; its
  spectrum is weighted towards coarse structure (`BAND_SIGMA`), which motion blur and
  compression leave intact, so a blurred frame still matches its sharp neighbour; a
  tile's correlation peak (0..1) says how much to trust it, and weak tiles (sky, a blank
  wall) are dropped rather than guessed;
* the tiles' shifts are fitted with a **homography** by iteratively reweighted DLT. A
  homography is exactly the motion of a camera that only rotates (or of a plane), so it
  is what "the view slid, turned or zoomed" means;
* **content motion** is how much of the view has changed since the window began: the
  window-start frame's rectangle carried through the composed homographies, and its
  overlap with the current frame (intersection over the larger of the two areas, so
  zooming in and zooming out both count);
* **parallax** is what the homography could *not* explain -- near things sliding past
  far ones, which only a camera that changed position produces. The per-tile residuals
  are summed as vectors, so the back-and-forth of a hand's tremor cancels while a real
  change of viewpoint (whose parallax keeps its direction) grows; the window's parallax
  is the 75th percentile of those sums, not the median, because in a scene with a
  dominant plane the homography fits that plane and the parallax lives in the rest.

Windows and thresholds. A window closes as soon as either measure reaches its budget,
and each window contributes its *sharpest* candidate -- A0's variance-of-Laplacian rank,
within the window, never an absolute cutoff (A0 #6: a 101x within-clip range). Because
the sharpest can sit anywhere in its window, two neighbouring keyframes are typically one
window apart and at most two. So each budget is half the worst gap to be tolerated:

* **overlap**: `overlap = 0.9` closes a window when 10% of the view has changed.
  Neighbouring keyframes then share ~90% of their view, and never less than ~80% --
  the 70-80% SfM front ends ask of sequential neighbours, met in the worst case rather
  than the typical one, because training wants more views of each surface than pose
  needs to chain them (each point stays in ~10 consecutive keyframes of a sweep);
* **parallax**: `parallax = 0.005` of the long side per window. For a camera circling a
  subject at distance D whose depth extent is about +/-0.4 D (a table filling the frame,
  a building from across the street), a 1 deg change of viewpoint shifts the near and far
  surfaces apart by roughly 0.3 x the image width x 0.0175 rad = 0.5% of it. Checked on
  Tanks and Temples' truck, 251 frames round it ~1.43 deg apart: a median 0.63% of the
  long side a frame, 0.44% a degree. So one window is ~1 deg of viewpoint: ~300-360
  keyframes round a full orbit before the ceiling, a little denser than the 150-300
  images of the Mip-NeRF 360 scenes 3DGS was tuned on, and the density at which the
  spool (8 fps: ~1 deg a frame if it was a half orbit in 22 s) measured better than
  4 fps did.

A frame that cannot be matched (motion blur, a hand across the lens) is held and the
next one matched across it, up to `MAX_HELD` in a row. A gap that still cannot be
bridged (a whip pan, a lens cap, a scene cut) counts as a whole window -- when unsure,
keep a frame rather than leave a gap pose cannot bridge.

The ceiling. When a capture's windows outnumber the ceiling (`keep_video`), every
budget is scaled up by the same factor, found by bisection, until they fit -- so the
keyframes are spread evenly *by motion*, which thinning by time would not do: a slow
stretch would keep as many frames as a fast one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

import video

__all__ = [
    "ANALYSIS_SIDE",
    "TILE",
    "Analysis",
    "MotionTracker",
    "PairMotion",
    "Segmentation",
    "Window",
    "analyse",
    "analysis_frame",
    "neighbour_motion",
    "overlap_of",
    "segment",
    "segment_within",
    "select_per_window",
]

Grey = npt.NDArray[np.float32]
Matrix = npt.NDArray[np.float64]

#: Long side, in pixels, motion is measured at. Phase correlation's precision is ~0.1 px
#: on a textured tile, so at 480 px the parallax budget (0.5% of it, 2.4 px) is ~25x the
#: noise of one pair and ~2.5x the sqrt-growing noise of a hundred. Measured: ~50-60 ms a
#: candidate on the (shared) development container, most of it the tiles' FFTs.
ANALYSIS_SIDE = 480
#: Tile edge for the local shifts, stepped by half a tile. 64 px is 13% of the long side:
#: large enough to carry texture and to find shifts up to ~20 px on top of the frame's
#: dominant one, small enough that a 480x270 frame gives ~80 tiles to fit 8 unknowns.
TILE = 64
#: The spread, in cycles per pixel, of the Gaussian that weights the correlation's
#: spectrum (1.0 is a perfect match under it), and the peak a tile must reach to be used.
#: Measured on 64 px tiles: unrelated real content (NeoVerse's clips against a random
#: texture) peaks at p99 0.39, unrelated random textures at p99 0.57; consecutive frames of
#: a real clip at a median 0.90 (p10 0.48), a pan whose every other frame is blurred
#: (Gaussian, 3 px) at 0.78 (p10 0.75), where the unweighted correlation that preceded
#: this peaked at 0.09 -- below its noise, so every blurred frame was unmeasurable.
BAND_SIGMA = 0.07
MIN_PEAK = 0.5
#: A tile whose grey standard deviation is below this carries nothing to correlate.
MIN_TILE_STD = 3.0
#: Fewer confident tiles than this and the pair counts as unmeasured. Eight unknowns in
#: a homography, and the reweighting needs spare equations to reject any.
MIN_TILES = 12
#: Frames in a row that may be held (bridged over) before a gap counts as unmeasured.
#: Three at 15 fps is a fifth of a second of blur or occlusion.
MAX_HELD = 3
#: The parallax statistic: the upper quartile of the tiles' accumulated residuals.
PARALLAX_PERCENTILE = 75.0
#: A last window below this fraction of a budget is folded into the one before it rather
#: than adding a keyframe a few candidates after the previous one.
TAIL_FRACTION = 0.5


def analysis_frame(image: Image.Image, side: int = ANALYSIS_SIDE) -> Grey:
    """The grey, `side`-px frame motion is measured on. Float32, 0..255."""
    grey = image.convert("L")
    width, height = grey.size
    scale = side / max(width, height)
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        grey = grey.resize(size, Image.Resampling.BILINEAR, reducing_gap=2.0)
    return np.asarray(grey, dtype=np.float32)


@dataclass(frozen=True)
class PairMotion:
    """How one candidate moved relative to the last one that was measured.

    Usually that is the candidate just before it. `homography` maps that frame's analysis
    pixels (x, y, 1) onto this one's; `residuals` is (rows, cols, 2): each tile's shift
    the homography did not explain, NaN where the tile was not trusted. `measured` is
    False when too few tiles were -- the homography is then the identity and says
    nothing. `held` marks a frame that could not be matched but was bridged: the next
    candidate was matched across it, so its motion is carried by that one's pair.
    """

    homography: Matrix
    residuals: npt.NDArray[np.float64]
    tiles: int
    measured: bool
    held: bool = False


@dataclass(frozen=True)
class Window:
    """A run of candidates within one motion budget of its first, [start, end]."""

    start: int
    end: int
    #: The window's motion when it closed, in budgets: >= 1 for every window but the
    #: last, which ran out of clip instead. Which budget closed it is `closed_by`.
    motion: float
    closed_by: str


@dataclass(frozen=True)
class Segmentation:
    """The windows, the scale they were found at, and whether the ceiling set it."""

    windows: tuple[Window, ...]
    scale: float
    ceiling: int | None
    bound: bool
    unmeasured_pairs: int
    #: How many windows there were before the ceiling scaled the budgets.
    uncapped_windows: int = 0


@dataclass(frozen=True)
class Analysis:
    """Every candidate's sharpness and its motion from the one before, in order."""

    scores: tuple[float, ...]
    pairs: tuple[PairMotion | None, ...]
    #: The analysis frames' (width, height): what homographies and budgets are in.
    frame_size: tuple[int, int]


def analyse(paths: Sequence[Path]) -> Analysis:
    """One decode per candidate: sharpness at full size, motion at `ANALYSIS_SIDE`.

    Sequential and one frame in memory at a time, because this runs on the worker (2 GB)
    over as many as `max_candidates` frames.
    """
    tracker = MotionTracker()
    scores: list[float] = []
    pairs: list[PairMotion | None] = []
    frame_size = (0, 0)
    for path in paths:
        with Image.open(path) as image:
            grey = image.convert("L")
            scores.append(video.laplacian_variance(np.asarray(grey, dtype=np.float64)))
            frame = analysis_frame(grey)
        frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        pairs.append(tracker.push(frame))
    return Analysis(tuple(scores), tuple(pairs), frame_size)


class MotionTracker:
    """Feed it analysis frames in order; it returns each one's motion from the last.

    A frame that cannot be matched to the last measured one (a burst of motion blur, a
    hand across the lens) is *held*: the reference stays where it was and the next frame
    is matched across the gap, up to `MAX_HELD` frames. Only when the gap cannot be
    bridged does a pair come back unmeasured, and the reference moves on.
    """

    def __init__(self, tile: int = TILE) -> None:
        self._tile = tile
        self._reference: Grey | None = None
        self._held = 0
        self._window_cache: dict[tuple[int, int], npt.NDArray[np.float32]] = {}

    def push(self, frame: Grey) -> PairMotion | None:
        """None for the first frame, and for a frame whose size changed (a new clip)."""
        reference = self._reference
        if reference is None or reference.shape != frame.shape:
            self._reference, self._held = frame, 0
            return None
        motion = pair_motion(reference, frame, tile=self._tile, cache=self._window_cache)
        if motion.measured:
            self._reference, self._held = frame, 0
            return motion
        if self._held < MAX_HELD:
            self._held += 1
            return PairMotion(motion.homography, motion.residuals, motion.tiles, False, True)
        # The gap is too long to bridge: count it, and measure on from here.
        self._reference, self._held = frame, 0
        return motion


def pair_motion(
    before: Grey,
    after: Grey,
    *,
    tile: int = TILE,
    cache: dict[tuple[int, int], npt.NDArray[np.float32]] | None = None,
) -> PairMotion:
    """The homography from `before` to `after` and what it leaves unexplained."""
    cache = {} if cache is None else cache
    height, width = before.shape
    step = tile // 2
    rows = max(0, (height - tile) // step + 1)
    cols = max(0, (width - tile) // step + 1)
    residuals = np.full((rows, cols, 2), np.nan)
    identity = np.eye(3)
    if rows == 0 or cols == 0:
        return PairMotion(identity, residuals, 0, False)

    # The dominant shift first, over the whole frame, so each tile searches about it
    # rather than about zero -- a pan of 10% of the frame between candidates is 48 px
    # here, beyond what a 64 px tile can find on its own.
    # At half resolution: only its whole-pixel part is used, and it is a quarter the work.
    small_before, small_after = _half(before), _half(after)
    (gy, gx), _ = _phase_shift(small_before, small_after, _hann(small_before.shape, cache))
    oy, ox = round(2.0 * gy), round(2.0 * gx)

    window = _hann((tile, tile), cache)
    centres: list[tuple[float, float]] = []
    flows: list[tuple[float, float]] = []
    cells: list[tuple[int, int]] = []
    firsts: list[Grey] = []
    seconds: list[Grey] = []
    for r in range(rows):
        y = r * step
        if not 0 <= y + oy <= height - tile:
            continue
        for c in range(cols):
            x = c * step
            if not 0 <= x + ox <= width - tile:
                continue
            a = before[y : y + tile, x : x + tile]
            if float(a.std()) < MIN_TILE_STD:
                continue
            firsts.append(a)
            seconds.append(after[y + oy : y + oy + tile, x + ox : x + ox + tile])
            centres.append((x + tile / 2.0, y + tile / 2.0))
            cells.append((r, c))
    if len(firsts) >= MIN_TILES:
        shifts, peaks = _phase_shifts(np.stack(firsts), np.stack(seconds), window)
        for (dy, dx), peak in zip(shifts, peaks, strict=True):
            flows.append((ox + dx, oy + dy) if peak >= MIN_PEAK else (math.nan, math.nan))
    good = [i for i, flow in enumerate(flows) if not math.isnan(flow[0])]
    if len(good) < MIN_TILES:
        return PairMotion(identity, residuals, len(good), False)

    source = np.array([centres[i] for i in good], dtype=np.float64)
    target = source + np.array([flows[i] for i in good], dtype=np.float64)
    homography = fit_homography(source, target)
    if homography is None or not np.all(np.isfinite(homography)):
        return PairMotion(identity, residuals, len(good), False)
    predicted = _apply(homography, source)
    for index, (r, c) in enumerate(cells[i] for i in good):
        residuals[r, c] = target[index] - predicted[index]
    return PairMotion(homography, residuals, len(good), True)


def fit_homography(
    source: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    *,
    iterations: int = 6,
    scale_px: float = 0.75,
) -> Matrix | None:
    """A homography from point pairs, robust to a minority of bad or off-plane ones.

    Normalised DLT (Hartley), then iteratively reweighted with Cauchy weights on each
    point's residual -- deterministic, unlike RANSAC, so the same clip always gives the
    same keyframes. `scale_px` is the residual (analysis pixels) at which a point's
    weight has fallen to 1/sqrt(2): a few times phase correlation's precision, so noise is
    kept and a tile on a nearer or farther surface is down-weighted, which leaves the fit
    on the dominant plane and the parallax in the residuals where it is measured.
    """
    if len(source) < 4:
        return None
    src_t, src_n = _normalise(source)
    dst_t, dst_n = _normalise(target)
    weights = np.ones(len(source))
    homography: Matrix | None = None
    for _ in range(iterations):
        rows = _dlt_rows(src_n, dst_n) * np.repeat(weights, 2)[:, None]
        try:
            _, _, vt = np.linalg.svd(rows, full_matrices=False)
        except np.linalg.LinAlgError:
            return homography
        normalised = vt[-1].reshape(3, 3)
        candidate = np.linalg.inv(dst_t) @ normalised @ src_t
        if abs(candidate[2, 2]) < 1e-12:
            return homography
        homography = candidate / candidate[2, 2]
        residual = np.linalg.norm(target - _apply(homography, source), axis=1)
        weights = 1.0 / np.sqrt(1.0 + (residual / scale_px) ** 2)
    return homography


def overlap_of(homography: Matrix, width: float, height: float) -> float:
    """How much of the view two frames share, 0..1.

    The first frame's rectangle carried by `homography` into the second's pixels,
    intersected with the second's rectangle, over the larger of the two areas -- so a
    zoom in and a zoom out by the same factor lose the same overlap, and a pan loses
    exactly the fraction that left the frame. A homography that folds the rectangle over
    (a degenerate fit) shares nothing.
    """
    corners = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    ones = np.ones((4, 1))
    projected = (homography @ np.hstack([corners, ones]).T).T
    if np.any(projected[:, 2] <= 1e-9):
        return 0.0
    quad = projected[:, :2] / projected[:, 2:3]
    area = _polygon_area(quad)
    if area <= 0:
        return 0.0
    inside = _clip_to_rectangle([(float(x), float(y)) for x, y in quad], width, height)
    shared = _polygon_area(np.array(inside)) if len(inside) >= 3 else 0.0
    return float(max(0.0, min(1.0, shared / max(area, width * height))))


def segment(
    pairs: Sequence[PairMotion | None],
    *,
    frame_size: tuple[int, int],
    overlap: float,
    parallax: float,
    ceiling: int | None = None,
) -> Segmentation:
    """Windows of one motion budget each; scaled evenly by motion to fit a ceiling.

    `pairs[i]` is candidate i's motion from candidate i-1 (`pairs[0]` is None). A None
    anywhere else counts as unmeasured, like a pair `measured` says was.
    """
    windows = segment_within(pairs, frame_size=frame_size, overlap=overlap, parallax=parallax)
    unmeasured = sum(1 for p in pairs[1:] if p is None or not (p.measured or p.held))
    if ceiling is None or len(windows) <= ceiling:
        return Segmentation(windows, 1.0, ceiling, False, unmeasured, uncapped_windows=len(windows))

    # Bisection on one scale for every budget. The count of windows is non-increasing in
    # the scale (a bigger budget never closes a window sooner), so the smallest scale
    # whose count fits is the densest spread the ceiling allows, even in motion.
    def at(scale: float) -> tuple[Window, ...]:
        return segment_within(
            pairs, frame_size=frame_size, overlap=overlap, parallax=parallax, scale=scale
        )

    low, high = 1.0, 2.0
    best = at(high)
    # Doubling ends: at a large enough scale the whole clip is one window.
    while len(best) > ceiling:
        low, high = high, high * 2.0
        best = at(high)
    for _ in range(40):
        middle = 0.5 * (low + high)
        trial = at(middle)
        if len(trial) <= ceiling:
            high, best = middle, trial
        else:
            low = middle
        if high - low < 1e-3 * high:
            break
    return Segmentation(best, high, ceiling, True, unmeasured, uncapped_windows=len(windows))


def segment_within(
    pairs: Sequence[PairMotion | None],
    *,
    frame_size: tuple[int, int],
    overlap: float,
    parallax: float,
    scale: float = 1.0,
) -> tuple[Window, ...]:
    """The greedy segmentation at one scale: a window closes when a budget is spent.

    Budgets, each multiplied by `scale`: `1 - overlap` of the view changed since the
    window's first candidate; `parallax` of the long side as the upper-quartile tile's
    accumulated residual; and one unmeasured pair.
    """
    count = len(pairs)
    if count == 0:
        return ()
    width, height = frame_size
    long_side = float(max(width, height))
    content_budget = max(1e-6, 1.0 - overlap) * scale
    parallax_budget = max(1e-6, parallax * long_side) * scale
    windows: list[Window] = []
    start = 0
    state = _WindowState.fresh()
    motion, closed_by = 0.0, "end"
    for index in range(1, count):
        pair = pairs[index]
        state.add(pair)
        content = (1.0 - overlap_of(state.homography, width, height)) / content_budget
        drift = state.parallax() / parallax_budget
        unknown = state.unmeasured / scale
        motion, closed_by = max((content, "overlap"), (drift, "parallax"), (unknown, "unmeasured"))
        if motion >= 1.0:
            windows.append(Window(start, index - 1, motion, closed_by))
            start = index
            state = _WindowState.fresh()
            motion, closed_by = 0.0, "end"
    if windows and motion < TAIL_FRACTION and windows[-1].closed_by != "unmeasured":
        # A sliver of clip after the last full window: fold it in rather than keep a
        # near-duplicate of the keyframe just before it. Not across an unmeasured pair,
        # though: how far the sliver is from the window before it is exactly what is
        # not known.
        last = windows.pop()
        windows.append(Window(last.start, count - 1, last.motion, last.closed_by))
    else:
        windows.append(Window(start, count - 1, motion, "end"))
    return tuple(windows)


def select_per_window(scores: Sequence[float], windows: Iterable[Window]) -> tuple[int, ...]:
    """The sharpest candidate of each window, in order; ties break on the earlier one.

    The same rank A0 measured (variance-of-Laplacian, Spearman +0.979 against blur),
    taken within a window of motion rather than a window of time or the whole clip. No
    cutoff: a window of nothing but blurred frames still gives its least blurred one,
    because a gap in the viewpoints is worse for pose than a soft frame.
    """
    return tuple(max(range(w.start, w.end + 1), key=lambda i: (scores[i], -i)) for w in windows)


def neighbour_motion(
    pairs: Sequence[PairMotion | None],
    chosen: Sequence[int],
    *,
    frame_size: tuple[int, int],
) -> tuple[list[float], list[float]]:
    """The overlap and parallax (fraction of the long side) between consecutive keyframes.

    What the budgets promise, measured on what was actually kept: the numbers that go in
    `source_meta.json` so a run can be checked against "~90% overlap, never under ~80%".
    An unmeasured pair inside a gap makes that gap's numbers NaN-free but optimistic, so
    the stage records the unmeasured count beside them.
    """
    width, height = frame_size
    long_side = float(max(width, height))
    overlaps: list[float] = []
    drifts: list[float] = []
    for a, b in pairwise(chosen):
        state = _WindowState.fresh()
        for index in range(a + 1, b + 1):
            state.add(pairs[index])
        overlaps.append(overlap_of(state.homography, width, height))
        drifts.append(state.parallax() / long_side)
    return overlaps, drifts


# --- internals ---------------------------------------------------------------------------


@dataclass
class _WindowState:
    """Motion accumulated since a window's first candidate."""

    homography: Matrix
    residual_sum: npt.NDArray[np.float64] | None
    seen: npt.NDArray[np.bool_] | None
    unmeasured: int

    @classmethod
    def fresh(cls) -> _WindowState:
        return cls(np.eye(3), None, None, 0)

    def add(self, pair: PairMotion | None) -> None:
        if pair is not None and pair.held:
            return  # its motion arrives with the pair that bridged it
        if pair is None or not pair.measured:
            self.unmeasured += 1
            return
        self.homography = pair.homography @ self.homography
        valid = ~np.isnan(pair.residuals[..., 0])
        step = np.where(valid[..., None], pair.residuals, 0.0)
        if self.residual_sum is None or self.residual_sum.shape != step.shape:
            self.residual_sum = step.copy()
            self.seen = valid.copy()
        else:
            self.residual_sum += step
            assert self.seen is not None
            self.seen |= valid

    def parallax(self) -> float:
        if self.residual_sum is None or self.seen is None or not self.seen.any():
            return 0.0
        magnitudes = np.linalg.norm(self.residual_sum[self.seen], axis=-1)
        return float(np.percentile(magnitudes, PARALLAX_PERCENTILE))


def _hann(
    shape: tuple[int, ...], cache: dict[tuple[int, int], npt.NDArray[np.float32]]
) -> npt.NDArray[np.float32]:
    key = (int(shape[0]), int(shape[1]))
    window = cache.get(key)
    if window is None:
        window = np.outer(np.hanning(key[0]), np.hanning(key[1])).astype(np.float32)
        cache[key] = window
    return window


def _half(frame: Grey) -> Grey:
    """2x2 block means: half the resolution, for the dominant shift."""
    height, width = (frame.shape[0] // 2) * 2, (frame.shape[1] // 2) * 2
    blocks = frame[:height, :width].reshape(height // 2, 2, width // 2, 2)
    return blocks.mean(axis=(1, 3), dtype=np.float32)


def _phase_shift(
    before: Grey, after: Grey, window: npt.NDArray[np.float32]
) -> tuple[tuple[float, float], float]:
    shifts, peaks = _phase_shifts(before[None], after[None], window)
    return shifts[0], peaks[0]


def _phase_shifts(
    before: npt.NDArray[np.float32],
    after: npt.NDArray[np.float32],
    window: npt.NDArray[np.float32],
) -> tuple[list[tuple[float, float]], list[float]]:
    """Batched phase correlation: how far each `after` tile's content moved from `before`.

    Returns (dy, dx) per tile, sub-pixel by a parabola through the peak and its
    neighbours, and the peak's height (1.0 for a pure shift of identical content, near
    zero for noise). Means are removed and a Hann window applied first, so the tile's
    border and its brightness do not correlate with themselves.
    """
    a = before - before.mean(axis=(1, 2), keepdims=True)
    b = after - after.mean(axis=(1, 2), keepdims=True)
    fa = np.fft.rfft2(a * window)
    fb = np.fft.rfft2(b * window)
    cross = fb * np.conj(fa)
    cross /= np.abs(cross) + 1e-9
    weights, unit = _band(before.shape[1], before.shape[2])
    surface = np.fft.irfft2(cross * weights, s=before.shape[1:]) / unit
    count, height, width = surface.shape
    peaks_at = surface.reshape(count, -1).argmax(axis=1)
    py, px = np.divmod(peaks_at, width)
    tiles = np.arange(count)
    centre = surface[tiles, py, px].astype(np.float64)
    dy = _parabola(
        surface[tiles, (py - 1) % height, px].astype(np.float64),
        centre,
        surface[tiles, (py + 1) % height, px].astype(np.float64),
    )
    dx = _parabola(
        surface[tiles, py, (px - 1) % width].astype(np.float64),
        centre,
        surface[tiles, py, (px + 1) % width].astype(np.float64),
    )
    y = py + dy
    x = px + dx
    # The correlation surface wraps: a peak past the middle is a negative shift.
    y = np.where(y > height / 2, y - height, y)
    x = np.where(x > width / 2, x - width, x)
    shifts = [(float(a), float(b)) for a, b in zip(y, x, strict=True)]
    return shifts, [float(c) for c in centre]


_BANDS: dict[tuple[int, int], tuple[npt.NDArray[np.float32], float]] = {}


def _band(height: int, width: int) -> tuple[npt.NDArray[np.float32], float]:
    """The spectral weighting, and the peak a perfect match reaches under it."""
    key = (height, width)
    cached = _BANDS.get(key)
    if cached is None:
        fy = np.fft.fftfreq(height)[:, None]
        fx = np.fft.rfftfreq(width)[None, :]
        weights = np.exp(-(fx**2 + fy**2) / (2.0 * BAND_SIGMA**2)).astype(np.float32)
        unit = float(np.fft.irfft2(weights.astype(np.complex64), s=(height, width))[0, 0])
        cached = (weights, unit)
        _BANDS[key] = cached
    return cached


def _parabola(
    left: npt.NDArray[np.float64], centre: npt.NDArray[np.float64], right: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The sub-pixel offset of a peak from its two neighbours, within half a pixel."""
    denominator = left - 2.0 * centre + right
    safe = np.where(np.abs(denominator) < 1e-12, 1.0, denominator)
    offset = np.where(np.abs(denominator) < 1e-12, 0.0, 0.5 * (left - right) / safe)
    return np.clip(offset, -0.5, 0.5)


def _normalise(points: npt.NDArray[np.float64]) -> tuple[Matrix, npt.NDArray[np.float64]]:
    centroid = points.mean(axis=0)
    spread = float(np.sqrt(((points - centroid) ** 2).sum(axis=1)).mean())
    scale = math.sqrt(2.0) / spread if spread > 1e-12 else 1.0
    transform = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0, 0, 1]]
    )
    return transform, (points - centroid) * scale


def _dlt_rows(
    source: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    x, y = source[:, 0], source[:, 1]
    u, v = target[:, 0], target[:, 1]
    zeros, ones = np.zeros_like(x), np.ones_like(x)
    first = np.stack([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u], axis=1)
    second = np.stack([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v], axis=1)
    rows = np.empty((2 * len(x), 9))
    rows[0::2], rows[1::2] = first, second
    return rows


def _apply(homography: Matrix, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ homography.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _polygon_area(points: npt.NDArray[np.float64]) -> float:
    """Shoelace; positive for either winding, so a folded quad reads as small, not negative."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_to_rectangle(
    polygon: list[tuple[float, float]], width: float, height: float
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman against [0, width] x [0, height]."""
    output = polygon
    # (axis, bound, sign): a point is inside when sign * (point[axis] - bound) >= 0.
    for axis, bound, sign in ((0, 0.0, 1.0), (0, width, -1.0), (1, 0.0, 1.0), (1, height, -1.0)):
        points, output = output, []
        if not points:
            break
        previous = points[-1]
        for current in points:
            current_in = sign * (current[axis] - bound) >= 0.0
            previous_in = sign * (previous[axis] - bound) >= 0.0
            if current_in != previous_in:
                output.append(_crossing(previous, current, axis, bound))
            if current_in:
                output.append(current)
            previous = current
    return output


def _crossing(
    p: tuple[float, float], q: tuple[float, float], axis: int, bound: float
) -> tuple[float, float]:
    """Where the segment p-q crosses the line `point[axis] == bound`."""
    t = (bound - p[axis]) / (q[axis] - p[axis])
    x, y = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
    return (bound, y) if axis == 0 else (x, bound)
