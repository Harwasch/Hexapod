"""The arithmetic of held-out accuracy, in numpy alone, so it is tested where it runs.

`holdout_error.py` runs on the training box with the trainer's CPython 3.10 and torch; it
renders each held-out frame and asks the rasteriser which gaussians painted each pixel.
Everything *around* that -- how wrong a pixel is, how a pixel's error is shared out among
the gaussians that painted it, what a gaussian's mean error is, what a frame's PSNR, bias
and sharpness are -- is here, as plain numpy, and is the same code on the GPU box and in
this project's tests. So it must stay importable by 3.10 with numpy 1.26: no 3.11+
syntax at runtime, nothing but numpy.

**The error map** is the trainer's own loss, per pixel: `(1 - w) * L1 + w * (1 - SSIM)`
with `w = 0.2` (gsplat v1.5.3's `ssim_lambda`), L1 the mean absolute difference over the
three channels and SSIM the local structural similarity in an 11-pixel gaussian window of
sigma 1.5 (the window `fused_ssim` uses). A held-out frame is judged by the measure the
splat was fitted with, so "accurate" means what training meant by it.

**Attribution.** A rendered pixel is `C(p) = sum_i c_i * a_i(p) * T_i(p)`: gaussian `i`'s
colour, its opacity at that pixel, and the light still passing through the gaussians in
front of it. `w_i(p) = a_i(p) T_i(p)` is how much of pixel `p` gaussian `i` painted. The
colour enters linearly, so the gradient of `sum_p e(p) * C(p)` with respect to `c_i` is
exactly `sum_p e(p) * w_i(p)`, and of `sum_p C(p)` exactly `sum_p w_i(p)`. The GPU
script renders a two-channel "colour" of zeros and reads both sums off one backward pass
(`holdout_error.attribute`); this module turns them into:

* **mean error** -- `sum e*w / sum w`: the error of the pixels a gaussian painted, weighted
  by how much of each it painted. Occlusion is in it for free: a hidden gaussian has
  `T ~ 0` and so no weight.
* **weight** -- `sum w` over every held-out frame: how many pixels' worth of held-out
  evidence the gaussian has. Zero means no held-out frame saw it, and its mean error is
  NaN rather than zero.

What attribution cannot do: tell apart the gaussians that share a pixel. A correct
gaussian behind a half-transparent wrong one is charged its share of that pixel's error.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

#: gsplat v1.5.3's `Config.ssim_lambda`: the trainer's loss is `0.8 * L1 + 0.2 * D-SSIM`.
SSIM_WEIGHT = 0.2
#: `fused_ssim`'s window: 11 taps, sigma 1.5.
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
_C1 = 0.01**2
_C2 = 0.03**2

#: The files a `holdout/` directory holds. Per gaussian, in `trained.ply`'s row order.
ERROR_FILE = "holdout_error.npy"
WEIGHT_FILE = "holdout_weight.npy"
VIEWS_FILE = "holdout_views.npy"
SUMMARY_FILE = "holdout.json"

#: A held-out frame "saw" a gaussian when the gaussian painted at least this many pixels'
#: worth of it (`holdout_views.npy` counts such frames).
VIEW_MIN_WEIGHT = 0.5


def gaussian_window(size: int = SSIM_WINDOW, sigma: float = SSIM_SIGMA) -> F64:
    """A normalised 1-D gaussian of `size` taps."""
    offsets = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    taps = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    window: F64 = taps / taps.sum()
    return window


def blur(image: npt.ArrayLike, window: F64 | None = None) -> F32:
    """Separable gaussian blur over the first two axes, same size, edges replicated."""
    array = np.asarray(image, dtype=np.float32)
    taps = gaussian_window() if window is None else window
    radius = len(taps) // 2
    height, width = array.shape[0], array.shape[1]
    pad_rows = [(radius, radius), (0, 0)] + [(0, 0)] * (array.ndim - 2)
    padded = np.pad(array, pad_rows, mode="edge")
    rows = np.zeros_like(array)
    for k, tap in enumerate(taps):
        rows += np.float32(tap) * padded[k : k + height]
    pad_cols = [(0, 0), (radius, radius)] + [(0, 0)] * (array.ndim - 2)
    padded = np.pad(rows, pad_cols, mode="edge")
    out = np.zeros_like(array)
    for k, tap in enumerate(taps):
        out += np.float32(tap) * padded[:, k : k + width]
    return out


def ssim_map(render: npt.ArrayLike, real: npt.ArrayLike) -> F32:
    """Local SSIM per pixel, averaged over channels. Both images in [0, 1], (H, W, C)."""
    a = np.asarray(render, dtype=np.float32)
    b = np.asarray(real, dtype=np.float32)
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a * mu_a
    var_b = blur(b * b) - mu_b * mu_b
    cov = blur(a * b) - mu_a * mu_b
    numerator = (2.0 * mu_a * mu_b + _C1) * (2.0 * cov + _C2)
    denominator = (mu_a * mu_a + mu_b * mu_b + _C1) * (var_a + var_b + _C2)
    ssim = np.clip(numerator / denominator, -1.0, 1.0)
    return ssim.mean(axis=2).astype(np.float32) if ssim.ndim == 3 else ssim.astype(np.float32)


def error_map(
    render: npt.ArrayLike, real: npt.ArrayLike, *, ssim_weight: float = SSIM_WEIGHT
) -> F32:
    """The trainer's loss per pixel: `(1 - w) * L1 + w * (1 - SSIM)`, (H, W), >= 0."""
    a = np.clip(np.asarray(render, dtype=np.float32), 0.0, 1.0)
    b = np.clip(np.asarray(real, dtype=np.float32), 0.0, 1.0)
    l1 = np.abs(a - b).mean(axis=2)
    dssim = 1.0 - ssim_map(a, b)
    error = (1.0 - ssim_weight) * l1 + ssim_weight * dssim
    return np.maximum(error, 0.0).astype(np.float32)


def psnr(render: npt.ArrayLike, real: npt.ArrayLike, mask: npt.ArrayLike | None = None) -> float:
    """PSNR in dB of two [0, 1] images, over `mask`'s pixels when given."""
    a = np.clip(np.asarray(render, dtype=np.float64), 0.0, 1.0)
    b = np.clip(np.asarray(real, dtype=np.float64), 0.0, 1.0)
    squared = ((a - b) ** 2).mean(axis=2)
    if mask is not None:
        squared = squared[np.asarray(mask, dtype=bool)]
    mse = float(squared.mean()) if squared.size else math.nan
    if not math.isfinite(mse):
        return math.nan
    return 99.0 if mse <= 1e-10 else -10.0 * math.log10(mse)


def _luminance(image: F32) -> F32:
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    lum: F32 = (image[..., :3] * weights).sum(axis=-1).astype(np.float32)
    return lum


def _gradient_energy(lum: F32, mask: npt.NDArray[np.bool_]) -> float:
    dx = np.abs(np.diff(lum, axis=1))
    dy = np.abs(np.diff(lum, axis=0))
    both = mask[:, 1:] & mask[:, :-1]
    down = mask[1:, :] & mask[:-1, :]
    total = float(dx[both].sum()) + float(dy[down].sum())
    count = int(both.sum()) + int(down.sum())
    return total / count if count else math.nan


def view_stats(
    render: npt.ArrayLike, real: npt.ArrayLike, error: F32, alpha: npt.ArrayLike
) -> dict[str, float | None]:
    """What one held-out frame says, over the pixels the splat covers (alpha > 0.5).

    * `psnr` and `meanError` -- how far the render is from the frame;
    * `bias` -- mean luminance of the render minus the frame's: an exposure or white
      balance change between frames shows as a bias the geometry cannot explain;
    * `sharpness` -- the frame's gradient energy over the render's. Well under 1: the
      held-out frame is blurrier than the splat (motion blur in that frame). Well over 1:
      the splat is blurrier than the frame (the frames it was trained on were blurred).
    """
    a = np.clip(np.asarray(render, dtype=np.float32), 0.0, 1.0)
    b = np.clip(np.asarray(real, dtype=np.float32), 0.0, 1.0)
    covered = np.asarray(alpha, dtype=np.float32).reshape(error.shape) > 0.5
    share = float(covered.mean()) if covered.size else 0.0
    if not covered.any():
        return {
            "coverage": 0.0,
            "psnr": None,
            "meanError": None,
            "bias": None,
            "sharpness": None,
        }
    lum_a, lum_b = _luminance(a), _luminance(b)
    energy_render = _gradient_energy(lum_a, covered)
    energy_real = _gradient_energy(lum_b, covered)
    sharpness = (
        energy_real / energy_render
        if math.isfinite(energy_render) and energy_render > 1e-6
        else math.nan
    )
    return {
        "coverage": _round(share, 4),
        "psnr": _round(psnr(a, b, covered), 3),
        "meanError": _round(float(error[covered].mean()), 5),
        "bias": _round(float(lum_a[covered].mean() - lum_b[covered].mean()), 4),
        "sharpness": _round(sharpness, 3),
    }


class Accumulator:
    """Per-gaussian sums over held-out frames: `sum e*w`, `sum w`, frames that saw it."""

    def __init__(self, count: int) -> None:
        self.error_sum = np.zeros(count, dtype=np.float64)
        self.weight_sum = np.zeros(count, dtype=np.float64)
        self.views = np.zeros(count, dtype=np.uint16)

    def add(self, error_weighted: npt.ArrayLike, weight: npt.ArrayLike) -> None:
        """One frame's two gradients. Negative weight is float noise and is clipped."""
        e = np.nan_to_num(np.asarray(error_weighted, dtype=np.float64).reshape(-1), nan=0.0)
        w = np.nan_to_num(np.asarray(weight, dtype=np.float64).reshape(-1), nan=0.0)
        if e.shape != self.error_sum.shape or w.shape != self.weight_sum.shape:
            raise ValueError(
                f"a frame's attribution has {e.shape[0]} and {w.shape[0]} values for "
                f"{self.error_sum.shape[0]} gaussians"
            )
        w = np.maximum(w, 0.0)
        e = np.maximum(e, 0.0)
        self.error_sum += e
        self.weight_sum += w
        self.views += (w >= VIEW_MIN_WEIGHT).astype(np.uint16)

    def mean_error(self) -> F32:
        return mean_error(self.error_sum, self.weight_sum)


def mean_error(error_sum: npt.ArrayLike, weight_sum: npt.ArrayLike) -> F32:
    """`sum e*w / sum w` per gaussian; NaN where no held-out frame gave it any weight."""
    e = np.asarray(error_sum, dtype=np.float64)
    w = np.asarray(weight_sum, dtype=np.float64)
    out = np.full(e.shape, np.nan, dtype=np.float64)
    seen = w > 0.0
    out[seen] = e[seen] / w[seen]
    return out.astype(np.float32)


def write(
    directory: Path,
    error: npt.ArrayLike,
    weight: npt.ArrayLike,
    views: npt.ArrayLike | None,
    summary: dict[str, Any],
) -> None:
    """The three arrays and `holdout.json`, into `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / ERROR_FILE, np.asarray(error, dtype=np.float32))
    np.save(directory / WEIGHT_FILE, np.asarray(weight, dtype=np.float32))
    if views is not None:
        np.save(directory / VIEWS_FILE, np.asarray(views, dtype=np.uint16))
    write_summary(directory, summary)


def write_summary(directory: Path, summary: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SUMMARY_FILE).write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def _round(value: float, digits: int) -> float | None:
    return round(float(value), digits) if math.isfinite(value) else None
