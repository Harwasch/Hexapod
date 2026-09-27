"""Not gsplat. A dense torch rasteriser with `gsplat.rasterization`'s signature, so that
`lod_optimise.py` can be run -- forward and backward -- on a machine with no CUDA.

It follows the 3DGS forward pass gsplat v1.5.3 implements (and scratchpad lodcmp/raster.py
did in numpy): EWA projection with the 0.3 px low-pass added to the 2D covariance,
`alpha = min(0.999, opacity * exp(power))` with the opacity/geometry gradient dropped where
that clamp is hit (RasterizeToPixels3DGSBwd.cu: `if (opac * vis <= 0.999f)`), fragments
under 1/255 skipped, front-to-back compositing by view depth, and degree-0 SH as
`max(SH_C0 * dc + 0.5, 0)`. Every (gaussian, pixel) pair is evaluated, so it is for scenes
of a few thousand gaussians and images of a few thousand pixels, which is all a test needs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

SH_C0 = 0.28209479177387814


def _rotations(quats: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quats, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def rasterization(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,  # noqa: N803
    width: int,
    height: int,
    sh_degree: int | None = None,
    packed: bool = True,
    rasterize_mode: str = "classic",
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    assert rasterize_mode == "classic"
    view, k = viewmats[0], Ks[0]
    if sh_degree is not None:
        colour = torch.clamp_min(SH_C0 * colors[:, 0, :] + 0.5, 0.0)
    else:
        colour = colors
    rotation, translation = view[:3, :3], view[:3, 3]
    cam = means @ rotation.T + translation
    keep = cam[:, 2] > 0.01
    empty = (
        torch.zeros(1, height, width, colour.shape[-1], dtype=means.dtype),
        torch.zeros(1, height, width, 1, dtype=means.dtype),
        {},
    )
    if not bool(keep.any()):
        return empty
    cam, colour, opacity = cam[keep], colour[keep], opacities[keep]
    axes = _rotations(quats[keep]) * scales[keep][:, None, :]
    cov_world = axes @ axes.transpose(1, 2)
    cov_cam = rotation @ cov_world @ rotation.T
    x, y, z = cam.unbind(-1)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    zero = torch.zeros_like(z)
    jacobian = torch.stack(
        [
            torch.stack([fx / z, zero, -fx * x / (z * z)], -1),
            torch.stack([zero, fy / z, -fy * y / (z * z)], -1),
        ],
        -2,
    )
    cov2 = jacobian @ cov_cam @ jacobian.transpose(1, 2)
    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + 0.3
    det = a * c - b * b
    u = fx * x / z + cx
    v = fy * y / z + cy
    rows = torch.arange(height, dtype=means.dtype)
    py, px = torch.meshgrid(rows, torch.arange(width, dtype=means.dtype), indexing="ij")
    dx = px.reshape(1, -1) + 0.5 - u[:, None]
    dy = py.reshape(1, -1) + 0.5 - v[:, None]
    power = -0.5 * (c[:, None] * dx * dx + a[:, None] * dy * dy) / det[:, None] + (
        b[:, None] * dx * dy / det[:, None]
    )
    raw = opacity[:, None] * torch.exp(power)
    alpha = torch.where(raw > 0.999, torch.full_like(raw, 0.999), raw)
    alpha = torch.where((power > 0) | (raw < 1.0 / 255.0), torch.zeros_like(alpha), alpha)
    order = torch.argsort(z)
    alpha = alpha[order]
    keep_light = torch.cumprod(1.0 - alpha, dim=0)
    transmit = torch.cat([torch.ones_like(keep_light[:1]), keep_light[:-1]], dim=0)
    weights = alpha * transmit
    image = weights.T @ colour[order]
    coverage = 1.0 - keep_light[-1]
    return (
        image.reshape(1, height, width, -1),
        coverage.reshape(1, height, width, 1),
        {},
    )
