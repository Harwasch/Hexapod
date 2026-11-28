"""Pluggable rasterizers: numpy reference and lazy gsplat production-settings adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .geometry import Array, unit


@dataclass
class Camera:
    name: str
    c2w: Array
    k: Array
    width: int
    height: int
    blur: float = float("nan")
    exposure: float = float("nan")

    @property
    def centre(self) -> Array:
        return self.c2w[:3, 3]

    def project(self, xyz: Array) -> tuple[Array, Array]:
        local = (xyz - self.centre) @ self.c2w[:3, :3]
        pixels = local @ self.k.T
        return pixels[:, :2] / np.maximum(local[:, 2:3], 1e-12), local[:, 2]


@dataclass
class Scene:
    xyz: Array
    scales: Array
    quats: Array  # wxyz
    opacity: Array
    colors: Array  # RGB for numpy; (N, bands, 3) SH for gsplat
    sh_degree: int | None = None


@dataclass
class Render:
    rgb: Array
    alpha: Array
    depth: Array  # expected camera-z depth, normalized by accumulated alpha


class Backend(Protocol):
    def render(self, scene: Scene, camera: Camera) -> Render: ...


def rotation(quats: Array) -> Array:
    w, x, y, z = unit(quats).T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


class NumpyBackend:
    """Perspective covariance projection, front-to-back alpha compositing; tiny fixtures only.

    Uses a 0.3 px² covariance floor, 3-sigma support and black background. Not gsplat
    equivalence: no tile sorting, SH, antialias compensation or CUDA cutoff replication.
    """

    def render(self, scene: Scene, camera: Camera) -> Render:
        if scene.sh_degree is not None:
            raise ValueError("numpy reference expects RGB, not SH")
        h, w = camera.height, camera.width
        rgb, alpha, depth_sum = np.zeros((h, w, 3)), np.zeros((h, w)), np.zeros((h, w))
        local = (scene.xyz - camera.centre) @ camera.c2w[:3, :3]
        pixels, z = camera.project(scene.xyz)
        rotations = rotation(scene.quats)
        cov = (rotations * scene.scales[:, None, :] ** 2) @ rotations.transpose(0, 2, 1)
        view_rotation = camera.c2w[:3, :3].T
        cov = view_rotation @ cov @ view_rotation.T
        for i in np.argsort(z):
            if z[i] <= 0.01:
                continue
            x, y, zi = local[i]
            jac = np.array(
                [
                    [camera.k[0, 0] / zi, 0, -camera.k[0, 0] * x / zi**2],
                    [0, camera.k[1, 1] / zi, -camera.k[1, 1] * y / zi**2],
                ]
            )
            cov2 = jac @ cov[i] @ jac.T + np.eye(2) * 0.3
            radius = 3 * np.sqrt(np.linalg.eigvalsh(cov2).max())
            lo = np.maximum(np.floor(pixels[i] - radius).astype(int), 0)
            hi = np.minimum(np.ceil(pixels[i] + radius).astype(int) + 1, [w, h])
            if np.any(lo >= hi):
                continue
            yy, xx = np.mgrid[lo[1] : hi[1], lo[0] : hi[0]]
            delta = np.stack([xx - pixels[i, 0], yy - pixels[i, 1]], -1)
            exponent = np.einsum("...i,ij,...j->...", delta, np.linalg.inv(cov2), delta)
            a = np.minimum(0.999, scene.opacity[i] * np.exp(-0.5 * exponent))
            a[exponent > 9] = 0
            area = np.s_[lo[1] : hi[1], lo[0] : hi[0]]
            weight = a * (1 - alpha[area])
            rgb[area] += weight[..., None] * scene.colors[i]
            depth_sum[area] += weight * zi
            alpha[area] += weight
        return Render(rgb, alpha, depth_sum / np.maximum(alpha, 1e-12))


class GsplatBackend:
    """Same pinhole/classic-or-antialiased/SH settings as holdout_error.py.

    RGB+ED adds expected depth. Cache immutable scene tensors across cameras. GPU-only
    imports are delayed until use; the shipped image supplies torch/gsplat.
    """

    def __init__(self, *, antialiased: bool = False, packed: bool = True) -> None:
        self.mode = "antialiased" if antialiased else "classic"
        self.packed = packed
        self.cached_scene: Scene | None = None
        self.tensors: dict[str, Any] = {}

    def render(self, scene: Scene, camera: Camera) -> Render:
        return self.render_many(scene, [camera])[0]

    def render_many(self, scene: Scene, cameras: Sequence[Camera]) -> list[Render]:
        if not cameras or len({(c.width, c.height) for c in cameras}) != 1:
            raise ValueError("nonempty same-size camera batch required")
        camera = cameras[0]
        import torch
        from gsplat import rasterization

        def tensor(array: Array) -> Any:
            return torch.as_tensor(np.asarray(array), dtype=torch.float32, device="cuda")

        if scene is not self.cached_scene:
            self.tensors = {
                "means": tensor(scene.xyz),
                "scales": tensor(scene.scales),
                "quats": tensor(scene.quats),
                "opacities": tensor(scene.opacity),
                "colors": tensor(scene.colors),
            }
            self.cached_scene = scene
        with torch.no_grad():
            rgbd, alpha, _ = rasterization(
                **self.tensors,
                viewmats=tensor(np.linalg.inv(np.array([c.c2w for c in cameras]))),
                Ks=tensor(np.array([c.k for c in cameras])),
                width=camera.width,
                height=camera.height,
                sh_degree=scene.sh_degree,
                packed=self.packed,
                rasterize_mode=self.mode,
                render_mode="RGB+ED",
            )
        # One scene upload per backend/run, one RGB+ED raster call per batch. Transfer
        # together rather than synchronizing and copying once per camera/channel.
        rgbd_cpu = rgbd.detach().cpu().numpy()
        alpha_cpu = alpha.detach().cpu().numpy()
        return [
            Render(np.clip(rgbd_cpu[i, ..., :3], 0, 1), alpha_cpu[i, ..., 0], rgbd_cpu[i, ..., 3])
            for i in range(len(cameras))
        ]
