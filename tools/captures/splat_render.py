"""A Gaussian splat renderer on the CPU: colour, depth, coverage and a label per pixel.

The teacher loops (generated motion and generated fill) need what a camera would see of a
scan: a still to condition a video model on, the depth and the coverage behind each pixel
so a generated frame can be checked against the scan and lifted back into 3D, and, per
pixel, which part of the scan it shows (a limb, a region). They need it on the machine that
plans the work, not on the GPU that runs the model, and they need it the same every time.

**How.** Each gaussian is drawn as Monte-Carlo samples of its own 3D normal (truncated at
2 sigma), as many as its projected area asks for (up to `max_samples`), and every sample's
opacity is the gaussian's spread over the pixels its samples land on. The samples of every
gaussian are then composited **exactly**, front to back, per pixel: a sort by (pixel, depth)
and a running transmittance. It is not the EWA rasteriser a viewer uses -- edges are
grainier and very large gaussians are sampled coarsely -- but what it shows is the scan,
seeded and repeatable, and fast enough for a few hundred thousand gaussians a frame. On Fort
Clatsop's 22.6M published gaussians a 900 x 600 frame took about a minute.

Frames are numpy arrays: `rgb` (h, w, 3) in [0, 1], `depth` (h, w) metres along the view
axis (the coverage-weighted mean of what was composited; `inf` where nothing was),
`alpha` (h, w) the coverage, `label` (h, w) int: the label that contributed most to the
pixel, summed over its gaussians (-1 where nothing did), and `purity` (h, w): that label's
share of the pixel's coverage.

`GsplatRenderer` is the same call on a GPU: gsplat's EWA rasteriser (as a viewer draws the
scan), with the same cameras, coverage and depth, no labels. An image model trained on 3DGS
renders (NVIDIA Fixer) is shown that, not the point samples (`teacher_fill --renderer`).

Cameras follow the OpenCV convention the rest of the pipeline uses (x right, y down,
z forward; `K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]`), in the scan's own frame.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "Camera",
    "Frame",
    "GsplatRenderer",
    "SplatIndex",
    "Splats",
    "load_ply",
    "load_tileset",
    "render",
    "save_ply",
]


@dataclass(frozen=True)
class Camera:
    """A pinhole camera: world-to-camera rotation rows, centre, focal length and size."""

    rotation: np.ndarray  # (3, 3): rows are the camera's x (right), y (down), z (forward)
    centre: np.ndarray  # (3,)
    focal: float  # pixels
    width: int
    height: int
    #: Gaussians whose centre is farther along the view axis than this are not drawn: a view
    #: meant for what is near does not show the horizon as specks.
    far: float = float("inf")

    @staticmethod
    def look_at(
        eye: np.ndarray | list[float],
        target: np.ndarray | list[float],
        *,
        fov_deg: float = 50.0,
        width: int = 640,
        height: int = 360,
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
        far: float = float("inf"),
    ) -> Camera:
        eye_a = np.asarray(eye, np.float64)
        forward = np.asarray(target, np.float64) - eye_a
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-9:  # looking straight up or down
            right = np.cross(forward, (0.0, 1.0, 0.0))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        focal = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
        return Camera(np.stack([right, down, forward]), eye_a, float(focal), width, height, far)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates (n, 2) and depths (n,) of world points (n, 3)."""
        p = (np.asarray(points, np.float64) - self.centre) @ self.rotation.T
        z = p[:, 2]
        safe = np.where(z > 1e-6, z, 1e-6)
        uv = np.stack(
            [
                self.focal * p[:, 0] / safe + self.width / 2,
                self.focal * p[:, 1] / safe + self.height / 2,
            ],
            axis=1,
        )
        return uv, z

    def rays(self) -> np.ndarray:
        """Unit world directions through every pixel centre, (h, w, 3)."""
        v, u = np.mgrid[0 : self.height, 0 : self.width].astype(np.float64)
        d = np.stack(
            [
                (u + 0.5 - self.width / 2) / self.focal,
                (v + 0.5 - self.height / 2) / self.focal,
                np.ones_like(u),
            ],
            axis=-1,
        )
        d = d @ self.rotation
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def to_json(self) -> dict[str, object]:
        data: dict[str, object] = {
            "rotation": self.rotation.tolist(),
            "centre": self.centre.tolist(),
            "focal": self.focal,
            "width": self.width,
            "height": self.height,
        }
        if math.isfinite(self.far):
            data["far"] = self.far
        return data

    @staticmethod
    def from_json(data: dict[str, object]) -> Camera:
        return Camera(
            np.asarray(data["rotation"], np.float64),
            np.asarray(data["centre"], np.float64),
            float(data["focal"]),  # type: ignore[arg-type]
            int(data["width"]),  # type: ignore[call-overload]
            int(data["height"]),  # type: ignore[call-overload]
            float(data.get("far", float("inf"))),  # type: ignore[arg-type]
        )


@dataclass
class Splats:
    """Gaussians in linear units: positions, unit quaternions (w, x, y, z), scales (metres),
    colours in [0, 1] and opacities in [0, 1]."""

    positions: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    colours: np.ndarray
    opacities: np.ndarray

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def take(self, index: np.ndarray) -> Splats:
        return Splats(
            self.positions[index],
            self.rotations[index],
            self.scales[index],
            self.colours[index],
            self.opacities[index],
        )

    @staticmethod
    def concat(parts: list[Splats]) -> Splats:
        return Splats(*(np.concatenate([getattr(p, f) for p in parts]) for f in _FIELDS))


_FIELDS = ("positions", "rotations", "scales", "colours", "opacities")
#: The renderer's sample offsets: standard normal, truncated at 2 sigma, fixed.
_NOISE = np.random.default_rng(20261001).standard_normal((1 << 20, 3)).clip(-2.0, 2.0)


def _noise(rows: np.ndarray) -> np.ndarray:
    return _NOISE[rows]


_SH_C0 = 0.28209479177387814


def _from_columns(c: dict[str, np.ndarray]) -> Splats:
    return Splats(
        np.stack([c["x"], c["y"], c["z"]], axis=1).astype(np.float64),
        np.stack([c["rot_0"], c["rot_1"], c["rot_2"], c["rot_3"]], axis=1).astype(np.float64),
        np.exp(np.stack([c["scale_0"], c["scale_1"], c["scale_2"]], axis=1).astype(np.float64)),
        np.clip(0.5 + _SH_C0 * np.stack([c["f_dc_0"], c["f_dc_1"], c["f_dc_2"]], axis=1), 0, 1),
        1.0 / (1.0 + np.exp(-c["opacity"].astype(np.float64))),
    )


def load_ply(path: Path) -> Splats:
    """A 3DGS PLY (degree-0 colour)."""
    from splat_tiles import read_ply

    return _from_columns(read_ply(path))


def load_tileset(tileset: Path) -> Splats:
    """Every leaf gaussian of a `splat_tiles.py` tileset, in its local ENU frame."""
    import json

    from rig_tiles import glb_spz
    from splat_tiles import unpack_spz

    document = json.loads(tileset.read_text(encoding="utf-8"))
    leaves: list[str] = []
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("children"):
            stack.extend(tile["children"])
        else:
            leaves.append(tile["content"]["uri"])
    return Splats.concat(
        [_from_columns(unpack_spz(glb_spz(tileset.parent / uri))) for uri in sorted(leaves)]
    )


_PLY_PROPERTIES = (
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
)  # fmt: skip


def save_ply(path: Path, splats: Splats, comment: str = "tools/captures/splat_render.py") -> None:
    """A binary 3DGS PLY (degree-0 colour) that `splat_tiles.py` packs."""
    rows = np.zeros(len(splats), dtype=np.dtype([(name, "<f4") for name in _PLY_PROPERTIES]))
    rows["x"], rows["y"], rows["z"] = splats.positions.T
    dc = (np.clip(splats.colours, 0, 1) - 0.5) / _SH_C0
    rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"] = dc.T
    p = np.clip(splats.opacities, 1e-6, 1 - 1e-6)
    rows["opacity"] = np.log(p / (1 - p))
    rows["scale_0"], rows["scale_1"], rows["scale_2"] = np.log(np.maximum(splats.scales, 1e-9)).T
    q = splats.rotations / np.linalg.norm(splats.rotations, axis=1, keepdims=True)
    rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"] = q.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment {comment}\n"
        f"element vertex {len(splats)}\n"
        + "".join(f"property float {name}\n" for name in _PLY_PROPERTIES)
        + "end_header\n"
    )
    path.write_bytes(header.encode("ascii") + rows.tobytes())


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    alpha: np.ndarray
    label: np.ndarray
    #: The winning label's share of the pixel's coverage (0 where nothing was labelled).
    purity: np.ndarray


def _rotation_matrices(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)  # fmt: skip


#: Gaussians per chunk of a `SplatIndex`.
CHUNK_SPLATS = 8192
#: Samples generated at a time: bounds the renderer's memory, not what it draws.
SAMPLE_BLOCK = 1 << 21


def _spread(v: np.ndarray) -> np.ndarray:
    """The low 21 bits of each value, two zero bits between each (a Morton code's axis)."""
    v = v & np.uint64(0x1FFFFF)
    for shift, mask in (
        (32, 0x1F00000000FFFF),
        (16, 0x1F0000FF0000FF),
        (8, 0x100F00F00F00F00F),
        (4, 0x10C30C30C30C30C3),
        (2, 0x1249249249249249),
    ):
        v = (v | (v << np.uint64(shift))) & np.uint64(mask)
    return v


@dataclass(frozen=True)
class SplatIndex:
    """Gaussians grouped into compact chunks (equal runs along a Morton curve), each with a
    bounding sphere padded by twice its largest scale. `render(..., index=)` projects only
    the chunks that can reach the frame (`visible`): the same frame as without it -- the test
    is conservative -- at a fraction of the cost when a view sees a part of a large scan."""

    #: Gaussian indices, chunk after chunk.
    order: np.ndarray
    #: Per chunk: bounding-sphere centre (c, 3) and padded radius (c,).
    centres: np.ndarray
    radii: np.ndarray
    size: int

    @staticmethod
    def build(splats: Splats, size: int = CHUNK_SPLATS) -> SplatIndex:
        pos = np.asarray(splats.positions, np.float64)
        if len(pos) == 0:
            return SplatIndex(np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0), size)
        lo = pos.min(axis=0)
        span = np.maximum(pos.max(axis=0) - lo, 1e-9)
        q = np.minimum((pos - lo) / span * (1 << 21), (1 << 21) - 1).astype(np.uint64)
        one, two = np.uint64(1), np.uint64(2)
        key = _spread(q[:, 0]) | (_spread(q[:, 1]) << one) | (_spread(q[:, 2]) << two)
        del q
        order = np.argsort(key, kind="stable")
        del key
        starts = np.arange(0, len(pos), size)
        ordered = pos[order]
        lo_c = np.minimum.reduceat(ordered, starts, axis=0)
        hi_c = np.maximum.reduceat(ordered, starts, axis=0)
        del ordered
        reach = np.maximum.reduceat(np.asarray(splats.scales).max(axis=1)[order], starts)
        radii = np.linalg.norm(hi_c - lo_c, axis=1) / 2 + 2.0 * reach
        return SplatIndex(order, (lo_c + hi_c) / 2, radii, size)

    def visible(self, camera: Camera) -> np.ndarray:
        """Sorted indices of the gaussians whose chunk's sphere meets `camera`'s frustum
        (between 0.05 and `far` along its axis)."""
        p = (self.centres - camera.centre) @ camera.rotation.T
        r = self.radii
        ok = (p[:, 2] + r > 0.05) & (p[:, 2] - r < camera.far)
        for axis, half in ((0, camera.width / 2), (1, camera.height / 2)):
            t = half / camera.focal
            norm = math.hypot(1.0, t)
            ok &= (p[:, axis] - t * p[:, 2]) / norm <= r
            ok &= (-p[:, axis] - t * p[:, 2]) / norm <= r
        chunks = np.flatnonzero(ok)
        if chunks.size == 0:
            return np.zeros(0, np.int64)
        starts = chunks * self.size
        lengths = np.minimum(starts + self.size, self.order.size) - starts
        offsets = np.repeat(starts - (np.cumsum(lengths) - lengths), lengths)
        return np.sort(self.order[offsets + np.arange(int(lengths.sum()))])


def render(
    splats: Splats,
    camera: Camera,
    *,
    labels: np.ndarray | None = None,
    opacity_scale: np.ndarray | None = None,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_samples: int = 24,
    sample_budget: int = 12_000_000,
    seed: int = 0,
    index: SplatIndex | None = None,
) -> Frame:
    """What `camera` sees of `splats`. `opacity_scale` (n,) multiplies each gaussian's
    opacity (a view-cone fade, a mask); `labels` (n,) are reported per pixel. `index`
    (`SplatIndex.build(splats)`, built once) skips what cannot be in the frame; the frame is
    the same without it."""
    rng = np.random.default_rng(seed)
    w_px, h_px = camera.width, camera.height
    candidates = None if index is None else index.visible(camera)
    count = len(splats) if candidates is None else candidates.size
    # Which gaussians reach the frame, tested in blocks (memory stays bounded).
    kept_index, kept_radius = [], []
    for start in range(0, count, SAMPLE_BLOCK):
        stop = min(start + SAMPLE_BLOCK, count)
        rows = np.arange(start, stop) if candidates is None else candidates[start:stop]
        op = splats.opacities[rows]
        if opacity_scale is not None:
            op = op * np.asarray(opacity_scale)[rows]
        uv, z = camera.project(splats.positions[rows])
        radius = camera.focal * splats.scales[rows].max(axis=1) / np.maximum(z, 1e-6) * 2.0
        keep = (
            (z > 0.05)
            & (z < camera.far)
            & (uv[:, 0] > -radius)
            & (uv[:, 0] < w_px + radius)
            & (uv[:, 1] > -radius)
            & (uv[:, 1] < h_px + radius)
            & (op > 0.004)
        )
        kept_index.append(rows[keep])
        kept_radius.append(radius[keep])
    index_g = np.concatenate(kept_index) if kept_index else np.zeros(0, np.int64)
    area = np.pi * (np.concatenate(kept_radius) if kept_radius else np.zeros(0)) ** 2
    del kept_index, kept_radius
    n = np.clip((area / 6.0).astype(np.int64), 1, max_samples)
    if n.sum() > sample_budget:
        kept = rng.random(index_g.size) < sample_budget / n.sum()
        index_g, n, area = index_g[kept], n[kept], area[kept]
    positions, scales = splats.positions[index_g], splats.scales[index_g]
    op = splats.opacities[index_g]
    if opacity_scale is not None:
        op = op * np.asarray(opacity_scale)[index_g]
    # Each gaussian's samples are its own, the same in every frame (a table indexed by the
    # gaussian and the sample), so a gaussian that moves carries its samples with it and
    # consecutive frames differ by the motion, not by sampling noise. Made in blocks of
    # gaussians (`SAMPLE_BLOCK` samples) so memory stays bounded; each sample is the same.
    first = np.cumsum(n) - n
    total_samples = int(n.sum())
    cuts = np.unique(
        np.r_[
            0,
            np.searchsorted(first, np.arange(SAMPLE_BLOCK, total_samples, SAMPLE_BLOCK)),
            index_g.size,
        ]
    )
    # Per sample, kept compact (int32 where it fits): its gaussian, pixel, depth and opacity.
    owners, pixels, depths, alphas = [], [], [], []
    for g0, g1 in itertools.pairwise(cuts.tolist()):
        counts = n[g0:g1]
        owner = np.repeat(np.arange(g0, g1), counts)
        within = np.arange(owner.size) - np.repeat(first[g0:g1] - first[g0], counts)
        local = _noise((index_g[owner] * 2654435761 + within * 40503 + seed) % _NOISE.shape[0])
        local = local * scales[owner]
        rot = _rotation_matrices(splats.rotations[index_g[g0:g1]])
        world = positions[owner] + np.einsum("nij,nj->ni", rot[owner - g0], local)
        del local, rot, within
        p = (world - camera.centre) @ camera.rotation.T
        del world
        ok = p[:, 2] > 0.02
        zs = np.where(ok, p[:, 2], 1)
        px = np.floor(camera.focal * p[:, 0] / zs + w_px / 2).astype(np.int64)
        py = np.floor(camera.focal * p[:, 1] / zs + h_px / 2).astype(np.int64)
        ok &= (px >= 0) & (px < w_px) & (py >= 0) & (py < h_px)
        owner = owner[ok]
        owners.append(owner.astype(np.int32))
        pixels.append((py[ok] * w_px + px[ok]).astype(np.int32))
        depths.append(p[ok, 2])
        # A gaussian's opacity spread over the pixels its samples cover (area / n each).
        per_sample_px = area[owner] / n[owner]
        alphas.append(
            np.clip(op[owner] * np.minimum(1.0, 6.0 / np.maximum(per_sample_px, 1e-6)), 0.0, 0.99)
        )
        del p, zs, px, py, ok, owner, per_sample_px
    del positions, scales, op, area, n, first
    owner = np.concatenate(owners) if owners else np.zeros(0, np.int32)
    del owners
    pixel = np.concatenate(pixels) if pixels else np.zeros(0, np.int32)
    del pixels
    depth = np.concatenate(depths) if depths else np.zeros(0)
    del depths
    a = np.concatenate(alphas) if alphas else np.zeros(0)
    del alphas
    order = np.lexsort((depth, pixel))
    pixel, a, depth, owner = pixel[order], a[order], depth[order], owner[order]
    del order
    log_t = np.log1p(-a)
    running = np.cumsum(log_t)
    starts = np.r_[0, np.flatnonzero(np.diff(pixel)) + 1] if pixel.size else np.zeros(0, np.int64)
    group = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, pixel.size]))
    before = running - log_t - (running[starts] - log_t[starts])[group] if pixel.size else running
    del running, group, starts
    weight = a * np.exp(before)
    del before, log_t, a
    total = h_px * w_px
    gaussian = index_g[owner]
    del owner
    rgb = np.stack(
        [np.bincount(pixel, weight * splats.colours[gaussian, c], total) for c in range(3)],
        axis=1,
    ).astype(np.float64)
    alpha = np.bincount(pixel, weight, total).astype(np.float64)
    depth_sum = np.bincount(pixel, weight * depth, total).astype(np.float64)
    del depth
    rgb += (1.0 - alpha)[:, None] * np.asarray(background)
    label_img = np.full(total, -1, np.int64)
    purity = np.zeros(total)
    if labels is not None and pixel.size:
        # Per pixel, each label's share of what was composited; the largest wins.
        sample_labels = np.asarray(labels)[gaussian].astype(np.int64)
        key = pixel.astype(np.int64) * (int(sample_labels.max()) + 2) + (sample_labels + 1)
        keys, inverse = np.unique(key, return_inverse=True)
        sums = np.bincount(inverse, weight)
        key_pixel = keys // (int(sample_labels.max()) + 2)
        best = np.lexsort((-sums, key_pixel))
        first = best[np.r_[True, key_pixel[best][1:] != key_pixel[best][:-1]]]
        label_img[key_pixel[first]] = keys[first] % (int(sample_labels.max()) + 2) - 1
        purity[key_pixel[first]] = sums[first] / np.maximum(alpha[key_pixel[first]], 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_depth = np.where(alpha > 1e-6, depth_sum / alpha, np.inf)
    shape = (h_px, w_px)
    return Frame(
        np.clip(rgb, 0, 1).reshape(shape + (3,)),
        mean_depth.reshape(shape),
        np.clip(alpha, 0, 1).reshape(shape),
        label_img.reshape(shape),
        purity.reshape(shape),
    )


class GsplatRenderer:
    """`render` on a CUDA GPU with gsplat (`distill_fill.gsplat_frame`): the same camera
    (pixel centres at +0.5, `K = [[f, 0, w/2], [0, f, h/2], [0, 0, 1]]`), gaussians nearer
    than 0.05 or past `camera.far` not drawn, a black background, depth the coverage-weighted
    mean along the view axis (`inf` where nothing was). `label` is -1 and `purity` 0: it is
    for the teacher's views, not segmentation. The last few scenes' tensors stay on the GPU
    (`keep`), so the views of one scan upload it once."""

    name = "gsplat"

    def __init__(self, device: str = "cuda", keep: int = 3) -> None:
        self.device = device
        self.keep = keep
        #: (the Splats object, its tensors), most recent last.
        self._scenes: list[tuple[Splats, dict]] = []

    def _tensors(self, splats: Splats) -> dict:
        import torch

        for k, (held, tensors) in enumerate(self._scenes):
            if held is splats:
                self._scenes.append(self._scenes.pop(k))
                return tensors
        del self._scenes[: max(0, len(self._scenes) - self.keep + 1)]
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)
        q = splats.rotations / np.maximum(
            np.linalg.norm(splats.rotations, axis=1, keepdims=True), 1e-12
        )
        tensors = {
            "means": t(splats.positions),
            "quats": t(q),
            "scales": t(splats.scales),
            "opacities": t(splats.opacities),
            "colours": t(splats.colours),
        }
        self._scenes.append((splats, tensors))
        return tensors

    def __call__(
        self,
        splats: Splats,
        camera: Camera,
        *,
        opacity_scale: np.ndarray | None = None,
        background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Frame:
        import torch

        from distill_fill import camera_tensors, gsplat_frame

        shape = (camera.height, camera.width)
        if len(splats) == 0:
            return Frame(
                np.zeros(shape + (3,)) + np.asarray(background),
                np.full(shape, np.inf),
                np.zeros(shape),
                np.full(shape, -1, np.int64),
                np.zeros(shape),
            )
        scene = self._tensors(splats)
        opacities = scene["opacities"]
        if opacity_scale is not None:
            scale = torch.as_tensor(np.asarray(opacity_scale), dtype=torch.float32)
            opacities = opacities * scale.to(self.device)
        viewmat, K, w, h = camera_tensors(camera.to_json(), torch, self.device)
        with torch.no_grad():
            rgb, alpha, depth = gsplat_frame(
                scene["means"],
                scene["quats"],
                scene["scales"],
                opacities.clamp(0.0, 1.0),
                scene["colours"],
                viewmat,
                K,
                w,
                h,
                near=0.05,
                far=camera.far if math.isfinite(camera.far) else 1e10,
            )
        a = alpha.double().clamp(0.0, 1.0).cpu().numpy()
        colour = rgb.double().cpu().numpy() + (1.0 - a)[..., None] * np.asarray(background)
        return Frame(
            np.clip(colour, 0, 1),
            np.where(a > 1e-6, depth.double().cpu().numpy(), np.inf),
            a,
            np.full(shape, -1, np.int64),
            np.zeros(shape),
        )
