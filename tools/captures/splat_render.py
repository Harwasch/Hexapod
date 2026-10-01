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

Cameras follow the OpenCV convention the rest of the pipeline uses (x right, y down,
z forward; `K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]`), in the scan's own frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Camera", "Frame", "Splats", "load_ply", "load_tileset", "render", "save_ply"]


@dataclass(frozen=True)
class Camera:
    """A pinhole camera: world-to-camera rotation rows, centre, focal length and size."""

    rotation: np.ndarray  # (3, 3): rows are the camera's x (right), y (down), z (forward)
    centre: np.ndarray  # (3,)
    focal: float  # pixels
    width: int
    height: int

    @staticmethod
    def look_at(
        eye: np.ndarray | list[float],
        target: np.ndarray | list[float],
        *,
        fov_deg: float = 50.0,
        width: int = 640,
        height: int = 360,
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
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
        return Camera(np.stack([right, down, forward]), eye_a, float(focal), width, height)

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
        return {
            "rotation": self.rotation.tolist(),
            "centre": self.centre.tolist(),
            "focal": self.focal,
            "width": self.width,
            "height": self.height,
        }

    @staticmethod
    def from_json(data: dict[str, object]) -> Camera:
        return Camera(
            np.asarray(data["rotation"], np.float64),
            np.asarray(data["centre"], np.float64),
            float(data["focal"]),  # type: ignore[arg-type]
            int(data["width"]),  # type: ignore[call-overload]
            int(data["height"]),  # type: ignore[call-overload]
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
) -> Frame:
    """What `camera` sees of `splats`. `opacity_scale` (n,) multiplies each gaussian's
    opacity (a view-cone fade, a mask); `labels` (n,) are reported per pixel."""
    rng = np.random.default_rng(seed)
    w_px, h_px = camera.width, camera.height
    op = splats.opacities if opacity_scale is None else splats.opacities * opacity_scale
    uv, z = camera.project(splats.positions)
    radius = camera.focal * splats.scales.max(axis=1) / np.maximum(z, 1e-6) * 2.0
    keep = (
        (z > 0.05)
        & (uv[:, 0] > -radius)
        & (uv[:, 0] < w_px + radius)
        & (uv[:, 1] > -radius)
        & (uv[:, 1] < h_px + radius)
        & (op > 0.004)
    )
    index = np.nonzero(keep)[0]
    area = np.pi * radius[index] ** 2
    n = np.clip((area / 6.0).astype(np.int64), 1, max_samples)
    if n.sum() > sample_budget:
        kept = rng.random(index.size) < sample_budget / n.sum()
        index, n, area = index[kept], n[kept], area[kept]
    rot = _rotation_matrices(splats.rotations[index])
    owner = np.repeat(np.arange(index.size), n)
    # Each gaussian's samples are its own, the same in every frame (a table indexed by the
    # gaussian and the sample), so a gaussian that moves carries its samples with it and
    # consecutive frames differ by the motion, not by sampling noise.
    within = np.arange(owner.size) - np.repeat(np.cumsum(n) - n, n)
    local = _noise((index[owner] * 2654435761 + within * 40503 + seed) % _NOISE.shape[0])
    local = local * splats.scales[index][owner]
    world = splats.positions[index][owner] + np.einsum("nij,nj->ni", rot[owner], local)
    p = (world - camera.centre) @ camera.rotation.T
    ok = p[:, 2] > 0.02
    px = np.floor(camera.focal * p[:, 0] / np.where(ok, p[:, 2], 1) + w_px / 2).astype(np.int64)
    py = np.floor(camera.focal * p[:, 1] / np.where(ok, p[:, 2], 1) + h_px / 2).astype(np.int64)
    ok &= (px >= 0) & (px < w_px) & (py >= 0) & (py < h_px)
    owner, depth, px, py = owner[ok], p[ok, 2], px[ok], py[ok]
    # A gaussian's opacity spread over the pixels its samples cover (about area / n each).
    per_sample_px = area[owner] / n[owner]
    a = np.clip(
        op[index][owner] * np.minimum(1.0, 6.0 / np.maximum(per_sample_px, 1e-6)), 0.0, 0.99
    )
    pixel = py * w_px + px
    order = np.lexsort((depth, pixel))
    pixel, a, depth, owner = pixel[order], a[order], depth[order], owner[order]
    log_t = np.log1p(-a)
    running = np.cumsum(log_t)
    starts = np.r_[0, np.flatnonzero(np.diff(pixel)) + 1] if pixel.size else np.zeros(0, np.int64)
    group = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, pixel.size]))
    before = running - log_t - (running[starts] - log_t[starts])[group] if pixel.size else running
    weight = a * np.exp(before)
    total = h_px * w_px
    colour = splats.colours[index][owner]
    rgb = np.stack(
        [np.bincount(pixel, weight * colour[:, c], total) for c in range(3)], axis=1
    ).astype(np.float64)
    alpha = np.bincount(pixel, weight, total).astype(np.float64)
    depth_sum = np.bincount(pixel, weight * depth, total).astype(np.float64)
    rgb += (1.0 - alpha)[:, None] * np.asarray(background)
    label_img = np.full(total, -1, np.int64)
    purity = np.zeros(total)
    if labels is not None and pixel.size:
        # Per pixel, each label's share of what was composited; the largest wins.
        sample_labels = np.asarray(labels)[index][owner]
        key = pixel * (int(sample_labels.max()) + 2) + (sample_labels + 1)
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
