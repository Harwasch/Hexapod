"""Frames of the committed synthetic tree, rendered from known poses, at test time.

A9 took 104 MB of tiles out of git and the repository is 5.3 MB tracked, so the pose
fixture cannot be 40 committed JPEGs. What is committed is
`data/tiles/synthetic-tree/source/splat.ply` -- 12,000 real 3D Gaussians with a
byte-identity gate on them -- and this module turns that into an orbit of images plus the
poses it used, deterministically, in a few seconds.

The renderer is a painter's-algorithm splatter: project every gaussian centre through a
pinhole, draw a disc whose radius is the gaussian's own size in pixels, back to front. It
is not a 3DGS rasteriser and does not try to be -- it has no opacity blending and no
anisotropy. What it has to be is **geometrically exact and repeatable**, because the whole
value of the fixture is that a feature COLMAP finds corresponds to a fixed 3D point.

The ground plane is not decoration either. A tree is a cloud of similar green blobs; a
non-repeating textured plane underneath it gives SIFT the well-conditioned, unambiguous
correspondences that a real capture gets from the ground it was standing on. The texture
is generated from a seeded RNG once and does not tile, so there are no repeated patterns
for the matcher to confuse.

**`det(R) == +1` is asserted on every pose this module builds.** A0's own evaluator built
an improper rotation and measured a constant 90.000 degree error that read exactly like
broken SfM; the assertion is in `Pose.__post_init__` so that trap cannot be re-entered
from any caller.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

import gaussians
from gaussians import SH_C0

F64 = npt.NDArray[np.float64]

#: How far a rotation matrix may be from proper before the fixture refuses to exist.
DET_TOLERANCE = 1e-9
#: The ground texture: this many texels across `GROUND_SPAN_M` metres, seeded once.
GROUND_TEXELS = 192
GROUND_SPAN_M = 26.0
GROUND_SEED = 20260922
#: Disc radius as a multiple of the gaussian's largest axis. Above 1 the tree closes up
#: into a surface, which is what gives SIFT something with texture to hold on to.
SPLAT_GAIN = 2.2
SKY_TOP = (150, 178, 210)
SKY_BOTTOM = (206, 216, 226)


@dataclass(frozen=True)
class Intrinsics:
    """A pinhole with no distortion -- what the renderer is, exactly."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def of(width: int, height: int, focal_px: float) -> Intrinsics:
        return Intrinsics(width, height, focal_px, focal_px, width / 2.0, height / 2.0)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }


@dataclass(frozen=True)
class Pose:
    """World-to-camera, in COLMAP's convention: `x_cam = R @ x_world + t`.

    Refuses to exist unless `R` is a proper rotation. That is A0 #7's third finding, and
    it is here rather than in the evaluator because an evaluator can be bypassed and a
    constructor cannot.
    """

    name: str
    rotation: F64
    translation: F64

    def __post_init__(self) -> None:
        det = float(np.linalg.det(self.rotation))
        if abs(det - 1.0) > DET_TOLERANCE:
            raise ValueError(
                f"{self.name}: det(R) = {det:.12f}, not +1. An improper rotation is not a "
                f"pose -- A0's evaluator built one and measured a constant 90.000 degree "
                f"error that looked exactly like broken SfM"
            )
        orthogonality = float(np.abs(self.rotation @ self.rotation.T - np.eye(3)).max())
        if orthogonality > 1e-9:
            raise ValueError(f"{self.name}: R is not orthonormal (off by {orthogonality:.3g})")

    @property
    def centre(self) -> F64:
        return -self.rotation.T @ self.translation

    @property
    def qvec(self) -> tuple[float, float, float, float]:
        """w-first, COLMAP's order."""
        m = self.rotation
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            q = (
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            )
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = (
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            )
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = (
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            )
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = (
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            )
        vector = np.asarray(q, dtype=np.float64)
        vector /= float(np.linalg.norm(vector))
        return (float(vector[0]), float(vector[1]), float(vector[2]), float(vector[3]))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "qvec": list(self.qvec),
            "tvec": [float(v) for v in self.translation],
            "centre": [float(v) for v in self.centre],
        }


def look_at(name: str, eye: F64, target: F64, up: F64 | None = None) -> Pose:
    """A camera at `eye` looking at `target`, x right, y down, z forward.

    Built as three orthonormal rows in that order, which is right-handed, so `det(R)` is
    `+1` by construction -- and asserted anyway by `Pose`.
    """
    world_up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64)
    forward = _unit(np.asarray(target, dtype=np.float64) - eye)
    right = _unit(np.cross(forward, world_up))
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return Pose(name=name, rotation=rotation, translation=-rotation @ np.asarray(eye, float))


def orbit(
    count: int,
    *,
    radius_m: float = 9.0,
    height_m: float = 3.6,
    target: tuple[float, float, float] = (0.0, 0.0, 3.0),
    stem: str = "frame",
) -> tuple[Pose, ...]:
    """`count` cameras evenly around a closed circle, all looking at `target`.

    A *closed* orbit on purpose: it is the case A0 #7 found `sequential_matcher` fails on
    (2 of 40 registered), because the last frame and the first frame overlap and only an
    exhaustive matcher ever compares them.
    """
    centre = np.asarray(target, dtype=np.float64)
    poses: list[Pose] = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        eye = np.array(
            [radius_m * math.cos(angle), radius_m * math.sin(angle), height_m], dtype=np.float64
        )
        poses.append(look_at(f"{stem}_{index:04d}.jpg", eye, centre))
    return tuple(poses)


def _unit(vector: F64) -> F64:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("cannot normalise a zero vector")
    return vector / norm


# --- the tree -----------------------------------------------------------------------


@dataclass(frozen=True)
class Cloud:
    """The committed splat, reduced to what this renderer needs."""

    position: F64
    colour: npt.NDArray[np.uint8]
    radius_m: F64

    @property
    def count(self) -> int:
        return int(self.position.shape[0])


def load_cloud(ply: Path) -> Cloud:
    """Read the committed 3DGS PLY through the same reader the pipeline uses."""
    splat = gaussians.read_splat(ply)
    columns = splat.columns
    position = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)
    sh = np.stack([columns[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float64)
    rgb = np.clip(sh * SH_C0 + 0.5, 0.0, 1.0)
    scales = np.exp(np.stack([columns[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64))
    finite = np.isfinite(position).all(axis=1) & np.isfinite(scales).all(axis=1)
    return Cloud(
        position=position[finite],
        colour=(rgb[finite] * 255.0).round().astype(np.uint8),
        radius_m=scales[finite].max(axis=1),
    )


# --- rendering ----------------------------------------------------------------------


def ground_texture(seed: int = GROUND_SEED, texels: int = GROUND_TEXELS) -> npt.NDArray[np.uint8]:
    """One non-repeating slab of high-contrast noise, seeded, in earth-ish colours.

    Non-repeating matters: a tiled texture gives the matcher many equally good wrong
    answers, which is the classic way to make an SfM benchmark that measures nothing.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 215, size=(texels, texels, 1), dtype=np.int64)
    tint = np.array([[[1.00, 0.86, 0.70]]])
    texture: npt.NDArray[np.uint8] = np.clip(base * tint, 0, 255).astype(np.uint8)
    return texture


def render(
    cloud: Cloud,
    pose: Pose,
    intrinsics: Intrinsics,
    texture: npt.NDArray[np.uint8],
    *,
    span_m: float = GROUND_SPAN_M,
) -> Image.Image:
    """One frame: sky, then the textured ground, then the tree back to front."""
    canvas = _sky(intrinsics)
    _draw_ground(canvas, pose, intrinsics, texture, span_m)
    image = Image.fromarray(canvas, mode="RGB")
    _draw_cloud(image, cloud, pose, intrinsics)
    return image


def _sky(intrinsics: Intrinsics) -> npt.NDArray[np.uint8]:
    ramp = np.linspace(0.0, 1.0, intrinsics.height)[:, None]
    top = np.asarray(SKY_TOP, dtype=np.float64)
    bottom = np.asarray(SKY_BOTTOM, dtype=np.float64)
    column = top[None, :] * (1.0 - ramp) + bottom[None, :] * ramp
    return np.broadcast_to(column[:, None, :], (intrinsics.height, intrinsics.width, 3)).astype(
        np.uint8
    )


def _draw_ground(
    canvas: npt.NDArray[np.uint8],
    pose: Pose,
    intrinsics: Intrinsics,
    texture: npt.NDArray[np.uint8],
    span_m: float,
) -> None:
    """Inverse ray-cast every pixel onto z = 0 and sample the texture there.

    Per-pixel and exact rather than a projected quad, so the plane is correct right out to
    the horizon and a feature on it is at the world position it looks like it is at.
    """
    height, width = intrinsics.height, intrinsics.width
    us, vs = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    directions = np.stack(
        [
            (us - intrinsics.cx) / intrinsics.fx,
            (vs - intrinsics.cy) / intrinsics.fy,
            np.ones_like(us),
        ],
        axis=-1,
    )
    world = directions @ pose.rotation  # (R^T d) for each pixel, as a row-vector product
    eye = pose.centre
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = -eye[2] / world[..., 2]
    hit = np.isfinite(distance) & (distance > 0.0) & (world[..., 2] < 0.0)
    points = eye[None, None, :] + world * distance[..., None]
    texels = texture.shape[0]
    grid = (points[..., :2] + span_m / 2.0) / span_m * texels
    inside = hit & (grid >= 0).all(axis=-1) & (grid < texels).all(axis=-1)
    rows = np.clip(grid[..., 1].astype(np.int64), 0, texels - 1)
    cols = np.clip(grid[..., 0].astype(np.int64), 0, texels - 1)
    sampled = texture[rows, cols]
    # Fade with distance so the far plane does not out-contrast the subject.
    fade = np.clip(1.0 - (distance / (span_m * 1.2)), 0.25, 1.0)[..., None]
    shaded = sampled * fade + np.asarray(SKY_BOTTOM, dtype=np.float64) * (1.0 - fade)
    canvas[inside] = shaded[inside].round().astype(np.uint8)


def _draw_cloud(image: Image.Image, cloud: Cloud, pose: Pose, intrinsics: Intrinsics) -> None:
    camera = cloud.position @ pose.rotation.T + pose.translation
    depth = camera[:, 2]
    visible = depth > 0.05
    u = intrinsics.fx * camera[:, 0] / np.where(visible, depth, 1.0) + intrinsics.cx
    v = intrinsics.fy * camera[:, 1] / np.where(visible, depth, 1.0) + intrinsics.cy
    radius = np.maximum(
        0.6, intrinsics.fx * cloud.radius_m * SPLAT_GAIN / np.where(visible, depth, 1.0)
    )
    margin = radius + 2.0
    visible &= (u > -margin) & (u < intrinsics.width + margin)
    visible &= (v > -margin) & (v < intrinsics.height + margin)
    order = np.argsort(-depth)
    order = order[visible[order]]
    draw = ImageDraw.Draw(image)
    colours = cloud.colour
    for index in order.tolist():
        x, y, r = float(u[index]), float(v[index]), float(radius[index])
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=(int(colours[index, 0]), int(colours[index, 1]), int(colours[index, 2])),
        )


# --- the fixture --------------------------------------------------------------------


@dataclass(frozen=True)
class Truth:
    """Everything the evaluation needs: the intrinsics, the poses, and the scene size."""

    intrinsics: Intrinsics
    poses: tuple[Pose, ...]
    extent_m: float
    splats: int

    def to_dict(self) -> dict[str, object]:
        return {
            "note": "ground truth for the rendered orbit; det(R) == +1 is asserted per pose",
            "intrinsics": self.intrinsics.to_dict(),
            "extentM": self.extent_m,
            "splats": self.splats,
            "poses": [pose.to_dict() for pose in self.poses],
        }

    def centres(self) -> F64:
        return np.stack([pose.centre for pose in self.poses])

    def by_name(self) -> dict[str, Pose]:
        return {pose.name: pose for pose in self.poses}


def render_orbit(
    ply: Path,
    out_dir: Path,
    *,
    count: int = 40,
    width: int = 640,
    height: int = 480,
    focal_px: float = 520.0,
    radius_m: float = 9.0,
    camera_height_m: float = 3.6,
    quality: int = 92,
) -> Truth:
    """Render `count` frames into `out_dir` and return the poses they were rendered from.

    Deterministic: the splats are read from a committed file, the ground texture is
    seeded, and nothing here touches a clock or the filesystem's ordering.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cloud = load_cloud(ply)
    intrinsics = Intrinsics.of(width, height, focal_px)
    poses = orbit(count, radius_m=radius_m, height_m=camera_height_m)
    texture = ground_texture()
    for pose in poses:
        frame = render(cloud, pose, intrinsics, texture)
        frame.save(out_dir / pose.name, format="JPEG", quality=quality, subsampling=0)
    low = cloud.position.min(axis=0)
    high = cloud.position.max(axis=0)
    truth = Truth(
        intrinsics=intrinsics,
        poses=poses,
        extent_m=float(np.linalg.norm(high - low)),
        splats=cloud.count,
    )
    (out_dir.parent / "truth.json").write_text(
        json.dumps(truth.to_dict(), indent=1) + "\n", encoding="utf-8"
    )
    return truth
