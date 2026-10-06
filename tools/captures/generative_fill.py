"""Generative fill: what a scan missed, made by a video model along camera paths that start
at a real camera, fused into an inferred layer of its own.

The owner's direction (2026-10-05): no rule-based geometry and no 3D object models -- a
video or world model makes the novel views, keeps them consistent with each other, and the
shape comes out of those views fused into gaussians. `teacher_fill.py` is the per-view
harness this grows from (its gate, lift and package are reused); what is new here:

1. **Unknown pixels** (`SeenDirections`, `ConeKnown`). With real cameras (the spool and the
   pumpkin: COLMAP poses), every gaussian has a *support* -- how many real cameras saw it,
   depth-tested -- and the directions they saw it from (64 octahedral bins). A gaussian is
   known from a viewpoint when at least `SUPPORT_MIN` cameras saw it and one of them from
   within about `KNOWN_DEG` of that viewpoint's direction. Without cameras (the camp) the
   view cones stand in (`teacher_fill.seen_weights`). A virtual frame's pixels are **known**
   where the known gaussians cover them; **to generate** where only unknown ones do, or
   nothing at all; **to lift** where they are to generate and the pixel's ray meets the
   region of interest (`roi`).
2. **Camera paths** (`plan_paths`) start at a real camera -- its own photo is frame 0 -- and
   move towards the viewpoint that sees the most unknown pixels of the region, so the
   generator sees real context first.
3. **Generate** (`ClipFiller`): a video model is given the known pixels and asked only for
   the rest (`video_fill_models`: Wan2.1-VACE natively, Wan2.2 TI2V and Cosmos-Predict2 by
   their own conditioning mechanisms). `PerViewClipFiller` runs a per-image inpainter
   (`world_model_client.GenerativeFiller`, LaMa) on the keyframes, as the baseline.
4. **Lift** (`align_depth`, `lift_clip`): a monocular depth model (Depth Anything V2 Small,
   Apache-2.0) on each generated keyframe, its affine inverse depth fitted to the measured
   depth of the frame's known pixels, places every lifted pixel. Confidence: distance from
   anything measured, how well the depth fitted, and -- with two seeds -- how far the
   seeds agree (`seed_agreement`).
5. **Consistency with the data**: the known-pixel gate (`gate_clip`), free-space carving
   (`carve`: an inferred gaussian in front of what a real camera saw is deleted), the
   region and footprint, and a gsplat distil (`distill_fill`) with the measured scan frozen,
   real photos at full weight on what they see and the generated frames at
   `GENERATED_WEIGHT` on their unknown pixels only.
6. **Progressive** rounds: a second round's paths are rendered with the first round's fill
   as known content, so the model continues it rather than inventing again.

Everything runs on the CPU with `splat_render.render` and stand-ins (`TeleaClipFiller`,
`StandInDepth`) for tests; `infra/modal/fill.py` (`gen:` and `holdout:` jobs) runs it on a
GPU with gsplat, the video models and Depth Anything.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

import teacher_fill as tf
import view_cones as vc
from splat_render import Camera, Frame, Splats, render

#: A gaussian is known only where at least this many real cameras saw it (LIVING_ENGINE §3).
SUPPORT_MIN = 3
#: ...and one of them from within about this angle of the viewing direction. (At 40 deg, a
#: lawn the phone saw from 16 deg up was unknown from 50 deg up and a clip there kept almost
#: no context: run 37381466964.)
KNOWN_DEG = 55.0
#: A path's end view must keep at least this share of what the scan shows there known
#: (context for the model).
MIN_CONTEXT = 0.3
#: Octahedral direction bins per side (64 bins of about 25 degrees).
DIR_BINS = 8
#: A real camera sees a gaussian when its centre is no more than this share of the
#: rendered depth behind the surface at its pixel.
DEPTH_TOLERANCE = 0.03
#: Coverage that counts as covered (and as known, when only known gaussians are drawn).
COVERED = 0.5
#: The mask to generate is grown by this (px) so the model blends across its edge.
GENERATE_GROW_PX = 4
#: Every this-many-th frame of a clip is lifted (and distilled); frame 0 is a real photo.
KEYFRAME_EVERY = 4
#: Lift every this-many-th pixel (each way) of a keyframe's pixels to lift.
LIFT_STRIDE = 4
#: The gate: a generated frame must keep the known pixels at this PSNR (dB), both frames
#: blurred (`teacher_fill.GATE_FULL_RENDER_PSNR_DB`: the models redraw every pixel).
GATE_DB = tf.GATE_FULL_RENDER_PSNR_DB
#: Frames with fewer known pixels than this share are not gated (nothing to hold them to).
GATE_MIN_KNOWN = 0.03
#: Depth fitting: known pixels needed, and the residual (relative) past which a frame's
#: depth is not trusted.
DEPTH_MIN_PIXELS = 200
DEPTH_MAX_RESIDUAL = 0.15
#: Seed agreement: colour (0..1, per channel RMS) and relative depth at which it halves.
SEED_COLOUR = 0.12
SEED_DEPTH = 0.05
#: The distil: generated views' weight against a real photo's 1, and their outside weight.
GENERATED_WEIGHT = 0.2
GENERATED_OUTSIDE = 0.5
#: Carving: an inferred gaussian is in seen-through space when it is nearer than the
#: measured surface by this share of its depth plus this share of the region's size.
CARVE_SHARE = 0.03
CARVE_REGION = 0.01
#: The layer's budget: at most this share of the measured gaussians (LIVING_ENGINE §3:
#: inferred <= 15 % of a site), and never fewer than `MIN_BUDGET`.
BUDGET_SHARE = 0.15
MIN_BUDGET = 20_000
#: Lifted points are kept within the region grown by this share of its size.
ROI_PAD = 0.25
#: The virtual cameras' horizontal field of view.
FOV_DEG = 60.0
#: What a clip is asked for when no hint is given.
PROMPT = (
    "A real outdoor place filmed on a phone: the camera moves slowly and smoothly; natural "
    "daylight, sharp detail, real footage; nothing in the scene moves."
)
RULE = (
    "unknown pixels (support < {support} real views, or none from within {deg:g} deg, or "
    "nothing at all) of camera paths that start at a real camera, generated by a video model "
    "given only the known pixels, lifted at monocular depth fitted to the measured depth, "
    "carved where a real camera saw through, distilled with the measured scan frozen"
)


# --- real cameras (COLMAP) --------------------------------------------------------------------

#: COLMAP camera model ids: name and parameter count (src/colmap/sensor/models.h).
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass(frozen=True)
class Intrinsics:
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    def matrix_and_distortion(self) -> tuple[np.ndarray, np.ndarray]:
        """OpenCV's K and distortion coefficients (fisheye models refused)."""
        p, m = self.params, self.model
        if m == "SIMPLE_PINHOLE":
            fx = fy = p[0]
            cx, cy, dist = p[1], p[2], []
        elif m == "PINHOLE":
            fx, fy, cx, cy, dist = p[0], p[1], p[2], p[3], []
        elif m == "SIMPLE_RADIAL":
            fx = fy = p[0]
            cx, cy, dist = p[1], p[2], [p[3], 0.0, 0.0, 0.0]
        elif m == "RADIAL":
            fx = fy = p[0]
            cx, cy, dist = p[1], p[2], [p[3], p[4], 0.0, 0.0]
        elif m in ("OPENCV", "FULL_OPENCV"):
            fx, fy, cx, cy = p[:4]
            dist = list(p[4:])
        else:
            raise ValueError(f"camera model {m} is not supported (pinhole and radial only)")
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return k, np.asarray(dist, np.float64)

    @property
    def focal(self) -> float:
        k, _ = self.matrix_and_distortion()
        return 0.5 * (k[0, 0] + k[1, 1])


@dataclass
class RealView:
    """One real photo: its camera in the scan's local frame (a centred pinhole at the photo's
    size, after undistortion) and where the photo is."""

    name: str
    camera: Camera
    intrinsics: Intrinsics | None = None
    image: Path | None = None

    def photo(self, width: int | None = None) -> np.ndarray | None:
        """The photo undistorted onto `camera` (and resized to `width`), uint8."""
        if self.image is None or not self.image.exists():
            return None
        import cv2
        from PIL import Image

        rgb = np.asarray(Image.open(self.image).convert("RGB"))
        w, h = self.camera.width, self.camera.height
        if rgb.shape[1] != w or rgb.shape[0] != h:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        if self.intrinsics is not None:
            k, dist = self.intrinsics.matrix_and_distortion()
            scale = w / self.intrinsics.width
            k = k.copy()
            k[:2] *= scale
            new = np.array(
                [[self.camera.focal, 0, w / 2], [0, self.camera.focal, h / 2], [0, 0, 1]]
            )
            if dist.size or abs(k[0, 2] - w / 2) > 0.5 or abs(k[1, 2] - h / 2) > 0.5:
                mx, my = cv2.initUndistortRectifyMap(k, dist, None, new, (w, h), cv2.CV_32FC1)
                rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if width is not None and width != w:
            rgb = cv2.resize(rgb, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        return rgb


def _u64(handle: Any) -> int:
    return struct.unpack("<Q", handle.read(8))[0]


def read_colmap(directory: Path) -> tuple[dict[int, Intrinsics], list[dict]]:
    """`cameras.bin` and `images.bin` of a COLMAP sparse model: intrinsics by camera id, and
    per image its name, world-to-camera rotation `r` and `t` (`x_cam = r x + t`), camera id.
    The model may sit in a folder below `directory` (`sparse/0/`)."""
    if not (directory / "cameras.bin").exists():
        found = sorted(directory.rglob("cameras.bin"))
        if not found:
            raise FileNotFoundError(f"no cameras.bin under {directory}")
        directory = found[0].parent
    cameras: dict[int, Intrinsics] = {}
    with (directory / "cameras.bin").open("rb") as f:
        for _ in range(_u64(f)):
            cid, model_id, width, height = struct.unpack("<IiQQ", f.read(24))
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"COLMAP camera model id {model_id} is unknown")
            name, count = CAMERA_MODELS[model_id]
            params = struct.unpack(f"<{count}d", f.read(8 * count))
            cameras[cid] = Intrinsics(name, int(width), int(height), tuple(params))
    images: list[dict] = []
    with (directory / "images.bin").open("rb") as f:
        for _ in range(_u64(f)):
            _iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<IdddddddI", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            f.read(24 * _u64(f))
            images.append(
                {
                    "name": name.decode("utf-8"),
                    "r": _quat_matrix(qw, qx, qy, qz),
                    "t": np.array([tx, ty, tz]),
                    "camera": cid,
                }
            )
    return cameras, sorted(images, key=lambda i: i["name"])


def _quat_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def real_views(
    poses: Path, placement: dict | None = None, frames: Path | None = None
) -> list[RealView]:
    """The COLMAP model's cameras in the scan's local frame: `placement` (the pipeline's
    `placement.json`, `x' = scale * R x + t` from COLMAP to canonical.ply, which is the
    tiles' local ENU frame) moves each centre like a point and turns its axes by R."""
    cameras, images = read_colmap(poses)
    s, rp, tp = 1.0, np.eye(3), np.zeros(3)
    if placement is not None:
        s = float(placement["scale"])
        rp = np.asarray(placement["rotation"], np.float64)
        tp = np.asarray(placement["translation"], np.float64)
    out = []
    for image in images:
        intr = cameras[image["camera"]]
        centre = -image["r"].T @ image["t"]
        rotation = image["r"] @ rp.T
        camera = Camera(rotation, s * rp @ centre + tp, intr.focal, intr.width, intr.height)
        path = None
        if frames is not None:
            path = frames / image["name"]
            if not path.exists():  # a model that names its frames by a folder of its own
                path = frames / Path(image["name"]).name
        out.append(RealView(image["name"], camera, intr, path))
    return out


def scaled(camera: Camera, width: int, height: int | None = None) -> Camera:
    """`camera` at another size, its horizontal field of view kept."""
    h = height if height is not None else max(1, round(camera.height * width / camera.width))
    return Camera(
        camera.rotation, camera.centre, camera.focal * width / camera.width, width, h, camera.far
    )


# --- what is known, from where ------------------------------------------------------------------


def _bin_centres(bins: int = DIR_BINS) -> np.ndarray:
    u = (np.arange(bins) + 0.5) / bins * 2 - 1
    gx, gy = np.meshgrid(u, u)
    return vc.decode_axis(
        np.stack([np.round((gx + 1) / 2 * 255), np.round((gy + 1) / 2 * 255)], -1).reshape(-1, 2)
    )


def direction_bins(directions: np.ndarray, bins: int = DIR_BINS) -> np.ndarray:
    """Octahedral bin (0..bins^2-1) of unit vectors (n, 3)."""
    rg = vc.encode_axis(np.asarray(directions, np.float64)).astype(np.float64)
    i = np.clip((rg[:, 0] / 255.0 * bins).astype(np.int64), 0, bins - 1)
    j = np.clip((rg[:, 1] / 255.0 * bins).astype(np.int64), 0, bins - 1)
    return j * bins + i


def _neighbour_masks(deg: float, bins: int = DIR_BINS) -> np.ndarray:
    """Per bin, the bitmask of the bins whose centres are within `deg` of its own."""
    centres = _bin_centres(bins)
    cos = np.clip(centres @ centres.T, -1, 1)
    near = np.degrees(np.arccos(cos)) <= deg
    weights = np.left_shift(np.uint64(1), np.arange(bins * bins, dtype=np.uint64))
    return np.array([np.bitwise_or.reduce(weights[row]) for row in near], dtype=np.uint64)


class KnownModel(Protocol):
    """Per gaussian of the measured scan, how known it is from `eye` (0..1)."""

    def weights(self, positions: np.ndarray, eye: np.ndarray) -> np.ndarray: ...


@dataclass
class SeenDirections:
    """Support and seen directions per gaussian, from real cameras."""

    counts: np.ndarray
    bits: np.ndarray
    support_min: int = SUPPORT_MIN
    known_deg: float = KNOWN_DEG
    _masks: np.ndarray | None = None

    def weights(self, positions: np.ndarray, eye: np.ndarray) -> np.ndarray:
        if self._masks is None:
            self._masks = _neighbour_masks(self.known_deg)
        d = np.asarray(positions, np.float64) - np.asarray(eye, np.float64)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        near = self._masks[direction_bins(d)]
        seen = (self.bits & near) != 0
        return ((self.counts >= self.support_min) & seen).astype(np.float64)


@dataclass
class ForcedUnknown:
    """Another model's knowledge with some gaussians never known: the held-out check's
    region, whose look the generator must make although the scan has its shape."""

    base: KnownModel
    never: np.ndarray

    def weights(self, positions: np.ndarray, eye: np.ndarray) -> np.ndarray:
        return self.base.weights(positions, eye) * ~self.never


@dataclass
class ConeKnown:
    """No cameras: the view cones (observers inferred from the scan's finest detail)."""

    grid: vc.ConeGrid

    def weights(self, positions: np.ndarray, eye: np.ndarray) -> np.ndarray:
        texels = vc.lookup(
            self.grid.texels, self.grid.origin, self.grid.cell, self.grid.dims, positions
        )
        d = np.asarray(positions, np.float64) - np.asarray(eye, np.float64)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        return vc.visibility(texels, d)


Renderer = Callable[..., Frame]


def seen_directions(
    splats: Splats, views: Sequence[RealView], renderer: Renderer = render, width: int = 320
) -> SeenDirections:
    """Which real camera saw which gaussian: in its frame, in front of it, and no farther
    than the rendered surface at its pixel (`DEPTH_TOLERANCE`)."""
    counts = np.zeros(len(splats), np.int32)
    bits = np.zeros(len(splats), np.uint64)
    one = np.uint64(1)
    # A centre may lie up to about its own extent behind the surface it makes.
    reach = 2.0 * splats.scales.max(axis=1)
    for view in views:
        camera = scaled(view.camera, width)
        frame = renderer(splats, camera)
        uv, z = camera.project(splats.positions)
        u = np.floor(uv[:, 0]).astype(np.int64)
        v = np.floor(uv[:, 1]).astype(np.int64)
        inside = (z > 1e-3) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        rows = np.flatnonzero(inside)
        surface = frame.depth[v[rows], u[rows]]
        limit = surface * (1 + DEPTH_TOLERANCE) + np.minimum(reach[rows], 0.05 * surface)
        seen = np.isfinite(surface) & (z[rows] <= limit)
        rows = rows[seen]
        counts[rows] += 1
        d = splats.positions[rows] - camera.centre
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        bits[rows] |= np.left_shift(one, direction_bins(d).astype(np.uint64))
    return SeenDirections(counts, bits)


# --- the region, rays ---------------------------------------------------------------------------


def ray_box(camera: Camera, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Per pixel, whether its ray (forward of the camera) meets the box."""
    rays = camera.rays()
    origin = camera.centre
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / rays
        t0 = (np.asarray(low) - origin) * inv
        t1 = (np.asarray(high) - origin) * inv
    near = np.nanmax(np.minimum(t0, t1), axis=-1)
    far = np.nanmin(np.maximum(t0, t1), axis=-1)
    return (far >= np.maximum(near, 0.0)) & np.isfinite(far)


def grown(low: np.ndarray, high: np.ndarray, share: float) -> tuple[np.ndarray, np.ndarray]:
    pad = share * (np.asarray(high) - np.asarray(low))
    return np.asarray(low) - pad, np.asarray(high) + pad


# --- the scene ------------------------------------------------------------------------------------


@dataclass
class Scene:
    """What one fill works on."""

    name: str
    #: Every measured gaussian: what the views are carved against and the renders show.
    measured: Splats
    #: How known each measured gaussian is from a viewpoint.
    known: KnownModel
    #: The region to fill (local ENU box).
    roi: tuple[np.ndarray, np.ndarray]
    #: Real cameras to start from, carve with and distil against (not the held-out ones).
    views: list[RealView] = field(default_factory=list)
    #: Real cameras held out of everything (the ground truth of a held-out check).
    held_out: list[RealView] = field(default_factory=list)
    #: Where to start without cameras (the view cones' observers, at eye height).
    observers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    #: Measured gaussians hidden from the generator (an object whose ground is filled).
    hidden: np.ndarray | None = None
    #: Measured gaussians left out of the scene altogether (what only held-out cameras saw).
    dropped: np.ndarray | None = None
    prompt: str = PROMPT
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def shown_index(self) -> np.ndarray:
        off = np.zeros(len(self.measured), bool)
        if self.hidden is not None:
            off |= self.hidden
        if self.dropped is not None:
            off |= self.dropped
        return np.flatnonzero(~off)

    def present_index(self) -> np.ndarray:
        """What exists in this scene (everything but the dropped)."""
        if self.dropped is None:
            return np.arange(len(self.measured))
        return np.flatnonzero(~self.dropped)

    def withheld(self) -> Splats:
        """The measured gaussians whose look the generator is not given (hidden, dropped, or
        forced unknown): kept out of the start photo and of the real views' loss. Cached."""
        cache = self.__dict__.setdefault("_withheld", {})
        if "w" not in cache:
            off = np.zeros(len(self.measured), bool)
            never = getattr(self.known, "never", None)
            for m in (self.hidden, self.dropped, never):
                if m is not None:
                    off |= m
            cache["w"] = self.measured.take(np.flatnonzero(off))
        return cache["w"]

    def given_index(self) -> np.ndarray:
        """The shown gaussians whose look is given too (not forced unknown): what the
        distil keeps frozen beside the layer."""
        shown = self.shown_index()
        never = getattr(self.known, "never", None)
        return shown if never is None else shown[~never[shown]]

    def conditioning(self, extra: Splats | None = None) -> tuple[Splats, np.ndarray]:
        """What the generator is shown (the shown gaussians, then `extra`) and the shown
        ones' indices into `measured`; the same objects for the same `extra`, so a GPU
        renderer uploads them once. A forced-unknown gaussian keeps its shape and loses its
        look (mid grey): a held-out check's answer is never drawn into a frame."""
        cache = self.__dict__.setdefault("_conditioning", {})
        key = id(extra) if extra is not None and len(extra) else None
        if key not in cache:
            shown = self.shown_index()
            splats = self.measured.take(shown)
            never = getattr(self.known, "never", None)
            if never is not None and never[shown].any():
                colours = splats.colours.copy()
                colours[never[shown]] = 0.5
                splats = Splats(
                    splats.positions, splats.rotations, splats.scales, colours, splats.opacities
                )
            if key is not None:
                splats = Splats.concat([splats, extra])  # type: ignore[list-item]
            if len(cache) > 4:
                cache.clear()
            cache[key] = (splats, shown, extra)
        splats, shown, _ = cache[key]
        return splats, shown

    @property
    def centre(self) -> np.ndarray:
        return (self.roi[0] + self.roi[1]) / 2

    @property
    def radius(self) -> float:
        return 0.5 * float(np.linalg.norm(self.roi[1] - self.roi[0]))


# --- paths ------------------------------------------------------------------------------------------


@dataclass
class CameraPath:
    """A camera path: per frame a world-to-camera rotation and a centre; frame 0 at `start`."""

    rotations: np.ndarray  # (n, 3, 3)
    centres: np.ndarray  # (n, 3)
    start: RealView | None
    label: str
    end_direction: np.ndarray
    fov_deg: float = FOV_DEG

    def camera(self, k: int, width: int, height: int) -> Camera:
        focal = 0.5 * width / math.tan(math.radians(self.fov_deg) / 2)
        return Camera(self.rotations[k], self.centres[k], focal, width, height)

    def cameras(self, width: int, height: int) -> list[Camera]:
        return [self.camera(k, width, height) for k in range(len(self.centres))]

    def to_json(self) -> dict[str, object]:
        return {
            "label": self.label,
            "start": self.start.name if self.start else None,
            "frames": len(self.centres),
            "from": self.centres[0].round(3).tolist(),
            "to": self.centres[-1].round(3).tolist(),
            "endDirection": self.end_direction.round(3).tolist(),
        }


def _quaternion(r: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix."""
    m = r
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    theta = math.acos(min(1.0, dot))
    return (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta)


def _smooth(t: float) -> float:
    return t * t * (3 - 2 * t)


def look_rotation(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    return Camera.look_at(eye, target, up=tuple(up)).rotation  # type: ignore[arg-type]


def make_path(
    start_rotation: np.ndarray,
    start_centre: np.ndarray,
    target: np.ndarray,
    end_direction: np.ndarray,
    end_distance: float,
    frames: int,
    *,
    start: RealView | None,
    label: str,
    up: np.ndarray,
) -> CameraPath:
    """From a real camera's pose to `target + end_distance * end_direction` looking at
    `target`: the centre on an arc about the target (direction slerped, distance
    interpolated), the orientation slerped from the real camera's to the look-at one."""
    d0 = start_centre - target
    r0 = float(np.linalg.norm(d0))
    d0 = d0 / max(r0, 1e-9)
    e = end_direction / np.linalg.norm(end_direction)
    q0 = _quaternion(start_rotation)
    looked_at = start_centre + start_rotation[2] * r0
    rotations, centres = [], []
    for k in range(frames):
        t = _smooth(k / max(frames - 1, 1))
        direction = _slerp(d0, e, t)
        radius = (1 - t) * r0 + t * end_distance
        centre = target + radius * direction
        aim = (1 - t) * looked_at + t * target
        q_look = _quaternion(look_rotation(centre, aim, up))
        w = _smooth(min(1.0, (k / max(frames - 1, 1)) / 0.35))
        q = _slerp(q0, q_look, w)
        rotations.append(_rotation_from_quaternion(q))
        centres.append(centre)
    return CameraPath(np.array(rotations), np.array(centres), start, label, e)


def _rotation_from_quaternion(q: np.ndarray) -> np.ndarray:
    return _quat_matrix(*q)


def candidate_directions(elevations: Sequence[float], azimuths: int) -> np.ndarray:
    out = []
    for elev in elevations:
        for k in range(azimuths):
            a = 2 * math.pi * k / azimuths
            e = math.radians(elev)
            out.append([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return np.array(out)


@dataclass
class Masks:
    """One frame's pixels: given colour (known), to generate, to lift, known depth."""

    rgb: np.ndarray  # (h, w, 3) uint8: known colour; the scan's render where unknown; 0 in void
    generate: np.ndarray
    lift: np.ndarray
    depth: np.ndarray  # measured depth of covered pixels not lifted (anchors), nan elsewhere
    known_alpha: np.ndarray
    void: np.ndarray  # nothing measured here
    surface: np.ndarray  # the measured surface's depth wherever the scan covers, nan elsewhere


def frame_masks(
    scene: Scene,
    camera: Camera,
    renderer: Renderer,
    extra: Splats | None = None,
    known: np.ndarray | None = None,
) -> Masks:
    """The known, to-generate and to-lift pixels of one virtual frame (module docstring).
    `extra` (an earlier round's fill) is drawn as known; `known` (per measured gaussian, from
    `scene.known.weights` at this camera) is computed here when not given."""
    import cv2

    splats, shown = scene.conditioning(extra)
    if known is None:
        known = scene.known.weights(scene.measured.positions, camera.centre)
    weights = known[shown]
    if len(splats) > len(shown):
        weights = np.concatenate([weights, np.ones(len(splats) - len(shown))])
    full = renderer(splats, camera)
    seen = renderer(splats, camera, opacity_scale=weights)
    # Coverage judged on alpha dilated a little: the CPU renderer's samples leave a
    # measured surface speckled, and a speck is neither a hole nor unknown.
    covered = tf._covered(full.alpha)
    unknown = tf.clean_mask(covered & ~tf._covered(seen.alpha))
    void = ~covered
    region = ray_box(camera, *scene.roi)
    todo = unknown | void
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * GENERATE_GROW_PX + 1, 2 * GENERATE_GROW_PX + 1)
    )
    generate = cv2.dilate(todo.astype(np.uint8), kernel) > 0
    lift = todo & region
    # Known colour: the seen render un-premultiplied, its sampling gaps filled from around.
    a = seen.alpha.astype(np.float32)
    w = cv2.GaussianBlur(a, (0, 0), 1.0)
    smooth = (
        cv2.GaussianBlur(seen.rgb.astype(np.float32), (0, 0), 1.0) / np.maximum(w, 1e-4)[..., None]
    )
    colour = np.where((a >= COVERED)[..., None], seen.rgb / np.maximum(a, 1e-6)[..., None], smooth)
    # Where the scan has a surface it did not see from here, its render stays as a hint (VACE
    # may repaint from it); void is black.
    af = full.alpha.astype(np.float32)
    wf = cv2.GaussianBlur(af, (0, 0), 1.0)
    full_colour = (
        cv2.GaussianBlur(full.rgb.astype(np.float32), (0, 0), 1.0) / np.maximum(wf, 1e-4)[..., None]
    )
    rgb = np.where(unknown[..., None], np.clip(full_colour, 0, 1), np.clip(colour, 0, 1))
    rgb = np.where(void[..., None], 0.0, rgb)
    # The depth a generated frame's depth is fitted to: every covered pixel outside what is
    # lifted -- known or not, the scan measured a surface there (only not its look from here).
    solid = (full.alpha >= COVERED) & np.isfinite(full.depth)
    depth = np.where(covered & ~lift & solid, full.depth, np.nan).astype(np.float32)
    surface = np.where(covered & solid, full.depth, np.nan).astype(np.float32)
    return Masks(tf.to_u8(rgb), generate, lift, depth, seen.alpha, void, surface)


def unknown_score(
    scene: Scene, camera: Camera, renderer: Renderer, extra: Splats | None
) -> tuple[int, float]:
    """How many pixels of the region a view would have to lift, and the share of what the
    scan shows there that is known (the context a model would have; empty sky does not count
    against it)."""
    m = frame_masks(scene, camera, renderer, extra)
    shown = ~m.void
    return int(m.lift.sum()), float((~m.generate & shown).sum() / max(int(shown.sum()), 1))


def _fit_distance(radius: float, fov_deg: float, aspect: float) -> float:
    """How far a camera stands for a sphere of `radius` to fill the frame's height."""
    vertical = 2 * math.atan(math.tan(math.radians(fov_deg) / 2) / aspect)
    return radius / math.tan(vertical / 2)


def plan_paths(
    scene: Scene,
    count: int,
    frames: int,
    renderer: Renderer,
    *,
    extra: Splats | None = None,
    avoid: Sequence[np.ndarray] = (),
    used_starts: Sequence[str] = (),
    elevations: Sequence[float] = (30.0, 50.0, 70.0, 85.0),
    azimuths: int = 12,
    probe: tuple[int, int] = (192, 108),
    min_separation_deg: float = 60.0,
    label: str = "r1",
) -> tuple[list[CameraPath], list[dict]]:
    """`count` paths: the end viewpoints (on a sphere about the region, at `elevations`)
    that would lift the most unknown pixels, at least `min_separation_deg` apart and from
    the directions in `avoid`; each from the real camera (or observer) nearest in direction
    to it, preferring one not in `used_starts`. Returns the paths and every candidate's
    score."""
    centre, radius = scene.centre, max(scene.radius, 1e-3)
    aspect = probe[0] / probe[1]
    fit = 1.15 * _fit_distance(radius, FOV_DEG, aspect)
    scores = []
    for d in candidate_directions(elevations, azimuths):
        eye = centre + fit * d
        camera = Camera.look_at(
            eye, centre, fov_deg=FOV_DEG, width=probe[0], height=probe[1], up=tuple(scene.up)
        )
        lift_px, context = unknown_score(scene, camera, renderer, extra)
        scores.append((lift_px, context, d))
    report = [
        {"direction": d.round(3).tolist(), "unknownPx": s, "known": round(c, 3)}
        for s, c, d in scores
    ]
    # Only views that keep enough of the frame known: a model given nothing invents the
    # whole frame (run 37381466964's clips, from 50 deg up).
    usable = [(s, d) for s, c, d in scores if c >= MIN_CONTEXT]
    if not usable:  # nowhere keeps that much: the best-known quarter of the views
        floor = float(np.quantile([c for _, c, _ in scores], 0.75)) if scores else 0.0
        usable = [(s, d) for s, c, d in scores if c >= floor]
    chosen: list[np.ndarray] = []
    for score, d in sorted(usable, key=lambda sd: -sd[0]):
        if score <= 0 or len(chosen) >= count:
            break
        others = [*avoid, *chosen]
        if all(
            math.degrees(math.acos(np.clip(d @ o, -1, 1))) >= min_separation_deg for o in others
        ):
            chosen.append(d)
    paths = []
    taken = set(used_starts)
    for k, d in enumerate(chosen):
        start_rot, start_centre, start = _start_for(scene, d, taken)
        if start is not None:
            taken.add(start.name)
        start_distance = float(np.linalg.norm(start_centre - centre))
        end = float(np.clip(fit, 0.7 * start_distance, 1.6 * start_distance))
        paths.append(
            make_path(
                start_rot,
                start_centre,
                centre,
                d,
                end,
                frames,
                start=start,
                label=f"{label}p{k}",
                up=scene.up,
            )
        )
    return paths, report


def _start_for(
    scene: Scene, direction: np.ndarray, taken: set[str]
) -> tuple[np.ndarray, np.ndarray, RealView | None]:
    """The real camera that looks at the region from nearest `direction` (not already used,
    when another will do); else the observer point nearest in direction, looking at it."""
    centre = scene.centre
    best: tuple[float, RealView] | None = None
    for view in scene.views:
        to = centre - view.camera.centre
        dist = float(np.linalg.norm(to))
        if dist < 1e-6:
            continue
        # The region's centre must be in front of it and inside its frame.
        uv, z = view.camera.project(centre[None])
        if z[0] <= 0 or not (
            0 <= uv[0, 0] < view.camera.width and 0 <= uv[0, 1] < view.camera.height
        ):
            continue
        away = (view.camera.centre - centre) / dist
        angle = math.degrees(math.acos(np.clip(away @ direction, -1, 1)))
        angle += 15.0 if view.name in taken else 0.0
        if best is None or angle < best[0]:
            best = (angle, view)
    if best is not None:
        view = best[1]
        return view.camera.rotation, view.camera.centre, view
    if len(scene.observers):
        eyes = scene.observers + scene.up * tf.EYE_HEIGHT_M
        away = eyes - centre
        dist = np.linalg.norm(away, axis=1)
        ok = dist > 0.5 * scene.radius
        if ok.any():
            cos = (away[ok] / dist[ok, None]) @ direction
            eye = eyes[ok][int(np.argmax(cos))]
            return look_rotation(eye, centre, scene.up), eye, None
    eye = centre + 2.0 * scene.radius * np.array([direction[0], direction[1], 0.2])
    return look_rotation(eye, centre, scene.up), eye, None


# --- clips ------------------------------------------------------------------------------------------


@dataclass
class Clip:
    """A path rendered at one size: what a model is given and asked."""

    path: CameraPath
    size: tuple[int, int]
    cameras: list[Camera]
    frames: np.ndarray  # (n, h, w, 3) uint8: known pixels (and the photo in frame 0)
    generate: np.ndarray  # (n, h, w) bool
    lift: np.ndarray  # (n, h, w) bool
    depth: np.ndarray  # (n, h, w) float: the anchors (`Masks.depth`), nan elsewhere
    photo_first: bool
    void: np.ndarray | None = None  # (n, h, w) bool: nothing measured
    surface: np.ndarray | None = None  # (n, h, w) float: the measured surface's depth


def remap_photo(view: RealView, camera: Camera) -> tuple[np.ndarray, np.ndarray] | None:
    """The real photo seen through `camera` (same pose, other intrinsics): colours and the
    pixels it covers."""
    import cv2

    src = view.camera
    # Downscaled first to about the virtual camera's pixel size (remap only interpolates).
    width = src.width
    if camera.focal < 0.75 * src.focal:
        width = max(16, round(src.width * camera.focal / src.focal))
    photo = view.photo(width=width)
    if photo is None:
        return None
    src = scaled(src, photo.shape[1], photo.shape[0])
    v, u = np.mgrid[0 : camera.height, 0 : camera.width].astype(np.float32)
    x = (u + 0.5 - camera.width / 2) / camera.focal
    y = (v + 0.5 - camera.height / 2) / camera.focal
    map_x = (x * src.focal + src.width / 2 - 0.5).astype(np.float32)
    map_y = (y * src.focal + src.height / 2 - 0.5).astype(np.float32)
    rgb = cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR)
    cover = (map_x >= 0) & (map_x <= src.width - 1) & (map_y >= 0) & (map_y <= src.height - 1)
    return rgb, cover


def render_clip(
    scene: Scene,
    path: CameraPath,
    size: tuple[int, int],
    renderer: Renderer,
    extra: Splats | None = None,
    known_cache: dict[int, np.ndarray] | None = None,
) -> Clip:
    """`path` rendered at `size`: per frame the known pixels and the masks; frame 0 is the
    start camera's photo where it has one (all known)."""
    w, h = size
    cameras = path.cameras(w, h)
    n = len(cameras)
    frames = np.zeros((n, h, w, 3), np.uint8)
    generate = np.zeros((n, h, w), bool)
    lift = np.zeros((n, h, w), bool)
    depth = np.full((n, h, w), np.nan, np.float32)
    void = np.zeros((n, h, w), bool)
    surface = np.full((n, h, w), np.nan, np.float32)
    photo_first = False
    for k, camera in enumerate(cameras):
        known = None
        if known_cache is not None:
            if k not in known_cache:
                known_cache[k] = scene.known.weights(scene.measured.positions, camera.centre)
            known = known_cache[k]
        m = frame_masks(scene, camera, renderer, extra, known)
        frames[k], generate[k], lift[k], depth[k] = m.rgb, m.generate, m.lift, m.depth
        void[k], surface[k] = m.void, m.surface
        if k == 0 and path.start is not None:
            seen = remap_photo(path.start, camera)
            if seen is not None:
                rgb, cover = seen
                # The photo shows what the scene withholds (a hidden object, a dropped top):
                # those pixels stay to generate.
                withheld = scene.withheld()
                if len(withheld):
                    cover &= ~tf._covered(renderer(withheld, camera).alpha * 4.0)
                frames[0] = np.where(cover[..., None], rgb, frames[0])
                generate[0] &= ~cover
                lift[0] &= ~cover
                void[0] &= ~cover
                photo_first = bool(cover.mean() > 0.5)
    return Clip(path, size, cameras, frames, generate, lift, depth, photo_first, void, surface)


def keyframes(clip: Clip) -> list[int]:
    start = KEYFRAME_EVERY if clip.photo_first else 0
    return list(range(start, len(clip.cameras), KEYFRAME_EVERY))


# --- generators ---------------------------------------------------------------------------------


@dataclass
class ClipRequest:
    clip: Clip
    prompt: str
    seed: int
    label: str


@dataclass
class ClipResult:
    #: (n, h, w, 3) uint8 as the model drew them; None when the call failed (`info["error"]`).
    frames: np.ndarray | None
    info: dict[str, object] = field(default_factory=dict)


class ClipFiller(Protocol):
    """Draws the pixels to generate of each clip. `start` sends the requests and returns a
    function that waits for and returns the results (so several fillers run at once)."""

    name: str
    key: str
    size: tuple[int, int]

    def start(self, requests: Sequence[ClipRequest]) -> Callable[[], list[ClipResult]]: ...


@dataclass
class TeleaClipFiller:
    """The CPU stand-in: every frame inpainted on its own (OpenCV Telea)."""

    size: tuple[int, int] = (160, 96)
    name: str = "opencv-telea-frames"
    key: str = "telea"

    def start(self, requests: Sequence[ClipRequest]) -> Callable[[], list[ClipResult]]:
        def run() -> list[ClipResult]:
            out = []
            for r in requests:
                f = tf.InpaintFiller()
                frames = np.stack(
                    [
                        f.fill(r.clip.frames[k], r.clip.generate[k])[0]
                        for k in range(len(r.clip.frames))
                    ]
                )
                out.append(ClipResult(frames, {"model": self.name, "seconds": 0.0}))
            return out

        return run


#: `submit(class, method, request) -> handle`, `wait(handle) -> response`: a GPU class's
#: method spawned (Modal) or called (tests).
Submit = Callable[[str, str, dict], Any]
Wait = Callable[[Any], dict]

#: The Modal class that runs each video model (`infra/modal/fill.py`).
VIDEO_CLASSES = {"vace": "FillVace", "wan22": "FillWan22", "cosmos": "FillCosmos"}
#: The name each generator's layer carries (`extras.evidence.filler`) and its folder's slug.
GENERATOR_NAMES = {
    "vace": "wan2.1-vace-1.3b",
    "wan22": "wan2.2-ti2v-5b",
    "cosmos": "cosmos-predict2-2b",
    "lama": "inpaint-lama",
    "telea": "opencv-telea-frames",
}


@dataclass
class RemoteClipFiller:
    """A video model on a GPU (`video_fill_models`), one call per clip."""

    key: str
    submit: Submit
    wait: Wait
    steps: int | None = None
    #: Added to the model's own negative prompt (what a hole must not be painted with).
    negative: str = ""
    name: str = ""
    size: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        import video_fill_models as vfm

        spec = vfm.MODELS[self.key]
        self.size = spec.size
        if not self.name:
            self.name = GENERATOR_NAMES[self.key]

    def start(self, requests: Sequence[ClipRequest]) -> Callable[[], list[ClipResult]]:
        import video_fill_models as vfm

        handles = []
        for r in requests:
            body: dict[str, object] = {
                "clip": vfm.pack_clip(r.clip.frames, r.clip.generate, r.clip.void),
                "prompt": r.prompt,
                "seed": int(r.seed),
            }
            if self.negative:
                body["negative"] = f"{vfm.NEGATIVE}, {self.negative}"
            if self.steps:
                body["steps"] = int(self.steps)
            handles.append(self.submit(VIDEO_CLASSES[self.key], "fill_clip", body))

        def collect() -> list[ClipResult]:
            # Every call is waited on, so none is left running unseen; a failed one is
            # reported and its clip skipped.
            out = []
            for h in handles:
                try:
                    response = self.wait(h)
                except Exception as error:  # noqa: BLE001 - recorded, the others go on
                    out.append(ClipResult(None, {"error": repr(error)[:1500]}))
                    continue
                if "error" in response:  # failed on the GPU: its seconds still count
                    out.append(ClipResult(None, dict(response)))
                    continue
                frames, _ = vfm.unpack_clip(response["clip"])
                info = {k: v for k, v in response.items() if k != "clip"}
                out.append(ClipResult(frames, info))
            return out

        return collect


@dataclass
class PerViewClipFiller:
    """A per-image filler (`teacher_fill.Filler`, e.g. `world_model_client.GenerativeFiller`)
    on each keyframe on its own: today's per-view inpainting, the baseline."""

    filler: Any
    size: tuple[int, int] = (1280, 704)
    name: str = ""
    key: str = "perview"

    def __post_init__(self) -> None:
        if not self.name:
            self.name = str(getattr(self.filler, "name", "per-view"))

    def start(self, requests: Sequence[ClipRequest]) -> Callable[[], list[ClipResult]]:
        def run() -> list[ClipResult]:
            out = []
            for r in requests:
                if hasattr(self.filler, "context"):
                    self.filler.context = {"prompt": r.prompt}
                frames = r.clip.frames.copy()
                for k in keyframes(r.clip):
                    if not r.clip.generate[k].any():
                        continue
                    extra = (
                        {"void": np.zeros_like(r.clip.generate[k])}
                        if getattr(self.filler, "reads_void", False)
                        else {}
                    )
                    frames[k] = self.filler.fill(r.clip.frames[k], r.clip.generate[k], **extra)[0]
                out.append(ClipResult(frames, {"model": self.name}))
            return out

        return run


# --- gate, depth, lift ----------------------------------------------------------------------------


def gate_clip(clip: Clip, frames: np.ndarray) -> tuple[np.ndarray, list[float | None]]:
    """Per frame, whether the model kept the known pixels (blurred PSNR >= `GATE_DB`)."""
    keep = np.ones(len(frames), bool)
    scores: list[float | None] = []
    for k in range(len(frames)):
        known = ~clip.generate[k]
        if known.mean() < GATE_MIN_KNOWN:
            scores.append(None)
            continue
        score = min(tf.psnr(tf._blur(frames[k]), tf._blur(clip.frames[k]), known), tf.GATE_CAP_DB)
        scores.append(round(score, 2))
        keep[k] = score >= GATE_DB
    return keep, scores


def composite(clip: Clip, frames: np.ndarray) -> np.ndarray:
    """The generated pixels where asked, the known ones where given."""
    return np.where(clip.generate[..., None], frames, clip.frames).astype(np.uint8)


class DepthModel(Protocol):
    """Relative inverse depth (any affine scale) per frame; `cameras` are the frames'
    (a learned model ignores them; a test's oracle renders with them)."""

    name: str

    def disparity(self, frames: np.ndarray, cameras: Sequence[Camera] = ()) -> np.ndarray: ...


@dataclass
class DepthAnything:
    """Depth Anything V2 Small (Apache-2.0) through transformers: relative inverse depth."""

    model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: str = "cuda"
    name: str = "depth-anything-v2-small"
    _pipe: Any = None

    def disparity(self, frames: np.ndarray, cameras: Sequence[Camera] = ()) -> np.ndarray:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if self._pipe is None:
            processor = AutoImageProcessor.from_pretrained(self.model)
            net = AutoModelForDepthEstimation.from_pretrained(self.model).to(self.device).eval()
            self._pipe = (processor, net)
        processor, net = self._pipe
        out = []
        for f in frames:
            inputs = processor(images=Image.fromarray(f), return_tensors="pt").to(self.device)
            with torch.no_grad():
                pred = net(**inputs).predicted_depth
            pred = torch.nn.functional.interpolate(
                pred[:, None], size=f.shape[:2], mode="bicubic", align_corners=False
            )[0, 0]
            out.append(pred.float().cpu().numpy())
        return np.stack(out)


def align_depth(
    disparity: np.ndarray, depth: np.ndarray, known: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray | None, dict[str, float]]:
    """Metric depth for `target` pixels: the affine map `1 / z = a * disparity + b` fitted to
    the measured depth of the `known` pixels (weighted to those near the target, two rounds
    dropping the worst fifth). None when too few known pixels or the fit is poor."""
    import cv2

    ok = known & np.isfinite(depth) & (depth > 0)
    if ok.sum() < DEPTH_MIN_PIXELS or not target.any():
        return None, {"knownPx": int(ok.sum())}
    gap = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.exp(-gap / (0.1 * max(target.shape)))[ok]
    x = disparity[ok].astype(np.float64)
    y = 1.0 / depth[ok]
    use = np.ones(x.size, bool)
    a = b = 0.0
    for _ in range(3):
        sw = weight[use]
        design = np.stack([x[use], np.ones(use.sum())], 1) * np.sqrt(sw)[:, None]
        (a, b), *_ = np.linalg.lstsq(design, y[use] * np.sqrt(sw), rcond=None)
        err = np.abs(a * x + b - y) / np.maximum(y, 1e-9)
        use = err <= np.quantile(err[use], 0.8)
    fitted = a * x + b
    residual = float(np.median(np.abs(fitted - y) / np.maximum(y, 1e-9)))
    info = {"knownPx": int(ok.sum()), "a": float(a), "b": float(b), "residual": round(residual, 4)}
    if a <= 0 or residual > DEPTH_MAX_RESIDUAL:
        return None, info
    inv = a * disparity + b
    out = np.where(target & (inv > 1e-9), 1.0 / np.maximum(inv, 1e-9), np.nan)
    limit = 4.0 * float(np.nanmax(depth[ok]))
    out = np.where(out <= limit, out, np.nan)
    return out, info


@dataclass
class Lifted:
    splats: Splats
    confidence: np.ndarray
    cameras: list[Camera]
    report: list[dict]


def lift_frame(
    camera: Camera,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    distance: np.ndarray,
    region: tuple[np.ndarray, np.ndarray],
    stride: int = LIFT_STRIDE,
    extra_confidence: np.ndarray | None = None,
) -> tuple[Splats, np.ndarray]:
    """`mask`'s pixels (every `stride`-th) as discs facing the camera at `depth`, inside the
    `region`; confidence from the distance to anything known (halving every
    `teacher_fill.CONFIDENCE_PX` or a quarter of the deepest, whichever is longer) times
    `extra_confidence` (per pixel)."""
    sel = np.zeros_like(mask)
    sel[::stride, ::stride] = mask[::stride, ::stride] & np.isfinite(depth[::stride, ::stride])
    v, u = np.nonzero(sel)
    empty = Splats(*(np.zeros((0, k)) for k in (3, 4, 3, 3)), np.zeros(0))
    if v.size == 0:
        return empty, np.zeros(0)
    z = depth[v, u]
    local = np.column_stack(
        [
            (u + 0.5 - camera.width / 2) * z / camera.focal,
            (v + 0.5 - camera.height / 2) * z / camera.focal,
            z,
        ]
    )
    positions = camera.centre + local @ camera.rotation
    inside = np.all((positions >= region[0]) & (positions <= region[1]), axis=1)
    if not inside.any():
        return empty, np.zeros(0)
    v, u, z, positions = v[inside], u[inside], z[inside], positions[inside]
    footprint = stride * z / camera.focal
    halving = max(tf.CONFIDENCE_PX, 0.25 * float(distance[mask].max()) if mask.any() else 0.0)
    confidence = np.exp(-distance[v, u] / halving * math.log(2))
    if extra_confidence is not None:
        confidence = confidence * extra_confidence[v, u]
    splats = Splats(
        positions,
        tf._disc_rotations(camera.centre - positions),
        np.column_stack([0.6 * footprint, 0.6 * footprint, 0.1 * footprint]),
        rgb[v, u].astype(np.float64) / 255.0,
        0.95 * np.clip(confidence, 0.05, 1.0),
    )
    return splats, confidence


def seed_agreement(
    a_rgb: np.ndarray, a_depth: np.ndarray, b_rgb: np.ndarray, b_depth: np.ndarray
) -> np.ndarray:
    """Per pixel, how far two seeds' fills of the same frame agree: colour and depth."""
    colour = np.sqrt(
        np.mean(((a_rgb.astype(np.float64) - b_rgb.astype(np.float64)) / 255.0) ** 2, -1)
    )
    rel = np.abs(a_depth - b_depth) / np.maximum(np.abs(a_depth), 1e-9)
    rel = np.where(np.isfinite(rel), rel, 1.0)
    return np.exp(-colour / SEED_COLOUR * math.log(2)) * np.exp(-rel / SEED_DEPTH * math.log(2))


@dataclass
class ClipFill:
    """One clip's generated keyframes, ready to lift (and to distil against)."""

    clip: Clip
    frames: np.ndarray  # composite, (n, h, w, 3)
    keep: np.ndarray
    gate: list[float | None]
    depth: dict[int, np.ndarray]
    depth_info: dict[int, dict]
    info: dict


def prepare_fill(clip: Clip, result: ClipResult, depth_model: DepthModel) -> ClipFill:
    """Gate the frames, composite, and fit each keyframe's depth."""
    frames = result.frames
    if frames.shape != clip.frames.shape:
        import cv2

        frames = np.stack([cv2.resize(f, clip.size, interpolation=cv2.INTER_AREA) for f in frames])
    keep, gate = gate_clip(clip, frames)
    full = composite(clip, frames)
    ks = [k for k in keyframes(clip) if keep[k] and clip.lift[k].any()]
    depth: dict[int, np.ndarray] = {}
    info: dict[int, dict] = {}
    if ks:
        disparity = depth_model.disparity(full[ks], [clip.cameras[k] for k in ks])
        for j, k in enumerate(ks):
            d, i = align_depth(
                disparity[j], clip.depth[k], np.isfinite(clip.depth[k]), clip.lift[k]
            )
            info[k] = i
            if d is not None:
                depth[k] = d
    return ClipFill(clip, full, keep, gate, depth, info, result.info)


def lift_fill(
    fill: ClipFill, region: tuple[np.ndarray, np.ndarray], other: ClipFill | None = None
) -> Lifted:
    """Every keyframe with a depth lifted; with `other` (another seed of the same clip),
    confidence also by how far the two agree."""
    import cv2

    parts, confs, cams, report = [], [], [], []
    for k, mono in sorted(fill.depth.items()):
        clip = fill.clip
        # Where the scan has a surface (only not seen from here), the fill is painted onto
        # it; the generated views' depth places only what the scan has nothing of.
        depth = mono
        if clip.surface is not None:
            depth = np.where(np.isfinite(clip.surface[k]), clip.surface[k], mono)
        mask = clip.lift[k]
        known = ~clip.generate[k]
        distance = cv2.distanceTransform((~known).astype(np.uint8), cv2.DIST_L2, 5)
        quality = math.exp(-fill.depth_info[k].get("residual", 0.0) / 0.1 * math.log(2))
        extra = np.full(mask.shape, quality)
        agreement = None
        if other is not None and k in other.depth:
            theirs = other.depth[k]
            if clip.surface is not None:
                theirs = np.where(np.isfinite(clip.surface[k]), clip.surface[k], theirs)
            agreement = seed_agreement(fill.frames[k], depth, other.frames[k], theirs)
            extra = extra * agreement
        s, c = lift_frame(
            clip.cameras[k], fill.frames[k], depth, mask, distance, region, extra_confidence=extra
        )
        if len(s):
            parts.append(s)
            confs.append(c)
            cams.append(clip.cameras[k])
        report.append(
            {
                "frame": k,
                "liftPx": int(mask.sum()),
                "lifted": len(s),
                "depthResidual": fill.depth_info[k].get("residual"),
                **(
                    {"seedAgreement": round(float(agreement[mask].mean()), 3)}
                    if agreement is not None
                    else {}
                ),
            }
        )
    if not parts:
        empty = Splats(*(np.zeros((0, k)) for k in (3, 4, 3, 3)), np.zeros(0))
        return Lifted(empty, np.zeros(0), [], report)
    return Lifted(Splats.concat(parts), np.concatenate(confs), cams, report)


def merge(lifted: Sequence[Lifted]) -> Lifted:
    parts = [x for x in lifted if len(x.splats)]
    if not parts:
        empty = Splats(*(np.zeros((0, k)) for k in (3, 4, 3, 3)), np.zeros(0))
        return Lifted(empty, np.zeros(0), [], [])
    return Lifted(
        Splats.concat([x.splats for x in parts]),
        np.concatenate([x.confidence for x in parts]),
        [c for x in parts for c in x.cameras],
        [r for x in lifted for r in x.report],
    )


def thin(lifted: Lifted, budget: int) -> Lifted:
    """The views fused: one gaussian per voxel (the voxel the median disc's size), placed and
    shaped as the voxel's most confident disc and coloured with the confidence-weighted mean
    of every disc there -- each frame's colour for that bit of surface, so one frame's
    stray pixel does not become a speck (run 37383986439 kept one frame's colour per voxel:
    the held-out top came out speckled); then at most `budget`, the most confident."""
    s = lifted.splats
    if len(s) == 0:
        return lifted
    size = max(float(np.median(s.scales[:, 0])) / 0.6, 1e-6)
    keys = np.floor(s.positions / size).astype(np.int64)
    order = np.argsort(-lifted.confidence, kind="stable")
    _, first, group = np.unique(keys[order], axis=0, return_index=True, return_inverse=True)
    group = np.asarray(group).reshape(-1)
    w = np.maximum(lifted.confidence[order], 1e-6)
    total = np.bincount(group, w)
    colours = (
        np.column_stack([np.bincount(group, w * s.colours[order, c]) for c in range(3)])
        / total[:, None]
    )
    keep = order[first]
    fused = s.take(keep)
    fused = Splats(fused.positions, fused.rotations, fused.scales, colours, fused.opacities)
    confidence = lifted.confidence[keep]
    chosen = np.argsort(keep, kind="stable")
    if chosen.size > budget:
        chosen = np.sort(np.argsort(-confidence, kind="stable")[:budget])
        chosen = chosen[np.argsort(keep[chosen], kind="stable")]
    return Lifted(fused.take(chosen), confidence[chosen], lifted.cameras, lifted.report)


# --- carving ---------------------------------------------------------------------------------------


def carving_cameras(scene: Scene, width: int = 320) -> list[Camera]:
    """The cameras whose sight lines carve: the real ones (not held out), else six faces of
    a cube at each of the observers nearest the region (eye height)."""
    if scene.views:
        return [scaled(v.camera, width) for v in scene.views]
    if not len(scene.observers):
        return []
    eyes = scene.observers + scene.up * tf.EYE_HEIGHT_M
    order = np.argsort(np.linalg.norm(eyes - scene.centre, axis=1))[:24]
    out = []
    size = max(64, width // 2)
    for eye in eyes[order]:
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            up = (0.0, 1.0, 0.0) if abs(d[2]) else (0.0, 0.0, 1.0)
            out.append(
                Camera.look_at(
                    eye, eye + np.asarray(d, float), fov_deg=90.0, width=size, height=size, up=up
                )
            )
    return out


def carve(
    positions: np.ndarray,
    measured: Splats,
    cameras: Sequence[Camera],
    renderer: Renderer,
    region_size: float,
) -> np.ndarray:
    """Which points survive: a point is carved when some camera's ray through it went on to
    a measured surface behind it (the pixel and its 3x3 neighbours covered, the point nearer
    than the nearest of their depths by `CARVE_SHARE` plus `CARVE_REGION` of the region) --
    the camera saw through where it stands."""
    from scipy.ndimage import maximum_filter, minimum_filter

    keep = np.ones(len(positions), bool)
    if not len(positions):
        return keep
    margin = CARVE_REGION * region_size
    for camera in cameras:
        frame = renderer(measured, camera)
        # Conservative at edges: the point must be in front of the *nearest* surface among
        # the pixel's 3x3 neighbours, and every one of them covered (coverage judged as
        # `teacher_fill._covered` does, so the CPU renderer's speckle counts as covered).
        depth = np.where(np.isfinite(frame.depth), frame.depth, np.inf)
        far = minimum_filter(depth, size=3)
        covered = maximum_filter(frame.alpha, size=3) >= COVERED
        solid = minimum_filter(covered.astype(np.uint8), size=3) > 0
        uv, z = camera.project(positions)
        u = np.floor(uv[:, 0]).astype(np.int64)
        v = np.floor(uv[:, 1]).astype(np.int64)
        inside = (z > 1e-3) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        rows = np.flatnonzero(inside & keep)
        if rows.size == 0:
            continue
        surf = far[v[rows], u[rows]]
        through = solid[v[rows], u[rows]] & (z[rows] < surf * (1 - CARVE_SHARE) - margin)
        keep[rows[through]] = False
    return keep


# --- fuse ---------------------------------------------------------------------------------------------


def _resize_view(camera: Camera, image: np.ndarray, mask: np.ndarray, width: int):
    import cv2

    if camera.width <= width:
        return camera, image, mask
    small = scaled(camera, width)
    image = cv2.resize(image, (small.width, small.height), interpolation=cv2.INTER_AREA)
    mask = (
        cv2.resize(
            mask.astype(np.uint8), (small.width, small.height), interpolation=cv2.INTER_NEAREST
        )
        > 0
    )
    return small, image, mask


def distil_request(
    scene: Scene,
    lifted: Lifted,
    fills: Sequence[ClipFill],
    renderer: Renderer,
    iterations: int,
    *,
    width: int = 640,
    real: int = 12,
) -> dict:
    """`distill_fill.run`'s request: the generated keyframes (their lift masks, weight
    `GENERATED_WEIGHT`) and up to `real` real photos (weight 1, on what the measured scan
    covers there and no withheld gaussian does), the measured scan (withheld gaussians out)
    cropped to the layer's neighbourhood and frozen."""
    import distill_fill as df

    cams, images, masks, weights, outside = [], [], [], [], []
    for fill in fills:
        for k in sorted(fill.depth):
            c, im, m = _resize_view(fill.clip.cameras[k], fill.frames[k], fill.clip.lift[k], width)
            cams.append(c.to_json())
            images.append(im)
            masks.append(m)
            weights.append(GENERATED_WEIGHT)
            outside.append(GENERATED_OUTSIDE)
    shown = scene.measured.take(scene.given_index())
    hidden = scene.withheld()
    if scene.views and real > 0:
        centre = scene.centre
        ranked = sorted(
            scene.views,
            key=lambda v: float(np.linalg.norm(v.camera.centre - centre)),
        )
        for view in ranked[:real]:
            photo = view.photo(width=min(width, view.camera.width))
            if photo is None:
                continue
            camera = scaled(view.camera, photo.shape[1], photo.shape[0])
            mask = renderer(shown, camera).alpha >= COVERED
            if hidden is not None and len(hidden):
                mask &= renderer(hidden, camera).alpha < 0.05
            cams.append(camera.to_json())
            images.append(photo)
            masks.append(mask)
            weights.append(1.0)
            outside.append(0.0)
    if not cams:
        raise ValueError("nothing to distil against")
    low, high = lifted.splats.positions.min(axis=0), lifted.splats.positions.max(axis=0)
    pad = 0.25 * float(np.max(high - low)) + 1e-6
    near = np.all((shown.positions >= low - pad) & (shown.positions <= high + pad), axis=1)
    # Views differ in size: pad each image and mask onto the largest frame (masked off).
    h = max(i.shape[0] for i in images)
    w = max(i.shape[1] for i in images)
    stack = np.zeros((len(images), h, w, 3), np.uint8)
    mstack = np.zeros((len(images), h, w), bool)
    for j, (im, m) in enumerate(zip(images, masks, strict=True)):
        stack[j, : im.shape[0], : im.shape[1]] = im
        mstack[j, : m.shape[0], : m.shape[1]] = m
    return {
        "measured": df.pack_scan(tf._arrays(shown.take(np.flatnonzero(near)))),
        "init": df.pack_scan(tf._arrays(lifted.splats)),
        "views": df.pack_views(cams, stack, mstack, weights=weights, outside=outside),
        "iterations": iterations,
    }


# --- packaging, provenance ---------------------------------------------------------------------------


def evidence_extra(
    filler: ClipFiller, *, rounds: int, seeds: int, clips: int, carved: int, frames: int, depth: str
) -> dict[str, object]:
    return {
        "generator": filler.name,
        "rounds": rounds,
        "seeds": seeds,
        "clips": clips,
        "framesLifted": frames,
        "carved": carved,
        "depth": depth,
        "provenance": "inferred-generated",
    }


def label_image(image: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 10 + 9 * len(text)), 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def grid(rows: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """Images (same size) as a grid."""
    return np.concatenate([np.concatenate(list(r), axis=1) for r in rows], axis=0)


def contact_sheet(fill: ClipFill, count: int = 6) -> np.ndarray:
    """A clip's frames: what the model was given (the mask outlined) over what it drew."""
    import cv2

    n = len(fill.frames)
    picks = np.linspace(0, n - 1, count).round().astype(int)
    top, bottom = [], []
    for k in picks:
        given = fill.clip.frames[k].copy()
        edge = cv2.morphologyEx(
            fill.clip.generate[k].astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        )
        given[edge > 0] = (255, 0, 255)
        small = lambda x: cv2.resize(x, (320, round(320 * x.shape[0] / x.shape[1])))
        top.append(label_image(small(given), f"given f{k}"))
        g = fill.gate[k]
        bottom.append(
            label_image(small(fill.frames[k]), f"drawn f{k} gate {g if g is not None else '-'}")
        )
    return grid([top, bottom])


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=_json_default), encoding="utf-8")


def _json_default(o: object) -> object:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)


# --- the scene from a tileset -------------------------------------------------------------------------


def subtree_ids(instances: Sequence[dict], root: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for i in instances:
        if i.get("parent"):
            children.setdefault(int(i["parent"]), []).append(int(i["id"]))
    out, stack = [], [int(root)]
    while stack:
        k = stack.pop()
        out.append(k)
        stack.extend(children.get(k, []))
    return sorted(out)


def instance_box(instances: Sequence[dict], root: int) -> tuple[np.ndarray, np.ndarray]:
    """The bounds of an instance and its descendants (instances.json)."""
    ids = set(subtree_ids(instances, root))
    rows = [i for i in instances if int(i["id"]) in ids]
    if not rows:
        raise ValueError(f"instance {root} is not in instances.json")
    low = np.min([r["bounds"]["min"] for r in rows], axis=0).astype(np.float64)
    high = np.max([r["bounds"]["max"] for r in rows], axis=0).astype(np.float64)
    return low, high


def load_labels(tileset: Path) -> np.ndarray:
    """Each leaf gaussian's instance id, in `splat_render.load_tileset`'s order (leaves by
    sorted uri), from instances.json's per-tile runs (keyed by position checksum)."""
    import split_objects as so

    document = json.loads(tileset.read_text(encoding="utf-8"))
    uri = document["root"].get("extras", {}).get("instances", {}).get("uri")
    if not uri:
        raise ValueError(f"{tileset} declares no instances")
    doc = json.loads((tileset.parent / uri).read_text(encoding="utf-8"))
    leaves: list[str] = []
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("children"):
            stack.extend(tile["children"])
        else:
            leaves.append(tile["content"]["uri"])
    out = []
    for leaf in sorted(leaves):
        spz = so.read_tile(tileset.parent / leaf)
        runs = doc["tiles"].get(so.checksum_positions(spz.positions()))
        if runs is None:
            raise ValueError(f"tile {leaf} is not in instances.json")
        labels = so.decode_runs(runs)
        if labels.size != len(spz):
            raise ValueError(f"tile {leaf}: {labels.size} ids for {len(spz)} gaussians")
        out.append(labels)
    return np.concatenate(out)


def object_gaussians(positions: np.ndarray, labels: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    """An object's gaussians, its fragments under other ids taken with it
    (`split_objects.absorb`'s rule: another id with `ABSORB_INSIDE` of its gaussians in the
    object's box and at most `ABSORB_SHARE` of its size)."""
    import split_objects as so

    mine = np.isin(labels, ids)
    if not mine.any():
        return mine
    low, high = np.percentile(positions[mine], [3, 97], axis=0)
    pad = so.ABSORB_PAD * float(np.max(high - low))
    inside = np.all((positions >= low - pad) & (positions <= high + pad), axis=1)
    counts = np.bincount(labels)
    within = np.bincount(labels[inside], minlength=counts.size)
    share = within / np.maximum(counts, 1)
    joined = np.flatnonzero((share >= so.ABSORB_INSIDE) & (counts <= so.ABSORB_SHARE * mine.sum()))
    joined_ids = [int(k) for k in joined if k > 0]
    return np.isin(labels, [*ids, *joined_ids])


def frame_check(
    splats: Splats, views: Sequence[RealView], renderer: Renderer, count: int = 4
) -> dict[str, Any]:
    """Whether the cameras are in the tiles' frame: the scan rendered from a few of them
    against their photos (PSNR on the covered pixels)."""
    scores: list[dict[str, Any]] = []
    for view in list(views)[:: max(1, len(views) // count)][:count]:
        photo = view.photo(width=320)
        if photo is None:
            continue
        camera = scaled(view.camera, photo.shape[1], photo.shape[0])
        frame = renderer(splats, camera)
        covered = frame.alpha >= 0.8
        if covered.mean() < 0.05:
            scores.append({"view": view.name, "covered": round(float(covered.mean()), 3)})
            continue
        colour = tf.to_u8(frame.rgb / np.maximum(frame.alpha, 1e-6)[..., None])
        scores.append(
            {
                "view": view.name,
                "covered": round(float(covered.mean()), 3),
                "psnr": round(tf.psnr(colour, photo, covered), 2),
            }
        )
    psnrs = [s["psnr"] for s in scores if "psnr" in s]
    return {"views": scores, "meanPsnr": round(float(np.mean(psnrs)), 2) if psnrs else None}


def elevation(view: RealView, centre: np.ndarray) -> float:
    d = view.camera.centre - centre
    return math.degrees(math.asin(np.clip(d[2] / max(np.linalg.norm(d), 1e-9), -1, 1)))


def sees(view: RealView, point: np.ndarray) -> bool:
    uv, z = view.camera.project(np.asarray(point)[None])
    w, h = view.camera.width, view.camera.height
    return bool(z[0] > 0 and 0 <= uv[0, 0] < w and 0 <= uv[0, 1] < h)


def split_held_out(
    views: Sequence[RealView], centre: np.ndarray, share: float
) -> tuple[list[RealView], list[RealView]]:
    """(kept, held out): the `share` of the cameras that see the region's centre from
    highest above it are held out."""
    seeing = [v for v in views if sees(v, centre)]
    ranked = sorted(seeing, key=lambda v: -elevation(v, centre))
    n = max(1, round(share * len(ranked))) if share > 0 and ranked else 0
    held = {v.name for v in ranked[:n]}
    return [v for v in views if v.name not in held], [v for v in views if v.name in held]


# --- the bake-off -------------------------------------------------------------------------------------


@dataclass
class Options:
    paths: int = 2
    seeds: int = 1
    rounds: int = 2
    round2_paths: int = 1
    frames: int = 49
    distill: int = 1500
    distill_width: int = 640
    real_views: int = 12
    #: Refuse to start more clips than this in total (the budget guard).
    max_clips: int = 64
    #: Path planning: azimuths tried per elevation, and the probe render's size.
    azimuths: int = 12
    probe: tuple[int, int] = (192, 108)


def seeds_for(filler: ClipFiller, options: Options) -> list[int]:
    """Video models: `options.seeds` seeds; a deterministic per-view filler: one."""
    n = options.seeds if isinstance(filler, RemoteClipFiller | TeleaClipFiller) else 1
    return [7 + 1000 * s for s in range(max(1, n))]


def planned_clips(fillers: Sequence[ClipFiller], options: Options) -> dict[str, int]:
    out = {}
    for f in fillers:
        n = options.paths * len(seeds_for(f, options))
        if options.rounds >= 2:
            n += options.round2_paths
        out[f.name] = n
    return out


def run_bakeoff(
    scene: Scene,
    fillers: Sequence[ClipFiller],
    depth_model: DepthModel,
    renderer: Renderer,
    measured_tileset: Path,
    out: Path,
    options: Options,
    distil_runner: Callable[[dict], dict] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Every filler on the same round-1 paths (and its own round-2 paths), each fill lifted,
    carved, distilled and packaged under `out/<filler>/inferred`; renders under
    `out/renders`; the report returned (and written to `out/report.json`)."""
    planned = planned_clips(fillers, options)
    if sum(planned.values()) > options.max_clips:
        raise SystemExit(
            f"{sum(planned.values())} clips planned, more than --max-clips {options.max_clips}"
        )
    report: dict[str, Any] = {
        "scene": scene.name,
        "roi": [scene.roi[0].round(3).tolist(), scene.roi[1].round(3).tolist()],
        "measured": len(scene.measured),
        "hidden": int(scene.hidden.sum()) if scene.hidden is not None else 0,
        "dropped": int(scene.dropped.sum()) if scene.dropped is not None else 0,
        "views": len(scene.views),
        "heldOut": [v.name for v in scene.held_out],
        "prompt": scene.prompt,
        "plannedClips": planned,
        "candidates": {},
    }
    region = grown(*scene.roi, ROI_PAD)
    budget = max(MIN_BUDGET, int(BUDGET_SHARE * len(scene.measured)))
    carvers = carving_cameras(scene)
    # Carving is against everything measured: what the (kept) real cameras saw, including
    # what the generator is not shown -- a hidden object, a dropped top -- since those
    # cameras did see it. It only ever removes what lies in front of it.
    present = scene.measured
    log(f"{scene.name}: planning {options.paths} paths")
    paths1, scores1 = plan_paths(
        scene,
        options.paths,
        options.frames,
        renderer,
        azimuths=options.azimuths,
        probe=options.probe,
        label="r1",
    )
    report["paths"] = [p.to_json() for p in paths1]
    report["pathScores"] = scores1
    if not paths1:
        report["skipped"] = "no viewpoint sees anything unknown in the region"
        write_json(out / "report.json", report)
        return report
    caches: dict[str, dict[int, np.ndarray]] = {p.label: {} for p in paths1}
    clips: dict[tuple[int, int], list[Clip]] = {}
    for f in fillers:
        if f.size not in clips:
            clips[f.size] = [
                render_clip(scene, p, f.size, renderer, None, caches[p.label]) for p in paths1
            ]
    pending: dict[str, tuple[list[ClipRequest], Callable[[], list[ClipResult]]]] = {}
    for f in fillers:
        requests = [
            ClipRequest(clip, scene.prompt, seed, f"{clip.path.label}s{seed}")
            for clip in clips[f.size]
            for seed in seeds_for(f, options)
        ]
        log(f"{f.name}: round 1, {len(requests)} clips")
        pending[f.name] = (requests, f.start(requests))
    state: dict[str, dict[str, Any]] = {}
    for f in fillers:
        requests, collect = pending[f.name]
        try:
            results = collect()
        except Exception as error:  # noqa: BLE001 - one model failing leaves the others
            log(f"{f.name}: round 1 failed: {error!r}")
            report["candidates"][f.name] = {"failed": repr(error)[:2000]}
            continue
        infos = [res.info for res in results]
        requests, results = _succeeded(requests, results)
        if not results:
            log(f"{f.name}: every round-1 clip failed")
            report["candidates"][f.name] = {"failed": "every clip failed", "calls": infos}
            continue
        fills = [
            prepare_fill(r.clip, res, depth_model) for r, res in zip(requests, results, strict=True)
        ]
        lifted = _lift_round(fills, requests, region)
        state[f.name] = {"fills": fills, "infos": infos, "lifted": [lifted]}
        log(f"{f.name}: round 1 lifted {len(lifted.splats)}")
    if options.rounds >= 2:
        pending = {}
        for f in fillers:
            if f.name not in state:
                continue
            first = _carved(merge(state[f.name]["lifted"]), present, carvers, renderer, scene)
            paths2, _ = plan_paths(
                scene,
                options.round2_paths,
                options.frames,
                renderer,
                extra=first.splats,
                avoid=[p.end_direction for p in paths1],
                used_starts=[p.start.name for p in paths1 if p.start],
                azimuths=options.azimuths,
                probe=options.probe,
                label="r2",
            )
            seed = seeds_for(f, options)[0]
            requests = [
                ClipRequest(
                    render_clip(scene, p, f.size, renderer, first.splats),
                    scene.prompt,
                    seed,
                    f"{p.label}s{seed}",
                )
                for p in paths2
            ]
            state[f.name]["paths2"] = [p.to_json() for p in paths2]
            log(f"{f.name}: round 2, {len(requests)} clips")
            pending[f.name] = (requests, f.start(requests))
        for name, (requests, collect) in pending.items():
            try:
                results = collect()
            except Exception as error:  # noqa: BLE001
                log(f"{name}: round 2 failed: {error!r}")
                state[name]["round2Failed"] = repr(error)[:2000]
                continue
            state[name]["infos"] += [res.info for res in results]
            requests, results = _succeeded(requests, results)
            fills = [
                prepare_fill(r.clip, res, depth_model)
                for r, res in zip(requests, results, strict=True)
            ]
            state[name]["fills"] += fills
            state[name]["lifted"].append(_lift_round(fills, requests, region))
    finish = Finish(
        scene,
        present,
        carvers,
        renderer,
        region,
        budget,
        measured_tileset,
        out,
        options,
        distil_runner,
        depth_model,
        log,
    )
    for f in fillers:
        if f.name in state:
            report["candidates"][f.name] = finish(f, state[f.name])
    _renders(scene, fillers, out, renderer, report)
    write_json(out / "report.json", report)
    return report


def _succeeded(
    requests: Sequence[ClipRequest], results: Sequence[ClipResult]
) -> tuple[list[ClipRequest], list[ClipResult]]:
    """The requests whose call came back with frames (a failed call is logged and skipped)."""
    kept = [(r, res) for r, res in zip(requests, results, strict=True) if res.frames is not None]
    for r, res in zip(requests, results, strict=True):
        if res.frames is None:
            _log(f"clip {r.label} failed: {res.info.get('error')}")
    return [r for r, _ in kept], [res for _, res in kept]


def _lift_round(
    fills: Sequence[ClipFill],
    requests: Sequence[ClipRequest],
    region: tuple[np.ndarray, np.ndarray],
) -> Lifted:
    """Each clip's first seed lifted, the other seeds of the same clip as its agreement."""
    by_clip: dict[int, list[ClipFill]] = {}
    for fill, r in zip(fills, requests, strict=True):
        by_clip.setdefault(id(r.clip), []).append(fill)
    return merge([lift_fill(g[0], region, g[1] if len(g) > 1 else None) for g in by_clip.values()])


def _carved(
    lifted: Lifted, measured: Splats, cameras: Sequence[Camera], renderer: Renderer, scene: Scene
) -> Lifted:
    if not len(lifted.splats) or not cameras:
        return lifted
    keep = carve(lifted.splats.positions, measured, cameras, renderer, 2 * scene.radius)
    rows = np.flatnonzero(keep)
    return Lifted(lifted.splats.take(rows), lifted.confidence[rows], lifted.cameras, lifted.report)


@dataclass
class Finish:
    """Carve, thin, distil, carve again, package: one candidate's layer and report entry."""

    scene: Scene
    #: What carving renders: everything measured.
    present: Splats
    carvers: list[Camera]
    renderer: Renderer
    region: tuple[np.ndarray, np.ndarray]
    budget: int
    measured_tileset: Path
    out: Path
    options: Options
    distil_runner: Callable[[dict], dict] | None
    depth_model: DepthModel
    log: Callable[[str], None]

    def __call__(self, filler: ClipFiller, s: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        scene = self.scene
        lifted = merge(s["lifted"])
        before = len(lifted.splats)
        lifted = _carved(lifted, self.present, self.carvers, self.renderer, scene)
        carved = before - len(lifted.splats)
        lifted = thin(lifted, self.budget)
        entry: dict[str, Any] = {
            "filler": filler.name,
            "calls": s["infos"],
            "gate": [
                {
                    "label": f.clip.path.label,
                    "kept": int(f.keep.sum()),
                    "of": len(f.keep),
                    "psnr": f.gate,
                }
                for f in s["fills"]
            ],
            "depthFits": [
                {"label": f.clip.path.label, **{str(k): v for k, v in f.depth_info.items()}}
                for f in s["fills"]
            ],
            "lifted": before,
            "carved": carved,
            "kept": len(lifted.splats),
        }
        if "paths2" in s:
            entry["paths2"] = s["paths2"]
        if "round2Failed" in s:
            entry["round2Failed"] = s["round2Failed"]
        # What each clip was given and drew, kept or not (a gate that keeps nothing is
        # judged by looking at it).
        name = slug(filler.name)
        renders = self.out / "renders"
        renders.mkdir(parents=True, exist_ok=True)
        for k, f in enumerate(s["fills"][:3]):
            sheet = contact_sheet(f)
            Image.fromarray(sheet).save(
                renders / f"clip-{scene.name}-{name}-{f.clip.path.label}-{k}.png"
            )
        if not len(lifted.splats):
            entry["skipped"] = "nothing lifted"
            return entry
        carved_after = 0
        if self.options.distill > 0 and self.distil_runner is not None:
            import distill_fill as df

            fills = [f for f in s["fills"] if f.depth]
            request = distil_request(
                scene,
                lifted,
                fills,
                self.renderer,
                self.options.distill,
                width=self.options.distill_width,
                real=self.options.real_views,
            )
            response = self.distil_runner(request)
            got = df.unpack_scan(response["inferred"])
            splats = Splats(*(got[k] for k in df.KEYS))
            inside = np.all(
                (splats.positions >= self.region[0]) & (splats.positions <= self.region[1]), axis=1
            )
            keep = inside & carve(
                splats.positions, self.present, self.carvers, self.renderer, 2 * scene.radius
            )
            rows = np.flatnonzero(keep)
            carved_after = int((~keep).sum())
            entry["distill"] = response["report"]
            entry["carvedAfterDistil"] = carved_after
            lifted = Lifted(
                splats.take(rows), lifted.confidence[rows], lifted.cameras, lifted.report
            )
        if not len(lifted.splats):
            entry["skipped"] = "nothing survived carving"
            return entry
        layer = self.out / name / "inferred"
        evidence = tf.package_inferred(
            lifted.splats,
            lifted.confidence,
            lifted.cameras,
            self.measured_tileset,
            layer,
            filler.name,
            rule=RULE.format(support=SUPPORT_MIN, deg=KNOWN_DEG),
            extra=evidence_extra(
                filler,
                rounds=1 + int(bool(s.get("paths2"))),
                seeds=len(seeds_for(filler, self.options)),
                clips=len(s["fills"]),
                carved=carved + carved_after,
                frames=sum(len(f.depth) for f in s["fills"]),
                depth=self.depth_model.name,
            ),
        )
        entry["evidence"] = evidence
        entry["layer"] = str(layer)
        self.log(
            f"{filler.name}: {len(lifted.splats)} gaussians, "
            f"mean confidence {evidence['meanConfidence']}"
        )
        return entry


def slug(name: str) -> str:
    out = "".join(c if c.isalnum() else "-" for c in name.lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def _layer(out: Path, filler: ClipFiller) -> Splats | None:
    from splat_render import load_tileset

    path = out / slug(filler.name) / "inferred" / "tileset.json"
    return load_tileset(path) if path.exists() else None


def _view(scene: Scene, elev: float, az: float, w: int, h: int) -> Camera:
    fit = 1.2 * _fit_distance(max(scene.radius, 1e-3), FOV_DEG, w / h)
    e, a = math.radians(elev), math.radians(az)
    d = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return Camera.look_at(
        scene.centre + fit * d, scene.centre, fov_deg=FOV_DEG, width=w, height=h, up=tuple(scene.up)
    )


#: The angles (elevation, azimuth) of the before/after renders.
RENDER_ANGLES = ((40.0, 30.0), (40.0, 150.0), (40.0, 270.0), (80.0, 0.0))


def _renders(
    scene: Scene, fillers: Sequence[ClipFiller], out: Path, renderer: Renderer, report: dict
) -> None:
    """Before and after from several angles; with held-out cameras, the comparison with
    their photos."""
    from PIL import Image

    folder = out / "renders"
    folder.mkdir(parents=True, exist_ok=True)
    layers = {f.name: _layer(out, f) for f in fillers}
    present = scene.measured.take(scene.present_index())
    w, h = 480, 270
    sets = [("", present)]
    if scene.hidden is not None and scene.hidden.any():
        sets.append(("hidden", scene.measured.take(scene.shown_index())))
    for tag, base in sets:
        rows = []
        for elev, az in RENDER_ANGLES:
            cam = _view(scene, elev, az, w, h)
            title = f"measured ({tag})" if tag else "measured"
            row = [label_image(tf.to_u8(renderer(base, cam).rgb), title)]
            for name, layer in layers.items():
                both = Splats.concat([base, layer]) if layer is not None and len(layer) else base
                row.append(label_image(tf.to_u8(renderer(both, cam).rgb), f"+ {name}"))
            rows.append(row)
        suffix = f"-{tag}" if tag else ""
        Image.fromarray(grid(rows)).save(folder / f"before-after-{scene.name}{suffix}.png")
    if scene.held_out:
        _held_out(scene, layers, present, renderer, folder, report)


def _held_out(
    scene: Scene,
    layers: dict[str, Splats | None],
    present: Splats,
    renderer: Renderer,
    folder: Path,
    report: dict,
) -> None:
    """Per held-out camera: its photo | the scan as the kept cameras know it | with each
    layer | the scan trained with them all. PSNR on the region's pixels, for reference."""
    from PIL import Image

    held = sorted(scene.held_out, key=lambda v: -elevation(v, scene.centre))[:4]
    rows, numbers = [], []
    for view in held:
        photo = view.photo(width=480)
        if photo is None:
            continue
        cam = scaled(view.camera, photo.shape[1], photo.shape[0])
        known = scene.known.weights(scene.measured.positions, cam.centre)[scene.present_index()]
        before = tf.to_u8(renderer(present, cam, opacity_scale=known).rgb)
        region = ray_box(cam, *scene.roi)
        row = [
            label_image(photo, f"real, held out: {view.name}"),
            label_image(before, "before: what the kept views know"),
        ]
        score: dict[str, Any] = {
            "view": view.name,
            "before": round(tf.psnr(before, photo, region), 2),
        }
        for name, layer in layers.items():
            if layer is None or not len(layer):
                row.append(label_image(np.zeros_like(photo), f"{name}: nothing"))
                continue
            weights = np.concatenate([known, np.ones(len(layer))])
            after = tf.to_u8(
                renderer(Splats.concat([present, layer]), cam, opacity_scale=weights).rgb
            )
            row.append(label_image(after, f"after: {name}"))
            score[name] = round(tf.psnr(after, photo, region), 2)
        full = tf.to_u8(renderer(scene.measured, cam).rgb)
        row.append(label_image(full, "scan trained with them"))
        rows.append(row)
        numbers.append(score)
    if rows:
        Image.fromarray(grid(rows)).save(folder / f"held-out-{scene.name}.png")
    report["heldOutPsnrInRegion"] = numbers


# --- the command line -----------------------------------------------------------------------------------

#: Set by a runner inside Modal (`infra/modal/fill.py`): how to spawn a GPU class's method and
#: wait for it. Unset, the video models cannot run (the stand-in can).
BACKEND: tuple[Submit, Wait] | None = None


def pick_roi(
    instances: Sequence[dict],
    splats: Splats,
    known: KnownModel,
    keyword: str,
    *,
    renderer: Renderer | None = None,
    min_splats: int = 5000,
    max_extent_m: float = 12.0,
    sample: int = 20000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The region to fill among the instances tagged `keyword` (a roof): the one the scan
    most missed from above, weighted by its size. Missed is unseen (its gaussians' `known`
    from a point over its centre) or, with a `renderer`, empty (the share of its footprint a
    view from straight above finds nothing in within a metre: the camp's view cones call
    every roof seen, but a roof seen only from the ground has holes on top). Only a pick:
    through such a hole the full scene shows what is below (run 37395213267, the camp: no
    viewpoint then has anything unknown or empty to fill there). Segmentation only says
    where to look; the fill does not use its shape."""
    rng = np.random.default_rng(0)
    scored = []
    for inst in instances:
        tags = " ".join(str(t["label"]).lower() for t in inst.get("tags", [])[:5])
        if keyword not in tags or int(inst.get("splats", 0)) < min_splats:
            continue
        low = np.asarray(inst["bounds"]["min"], np.float64)
        high = np.asarray(inst["bounds"]["max"], np.float64)
        if float(np.max((high - low)[:2])) > max_extent_m:
            continue
        rows = np.flatnonzero(
            np.all((splats.positions >= low) & (splats.positions <= high), axis=1)
        )
        if rows.size < 100:
            continue
        if rows.size > sample:
            rows = rng.choice(rows, sample, replace=False)
        centre = (low + high) / 2
        eye = centre + np.array([0.0, 0.0, 2.0 * max(float(np.max(high - low)), 1.0)])
        if isinstance(known, ConeKnown):
            weights = known.weights(splats.positions[rows], eye)
        else:
            weights = known.weights(splats.positions, eye)[rows]
        unseen = 1.0 - float(weights.mean())
        empty = _empty_from_above(splats, low, high, renderer) if renderer is not None else 0.0
        missed = 1.0 - (1.0 - unseen) * (1.0 - empty)
        scored.append(
            (missed * math.log1p(int(inst["splats"])), int(inst["id"]), unseen, empty, low, high)
        )
    if not scored:
        raise SystemExit(f"no instance tagged {keyword!r} to fill")
    scored.sort(key=lambda s: -s[0])
    _, iid, unseen, empty, low, high = scored[0]
    return (
        low,
        high,
        {
            "instance": iid,
            "unseenFromAbove": round(unseen, 3),
            "emptyFromAbove": round(empty, 3),
            "candidates": [
                {
                    "instance": s[1],
                    "score": round(s[0], 3),
                    "unseen": round(s[2], 3),
                    "empty": round(s[3], 3),
                }
                for s in scored[:8]
            ],
        },
    )


def _empty_from_above(
    splats: Splats, low: np.ndarray, high: np.ndarray, renderer: Renderer, size: int = 96
) -> float:
    """The share of a box's footprint that a view from straight above finds nothing in (the
    gaussians within a metre of the box drawn)."""
    pad = 1.0
    near = np.all((splats.positions >= low - pad) & (splats.positions <= high + pad), axis=1)
    if not near.any():
        return 1.0
    centre = (low + high) / 2
    extent = float(np.max((high - low)[:2])) + 0.5
    fov = 40.0
    height = (high[2] - centre[2]) + extent / (2 * math.tan(math.radians(fov / 2)))
    camera = Camera.look_at(
        centre + np.array([0.0, 0.0, height]),
        centre,
        fov_deg=fov,
        width=size,
        height=size,
        up=(0.0, 1.0, 0.0),
    )
    footprint = ray_box(camera, low, high)
    if not footprint.any():
        return 0.0
    frame = renderer(splats.take(np.flatnonzero(near)), camera)
    return float((~tf._covered(frame.alpha))[footprint].mean())


def shape_roi(
    low: np.ndarray, high: np.ndarray, grow: float = 0.0, top: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """An object's box made the region around and under it: grown by `grow` of its size on
    each side, its bottom lowered by 15 % of its height (the surface it stands on), and
    with `top` its top cut to that share of its height."""
    low, high = np.array(low, np.float64), np.array(high, np.float64)
    size = high - low
    if top is not None:
        high[2] = low[2] + top * size[2]
    if grow > 0:
        low[:2] -= grow * size[:2]
        high[:2] += grow * size[:2]
        low[2] -= 0.15 * size[2]
    return low, high


def make_fillers(
    keys: Sequence[str], steps: int | None = None, negative: str = ""
) -> list[ClipFiller]:
    out: list[ClipFiller] = []
    for key in keys:
        if key in VIDEO_CLASSES:
            if BACKEND is None:
                raise SystemExit(f"{key}: no GPU backend (run through infra/modal/fill.py)")
            out.append(RemoteClipFiller(key, *BACKEND, steps=steps, negative=negative))
        elif key == "lama":
            from world_model_client import GenerativeFiller

            out.append(PerViewClipFiller(GenerativeFiller(model="lama", reads_void=False)))
        elif key == "telea":
            out.append(TeleaClipFiller())
        else:
            known = ", ".join([*VIDEO_CLASSES, "lama", "telea"])
            raise SystemExit(f"generator {key!r}: one of {known}")
    return out


def build_scene(
    args: Any, renderer: Renderer, log: Callable[[str], None] = print
) -> tuple[Scene, dict[str, Any]]:
    """The scene a run fills, from the command line's arguments."""
    from splat_render import load_tileset

    tileset = Path(args.tileset)
    splats = load_tileset(tileset)
    info: dict[str, Any] = {"gaussians": len(splats)}
    instances = tf.read_instances(tileset)
    grid = None
    if args.roi_pick:
        if args.poses:
            raise SystemExit("--roi-pick is for a scan without cameras (its view cones)")
        grid = vc.cone_grid_from_tileset(tileset)
        low, high, picked = pick_roi(
            instances, splats, ConeKnown(grid), args.roi_pick, renderer=renderer
        )
        info["roiPicked"] = picked
        log(f"region picked: {json.dumps(picked)}")
    elif args.roi_instance is not None:
        low, high = instance_box(instances, args.roi_instance)
    elif args.roi:
        low, high = np.asarray(args.roi[:3], float), np.asarray(args.roi[3:], float)
    else:
        low, high = np.percentile(splats.positions, [5, 95], axis=0)
    low, high = shape_roi(low, high, args.roi_grow, args.roi_top)
    info["roi"] = [low.round(3).tolist(), high.round(3).tolist()]
    hidden = None
    if args.hide_instance is not None:
        if args.crop_m:
            raise SystemExit("--hide-instance with --crop-m is not supported")
        labels = load_labels(tileset)
        hidden = object_gaussians(
            splats.positions, labels, subtree_ids(instances, args.hide_instance)
        )
        info["hiddenGaussians"] = int(hidden.sum())
    if args.crop_m:
        c = (low + high) / 2
        reach = 0.5 * float(np.max(high - low)) + args.crop_m
        near = np.all(np.abs(splats.positions[:, :2] - c[:2]) <= reach, axis=1)
        splats = splats.take(np.flatnonzero(near))
        info["cropped"] = len(splats)
    views: list[RealView] = []
    held: list[RealView] = []
    observers = np.zeros((0, 3))
    dropped = None
    if args.poses:
        placement = json.loads(Path(args.placement).read_text()) if args.placement else None
        views = real_views(Path(args.poses), placement, Path(args.frames) if args.frames else None)
        info["realViews"] = len(views)
        check = frame_check(splats, views, renderer)
        info["frameCheck"] = check
        log(f"frame check: {json.dumps(check)}")
        if check["meanPsnr"] is None or check["meanPsnr"] < args.min_frame_psnr:
            raise SystemExit(
                f"the cameras do not match the tiles (frame check {check['meanPsnr']} dB)"
            )
        centre = (low + high) / 2
        if args.holdout_above > 0:
            views, held = split_held_out(views, centre, args.holdout_above)
            info["heldOutElevations"] = [round(elevation(v, centre), 1) for v in held]
        known = seen_directions(splats, views, renderer)
        if held:
            everyone = seen_directions(splats, held, renderer)
            dropped = (known.counts == 0) & (everyone.counts > 0)
            info["dropped"] = int(dropped.sum())
        info["support"] = {
            "median": float(np.median(known.counts)),
            "belowMin": int((known.counts < SUPPORT_MIN).sum()),
        }
        known_model: KnownModel = known
        if args.unknown_roi:
            never = np.all((splats.positions >= low) & (splats.positions <= high), axis=1)
            known_model = ForcedUnknown(known, never)
            info["forcedUnknown"] = int(never.sum())
    else:
        grid = grid if grid is not None else vc.cone_grid_from_tileset(tileset)
        known_model = ConeKnown(grid)
        observers = grid.observers
    scene = Scene(
        name=args.scan,
        measured=splats,
        known=known_model,
        roi=(low, high),
        views=views,
        held_out=held,
        observers=observers,
        hidden=hidden,
        dropped=dropped,
        prompt=args.prompt or PROMPT,
    )
    return scene, info


@dataclass
class FlatDepth:
    """The stand-in for a run without a depth model: the same disparity everywhere, so the
    fit puts each lifted pixel at the known pixels' depth (a plane facing the camera)."""

    name: str = "flat"

    def disparity(self, frames: np.ndarray, cameras: Sequence[Camera] = ()) -> np.ndarray:
        n, h, w = frames.shape[:3]
        return np.ones((n, h, w)) + 1e-6 * np.arange(w)[None, None, :]


def parser() -> Any:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="every generator on one scan's region: layers and renders")
    run.add_argument("tileset", type=Path)
    run.add_argument("out", type=Path)
    run.add_argument("--scan", default="scan")
    run.add_argument("--generators", default="telea", help="vace,wan22,cosmos,lama,telea")
    run.add_argument("--roi-instance", type=int)
    run.add_argument("--roi", type=float, nargs=6, help="low xyz, high xyz (local ENU)")
    run.add_argument("--hide-instance", type=int, help="hide this object from the generator")
    run.add_argument("--crop-m", type=float, help="keep the scan within this of the region")
    run.add_argument("--poses", type=Path, help="COLMAP sparse model (cameras.bin, images.bin)")
    run.add_argument("--placement", type=Path, help="placement.json, COLMAP to the tiles' frame")
    run.add_argument("--frames", type=Path, help="the photos, by the model's image names")
    run.add_argument(
        "--holdout-above",
        type=float,
        default=0.0,
        help="hold out this share of the cameras that see the region from highest",
    )
    run.add_argument("--min-frame-psnr", type=float, default=13.0)
    run.add_argument(
        "--unknown-roi",
        action="store_true",
        help="the held-out check: the region's look is never known (its shape is), so the "
        "generator must make it; compared with the held-out photos",
    )
    run.add_argument("--paths", type=int, default=2)
    run.add_argument("--seeds", type=int, default=1)
    run.add_argument("--rounds", type=int, default=2)
    run.add_argument("--round2-paths", type=int, default=1)
    run.add_argument("--frames-per-clip", type=int, default=49)
    run.add_argument("--steps", type=int)
    run.add_argument("--distill", type=int, default=1500)
    run.add_argument("--distill-on", choices=("local", "modal"), default="local")
    run.add_argument("--max-clips", type=int, default=64)
    run.add_argument("--prompt", default="")
    run.add_argument("--negative", default="", help="also never paint these")
    run.add_argument(
        "--roi-grow",
        type=float,
        default=0.0,
        help="grow the object's box by this share of its size each side",
    )
    run.add_argument("--roi-top", type=float, help="cut the box's top to this share of its height")
    run.add_argument("--roi-pick", default="", help="pick the region among instances so tagged")
    run.add_argument("--renderer", choices=("cpu", "gsplat"), default="cpu")
    run.add_argument("--depth", choices=("depth-anything", "flat"), default="depth-anything")
    return p


def _log(line: str) -> None:
    """Progress to stderr: stdout carries the report alone."""
    print(line, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    import distill_fill

    args = parser().parse_args(argv)
    renderer = tf.make_renderer(args.renderer)
    scene, info = build_scene(args, renderer, _log)
    keys = [k.strip() for k in args.generators.split(",") if k.strip()]
    fillers = make_fillers(keys, args.steps, args.negative)
    options = Options(
        paths=args.paths,
        seeds=args.seeds,
        rounds=args.rounds,
        round2_paths=args.round2_paths,
        frames=args.frames_per_clip,
        distill=args.distill,
        max_clips=args.max_clips,
    )
    depth: DepthModel = DepthAnything() if args.depth == "depth-anything" else FlatDepth()
    runner = None
    if args.distill > 0:
        runner = distill_fill.run if args.distill_on == "local" else tf._distill_runner("modal")
    report = run_bakeoff(
        scene, fillers, depth, renderer, args.tileset, args.out, options, runner, log=_log
    )
    report["setup"] = info
    write_json(args.out / "report.json", report)
    brief = {
        n: e.get("evidence") or {k: e.get(k) for k in ("failed", "skipped")}
        for n, e in report["candidates"].items()
    }
    print(json.dumps({"scene": scene.name, "candidates": brief}, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
