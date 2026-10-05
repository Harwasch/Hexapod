"""Objects from a scale-conditioned feature field on the splat: bake-off candidate B.

docs/SCENE_OBJECTS.md §3 is today's method (masks voted onto cells, merged by co-occurrence);
this is the alternative the 2026-10-05 research brief calls "per-scene feature fields"
(§1.4 there): the splat's geometry is frozen, every gaussian gets a small feature vector, and
the features are trained so that, rendered into a view and read at a physical *scale*, two
pixels are alike exactly when some 2D mask of about that size holds both. Whole objects and
their parts then come out of one field by reading it at large and small scales. It writes
the same `instances.json` the viewer reads (§4 there), so the owner can switch to it.

**References, reimplemented here on gsplat (none of their code is used):**

* GARField (Kim et al., CVPR 2024; nerfstudio code, MIT): an affinity field conditioned on a
  physical scale, trained contrastively from SAM masks whose scale is their 3D extent, and
  turned into a tree by sweeping the scale. The scale definition and the tree are theirs.
* SAGA (Cen et al., AAAI 2025; Apache-2.0 repository built on Inria's non-commercial 3DGS):
  a feature per gaussian, rasterized, read through a *scale gate* -- the feature times a
  sigmoid of a small MLP of the scale -- and a feature-norm consistency term. The gate and the
  consistency term are theirs.
* LangSplatV2 (NeurIPS 2025): a language feature as sparse codes. Not built (budget); cover
  classes are SigLIP 2 labels voted per ground cluster instead (`cover_classes`).
* gsplat 1.5.3 (Apache-2.0) rasterizes the N-D features; SAM 2.1 hiera-large (Apache-2.0)
  gives the masks; SigLIP 2 base (Apache-2.0) describes the instances, as today.

**The field.** Per gaussian a feature `f_i` (`FEATURE_DIM`), unit-normalised before it is
rasterized (gsplat, colours of any width), so a pixel's feature `F(p)` is the alpha blend of
its gaussians'. A gate `g(s) = sigmoid(MLP(log s))` (`GATE_HIDDEN`) gives the feature at scale
`s`: `normalize(g(s) * F(p))`. Training images are the capture's own photos where we have
them (`colmap_views`: COLMAP poses, undistorted, moved into the tileset's frame by the
`place` stage's placement), else views rendered from the splat (`rendered_views`). SAM 2.1's
automatic masks of each image (`Sam2Masks`, the whole-view ceiling raised to `MASK_MAX_AREA`
so the ground's big masks count) each get a scale: the robust 3D diameter of the splat's
expected depth under it (`mask_scales`).

**The loss** (`pair_loss`). Per step, one view, `PIXELS` pixels with the splat behind them,
and per anchor pixel `i` a scale `s_i` drawn log-uniformly over the masks' scales. At that
scale `i`'s group is the masks holding it whose scale is at most `s_i`: a pixel `j` in one of
them is a positive pair (cosine pulled to 1), any other a negative (cosine pushed below
`NEGATIVE_MARGIN`) -- what the view's masks say exhaustively, as GARField's containment rule
reads them: grouped at some scale at most `s` means grouped at `s`. An anchor in no mask that
small says nothing and is skipped. Views disagree (SAM's levels are relative to a click), and
the field averages them in 3D -- which is the point. Plus the consistency term
`CONSISTENCY_WEIGHT * (A - |F|)`: the blend of unit features is as long as the coverage `A`
only where the gaussians along a ray agree.

**The tree** (`build_tree`, numpy: it reruns on a CPU from the saved field). Ground first
(`ground_layer`: the shared ground pass when it is importable, else `scene_plants`' slope
filter), then:

1. A spatial graph: each gaussian's `NEIGHBOURS` nearest, not past `EDGE_QUANTILE` of the
   k-th distances; floaters (`floaters`) stay out of it.
2. *Objects*: the connected parts of the graph above the ground, each split by the field at
   its own size (`partition` at `TOP_SCALE` x its robust diameter): edges whose gated cosine
   is at least `EDGE_COSINE` make fragments, adjacent fragments whose mean features agree to
   `MERGE_COSINE` join, crumbs join a neighbour.
3. *Parts*: each node split again at `CHILD_SCALE` of its own diameter (and at that squared
   when the first scale does not split it), up to `MAX_DEPTH` levels.
4. *Contact*: a ground-layer gaussian beside an object whose feature is the object's rather
   than the ground beside it moves to the object (`claim_contact`) -- the spool's bottom
   flange, which today's method fused with the ground.
5. *Ground cover*: the ground split at `COVER_SCALE` of the masks' scales into regions, each
   described by SigLIP 2 like any instance and labelled with its best `COVER_CLASSES` phrase
   (`cover_classes`); regions of one class are grouped under it, and the classes under one
   `Ground` instance (`ground_records`).

Then everything `segment_scene` does after its lift: views planned and rendered with per
pixel the dominant instance (`segment_scene.RenderPool`), `segment_scene.describe` (crops,
SigLIP 2 tags, properties, categories), the tile binding and `instances.json`.

Usage (the GPU half, then the CPU-able half):

    python feature_fields.py train TILES/tileset.json --out field.npz \\
        [--frames FRAMES --poses SPARSE --placement placement.json] [--steps 3000]
    python feature_fields.py finish TILES/tileset.json --field field.npz --out OUT \\
        [--embedder segment_models:SiglipEmbedder --vocabulary data/open_vocabulary.txt] \\
        [--overview OUT/overview.png]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import scene_categories
import scene_plants
import segment_scene
from splat_render import Camera, Splats

# ----------------------------------------------------------------------------- constants

#: The method's name in `instances.json` and the variant it is published as.
METHOD = "feature-field"
VARIANT = "feature-fields"

#: Feature width per gaussian (gsplat rasterizes 32 channels without padding).
FEATURE_DIM = 32
GATE_HIDDEN = 32
#: Training: steps (one view each), anchor pixels per step, learning rates.
STEPS = 3000
PIXELS = 4096
FEATURE_LR = 0.01
GATE_LR = 0.005
NEGATIVE_MARGIN = 0.1
CONSISTENCY_WEIGHT = 0.5
#: Pixels with less coverage than this show no splat and take no part.
MIN_ALPHA = 0.5
#: Long side of the images SAM sees, and of the feature maps trained on.
MASK_SIDE = 1024
TRAIN_SIDE = 512
#: At most this many photos (evenly spaced), to keep SAM's share of the GPU bounded.
MAX_FRAMES = 128
#: SAM 2.1 for the field: the large checkpoint (Apache-2.0), and a whole-view ceiling high
#: enough that the ground's masks count (they say what is *not* ground).
SAM_MODEL = "facebook/sam2.1-hiera-large"
MASK_MAX_AREA = 0.95
#: A mask's scale needs depth under at least this share of it, and is measured on at most
#: this many of its pixels.
MASK_MIN_DEPTH_SHARE = 0.5
MASK_SCALE_POINTS = 4000
#: Scales are drawn (and the tree read) within these quantiles of the masks' scales.
SCALE_QUANTILES = (0.05, 0.98)
#: A gaussian that got a gradient in fewer steps than this takes its neighbours' feature.
MIN_HITS = 2
#: Photos whose render matches them worse than this (median PSNR, dB) are not the splat's
#: frame: the run trains on rendered views instead.
MIN_FRAME_PSNR = 15.0

#: The graph and the tree.
NEIGHBOURS = 10
EDGE_QUANTILE = 0.9
EDGE_COSINE = 0.75
MERGE_COSINE = 0.6
MERGE_ROUNDS = 12
TOP_SCALE = 1.0
CHILD_SCALE = 0.5
MAX_DEPTH = 3
MIN_OBJECT_SPLATS = 40
MIN_CHILD_SHARE = 0.03
#: Floaters: gaussians larger than this many times the median largest axis, or past the
#: 99.5th percentile, stay out of the graph and take a neighbour's instance at the end.
FLOATER_SCALE = 10.0
FLOATER_QUANTILE = 0.995
#: Ground cover: the ground is split at this quantile of the masks' scales; regions under
#: `MIN_COVER_SHARE` of the ground join a neighbour.
COVER_SCALE_QUANTILE = 0.5
MIN_COVER_SHARE = 0.01
#: A ground-layer gaussian joins the object beside it when its gated cosine to the object's
#: mean beats the ground around it by this much.
CONTACT_MARGIN = 0.1

#: Views for describing (as segment_scene's whole-scan views).
DESCRIBE_VIEWS = 32

#: Ground-cover phrases (research brief §1.2.5's vocabulary), each with the scene category
#: (scene_categories) a region of it is listed under.
COVER_CLASSES: dict[str, tuple[str, str]] = {
    "lawn": ("a photo of a mown lawn", "grass"),
    "tall grass": ("a photo of tall grass in a meadow", "grass"),
    "moss": ("a photo of moss on the ground", "grass"),
    "dirt": ("a photo of bare soil and dirt", "ground"),
    "mud": ("a photo of mud", "ground"),
    "sand": ("a photo of sand", "ground"),
    "gravel": ("a photo of gravel", "ground"),
    "crushed stone": ("a photo of crushed stone", "ground"),
    "bedrock": ("a photo of bare bedrock", "rock"),
    "asphalt": ("a photo of asphalt", "paths"),
    "concrete": ("a photo of concrete", "paths"),
    "pavers": ("a photo of paving stones", "paths"),
    "brick": ("a photo of a brick pavement", "paths"),
    "wooden deck": ("a photo of a wooden deck", "paths"),
    "mulch": ("a photo of bark mulch", "ground"),
    "wood chips": ("a photo of wood chips", "ground"),
    "leaf litter": ("a photo of fallen leaves on the ground", "grass"),
    "forest floor": ("a photo of a pine needle forest floor", "ground"),
    "hay": ("a photo of hay and straw", "produce"),
    "crop rows": ("a photo of crop rows in a field", "produce"),
    "ploughed field": ("a photo of a ploughed field", "ground"),
    "snow": ("a photo of snow", "snow"),
    "ice": ("a photo of ice", "snow"),
    "water": ("a photo of water", "water"),
    "puddle": ("a photo of a puddle", "water"),
    "artificial turf": ("a photo of artificial turf", "grass"),
}
GROUND_LABEL = "ground"


# ---------------------------------------------------------------------- training views


@dataclass
class TrainView:
    """An image the field is trained on, with the camera that saw it in the tileset's
    frame: OpenCV axes (x right, y down, z forward), `K` for `image`'s own size."""

    name: str
    image: np.ndarray  # (h, w, 3) uint8
    K: np.ndarray  # (3, 3)
    viewmat: np.ndarray  # (4, 4) world to camera

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def at(self, width: int, height: int) -> np.ndarray:
        """`K` for the same camera drawn at another size."""
        K = self.K.copy()
        K[0] *= width / self.width
        K[1] *= height / self.height
        return K

    def centre(self) -> np.ndarray:
        r, t = self.viewmat[:3, :3], self.viewmat[:3, 3]
        return -r.T @ t


def fit_side(width: int, height: int, side: int) -> tuple[int, int]:
    """`(width, height)` scaled so the long side is `side` (never up)."""
    scale = min(1.0, side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def quat_rotation(q: Sequence[float]) -> np.ndarray:
    """COLMAP's (w, x, y, z) unit quaternion as a rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def placed_viewmat(qvec: Sequence[float], tvec: Sequence[float], placement: dict) -> np.ndarray:
    """A COLMAP image's world-to-camera, moved into the placed (tileset) frame.

    `place` moves every point by `x' = s R x + t` (placement.json); a camera's centre moves
    the same way and its axes by `R`, so the world-to-camera rotation becomes `R_c R^T`. The
    `1/s` between camera coordinates before and after changes no projection
    (`tools/pipeline/lod_maths.placed_camtoworld` is the same move)."""
    rc = quat_rotation(qvec)
    tc = np.asarray(tvec, np.float64)
    rp = np.asarray(placement["rotation"], np.float64)
    tp = np.asarray(placement["translation"], np.float64)
    s = float(placement["scale"])
    centre = s * (rp @ (-rc.T @ tc)) + tp
    rotation = rc @ rp.T
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = -rotation @ centre
    return out


def intrinsics(model: str, params: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """`K` and OpenCV distortion coefficients of a COLMAP camera."""
    p = [float(v) for v in params]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        k = p[3:]
        dist = [k[0] if k else 0.0, k[1] if len(k) > 1 else 0.0, 0.0, 0.0]
    elif model in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = p[:4]
        dist = (p[4:8] + [0.0] * 4)[:4]
    else:
        raise ValueError(f"camera model {model} is not one this reads")
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, np.asarray(dist, np.float64)


def _pipeline_sfm() -> Any:
    """`tools/pipeline/sfm.py` (numpy only), imported by path as estimate_scale.py does."""
    pipeline = Path(__file__).resolve().parent.parent / "pipeline"
    if str(pipeline) not in sys.path:
        sys.path.append(str(pipeline))
    import sfm

    return sfm


def colmap_views(
    frames: Path,
    poses: Path,
    placement: dict,
    *,
    side: int = MASK_SIDE,
    limit: int = MAX_FRAMES,
) -> list[TrainView]:
    """The capture's photos with their COLMAP poses, undistorted, at most `side` a side and
    at most `limit` of them (evenly spaced, in name order), in the tileset's frame."""
    import cv2

    sfm = _pipeline_sfm()
    model = sfm.read_model(poses)
    cameras = {c.id: c for c in model.cameras}
    images = [i for i in model.images if (frames / i.name).is_file()]
    if limit and len(images) > limit:
        images = [images[k] for k in np.linspace(0, len(images) - 1, limit).round().astype(int)]
    out: list[TrainView] = []
    for image in images:
        camera = cameras[image.camera_id]
        K, dist = intrinsics(camera.model, camera.params)
        bgr = cv2.imread(str(frames / image.name), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        if (w, h) != (camera.width, camera.height):
            # Frames stored at another size than COLMAP posed: scale the intrinsics.
            K[0] *= w / camera.width
            K[1] *= h / camera.height
        if np.any(dist != 0):
            bgr = cv2.undistort(bgr, K, dist)
        size = fit_side(w, h, side)
        if size != (w, h):
            bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        K[0] *= size[0] / w
        K[1] *= size[1] / h
        viewmat = placed_viewmat(image.qvec, image.tvec, placement)
        out.append(TrainView(image.name, np.ascontiguousarray(bgr[..., ::-1]), K, viewmat))
    return out


def camera_view(camera: Camera, image: np.ndarray, name: str = "") -> TrainView:
    """A `splat_render.Camera` (principal point at the centre) as a `TrainView`."""
    K = np.array(
        [
            [camera.focal, 0.0, camera.width / 2],
            [0.0, camera.focal, camera.height / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    viewmat = np.eye(4)
    viewmat[:3, :3] = camera.rotation
    viewmat[:3, 3] = -camera.rotation @ camera.centre
    return TrainView(name, image, K, viewmat)


# ------------------------------------------------------------------ rasterizing on torch


@dataclass
class Scene:
    """The frozen gaussians as tensors on a device."""

    means: Any
    quats: Any
    scales: Any
    opacities: Any
    colours: Any

    @staticmethod
    def of(splats: Splats, device: str) -> Scene:
        import torch

        def t(a: np.ndarray) -> Any:
            return torch.as_tensor(np.asarray(a, np.float32), device=device)

        q = splats.rotations / np.maximum(
            np.linalg.norm(splats.rotations, axis=1, keepdims=True), 1e-12
        )
        return Scene(
            t(splats.positions), t(q), t(splats.scales), t(splats.opacities), t(splats.colours)
        )


#: `(scene, colours (n, c), viewmat (4, 4), K (3, 3), width, height, depth) ->
#: (out (h, w, c), alpha (h, w), depth (h, w) or None)`, differentiable in `colours`.
Rasterize = Callable[..., tuple[Any, Any, Any]]


def gsplat_rasterize(scene, colours, viewmat, K, width, height, depth=False):
    """gsplat's rasterization of any number of channels (and its expected depth)."""
    from gsplat import rasterization

    out, alpha, _ = rasterization(
        scene.means,
        scene.quats,
        scene.scales,
        scene.opacities,
        colours,
        viewmat[None],
        K[None],
        int(width),
        int(height),
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB+ED" if depth else "RGB",
    )
    out = out[0]
    if depth:
        return out[..., :-1], alpha[0, ..., 0], out[..., -1]
    return out, alpha[0, ..., 0], None


def dense_rasterize(scene, colours, viewmat, K, width, height, depth=False):
    """A reference rasterizer in plain torch (tests, tiny images): each gaussian an
    isotropic splat of its mean scale, composited front to back. Dense in (n, h, w)."""
    import torch

    cam = scene.means @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = cam[:, 2].clamp(min=1e-3)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    sigma = (K[0, 0] * scene.scales.mean(dim=1) / z).clamp(min=0.3)
    order = torch.argsort(z)
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=z.dtype) + 0.5,
        torch.arange(width, dtype=z.dtype) + 0.5,
        indexing="ij",
    )
    d2 = (xs[None] - u[order, None, None]) ** 2 + (ys[None] - v[order, None, None]) ** 2
    a = scene.opacities[order, None, None] * torch.exp(-0.5 * d2 / sigma[order, None, None] ** 2)
    a = (a * (cam[order, 2, None, None] > 1e-3)).clamp(max=0.99)
    transmit = torch.cumprod(torch.cat([torch.ones_like(a[:1]), 1 - a[:-1]]), dim=0)
    weight = a * transmit
    out = torch.einsum("nhw,nc->hwc", weight, colours[order])
    alpha = weight.sum(dim=0)
    if depth:
        expected = (weight * z[order, None, None]).sum(dim=0) / alpha.clamp(min=1e-6)
        return out, alpha, expected
    return out, alpha, None


def view_tensors(view: TrainView, width: int, height: int, device: str) -> tuple[Any, Any]:
    import torch

    viewmat = torch.as_tensor(view.viewmat, dtype=torch.float32, device=device)
    K = torch.as_tensor(view.at(width, height), dtype=torch.float32, device=device)
    return viewmat, K


# ------------------------------------------------------------------- masks and scales


class MaskSource:
    """What `masks(rgb)` returns: `segment_scene.Mask`s (bool (h, w), level, score)."""

    name: str

    def masks(self, rgb: np.ndarray) -> list[segment_scene.Mask]:  # pragma: no cover
        raise NotImplementedError


def sam_masks(model: str = SAM_MODEL) -> Any:
    """SAM 2.1 automatic masks for the field: `segment_models.Sam2Masks` with the larger
    checkpoint and the whole-view ceiling at `MASK_MAX_AREA`."""
    import segment_models

    return segment_models.Sam2Masks(model=model, max_area=MASK_MAX_AREA)


def robust_diameter(points: np.ndarray) -> float:
    """The diameter of a set of points that ignores its outer tenth on every axis: the
    10th-90th percentile box's diagonal, over 0.8 (exact for a uniform box)."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    if len(p) < 2:
        return 0.0
    lo, hi = np.percentile(p, [10, 90], axis=0)
    return float(np.linalg.norm(hi - lo) / 0.8)


def unproject(view_K: np.ndarray, viewmat: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """World points (h, w, 3) at each pixel's depth along the view axis."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    x = (u + 0.5 - view_K[0, 2]) / view_K[0, 0]
    y = (v + 0.5 - view_K[1, 2]) / view_K[1, 1]
    cam = np.stack([x, y, np.ones_like(x)], axis=-1) * depth[..., None]
    r, t = viewmat[:3, :3], viewmat[:3, 3]
    return (cam - t) @ r


def mask_scales(
    masks: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    *,
    seed: int = 0,
    max_points: int = MASK_SCALE_POINTS,
) -> np.ndarray:
    """Per mask (m, h, w), the robust diameter of the 3D points under it (`points`, where
    `valid`); NaN where less than `MASK_MIN_DEPTH_SHARE` of it has depth."""
    rng = np.random.default_rng(seed)
    out = np.full(len(masks), np.nan)
    for k, mask in enumerate(masks):
        inside = np.flatnonzero(mask.reshape(-1))
        if inside.size == 0:
            continue
        good = inside[valid.reshape(-1)[inside]]
        if good.size < max(3, MASK_MIN_DEPTH_SHARE * inside.size):
            continue
        if good.size > max_points:
            good = rng.choice(good, max_points, replace=False)
        out[k] = robust_diameter(points.reshape(-1, 3)[good])
    return out


def shrink_masks(masks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bool masks (m, H, W) at (height, width): a pixel is in a mask when most of it is."""
    import cv2

    if masks.shape[1:] == (height, width):
        return masks.astype(bool)
    out = np.zeros((len(masks), height, width), bool)
    for k, mask in enumerate(masks):
        out[k] = cv2.resize(mask.astype(np.float32), (width, height), cv2.INTER_AREA) > 0.5
    return out


@dataclass
class Supervision:
    """What one view teaches: its camera, the masks' membership per pixel (bit-packed), each
    mask's scale, and which pixels show the splat, at the training size."""

    view: TrainView
    width: int
    height: int
    bits: np.ndarray  # (h * w, ceil(m / 8)) uint8
    scales: np.ndarray  # (m,) float32
    valid: np.ndarray  # (h * w,) bool

    @property
    def count(self) -> int:
        return int(self.scales.size)

    def membership(self, pixels: np.ndarray) -> np.ndarray:
        """(len(pixels), m) bool."""
        if self.count == 0:
            return np.zeros((len(pixels), 0), bool)
        return np.unpackbits(self.bits[pixels], axis=1, count=self.count).astype(bool)


def supervise(
    view: TrainView,
    masks: Sequence[segment_scene.Mask],
    depth: np.ndarray,
    alpha: np.ndarray,
    *,
    side: int = TRAIN_SIDE,
    seed: int = 0,
) -> Supervision:
    """One view's `Supervision` from its masks (at the image's size) and the splat's depth
    and coverage drawn with the same camera at that size."""
    stack = np.array([m.mask for m in masks], bool).reshape(len(masks), view.height, view.width)
    valid = (alpha >= MIN_ALPHA) & np.isfinite(depth)
    points = unproject(view.K, view.viewmat, np.where(valid, depth, 0.0))
    scales = mask_scales(stack, points, valid, seed=seed)
    keep = np.isfinite(scales) & (scales > 0)
    width, height = fit_side(view.width, view.height, side)
    small = shrink_masks(stack[keep], width, height)
    small_valid = shrink_masks(valid[None], width, height)[0]
    flat = small.reshape(len(small), -1).T
    bits = np.packbits(flat, axis=1) if len(small) else np.zeros((width * height, 0), np.uint8)
    return Supervision(
        view,
        width,
        height,
        np.ascontiguousarray(bits),
        scales[keep].astype(np.float32),
        small_valid.reshape(-1),
    )


# ------------------------------------------------------------------------- the field


@dataclass
class Field:
    """A trained field: per gaussian a unit feature; the gate's weights; how often each
    gaussian got a gradient; every mask's scale (the range the field was trained over)."""

    features: np.ndarray  # (n, d) float32, unit rows
    gate: dict[str, np.ndarray]
    hits: np.ndarray  # (n,) int32
    scales: np.ndarray  # (masks,) float32
    stats: dict[str, Any] = field(default_factory=dict)

    def scale_range(self) -> tuple[float, float]:
        if self.scales.size == 0:
            return (1e-3, 1.0)
        lo, hi = np.quantile(self.scales, SCALE_QUANTILES)
        return float(lo), float(max(hi, lo * 1.01))

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            features=self.features.astype(np.float16),
            hits=self.hits.astype(np.int32),
            scales=self.scales.astype(np.float32),
            stats=np.frombuffer(json.dumps(self.stats).encode(), np.uint8),
            **{f"gate_{k}": v.astype(np.float32) for k, v in self.gate.items()},
        )

    @staticmethod
    def load(path: Path) -> Field:
        with np.load(path) as z:
            gate = {k[5:]: z[k].astype(np.float64) for k in z.files if k.startswith("gate_")}
            stats = json.loads(bytes(z["stats"]).decode()) if "stats" in z.files else {}
            features = z["features"].astype(np.float32)
            norm = np.linalg.norm(features, axis=1, keepdims=True)
            return Field(
                features / np.maximum(norm, 1e-12),
                gate,
                z["hits"].astype(np.int32),
                z["scales"].astype(np.float32),
                stats,
            )


def gate_values(gate: dict[str, np.ndarray], scales: np.ndarray) -> np.ndarray:
    """`g(s)` (len(scales), d) in (0, 1): `sigmoid(w2 relu(w1 u + b1) + b2)`, `u` the log of
    the scale over the gate's reference."""
    u = (np.log(np.maximum(np.asarray(scales, np.float64), 1e-9)) - gate["ref"][0])[:, None]
    hidden = np.maximum(u @ gate["w1"].T + gate["b1"], 0.0)
    return 1.0 / (1.0 + np.exp(-(hidden @ gate["w2"].T + gate["b2"])))


def gated(features: np.ndarray, gate_row: np.ndarray) -> np.ndarray:
    """Unit rows of `g * f` (one gate row for all, or one per row)."""
    out = np.asarray(features, np.float64) * gate_row
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)


def pair_loss(features, alpha_norm, member, mask_scales_t, anchor_scales, gate_fn, margin):
    """The contrastive loss of one step (torch): `features` (k, d) the sampled pixels'
    rendered features, `member` (k, m) float their masks, `mask_scales_t` (m,), `anchor_scales`
    (k,) each row's scale, `gate_fn(scales) -> (k, d)`. Returns (loss, stats)."""
    import torch

    g = gate_fn(anchor_scales)  # (k, d)
    g2 = g * g
    num = (features * g2) @ features.T  # sum_d g_i^2 f_i f_j
    own = torch.sqrt((g2 * features * features).sum(dim=1).clamp(min=1e-12))  # |g_i f_i|
    other = torch.sqrt((g2 @ (features * features).T).clamp(min=1e-12))  # |g_i f_j|
    cos = num / (own[:, None] * other)
    small = (mask_scales_t[None, :] <= anchor_scales[:, None]).float()  # (k, m)
    held = member * small
    rows = held.sum(dim=1) > 0  # the anchor is in some mask at most its scale
    positive = (held @ member.T) > 0
    eye = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive &= ~eye
    negative = ~positive & ~eye
    positive &= rows[:, None]
    negative &= rows[:, None]
    zero = features.sum() * 0.0
    pos = (1.0 - cos)[positive].mean() if positive.any() else zero
    neg = torch.relu(cos - margin)[negative].mean() if negative.any() else zero
    loss = pos + neg + CONSISTENCY_WEIGHT * alpha_norm
    return loss, {
        "positive": float(pos.detach()),
        "negative": float(neg.detach()),
        "consistency": float(alpha_norm.detach()),
        "positiveShare": float(positive.float().mean().detach()),
    }


def train_field(
    splats: Splats,
    supervision: Sequence[Supervision],
    *,
    dim: int = FEATURE_DIM,
    steps: int = STEPS,
    pixels: int = PIXELS,
    seed: int = 0,
    device: str | None = None,
    rasterize: Rasterize | None = None,
    progress: Callable[[str], None] | None = None,
) -> Field:
    """Train per-gaussian features and the scale gate from the views' masks (module
    docstring, "The loss"). The gaussians stay as they are."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rasterize = rasterize or (gsplat_rasterize if device == "cuda" else dense_rasterize)
    scene = Scene.of(splats, device)
    n = len(splats)
    all_scales = np.concatenate([s.scales for s in supervision] or [np.zeros(0, np.float32)])
    usable = [k for k, s in enumerate(supervision) if s.count and s.valid.any()]
    if not usable or all_scales.size == 0:
        raise ValueError("no view has a mask with a scale: nothing to train on")
    lo, hi = np.quantile(all_scales, SCALE_QUANTILES)
    hi = max(float(hi), float(lo) * 1.01)
    ref = float(np.exp(0.5 * (np.log(lo) + np.log(hi))))
    raw = torch.nn.Parameter(torch.randn(n, dim, device=device))
    w1 = torch.nn.Parameter(torch.randn(GATE_HIDDEN, 1, device=device))
    b1 = torch.nn.Parameter(torch.zeros(GATE_HIDDEN, device=device))
    w2 = torch.nn.Parameter(torch.randn(dim, GATE_HIDDEN, device=device) / math.sqrt(GATE_HIDDEN))
    b2 = torch.nn.Parameter(torch.full((dim,), 2.0, device=device))
    optimiser = torch.optim.Adam(
        [{"params": [raw], "lr": FEATURE_LR}, {"params": [w1, b1, w2, b2], "lr": GATE_LR}]
    )

    def gate_fn(scales: Any) -> Any:
        u = (torch.log(scales.clamp(min=1e-9)) - math.log(ref))[:, None]
        return torch.sigmoid(torch.relu(u @ w1.T + b1) @ w2.T + b2)

    cameras = {k: view_tensors(supervision[k].view, supervision[k].width,
                               supervision[k].height, device) for k in usable}  # fmt: skip
    scales_t = {k: torch.as_tensor(supervision[k].scales, device=device) for k in usable}
    hits = torch.zeros(n, dtype=torch.int32, device=device)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for step in range(steps):
        k = usable[int(rng.integers(len(usable)))]
        sup = supervision[k]
        viewmat, K = cameras[k]
        colours = torch.nn.functional.normalize(raw, dim=1)
        out, alpha, _ = rasterize(scene, colours, viewmat, K, sup.width, sup.height)
        flat = out.reshape(-1, dim)
        a = alpha.reshape(-1)
        seen = (a.detach().cpu().numpy() >= MIN_ALPHA) & sup.valid
        candidates = np.flatnonzero(seen)
        if candidates.size < 2:
            continue
        chosen = rng.choice(candidates, min(pixels, candidates.size), replace=False)
        index = torch.as_tensor(chosen, device=device)
        member = torch.as_tensor(sup.membership(chosen), dtype=torch.float32, device=device)
        anchor = torch.as_tensor(
            np.exp(rng.uniform(np.log(lo), np.log(hi), chosen.size)), dtype=torch.float32,
            device=device,
        )  # fmt: skip
        valid_t = torch.as_tensor(seen, device=device)
        norm = flat.norm(dim=1)
        consistency = ((a - norm).clamp(min=0) / a.clamp(min=1e-6))[valid_t].mean()
        loss, stats = pair_loss(
            flat[index], consistency, member, scales_t[k], anchor, gate_fn, NEGATIVE_MARGIN
        )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if raw.grad is not None:
            hits += (raw.grad.abs().sum(dim=1) > 0).int()
        optimiser.step()
        if step % 100 == 0 or step == steps - 1:
            stats.update(step=step, loss=float(loss.detach()))
            history.append(stats)
            if progress:
                progress(
                    f"step {step}: loss {stats['loss']:.4f} (+{stats['positive']:.3f} "
                    f"-{stats['negative']:.3f} c{stats['consistency']:.3f})"
                )
    features = torch.nn.functional.normalize(raw.detach(), dim=1).cpu().numpy()
    gate = {
        "w1": w1.detach().cpu().numpy().astype(np.float64),
        "b1": b1.detach().cpu().numpy().astype(np.float64),
        "w2": w2.detach().cpu().numpy().astype(np.float64),
        "b2": b2.detach().cpu().numpy().astype(np.float64),
        "ref": np.array([math.log(ref)]),
    }
    return Field(
        features.astype(np.float32),
        gate,
        hits.cpu().numpy().astype(np.int32),
        all_scales.astype(np.float32),
        {
            "steps": steps,
            "pixels": pixels,
            "dim": dim,
            "views": len(usable),
            "masks": int(all_scales.size),
            "scaleRange": [round(float(lo), 5), round(float(hi), 5)],
            "trainS": round(time.perf_counter() - started, 1),
            "history": history,
        },
    )


# ------------------------------------------------------------------------- the graph


@dataclass
class Graph:
    """Undirected edges `a < b` between gaussians (indices into the scan)."""

    a: np.ndarray
    b: np.ndarray
    n: int


def floaters(splats: Splats) -> np.ndarray:
    """Gaussians too large to be surface samples (`FLOATER_SCALE`, `FLOATER_QUANTILE`)."""
    largest = np.asarray(splats.scales).max(axis=1)
    if largest.size == 0:
        return np.zeros(0, bool)
    bound = min(FLOATER_SCALE * float(np.median(largest)), float(np.quantile(largest, FLOATER_QUANTILE)))
    return largest > max(bound, 1e-9)


def knn_graph(positions: np.ndarray, keep: np.ndarray, k: int = NEIGHBOURS) -> Graph:
    """Each kept gaussian joined to its `k` nearest kept ones, not farther than
    `EDGE_QUANTILE` of the k-th neighbours' distances."""
    rows = np.flatnonzero(keep)
    n = len(positions)
    if rows.size < 2:
        return Graph(np.zeros(0, np.int64), np.zeros(0, np.int64), n)
    pts = np.asarray(positions, np.float64)[rows]
    kk = min(k + 1, rows.size)
    distance, index = cKDTree(pts).query(pts, k=kk, workers=-1)
    cap = float(np.quantile(distance[:, -1], EDGE_QUANTILE))
    a = np.repeat(np.arange(rows.size), kk - 1)
    b = index[:, 1:].reshape(-1)
    d = distance[:, 1:].reshape(-1)
    ok = (d <= cap) & (b < rows.size) & (a != b)
    a, b = rows[a[ok]], rows[b[ok]]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = np.unique(lo * n + hi)
    return Graph(key // n, key % n, n)


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    graph = coo_matrix((np.ones(a.size, np.int8), (a, b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def edge_cosines(
    features: np.ndarray,
    gate: dict[str, np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    edge_scale: np.ndarray,
    chunk: int = 1 << 18,
) -> np.ndarray:
    """Per edge, the cosine of its two gaussians' features gated at its scale."""
    out = np.zeros(a.size)
    for start in range(0, a.size, chunk):
        sl = slice(start, start + chunk)
        g = gate_values(gate, edge_scale[sl])
        fa = gated(features[a[sl]], g)
        fb = gated(features[b[sl]], g)
        out[sl] = (fa * fb).sum(axis=1)
    return out


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Labels >= 0 renumbered 0..k-1 in order of first appearance; -1 kept."""
    out = np.full(labels.shape, -1, np.int64)
    has = labels >= 0
    _, first, inverse = np.unique(labels[has], return_index=True, return_inverse=True)
    rank = np.argsort(np.argsort(first))
    out[has] = rank[inverse]
    return out


def partition(
    node: np.ndarray,
    node_scale: np.ndarray,
    features: np.ndarray,
    gate: dict[str, np.ndarray],
    graph: Graph,
    *,
    edge_cosine: float = EDGE_COSINE,
    merge_cosine: float = MERGE_COSINE,
    min_splats: int = MIN_OBJECT_SPLATS,
    min_share: float = 0.0,
) -> np.ndarray:
    """Split every node (`node`: per gaussian its node, -1 none) by the field gated at the
    node's scale (`node_scale`, per node): fragments of edges at least `edge_cosine`;
    adjacent fragments whose mean features agree to `merge_cosine` joined, mutual best pairs
    first, in rounds; then fragments under `min_splats` (or `min_share` of their node) join
    the neighbour they share most edges with, and ones with no neighbour the node's largest.
    Returns per gaussian its piece (0..k-1, -1 where `node` is), every piece inside one node."""
    n = graph.n
    a, b = graph.a, graph.b
    same = (node[a] >= 0) & (node[a] == node[b])
    a, b = a[same], b[same]
    scale = node_scale[node[a]]
    cos = edge_cosines(features, gate, a, b, scale)
    strong = cos >= edge_cosine
    piece = _components(n, a[strong], b[strong]).astype(np.int64)
    piece[node < 0] = -1
    piece = _relabel(piece)
    k = int(piece.max()) + 1 if (piece >= 0).any() else 0
    if k == 0:
        return piece
    piece_node = np.zeros(k, np.int64)
    piece_node[piece[node >= 0]] = node[node >= 0]

    def means(piece: np.ndarray, k: int) -> np.ndarray:
        has = piece >= 0
        g = gate_values(gate, node_scale[piece_node_of(piece, k)])
        sums = np.zeros((k, features.shape[1]))
        rows = np.flatnonzero(has)
        for start in range(0, rows.size, 1 << 18):
            r = rows[start : start + (1 << 18)]
            np.add.at(sums, piece[r], gated(features[r], g[piece[r]]))
        return sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)

    def piece_node_of(piece: np.ndarray, k: int) -> np.ndarray:
        out = np.zeros(k, np.int64)
        has = piece >= 0
        out[piece[has]] = node[has]
        return out

    for _ in range(MERGE_ROUNDS):
        pa, pb = piece[a], piece[b]
        cross = pa != pb
        if not cross.any():
            break
        lo = np.minimum(pa[cross], pb[cross])
        hi = np.maximum(pa[cross], pb[cross])
        pairs = np.unique(lo * k + hi)
        pl, ph = pairs // k, pairs % k
        mean = means(piece, k)
        agree = (mean[pl] * mean[ph]).sum(axis=1)
        good = agree >= merge_cosine
        if not good.any():
            break
        pl, ph, agree = pl[good], ph[good], agree[good]
        # Mutual best: each piece's best partner, joined where the choice is mutual.
        best = np.full(k, -1.0)
        partner = np.full(k, -1, np.int64)
        for x, y in ((pl, ph), (ph, pl)):
            order = np.argsort(agree, kind="stable")
            best[x[order]] = agree[order]
            partner[x[order]] = y[order]
        mutual = partner[pl] == ph
        mutual &= partner[ph] == pl
        if not mutual.any():
            mutual = agree == agree.max()
        joined = _components(k, pl[mutual], ph[mutual])
        piece = np.where(piece >= 0, joined[np.maximum(piece, 0)], -1)
        piece = _relabel(piece)
        k = int(piece.max()) + 1
    # Crumbs join the neighbour they share most edges with.
    size_of_node = np.bincount(node[node >= 0], minlength=int(node.max()) + 1)
    for _ in range(MERGE_ROUNDS):
        size = np.bincount(piece[piece >= 0], minlength=k)
        floor = np.maximum(min_splats, min_share * size_of_node[piece_node_of(piece, k)])
        small = size < floor
        if not small.any():
            break
        pa, pb = piece[a], piece[b]
        cross = pa != pb
        src = np.concatenate([pa[cross], pb[cross]])
        dst = np.concatenate([pb[cross], pa[cross]])
        from_small = small[src]
        if not from_small.any():
            break
        key = src[from_small] * k + dst[from_small]
        keys, counts = np.unique(key, return_counts=True)
        # Prefer the larger neighbour on ties: count + its size as a fraction.
        target_size = size[keys % k]
        score = counts + target_size / (target_size.max() + 1.0)
        order = np.lexsort((score, keys // k))
        last = np.r_[np.flatnonzero(np.diff((keys // k)[order])), order.size - 1]
        mapping = np.arange(k)
        mapping[(keys // k)[order][last]] = (keys % k)[order][last]
        # Resolve chains (a crumb into a crumb that moves too).
        for _ in range(8):
            mapping = mapping[mapping]
        piece = np.where(piece >= 0, mapping[np.maximum(piece, 0)], -1)
        piece = _relabel(piece)
        k = int(piece.max()) + 1
    # Isolated crumbs (no edge to another piece of their node) join their node's largest.
    size = np.bincount(piece[piece >= 0], minlength=k)
    nodes_of = piece_node_of(piece, k)
    floor = np.maximum(min_splats, min_share * size_of_node[nodes_of])
    small = size < floor
    if small.any():
        largest = np.full(int(node.max()) + 1, -1, np.int64)
        order = np.argsort(size, kind="stable")
        largest[nodes_of[order]] = order
        target = np.where(small, largest[nodes_of], np.arange(k))
        piece = np.where(piece >= 0, target[np.maximum(piece, 0)], -1)
        piece = _relabel(piece)
    return piece


# --------------------------------------------------------------------------- the ground


@dataclass
class GroundLayer:
    """Which gaussians are ground and how high each is above the terrain."""

    is_ground: np.ndarray  # (n,) bool
    height: np.ndarray  # (n,) float, above the terrain
    source: str
    info: dict[str, Any] = field(default_factory=dict)


def ground_layer(splats: Splats, skip: np.ndarray | None = None) -> GroundLayer:
    """The ground: the shared ground pass (`ground_pass`, bake-off branch
    bakeoff-seg-ground-first) when it is importable, else `scene_plants`' slope filter
    (Vosselman 2000) on the gaussians that are not floaters (`skip`), the ground layer being
    its own estimate of how far resting splats scatter above the surface."""
    try:
        import ground_pass  # type: ignore[import-not-found]
    except ImportError:
        ground_pass = None
    if ground_pass is not None and hasattr(ground_pass, "feature_field_ground"):
        found = ground_pass.feature_field_ground(splats)
        return GroundLayer(
            np.asarray(found["isGround"], bool),
            np.asarray(found["height"], np.float64),
            "ground_pass",
            dict(found.get("info", {})),
        )
    pos = np.asarray(splats.positions, np.float64)
    keep = np.ones(len(pos), bool) if skip is None else ~skip
    keep &= np.asarray(splats.opacities) >= 0.1
    sample = pos[keep]
    if len(sample) < 16:
        return GroundLayer(np.zeros(len(pos), bool), np.zeros(len(pos)), "none")
    rng = np.random.default_rng(0)
    probe = sample[rng.choice(len(sample), min(len(sample), 20000), replace=False)]
    spacing = float(np.median(cKDTree(sample).query(probe, k=2)[0][:, 1]))
    model = scene_plants.ground_model(sample, spacing)
    height = pos[:, 2] - model.at(pos[:, :2])
    return GroundLayer(
        height <= model.layer_m,
        height,
        "scene_plants slope filter",
        {"layer": round(float(model.layer_m), 4), **model.to_json()},
    )


def claim_contact(
    labels: np.ndarray,
    ground: np.ndarray,
    features: np.ndarray,
    gate: dict[str, np.ndarray],
    graph: Graph,
    object_scale: np.ndarray,
    margin: float = CONTACT_MARGIN,
) -> np.ndarray:
    """Ground gaussians that touch an object (`labels` > 0, per gaussian its top object)
    and whose feature, gated at the object's scale, is nearer the object's mean than the mean
    of the ground gaussians beside them by `margin` join the object. Repeated while any do,
    so a flange's lowest ring is claimed from the outside in."""
    labels = labels.copy()
    a, b = graph.a, graph.b
    k = int(labels.max()) + 1
    if k <= 1:
        return labels
    claimed_total = 0
    for _ in range(6):
        on_ground = ground & (labels <= 0)
        touch = (on_ground[a] & (labels[b] > 0)) | (on_ground[b] & (labels[a] > 0))
        if not touch.any():
            break
        ga = np.where(on_ground[a[touch]], a[touch], b[touch])
        ob = np.where(on_ground[a[touch]], b[touch], a[touch])
        target = labels[ob]
        # The object's mean feature, gated at its scale.
        has = labels > 0
        g_obj = gate_values(gate, object_scale)
        sums = np.zeros((k, features.shape[1]))
        rows = np.flatnonzero(has)
        np.add.at(sums, labels[rows], gated(features[rows], g_obj[labels[rows]]))
        mean = sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)
        # The ground beside each candidate: its ground neighbours' mean, at the same scale.
        both_ground = on_ground[a] & on_ground[b]
        src = np.concatenate([a[both_ground], b[both_ground]])
        dst = np.concatenate([b[both_ground], a[both_ground]])
        cand, first = np.unique(ga, return_index=True)
        obj = target[first]
        position = np.full(len(labels), -1, np.int64)
        position[cand] = np.arange(cand.size)
        sel = position[src] >= 0
        g_c = g_obj[obj]
        nsum = np.zeros((cand.size, features.shape[1]))
        np.add.at(nsum, position[src[sel]], gated(features[dst[sel]], g_c[position[src[sel]]]))
        nmean = nsum / np.maximum(np.linalg.norm(nsum, axis=1, keepdims=True), 1e-12)
        own = gated(features[cand], g_c)
        to_object = (own * mean[obj]).sum(axis=1)
        to_ground = np.where(np.linalg.norm(nsum, axis=1) > 0, (own * nmean).sum(axis=1), -1.0)
        win = to_object >= to_ground + margin
        if not win.any():
            break
        labels[cand[win]] = obj[win]
        claimed_total += int(win.sum())
    return labels


# ---------------------------------------------------------------------------- the tree


@dataclass
class Tree:
    """Per gaussian its deepest node (1-based; 0 none); per node its parent (0 none), depth,
    scale (the robust diameter it was split at) and kind ("object" | "ground-region")."""

    leaf: np.ndarray
    parent: np.ndarray
    level: np.ndarray
    scale: np.ndarray
    kind: list[str]
    stats: dict[str, Any] = field(default_factory=dict)


def _diameters(positions: np.ndarray, label: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Per label 0..k-1, the robust diameter of its gaussians (a sample of each)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(label, kind="stable")
    bounds = np.searchsorted(label[order], np.arange(k + 1))
    out = np.zeros(k)
    for j in range(k):
        rows = order[bounds[j] : bounds[j + 1]]
        if rows.size > 4000:
            rows = rng.choice(rows, 4000, replace=False)
        out[j] = robust_diameter(positions[rows])
    return out


def fill_unseen(features: np.ndarray, hits: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Gaussians the training barely reached (`MIN_HITS`) take the mean feature of their
    eight nearest reached neighbours."""
    reached = hits >= MIN_HITS
    if reached.all() or not reached.any():
        return features
    tree = cKDTree(np.asarray(positions, np.float64)[reached])
    missing = np.flatnonzero(~reached)
    _, index = tree.query(np.asarray(positions, np.float64)[missing], k=min(8, int(reached.sum())))
    index = np.asarray(index).reshape(missing.size, -1)
    source = np.flatnonzero(reached)[index]
    out = features.copy()
    mean = features[source].mean(axis=1)
    out[missing] = mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    return out


def build_tree(
    splats: Splats,
    field_: Field,
    ground: GroundLayer,
    *,
    graph: Graph | None = None,
    skip: np.ndarray | None = None,
    max_depth: int = MAX_DEPTH,
) -> Tree:
    """Objects (and their parts) above the ground and regions of the ground, from the field
    (module docstring, "The tree"). Floaters (`skip`) and specks get no node here."""
    pos = np.asarray(splats.positions, np.float64)
    n = len(pos)
    skip = floaters(splats) if skip is None else skip
    graph = graph or knn_graph(pos, ~skip)
    features = fill_unseen(field_.features, field_.hits, pos)
    lo, hi = field_.scale_range()
    clamp = lambda s: np.clip(s, lo, hi)  # noqa: E731
    stats: dict[str, Any] = {"edges": int(graph.a.size), "floaters": int(skip.sum())}

    # 1. The connected parts above the ground.
    above = ~ground.is_ground & ~skip
    keep_edge = above[graph.a] & above[graph.b]
    comp = _components(n, graph.a[keep_edge], graph.b[keep_edge]).astype(np.int64)
    comp[~above] = -1
    comp = _relabel(comp)
    kc = int(comp.max()) + 1 if (comp >= 0).any() else 0
    size = np.bincount(comp[comp >= 0], minlength=kc)
    comp = np.where((comp >= 0) & (size[np.maximum(comp, 0)] >= MIN_OBJECT_SPLATS), comp, -1)
    comp = _relabel(comp)
    kc = int(comp.max()) + 1 if (comp >= 0).any() else 0
    stats["components"] = kc

    # 2. Objects: each component split at its own size.
    nodes: list[tuple[int, int, float, str]] = []  # (parent node id, depth, scale, kind)
    leaf = np.zeros(n, np.int64)
    if kc:
        diam = _diameters(pos, comp, kc)
        top = partition(comp, clamp(TOP_SCALE * diam), features, field_.gate, graph)
        kt = int(top.max()) + 1 if (top >= 0).any() else 0
        top_diam = _diameters(pos, top, kt)
        # 4. Contact: ground beside an object, the object's by its feature.
        claimed = claim_contact(
            np.where(top >= 0, top + 1, 0),
            ground.is_ground & ~skip,
            features,
            field_.gate,
            graph,
            np.r_[lo, clamp(TOP_SCALE * top_diam)],
        )
        stats["contactClaimed"] = int(((claimed > 0) & (top < 0)).sum())
        top = claimed - 1
        for j in range(kt):
            nodes.append((0, 0, float(top_diam[j]), "object"))
        leaf = np.where(top >= 0, top + 1, 0)
        # 3. Parts, depth by depth.
        current = leaf.copy()  # node id per gaussian at the depth being split
        for depth in range(1, max_depth):
            ids = np.unique(current[current > 0])
            ids = np.array([i for i in ids if nodes[i - 1][1] == depth - 1], np.int64)
            if ids.size == 0:
                break
            node = np.full(n, -1, np.int64)
            index = np.full(len(nodes) + 1, -1, np.int64)
            index[ids] = np.arange(ids.size)
            node[current > 0] = index[current[current > 0]]
            node_d = np.array([nodes[i - 1][2] for i in ids])
            pieces = None
            for factor in (CHILD_SCALE, CHILD_SCALE**2):
                attempt = partition(
                    node, clamp(factor * node_d), features, field_.gate, graph,
                    min_share=MIN_CHILD_SHARE,
                )  # fmt: skip
                if pieces is None:
                    pieces = attempt
                    continue
                # Nodes the first scale left whole take the second scale's split.
                kp = int(pieces.max()) + 1
                count = np.bincount(node[node >= 0], minlength=ids.size)
                per_node = np.zeros(ids.size, np.int64)
                firsts = np.unique(np.c_[node[pieces >= 0], pieces[pieces >= 0]], axis=0)
                np.add.at(per_node, firsts[:, 0], 1)
                whole = (per_node <= 1) & (count > 0)
                take = (node >= 0) & whole[np.maximum(node, 0)]
                pieces = np.where(take, attempt + kp, pieces)
                pieces = _relabel(pieces)
            assert pieces is not None
            kp = int(pieces.max()) + 1 if (pieces >= 0).any() else 0
            if kp == 0:
                break
            piece_node = np.zeros(kp, np.int64)
            piece_node[pieces[pieces >= 0]] = node[pieces >= 0]
            per_node = np.bincount(piece_node, minlength=ids.size)
            piece_d = _diameters(pos, pieces, kp)
            new_current = current.copy()
            made = 0
            for p in range(kp):
                if per_node[piece_node[p]] < 2:
                    continue  # a node split into one piece is not split
                nodes.append((int(ids[piece_node[p]]), depth, float(piece_d[p]), "object"))
                new_current[pieces == p] = len(nodes)
                made += 1
            stats[f"depth{depth}Nodes"] = made
            if made == 0:
                break
            current = new_current
        leaf = current
    stats["objects"] = int(sum(1 for nd in nodes if nd[1] == 0))

    # 5. Ground regions.
    ground_rows = ground.is_ground & ~skip & (leaf == 0)
    keep_edge = ground_rows[graph.a] & ground_rows[graph.b]
    gcomp = _components(n, graph.a[keep_edge], graph.b[keep_edge]).astype(np.int64)
    gcomp[~ground_rows] = -1
    gcomp = _relabel(gcomp)
    kg = int(gcomp.max()) + 1 if (gcomp >= 0).any() else 0
    if kg:
        cover = float(np.quantile(field_.scales, COVER_SCALE_QUANTILE)) if field_.scales.size else lo
        regions = partition(
            gcomp, np.full(kg, clamp(cover)), features, field_.gate, graph,
            min_splats=max(MIN_OBJECT_SPLATS, int(MIN_COVER_SHARE * ground_rows.sum())),
        )  # fmt: skip
        kr = int(regions.max()) + 1 if (regions >= 0).any() else 0
        region_d = _diameters(pos, regions, kr)
        base = len(nodes)
        for r in range(kr):
            nodes.append((0, 0, float(region_d[r]), "ground-region"))
        leaf = np.where(regions >= 0, base + 1 + regions, leaf)
        stats["groundRegions"] = kr
    parent = np.array([nd[0] for nd in nodes], np.int64)
    level = np.array([nd[1] for nd in nodes], np.int64)
    scale = np.array([nd[2] for nd in nodes], np.float64)
    kind = [nd[3] for nd in nodes]
    return Tree(leaf, parent, level, scale, kind, stats)


def fill_rest(leaf: np.ndarray, positions: np.ndarray, reach: np.ndarray) -> np.ndarray:
    """Gaussians without a node (floaters, specks) take their nearest labelled gaussian's,
    within `reach` (per gaussian)."""
    missing = np.flatnonzero(leaf == 0)
    labelled = np.flatnonzero(leaf > 0)
    if missing.size == 0 or labelled.size == 0:
        return leaf
    pos = np.asarray(positions, np.float64)
    distance, index = cKDTree(pos[labelled]).query(pos[missing], k=1, workers=-1)
    ok = distance <= reach[missing]
    out = leaf.copy()
    out[missing[ok]] = leaf[labelled[index[ok]]]
    return out


def renumber(tree: Tree) -> tuple[Tree, np.ndarray]:
    """Breadth first, largest first (as `segment_scene._hierarchy` numbers instances), so
    a parent's id is below its children's. Returns the tree and old-to-new ids (index 0: 0)."""
    k = tree.parent.size
    own = np.bincount(tree.leaf, minlength=k + 1)[1:].astype(np.float64)
    total = segment_scene._up(own, tree.parent, tree.level, "sum")
    new = np.zeros(k + 1, np.int64)
    next_id = 1
    for depth in range(int(tree.level.max()) + 1 if k else 0):
        nodes = np.flatnonzero(tree.level == depth)
        order = np.lexsort((nodes, -total[nodes], new[tree.parent[nodes]]))
        for j in nodes[order]:
            new[j + 1] = next_id
            next_id += 1
    parent = np.zeros(k, np.int64)
    level = np.zeros(k, np.int64)
    scale = np.zeros(k)
    kind = [""] * k
    for j in range(k):
        parent[new[j + 1] - 1] = new[tree.parent[j]]
        level[new[j + 1] - 1] = tree.level[j]
        scale[new[j + 1] - 1] = tree.scale[j]
        kind[new[j + 1] - 1] = tree.kind[j]
    return Tree(new[tree.leaf], parent, level, scale, kind, tree.stats), new


# ------------------------------------------------------------------ views to train on


def frame_check(
    scene: Scene, views: Sequence[TrainView], rasterize: Rasterize, device: str, count: int = 16
) -> dict[str, float]:
    """How well the splat drawn from each photo's pose matches the photo: the median PSNR
    over the pixels it covers, and its median coverage, over up to `count` photos. A pose
    in the wrong frame scores near noise."""
    import cv2
    import torch

    psnr, cover = [], []
    picked = list(views)[:: max(1, len(views) // count)][:count]
    for view in picked:
        width, height = fit_side(view.width, view.height, 256)
        viewmat, K = view_tensors(view, width, height, device)
        with torch.no_grad():
            out, alpha, _ = rasterize(scene, scene.colours, viewmat, K, width, height)
        rgb = out.clamp(0, 1).cpu().numpy()
        a = alpha.cpu().numpy()
        photo = cv2.resize(view.image, (width, height), interpolation=cv2.INTER_AREA) / 255.0
        good = a >= MIN_ALPHA
        cover.append(float(good.mean()))
        if good.sum() < 16:
            psnr.append(0.0)
            continue
        mse = float(((rgb[good] - photo[good]) ** 2).mean())
        psnr.append(10.0 * math.log10(1.0 / max(mse, 1e-10)))
    return {
        "psnr": round(float(np.median(psnr)) if psnr else 0.0, 2),
        "coverage": round(float(np.median(cover)) if cover else 0.0, 3),
        "checked": len(picked),
    }


def rendered_views(
    splats: Splats,
    scene: Scene,
    rasterize: Rasterize,
    device: str,
    *,
    count: int = 48,
    max_views: int = 96,
) -> list[TrainView]:
    """Views drawn from the splat (`segment_scene.plan_views`: rings, eye-height views where
    it was seen from, and local views on a wide scan), for a scan without photos."""
    import torch

    _, centroids, _, edge = segment_scene.supervoxels(splats.positions)
    cameras = segment_scene.plan_views(
        splats.positions,
        count,
        observers=segment_scene.observer_points(splats),
        edge=edge,
        solid=centroids,
        max_views=max_views,
    )
    out = []
    for k, camera in enumerate(cameras):
        view = camera_view(camera, np.zeros((camera.height, camera.width, 3), np.uint8), f"r{k}")
        viewmat, K = view_tensors(view, camera.width, camera.height, device)
        with torch.no_grad():
            rgb, _, _ = rasterize(scene, scene.colours, viewmat, K, camera.width, camera.height)
        view.image = np.round(rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        out.append(view)
    return out


def supervision_of(
    views: Sequence[TrainView],
    source: Any,
    scene: Scene,
    rasterize: Rasterize,
    device: str,
    progress: Callable[[str], None] | None = None,
) -> list[Supervision]:
    """Every view's masks, measured in 3D on the splat's depth from the same camera."""
    import torch

    out = []
    for k, view in enumerate(views):
        masks = source.masks(view.image)
        viewmat, K = view_tensors(view, view.width, view.height, device)
        with torch.no_grad():
            _, alpha, depth = rasterize(
                scene, scene.colours, viewmat, K, view.width, view.height, depth=True
            )
        sup = supervise(view, masks, depth.cpu().numpy(), alpha.cpu().numpy(), seed=k)
        out.append(sup)
        if progress:
            progress(f"view {k + 1}/{len(views)}: {len(masks)} masks, {sup.count} with a scale")
    return out


# ------------------------------------------------------------------------ describing


def cover_classes(embedder: Any, embedding: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Per row of `embedding` (unit, or zero where undescribed), the probability of each
    `COVER_CLASSES` phrase: a softmax at `segment_scene.LOGIT_SCALE` over the phrases'
    text embeddings. Returns the class names and (rows, classes)."""
    names = list(COVER_CLASSES)
    text = np.asarray(embedder.embed_texts([COVER_CLASSES[c][0] for c in names]), np.float64)
    text /= np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    logits = segment_scene.LOGIT_SCALE * np.asarray(embedding, np.float64) @ text.T
    return names, segment_scene._softmax(logits)


def describe_views(
    splats: Splats,
    leaf: np.ndarray,
    skip: np.ndarray,
    *,
    count: int = DESCRIBE_VIEWS,
    workers: int = 1,
) -> list[segment_scene.View]:
    """Views of the scan for describing (`segment_scene.plan_views`), rendered on the CPU
    with each pixel's dominant leaf instance (`segment_scene.RenderPool`; call it before any
    model starts its threads, which a fork would copy mid-flight), without the floaters."""
    keep = np.flatnonzero(~skip)
    shown = splats.take(keep)
    cells = leaf[keep].astype(np.int64)
    _, centroids, _, edge = segment_scene.supervoxels(shown.positions)
    cameras = segment_scene.plan_views(
        shown.positions,
        count,
        observers=segment_scene.observer_points(shown),
        edge=edge,
        solid=centroids,
        max_views=count * 2,
    )
    index = segment_scene.SplatIndex.build(shown)
    with segment_scene.RenderPool(shown, cells, index=index, workers=workers) as pool:
        return list(pool.views(cameras))


@dataclass
class Described:
    """The instances in their final numbering, per gaussian its leaf instance, and per
    instance its kind, scale and (for a ground-cover class) what it covers."""

    instances: list[segment_scene.Instance]
    leaf: np.ndarray
    kind: dict[int, str]
    scale: dict[int, float]
    cover: dict[int, dict[str, Any]]


def ground_records(
    tree: Tree,
    instances: Sequence[segment_scene.Instance],
    positions: np.ndarray,
    embedder: Any,
) -> Described:
    """The final instances: the objects as described, and the ground regions gathered under
    one `Ground` instance by cover class (`cover_classes`): Ground, then a class, then (where
    a class has more than one) its regions. Renumbered breadth first, largest first."""
    k = tree.parent.size
    regions = [j for j in range(k) if tree.kind[j] == "ground-region"]
    dim = int(instances[0].embedding.size) if instances else 0
    by_class: dict[str, list[int]] = {}
    probability: dict[int, float] = {}
    if regions:
        emb = np.stack([instances[j].embedding for j in regions])
        names, p = cover_classes(embedder, emb)
        for row, j in enumerate(regions):
            described = bool(np.any(emb[row]))
            c = names[int(p[row].argmax())] if described else "unknown"
            probability[j] = float(p[row].max()) if described else 0.0
            by_class.setdefault(c, []).append(j)
    # The grown tree: objects as they are; Ground, its classes, their regions.
    parent = list(tree.parent)
    level = list(tree.level)
    scale = list(tree.scale)
    kind = list(tree.kind)
    #: Per added or relabelled node (old 1-based id): its label and its regions (old ids).
    labelled: dict[int, tuple[str, list[int]]] = {}
    if regions:
        parent.append(0)
        level.append(0)
        scale.append(0.0)
        kind.append("ground")
        ground_node = len(parent)
        labelled[ground_node] = (GROUND_LABEL, [])
        for c, members in sorted(by_class.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(members) == 1:
                j = members[0]
                parent[j], level[j], kind[j] = ground_node, 1, "ground-cover"
                labelled[j + 1] = (c, [j])
                continue
            parent.append(ground_node)
            level.append(1)
            scale.append(0.0)
            kind.append("ground-cover")
            labelled[len(parent)] = (c, members)
            for j in members:
                parent[j], level[j] = len(parent), 2
    grown = Tree(
        tree.leaf.copy(), np.asarray(parent, np.int64), np.asarray(level, np.int64),
        np.asarray(scale, np.float64), kind, tree.stats,
    )  # fmt: skip
    final, new = renumber(grown)
    count = final.parent.size
    out: list[segment_scene.Instance | None] = [None] * count
    for j, instance in enumerate(instances):
        i = int(new[j + 1])
        instance.id = i
        instance.parent = int(final.parent[i - 1]) or None
        instance.level = int(final.level[i - 1])
        out[i - 1] = instance
    own = np.bincount(final.leaf, minlength=count + 1)[1:]
    below: dict[int, list[int]] = {}
    for i in range(1, count + 1):
        p = int(final.parent[i - 1])
        while p:
            below.setdefault(p, []).append(i)
            p = int(final.parent[p - 1])
    cover: dict[int, dict[str, Any]] = {}
    # Deepest first, so a class is finished before Ground averages over it.
    for old in sorted(labelled, key=lambda o: -int(final.level[new[o] - 1])):
        label, members = labelled[old]
        i = int(new[old])
        subtree = np.asarray([i] + below.get(i, []), np.int64)
        rows = np.flatnonzero(np.isin(final.leaf, subtree))
        p = positions[rows] if rows.size else np.zeros((1, 3))
        lo, hi, centroid = p.min(axis=0), p.max(axis=0), p.mean(axis=0)
        instance = out[i - 1]
        if instance is None:  # a node made here: Ground, or a class of several regions
            parts = [out[d - 1] for d in below.get(i, []) if out[d - 1] is not None]
            emb = sum((b.embedding for b in parts), np.zeros(dim))
            norm = float(np.linalg.norm(emb))
            props = {
                name: round(float(np.mean([b.properties.get(name, 0.0) for b in parts])), 4)
                if parts
                else 0.0
                for name in segment_scene.PROPERTY_PROMPTS
            }
            instance = segment_scene.Instance(
                id=i, parent=int(final.parent[i - 1]) or None, level=int(final.level[i - 1]),
                splats=int(own[i - 1]), bounds_min=lo, bounds_max=hi, centroid=centroid,
                views=max((b.views for b in parts), default=0),
                embedding=emb / norm if norm > 0 else np.zeros(dim), tags=[],
                properties=props, behaviour="static",
            )  # fmt: skip
            out[i - 1] = instance
        instance.bounds_min, instance.bounds_max, instance.centroid = lo, hi, centroid
        instance.behaviour = "static"
        if label == GROUND_LABEL:
            instance.tags = [{"label": GROUND_LABEL, "score": 1.0}]
            instance.category = "ground"
            continue
        category = COVER_CLASSES.get(label, ("", "ground"))[1]
        score = round(float(np.mean([probability[j] for j in members])), 4)
        instance.tags = [{"label": label, "score": score}]
        instance.category = category
        cover[i] = {"class": label, "regions": [int(new[j + 1]) for j in members], "score": score}
        for j in members:
            region = out[int(new[j + 1]) - 1]
            if region is not None and region is not instance:
                region.tags = [{"label": label, "score": round(probability[j], 4)}]
                region.category = category
    instances_out = [i for i in out if i is not None]
    kinds = {i + 1: final.kind[i] for i in range(count)}
    scales = {i + 1: round(float(final.scale[i]), 4) for i in range(count)}
    return Described(instances_out, final.leaf, kinds, scales, cover)
