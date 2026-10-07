"""Inferred fill, object round: an object's unseen side completed by a 3D object-completion model,
registered to the real scan, the measured splats untouched.

Round 2 (`anchor_fill`) fills what the cameras saw badly by painting views and lifting them. An
object the cameras saw from one side only -- the pumpkins on their straw, whose undersides and
the ground under them no camera saw (the cameras are at 38-40 degrees and above) -- is better
made whole by a model that knows what such objects are like all round. Nothing is picked by
hand: the objects are the segmentation's (`extras.variants.objects`), the photos the ones that
see each object best.

1. **Objects** (`object_targets`): the segmentation variant's instances whose concept is the
   object's (the pumpkin), each with its subtree; the gaussians that carry them (the
   instances' run-length ids per leaf tile, `leaf_instance_ids`), cleaned to the connected
   body. The largest two, then up to three small ones.
2. **Frames** (`target_masks`, `choose_frames`): each real camera's 2D mask of the object, the
   object's gaussians rendered where they are the front surface (the segmentation's mask in
   that photo). Up to three frames, the most of the object unoccluded and uncut by the frame
   edge, lower views first among near ties, at least `FRAME_SEPARATION_DEG` apart.
3. **Generate** (`object_models`): the photo crops with their masks as alpha, to the model on
   a GPU; it returns the whole object as gaussians in its own frame.
4. **Register** (`register`): scale, yaw and position from the measured body's extent and top,
   a yaw search, then trimmed ICP with scale: the measured points onto the generated surface,
   and the generated surface above the lowest measured point onto the measured points (what
   lies below that, the underside, has nothing to match). The best of the candidates.
5. **Free space and silhouettes** (`anchor_fill.Carver`, `strict_keep`, `silhouette_keep`):
   anything generated that a real camera saw through is removed, and anything a real camera
   would see outside the object's mask in its photo (the object must not grow).
6. **The unseen side only** (`shell_coverage`, `unseen`): a generated gaussian stays only in
   directions (from the completed object's centre) where the object's well-seen (`fill_quality`
   known) surface is not, and where no well-seen measured gaussian is within
   `UNSEEN_SPACINGS` of the object's spacing (the straw it rests on): a generated side that
   lies a little inside or outside the measured one is not kept for being displaced.
7. **Colour** (`colour_band`): a gain and offset per channel taking the generated colours'
   mean and spread to the scan's, fitted on the band where the kept part begins (covered
   directions next to uncovered ones) against the scan's colour in the same direction.
8. **Layer**: the kept gaussians of every object, one inferred layer (`teacher_fill.
   package_inferred`), shown purple in the viewer's Highlight style like round 2's.

Grading (no new photos): silhouette agreement in every real frame (`silhouettes`: the completed
object against the segmentation's mask; it must not grow past it), free-space violations at a
grading width and a finer one (`free_space`), the round-2 leave-out (`leave_out_scores`: the lowest
cameras held out, their look withheld; the object pipeline run on the kept cameras and scored
on the held-out photos in round 2's region and on the objects' lower edges there), and a sheet
per method (`sheet`).

Everything runs on the CPU with a stand-in generator (`StandInGenerator`) and the CPU renderer
for the tests; `infra/modal/fill_objects.py` runs it with gsplat on an L4 and the models on
their GPUs.
"""

from __future__ import annotations

import io
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

import anchor_fill as af
import fill_quality as fq
import fill_views as fv
import generative_fill as gf
import teacher_fill as tf
from splat_render import Camera, Splats

# --- constants ------------------------------------------------------------------------------------

#: The objects: the instances whose concept (or name) is this, the largest `LARGE`, then up
#: to `SMALL` more with at least `SMALL_MIN` gaussians.
CONCEPT = "pumpkin"
LARGE = 2
SMALL = 3
SMALL_MIN = 50
#: An object's body: voxels this many median spacings across join; parts smaller than
#: `BODY_SHARE` of the largest are left out (straw the segmentation gave the object).
BODY_VOXEL_SPACINGS = 4.0
BODY_SHARE = 0.2
#: Registration uses gaussians at least this opaque.
OPAQUE = 0.3
#: Masks and frame choice at this width.
MASK_WIDTH = 480
#: A gaussian of the object is the front surface where it is no farther than the whole
#: scene's rendered depth by this share.
FRONT_SHARE = 0.02
#: Frames: at most this many, at least this far apart around the object.
FRAMES = 3
FRAME_SEPARATION_DEG = 50.0
#: A frame whose mask touches the photo's edge over more than this share of its border
#: pixels is cut off and not used.
EDGE_SHARE = 0.02
#: The mask sent is opened by a disc this share of the mask's size across (`clean_mask`).
MASK_OPEN_SHARE = 0.03
#: The photo crop sent: the mask's box padded by this share of its size, at most this wide.
CROP_PAD = 0.2
CROP_MAX = 1024
#: A candidate fitting within this factor of the best one's error may be chosen for being
#: more complete underneath (`underside_share`).
FIT_SLACK = 1.3
#: The underside: directions more than this far below the horizontal from the object's centre.
UNDERSIDE_DEG = 45.0
#: Registration: yaw steps, ICP iterations, the share of correspondences kept; an object
#: whose best candidate fits worse than `MAX_ERROR` (trimmed mean distance over its size) is
#: left unfilled.
YAW_STEPS = 36
ICP_ITERATIONS = 30
TRIM = 0.8
REGISTER_SAMPLE = 6000
MAX_ERROR = 0.05
#: A generated gaussian is on the seen side when a known measured gaussian is within this many
#: of the object's median spacings.
UNSEEN_SPACINGS = 3.0
#: The colour band: generated gaussians in covered directions within this many direction
#: bins of a direction holding kept ones (`colour_band`).
BAND_BINS = 2
BAND_MIN = 50
#: The object's well-seen shell, by direction from the completed object's centre: bins this
#: many degrees square, covered with at least this many of its known gaussians.
SHELL_BIN_DEG = 5
SHELL_MIN = 2
#: The silhouette: a generated gaussian a real camera would see (no farther than the scan's
#: surface at its pixel) outside the object's mask there, grown by this many pixels at
#: `MASK_WIDTH`, is removed: the object must not grow past the real silhouette.
SILHOUETTE_GROW_PX = 1
SILHOUETTE_SIGMAS = 1.5
#: Strict free-space test: in front of the measured surface by more than this share of the
#: depth (and the gaussian's own size) where the scan covers the pixel solidly.
STRICT_SHARE = 0.02
STRICT_WIDTH = 480
GRADE_WIDTH = 640
FINE_WIDTH = 960
#: Round 2's carver works at this width.
CARVE_WIDTH = 320
#: The layer's budget (round 1 and 2's): this share of the measured gaussians, at least
#: `MIN_BUDGET`.
BUDGET_SHARE = gf.BUDGET_SHARE
MIN_BUDGET = gf.MIN_BUDGET
#: Round 2's leave-out: the lowest (or highest) share of the cameras held out.
LEAVE_SHARE = 0.15
#: Generated gaussians fainter than this are dropped.
MIN_OPACITY = 0.05
#: Sheet tiles.
TILE = (256, 192)
SHEET_VIEWS = (
    ("low 0", 5.0, 0.0),
    ("low 120", 5.0, 120.0),
    ("low 240", 5.0, 240.0),
    ("45 deg", 45.0, 60.0),
    ("below", -89.0, 0.0),
)
HIGHLIGHT = np.array([0.62, 0.31, 0.95])

#: Each method's layer folder and the filler name its evidence carries.
LAYERS = {
    "sam3d": "objects-sam3d",
    "pixal3d": "objects-pixal3d",
    "trellis2": "objects-trellis2",
    "trellis": "objects-trellis",
    "standin": "objects-standin",
}
RULE = (
    "objects the segmentation names ({concept}) completed by a 3D object-completion model "
    "({model}) from {frames} real photo(s) each with the object's mask, registered to the "
    "measured gaussians (scale, yaw search, trimmed ICP), carved by real sight lines, kept only "
    "where no well-seen measured gaussian is near, colour matched to the measured band above "
    "where the kept part begins"
)


# --- objects -----------------------------------------------------------------------------------------


def leaf_order(tileset: Path) -> list[str]:
    """The leaf tiles' uris in the order `splat_render.load_tileset` concatenates them."""
    document = json.loads(tileset.read_text(encoding="utf-8"))
    leaves: list[str] = []
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("children"):
            stack.extend(tile["children"])
        else:
            leaves.append(tile["content"]["uri"])
    return sorted(leaves)


def leaf_instance_ids(tileset: Path, instances: dict) -> tuple[np.ndarray, dict[str, Any]]:
    """Per gaussian of `load_tileset(tileset)`, the instance id its leaf tile's runs give it
    (0 where the tile is not listed or its runs do not match its count), and a summary."""
    import rebind_instances as ri
    from rig_tiles import tile_positions
    from synthetic_tree import checksum_positions

    tiles = instances.get("tiles") or {}
    parts, matched, missing = [], 0, []
    for uri in leaf_order(tileset):
        at = tile_positions(tileset.parent / uri)
        runs = tiles.get(checksum_positions(at))
        ids = ri.decode_runs(runs) if runs is not None else np.zeros(0, np.int64)
        if ids.size != len(at):
            missing.append(uri)
            ids = np.zeros(len(at), np.int64)
        else:
            matched += 1
        parts.append(ids)
    ids = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    return ids, {"leaves": matched + len(missing), "matched": matched, "missing": missing}


@dataclass
class Target:
    """One object: its instance, the measured gaussians that carry it (cleaned to its body)
    and its size."""

    instance: int
    name: str
    rows: np.ndarray
    centre: np.ndarray
    radius: float
    spacing: float
    size_class: str

    def to_json(self) -> dict[str, Any]:
        return {
            "instance": self.instance,
            "name": self.name,
            "gaussians": int(self.rows.size),
            "centre": np.round(self.centre, 4).tolist(),
            "radius": round(self.radius, 4),
            "spacing": round(self.spacing, 5),
            "size": self.size_class,
        }


def _subtree(instances: Sequence[dict], root: int) -> set[int]:
    return set(gf.subtree_ids(instances, root))


def body_rows(measured: Splats, rows: np.ndarray) -> np.ndarray:
    """`rows` without what is not connected to the object's body (voxel components)."""
    if rows.size < 20:
        return rows
    p = measured.positions[rows]
    spacing = fq.median_spacing(p)
    labels = fq.voxel_components(p, max(spacing * BODY_VOXEL_SPACINGS, 1e-6))
    counts = np.bincount(labels)
    keep = counts >= BODY_SHARE * counts.max()
    return rows[keep[labels]]


def object_targets(
    measured: Splats,
    ids: np.ndarray,
    instances: dict,
    concept: str = CONCEPT,
    large: int = LARGE,
    small: int = SMALL,
    small_min: int = SMALL_MIN,
) -> list[Target]:
    """The variant's objects of `concept`: each root instance named so, with its subtree, the
    largest `large` and then up to `small` more of at least `small_min` gaussians."""
    items = instances.get("instances") or []
    roots = [
        i
        for i in items
        if not i.get("parent")
        and concept in f"{i.get('concept') or ''} {i.get('name') or ''}".lower()
    ]
    found = []
    for item in roots:
        members = np.array(sorted(_subtree(items, int(item["id"]))), np.int64)
        rows = np.flatnonzero(np.isin(ids, members))
        rows = body_rows(measured, rows)
        if rows.size:
            found.append((item, rows))
    found.sort(key=lambda x: -x[1].size)
    out = []
    for k, (item, rows) in enumerate(found):
        if k >= large and (rows.size < small_min or len(out) >= large + small):
            continue
        p = measured.positions[rows]
        lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
        out.append(
            Target(
                int(item["id"]),
                str(item.get("name") or item.get("concept") or concept),
                rows,
                (lo + hi) / 2,
                float(np.linalg.norm(hi - lo) / 2),
                fq.median_spacing(p),
                "large" if k < large else "small",
            )
        )
    return out


# --- masks and frames -------------------------------------------------------------------------------


def draw(renderer: Any, splats: Splats, camera: Camera, rows: np.ndarray | None = None, **kw: Any):
    """`renderer` on `splats` (only `rows` of them when given; gsplat keeps the scene uploaded)."""
    if rows is None:
        return renderer(splats, camera, **kw)
    if getattr(renderer, "name", "") == "gsplat":
        return renderer(splats, camera, rows=np.asarray(rows, np.int64), **kw)
    return renderer(splats.take(np.asarray(rows, np.int64)), camera, **kw)


def front_mask(
    renderer: Any,
    scene: Splats,
    rows: np.ndarray,
    camera: Camera,
    whole_depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Where the gaussians `rows` of `scene` are drawn (alpha 0.5 or more) and where they are
    also the front surface (no farther than the whole scene's depth): (drawn, front)."""
    part = draw(renderer, scene, camera, rows)
    if whole_depth is None:
        whole_depth = renderer(scene, camera).depth
    drawn = part.alpha >= 0.5
    whole = np.where(np.isfinite(whole_depth), whole_depth, np.inf)
    front = drawn & (part.depth <= whole * (1 + FRONT_SHARE) + 1e-6)
    return drawn, front


@dataclass
class ViewMask:
    view: int
    name: str
    front: np.ndarray
    area: int
    unoccluded: float
    edge: float
    elevation: float
    azimuth: float

    def to_json(self) -> dict[str, Any]:
        return {
            "view": self.name,
            "area": self.area,
            "unoccluded": round(self.unoccluded, 3),
            "edge": round(self.edge, 3),
            "elevation": round(self.elevation, 1),
            "azimuth": round(self.azimuth, 1),
        }


def _angles(target: Target, camera: Camera) -> tuple[float, float]:
    d = camera.centre - target.centre
    el = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    az = math.degrees(math.atan2(d[1], d[0])) % 360
    return el, az


def edge_share(mask: np.ndarray) -> float:
    """The share of the image's border pixels the mask covers."""
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return float(border.mean()) if border.size else 0.0


def target_masks(
    renderer: Any,
    scene: Splats,
    target: Target,
    views: Sequence[gf.RealView],
    width: int | None = None,
    depths: dict[str, np.ndarray] | None = None,
) -> list[ViewMask]:
    """Every view's mask of the object (its gaussians where they are the front surface)."""
    width = width or MASK_WIDTH
    out = []
    for k, view in enumerate(views):
        cam = fv.scaled(view.camera, width)
        whole = None if depths is None else depths.get(view.name)
        drawn, front = front_mask(renderer, scene, target.rows, cam, whole)
        el, az = _angles(target, view.camera)
        out.append(
            ViewMask(
                k,
                view.name,
                front,
                int(front.sum()),
                float(front.sum() / max(1, drawn.sum())),
                edge_share(front),
                el,
                az,
            )
        )
    return out


def choose_frames(
    masks: Sequence[ViewMask], limit: int = FRAMES, separation: float = FRAME_SEPARATION_DEG
) -> list[ViewMask]:
    """Up to `limit` frames: the largest unoccluded masks not cut by the frame's edge (area x
    unoccluded share squared, the lower views a little ahead), at least `separation` degrees
    apart in azimuth about the object."""
    ok = [m for m in masks if m.area >= 200 and m.edge <= EDGE_SHARE]
    if not ok:
        ok = [m for m in masks if m.area >= 50]
    if not ok:
        return []
    els = np.array([m.elevation for m in ok])
    lo, hi = float(els.min()), float(els.max())
    score = {
        id(m): m.area * m.unoccluded**2 * (1.0 + 0.25 * (hi - m.elevation) / max(hi - lo, 1e-6))
        for m in ok
    }
    picked: list[ViewMask] = []
    for m in sorted(ok, key=lambda x: -score[id(x)]):
        if all(abs((m.azimuth - p.azimuth + 180) % 360 - 180) >= separation for p in picked):
            picked.append(m)
        if len(picked) >= limit:
            break
    return picked


# --- requests -----------------------------------------------------------------------------------------


def _png(rgba: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


def crop_box(mask: np.ndarray, pad: float = CROP_PAD) -> tuple[int, int, int, int]:
    """The mask's box padded by `pad` of its size, squared, inside the image: (x0, y0, x1, y1)."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    side = min(side, w, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx = int(round(np.clip(cx - side / 2, 0, w - side)))
    by = int(round(np.clip(cy - side / 2, 0, h - side)))
    s = int(round(side))
    return bx, by, bx + s, by + s


def pointmap(depth: np.ndarray, alpha: np.ndarray, camera: Camera) -> np.ndarray:
    """The scan's own points in `camera`'s frame, per pixel (h, w, 3): the rendered depth
    along the view axis back-projected through each pixel centre, in the PyTorch3D camera
    convention SAM 3D Objects takes (x left, y up, z forward: OpenCV's x and y negated); NaN
    where the scan covers less than half the pixel."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    z = np.where(np.isfinite(depth) & (alpha >= 0.5), depth, np.nan)
    x = (u + 0.5 - camera.width / 2) / camera.focal * z
    y = (v + 0.5 - camera.height / 2) / camera.focal * z
    return np.stack([-x, -y, z], axis=-1)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """A model's mask: opened by a disc `MASK_OPEN_SHARE` of the mask's size across (thin
    straw the segmentation gave the object goes, or the model makes it part of the object),
    its largest piece, holes filled."""
    import cv2
    from scipy.ndimage import binary_fill_holes, label

    ys, xs = np.nonzero(mask)
    if ys.size < 50:
        return mask
    size = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
    k = max(3, int(round(MASK_OPEN_SHARE * size)) | 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, disc) > 0
    labels, n = label(opened)
    if n == 0:
        return mask
    biggest = 1 + int(np.argmax(np.bincount(labels.reshape(-1))[1:]))
    return binary_fill_holes(labels == biggest)


def frame_crop(
    view: gf.RealView,
    renderer: Any,
    scene: Splats,
    target: Target,
    width: int | None = None,
    with_pointmap: bool = False,
) -> dict[str, Any] | None:
    """One frame as a model takes it: the photo crop about the object, its mask as alpha
    (RGBA PNG, at most `CROP_MAX` across), the crop's box and camera in the photo, its own
    intrinsics (focal and principal point in crop pixels: a pixel-aligned model's field of
    view), the mask's share of the crop and, with `with_pointmap`, the scan's points there
    (`pointmap`, float16, the crop's size)."""
    photo = view.photo(width=width)
    if photo is None:
        return None
    cam = fv.scaled(view.camera, photo.shape[1], photo.shape[0])
    whole = renderer(scene, cam)
    _, front = front_mask(renderer, scene, target.rows, cam, whole.depth)
    if front.sum() < 50:
        return None
    front = clean_mask(front)
    x0, y0, x1, y1 = crop_box(front)
    rgb = photo[y0:y1, x0:x1]
    alpha = (front[y0:y1, x0:x1] * 255).astype(np.uint8)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    points = pointmap(whole.depth, whole.alpha, cam)[y0:y1, x0:x1] if with_pointmap else None
    scale = 1.0
    if rgba.shape[1] > CROP_MAX:
        import cv2

        scale = CROP_MAX / rgba.shape[1]
        rgba = cv2.resize(rgba, (CROP_MAX, CROP_MAX), interpolation=cv2.INTER_AREA)
        if points is not None:
            points = cv2.resize(
                points.astype(np.float32), (CROP_MAX, CROP_MAX), interpolation=cv2.INTER_NEAREST
            )
    out = {
        "view": view.name,
        "png": _png(rgba),
        "box": [x0, y0, x1, y1],
        "photo": [int(photo.shape[1]), int(photo.shape[0])],
        "camera": cam.to_json(),
        "intrinsics": {
            "focal": cam.focal * scale,
            "cx": (cam.width / 2 - x0) * scale,
            "cy": (cam.height / 2 - y0) * scale,
            "size": int(rgba.shape[1]),
        },
        "maskShare": round(float(front[y0:y1, x0:x1].mean()), 4),
    }
    if points is not None:
        import object_models as om

        out["pointmap"] = om.pack_array(points, "float16")
    return out


# --- generators ---------------------------------------------------------------------------------------


@dataclass
class Generated:
    """A model's object: gaussians in its own frame, the frame's convention, and what it
    reported."""

    key: str
    splats: Splats | None
    frame: str
    info: dict[str, Any] = field(default_factory=dict)


class Generator(Protocol):
    name: str
    frame: str

    def start(self, requests: Sequence[dict[str, Any]]) -> Callable[[], list[Generated]]: ...


def splats_from_arrays(data: dict[str, Any]) -> Splats:
    """Gaussians from a model's reply (`object_models.pack_gaussians`)."""
    import object_models as om

    a = {k: om.unpack_array(data[k]).astype(np.float64) for k in om.GAUSSIAN_FIELDS}
    return Splats(a["positions"], a["rotations"], a["scales"], a["colours"], a["opacities"])


class StandInGenerator:
    """A CPU stand-in for the tests: the object as an ellipsoid of gaussians in a z-up frame
    of unit size, turned by `yaw` degrees, its colour the request's mean masked colour."""

    name = "standin"
    frame = "z-up"

    def __init__(self, yaw: float = 30.0, count: int = 4000, squash: float = 0.6) -> None:
        self.yaw, self.count, self.squash = yaw, count, squash
        self.calls = 0

    def start(self, requests: Sequence[dict[str, Any]]) -> Callable[[], list[Generated]]:
        from PIL import Image

        out = []
        for request in requests:
            self.calls += 1
            colours = []
            for crop in request["frames"]:
                rgba = np.asarray(Image.open(io.BytesIO(crop["png"])).convert("RGBA"))
                m = rgba[..., 3] > 127
                colours.append(rgba[..., :3][m].mean(axis=0) / 255 if m.any() else [0.5] * 3)
            colour = np.mean(colours, axis=0)
            rng = np.random.default_rng(int(request.get("seed", 0)))
            d = rng.standard_normal((self.count, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            p = d * np.array([0.5, 0.5, 0.5 * self.squash])
            a = math.radians(self.yaw)
            rz = np.array(
                [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
            )
            p = p @ rz.T
            n = len(p)
            splats = Splats(
                p,
                np.tile([1.0, 0, 0, 0], (n, 1)),
                np.full((n, 3), 0.02),
                np.tile(colour, (n, 1)),
                np.full(n, 0.9),
            )
            out.append(Generated(request["key"], splats, self.frame, {"standin": True}))
        return lambda: out


Submit = Callable[[str, str, dict], Any]
Wait = Callable[[Any], dict]


class RemoteGenerator:
    """A model on Modal (`infra/modal/fill_objects.py` sets `BACKEND`): every request spawned
    at once, collected in order."""

    def __init__(self, submit: Submit, wait: Wait, cls: str, name: str, frame: str) -> None:
        self.submit, self.wait, self.cls, self.name, self.frame = submit, wait, cls, name, frame
        self.calls: list[dict[str, Any]] = []

    def start(self, requests: Sequence[dict[str, Any]]) -> Callable[[], list[Generated]]:
        handles = [(r["key"], self.submit(self.cls, "generate", r)) for r in requests]

        def collect() -> list[Generated]:
            out = []
            for key, handle in handles:
                try:
                    reply = self.wait(handle)
                except Exception as error:  # noqa: BLE001 - this candidate fails, others go on
                    reply = {"error": repr(error)[:1500]}
                info = {k: v for k, v in reply.items() if k not in ("gaussians",)}
                self.calls.append({"key": key, **info})
                splats = None
                if not reply.get("error") and reply.get("gaussians"):
                    splats = splats_from_arrays(reply["gaussians"])
                out.append(Generated(key, splats, reply.get("frame", self.frame), info))
            return out

        return collect


# --- registration ---------------------------------------------------------------------------------------


@dataclass
class Similarity:
    """`x' = scale * R x + t`."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(points) @ self.rotation.T + self.translation

    def splats(self, s: Splats) -> Splats:
        """The gaussians moved: positions, scales and orientations."""
        q = _matrix_quaternion(self.rotation)
        return Splats(
            self.apply(s.positions),
            _quat_mul(np.broadcast_to(q, s.rotations.shape), s.rotations),
            s.scales * self.scale,
            s.colours.copy(),
            s.opacities.copy(),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "scale": round(float(self.scale), 6),
            "rotation": np.round(self.rotation, 6).tolist(),
            "translation": np.round(self.translation, 6).tolist(),
        }


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton products of (w, x, y, z) quaternions, row by row."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _matrix_quaternion(r: np.ndarray) -> np.ndarray:
    """The unit quaternion (w, x, y, z) of a rotation matrix."""
    m = np.asarray(r, np.float64)
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


def umeyama(src: np.ndarray, dst: np.ndarray) -> Similarity:
    """The similarity taking `src` onto `dst` in the least-squares sense."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    r = u @ s @ vt
    var = (a**2).sum() / len(src)
    scale = float(np.trace(np.diag(d) @ s) / max(var, 1e-12))
    return Similarity(scale, r, mu_d - scale * r @ mu_s)


def _rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0, 0, 1]])


def _sample(p: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(p) <= n:
        return p
    return p[np.random.default_rng(seed).choice(len(p), n, replace=False)]


def _fit_error(
    gen: np.ndarray, measured: np.ndarray, t: Similarity, zcut: float, size: float
) -> tuple[float, dict[str, np.ndarray]]:
    """The trimmed mean of both ways' distances (measured onto the moved generated surface, and
    the generated surface above `zcut` onto the measured), over `size`; and the pairs."""
    from scipy.spatial import cKDTree

    g = t.apply(gen)
    d1, i1 = cKDTree(g).query(measured)
    upper = np.flatnonzero(g[:, 2] > zcut)
    d2, i2 = (
        cKDTree(measured).query(g[upper]) if upper.size else (np.zeros(0), np.zeros(0, np.int64))
    )
    d = np.concatenate([d1, d2])
    cut = np.quantile(d, TRIM) if d.size else 0.0
    err = float(d[d <= cut].mean()) / max(size, 1e-12) if d.size else float("inf")
    pairs = {
        "src": np.concatenate([gen[i1], gen[upper]]),
        "dst": np.concatenate([measured, measured[i2]]),
        "d": d,
    }
    return err, pairs


def register(
    gen_points: np.ndarray,
    measured: np.ndarray,
    *,
    up_known: bool = True,
    inits: Sequence[Similarity] = (),
    yaw_steps: int = YAW_STEPS,
    iterations: int = ICP_ITERATIONS,
) -> tuple[Similarity, dict[str, Any]]:
    """The similarity putting the generated object (`gen_points`, its frame z-up when
    `up_known`) on the measured body (`measured`, world z-up). Starts: `inits` (a model's own
    pose) and, when up is known, a yaw search after matching the horizontal extent, the
    centre and the top; then trimmed ICP with scale. Returns the best and its errors (over the
    body's size)."""
    gen = _sample(np.asarray(gen_points, np.float64), REGISTER_SAMPLE, 1)
    meas = _sample(np.asarray(measured, np.float64), REGISTER_SAMPLE, 2)
    lo_m, hi_m = np.percentile(meas, 2, axis=0), np.percentile(meas, 98, axis=0)
    size = float(np.linalg.norm(hi_m - lo_m))
    zcut = float(np.percentile(meas[:, 2], 5))
    starts: list[Similarity] = list(inits)
    if up_known:
        lo_g, hi_g = np.percentile(gen, 2, axis=0), np.percentile(gen, 98, axis=0)
        span_g = float(np.mean(hi_g[:2] - lo_g[:2]))
        span_m = float(np.mean(hi_m[:2] - lo_m[:2]))
        s0 = span_m / max(span_g, 1e-9)
        mid_g = (lo_g + hi_g) / 2
        mid_m = (lo_m + hi_m) / 2
        for k in range(yaw_steps):
            r = _rz(360.0 * k / yaw_steps)
            t = np.array([mid_m[0], mid_m[1], hi_m[2]]) - s0 * r @ np.array(
                [mid_g[0], mid_g[1], hi_g[2]]
            )
            starts.append(Similarity(s0, r, t))
    if not starts:
        raise ValueError("no starting pose: give inits or a z-up generator")
    scored = sorted(((_fit_error(gen, meas, s, zcut, size)[0], k) for k, s in enumerate(starts)))
    best_err, best = float("inf"), starts[scored[0][1]]
    tried = []
    for err0, k in scored[:3]:
        t = starts[k]
        err = err0
        for _ in range(iterations):
            err, pairs = _fit_error(gen, meas, t, zcut, size)
            keep = pairs["d"] <= np.quantile(pairs["d"], TRIM)
            t_new = umeyama(pairs["src"][keep], pairs["dst"][keep])
            if not np.isfinite(t_new.scale) or t_new.scale <= 0:
                break
            t = t_new
        err, _ = _fit_error(gen, meas, t, zcut, size)
        tried.append({"start": k, "startError": round(err0, 5), "error": round(err, 5)})
        if err < best_err:
            best_err, best = err, t
    # The final errors, each way, as shares of the body's size.
    from scipy.spatial import cKDTree

    g = best.apply(gen)
    d1, _ = cKDTree(g).query(meas)
    info = {
        "error": round(best_err, 5),
        "medianMeasuredToGenerated": round(float(np.median(d1)) / size, 5),
        "within2pc": round(float((d1 <= 0.02 * size).mean()), 4),
        "size": round(size, 4),
        "tried": tried,
        "transform": best.to_json(),
    }
    return best, info


# --- completion -------------------------------------------------------------------------------------------


def strict_keep(
    positions: np.ndarray,
    reach: np.ndarray,
    measured: Splats,
    cameras: Sequence[Camera],
    renderer: Any,
    width: int | None = None,
    depths: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Per point, whether no real camera saw through it: it is not in front of the measured
    surface by more than `STRICT_SHARE` of the depth (and its own reach) where the scan covers
    the pixel solidly (no exemption for weak surfaces). `depths` caches each camera's maps."""
    from scipy.ndimage import minimum_filter

    width = width or STRICT_WIDTH
    keep = np.ones(len(positions), bool)
    for c, full in enumerate(cameras):
        cam = fv.scaled(full, width)
        if depths is not None and c in depths:
            surface, solid = depths[c]
        else:
            frame = renderer(measured, cam)
            surface = fq.nearest_surface(frame.depth)
            solid = minimum_filter((frame.alpha >= 0.5).astype(np.uint8), size=3) > 0
            if depths is not None:
                depths[c] = (surface, solid)
        uv, z = cam.project(positions)
        u = np.floor(uv[:, 0]).astype(np.int64)
        v = np.floor(uv[:, 1]).astype(np.int64)
        inside = (z > 1e-3) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height) & keep
        rows = np.flatnonzero(inside)
        if rows.size == 0:
            continue
        s = surface[v[rows], u[rows]]
        through = (
            solid[v[rows], u[rows]]
            & np.isfinite(s)
            & (z[rows] < s * (1 - STRICT_SHARE) - reach[rows])
        )
        keep[rows[through]] = False
    return keep


def silhouette_keep(
    positions: np.ndarray,
    sigma: np.ndarray,
    masks: Sequence[ViewMask],
    depths: dict[str, np.ndarray],
    views: Sequence[gf.RealView],
    width: int | None = None,
) -> np.ndarray:
    """Per point, whether no real camera would see it outside the object's mask: where a
    camera's pixel is outside the mask (grown by `SILHOUETTE_GROW_PX`), a point no farther than
    the scan's rendered surface there (or where the scan shows nothing) would grow the object
    past its real silhouette, and goes. A point behind the surface (under the straw) stays.
    The test is made at the centre and `SILHOUETTE_SIGMAS` of its largest scale (`sigma`) to
    either side, each sample against the depth at its own pixel, so a kept gaussian's core does
    not spill past the mask (a point just behind the object's edge shows beside it)."""
    width = width or MASK_WIDTH
    keep = np.ones(len(positions), bool)
    for m in masks:
        view = views[m.view]
        cam = fv.scaled(view.camera, width)
        allowed = af._dilate(m.front, SILHOUETTE_GROW_PX)
        depth = depths.get(view.name)
        if depth is None:
            continue
        uv, z = cam.project(positions)
        u = np.floor(uv[:, 0]).astype(np.int64)
        v = np.floor(uv[:, 1]).astype(np.int64)
        inside = (z > 1e-3) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height) & keep
        rows = np.flatnonzero(inside)
        if rows.size == 0:
            continue
        r = SILHOUETTE_SIGMAS * cam.focal * sigma[rows] / np.maximum(z[rows], 1e-6)
        spills = np.zeros(rows.size, bool)
        for du, dv in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            uu = np.clip(np.floor(uv[rows, 0] + du * r).astype(np.int64), 0, cam.width - 1)
            vv = np.clip(np.floor(uv[rows, 1] + dv * r).astype(np.int64), 0, cam.height - 1)
            s = depth[vv, uu]
            s = np.where(np.isfinite(s), s, np.inf)
            spills |= ~allowed[vv, uu] & (z[rows] <= s * (1 + FRONT_SHARE))
        keep[rows[spills]] = False
    return keep


def _direction_bins(d: np.ndarray) -> np.ndarray:
    """Each direction's (elevation, azimuth) bin, `SHELL_BIN_DEG` square."""
    r = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
    el = np.degrees(np.arcsin(np.clip(d[:, 2] / r, -1, 1)))
    az = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360
    n_el, n_az = 180 // SHELL_BIN_DEG, 360 // SHELL_BIN_DEG
    i = np.clip(((el + 90) / SHELL_BIN_DEG).astype(np.int64), 0, n_el - 1)
    j = np.clip((az / SHELL_BIN_DEG).astype(np.int64), 0, n_az - 1)
    return i * n_az + j


def underside_share(points: np.ndarray, centre: np.ndarray) -> float:
    """The share of the underside's direction bins (more than `UNDERSIDE_DEG` below the
    horizontal from `centre`) that hold points: how complete an object is underneath."""
    n_el, n_az = 180 // SHELL_BIN_DEG, 360 // SHELL_BIN_DEG
    rows = int((90 - UNDERSIDE_DEG) // SHELL_BIN_DEG)
    held = np.zeros(n_el * n_az, bool)
    if len(points):
        held[np.unique(_direction_bins(np.asarray(points) - centre))] = True
    return float(held.reshape(n_el, n_az)[:rows].mean())


def shell_coverage(centre: np.ndarray, shell: np.ndarray) -> np.ndarray:
    """Per direction bin from `centre` (`_direction_bins`), whether the object's well-seen
    measured surface (`shell`) is there: `SHELL_MIN` points or more, closed over single-bin
    gaps (azimuth wraps)."""
    from scipy.ndimage import binary_closing

    n_el, n_az = 180 // SHELL_BIN_DEG, 360 // SHELL_BIN_DEG
    counts = np.bincount(_direction_bins(shell - centre), minlength=n_el * n_az)
    grid = (counts >= SHELL_MIN).reshape(n_el, n_az)
    padded = np.concatenate([grid[:, -2:], grid, grid[:, :2]], axis=1)
    closed = binary_closing(padded, structure=np.ones((3, 3), bool))[:, 2:-2]
    return (closed | grid).reshape(-1)


def shell_colours(centre: np.ndarray, shell: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """Per direction bin, the mean colour of the shell there (NaN where none)."""
    bins = _direction_bins(shell - centre)
    n = (180 // SHELL_BIN_DEG) * (360 // SHELL_BIN_DEG)
    count = np.bincount(bins, minlength=n).astype(np.float64)
    out = np.full((n, 3), np.nan)
    for ch in range(3):
        total = np.bincount(bins, weights=colours[:, ch], minlength=n)
        out[:, ch] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


def unseen(
    positions: np.ndarray, known: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per point, whether no known measured gaussian (`known`, positions) is within `radius`,
    and the distance to the nearest."""
    from scipy.spatial import cKDTree

    if len(known) == 0:
        return np.ones(len(positions), bool), np.full(len(positions), np.inf)
    d, _ = cKDTree(known).query(positions)
    return d > radius, d


def colour_transfer(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per channel a gain and offset taking `src`'s mean and spread to `dst`'s (the gain
    within 0.67-1.5): a regression would shrink a texture that correlates little with the
    scan's to its mean."""
    sx, sy = src.std(axis=0), dst.std(axis=0)
    gain = np.clip(np.where(sx > 1e-6, sy / np.maximum(sx, 1e-6), 1.0), 0.67, 1.5)
    return gain, dst.mean(axis=0) - gain * src.mean(axis=0)


def colour_band(
    world: Splats,
    kept: np.ndarray,
    covered: np.ndarray,
    bins: np.ndarray,
    centre: np.ndarray,
    measured: Splats,
    shell_rows: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The kept gaussians' colours matched to the scan (`colour_transfer`), fitted on the band
    where the kept part begins: generated gaussians in covered directions within
    `BAND_BINS` bins of an uncovered direction holding kept ones, against the mean colour of
    the object's well-seen surface in their own direction."""
    from scipy.ndimage import binary_dilation

    out = world.colours.copy()
    info: dict[str, Any] = {"applied": False}
    if not kept.any() or shell_rows.size == 0:
        return out, info
    n_el, n_az = 180 // SHELL_BIN_DEG, 360 // SHELL_BIN_DEG
    holds = np.zeros(n_el * n_az, bool)
    holds[np.unique(bins[kept])] = True
    grid = holds.reshape(n_el, n_az)
    padded = np.concatenate([grid[:, -BAND_BINS:], grid, grid[:, :BAND_BINS]], axis=1)
    near = binary_dilation(padded, iterations=BAND_BINS)[:, BAND_BINS:-BAND_BINS].reshape(-1)
    target_colour = shell_colours(
        centre, measured.positions[shell_rows], measured.colours[shell_rows]
    )
    band = covered & near[bins] & np.isfinite(target_colour[bins, 0])
    if band.sum() < BAND_MIN:
        info["pairs"] = int(band.sum())
        return out, info
    src = world.colours[band]
    dst = target_colour[bins[band]]
    gain, offset = colour_transfer(src, dst)
    out[kept] = np.clip(world.colours[kept] * gain + offset, 0, 1)
    info = {
        "applied": True,
        "on": "band",
        "pairs": int(band.sum()),
        "gain": np.round(gain, 4).tolist(),
        "offset": np.round(offset, 4).tolist(),
        "bandMean": np.round(dst.mean(axis=0), 4).tolist(),
        "generatedMean": np.round(src.mean(axis=0), 4).tolist(),
        "meanAbsErrorBefore": round(float(np.abs(src - dst).mean()), 4),
        "meanAbsErrorAfter": round(
            float(np.abs(np.clip(src * gain + offset, 0, 1) - dst).mean()), 4
        ),
    }
    return out, info


def thin(splats: Splats, limit: int, voxel: float) -> Splats:
    """At most `limit` gaussians: the most opaque per voxel, the voxel grown until it fits."""
    if len(splats) <= limit:
        return splats
    v = max(voxel, 1e-6)
    for _ in range(12):
        keys = np.floor(splats.positions / v).astype(np.int64)
        order = np.argsort(-splats.opacities, kind="stable")
        _, first = np.unique(keys[order], axis=0, return_index=True)
        rows = np.sort(order[first])
        if rows.size <= limit:
            return splats.take(rows)
        v *= 1.25
    return splats.take(np.argsort(-splats.opacities)[:limit])


@dataclass
class Part:
    """One object's completion: the kept gaussians (world frame) and how they were chosen."""

    target: Target
    splats: Splats | None
    confidence: np.ndarray
    info: dict[str, Any]


def complete_target(
    target: Target,
    candidates: Sequence[Generated],
    setup: af.Setup,
    carver: af.Carver,
    cameras: Sequence[Camera],
    renderer: Any,
    strict_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    masks: Sequence[ViewMask] = (),
    depths: dict[str, np.ndarray] | None = None,
    up_known: bool = True,
    grade_cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    debug: Path | None = None,
    log: Callable[[str], None] = print,
) -> Part:
    """The best registered candidate, carved, kept inside the real silhouettes and to the
    unseen side, colour matched."""
    measured = setup.measured
    shown = np.zeros(len(measured), bool)
    shown[setup.shown] = True
    rows = target.rows[shown[target.rows]]
    body = rows[measured.opacities[rows] >= OPAQUE]
    if body.size < 30:
        body = rows
    info: dict[str, Any] = {"target": target.to_json(), "candidates": []}
    fitted: list[tuple[float, float, Generated, Similarity, dict]] = []
    for cand in candidates:
        entry: dict[str, Any] = {"key": cand.key, **{k: v for k, v in cand.info.items()}}
        if cand.splats is None or len(cand.splats) < 50:
            entry["skipped"] = "no object"
            info["candidates"].append(entry)
            continue
        g = cand.splats.take(np.flatnonzero(cand.splats.opacities >= MIN_OPACITY))
        solid = g.positions[g.opacities >= OPAQUE]
        if len(solid) < 50:
            solid = g.positions
        try:
            t, reg = register(solid, measured.positions[body], up_known=up_known)
        except Exception as error:  # noqa: BLE001 - this candidate fails
            entry["skipped"] = f"registration failed: {error!r}"[:300]
            info["candidates"].append(entry)
            continue
        entry["registration"] = reg
        moved = t.apply(g.positions)
        lo, hi = np.percentile(moved, 1, axis=0), np.percentile(moved, 99, axis=0)
        entry["underside"] = round(underside_share(moved, (lo + hi) / 2), 4)
        info["candidates"].append(entry)
        fitted.append((reg["error"], entry["underside"], cand, t, reg))
    if not fitted:
        info["skipped"] = "no candidate registered"
        return Part(target, None, np.zeros(0), info)
    # Among the candidates that fit nearly as well as the best, the most complete underneath.
    floor = min(f[0] for f in fitted)
    good = [f for f in fitted if f[0] <= max(FIT_SLACK * floor, floor + 0.003)]
    err, _, cand, t, reg = max(good, key=lambda f: (f[1], -f[0]))
    info["chosen"] = cand.key
    if err > MAX_ERROR:
        info["skipped"] = (
            f"the best candidate fits at {err:.4f} of the object's size (> {MAX_ERROR})"
        )
        log(f"object {target.instance}: {info['skipped']}")
        return Part(target, None, np.zeros(0), info)
    assert cand.splats is not None
    g = cand.splats.take(np.flatnonzero(cand.splats.opacities >= MIN_OPACITY))
    world = t.splats(g)
    info["generated"] = len(world)
    # Free space: round 2's carver, then the strict test (no weak-surface exemption) at the
    # carve's width and at the grading's, so the grade finds nothing the carve let through.
    carved = ~carver.keep(world.positions)
    reach = 2.0 * world.scales.max(axis=1)
    strict = ~strict_keep(world.positions, reach, measured, cameras, renderer, depths=strict_cache)
    strict |= ~strict_keep(
        world.positions, reach, measured, cameras, renderer, GRADE_WIDTH, grade_cache
    )
    keep = ~carved & ~strict
    info["carved"] = int((~keep).sum())
    outside = np.zeros(len(world), bool)
    if masks and depths is not None:
        outside = ~silhouette_keep(
            world.positions, world.scales.max(axis=1), masks, depths, setup.views
        )
        info["outsideSilhouette"] = int((keep & outside).sum())
        keep &= ~outside
    # The unseen side: in no direction (from the completed object's centre) the object's
    # well-seen surface covers, and no well-seen measured gaussian near (the straw it rests
    # on included).
    classes = np.full(len(measured), fq.UNKNOWN, np.int8)
    classes[setup.shown] = setup.shown_classes()
    known = measured.positions[classes == fq.KNOWN]
    radius = UNSEEN_SPACINGS * target.spacing
    away, _ = unseen(world.positions, known, radius)
    known_body = body[classes[body] == fq.KNOWN]
    lo, hi = np.percentile(world.positions, 1, axis=0), np.percentile(world.positions, 99, axis=0)
    centre = (lo + hi) / 2
    covered_bins = shell_coverage(centre, measured.positions[known_body])
    bins = _direction_bins(world.positions - centre)
    covered = covered_bins[bins]
    kept = keep & away & ~covered
    seen_side = keep & ~kept
    info["shell"] = {
        "centre": np.round(centre, 4).tolist(),
        "coveredBins": int(covered_bins.sum()),
        "covered": int((keep & covered).sum()),
        "nearKnown": int((keep & ~away).sum()),
    }
    info["seenSide"] = int(seen_side.sum())
    info["kept"] = int(kept.sum())
    info["underside"] = {
        "generated": round(underside_share(world.positions, centre), 4),
        "kept": round(underside_share(world.positions[kept], centre), 4),
        "measured": round(underside_share(measured.positions[known_body], centre), 4),
    }
    # Colour: matched on the band where the kept part begins: the covered directions next to
    # an uncovered one, against the scan's colour in the same direction.
    colours, cinfo = colour_band(world, kept, covered, bins, centre, measured, known_body)
    info["colour"] = cinfo
    if debug is not None:
        debug.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            debug,
            positions=world.positions.astype(np.float32),
            colours=np.round(np.clip(world.colours, 0, 1) * 255).astype(np.uint8),
            carved=carved,
            strict=strict,
            outside=outside,
            near_known=~away,
            covered=covered,
            kept=kept,
            centre=centre,
            body=body,
            body_classes=classes[body],
        )
    world = Splats(world.positions, world.rotations, world.scales, colours, world.opacities)
    out = world.take(np.flatnonzero(kept))
    # Confidence: the registration's fit, fading with distance from the seen band.
    fit = float(np.clip(1.0 - 10.0 * err, 0.1, 1.0))
    from scipy.spatial import cKDTree

    if seen_side.any() and kept.any():
        d_band, _ = cKDTree(world.positions[seen_side]).query(out.positions)
        conf = fit * np.exp(-d_band / max(target.radius, 1e-9))
    else:
        conf = np.full(len(out), fit * 0.5)
    info["fit"] = round(fit, 4)
    log(
        f"object {target.instance} ({target.size_class}): {cand.key}, error {err:.4f}, "
        f"{len(world)} generated, {info['carved']} carved, {info['kept']} kept"
    )
    return Part(target, out if len(out) else None, conf, info)


# --- grading --------------------------------------------------------------------------------------------


def silhouettes(
    renderer: Any,
    measured: Splats,
    parts: Sequence[Part],
    layer: Splats | None,
    views: Sequence[gf.RealView],
    width: int | None = None,
) -> dict[str, Any]:
    """Per object, over every real frame: the completed object's mask (its gaussians and its
    part of the layer, drawn and in front of everything else: the rest of the scan and the
    other objects' fills) against the segmentation's (its gaussians, drawn and in front of the
    rest of the scan): IoU, and growth (the completed mask's pixels outside the real one, over
    the real one's), also past a one-pixel edge."""
    if layer is None or not len(layer):
        return {}
    width = width or MASK_WIDTH
    scene = Splats.concat([measured, layer])
    n = len(measured)
    offsets, start = {}, n
    for part in parts:
        size = 0 if part.splats is None else len(part.splats)
        offsets[part.target.instance] = np.arange(start, start + size)
        start += size
    out: dict[str, Any] = {}
    stats: dict[int, list[tuple[float, float, int, float]]] = {p.target.instance: [] for p in parts}
    everything = np.arange(len(scene))
    for view in views:
        cam = fv.scaled(view.camera, width)
        for part in parts:
            rows = part.target.rows
            mine = np.concatenate([rows, offsets[part.target.instance]])
            rest_m = np.setdiff1d(np.arange(n), rows, assume_unique=True)
            rest_c = np.setdiff1d(everything, mine, assume_unique=True)
            _, real = front_mask(
                renderer, measured, rows, cam, draw(renderer, measured, cam, rest_m).depth
            )
            if real.sum() < 20:
                continue
            _, done = front_mask(
                renderer, scene, mine, cam, draw(renderer, scene, cam, rest_c).depth
            )
            inter = float((real & done).sum())
            union = float((real | done).sum())
            grow = float((done & ~real).sum()) / float(real.sum())
            past = float((done & ~af._dilate(real, 1)).sum()) / float(real.sum())
            stats[part.target.instance].append(
                (inter / max(union, 1.0), grow, int(real.sum()), past)
            )
    for inst, rows in stats.items():
        if not rows:
            continue
        a = np.asarray(rows)
        out[str(inst)] = {
            "frames": len(rows),
            "meanIoU": round(float(a[:, 0].mean()), 4),
            "minIoU": round(float(a[:, 0].min()), 4),
            "meanGrowth": round(float(a[:, 1].mean()), 5),
            "maxGrowth": round(float(a[:, 1].max()), 5),
            "framesGrowingOverHalfPercent": int((a[:, 1] > 0.005).sum()),
            "meanGrowthPastEdge": round(float(a[:, 3].mean()), 5),
            "maxGrowthPastEdge": round(float(a[:, 3].max()), 5),
        }
    return out


def free_space(
    layer: Splats | None,
    measured: Splats,
    cameras: Sequence[Camera],
    renderer: Any,
    width: int | None = None,
) -> dict[str, Any]:
    """Generated gaussians a real camera saw through (the strict test, no weak-surface
    exemption), at the grading width (`GRADE_WIDTH`, which the carve also tests) and, as a
    check the carve never made, at `FINE_WIDTH`."""
    if layer is None or not len(layer):
        return {"gaussians": 0, "violations": 0, "violationsFine": 0}
    width = width or GRADE_WIDTH
    reach = 2.0 * layer.scales.max(axis=1)
    keep = strict_keep(layer.positions, reach, measured, cameras, renderer, width=width)
    fine = strict_keep(layer.positions, reach, measured, cameras, renderer, width=FINE_WIDTH)
    return {
        "gaussians": len(layer),
        "violations": int((~keep).sum()),
        "width": width,
        "violationsFine": int((~fine).sum()),
        "fineWidth": FINE_WIDTH,
        "cameras": len(cameras),
    }


def leave_out_scores(
    setup: af.Setup,
    renderer: Any,
    perceptual: Any,
    layers: dict[str, Splats | None],
    targets: Sequence[Target],
    out: Path,
    max_views: int = 6,
    width: int = 480,
) -> dict[str, Any]:
    """Round 2's leave-out score (`anchor_fill.score_held_out`) with the object layers: per
    held-out photo (the six with the most withheld pixels), in round 2's region (where a
    withheld gaussian is the front surface) and in its part on the objects' lower edges (the
    region within the objects' masks there): the scan as the kept cameras know it (before) and
    with each layer. A side by side is saved."""
    from PIL import Image

    scene = setup.scene()
    shown = setup.shown
    c = setup.shown_classes()
    visible = ((c != fq.UNKNOWN) & ~setup.withheld[shown]).astype(np.float64)
    withheld = setup.measured.take(np.flatnonzero(setup.withheld & ~setup.dropped))
    index_of = np.full(len(setup.measured), -1, np.int64)
    index_of[shown] = np.arange(len(shown))
    object_rows = np.concatenate([index_of[t.rows] for t in targets]) if targets else np.zeros(0)
    object_rows = object_rows[object_rows >= 0].astype(np.int64)
    candidates = []
    for view in setup.held_out:
        photo = view.photo(width=width)
        if photo is None:
            continue
        cam = fv.scaled(view.camera, photo.shape[1], photo.shape[0])
        region = np.zeros(photo.shape[:2], bool)
        if len(withheld):
            drawn = renderer(withheld, cam)
            front = renderer(scene, cam).depth
            ahead = np.isfinite(drawn.depth) & (
                drawn.depth <= np.where(np.isfinite(front), front, np.inf) * 1.03 + 1e-6
            )
            region = af._close(tf._covered(drawn.alpha) & ahead, 3)
        objects = np.zeros_like(region)
        if object_rows.size:
            objects = af._dilate(draw(renderer, scene, cam, object_rows).alpha >= 0.5, 4)
        candidates.append((int(region.sum()), view, photo, cam, region, region & objects))
    candidates.sort(key=lambda x: -x[0])
    rows, numbers = [], []
    for area, view, photo, cam, region, lower in candidates[:max_views]:
        if area < 50:
            continue
        before = tf.to_u8(renderer(scene, cam, opacity_scale=visible).rgb)
        row = [
            gf.label_image(photo, f"real, held out: {view.name}"),
            gf.label_image(before, "before"),
        ]
        score: dict[str, Any] = {
            "view": view.name,
            "regionPx": area,
            "objectEdgePx": int(lower.sum()),
            "before": _metrics(perceptual, before, photo, region, lower),
        }
        for name, layer in layers.items():
            if layer is None or not len(layer):
                row.append(gf.label_image(np.zeros_like(photo), f"{name}: nothing"))
                continue
            weights = np.concatenate([visible, np.ones(len(layer))])
            after = tf.to_u8(
                renderer(Splats.concat([scene, layer]), cam, opacity_scale=weights).rgb
            )
            row.append(gf.label_image(after, name))
            score[name] = _metrics(perceptual, after, photo, region, lower)
        row[0] = row[0].copy()
        row[0][af._dilate(region, 1) & ~region] = (255, 0, 255)
        row[0][af._dilate(lower, 1) & ~lower] = (0, 255, 255)
        rows.append(row)
        numbers.append(score)
    if rows:
        Image.fromarray(gf.grid(rows)).save(out / f"held-out-{setup.name}.png")
    means: dict[str, Any] = {}
    for key in ["before", *layers]:
        vals = [n[key] for n in numbers if key in n]
        if not vals:
            continue
        means[key] = {
            where: {
                m: af._mean([v[where].get(m) for v in vals]) for m in ("psnr", "lpips", "dreamsim")
            }
            for where in ("region", "objectEdges")
        }
    return {"views": numbers, "mean": means}


def _metrics(
    perceptual: Any, image: np.ndarray, photo: np.ndarray, region: np.ndarray, lower: np.ndarray
) -> dict[str, Any]:
    out = {}
    for where, mask in (("region", region), ("objectEdges", lower)):
        if mask.sum() < 20:
            out[where] = {"psnr": None, "lpips": None, "dreamsim": None}
            continue
        out[where] = {
            "psnr": round(tf.psnr(image, photo, mask), 3),
            "lpips": af._round(perceptual.lpips(image, photo, mask)),
            "dreamsim": af._round(perceptual.dreamsim(image, photo, mask)),
        }
    return out


# --- sheets --------------------------------------------------------------------------------------------


def _view_camera(target: Target, elevation: float, azimuth: float) -> Camera:
    el, az = math.radians(elevation), math.radians(azimuth)
    d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    fov = 40.0
    dist = 1.25 * target.radius / math.tan(math.radians(fov) / 2)
    w, h = TILE
    return Camera.look_at(target.centre + dist * d, target.centre, fov_deg=fov, width=w, height=h)


def sheet(
    renderer: Any,
    measured: Splats,
    parts: Sequence[Part],
    title: str,
    path: Path,
    around: np.ndarray | None = None,
) -> None:
    """Per object, from low side views, a 45-degree view and straight below (looking up):
    the measured scan, the completed scan, the measured object alone, the completed object
    alone with the inferred part highlighted (purple, as the viewer's Highlight shows it), and
    the inferred part alone in its own colours. The views' azimuths are counted from the
    object's outward direction from `around` (the capture's focus): the first looks at the
    side facing away from the capture, the one its cameras saw least."""
    from PIL import Image

    rows = []
    for part in parts:
        t = part.target
        layer = part.splats
        n_layer = 0 if layer is None else len(layer)
        obj = measured.take(t.rows)
        scene_done = measured if layer is None else Splats.concat([measured, layer])
        obj_done = obj if layer is None else Splats.concat([obj, layer])
        lit = None
        if layer is not None:
            lit = Splats(
                layer.positions,
                layer.rotations,
                layer.scales,
                np.tile(HIGHLIGHT, (n_layer, 1)),
                layer.opacities,
            )
        obj_lit = obj if lit is None else Splats.concat([obj, lit])
        for label, el, az in SHEET_VIEWS:
            out_az = 0.0
            if around is not None:
                d = t.centre - np.asarray(around, np.float64)
                out_az = math.degrees(math.atan2(d[1], d[0]))
            cam = _view_camera(t, el, (out_az + az) % 360)
            tiles = []
            for name, splats in (
                ("scan", measured),
                ("scan+fill", scene_done),
                ("object", obj),
                ("object+fill", obj_lit),
                ("fill alone", layer),
            ):
                if splats is None or not len(splats):
                    img = np.zeros((cam.height, cam.width, 3), np.uint8)
                else:
                    img = tf.to_u8(renderer(splats, cam).rgb)
                tiles.append(gf.label_image(img, f"#{t.instance} {label}: {name}"))
            rows.append(tiles)
        del obj_done
    if not rows:
        return
    image = gf.grid(rows)
    head = np.zeros((28, image.shape[1], 3), np.uint8)
    head = gf.label_image(head, title)
    Image.fromarray(np.concatenate([head, image], axis=0)).save(path)


# --- the run ---------------------------------------------------------------------------------------------


def make_setups(
    name: str,
    caption: str,
    measured: Splats,
    views: Sequence[gf.RealView],
    renderer: Any,
    *,
    leave_out: str = "low",
    share: float = 0.15,
    width: int = 320,
    log: Callable[[str], None] = print,
) -> tuple[af.Setup, af.Setup | None, dict[str, Any]]:
    """Round 2's setups (`anchor_fill.make_setup`) from one quality pass: every camera, and
    the leave-out (`leave_out` cameras held out, their look withheld)."""
    views = list(views)
    focus = fq.capture_focus([v.camera for v in views])
    sharp = fq.photo_sharpness([v.photo(width=256) for v in views])
    t = time.time()
    everyone = fq.measure_quality(
        measured, [v.camera for v in views], renderer, focus, width=width, sharpness=sharp
    )
    log(
        f"{name}: quality of {len(measured)} gaussians over {len(views)} cameras in {time.time() - t:.0f}s"
    )
    none = np.zeros(len(measured), bool)
    clusters = fq.Clusters(np.full(len(measured), -1, np.int64), [])
    full = af.Setup(
        name, caption, measured, views, [], everyone, everyone.classes.copy(), none, none,
        focus, clusters, "none",
    )  # fmt: skip
    info: dict[str, Any] = {
        "focus": focus.to_json(),
        "cameras": len(views),
        "quality": everyone.summary(),
    }
    leave = None
    if leave_out != "none":
        kept_idx, held_idx = af.split_leave_out(views, focus, leave_out, share)
        kept = everyone.subset(kept_idx)
        withheld, dropped = fq.withheld_by_holdout(kept, everyone, everyone.subset(held_idx))
        classes = kept.classes.copy()
        classes[withheld] = fq.UNKNOWN
        leave = af.Setup(
            name, caption, measured, [views[k] for k in kept_idx], [views[k] for k in held_idx],
            kept, classes, withheld, dropped, focus, clusters, leave_out,
        )  # fmt: skip
        info["leaveOut"] = {
            "kept": len(kept_idx),
            "heldOut": [views[k].name for k in held_idx],
            "heldOutElevations": [
                round(focus.elevation(views[k].camera.centre), 1) for k in held_idx
            ],
            "withheld": int(withheld.sum()),
            "dropped": int(dropped.sum()),
            "quality": kept.summary(),
        }
    return full, leave, info


@dataclass
class ArmResult:
    setup: af.Setup
    parts: list[Part]
    layer: Splats | None
    confidence: np.ndarray
    frames: dict[int, list[str]]


@dataclass
class Plan:
    """One setup's generation requests, each object's chosen frames and masks, and the scan's
    depth in every camera (at `MASK_WIDTH`)."""

    requests: list[dict[str, Any]]
    chosen: dict[int, list[ViewMask]]
    masks: dict[int, list[ViewMask]]
    depths: dict[str, np.ndarray]


def requests_for(
    measured: Splats,
    views: Sequence[gf.RealView],
    targets: Sequence[Target],
    renderer: Any,
    seeds: Sequence[int],
    per_frame: bool,
    log: Callable[[str], None] = print,
) -> tuple[Plan, dict[str, Any]]:
    """The generation requests of one setup: per object its chosen frames (from the setup's
    cameras, `views`), one request per seed with all of them (`per_frame` False: a
    multi-image model) or one per frame and seed."""
    scene = measured
    depths = {}
    for view in views:
        depths[view.name] = renderer(scene, fv.scaled(view.camera, MASK_WIDTH)).depth
    requests, chosen, info, all_masks = [], {}, {}, {}
    for target in targets:
        masks = target_masks(renderer, scene, target, views, depths=depths)
        all_masks[target.instance] = masks
        frames = choose_frames(masks)
        chosen[target.instance] = frames
        info[str(target.instance)] = {
            "frames": [f.to_json() for f in frames],
            "masks": [m.to_json() for m in masks],
        }
        crops = []
        by_name = {v.name: v for v in views}
        for f in frames:
            crop = frame_crop(by_name[f.name], renderer, scene, target)
            if crop is not None:
                crops.append(crop)
        if not crops:
            log(f"object {target.instance}: no usable frame")
            continue
        groups = [[c] for c in crops]
        if not per_frame and len(crops) > 1:
            groups = [crops, *groups]
        for group in groups:
            for seed in seeds:
                key = f"o{target.instance}-{'+'.join(c['view'] for c in group)}-s{seed}"
                requests.append(
                    {"key": key, "target": target.instance, "frames": group, "seed": int(seed)}
                )
    return Plan(requests, chosen, all_masks, depths), info


def run_arm(
    setup: af.Setup,
    targets: Sequence[Target],
    generated: dict[str, Generated],
    plan: Plan,
    renderer: Any,
    *,
    up_known: bool,
    debug: Path | None = None,
    log: Callable[[str], None] = print,
) -> ArmResult:
    """Every object of one setup completed from its candidates."""
    carver = af.Carver(
        setup.measured,
        [fv.scaled(v.camera, CARVE_WIDTH) for v in setup.views],
        renderer,
        2 * setup.focus.radius,
        af.soft_gaussians(setup),
    )
    cameras = [v.camera for v in setup.views]
    strict_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    grade_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    parts = []
    for target in targets:
        keys = [r["key"] for r in plan.requests if r["target"] == target.instance]
        cands = [generated[k] for k in keys if k in generated]
        parts.append(
            complete_target(
                target,
                cands,
                setup,
                carver,
                cameras,
                renderer,
                strict_cache,
                masks=plan.masks.get(target.instance, []),
                depths=plan.depths,
                up_known=up_known,
                grade_cache=grade_cache,
                debug=None if debug is None else debug / f"{setup.leave_out}-{target.instance}.npz",
                log=log,
            )  # fmt: skip
        )
    pieces = [p.splats for p in parts if p.splats is not None and len(p.splats)]
    confs = [p.confidence for p in parts if p.splats is not None and len(p.splats)]
    layer = Splats.concat(pieces) if pieces else None
    conf = np.concatenate(confs) if confs else np.zeros(0)
    budget = max(MIN_BUDGET, int(BUDGET_SHARE * len(setup.measured)))
    if layer is not None and len(layer) > budget:
        # Thinned per object in proportion, so a large one does not crowd out the rest.
        share = budget / len(layer)
        new_parts = []
        for p in parts:
            if p.splats is None or not len(p.splats):
                new_parts.append(p)
                continue
            limit = max(1, int(share * len(p.splats)))
            thinned = thin(p.splats, limit, p.target.spacing)
            keep_conf = (
                p.confidence[: len(thinned)]
                if len(thinned) == len(p.splats)
                else np.full(len(thinned), float(p.confidence.mean()) if p.confidence.size else 0.5)
            )
            p.info["thinnedTo"] = len(thinned)
            new_parts.append(Part(p.target, thinned, keep_conf, p.info))
        parts = new_parts
        pieces = [p.splats for p in parts if p.splats is not None and len(p.splats)]
        layer = Splats.concat(pieces) if pieces else None
        conf = np.concatenate(
            [p.confidence for p in parts if p.splats is not None and len(p.splats)]
        )
    return ArmResult(setup, parts, layer, conf, {})


def run(
    tileset: Path,
    instances: dict,
    measured: Splats,
    views: Sequence[gf.RealView],
    generator: Generator,
    renderer: Any,
    perceptual: Any,
    out: Path,
    *,
    scan: str = "scan",
    caption: str = "",
    concept: str = CONCEPT,
    leave_out: str = "low",
    seeds: Sequence[int] = (1, 2),
    per_frame: bool = False,
    up_known: bool = True,
    quality_width: int = 320,
    extra_layers: dict[str, Splats] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The whole round for one generator: setups, objects, frames, generation, completion,
    the layer (`out/<layer>/inferred`), grades, sheets and `out/report.json`."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "renders").mkdir(exist_ok=True)
    started = time.time()
    report: dict[str, Any] = {"scan": scan, "method": generator.name, "concept": concept}
    timings: dict[str, float] = {}
    ids, report["instances"] = leaf_instance_ids(tileset, instances)
    targets = object_targets(measured, ids, instances, concept)
    report["targets"] = [t.to_json() for t in targets]
    log(f"{scan}: {len(targets)} objects: {[t.to_json() for t in targets]}")
    if not targets:
        report["skipped"] = f"no {concept} in the segmentation"
        gf.write_json(out / "report.json", report)
        return report
    # The frames of every setup (all cameras; the leave-out's kept ones), and their requests,
    # sent at once (the same frames and seed are asked once); the quality pass runs while the
    # model works.
    t0 = time.time()
    views = list(views)
    camera_sets = {"full": views}
    if leave_out != "none":
        focus = fq.capture_focus([v.camera for v in views])
        kept_idx, _ = af.split_leave_out(views, focus, leave_out, LEAVE_SHARE)
        camera_sets["leaveout"] = [views[k] for k in kept_idx]
    plans, all_requests = {}, {}
    for name, cams in camera_sets.items():
        plan, info = requests_for(measured, cams, targets, renderer, seeds, per_frame, log)
        plans[name] = plan
        report.setdefault("frames", {})[name] = info
        for r in plan.requests:
            all_requests.setdefault(r["key"], r)
    timings["framesS"] = round(time.time() - t0, 1)
    log(f"{len(all_requests)} generation requests")
    t0 = time.time()
    collect = generator.start(list(all_requests.values()))
    full, leave, report["setup"] = make_setups(
        scan, caption, measured, views, renderer, leave_out=leave_out, share=LEAVE_SHARE,
        width=quality_width, log=log,
    )  # fmt: skip
    timings["setupS"] = round(time.time() - t0, 1)
    setups = [("full", full)] + ([("leaveout", leave)] if leave is not None else [])
    for name, setup in setups:
        assert [v.name for v in setup.views] == [v.name for v in camera_sets[name]]
    t0 = time.time()
    generated = {g.key: g for g in collect()}
    timings["generateWaitS"] = round(time.time() - t0, 1)
    report["generated"] = {
        k: {
            "ok": g.splats is not None,
            "gaussians": 0 if g.splats is None else len(g.splats),
            **g.info,
        }
        for k, g in generated.items()
    }
    layer_name = LAYERS.get(generator.name, f"objects-{generator.name}")
    results: dict[str, ArmResult] = {}
    for name, setup in setups:
        t0 = time.time()
        arm = run_arm(
            setup, targets, generated, plans[name], renderer, up_known=up_known,
            debug=out / "debug", log=log,
        )  # fmt: skip
        results[name] = arm
        timings[f"{name}CompleteS"] = round(time.time() - t0, 1)
        report.setdefault("parts", {})[name] = [p.info for p in arm.parts]
    arm = results["full"]
    # The layer to publish (every camera).
    if arm.layer is not None and len(arm.layer):
        layer_dir = out / layer_name / "inferred"
        frames_used = sorted({c["view"] for r in plans["full"].requests for c in r["frames"]})
        by_name = {v.name: v for v in full.views}
        cams = [by_name[n].camera for n in frames_used if n in by_name] or [
            v.camera for v in full.views
        ]
        report["evidence"] = tf.package_inferred(
            arm.layer,
            arm.confidence,
            cams,
            tileset,
            layer_dir,
            generator.name,
            rule=RULE.format(concept=concept, model=generator.name, frames=f"1-{FRAMES}"),
            extra={
                "provenance": "inferred-generated",
                "method": generator.name,
                "objects": [p.target.instance for p in arm.parts if p.splats is not None],
                "segmentation": instances.get("variant", {}).get("name")
                if isinstance(instances.get("variant"), dict)
                else None,
            },
        )
        report["layer"] = str(layer_dir)
    # Grades.
    t0 = time.time()
    report["silhouette"] = silhouettes(renderer, measured, arm.parts, arm.layer, full.views)
    report["freeSpace"] = free_space(arm.layer, measured, [v.camera for v in full.views], renderer)
    if leave is not None:
        lo = results["leaveout"]
        report["freeSpaceLeaveOut"] = free_space(
            lo.layer, measured, [v.camera for v in leave.views], renderer
        )
        layers: dict[str, Splats | None] = {layer_name: lo.layer}
        for name, extra in (extra_layers or {}).items():
            layers[name] = extra
            if lo.layer is not None and len(lo.layer):
                layers[f"{name}+{layer_name}"] = Splats.concat([extra, lo.layer])
        report["heldOut"] = leave_out_scores(
            leave, renderer, perceptual, layers, targets, out / "renders"
        )
    timings["gradeS"] = round(time.time() - t0, 1)
    t0 = time.time()
    sheet(
        renderer,
        measured,
        arm.parts,
        f"{scan}: {layer_name} ({generator.name})",
        out / "renders" / f"sheet-{layer_name}.png",
        around=full.focus.centre,
    )
    timings["sheetS"] = round(time.time() - t0, 1)
    timings["totalS"] = round(time.time() - started, 1)
    report["timings"] = timings
    report["calls"] = getattr(generator, "calls", None)
    gf.write_json(out / "report.json", report)
    return report


# --- the command line ----------------------------------------------------------------------------------------

#: Set inside Modal (`infra/modal/fill_objects.py`): spawn a GPU class's method and wait for it.
BACKEND: tuple[Submit, Wait] | None = None
#: The generators `--method` names: (Modal class, frame convention, multi-image, up known).
METHODS = {
    "trellis": ("GenTrellis", "z-up", True),
    "standin": ("", "z-up", True),
}


def _log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def smoke_request() -> dict[str, Any]:
    """A tiny request (an orange disc on its mask, four steps each) that checks the model
    runs before a job sends it the real ones."""
    yy, xx = np.mgrid[0:128, 0:128]
    disc = (xx - 64) ** 2 + ((yy - 70) * 1.3) ** 2 < 44**2
    rgba = np.zeros((128, 128, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 230, 120, 30
    rgba[..., 3] = disc * 255
    frame = {"view": "smoke", "png": _png(rgba), "box": [0, 0, 128, 128]}
    return {
        "key": "smoke",
        "target": -1,
        "frames": [frame],
        "seed": 1,
        "ss": {"steps": 4},
        "slat": {"steps": 4},
    }


def load_layer(archive: Path, work: Path) -> Splats:
    """An inferred layer (`inferred.tar.gz` as fill.yml uploads it) as gaussians."""
    import tarfile

    from splat_render import load_tileset

    work.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if member.isfile() and len(parts) == 2 and parts[0] == "inferred":
                source = tar.extractfile(member)
                assert source is not None
                (work / parts[1]).write_bytes(source.read())
    return load_tileset(work / "tileset.json")


def main(argv: list[str] | None = None) -> int:
    import argparse

    from splat_render import load_tileset

    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="complete the objects of one scan with one method")
    r.add_argument("tileset", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--instances", type=Path, required=True, help="the segmentation variant's")
    r.add_argument("--poses", type=Path, required=True)
    r.add_argument("--frames", type=Path)
    r.add_argument("--placement", type=Path)
    r.add_argument("--scan", default="scan")
    r.add_argument("--caption", default="")
    r.add_argument("--concept", default=CONCEPT)
    r.add_argument("--method", default="trellis", choices=sorted(METHODS))
    r.add_argument("--leave-out", choices=("none", "high", "low"), default="low")
    r.add_argument("--seeds", default="1,2")
    r.add_argument("--renderer", choices=("cpu", "gsplat"), default="gsplat")
    r.add_argument("--perceptual", choices=("lpips", "standin"), default="lpips")
    r.add_argument("--quality-width", type=int, default=320)
    r.add_argument("--min-frame-psnr", type=float, default=13.0)
    r.add_argument(
        "--compare", action="append", default=[], help="name=inferred.tar.gz (leave-out layers)"
    )
    args = p.parse_args(argv)
    cls, frame, multi = METHODS[args.method]
    smoke = None
    if args.method == "standin":
        generator: Generator = StandInGenerator()
    else:
        if BACKEND is None:
            raise SystemExit(
                "the remote models need a backend (run through infra/modal/fill_objects.py)"
            )
        generator = RemoteGenerator(*BACKEND, cls=cls, name=args.method, frame=frame)
        # Sent now, read after the frame check: the model's container loads meanwhile, and a
        # model that cannot run stops the job before its quality pass.
        smoke = generator.start([smoke_request()])
    renderer = tf.make_renderer(args.renderer)
    measured = load_tileset(args.tileset)
    placement = json.loads(args.placement.read_text()) if args.placement else None
    views = gf.real_views(args.poses, placement, args.frames)
    check = gf.frame_check(measured, views, renderer)
    _log(f"frame check: {json.dumps(check)}")
    if check["meanPsnr"] is None or check["meanPsnr"] < args.min_frame_psnr:
        raise SystemExit(f"the cameras do not match the tiles (frame check {check['meanPsnr']} dB)")
    instances = json.loads(args.instances.read_text(encoding="utf-8"))
    smoked: dict[str, Any] = {}
    if smoke is not None:
        (result,) = smoke()
        smoked = {k: v for k, v in result.info.items() if k != "gaussians"}
        _log(f"model smoke test: {json.dumps(smoked, default=str)[:2000]}")
        if result.splats is None:
            raise SystemExit(f"the model failed its smoke test: {smoked.get('error')}")
    if args.perceptual == "lpips":
        import anchor_models as am

        perceptual: Any = am.Perceptual()
    else:
        perceptual = af.StandInPerceptual()
    extra = {}
    for spec in args.compare:
        name, _, path = spec.partition("=")
        extra[name] = load_layer(Path(path), args.out / f".compare-{name}")
    report = run(
        args.tileset,
        instances,
        measured,
        views,
        generator,
        renderer,
        perceptual,
        args.out,
        scan=args.scan,
        caption=args.caption,
        concept=args.concept,
        leave_out=args.leave_out,
        seeds=[int(s) for s in args.seeds.split(",") if s],
        per_frame=not multi,
        up_known=frame == "z-up",
        quality_width=args.quality_width,
        extra_layers=extra,
        log=_log,
    )
    report["frameCheck"] = check
    report["smoke"] = smoked
    gf.write_json(args.out / "report.json", report)
    print(
        json.dumps(
            {
                "silhouette": report.get("silhouette"),
                "freeSpace": report.get("freeSpace"),
                "heldOut": (report.get("heldOut") or {}).get("mean"),
            },
            default=gf._json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
