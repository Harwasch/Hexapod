"""Weak surfaces made solid from the photos that saw them: the first treatment of a weak region,
ahead of any generator.

A surface the cameras saw only from one side, at grazing angles (the spool's top), was fitted
by semi-transparent, strongly view-dependent gaussians: from the angles it was filmed it looks
right, from above or the far side it is see-through. What it looks like is known -- from the
photos that saw it -- and so is roughly where it is. So, with nothing picked by hand:

1. **Where the surface is** (`estimate_surface`): the views that saw the region best
   (`best_cameras`, by the per-camera quality of `fill_quality`) render the scan's depth;
   their pixels that the region covers solidly are back-projected and fused on a voxel grid
   (a mean point per voxel), with each point's normal from the plane of its neighbours.
2. **What it looks like** (`colour_surface`): each point takes the colour the best real
   photos show there (projected, depth-tested against the scan, weighted by viewing angle x
   footprint, squared, so the squarest look wins) -- true pixels moved onto the surface. A
   point no photo saw usefully gets none (`MIN_SUPPORT`) and is left to the generator.
3. **Solid** (`surface_splats`): one flat, opaque disc per coloured point, one colour each
   (no view dependence). Rendered from a new pose with the scan, they are the pseudo-views:
   the photos' pixels at new angles, for a generator only to clean at low strength. Fused
   into the inferred layer with their shape and a floor on their opacity held
   (`anchor_fill.fuse`), they keep the top from being see-through.

Numpy and scipy; renders through any `splat_render`-style renderer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import fill_quality as fq
from splat_render import Camera, Splats

#: Real views whose depth is fused, and real photos warped onto the surface (best first,
#: directions closer than `VIEW_NMS_DEG` suppressed).
SURFACE_VIEWS = 8
COLOUR_VIEWS = 12
VIEW_NMS_DEG = 6.0
#: A pixel is back-projected when the scan covers it at least this much, and the region at
#: least this share of that.
SURFACE_ALPHA = 0.85
SURFACE_SHARE = 0.6
#: A point whose best photo weight (|cos| x footprint) is below this is not coloured.
MIN_SUPPORT = 0.05
#: The discs: opacity, and the floor the distil may not take it below; size against the
#: grid spacing (overlapping, so no gap opens between them).
DISC_OPACITY = 0.95
OPACITY_FLOOR = 0.85
DISC_SCALE = 0.75
DISC_REACH = 3.0
#: The grid spacing: the region's median gaussian spacing, within these shares of the
#: focus radius.
SPACING_RANGE = (1.0 / 400.0, 1.0 / 40.0)
#: The region's own gaussians (at least this opaque) are surface samples too: they fill
#: where the best views' pixels leave gaps (far edges seen at the most grazing angles).
MEMBER_OPACITY = 0.05


@dataclass
class Surface:
    """The estimated surface: points, unit normals (towards the cameras that saw it), each
    point's photo colour (0..1; NaN where none) and best photo weight, the grid spacing,
    and what was used."""

    positions: np.ndarray
    normals: np.ndarray
    colours: np.ndarray
    support: np.ndarray
    spacing: float
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def coloured(self) -> np.ndarray:
        return (self.support >= MIN_SUPPORT) & np.isfinite(self.colours).all(axis=1)

    def __len__(self) -> int:
        return len(self.positions)


def best_cameras(
    per_camera: Any,
    rows: np.ndarray,
    centres: np.ndarray,
    focus_centre: np.ndarray,
    k: int,
    nms_deg: float = VIEW_NMS_DEG,
) -> list[int]:
    """The `k` cameras that saw the gaussians `rows` best (summed quality), none within
    `nms_deg` (as seen from the focus) of one taken."""
    if not len(rows):
        return []
    score = np.asarray(per_camera[:, rows].sum(axis=1)).ravel()
    dirs = np.asarray(centres, np.float64) - np.asarray(focus_centre, np.float64)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    out: list[int] = []
    cos_min = math.cos(math.radians(nms_deg))
    for c in np.argsort(-score, kind="stable"):
        if score[c] <= 0 or len(out) >= k:
            break
        if any(float(dirs[c] @ dirs[o]) > cos_min for o in out):
            continue
        out.append(int(c))
    return out


def back_project(
    camera: Camera, depth: np.ndarray, mask: np.ndarray, stride: int = 1
) -> np.ndarray:
    """World points of `mask`'s pixels (every `stride`-th) at `depth` along the view axis."""
    sel = np.zeros_like(mask)
    sel[::stride, ::stride] = mask[::stride, ::stride]
    v, u = np.nonzero(sel & np.isfinite(depth))
    z = depth[v, u]
    local = np.column_stack(
        [
            (u + 0.5 - camera.width / 2) * z / camera.focal,
            (v + 0.5 - camera.height / 2) * z / camera.focal,
            z,
        ]
    )
    return camera.centre + local @ camera.rotation


def grid_spacing(positions: np.ndarray, focus: fq.Focus) -> float:
    s = fq.median_spacing(positions) if len(positions) > 1 else focus.radius / 100
    low, high = (r * focus.radius for r in SPACING_RANGE)
    return float(np.clip(s, low, high))


def _indicator(scene: Splats, region: np.ndarray) -> Splats:
    colours = np.repeat(region.astype(np.float64)[:, None], 3, axis=1)
    return Splats(scene.positions, scene.rotations, scene.scales, colours, scene.opacities)


def estimate_surface(
    scene: Splats,
    region: np.ndarray,
    cameras: Sequence[Camera],
    renderer: Any,
    focus: fq.Focus,
    *,
    width: int = 480,
    stride: int = 2,
    spacing: float | None = None,
    facing: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Points on the surface of `region` (per gaussian of `scene`) and their normals: each
    of `cameras` (the best views of it) renders the scan; its pixels that the scan covers
    solidly (`SURFACE_ALPHA`) and mostly with the region (`SURFACE_SHARE`) are
    back-projected at the rendered depth; with the region's own gaussians
    (`MEMBER_OPACITY`) they are fused one per voxel of the grid spacing (their mean), kept
    within two spacings of a region gaussian. Normals: the plane of the 16 nearest points,
    turned towards the cameras that saw it (`facing`, per gaussian, for the region's own)."""
    from scipy.spatial import cKDTree

    region = np.asarray(region, bool)
    members = scene.positions[region]
    if spacing is None:
        spacing = grid_spacing(members, focus)
    pts, eyes = [], []
    marked = _indicator(scene, region)
    per_view = []
    for cam in cameras:
        small = _scaled(cam, width)
        frame = renderer(scene, small)
        share_frame = renderer(marked, small)
        share = share_frame.rgb[..., 0] / np.maximum(share_frame.alpha, 1e-6)
        mask = (frame.alpha >= SURFACE_ALPHA) & (share >= SURFACE_SHARE)
        p = back_project(small, frame.depth, mask, stride)
        per_view.append(len(p))
        if len(p):
            pts.append(p)
            eyes.append(np.repeat(np.asarray(cam.centre, np.float64)[None], len(p), axis=0))
    report: dict[str, Any] = {"views": len(cameras), "pixels": per_view, "spacing": spacing}
    solid = region & (np.asarray(scene.opacities) >= MEMBER_OPACITY)
    if solid.any():
        own = scene.positions[solid]
        if facing is not None:
            ahead = own + np.asarray(facing, np.float64)[solid]
        else:
            centre = np.mean([c.centre for c in cameras], axis=0) if cameras else own.mean(0)
            ahead = np.repeat(np.asarray(centre, np.float64)[None], len(own), axis=0)
        pts.append(own)
        eyes.append(ahead)
        report["members"] = int(solid.sum())
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3)), spacing, report
    p = np.concatenate(pts)
    e = np.concatenate(eyes)
    keys = np.floor(p / spacing).astype(np.int64)
    _, group, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    group = np.asarray(group).reshape(-1)
    n = len(counts)
    fused = np.column_stack([np.bincount(group, p[:, j], n) for j in range(3)]) / counts[:, None]
    toward = np.column_stack([np.bincount(group, e[:, j], n) for j in range(3)]) / counts[:, None]
    if len(members):
        near, _ = cKDTree(members).query(fused, k=1)
        keep = near <= 2.0 * spacing
        fused, toward = fused[keep], toward[keep]
    if len(fused) >= 4:
        normals, _ = fq.local_normals(fused, 16)
    else:
        normals = np.tile([0.0, 0.0, 1.0], (len(fused), 1))
    flip = np.einsum("ij,ij->i", normals, toward - fused) < 0
    normals[flip] *= -1
    report.update({"points": len(p), "voxels": len(fused)})
    return fused, normals, spacing, report


def _scaled(camera: Camera, width: int) -> Camera:
    if camera.width <= width:
        return camera
    h = max(1, round(camera.height * width / camera.width))
    return Camera(
        camera.rotation, camera.centre, camera.focal * width / camera.width, width, h, camera.far
    )


def _bilinear(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    import cv2

    img = np.asarray(image, np.float32)
    mx = uv[:, 0].astype(np.float32).reshape(-1, 1) - 0.5
    my = uv[:, 1].astype(np.float32).reshape(-1, 1) - 0.5
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(-1, img.shape[2]) if img.ndim == 3 else out.reshape(-1)


def colour_surface(
    positions: np.ndarray,
    normals: np.ndarray,
    views: Sequence[tuple[Camera, np.ndarray]],
    scene: Splats,
    renderer: Any,
    density: float,
    *,
    depth_width: int = 320,
) -> tuple[np.ndarray, np.ndarray]:
    """Each point's colour from the real photos `views` ((camera at the photo's size,
    photo)): projected into each, kept where the point is no farther than the scan's surface
    there (rendered at `depth_width`), weighted by |cos| (ray, normal) x footprint (the
    photo's pixels per metre there against `density`) squared. Returns colours (0..1, NaN
    where no photo saw the point) and each point's best weight."""
    m = len(positions)
    acc = np.zeros((m, 3))
    wsum = np.zeros(m)
    best = np.zeros(m)
    for cam, photo in views:
        if photo is None:
            continue
        small = _scaled(cam, depth_width)
        depth = fq.nearest_surface(renderer(scene, small).depth)
        uv, z = cam.project(positions)
        inside = (
            (z > 1e-3)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < cam.width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < cam.height)
        )
        rows = np.flatnonzero(inside)
        if not rows.size:
            continue
        f = small.width / cam.width
        u = np.clip(np.floor(uv[rows, 0] * f).astype(np.int64), 0, small.width - 1)
        v = np.clip(np.floor(uv[rows, 1] * f).astype(np.int64), 0, small.height - 1)
        surf = depth[v, u]
        seen = np.isfinite(surf) & (z[rows] <= surf * (1 + fq.DEPTH_TOLERANCE) + 0.02 * surf)
        rows = rows[seen]
        if not rows.size:
            continue
        ray = positions[rows] - cam.centre
        ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12)
        cos = np.abs(np.einsum("ij,ij->i", ray, normals[rows]))
        foot = np.clip((cam.focal / np.maximum(z[rows], 1e-9)) / max(density, 1e-9), 0.0, 1.0)
        w = cos * foot
        colour = _bilinear(photo, uv[rows]) / 255.0
        acc[rows] += (w**2)[:, None] * colour
        wsum[rows] += w**2
        best[rows] = np.maximum(best[rows], w)
    colours = np.where(wsum[:, None] > 0, acc / np.maximum(wsum, 1e-12)[:, None], np.nan)
    return colours, best


def surface_splats(surface: Surface) -> Splats:
    """One flat, opaque disc per coloured point: facing its normal, `DISC_OPACITY`, the
    photo colour; across `DISC_SCALE` of the spacing, or of the distance to its fourth
    nearest neighbour where the points are sparser (at most `DISC_REACH` spacings), so no
    gap opens between them."""
    keep = surface.coloured
    s = surface.spacing
    n = int(keep.sum())
    if not n:
        return Splats(*(np.zeros((0, k)) for k in (3, 4, 3, 3)), np.zeros(0))
    from scipy.spatial import cKDTree

    from teacher_fill import _disc_rotations

    pos = surface.positions[keep]
    if n > 4:
        d, _ = cKDTree(pos).query(pos, k=5)
        local = d[:, 4]
    else:
        local = np.full(n, s)
    size = DISC_SCALE * np.clip(local, s, DISC_REACH * s)
    return Splats(
        pos,
        _disc_rotations(surface.normals[keep]),
        np.column_stack([size, size, 0.1 * size]),
        np.clip(surface.colours[keep], 0.0, 1.0),
        np.full(n, DISC_OPACITY),
    )


def thin_surface(surface: Surface, limit: int) -> Surface:
    """At most `limit` coloured points: the grid coarsened (voxels of twice, three times...
    the spacing, the best-supported point of each) until they fit."""
    keep = np.flatnonzero(surface.coloured)
    if keep.size <= limit:
        return surface
    factor = 1.0
    while keep.size > limit and factor < 16:
        factor *= 1.25
        keys = np.floor(surface.positions[keep] / (surface.spacing * factor)).astype(np.int64)
        order = np.argsort(-surface.support[keep], kind="stable")
        _, first = np.unique(keys[order], axis=0, return_index=True)
        keep = np.sort(keep[order[first]])
    return Surface(
        surface.positions[keep],
        surface.normals[keep],
        surface.colours[keep],
        surface.support[keep],
        surface.spacing * factor,
        {**surface.report, "thinnedBy": round(factor, 3)},
    )
