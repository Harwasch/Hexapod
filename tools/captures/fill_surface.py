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
    """`image` sampled bilinearly at pixel coordinates `uv` (n, 2), edges clamped. In numpy:
    OpenCV's `remap` refuses maps longer than 32767 (run 37537234426)."""
    img = np.asarray(image, np.float32)
    h, w = img.shape[:2]
    x = np.clip(uv[:, 0] - 0.5, 0.0, w - 1.0)
    y = np.clip(uv[:, 1] - 0.5, 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    if img.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


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


def ewa_alpha(splats: Splats, camera: Camera) -> np.ndarray:
    """The coverage (alpha) `splats` render to at `camera`, by EWA splatting as 3DGS draws it:
    each gaussian's projected 2D covariance (plus a 0.3 px low-pass), front to back, alpha
    capped at 0.99 and dropped below 1/255. Exact where `splat_render.render` samples;
    numpy, one gaussian at a time: for checks on a few thousand, not for scenes."""
    w, h = camera.width, camera.height
    pc = (np.asarray(splats.positions, np.float64) - camera.centre) @ camera.rotation.T
    z = pc[:, 2]
    rot = fq._rotation_matrices(np.asarray(splats.rotations, np.float64))
    sg = rot * np.asarray(splats.scales, np.float64)[:, None, :]
    cov3 = sg @ sg.transpose(0, 2, 1)
    covc = camera.rotation[None] @ cov3 @ camera.rotation.T[None]
    f = camera.focal
    zs = np.maximum(z, 1e-6)
    jac = np.zeros((len(z), 2, 3))
    jac[:, 0, 0] = f / zs
    jac[:, 0, 2] = -f * pc[:, 0] / zs**2
    jac[:, 1, 1] = f / zs
    jac[:, 1, 2] = -f * pc[:, 1] / zs**2
    cov2 = jac @ covc @ jac.transpose(0, 2, 1)
    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + 0.3
    det = a * c - b * b
    u = f * pc[:, 0] / zs + w / 2
    v = f * pc[:, 1] / zs + h / 2
    mid = 0.5 * (a + c)
    radius = np.ceil(3 * np.sqrt(mid + np.sqrt(np.maximum(mid * mid - det, 0.1))))
    keep = (
        (z > 0.05)
        & (det > 0)
        & (u + radius > 0)
        & (u - radius < w)
        & (v + radius > 0)
        & (v - radius < h)
        & (np.asarray(splats.opacities) > 1 / 255)
    )
    rows = np.flatnonzero(keep)
    rows = rows[np.argsort(z[rows])]
    transmit = np.ones((h, w))
    ia, ib, ic = (
        c / np.where(det > 0, det, 1),
        -b / np.where(det > 0, det, 1),
        a / np.where(det > 0, det, 1),
    )
    for i in rows:
        x0, x1 = int(max(0, u[i] - radius[i])), int(min(w, u[i] + radius[i] + 1))
        y0, y1 = int(max(0, v[i] - radius[i])), int(min(h, v[i] + radius[i] + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        dx = np.arange(x0, x1) + 0.5 - u[i]
        dy = np.arange(y0, y1) + 0.5 - v[i]
        q = (
            ia[i] * dx[None, :] ** 2
            + 2 * ib[i] * dx[None, :] * dy[:, None]
            + ic[i] * dy[:, None] ** 2
        )
        al = np.minimum(0.99, splats.opacities[i] * np.exp(-0.5 * q))
        al[al < 1 / 255] = 0
        transmit[y0:y1, x0:x1] *= 1 - al
    return 1 - transmit


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


#: A measured gaussian is superseded by the rebuilt surface when it lies on a disc: within
#: this many of the disc's sizes of its centre in the disc's plane, and this many across it.
SUPERSEDE_ALONG = 1.5
SUPERSEDE_ACROSS = 1.0


def superseded(positions: np.ndarray, discs: Splats) -> np.ndarray:
    """Per measured gaussian (`positions`), whether a disc of the rebuilt surface replaces
    it: it lies on the disc -- in its plane (`SUPERSEDE_ACROSS` sizes off it at most) and
    over it (`SUPERSEDE_ALONG` sizes from its centre). The measured gaussians a surface
    patch is rebuilt over, whatever their class; the board's rim below the patch stays."""
    from scipy.spatial import cKDTree

    out = np.zeros(len(positions), bool)
    if not len(discs) or not len(positions):
        return out
    size = discs.scales[:, :2].max(axis=1)
    normals = fq.shortest_axes(discs)
    tree = cKDTree(discs.positions)
    reach = SUPERSEDE_ALONG * float(size.max())
    d, j = tree.query(positions, k=1, distance_upper_bound=reach)
    rows = np.flatnonzero(np.isfinite(d))
    if not rows.size:
        return out
    j = j[rows]
    offset = positions[rows] - discs.positions[j]
    across = np.abs(np.einsum("ij,ij->i", offset, normals[j]))
    along = np.sqrt(np.maximum((offset**2).sum(axis=1) - across**2, 0.0))
    out[rows] = (across <= SUPERSEDE_ACROSS * size[j]) & (along <= SUPERSEDE_ALONG * size[j])
    return out


SUPERSEDE_ENCODING = (
    "rle per tile (instances.json's addressing): 1 a measured splat the fill's rebuilt "
    "surface replaces, 0 one it leaves; a coarse tile's splat takes the plurality of the 8 "
    "leaf splats nearest to it"
)


def supersede_document(tiles_dir: Any, mask: np.ndarray) -> dict[str, Any] | None:
    """`mask` (per leaf splat, in `splat_render.load_tileset`'s order) as the viewer reads
    per-splat ids (`instances.json`'s `tiles`: per tile checksum, run-length ids, coarse tiles
    by their nearest leaf splats, `rebind_instances`), or None when the tiles do not match
    the mask (not this scan's package)."""
    import json
    from pathlib import Path

    from rebind_instances import encode_runs, rebind
    from rig_tiles import tile_positions
    from synthetic_tree import checksum_positions

    tiles_dir = Path(tiles_dir)
    path = tiles_dir / "tileset.json"
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    leaves: list[str] = []
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("children"):
            stack.extend(tile["children"])
        elif tile.get("content", {}).get("uri"):
            leaves.append(tile["content"]["uri"])
    tiles: dict[str, list[int]] = {}
    at = 0
    for uri in sorted(leaves):
        positions = tile_positions(tiles_dir / uri)
        part = np.asarray(mask[at : at + len(positions)], np.int64)
        if part.size != len(positions):
            return None
        tiles[checksum_positions(positions)] = encode_runs(part)
        at += len(positions)
    if at != len(mask):
        return None
    return {
        "encoding": SUPERSEDE_ENCODING,
        "superseded": int(np.count_nonzero(mask)),
        "tiles": rebind(tiles_dir, tiles),
    }
