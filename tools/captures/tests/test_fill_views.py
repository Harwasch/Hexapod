"""`fill_views`: pixel classes, greedy view selection (high views for a table seen from below
its top, low views for a ball seen from above), orders and context retrieval."""

from __future__ import annotations

import numpy as np
from fill_scenes import ball_scene, pumpkin_cameras, splats, spool_cameras, table_scene

import fill_quality as fq
import fill_views as fv
from splat_render import Camera, render


def _plane(n: int, size: float, z: float = 0.0) -> np.ndarray:
    g = np.linspace(-size / 2, size / 2, n)
    gx, gy = np.meshgrid(g, g)
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def test_pixel_classes_partition_and_find_enclosed_holes() -> None:
    p = _plane(31, 1.2)
    p = p[np.linalg.norm(p[:, :2], axis=1) > 0.2]  # a hole in the middle
    s = splats(p, [0.5, 0.5, 0.5], size=0.03)
    known = p[:, 0] < 0.0
    weak = ~known
    cam = Camera.look_at(
        [0.0, 0.0, 1.5], [0.0, 0.0, 0.0], fov_deg=70, width=96, height=72, up=(0, 1, 0)
    )
    pc = fv.pixel_classes(s, cam, render, known, weak)
    total = pc.known.astype(int) + pc.weak + pc.unknown + pc.void
    assert (total == 1).all()
    # The hole the plane surrounds is unknown; the background around it void.
    assert pc.unknown[36, 48] and pc.void[2, 2]
    assert pc.known[36, 20] and pc.weak[36, 76]
    shares = pc.shares()
    assert 0 < shares["unknown"] < 0.2 and shares["shown"] > 0.5


def test_something_in_front_of_a_known_surface_is_not_known() -> None:
    back = splats(_plane(21, 1.0, 0.0), [0.2, 0.2, 0.8], size=0.04)
    front = splats(_plane(9, 0.3, 0.5), [0.8, 0.2, 0.2], size=0.04)
    s = type(back).concat([back, front])
    known = np.r_[np.ones(len(back), bool), np.zeros(len(front), bool)]
    cam = Camera.look_at(
        [0.0, 0.0, 2.0], [0.0, 0.0, 0.0], fov_deg=60, width=80, height=80, up=(0, 1, 0)
    )
    pc = fv.pixel_classes(s, cam, render, known, np.zeros(len(s), bool))
    assert pc.unknown[40, 40] and pc.known[27, 27]


def test_the_back_of_what_the_cameras_saw_is_not_known() -> None:
    plane = splats(_plane(21, 1.0), [0.3, 0.6, 0.3], size=0.04)
    above = [
        Camera.look_at([x, 0.3, 1.5], [0, 0, 0], fov_deg=60, width=80, height=80)
        for x in (-0.5, 0.5)
    ]
    focus = fq.capture_focus(above)
    q = fq.measure_quality(plane, above, render, focus, width=80)
    assert (q.classes == fq.KNOWN).mean() > 0.8
    assert (q.facing[:, 2] > 0.9).mean() > 0.9  # turned towards the cameras: up
    below = Camera.look_at([0.0, 0.2, -1.5], [0, 0, 0], fov_deg=60, width=80, height=80)
    known = q.classes == fq.KNOWN
    weak = q.classes == fq.WEAK
    seen_from_below = fv.pixel_classes(
        plane, below, render, known, weak, fq.facing(q, below.centre, plane.positions)
    )
    assert seen_from_below.unknown.sum() > 100
    assert seen_from_below.known.sum() < 0.03 * seen_from_below.unknown.sum()
    from_above = fv.pixel_classes(
        plane, above[0], render, known, weak, fq.facing(q, above[0].centre, plane.positions)
    )
    assert from_above.known.sum() > 100


def test_select_views_covers_greedily_twice_with_suppression() -> None:
    # 6 candidates on a circle; elements 0-9 seen by 0 and 1 (5 deg apart), 10-19 by 3 and 4.
    angles = np.radians([0, 5, 60, 120, 180, 240])
    dirs = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(6)])
    covers = np.zeros((6, 20), bool)
    covers[0, :10] = covers[1, :10] = True
    covers[3, 10:] = covers[4, 10:] = True
    covers[2, :5] = True
    w = np.ones(20)
    sel = fv.select_views(dirs, covers, w, np.ones(6, bool), anchors=(1, 3), propagation=(1, 4))
    assert set(sel.anchors) <= {0, 1, 3, 4} and len(sel.anchors) == 2
    assert 1 not in sel.anchors or 0 not in sel.anchors  # 5 degrees apart: one is suppressed
    # Twice: 1 is suppressed by 0, so the second sight of 0-4 is view 2, and 10-19 the other.
    assert sel.report["seenOnce"] == 1.0
    chosen = sel.anchors + sel.propagation
    assert {3, 4} <= set(chosen) and 2 in chosen
    unusable = np.ones(6, bool)
    unusable[[0, 1]] = False
    sel2 = fv.select_views(dirs, covers, w, unusable, anchors=(1, 2), propagation=(0, 2))
    assert 0 not in sel2.anchors + sel2.propagation


def _choose(scene, cams, **kw):
    focus = fq.capture_focus(cams)
    q = fq.measure_quality(scene, cams, render, focus, width=96)
    clusters = fq.hole_clusters(scene, q.classes, focus, best=q.best)
    cands, sel, chosen = fv.choose_views(
        scene,
        q.classes,
        q.normals,
        q.best,
        clusters,
        focus,
        render,
        fill_size=(160, 96),
        probe_size=(80, 48),
        azimuths=12,
        quality=q,
        **kw,
    )
    return focus, q, clusters, cands, sel, chosen


def test_the_spools_high_views_are_found() -> None:
    scene, _ = table_scene()
    focus, _q, clusters, cands, sel, chosen = _choose(
        scene, spool_cameras(), anchors=(4, 6), propagation=(4, 8)
    )
    assert clusters.info, "the top should be a hole cluster"
    assert 4 <= len(sel.anchors) <= 6
    elev = [cands[c].elevation for c in sel.anchors]
    assert max(elev) >= 65.0 and np.mean(elev) >= 40.0, elev
    # The views see the top well: most of its fill weight is seen twice.
    assert sel.report["seenTwice"] > 0.8
    for c in sel.anchors:
        assert cands[c].eligible
    # The chosen cameras look at what the real cameras filmed.
    for cam in chosen.values():
        uv, z = cam.project(focus.centre[None])
        assert z[0] > 0 and abs(uv[0, 0] - cam.width / 2) < 1 and abs(uv[0, 1] - cam.height / 2) < 1


def test_the_balls_low_views_are_found() -> None:
    scene, _ = ball_scene()
    _focus, _q, _clusters, cands, sel, _chosen = _choose(
        scene, pumpkin_cameras(), anchors=(4, 6), propagation=(4, 8)
    )
    elev = [cands[c].elevation for c in sel.anchors + sel.propagation]
    assert min(elev) <= 15.0, elev
    low_anchor = [cands[c].elevation for c in sel.anchors]
    assert min(low_anchor) <= 15.0


def test_propagation_moves_outwards_and_the_tour_is_smooth() -> None:
    angles = np.radians(np.arange(0, 360, 30))
    dirs = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(angles.size)])
    order = fv.propagation_order(dirs, [0], [6, 1, 3, 11])
    assert order[0] in (1, 11) and order[-1] == 6
    walk = fv.tour(dirs, [0, 6, 3, 9, 1], start=0)
    assert walk[0] == 0 and walk[1] == 1 and walk[2] == 3


def test_context_photos_are_voted_by_the_holes_border() -> None:
    from scipy.sparse import csr_matrix

    # 5 cameras around a hole at the origin; gaussians 0-9 are its border.
    centres = np.array([[3, 0, 1], [2.9, 0.3, 1], [0, 3, 1], [-6, 0, 1], [0, -1.5, 1]], float)
    q = np.zeros((5, 20))
    q[0, :10] = 0.9
    q[1, :10] = 0.8  # nearly the same direction as camera 0: suppressed for "top"
    q[2, :10] = 0.5
    q[3, :10] = 0.4  # far: the wide shot
    q[4, :10] = 0.3  # near: the close-up
    q[:, 10:] = 0.9  # not the border: no say
    ctx = fv.retrieve_context(csr_matrix(q), np.arange(10), centres, np.zeros(3))
    assert ctx["top"] == [0, 2]
    assert ctx["wide"] == 3 and ctx["close"] == 4
    seeds = [fv.context_for_seed(ctx, k) for k in range(4)]
    assert seeds[0] == [("top0", 0), ("top1", 2)]
    assert seeds[1] == [("top0", 0), ("wide", 3)]
    assert seeds[2] == [("top1", 2), ("close", 4)]
    assert seeds[3] == [("wide", 3), ("close", 4)]
    assert fv.context_for_seed({"top": []}, 0) == []


def test_border_gaussians_are_the_known_ones_touching_the_hole() -> None:
    p = np.array([[0, 0, 0], [0.1, 0, 0], [0.15, 0, 0], [1.0, 0, 0]], float)
    border = fv.border_gaussians(p, np.array([0]), np.array([1, 2, 3]), radius=0.12)
    assert border.tolist() == [1]
