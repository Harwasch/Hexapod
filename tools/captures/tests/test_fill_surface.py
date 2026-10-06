"""`fill_surface` and the weak-surface treatment: the spool-like table's top, seen only at
grazing angles, found where it is, coloured from the photos (its dark ring included), made
solid; and a ray through a weak surface does not carve (CPU renderer)."""

from __future__ import annotations

import numpy as np
from fill_scenes import splats, spool_cameras, table_scene
from test_anchor_fill import _measured_tileset

import anchor_fill as af
import fill_quality as fq
import fill_surface as fs
import generative_fill as gf
from splat_render import Camera, Splats, render


def _top_surface():
    scene, m = table_scene()
    cams = spool_cameras()
    focus = fq.capture_focus(cams)
    q = fq.measure_quality(scene, cams, render, focus, width=96)
    clusters = fq.hole_clusters(scene, q.classes, focus, best=q.best)
    region = clusters.labels == 0
    centres = np.array([c.centre for c in cams])
    best = fs.best_cameras(q.per_camera, np.flatnonzero(region), centres, focus.centre, 12)
    return scene, m, cams, focus, region, best


def test_the_top_seen_at_grazing_angles_is_found_and_coloured_from_the_photos() -> None:
    scene, m, cams, focus, region, best = _top_surface()
    assert (region & m["top"]).sum() > 0.6 * m["top"].sum()
    assert len(best) >= 4
    positions, normals, spacing, info = fs.estimate_surface(
        scene, region, [cams[c] for c in best[: fs.SURFACE_VIEWS]], render, focus, width=160
    )
    assert len(positions) > 50 and info["voxels"] == len(positions)
    on_top = (np.abs(positions[:, 2] - 0.7) < 0.04) & (
        np.linalg.norm(positions[:, :2], axis=1) < 0.36
    )
    assert on_top.sum() > 100 and on_top.mean() > 0.6
    assert (np.abs(normals[on_top, 2]) > 0.9).mean() > 0.8
    assert (normals[on_top, 2] > 0).mean() > 0.8  # turned towards the cameras: up
    # Reprojected: the photos' colours on the surface, the top's dark ring included.
    photos = [(cams[c], (render(scene, cams[c]).rgb * 255).astype(np.uint8)) for c in best]
    colours, support = fs.colour_surface(
        positions, normals, photos, scene, render, focus.density, depth_width=96
    )
    surface = fs.Surface(positions, normals, colours, support, spacing)
    assert surface.coloured[on_top].mean() > 0.8
    r = np.linalg.norm(positions[:, :2], axis=1)
    ring = on_top & surface.coloured & (np.abs(r - 0.22) < 0.025)
    plain = on_top & surface.coloured & (r < 0.12)
    assert ring.sum() > 3 and plain.sum() > 3
    assert colours[plain].mean() - colours[ring].mean() > 0.12
    # Made solid: from straight above the discs alone cover the top.
    discs = fs.surface_splats(surface)
    assert len(discs) == surface.coloured.sum() and (discs.opacities >= 0.9).all()
    inner = scene.positions[m["top"] & (np.linalg.norm(scene.positions[:, :2], axis=1) < 0.3)]
    for eye in ([0.0, 0.01, 3.0], [-2.0, 0.3, 1.25]):  # straight above, and low from the side
        cam = Camera.look_at(eye, [0.0, 0.0, 0.7], fov_deg=40, width=96, height=96)
        alpha = fs.ewa_alpha(discs, cam)
        uv, _ = cam.project(inner)
        u, v = np.floor(uv).astype(int).T
        assert (alpha[v, u] >= 0.5).mean() > 0.95, eye
    # Thinned to a limit by coarsening the grid.
    fewer = fs.thin_surface(surface, max(10, int(surface.coloured.sum()) // 3))
    assert fewer.coloured.sum() <= max(10, int(surface.coloured.sum()) // 3)
    assert fewer.spacing > surface.spacing


def test_a_ray_through_a_weak_surface_does_not_carve() -> None:
    # An opaque ground and, above it, a faint (under-constrained) top of flat gaussians: a
    # camera above sees mostly the ground through the top.
    g = np.linspace(-1.5, 1.5, 61)
    gx, gy = np.meshgrid(g, g)
    ground = splats(np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]), [0.2, 0.5, 0.2])
    t = np.linspace(-0.3, 0.3, 7)
    tx, ty = np.meshgrid(t, t)
    n = tx.size
    top = Splats(
        np.column_stack([tx.ravel(), ty.ravel(), np.full(n, 0.5)]),
        np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        np.tile([0.07, 0.07, 0.002], (n, 1)),
        np.tile([0.8, 0.7, 0.5], (n, 1)),
        np.full(n, 0.4),
    )
    measured = Splats.concat([ground, top])
    soft = np.r_[np.zeros(len(ground), bool), np.ones(len(top), bool)]
    cam = [Camera.look_at([0.0, 0.01, 2.5], [0.0, 0.0, 0.0], fov_deg=50, width=96, height=96)]
    over_top = np.array([[0.0, 0.0, 1.0], [0.1, -0.1, 0.9]])
    away = np.array([[0.4, 0.4, 1.0], [-0.45, 0.4, 1.2]])
    hard = af.Carver(measured, cam, render, 1.0)
    assert not hard.keep(over_top).any()  # without the rule, seen "through" the top
    gentle = af.Carver(measured, cam, render, 1.0, soft)
    assert gentle.keep(over_top).all()
    # Free space away from the weak surface is still carved.
    assert not gentle.keep(away).any()


def test_the_photo_arm_makes_the_top_solid_and_the_headline_shows_it(tmp_path) -> None:
    from fill_scenes import photos

    scene, _ = table_scene()
    cams = spool_cameras()
    paths = photos(scene, cams, tmp_path / "frames")
    views = [gf.RealView(p.name, c, None, p) for p, c in zip(paths, cams, strict=True)]
    setup, _ = af.make_setup(
        "table", "a round wooden table", scene, views, render, width=96, log=lambda s: None
    )
    options = af.Options(
        arms=("refs", "norefs"),
        fill_size=(96, 56),
        set_size=(64, 36),
        probe_size=(64, 36),
        quality_width=96,
        azimuths=8,
        anchors=(2, 3),
        propagation=(2, 3),
        seeds=2,
        prop_seeds=2,
        distill=0,
        update_distill=0,
        lift_stride=2,
        carve_width=96,
        score_width=96,
    )
    report = af.run_arms(
        setup,
        af.StandInEditor(),
        None,
        af.OracleDepth(scene, render),
        render,
        af.StandInPerceptual(),
        options,
        _measured_tileset(tmp_path / "scan"),
        tmp_path / "out",
        log=lambda s: None,
    )
    assert report["surface"]["gaussians"] > 0
    refs, norefs = report["candidates"]["refs"], report["candidates"]["norefs"]
    assert refs["surface"] == report["surface"]["gaussians"] and norefs["surface"] == 0
    # The photo arm's views were drawn with the reprojected surface (cleaned, not made).
    head = report["headline"]
    assert (tmp_path / "out" / "renders" / "headline-table.png").exists()
    assert len(head["views"]) == len(af.HEADLINE_VIEWS)
    above = head["views"][0]
    # (Opacity itself is checked with the exact EWA alpha below: the CPU renderer samples.)
    assert above["elevation"] == 90.0 and above["refs"] is not None
    assert set(head["verdict"]) >= {"before", "refs", "norefs"}
    # Swap, don't stack: the photo arm names the measured gaussians its surface replaces
    # (the top's), and the control none.
    assert refs["superseded"] > 0 and "superseded" not in norefs
    # The hemisphere views were added where the chosen ones left directions unseen.
    assert report["plan"]["hemisphere"]


def test_superseded_gaussians_lie_on_the_discs_and_round_trip_per_tile(tmp_path) -> None:
    import splat_tiles
    from rebind_instances import decode_runs
    from rig_tiles import tile_positions
    from splat_render import load_tileset, save_ply
    from synthetic_tree import checksum_positions

    scene, _ = table_scene()
    save_ply(tmp_path / "scan.ply", scene)
    # Small tiles: leaves under merged parents, as a real scan's.
    tiles_dir = tmp_path / "tiles"
    splat_tiles.convert(tmp_path / "scan.ply", tiles_dir, 0.0, 0.0, 0.0, 0.0, 400)
    measured = load_tileset(tiles_dir / "tileset.json")
    g = np.linspace(-0.35, 0.35, 15)
    gx, gy = np.meshgrid(g, g)
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 0.7)])
    pts = pts[np.linalg.norm(pts[:, :2], axis=1) < 0.36]
    surface = fs.Surface(
        pts,
        np.tile([0.0, 0.0, 1.0], (len(pts), 1)),
        np.full((len(pts), 3), 0.5),
        np.ones(len(pts)),
        0.05,
    )
    discs = fs.surface_splats(surface)
    mask = fs.superseded(measured.positions, discs)
    on_top = np.abs(measured.positions[:, 2] - 0.7) < 0.01
    inner = on_top & (np.linalg.norm(measured.positions[:, :2], axis=1) < 0.3)
    assert mask[inner].all()
    assert not mask[measured.positions[:, 2] < 0.6].any()  # the side and the ground stay
    doc = fs.supersede_document(tiles_dir, mask)
    assert doc is not None and doc["superseded"] == int(mask.sum())
    # Leaf tiles carry the mask exactly, in load_tileset's order.
    import json

    tileset = json.loads((tiles_dir / "tileset.json").read_text())
    leaves, parents, stack = [], [], [tileset["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("children"):
            stack.extend(tile["children"])
            parents += [tile["content"]["uri"]] if tile.get("content") else []
        else:
            leaves.append(tile["content"]["uri"])
    got = np.concatenate(
        [
            decode_runs(doc["tiles"][checksum_positions(tile_positions(tiles_dir / u))])
            for u in sorted(leaves)
        ]
    )
    assert (got == mask.astype(int)).all()
    # Every merged parent is listed too, so a far view hides the same patch: its flagged
    # splats are the top's, and some are flagged.
    assert parents
    flagged = 0
    for uri in parents:
        at = tile_positions(tiles_dir / uri)
        flags = decode_runs(doc["tiles"][checksum_positions(at)]).astype(bool)
        assert flags.size == len(at)
        assert (np.abs(at[flags, 2] - 0.7) < 0.1).all()
        flagged += int(flags.sum())
    assert flagged > 0
    assert fs.supersede_document(tmp_path / "nowhere", mask) is None
    # A parent's file missing (a job that fetched the leaves only) is an error the writer
    # reports, not a failed run.
    (tiles_dir / parents[0]).unlink()
    layer = tmp_path / "layer"
    layer.mkdir()
    written = af.write_supersedes(layer, tiles_dir, mask, log=lambda _: None)
    assert written["written"] is False and "error" in written
    assert not (layer / "supersedes.json").exists()


def test_bilinear_samples_any_number_of_points() -> None:
    image = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    uv = np.tile([[2.5, 1.5]], (40000, 1))  # a pixel centre, many times (past remap's limit)
    out = fs._bilinear(image, uv)
    assert out.shape == (40000, 3) and np.allclose(out[0], image[1, 2])
    mid = fs._bilinear(image, np.array([[3.0, 1.5]]))
    assert np.allclose(mid[0], (image[1, 2] + image[1, 3]) / 2)
