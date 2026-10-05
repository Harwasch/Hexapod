"""feature_fields.py: the camera move into the tileset's frame, mask scales, the tree read out
of a field (on the synthetic yard with an oracle field), the contact claim, the ground's
records, and -- with torch -- that the loss teaches a gate to group at large scales and part
at small ones."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import feature_fields as ff
import segment_scene
import synthetic_yard as sy
from splat_render import Splats


def _splats(positions: np.ndarray, scale: float = 0.01, opacity: float = 0.9) -> Splats:
    n = len(positions)
    rot = np.zeros((n, 4))
    rot[:, 0] = 1.0
    return Splats(
        np.asarray(positions, np.float64), rot, np.full((n, 3), scale),
        np.full((n, 3), 0.5), np.full(n, opacity),
    )  # fmt: skip


def _gate(dim: int = 32, split: int = 16, ref: float = 1.0) -> dict[str, np.ndarray]:
    """A hand-made gate: the first `split` channels always open, the rest open only below
    `ref` (objects' code at every scale, parts' code at small scales)."""
    w2 = np.zeros((dim, 1))
    w2[split:, 0] = 10.0
    b2 = np.full(dim, 10.0)
    b2[split:] = -3.0
    return {
        "w1": np.array([[-1.0]]), "b1": np.array([0.0]), "w2": w2, "b2": b2,
        "ref": np.array([math.log(ref)]),
    }  # fmt: skip


def _codes(labels: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """A unit code per label: one-hot while the labels fit, else random."""
    if int(labels.max()) < dim:
        return np.eye(dim)[labels]
    rng = np.random.default_rng(seed)
    table = rng.standard_normal((int(labels.max()) + 1, dim))
    table /= np.linalg.norm(table, axis=1, keepdims=True)
    return table[labels]


def oracle_field(objects: np.ndarray, parts: np.ndarray, scales=(0.05, 20.0)) -> ff.Field:
    """Features [object code | part code], unit; the gate keeps the part code below 1 m."""
    features = np.concatenate([_codes(objects, 16, 1), _codes(parts, 16, 2)], axis=1)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    n = len(objects)
    return ff.Field(
        features.astype(np.float32), _gate(), np.full(n, 10, np.int32),
        np.geomspace(*scales, 200).astype(np.float32),
    )  # fmt: skip


# --------------------------------------------------------------------------- cameras


def test_placed_viewmat_sees_the_placed_point_where_colmap_saw_the_original():
    rng = np.random.default_rng(0)
    q = rng.standard_normal(4)
    t = rng.standard_normal(3)
    angle = 0.7
    rp = np.array(
        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    )
    placement = {"scale": 2.5, "rotation": rp.tolist(), "translation": [3.0, -1.0, 0.5]}
    x = rng.standard_normal(3) + np.array([0, 0, 6.0])
    cam = ff.quat_rotation(q) @ x + t
    placed = 2.5 * rp @ x + np.array([3.0, -1.0, 0.5])
    view = ff.placed_viewmat(q, t, placement)
    cam_placed = view[:3, :3] @ placed + view[:3, 3]
    # The same ray: camera coordinates scaled by the placement's scale, nothing else.
    assert np.allclose(cam_placed, 2.5 * cam, atol=1e-9)


def _write_colmap(folder: Path, camera: tuple, images: list[tuple]) -> None:
    """cameras.bin, images.bin and an empty points3D.bin in COLMAP's binary layout."""
    import struct

    folder.mkdir(parents=True, exist_ok=True)
    model_id, width, height, params = camera
    (folder / "cameras.bin").write_bytes(
        struct.pack("<Q", 1)
        + struct.pack("<IiQQ", 1, model_id, width, height)
        + struct.pack(f"<{len(params)}d", *params)
    )
    out = struct.pack("<Q", len(images))
    for k, (name, q, t) in enumerate(images):
        out += struct.pack("<IdddddddI", k + 1, *q, *t, 1) + name.encode() + b"\0"
        out += struct.pack("<Q", 0)
    (folder / "images.bin").write_bytes(out)
    (folder / "points3D.bin").write_bytes(struct.pack("<Q", 0))


def test_colmap_views_project_a_placed_point_where_the_photo_shows_it(tmp_path: Path):
    import cv2

    f, cx, cy = 150.0, 96.0, 52.0
    q, t = (0.9, 0.1, -0.3, 0.2), (0.3, -0.2, 4.0)
    _write_colmap(tmp_path / "poses", (2, 200, 100, (f, cx, cy, 0.0)), [("a.jpg", q, t)])
    (tmp_path / "frames").mkdir()
    cv2.imwrite(str(tmp_path / "frames" / "a.jpg"), np.zeros((100, 200, 3), np.uint8))
    placement = {"scale": 1.7, "rotation": np.eye(3).tolist(), "translation": [1.0, 2.0, 0.0]}
    views = ff.colmap_views(tmp_path / "frames", tmp_path / "poses", placement, side=100)
    assert len(views) == 1 and (views[0].width, views[0].height) == (100, 50)
    x = np.array([0.2, -0.1, 0.5])
    cam = ff.quat_rotation(q) @ x + np.asarray(t)
    pixel = np.array([f * cam[0] / cam[2] + cx, f * cam[1] / cam[2] + cy]) * 0.5
    placed = 1.7 * x + np.array([1.0, 2.0, 0.0])
    p = views[0].viewmat[:3, :3] @ placed + views[0].viewmat[:3, 3]
    assert np.allclose(views[0].K[:2, :2] @ (p[:2] / p[2]) + views[0].K[:2, 2], pixel)


def test_intrinsics_read_colmap_models():
    K, dist = ff.intrinsics("SIMPLE_RADIAL", [500.0, 320.0, 240.0, 0.01])
    assert K[0, 0] == K[1, 1] == 500.0 and K[0, 2] == 320.0 and dist[0] == 0.01
    K, dist = ff.intrinsics("PINHOLE", [500.0, 510.0, 320.0, 240.0])
    assert K[1, 1] == 510.0 and not dist.any()


def test_camera_view_matches_the_renderers_convention():
    from splat_render import Camera

    camera = Camera.look_at([3.0, 1.0, 2.0], [0.0, 0.0, 0.0], width=64, height=48)
    view = ff.camera_view(camera, np.zeros((48, 64, 3), np.uint8))
    point = np.array([0.2, -0.1, 0.3])
    uv, _ = camera.project(point[None])
    p = view.viewmat[:3, :3] @ point + view.viewmat[:3, 3]
    assert np.allclose(view.K[:2, :2] @ (p[:2] / p[2]) + view.K[:2, 2], uv[0])
    assert np.allclose(view.centre(), camera.centre)


# ----------------------------------------------------------------------------- masks


def test_mask_scale_is_the_robust_diameter_under_the_mask():
    rng = np.random.default_rng(0)
    h, w = 40, 60
    points = np.zeros((h, w, 3))
    points[..., 0] = rng.uniform(0, 2.0, (h, w))  # a 2 m by 1 m patch
    points[..., 1] = rng.uniform(0, 1.0, (h, w))
    valid = np.ones((h, w), bool)
    whole = np.ones((h, w), bool)
    half = np.zeros((h, w), bool)
    half[:, :30] = True
    holes = np.zeros((h, w), bool)
    holes[:5, :5] = True
    valid_holes = valid.copy()
    valid_holes[:5, :4] = False  # most of it has no depth
    scales = ff.mask_scales(np.stack([whole, half, holes]), points, valid_holes)
    assert scales[0] == pytest.approx(math.hypot(2.0, 1.0), rel=0.1)
    assert np.isfinite(scales[1])
    assert np.isnan(scales[2])


def test_supervise_packs_membership_at_the_training_size():
    view = ff.TrainView("v", np.zeros((40, 80, 3), np.uint8), np.eye(3), np.eye(4))
    view.K = np.array([[80.0, 0, 40], [0, 80.0, 20], [0, 0, 1]])
    left = np.zeros((40, 80), bool)
    left[:, :40] = True
    masks = [segment_scene.Mask(left, 0, 1.0), segment_scene.Mask(~left, 0, 1.0)]
    depth = np.full((40, 80), 2.0)
    sup = ff.supervise(view, masks, depth, np.ones((40, 80)), side=20)
    assert (sup.width, sup.height) == (20, 10)
    member = sup.membership(np.array([0, 19]))
    assert member.tolist() == [[True, False], [False, True]]
    assert np.all(sup.scales > 0)


# ------------------------------------------------------------------------------ tree


def _two_blobs(gap: float = 0.0):
    rng = np.random.default_rng(1)
    a = rng.uniform([-1.0, -0.5, 0.2], [0.0, 0.5, 1.0], (3000, 3))
    b = rng.uniform([gap, -0.5, 0.2], [1.0 + gap, 0.5, 1.0], (3000, 3))
    return np.concatenate([a, b]), np.r_[np.zeros(3000, int), np.ones(3000, int)]


def test_partition_splits_touching_blobs_by_their_features_only():
    pos, truth = _two_blobs()
    keep = np.ones(len(pos), bool)
    graph = ff.knn_graph(pos, keep)
    node = np.zeros(len(pos), np.int64)
    gate = _gate()
    apart = np.concatenate([_codes(truth, 16, 1), _codes(np.zeros_like(truth), 16, 2)], axis=1)
    pieces = ff.partition(node, np.array([2.0]), apart, gate, graph)
    assert len(np.unique(pieces)) == 2
    assert all(len(np.unique(pieces[truth == t])) == 1 for t in (0, 1))
    together = np.concatenate([_codes(np.zeros_like(truth), 16, 1), _codes(truth, 16, 2)], axis=1)
    # Grouped at a large scale (the part code gated off), parted at a small one.
    assert len(np.unique(ff.partition(node, np.array([5.0]), together, gate, graph))) == 1
    assert len(np.unique(ff.partition(node, np.array([0.1]), together, gate, graph))) == 2


def test_tree_on_the_yard_finds_its_objects_and_parts_above_the_ground():
    data, klass, inst, instances = sy.generate_yard(11)
    splats = Splats(
        data["position"].astype(np.float64), data["quat_wxyz"], np.exp(data["log_scale"]),
        data["rgb"], 1.0 / (1.0 + np.exp(-data["opacity_logit"])),
    )  # fmt: skip
    rows = np.arange(len(klass))
    objects, parts = segment_scene.truth_levels(
        {"class": klass, "instance": inst, "instances": instances}, rows, splats.colours
    )
    # The ground is one thing at large scales (SAM masks it whole), lawn and path alike.
    objects = np.where(np.isin(klass, (3, 4)), len(instances) + 3, objects)
    ground = ff.ground_layer(splats)
    field_ = oracle_field(objects, parts)
    tree = ff.build_tree(splats, field_, ground)
    # Each object of the truth (trees, shrubs, snags, the building) is mostly one top node.
    up = np.arange(tree.parent.size + 1)
    for i in range(1, tree.parent.size + 1):
        p = tree.parent[i - 1]
        while p:
            up[i] = p
            p = tree.parent[p - 1]
    top = up[tree.leaf]
    found = 0
    for k in range(len(instances)):
        mine = (inst == k) & (top > 0)
        if mine.sum() < 50:
            continue
        values, counts = np.unique(top[mine], return_counts=True)
        best = values[counts.argmax()]
        share = counts.max() / mine.sum()
        purity = (inst[top == best] == k).mean()
        assert share > 0.8, (k, share)
        assert purity > 0.9, (k, purity)
        found += 1
    assert found >= len(instances) - 1
    # The objects' bases in the ground layer came back to them, and almost no ground with them.
    assert tree.stats["groundRegionsClaimed"] > 0
    objects_ids = [j + 1 for j, kind in enumerate(tree.kind) if kind == "object"]
    assert np.isin(top[np.isin(klass, (3, 4))], objects_ids).mean() < 0.02
    # Parts: some object has children, and every child's splats are inside its parent.
    assert (tree.level > 0).any()
    # The ground left over is the ground classes.
    assert tree.ground.any()
    assert np.isin(klass[tree.ground], (3, 4)).mean() > 0.9  # grass/low, ground


def test_pieces_the_graph_left_apart_join_when_the_field_says_one_thing():
    """A table top over its leg with a shadowed gap between them (no graph edge crosses it):
    one object, with the two as parts; a box beside it with its own code stays apart."""
    rng = np.random.default_rng(4)
    xy = np.stack(np.meshgrid(np.arange(-3, 3, 0.03), np.arange(-2, 2, 0.03)), -1).reshape(-1, 2)
    floor = np.c_[xy, rng.normal(0, 0.002, len(xy))]
    leg = rng.uniform([-0.2, -0.2, 0.05], [0.2, 0.2, 0.7], (4000, 3))
    top = rng.uniform([-0.8, -0.8, 0.85], [0.8, 0.8, 0.95], (6000, 3))
    box = rng.uniform([1.8, -0.3, 0.05], [2.4, 0.3, 0.6], (3000, 3))
    pos = np.concatenate([floor, leg, top, box])
    n = len(floor)
    objects = np.r_[np.zeros(n, int), np.ones(4000 + 6000, int), np.full(3000, 2)]
    parts = np.r_[np.zeros(n, int), np.full(4000, 1), np.full(6000, 2), np.full(3000, 3)]
    splats = _splats(pos, scale=0.015)
    ground = ff.ground_layer(splats)
    tree = ff.build_tree(splats, oracle_field(objects, parts), ground)
    assert tree.stats["adjacentJoins"] >= 1
    up = np.arange(tree.parent.size + 1)
    for i in range(1, tree.parent.size + 1):
        p = int(tree.parent[i - 1])
        while p:
            up[i] = p
            p = int(tree.parent[p - 1])
    held = up[tree.leaf[n : n + 10000]]
    values, counts = np.unique(held[held > 0], return_counts=True)
    assert values.size == 1 and counts[0] > 0.95 * held.size  # one object: leg and top
    assert not np.isin(up[tree.leaf[n + 10000 :]], values).any()  # the box apart
    children = [j + 1 for j in range(tree.parent.size) if tree.parent[j] == values[0]]
    assert len(children) == 2


def test_a_flange_the_height_filter_calls_ground_goes_to_its_object():
    rng = np.random.default_rng(3)
    xy = np.stack(np.meshgrid(np.arange(-2, 2, 0.02), np.arange(-2, 2, 0.02)), -1).reshape(-1, 2)
    ground_pts = np.c_[xy, rng.normal(0, 0.002, len(xy))]
    r = np.hypot(ground_pts[:, 0], ground_pts[:, 1])
    flange = r < 0.45  # a disc lying on the ground, part of the object above it
    ground_pts[flange, 2] = 0.006
    body = rng.uniform([-0.35, -0.35, 0.01], [0.35, 0.35, 0.8], (30000, 3))
    pos = np.concatenate([ground_pts, body])
    truth = np.r_[flange.astype(int), np.ones(len(body), int)]
    splats = _splats(pos)
    ground = ff.ground_layer(splats)
    assert ground.is_ground[: len(ground_pts)][flange].mean() > 0.9  # geometry: ground
    field_ = oracle_field(truth, truth)
    tree = ff.build_tree(splats, field_, ground)
    objects = [j + 1 for j, kind in enumerate(tree.kind) if kind == "object" and tree.level[j] == 0]
    assert len(objects) == 1
    up = np.arange(tree.parent.size + 1)
    for i in range(1, tree.parent.size + 1):
        p = int(tree.parent[i - 1])
        while p:
            up[i] = p
            p = int(tree.parent[p - 1])
    in_object = up[tree.leaf] == objects[0]
    assert in_object[: len(ground_pts)][flange].mean() > 0.9
    assert in_object[: len(ground_pts)][~flange].mean() < 0.01


def test_low_pieces_that_read_as_the_ground_are_ground_cover():
    """A tuft and a gourd, both just above the ground layer: the field says the tuft is the
    lawn beside it and the gourd is not."""
    rng = np.random.default_rng(5)
    xy = np.stack(np.meshgrid(np.arange(-2, 2, 0.02), np.arange(-2, 2, 0.02)), -1).reshape(-1, 2)
    lawn = np.c_[xy, rng.normal(0, 0.003, len(xy))]
    tuft = rng.uniform([-1.2, -0.1, 0.06], [-0.9, 0.2, 0.14], (1500, 3))
    gourd = rng.uniform([0.8, -0.15, 0.06], [1.1, 0.15, 0.3], (3000, 3))
    pos = np.concatenate([lawn, tuft, gourd])
    truth = np.r_[np.zeros(len(lawn), int), np.zeros(len(tuft), int), np.ones(len(gourd), int)]
    splats = _splats(pos)
    ground = ff.ground_layer(splats)
    tree = ff.build_tree(splats, oracle_field(truth, truth), ground)
    assert tree.stats["groundLikePieces"] >= 1
    n_lawn, n_tuft = len(lawn), len(tuft)
    assert tree.ground[n_lawn : n_lawn + n_tuft].mean() > 0.9
    assert (tree.leaf[n_lawn + n_tuft :] > 0).mean() > 0.9


def test_ground_records_write_the_shared_ground_schema():
    """Things first; then one top-level instance per cover class (kind ground, category
    ground, its `cover` and `name`), with its connected regions as children when it has more
    than one; a low thing described as ground cover, or not described at all, joins the
    ground (candidate A's stuff rule); a tall one not described stays a thing."""
    import segment_ground_first as sgf

    rng = np.random.default_rng(2)
    box = rng.uniform([0, 0, 0.3], [0.5, 0.5, 1.0], (500, 3))
    tuft = rng.uniform([2.0, 0.2, 0.0], [2.1, 0.3, 0.02], (60, 3))
    speck = rng.uniform([1.5, 0.5, 0.0], [1.55, 0.55, 0.08], (20, 3))
    post = rng.uniform([2.5, 0.5, 0.0], [2.55, 0.55, 0.8], (40, 3))
    xy = np.stack(np.meshgrid(np.arange(0, 3, 0.05), np.arange(0, 1, 0.05)), -1).reshape(-1, 2)
    floor = np.c_[xy, np.zeros(len(xy))]
    pos = np.concatenate([box, tuft, speck, post, floor])
    n_box, n_tuft, n_speck, n_post = len(box), len(tuft), len(speck), len(post)
    leaf = np.r_[
        np.ones(n_box, int), np.full(n_tuft, 2), np.full(n_speck, 3), np.full(n_post, 4),
        np.zeros(len(floor), int),
    ]  # fmt: skip
    things = n_box + n_tuft + n_speck + n_post
    ground = np.r_[np.zeros(things, bool), np.ones(len(floor), bool)]
    tree = ff.Tree(
        leaf, np.zeros(4, np.int64), np.zeros(4, np.int64), np.ones(4), ["object"] * 4,
        ground=ground,
    )  # fmt: skip
    # Ground cells along x: grass, dirt, grass -- two grass regions, apart.
    third = np.minimum((floor[:, 0] // 1.0).astype(int), 2)
    classes, _ = sgf.cover_classes()
    names = [c.id for c in classes]
    klass = np.array([names.index("grass"), names.index("dirt"), names.index("grass")])
    cover = ff.GroundCover(
        classes, third, klass, np.array([0.9, 0.7, 0.8]), np.array([0, 1, 2]),
        np.ones((len(classes), 8)),
    )  # fmt: skip
    dim = 8

    def instance(i: int, category: str, described: bool = True) -> segment_scene.Instance:
        return segment_scene.Instance(
            id=i, parent=None, level=0, splats=10, bounds_min=np.zeros(3),
            bounds_max=np.ones(3), centroid=np.zeros(3), views=3, embedding=np.zeros(dim),
            tags=[{"label": "x", "score": 0.5}] if described else [],
            properties={name: 0.1 for name in segment_scene.PROPERTY_PROMPTS},
            behaviour="static", category=category,
        )  # fmt: skip

    height = pos[:, 2]
    described = [
        instance(1, "furniture"), instance(2, "grass"), instance(3, "household", False),
        instance(4, "household", False),
    ]  # fmt: skip
    out = ff.ground_records(tree, described, pos, cover, dim, 0.05, height)
    by_id = {i.id: i for i in out.instances}
    assert [i.id for i in out.instances] == list(range(1, len(out.instances) + 1))
    assert out.extra[1]["kind"] == "thing" and by_id[1].parent is None
    # The tuft and the speck are ground; the box and the post are things.
    assert sum(1 for e in out.extra.values() if e["kind"] == "thing") == 2
    assert out.extra[2]["kind"] == "thing" and (out.leaf[things - n_post : things] == 2).all()
    tops = [i for i in out.instances if i.parent is None and out.extra[i.id]["kind"] == "ground"]
    assert sorted(out.extra[i.id]["cover"] for i in tops) == ["dirt", "grass"]
    for i in tops:
        assert i.category == "ground" and out.extra[i.id]["nameSource"] == "ground-cover"
    grass = next(i for i in tops if out.extra[i.id]["cover"] == "grass")
    dirt = next(i for i in tops if out.extra[i.id]["cover"] == "dirt")
    children = [i for i in out.instances if i.parent == grass.id]
    assert len(children) == 2 and grass.splats == 0
    assert all(out.extra[c.id]["cover"] == "grass" for c in children)
    assert not [i for i in out.instances if i.parent == dirt.id] and dirt.splats > 0
    assert set(np.unique(out.leaf)) <= set(by_id)
    lows = out.leaf[n_box : n_box + n_tuft + n_speck]  # the tuft's and speck's splats
    assert all(out.extra[int(i)]["kind"] == "ground" for i in np.unique(lows))


def test_granularities_follow_the_tree():
    parent = np.array([0, 1, 1, 2])
    level = np.array([0, 1, 1, 2])
    leaf = np.array([4, 3, 2, 0])
    layers = ff.granularities(parent, level, leaf)
    assert layers["objects"].tolist() == [1, 1, 1, 0]
    assert layers["parts"].tolist() == [2, 3, 2, 0]
    assert layers["leaves"].tolist() == [4, 3, 2, 0]


def test_field_round_trips(tmp_path: Path):
    field_ = oracle_field(np.array([0, 1, 1]), np.array([0, 1, 2]))
    field_.stats = {"steps": 3}
    field_.save(tmp_path / "field.npz")
    back = ff.Field.load(tmp_path / "field.npz")
    assert np.allclose(back.features, field_.features, atol=1e-3)
    assert back.stats == {"steps": 3}
    assert np.allclose(ff.gate_values(back.gate, [0.1, 10]), ff.gate_values(field_.gate, [0.1, 10]))


# --------------------------------------------------------------------------- training


def test_the_loss_teaches_the_gate_to_group_large_and_part_small():
    """Two boxes side by side, seen from four sides; each view's masks: each box alone
    (small) and both together (large). After training, features read at a large scale say
    "one thing" and at a small scale "two"."""
    torch = pytest.importorskip("torch")
    from splat_render import Camera

    rng = np.random.default_rng(0)
    a = rng.uniform([-0.6, -0.25, 0.0], [-0.05, 0.25, 0.5], (400, 3))
    b = rng.uniform([0.05, -0.25, 0.0], [0.6, 0.25, 0.5], (400, 3))
    pos = np.concatenate([a, b])
    splats = _splats(pos, scale=0.03, opacity=0.8)
    scene = ff.Scene.of(splats, "cpu")
    truth = np.r_[np.zeros(400, int), np.ones(400, int)]
    supervision = []
    for azimuth in (0.3, 1.9, 3.4, 4.9):
        eye = np.array([3 * math.cos(azimuth), 3 * math.sin(azimuth), 1.2])
        camera = Camera.look_at(eye, [0, 0, 0.25], width=40, height=30, fov_deg=45)
        view = ff.camera_view(camera, np.zeros((30, 40, 3), np.uint8))
        viewmat, K = ff.view_tensors(view, 40, 30, "cpu")
        with torch.no_grad():
            one_hot = torch.as_tensor(np.eye(2)[truth], dtype=torch.float32)
            label, alpha, depth = ff.dense_rasterize(scene, one_hot, viewmat, K, 40, 30, depth=True)
        owner = label.argmax(dim=-1).numpy()
        covered = alpha.numpy() > 0.5
        masks = [
            segment_scene.Mask(covered & (owner == 0), 1, 1.0),
            segment_scene.Mask(covered & (owner == 1), 1, 1.0),
            segment_scene.Mask(covered, 0, 1.0),
        ]
        supervision.append(ff.supervise(view, masks, depth.numpy(), alpha.numpy(), side=40))
    torch.set_num_threads(1)
    trained = ff.train_field(
        splats, supervision, dim=8, steps=800, pixels=256, device="cpu",
        rasterize=ff.dense_rasterize,
    )  # fmt: skip
    lo, hi = trained.scale_range()
    graph = ff.knn_graph(pos, np.ones(len(pos), bool))
    node = np.zeros(len(pos), np.int64)
    whole = ff.partition(node, np.array([hi]), trained.features, trained.gate, graph)
    parts = ff.partition(node, np.array([lo]), trained.features, trained.gate, graph)
    assert len(np.unique(whole)) == 1, trained.stats["history"][-1]
    assert len(np.unique(parts)) == 2, trained.stats["history"][-1]
    assert all(len(np.unique(parts[truth == t])) == 1 for t in (0, 1))


def test_finish_writes_instances_bound_to_every_tile(tmp_path: Path):
    """The CPU half end to end on the committed yard tiles, with a field of random
    features (the plumbing: tree, views, descriptions, binding, overview)."""
    tileset = Path(__file__).resolve().parents[3] / "data/tiles/synthetic-yard/splat/tileset.json"
    from splat_render import load_tileset

    n = len(load_tileset(tileset))
    rng = np.random.default_rng(0)
    features = rng.standard_normal((n, 16)).astype(np.float32)
    field_ = ff.Field(
        features / np.linalg.norm(features, axis=1, keepdims=True), _gate(16, 8),
        np.full(n, 5, np.int32), np.geomspace(0.1, 10, 50).astype(np.float32),
    )  # fmt: skip
    field_.save(tmp_path / "field.npz")
    out = tmp_path / "out"
    assert ff.main([
        "finish", str(tileset), "--field", str(tmp_path / "field.npz"), "--out", str(out),
        "--views", "2", "--cpus", "3",
    ]) == 0  # fmt: skip
    document = json.loads((out / "instances.json").read_text())
    records = document["instances"]
    assert [r["id"] for r in records] == list(range(1, len(records) + 1))
    assert all(r["parent"] is None or r["parent"] < r["id"] for r in records)
    ground = [r for r in records if r["kind"] == "ground"]
    assert ground and all(r["category"] == "ground" and r.get("cover") for r in ground)
    assert all(r["parent"] is None for r in ground if r["level"] == 0)
    assert document["variant"]["name"] == "feature-fields" and document["variant"]["label"]
    emb = (out / "instances.emb").stat().st_size
    assert emb == len(records) * document["embedding"]["dim"] * 2
    from rig_tiles import tile_uris

    tiles = json.loads(tileset.read_text())
    assert len(document["tiles"]) == len(set(tile_uris(tiles)))
    assert (out / "overview.png").stat().st_size > 1000
    summary = json.loads((out / "summary.json").read_text())
    assert summary["assignedShare"] > 0.9


def test_train_then_finish_on_a_small_tileset(tmp_path: Path):
    """Both halves of the CLI on a few thousand of the yard's splats, rendered views (no
    photos), colour regions for masks and the reference rasterizer: the plumbing a GPU run
    goes through."""
    torch = pytest.importorskip("torch")
    import splat_tiles
    from splat_render import save_ply

    torch.set_num_threads(1)
    data, klass, _, _ = sy.generate_yard(11)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(klass), 1500, replace=False))
    splats = Splats(
        data["position"][rows].astype(np.float64), data["quat_wxyz"][rows],
        np.exp(data["log_scale"][rows]), data["rgb"][rows],
        1.0 / (1.0 + np.exp(-data["opacity_logit"][rows])),
    )  # fmt: skip
    save_ply(tmp_path / "small.ply", splats)
    splat_tiles.convert(tmp_path / "small.ply", tmp_path / "tiles", 46.0, -123.0, 0.0,
                        tile_gaussians=600)  # fmt: skip
    tileset = tmp_path / "tiles" / "tileset.json"
    assert ff.main([
        "train", str(tileset), "--out", str(tmp_path / "field.npz"), "--steps", "4",
        "--pixels", "64", "--masks", "feature_fields:ColourMasks", "--render-views", "3",
        "--render-side", "48", "--train-side", "32", "--summary", str(tmp_path / "train.json"),
        "--sheet", str(tmp_path / "sheet.jpg"),
    ]) == 0  # fmt: skip
    trained = ff.Field.load(tmp_path / "field.npz")
    from splat_render import load_tileset

    assert trained.features.shape[0] == len(load_tileset(tileset))
    assert json.loads((tmp_path / "train.json").read_text())["views"] == "rendered"
    assert (tmp_path / "sheet.jpg").stat().st_size > 0
    assert ff.main([
        "finish", str(tileset), "--field", str(tmp_path / "field.npz"),
        "--out", str(tmp_path / "out"), "--views", "2", "--cpus", "3",
    ]) == 0  # fmt: skip
    assert (tmp_path / "out" / "instances.json").exists()


def test_cli_parses_both_commands(tmp_path: Path):
    with pytest.raises(SystemExit):
        ff.main(["train", str(tmp_path / "tileset.json"), "--out", "x", "--frames", "f"])
    assert ff.VARIANT == "feature-fields" and ff.VARIANT_LABEL.startswith("B")
