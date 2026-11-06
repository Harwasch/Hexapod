import numpy as np

from living_weather_detail import has_segmented_vegetation, leaf_uris, measure


def test_only_terminal_tiles_are_counted():
    assert leaf_uris(
        {"root": {"content": {"uri": "parent.glb"}, "children": [{"content": {"uri": "leaf.glb"}}]}}
    ) == ["leaf.glb"]


def test_separable_leaf_scale_pieces_require_skeleton_contact():
    # Two internally connected pieces, separated by 20 cm; 16 splats per piece.
    p = np.array(
        [
            [x * 0.004, y * 0.004 + offset, 0.5]
            for offset in [0, 0.2]
            for x in range(4)
            for y in range(4)
        ]
    )
    rgb = np.tile([0.1, 0.6, 0.1], (len(p), 1))
    scales = np.full_like(p, 0.006)
    quat = np.tile([0, 0, 0, 1], (len(p), 1))
    alpha = np.ones(len(p))
    nodes = [{"parent": -1, "position": [0, 0, 0]}, {"parent": 0, "position": [0, 0, 1]}]
    r = measure(p, rgb, scales, quat, alpha, nodes, True)
    assert r["separableCrownShare"] == 1
    assert r["attachedCrownShare"] == 0.5
    assert r["score"] == 0.5
    assert r["tier"] == "B" and not r["leafBindings"]
    assert measure(p, rgb, scales, quat, alpha, [], True)["tier"] == "A"
    assert measure(p, rgb, scales, quat, alpha, [], False)["tier"] == "off"


def test_vegetation_scores_never_override_ground_wood_or_rigidity():
    assert not has_segmented_vegetation(
        [{"category": "ground", "properties": {"vegetation": 0.99, "rigid": 0.1}}]
    )
    assert not has_segmented_vegetation(
        [{"category": "trees", "properties": {"vegetation": 0.99, "rigid": 0.9}}]
    )
    assert has_segmented_vegetation(
        [{"category": "shrubs", "properties": {"vegetation": 0.99, "rigid": 0.3}}]
    )
