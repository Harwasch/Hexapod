import numpy as np

from living_weather_forest import coarse_binding, coarse_nodes, crown_measurement, layout


def test_forest_is_repeatable_jittered_and_within_limits():
    a, b = layout(), layout()
    assert a == b and len(a) == 80
    assert all(0.9 <= p["scale"] <= 1.1 for p in a)
    assert np.ptp([p["x"] for p in a]) > 30
    assert len({p["yaw"] for p in a}) == 80
    assert 80 * 30000 <= 3000000


def test_coarse_rig_keeps_connectivity_and_maps_every_original_joint():
    nodes = [
        {"parent": i - 1, "position": [0, i * 0.1, i * 0.2], "band": "trunk" if i < 3 else "branch"}
        for i in range(50)
    ]
    out, nearest = coarse_nodes(nodes, 10)
    assert len(out) == 10
    assert all(n["parent"] < i for i, n in enumerate(out))
    assert np.all(nearest >= 0) and np.all(nearest < len(out))
    assert out[1]["band"] == "trunk"


def test_crowns_touch_without_overlapping_trunks():
    a = layout()
    xy = np.array([[p["x"], p["y"]] for p in a])
    distance = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = distance.min(axis=1)
    assert np.all(nearest >= 0.8 * 3.889033203125)
    assert np.all(nearest <= 3.889033203125)
    # 0.4 m is a conservative assumed trunk diameter, not a scan measurement.
    assert distance.min() > 0.4 * 1.1
    assert {p["mirror"] for p in a} == {True, False}


def test_crown_measurement_excludes_ground_and_sparse_scan_tails():
    angle = np.linspace(0, 2 * np.pi, 10000)
    crown = np.column_stack([2 * np.cos(angle), 3 * np.sin(angle), np.full(len(angle), 4)])
    points = np.concatenate([crown, [[99, 99, 0], [100, 100, 4]]])
    result = crown_measurement(points)
    assert abs(result["diameterM"] - 5) < 0.02
    assert result["splatsMeasured"] == len(points)


def test_coarse_lod_does_not_pin_unretained_crown_branches_to_the_trunk():
    original = [
        {"parent": -1, "band": "trunk", "position": [0, 0, 0]},
        {"parent": 0, "band": "trunk", "position": [0, 0, 1]},
        {"parent": 1, "band": "branch", "position": [0.5, 0, 2]},
        {"parent": 2, "band": "branch", "position": [1, 0, 3]},
        {"parent": 3, "band": "branch", "position": [1.5, 0, 4]},
    ]
    nodes, _ = coarse_nodes(original, 3)
    points = np.array([[0, 0, 0.5], [0.5, 0, 2], [1, 0, 3], [1.5, 0, 4]])
    binding = coarse_binding(points, original, nodes)
    assert np.all(binding[0] == 0)
    assert all(nodes[owner - 1]["band"] != "trunk" for owner in binding[1:, 0])
