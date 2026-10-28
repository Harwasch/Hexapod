import numpy as np

from living_weather_forest import coarse_nodes, layout


def test_forest_is_repeatable_jittered_and_within_limits():
    a, b = layout(), layout()
    assert a == b and len(a) == 36
    assert all(0.9 <= p["scale"] <= 1.1 for p in a)
    assert np.ptp([p["x"] for p in a]) > 115
    assert len({p["yaw"] for p in a}) == 36
    assert 36 * 40000 <= 3000000


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
