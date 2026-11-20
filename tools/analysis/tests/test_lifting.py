from typing import Any

import numpy as np
import pytest

from hex_analysis.lifting import bounds, lift_depth


def lift(
    depth: Any, mask: Any = None, k: Any = None, transform: Any = None, size: Any = None
) -> Any:
    depth = np.asarray(depth, dtype=np.float64)
    return lift_depth(
        depth,
        np.ones(depth.shape, dtype=np.bool_) if mask is None else mask,
        np.eye(3) if k is None else k,
        np.eye(4) if transform is None else transform,
        (depth.shape[1], depth.shape[0]) if size is None else size,
    )


def test_known_plane_signs_and_axial_not_radial_depth() -> None:
    points = lift([[2, 2], [2, 2]])
    np.testing.assert_array_equal(points, [[0, 0, -2], [2, 0, -2], [0, -2, -2], [2, -2, -2]])
    measured = bounds(points, 4)
    assert measured is not None
    assert measured.center == (1, -1, -2)
    assert measured.extent == (2, 2, 0)
    assert measured.valid_pixels == 4
    assert np.linalg.norm(points[-1]) > 2


def test_principal_point_focal_lengths_and_depth_scaling() -> None:
    k = np.array([[4.0, 0, 1.5], [0, 8, 3.5], [0, 0, 1]])
    points = lift([[8, 8]], k=k, size=(4, 8))
    np.testing.assert_array_equal(points, [[-2, 0, -8], [2, 0, -8]])


def test_row_major_rotation_and_translation() -> None:
    t = np.array([[0.0, 0, 1, 10], [0, 1, 0, 20], [-1, 0, 0, 30], [0, 0, 0, 1]])
    np.testing.assert_array_equal(lift([[3, 3]], transform=t), [[7, 20, 30], [7, 20, 27]])


def test_holes_and_mask_do_not_create_anchors() -> None:
    points = lift([[0, -1, np.nan, np.inf, 3]], mask=np.array([[True, True, True, True, False]]))
    assert points.shape == (0, 3)
    assert bounds(points, 4) is None
    single = lift([[0, 2]])
    measured = bounds(single, 2)
    assert measured is not None
    assert measured.center == (2, 0, -2)
    assert measured.extent == (0, 0, 0)
    assert measured.valid_pixels == 1
    assert measured.selected_pixels == 2


@pytest.mark.parametrize(
    "bad", [np.zeros((3, 3)), np.eye(3) * np.nan, np.array([[1.0, 0.1, 0], [0, 1, 0], [0, 0, 1]])]
)
def test_invalid_intrinsics(bad: Any) -> None:
    with pytest.raises(ValueError):
        lift([[1]], k=bad)


@pytest.mark.parametrize(
    "bad",
    [
        np.eye(3),
        np.diag([1.0, 1, -1, 1]),
        np.diag([2.0, 1, 1, 1]),
        np.eye(4) * np.nan,
        np.zeros((4, 4)),
    ],
)
def test_nonrigid_or_invalid_transforms(bad: Any) -> None:
    with pytest.raises(ValueError):
        lift([[1]], transform=bad)


def test_shape_and_bounds_validation() -> None:
    with pytest.raises(ValueError):
        lift([[1]], mask=np.array([[1]]))
    with pytest.raises(ValueError):
        lift([[1]], mask=np.array([[True, True]]))
    with pytest.raises(ValueError):
        lift([[1]], size=(0, 1))
    with pytest.raises(ValueError):
        lift_depth(np.empty((0, 0)), np.empty((0, 0), dtype=np.bool_), np.eye(3), np.eye(4), (1, 1))
    with pytest.raises(ValueError):
        bounds(np.array([[1.0, 2, 3]]), 0)
    with pytest.raises(ValueError):
        bounds(np.array([[1.0, 2, np.nan]]), 1)


def test_camera_float32_roundtrip_accepted() -> None:
    angle = 0.7
    c, s = (np.cos(angle), np.sin(angle))
    t = np.array(
        [[c, 0, s, 1], [0, 1, 0, 2], [-s, 0, c, 3], [0, 0, 0, 1]], dtype=np.float32
    ).astype(np.float64)
    point = lift([[2]], transform=t)[0]
    np.testing.assert_allclose(point, t[:3, 3] - 2 * t[:3, 2], rtol=0, atol=0)


def test_projection_roundtrip_on_a_transformed_nonuniform_surface() -> None:
    from hex_analysis.lifting import F64

    k: F64 = np.array([[320.0, 0, 1.2], [0, 240, 0.7], [0, 0, 1]])
    t: F64 = np.array([[0.0, -1, 0, 4], [1, 0, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]])
    depth: F64 = np.array([[1.0, 2, 3], [4, 5, 6]])
    world = lift_depth(depth, np.ones((2, 3), dtype=np.bool_), k, t, (3, 2))
    camera = (world - t[:3, 3]) @ t[:3, :3]
    u = camera[:, 0] / -camera[:, 2] * k[0, 0] + k[0, 2]
    v = -camera[:, 1] / -camera[:, 2] * k[1, 1] + k[1, 2]
    # A conservative 64-operation roundoff bound, scaled by world magnitude and
    # focal length; these test depths are >=1, so division cannot amplify further.
    roundoff = 64 * np.finfo(np.float64).eps * float(np.abs(world).max()) * float(k[0, 0])
    np.testing.assert_allclose(u, [0, 1, 2, 0, 1, 2], rtol=0, atol=roundoff)
    np.testing.assert_allclose(v, [0, 0, 0, 1, 1, 1], rtol=0, atol=roundoff)
    np.testing.assert_array_equal(-camera[:, 2], depth.ravel())
