"""The supported region as a voxel mask: any shape the well-supported data makes."""

from __future__ import annotations

import numpy as np
import pytest

import support_mask
import training


def _l_shape_and_island(seed: int = 0) -> np.ndarray:
    """Two walls meeting in an L, plus a separate object 3 m away: no sphere or box fits
    these without taking in the empty space between them."""
    rng = np.random.default_rng(seed)
    wall_a = np.column_stack(
        [rng.uniform(0, 4, 20000), rng.uniform(0, 0.1, 20000), rng.uniform(0, 2, 20000)]
    )
    wall_b = np.column_stack(
        [rng.uniform(0, 0.1, 20000), rng.uniform(0, 4, 20000), rng.uniform(0, 2, 20000)]
    )
    island = rng.normal([3.5, 3.5, 0.5], 0.15, size=(2000, 3))
    return np.concatenate([wall_a, wall_b, island])


def test_the_mask_follows_the_data_not_a_bounding_shape() -> None:
    mask = support_mask.build(_l_shape_and_island())
    assert mask is not None
    # On the walls and on the island.
    assert mask.contains([[2.0, 0.05, 1.0], [0.05, 2.0, 1.0], [3.5, 3.5, 0.5]]).all()
    # The empty middle of the L, which any sphere or box round the data would include.
    assert not mask.contains([[2.0, 2.0, 1.0], [1.5, 1.5, 0.5]]).any()
    # Far away, and nowhere.
    assert not mask.contains([[20.0, 20.0, 20.0], [np.nan, 0.0, 0.0]]).any()


def test_the_mask_round_trips_through_a_job_parameter() -> None:
    mask = support_mask.build(_l_shape_and_island())
    assert mask is not None
    encoded = mask.to_dict()
    assert len(str(encoded)) < 64_000
    back = support_mask.SupportMask.parse(encoded)
    assert back is not None
    assert back.dims == mask.dims and back.voxels == mask.voxels
    probe = _l_shape_and_island(seed=1)
    assert (back.contains(probe) == mask.contains(probe)).all()


def test_a_malformed_mask_is_refused_by_name() -> None:
    assert support_mask.SupportMask.parse(None) is None
    with pytest.raises(ValueError, match="support_mask"):
        support_mask.SupportMask.parse({"kind": "sphere"})
    mask = support_mask.build(_l_shape_and_island())
    assert mask is not None
    broken = {**mask.to_dict(), "dims": [mask.dims[0], mask.dims[1], 10_000]}
    with pytest.raises(ValueError, match="support_mask"):
        support_mask.SupportMask.parse(broken)


def test_too_little_support_gives_no_mask() -> None:
    assert support_mask.build(np.zeros((10, 3))) is None


def test_a_trained_splat_is_cropped_to_the_mask() -> None:
    mask = support_mask.build(_l_shape_and_island())
    assert mask is not None
    xyz = np.asarray([[2.0, 0.05, 1.0], [2.0, 2.0, 1.0], [3.5, 3.5, 0.5]], dtype=np.float32)
    columns = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "opacity": np.zeros(3, np.float32)}
    cropped, kept = training.crop_splat(columns, mask)
    assert kept == 2
    assert cropped["x"].tolist() == pytest.approx([2.0, 3.5])
