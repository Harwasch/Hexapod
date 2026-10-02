"""The supported region as a voxel mask: any shape the well-supported data makes, at the
data's own resolution, carried sparsely."""

from __future__ import annotations

import base64
import json
import math
import zlib

import numpy as np
import pytest

import support_mask
import training
from support_mask import SupportMask

#: A pixel footprint for `_l_shape_and_island`: 4 mm a pixel, a phone a couple of metres
#: from two walls.
FOOTPRINT = 0.004


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


def _footprint(points: np.ndarray, value: float = FOOTPRINT, seed: int = 0) -> np.ndarray:
    """A footprint per point that varies as near and far parts of a capture do."""
    rng = np.random.default_rng(seed)
    prints: np.ndarray = rng.uniform(0.6, 1.4, size=(len(points),)) * value
    return prints


def _building(count: int, seed: int = 0) -> np.ndarray:
    """A 50 x 20 x 15 m box's four walls and roof: the large scene a fixed cell count
    made 65 cm voxels of."""
    rng = np.random.default_rng(seed)
    k = count // 5
    return np.concatenate(
        [
            np.column_stack([rng.uniform(0, 50, k), np.zeros(k), rng.uniform(0, 15, k)]),
            np.column_stack([rng.uniform(0, 50, k), np.full(k, 20.0), rng.uniform(0, 15, k)]),
            np.column_stack([np.zeros(k), rng.uniform(0, 20, k), rng.uniform(0, 15, k)]),
            np.column_stack([np.full(k, 50.0), rng.uniform(0, 20, k), rng.uniform(0, 15, k)]),
            np.column_stack([rng.uniform(0, 50, k), rng.uniform(0, 20, k), np.full(k, 15.0)]),
        ]
    )


def _legacy(mask: SupportMask) -> dict[str, object]:
    """The dense format an earlier quality stage wrote, exactly as it wrote it."""
    occupied = np.zeros(int(np.prod(mask.dims)), dtype=bool)
    occupied[mask.indices()] = True
    return {
        "kind": "voxels",
        "frame": "colmap",
        "origin": [round(v, 6) for v in mask.origin],
        "voxel": round(mask.voxel, 9),
        "dims": list(mask.dims),
        "voxels": mask.voxels,
        "bits": base64.b64encode(zlib.compress(np.packbits(occupied).tobytes(), 9)).decode(),
    }


def _runs_payload(values: list[int]) -> str:
    """Hand-made runs data: LEB128 varints, zlib, base64."""
    out = bytearray()
    for value in values:
        while True:
            byte = value & 0x7F
            value >>= 7
            out.append(byte | (0x80 if value else 0))
            if not value:
                break
    return base64.b64encode(zlib.compress(bytes(out))).decode()


# --- the shape ---------------------------------------------------------------------------


@pytest.mark.parametrize("footprint", [True, False], ids=["footprint", "extent"])
def test_the_mask_follows_the_data_not_a_bounding_shape(footprint: bool) -> None:
    points = _l_shape_and_island()
    mask = support_mask.build(points, _footprint(points) if footprint else None)
    assert mask is not None and mask.sizing is not None
    assert mask.sizing.rule == ("footprint" if footprint else "extent")
    # On the walls and on the island.
    assert mask.contains([[2.0, 0.05, 1.0], [0.05, 2.0, 1.0], [3.5, 3.5, 0.5]]).all()
    # The empty middle of the L, which any sphere or box round the data would include.
    assert not mask.contains([[2.0, 2.0, 1.0], [1.5, 1.5, 0.5]]).any()
    # Far away, and nowhere.
    assert not mask.contains([[20.0, 20.0, 20.0], [np.nan, 0.0, 0.0]]).any()
    # Nearly every supporting point is inside its own mask.
    assert mask.contains(points).mean() > 0.97


def test_the_voxel_is_a_multiple_of_the_footprint_not_a_share_of_the_extent() -> None:
    """A densely trained 4 x 2 m wall (a gaussian every 6 mm) seen at 1 mm a pixel."""
    rng = np.random.default_rng(2)
    points = np.column_stack(
        [rng.uniform(0, 4, 200_000), np.zeros(200_000), rng.uniform(0, 2, 200_000)]
    )
    prints = _footprint(points, 0.001)
    mask = support_mask.build(points, prints)
    assert mask is not None and mask.sizing is not None
    sizing = mask.sizing
    assert sizing.footprint == pytest.approx(np.percentile(prints, 90))
    assert sizing.multiple == support_mask.FOOTPRINT_MULTIPLE
    assert sizing.footprint is not None
    assert sizing.bound is None and mask.voxel == sizing.resolution
    # Footprint x multiple, grown by whole sqrt(2) steps only if the points were too sparse.
    assert mask.voxel == pytest.approx(
        sizing.footprint * sizing.multiple * math.sqrt(2) ** sizing.spacing_steps
    )
    # Much finer than the old 80-across grid on this 4 m scene (5 cm).
    assert mask.voxel < 0.5 * 4.0 / 76


def test_the_same_shape_at_25x_the_scale_gets_the_same_voxels() -> None:
    """Sized from the footprint, which scales with the scene, the voxels scale too: the
    same topology, not a coarser one because the scene is bigger."""
    points = _l_shape_and_island()
    prints = _footprint(points)
    small = support_mask.build(points, prints)
    large = support_mask.build(points * 25.0, prints * 25.0)
    assert small is not None and large is not None
    assert small.sizing is not None and large.sizing is not None
    assert large.voxel == pytest.approx(25.0 * small.voxel, rel=1e-9)
    assert large.dims == small.dims
    assert large.sizing.spacing_steps == small.sizing.spacing_steps
    np.testing.assert_array_equal(large.starts, small.starts)
    np.testing.assert_array_equal(large.stops, small.stops)
    probe = _l_shape_and_island(seed=1)
    assert (large.contains(probe * 25.0) == small.contains(probe)).all()


def test_a_bigger_scene_of_the_same_detail_gets_the_same_voxel() -> None:
    """The point of sizing from the footprint: three copies of the L across 45 m, seen
    at the same 4 mm a pixel, keep the one L's voxel. A share of the extent would have
    made them 10x coarser, and the mask of each L 10x looser."""
    one = _l_shape_and_island()
    three = np.concatenate(
        [one, one + np.array([20.0, 0.0, 0.0]), one + np.array([40.0, 5.0, 0.0])]
    )
    alone = support_mask.build(one, _footprint(one))
    many = support_mask.build(three, _footprint(three))
    extent = support_mask.build(three)
    assert alone is not None and many is not None and extent is not None
    assert many.voxel == pytest.approx(alone.voxel, rel=0.01)  # footprints drawn afresh
    assert extent.voxel > 5 * many.voxel
    probe = _l_shape_and_island(seed=1)
    for shift in (np.zeros(3), np.array([20.0, 0.0, 0.0]), np.array([40.0, 5.0, 0.0])):
        agree = many.contains(probe + shift) == alone.contains(probe)
        assert agree.mean() > 0.99


def test_gaussians_sparser_than_the_footprint_implies_grow_the_voxel() -> None:
    """A gaussian budget spread thin: the footprint says 2 mm, the points are ~9 cm apart,
    so footprint x 6 would leave almost every voxel with one gaussian and no support."""
    rng = np.random.default_rng(4)
    wall = np.column_stack([rng.uniform(0, 4, 2000), np.zeros(2000), rng.uniform(0, 2, 2000)])
    mask = support_mask.build(wall, np.full(2000, 0.002))
    assert mask is not None and mask.sizing is not None
    assert mask.sizing.spacing_steps > 0
    assert mask.voxel > 0.012
    # The wall is one connected region, not a sieve.
    probe = np.column_stack(
        [rng.uniform(0.2, 3.8, 1000), np.zeros(1000), rng.uniform(0.2, 1.8, 1000)]
    )
    assert mask.contains(probe).mean() > 0.97


def test_a_footprint_too_coarse_for_the_extent_still_leaves_a_shape() -> None:
    points = _l_shape_and_island()
    mask = support_mask.build(points, np.full(points.shape[0], 1.0))  # 1 m a pixel
    assert mask is not None and mask.sizing is not None
    assert mask.sizing.bound == "min-cells"
    assert mask.voxel == pytest.approx(
        float(np.max(np.percentile(points, 99.5, axis=0) - np.percentile(points, 0.5, axis=0)))
        / support_mask.MIN_CELLS_ACROSS
    )


def test_too_little_support_gives_no_mask() -> None:
    assert support_mask.build(np.zeros((10, 3))) is None
    assert support_mask.build(np.zeros((10, 3)), np.ones(10)) is None
    # Plenty of points, all in one place: no extent to make a grid of.
    assert support_mask.build(np.zeros((100, 3)), np.ones(100)) is None


def test_a_footprint_must_have_one_value_a_point() -> None:
    with pytest.raises(ValueError, match="support_mask"):
        support_mask.build(_l_shape_and_island(), np.ones(3))


# --- size ---------------------------------------------------------------------------------


def test_a_large_scene_at_fine_voxels_fits_a_job_parameter() -> None:
    """A 50 m building at a 1 cm footprint: voxels of centimetres, not the 65 cm a fixed
    80 across made, and still well inside the limit because only runs are stored."""
    points = _building(500_000)
    mask = support_mask.build(points, _footprint(points, 0.01))
    assert mask is not None and mask.sizing is not None
    assert mask.sizing.bound is None
    assert mask.voxel < 0.25 < 50.0 / 76
    encoded = mask.to_dict()
    assert isinstance(encoded["data"], str)
    assert len(encoded["data"]) <= support_mask.MAX_ENCODED_BYTES
    assert len(json.dumps(encoded)) < support_mask.MAX_ENCODED_BYTES
    back = SupportMask.parse(json.loads(json.dumps(encoded)))
    assert back is not None and back.voxels == mask.voxels > 100_000
    assert mask.contains([[25.0, 0.0, 7.0], [25.0, 10.0, 15.0]]).all()
    assert not mask.contains([[25.0, 10.0, 7.0]]).any()  # inside the building: air


@pytest.mark.parametrize(
    ("limits", "bound"),
    [({"max_bytes": 8_000}, "max-bytes"), ({"max_voxels": 50_000}, "max-voxels")],
)
def test_a_cap_coarsens_only_when_it_binds_and_says_so(limits: dict[str, int], bound: str) -> None:
    points = _building(200_000)
    prints = _footprint(points, 0.01)
    free = support_mask.build(points, prints)
    capped = support_mask.build(points, prints, **limits)
    assert free is not None and capped is not None
    assert free.sizing is not None and capped.sizing is not None
    assert free.sizing.bound is None
    assert capped.sizing.bound == bound
    assert capped.sizing.resolution == free.sizing.resolution == free.voxel
    assert capped.voxel > free.voxel
    assert capped.sizing.to_dict()["coarsenedBy"] == pytest.approx(
        capped.voxel / free.voxel, rel=1e-3
    )
    assert len(str(capped.to_dict()["data"])) <= limits.get("max_bytes", 1 << 30)
    assert capped.voxels <= limits.get("max_voxels", 1 << 30)


# --- the format ---------------------------------------------------------------------------


def test_the_mask_round_trips_through_a_job_parameter() -> None:
    points = _l_shape_and_island()
    mask = support_mask.build(points, _footprint(points))
    assert mask is not None
    encoded = json.loads(json.dumps(mask.to_dict()))
    assert encoded["kind"] == support_mask.KIND
    assert len(json.dumps(encoded)) < 64_000
    back = SupportMask.parse(encoded)
    assert back is not None
    assert back.dims == mask.dims and back.voxels == mask.voxels
    assert back.origin == mask.origin and back.voxel == mask.voxel
    np.testing.assert_array_equal(back.starts, mask.starts)
    np.testing.assert_array_equal(back.stops, mask.stops)
    probe = _l_shape_and_island(seed=1)
    assert (back.contains(probe) == mask.contains(probe)).all()


def test_an_earlier_runs_dense_mask_still_reads() -> None:
    """quality.json from before the sparse format holds packed bits; a Refine of such a
    run must crop exactly as that run's mask said."""
    mask = support_mask.build(_l_shape_and_island())  # the extent rule: dims <= 320
    assert mask is not None
    old = _legacy(mask)
    back = SupportMask.parse(old)
    assert back is not None
    assert back.dims == mask.dims and back.voxels == mask.voxels
    np.testing.assert_array_equal(back.indices(), mask.indices())
    probe = _l_shape_and_island(seed=1)
    assert (back.contains(probe) == mask.contains(probe)).mean() > 0.9999
    # And its bits must fill its dims, exactly.
    with pytest.raises(ValueError, match="support_mask"):
        SupportMask.parse({**old, "dims": [mask.dims[0], mask.dims[1], mask.dims[2] + 1]})
    # Its old per-axis limit still bounds how far its bits may expand.
    with pytest.raises(ValueError, match="support_mask"):
        SupportMask.parse({**old, "dims": [mask.dims[0], mask.dims[1], 321]})


def _valid() -> dict[str, object]:
    return {
        "kind": "voxels.v2",
        "frame": "colmap",
        "origin": [0.0, 0.0, 0.0],
        "voxel": 1.0,
        "dims": [2, 3, 4],
        "voxels": 3,
        "data": _runs_payload([1, 2, 7, 1]),  # [1, 3) and [10, 11)
    }


def test_a_hand_made_mask_parses_to_its_runs() -> None:
    mask = SupportMask.parse(_valid())
    assert mask is not None
    assert mask.starts.tolist() == [1, 10] and mask.stops.tolist() == [3, 11]
    assert mask.indices().tolist() == [1, 2, 10]


def _bomb() -> str:
    return base64.b64encode(zlib.compress(b"\x01" * (100 << 20), 9)).decode()


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "sphere"},
        {"kind": "voxels.v3"},
        {"frame": "enu"},
        {"origin": [0.0, 0.0]},
        {"origin": [0.0, float("nan"), 0.0]},
        {"origin": [0.0, True, 0.0]},
        {"voxel": 0.0},
        {"voxel": -1.0},
        {"voxel": float("inf")},
        {"voxel": "1"},
        {"dims": [2, 3]},
        {"dims": [2, 3, 0]},
        {"dims": [2, 3, True]},
        {"dims": [2, 3, 4.0]},
        {"dims": [2, 3, support_mask.MAX_DIM + 1]},
        {"dims": [1, 1, 10]},  # the runs go past its end
        {"voxels": 4},  # not what the runs hold
        {"voxels": True},
        {"runs": 3},  # nor what they number
        {"data": None},
        {"data": "not base64!"},
        {"data": base64.b64encode(b"not zlib").decode()},
        {"data": _runs_payload([1, 2, 7, 1])[:-4]},  # truncated
        {"data": base64.b64encode(zlib.compress(b"\x01\x02") + b"junk").decode()},
        {"data": "A" * (support_mask.MAX_ENCODED_BYTES + 4)},
        {"data": _bomb()},  # decompresses into 100 MB
        {"data": _runs_payload([1, 2, 7])},  # not (gap, length) pairs
        {"data": _runs_payload([1, 0, 7, 1])},  # an empty run
        {"data": _runs_payload([1, 2, 0, 1])},  # adjacent: not maximal, or overlapping
        {"data": _runs_payload([])},  # empty: it would crop everything away
        {"data": base64.b64encode(zlib.compress(b"\x81")).decode()},  # ends mid-number
        {"data": base64.b64encode(zlib.compress(b"\x81\x00\x01")).decode()},  # padded
        {"data": base64.b64encode(zlib.compress(b"\xff" * 8 + b"\x01")).decode()},  # too long
    ],
)
def test_a_malformed_mask_is_refused_by_name(change: dict[str, object]) -> None:
    assert SupportMask.parse(_valid()) is not None
    with pytest.raises(ValueError, match="support_mask"):
        SupportMask.parse({**_valid(), **change})


def test_no_mask_is_none_and_a_non_mapping_is_refused() -> None:
    assert SupportMask.parse(None) is None
    values: list[object] = [[], "voxels", 3]
    for value in values:
        with pytest.raises(ValueError, match="support_mask"):
            SupportMask.parse(value)


# --- contains -----------------------------------------------------------------------------


def test_contains_is_exact_at_voxel_and_run_edges() -> None:
    """Grid (2, 3, 4) of unit voxels at the origin, occupied: linear 1, 2 and 10, i.e.
    (0,0,1), (0,0,2) and (0,2,2)."""
    mask = SupportMask.parse(_valid())
    assert mask is not None
    centres = np.array(
        [[i + 0.5, j + 0.5, k + 0.5] for i in range(2) for j in range(3) for k in range(4)]
    )
    assert np.flatnonzero(mask.contains(centres)).tolist() == [1, 2, 10]
    # A voxel's lower faces are in it, its upper faces are the next voxel's.
    assert mask.contains([[0.0, 0.0, 1.0], [0.0, 2.0, 2.0], [0.999, 0.999, 2.999]]).all()
    assert not mask.contains([[0.5, 0.5, 3.0], [0.5, 0.5, 0.999], [0.5, 2.5, 3.0]]).any()
    # Outside the grid on every side -- including exactly on its far faces -- is outside,
    # even where the linear index would alias into an occupied voxel.
    outside = [[-0.001, 0.5, 1.5], [0.5, -0.5, 1.5], [2.0, 0.5, 1.5], [0.5, 3.0, 1.5]]
    outside += [[0.5, 0.5, 4.0], [0.5, 0.5, 5.5], [0.5, 0.5, -3.5]]
    assert not mask.contains(outside).any()
    # Non-finite points are nowhere; float32 and a flat list both work; none is none.
    assert not mask.contains([[np.nan, 0.5, 1.5], [np.inf, 0.5, 1.5], [0.5, -np.inf, 1.5]]).any()
    assert mask.contains(np.array([0.5, 0.5, 1.5], dtype=np.float32)).tolist() == [True]
    assert mask.contains(np.zeros((0, 3))).shape == (0,)


def test_contains_agrees_with_brute_force_on_millions_of_points() -> None:
    points = _building(200_000)
    mask = support_mask.build(points, _footprint(points, 0.01))
    assert mask is not None
    rng = np.random.default_rng(9)
    probe = rng.uniform([-1, -1, -1], [51, 21, 16], size=(2_000_000, 3))
    brute = np.isin(_linear_or_minus_one(mask, probe), mask.indices(), assume_unique=False)
    assert (mask.contains(probe) == brute).all()


def _linear_or_minus_one(mask: SupportMask, xyz: np.ndarray) -> np.ndarray:
    ijk = np.floor((xyz - np.asarray(mask.origin)) / mask.voxel).astype(np.int64)
    dims = np.asarray(mask.dims)
    ok = (ijk >= 0).all(axis=1) & (ijk < dims).all(axis=1)
    linear = np.ravel_multi_index(tuple(np.clip(ijk, 0, dims - 1).T), mask.dims)
    return np.where(ok, linear, -1)


# --- the consumer ---------------------------------------------------------------------------


def test_a_trained_splat_is_cropped_to_the_mask() -> None:
    points = _l_shape_and_island()
    mask = support_mask.build(points, _footprint(points))
    assert mask is not None
    xyz = np.asarray([[2.0, 0.05, 1.0], [2.0, 2.0, 1.0], [3.5, 3.5, 0.5]], dtype=np.float32)
    columns = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "opacity": np.zeros(3, np.float32)}
    cropped, kept = training.crop_splat(columns, mask)
    assert kept == 2
    assert cropped["x"].tolist() == pytest.approx([2.0, 3.5])
