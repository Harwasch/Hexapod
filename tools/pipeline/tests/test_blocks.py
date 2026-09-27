"""Block training (`blocks.py`, `block_maths.py`): everything but the GPU.

**Nothing here trains or renders anything.** The rules are plain geometry and are tested
as that, each against the source it was taken from where one exists. The stage is driven
end to end by the stand-in trainers (`gsplat_stand_in.py`, and `loop_stand_in/` for the
convergence stop) and a stand-in for the camera test's renders
(`block_views_stand_in.py`), which is evidence that the blocks are planned, trained,
resumed and merged as described -- not that a real run's seams are invisible, which only
the spool capture at `blocks: 2` against `blocks: 1` on a GPU can show.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import block_maths
import blocks
import gaussian_budget
import gaussians
import sfm
import synthetic_scene
from adapters import LocalTransfer, SubprocessAdapter
from captures_bridge import read_ply
from cloud import AttemptLedger, CloudRunner, Placement
from errors import RemoteStageError, StageFailedError
from executor import execute
from providers import Rate
from runners import LocalRunner, RunnerSet
from test_train_gsplat import stand_in_params, train_recipe
from workdir import Workdir

HERE = Path(__file__).resolve().parent
VIEWS_STAND_IN = HERE / "block_views_stand_in.py"
LOOP_STAND_IN = HERE / "loop_stand_in" / "simple_trainer.py"


# --- the partition --------------------------------------------------------------------


def walk_points(count: int = 6000, seed: int = 0) -> np.ndarray:
    """A long walk along x, denser in the middle: quantiles, not equal widths, balance it."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 3.0, count) + rng.choice([-4.0, 0.0, 4.0], count, p=[0.2, 0.6, 0.2])
    return np.stack([x, rng.normal(0.0, 1.0, count), rng.normal(0.0, 0.2, count)], axis=1)


def test_blocks_hold_equal_counts_per_column_and_row_at_the_quantiles() -> None:
    """CityGS V2 build_partition_coordinates: per-axis quantiles of the coarse gaussians."""
    xyz = walk_points()
    part = block_maths.partition(xyz, [0.0, 0.0, 1.0], 4, exact=True)
    uv = part.project(xyz)
    cells = part.cell_of(uv)
    columns = np.bincount(cells // part.rows, minlength=part.columns)
    rows = np.bincount(cells % part.rows, minlength=part.rows)
    assert part.count == 4
    assert columns.max() - columns.min() <= 1
    assert rows.max() - rows.min() <= 1
    # The walk runs along x, so it is cut across its length: the first axis is x.
    assert abs(float(np.dot(part.axis_u, [1.0, 0.0, 0.0]))) > 0.99
    assert (part.columns, part.rows) == (4, 1)


def test_the_partition_is_deterministic_and_ignores_input_order() -> None:
    xyz = walk_points()
    first = block_maths.partition(xyz, [0.0, 0.0, 1.0], 6, exact=False)
    again = block_maths.partition(xyz[::-1].copy(), [0.0, 0.0, 1.0], 6, exact=False)
    assert first.to_dict() == again.to_dict()
    assert block_maths.Partition.from_dict(json.loads(json.dumps(first.to_dict()))) == first


def test_the_ground_plane_follows_the_cameras_up() -> None:
    """The same walk with up along y (COLMAP's frame is whatever the first camera was)."""
    xyz = walk_points()[:, [0, 2, 1]]
    part = block_maths.partition(xyz, [0.0, 1.0, 0.0], 2, exact=True)
    up = np.cross(part.axis_u, part.axis_v)
    assert abs(float(up[1])) > 0.999


def test_a_forced_count_is_exact_and_a_budget_count_is_a_minimum() -> None:
    assert block_maths.grid_shape(2, 1.0, 1.0, exact=True) == (2, 1)
    assert block_maths.grid_shape(4, 1.0, 1.0, exact=True) == (2, 2)
    assert block_maths.grid_shape(3, 1.0, 1.0, exact=False) == (2, 2)
    assert block_maths.grid_shape(5, 10.0, 1.0, exact=False) == (5, 1)


def test_every_point_belongs_to_exactly_one_block_the_outer_edges_open() -> None:
    xyz = walk_points()
    part = block_maths.partition(xyz, [0.0, 0.0, 1.0], 4, exact=True)
    far = np.array([[1e6, 0.0, 0.0], [-1e6, 5e5, 0.0]])
    blocks_of = part.block_of(part.project(np.vstack([xyz, far])))
    assert (blocks_of >= 0).all() and (blocks_of < part.count).all()
    assert blocks_of[-2] == part.count - 1 and blocks_of[-1] == 0


# --- block count ----------------------------------------------------------------------


def auto_budget(raw: int, *, ceiling: int, budget_max: int | None) -> gaussian_budget.Budget:
    return gaussian_budget.Budget(
        mode="auto", cap=min(raw, ceiling), raw=raw, memory_ceiling=ceiling, budget_max=budget_max
    )


def test_blocks_are_only_used_when_the_budget_is_more_than_one_gpu() -> None:
    fits = auto_budget(1_500_000, ceiling=8_700_000, budget_max=2_000_000)
    assert blocks.block_count(None, fits)[0] == 1
    over = auto_budget(5_000_000, ceiling=8_700_000, budget_max=2_000_000)
    count, reason = blocks.block_count(None, over)
    # The per-GPU cap is read off the budget: here budget_max (2M) is the smaller ceiling.
    assert blocks.gpu_cap(over) == 2_000_000
    assert count == 3 and "2.50x" in reason
    raised = auto_budget(20_000_000, ceiling=8_700_000, budget_max=None)
    assert blocks.block_count(None, raised)[0] == 3
    explicit = gaussian_budget.Budget(mode="explicit", cap=50_000_000, memory_ceiling=8_700_000)
    assert blocks.block_count(None, explicit)[0] == 1
    assert blocks.block_count(2, fits) == (2, "blocks: 2 given")


@pytest.mark.parametrize("value", ["two", 0, -1, True, 1.5])
def test_a_block_count_that_is_not_auto_or_a_positive_integer_is_refused(value: object) -> None:
    with pytest.raises(ValueError, match="blocks must be"):
        blocks.parse_blocks(value)


def test_auto_and_unset_mean_the_budget_decides() -> None:
    assert blocks.parse_blocks(None) is None
    assert blocks.parse_blocks("auto") is None
    assert blocks.parse_blocks(3) == 3
    assert blocks.parse_blocks("2") == 2


# --- cameras --------------------------------------------------------------------------


def two_blocks() -> block_maths.Partition:
    return block_maths.Partition(
        origin=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 1.0, 0.0),
        u_edges=(-4.0, 0.0, 4.0),
        v_edges=(-1.0, 1.0),
        blocks=((0,), (1,)),
    )


def test_a_camera_standing_in_a_block_is_assigned_to_it() -> None:
    part = two_blocks()
    centres = np.array([[-2.0, 0.0, 1.0], [3.0, 0.0, 1.0], [-50.0, 9.0, 1.0], [0.0, 0.0, 1.0]])
    inside = block_maths.cameras_inside(part, centres)
    # Outside the grid still counts, by the nearest edge block; an edge goes right.
    assert inside.tolist() == [[True, False], [False, True], [True, False], [False, True]]


def test_ssim_is_one_for_equal_images_and_falls_with_what_changes() -> None:
    rng = np.random.default_rng(1)
    image = rng.random((32, 40, 3))
    assert block_maths.ssim(image, image) == pytest.approx(1.0)
    small = image.copy()
    small[:4, :4] = 0.0
    large = image.copy()
    large[:, :20] = 0.0
    assert 1.0 - block_maths.ssim(small, image) < 1.0 - block_maths.ssim(large, image)


def test_the_camera_test_is_one_minus_ssim_of_the_render_without_the_block() -> None:
    """CityGS V2 projection_based_partition_assignment, with a stub renderer: camera 0 sees
    block 1 fill half its view, camera 1 sees a sliver of it, camera 2 sees none of it."""
    rng = np.random.default_rng(2)
    base = rng.random((24, 32, 3))
    members = [np.array([True, False]), np.array([False, True])]
    covers = {0: 16, 1: 1, 2: 0}  # columns of the image block 1 paints, per camera
    calls: list[tuple[int, int | None]] = []

    def render(camera: int, drop: Any) -> np.ndarray:
        dropped = None if drop is None else next(i for i, m in enumerate(members) if m is drop)
        calls.append((camera, dropped))
        image = base.copy()
        if dropped == 1:
            image[:, : covers[camera]] = 0.0
        return image

    inside = np.array([[True, False], [True, False], [True, False]])
    loss = block_maths.contribution(render, 3, members, skip=inside)
    assert loss[0, 1] > block_maths.DEFAULT_EPSILON > loss[1, 1] > 0.0
    assert loss[2, 1] == 0.0
    # A camera standing in a block is not rendered for it (CityGS renders only the rest).
    assert all(dropped != 0 for _camera, dropped in calls)
    assigned = block_maths.assign(inside, loss, block_maths.DEFAULT_EPSILON)
    assert assigned.tolist() == [[True, True], [True, False], [True, False]]


def test_a_block_under_the_minimum_joins_its_emptiest_neighbour() -> None:
    """CityGS V2 asserts > 50 images a block; one short of that is merged, not trained."""
    part = block_maths.Partition(
        origin=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 1.0, 0.0),
        u_edges=(0.0, 1.0, 2.0, 3.0),
        v_edges=(0.0, 1.0),
        blocks=((0,), (1,), (2,)),
    )
    assigned = np.zeros((200, 3), dtype=bool)
    assigned[:60, 0] = True
    assigned[60:70, 1] = True
    assigned[70:140, 2] = True
    train = np.ones(200, dtype=bool)
    merged, matrix, merges = block_maths.merge_small(part, assigned, train, [900, 100, 500], 50)
    assert merged.blocks == ((0,), (1, 2))  # into block 2: fewer gaussians than block 0
    assert merges == [(2, 1, 10)]
    assert matrix[:, 1].sum() == 80 and matrix[:, 0].sum() == 60
    # Nothing under the minimum: nothing changes.
    same, _, none = block_maths.merge_small(part, assigned, train, [1, 1, 1], 5)
    assert same.blocks == part.blocks and none == []


def test_a_block_count_that_cannot_reach_the_minimum_comes_down_to_one() -> None:
    part = two_blocks()
    assigned = np.ones((30, 2), dtype=bool)
    merged, matrix, merges = block_maths.merge_small(part, assigned, np.ones(30, bool), [5, 5])
    assert merged.count == 1 and matrix.shape == (30, 1) and len(merges) == 1


# --- the frozen ring and the blend ----------------------------------------------------


def test_the_frozen_ring_is_h3dgs_scaffold_outside_the_trainable_region() -> None:
    """H3DGS gaussian_model.py: 0.5 < max(|dx|, |dy|) / size < 1.5, here from the margin's
    edge outward, per axis for a rectangular block."""
    part = block_maths.Partition(
        origin=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 1.0, 0.0),
        u_edges=(-10.0, 0.0, 2.0, 4.0, 14.0),
        v_edges=(-10.0, 0.0, 1.0, 11.0),
        blocks=tuple((c,) for c in range(12)),
    )
    block = 1 * part.rows + 1  # the cell u in [0, 2], v in [0, 1]
    uv = np.array(
        [
            [1.0, 0.5],  # its centre: trainable
            [2.1, 0.5],  # inside the 10% margin: trainable, not ring
            [2.5, 0.5],  # past the margin: ring
            [3.9, 0.5],  # 1.45 sizes from the centre along u: ring
            [4.1, 0.5],  # 1.55: beyond the ring
            [1.0, 1.9],  # 1.4 sizes along v: ring
            [1.0, 2.1],  # 1.6 along v: beyond
        ]
    )
    ring = block_maths.ring(part, uv, block, margin=0.1, outer=1.5)
    assert ring.tolist() == [False, False, True, True, False, True, False]
    assert part.in_expanded(uv, block, 0.1).tolist()[:3] == [True, True, False]


def get_weight(pos: np.ndarray, chunk: int, centres: np.ndarray, falloff: float = 0.05) -> float:
    """H3DGS hierarchy_explicit_loader.cpp::getWeight, transcribed."""
    own = float(np.linalg.norm(pos - centres[chunk]))
    other = min(float(np.linalg.norm(pos - c)) for i, c in enumerate(centres) if i != chunk)
    if own <= (1 - falloff) * other:
        return 1.0
    if own > (1 + falloff) * other:
        return 0.0
    return (-1 / (2 * falloff * other)) * own + (1 + falloff) / (2 * falloff)


def test_the_blend_matches_h3dgs_get_weight_on_its_uniform_grid() -> None:
    part = two_blocks()
    centres = np.array([[-2.0, 0.0], [2.0, 0.0]])
    xs = np.linspace(-0.3, 0.3, 121)
    uv = np.stack([xs, np.zeros_like(xs)], axis=1)
    ours = block_maths.weights(part, uv, 0)
    theirs = np.array([get_weight(p, 0, centres) for p in uv])
    # getWeight is linear in the distance to its own centre, so its band sits 0.0128 D
    # before the edge and 0.0122 D after it; ours is the symmetric +-0.0125 D.
    assert np.abs(ours - theirs).max() < 0.02
    band = xs[(theirs > 0) & (theirs < 1)]
    ours_band = xs[(ours > 0) & (ours < 1)]
    assert abs(band.min() - ours_band.min()) < 0.01
    assert abs(band.max() - ours_band.max()) < 0.01
    assert ours[0] == 1.0 and ours[-1] == 0.0
    assert block_maths.weights(part, np.array([[0.0, 0.0]]), 0)[0] == pytest.approx(0.5)


def test_the_blend_is_a_partition_of_unity_so_the_band_never_doubles_density() -> None:
    xyz = walk_points()
    part = block_maths.partition(xyz, [0.0, 0.0, 1.0], 6, exact=True)
    uv = part.project(xyz)
    total = sum(block_maths.weights(part, uv, b) for b in range(part.count))
    assert np.allclose(total, 1.0)
    # A merged block is its cells' sum, so the same holds after the minimum-images rule.
    merged = block_maths.Partition(**{**part.__dict__, "blocks": ((0, 1), *part.blocks[2:])})
    total = sum(block_maths.weights(merged, uv, b) for b in range(merged.count))
    assert np.allclose(total, 1.0)


def test_a_zero_falloff_is_the_hard_crop_of_cityg_and_vastgs() -> None:
    part = two_blocks()
    uv = np.array([[-0.01, 0.0], [0.0, 0.0], [0.01, 0.0]])
    assert block_maths.weights(part, uv, 0, falloff=0.0).tolist() == [1.0, 0.0, 0.0]


def test_fading_scales_alpha_and_leaves_full_weight_bit_for_bit() -> None:
    logits = np.array([2.0, -1.0, 0.3], dtype=np.float32)
    out = block_maths.blended_opacity(logits, np.array([1.0, 0.5, 0.25]))
    assert out[0] == logits[0]
    alpha = 1 / (1 + np.exp(-out.astype(np.float64)))
    expected = 1 / (1 + np.exp(-logits[1:].astype(np.float64))) * np.array([0.5, 0.25])
    assert np.allclose(alpha[1:], expected, atol=1e-6)


# --- the held-out frames of a block ---------------------------------------------------


def test_a_blocks_dataset_holds_out_exactly_its_global_val_frames() -> None:
    """gsplat v1.5.3's parser holds out `index % test_every == 0` of the sorted names."""
    train = [f"t{i:03d}" for i in range(61)]
    val = [f"v{i:03d}" for i in range(9)]
    order, every = block_maths.val_order(train, val)
    renamed = sorted(f"{pos:05d}_{name}" for pos, (name, _) in enumerate(order))
    held = [renamed[i].split("_", 1)[1] for i in range(len(renamed)) if i % every == 0]
    trained = [renamed[i].split("_", 1)[1] for i in range(len(renamed)) if i % every != 0]
    assert set(held) <= set(val) and len(held) >= 8
    assert sorted(trained) == train


def test_a_block_with_no_val_frame_holds_out_its_first() -> None:
    order, every = block_maths.val_order(["a", "b", "c"], [])
    assert every == 4 and [flag for _, flag in order] == [True, False, False]


# --- PLYs -----------------------------------------------------------------------------


def columns(count: int, seed: int, rest: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    names = [*gaussians.CANONICAL_PROPERTIES, *(f"f_rest_{i}" for i in range(rest))]
    return {name: rng.normal(size=count).astype(np.float32) for name in names}


def test_the_merged_ply_is_the_concatenation_of_the_cropped_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each block is kept as small PLY parts in the checkpoint (`write_parts`).
    monkeypatch.setattr(blocks, "PART_ROWS", 300)
    parts = []
    for index, count in enumerate((1000, 0, 2500)):
        directory = tmp_path / f"block_{index}"
        blocks.write_parts(directory, columns(count, index, rest=9), list(columns(1, 0, rest=9)))
        found = blocks.block_parts(directory)
        assert len(found) == max(1, -(-count // 300))
        parts.extend(found)
    full = tmp_path / "full.ply"
    total, _ = blocks.merge_plys(parts, full, chunk=700)
    assert total == 3500
    merged = read_ply(full)
    for name in ("x", "opacity", "f_rest_8"):
        expected = np.concatenate([read_ply(p)[name] for p in parts])
        assert np.array_equal(merged[name], expected)
    canonical = tmp_path / "trained.ply"
    blocks.merge_plys(parts, canonical, gaussians.CANONICAL_PROPERTIES, chunk=333)
    reference = tmp_path / "reference.ply"
    gaussians.write_ply(reference, read_ply(full))
    # Byte-identical to the in-memory writer every other trained.ply comes from.
    assert canonical.read_bytes() == reference.read_bytes()


# --- the prior ------------------------------------------------------------------------


def write_walk(poses: Path, cameras: int = 48) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cameras walking along x at y = -3, each looking across at the wall y = 0, and SfM
    points on the wall each seen by the three nearest cameras."""
    rig = []
    for index in range(cameras):
        x = -6.0 + 12.0 * index / (cameras - 1)
        centre = np.array([x, -3.0, 1.0])
        rig.append((synthetic_scene.look_at(centre, np.array([x, 0.0, 0.8])), centre))
    synthetic_scene.write_model(poses, rig, width=64, height=48, focal=50.0)
    xyz, tracks = [], []
    for index in range(cameras):
        x = -6.0 + 12.0 * index / (cameras - 1)
        for k in range(8):
            xyz.append([x + 0.02 * k, 0.0, 0.2 * k])
            seen = [i for i in (index - 1, index, index + 1) if 0 <= i < cameras]
            tracks.append(np.array([[i + 1, k] for i in seen], dtype=np.uint32))
    lengths = [len(t) for t in tracks]
    points = sfm.Points3D(
        ids=np.arange(1, len(xyz) + 1, dtype=np.uint64),
        xyz=np.asarray(xyz, dtype=np.float64),
        rgb=np.full((len(xyz), 3), 120, dtype=np.uint8),
        error=np.full(len(xyz), 0.5),
        track=np.concatenate(tracks),
        track_offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
    )
    sfm.write_points3d(poses / "points3D.bin", points)
    (poses / "poses.json").write_text(
        json.dumps({"registered": cameras, "upEstimate": {"up": [0.0, 0.0, 1.0]}}),
        encoding="utf-8",
    )
    return rig


def seed_walk(workdir: Workdir, cameras: int = 48) -> None:
    frames = workdir.input_path("frames")
    frames.mkdir(parents=True, exist_ok=True)
    for index in range(cameras):
        Image.new("RGB", (64, 48), (index * 5 % 255, 90, 160)).save(
            frames / f"frame_{index:04d}.jpg"
        )
    poses = workdir.input_path("poses")
    poses.mkdir(parents=True, exist_ok=True)
    write_walk(poses, cameras)


def wall_splat(count: int = 400) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    out = columns(count, 7)
    out["x"] = rng.uniform(-6.0, 6.0, count).astype(np.float32)
    out["y"] = rng.normal(0.0, 0.05, count).astype(np.float32)
    out["z"] = rng.uniform(0.0, 1.6, count).astype(np.float32)
    out["opacity"] = np.full(count, 2.0, dtype=np.float32)
    return out


def seed_prior(workdir: Workdir, tmp_path: Path) -> int:
    checkpoint = workdir.checkpoint_dir("train")
    checkpoint.mkdir(parents=True, exist_ok=True)
    saved = blocks.save_prior(
        checkpoint, wall_splat(), workdir.input_path("poses"), scratch=tmp_path, source="test"
    )
    assert saved is not None
    return saved


def test_the_prior_round_trips_and_is_refused_in_other_poses(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_walk(workdir)
    assert seed_prior(workdir, tmp_path / "scratch") == 400
    prior, why = blocks.load_prior(workdir.checkpoint_dir("train"), workdir.input_path("poses"))
    assert prior is not None and why == "" and prior.count == 400
    assert np.allclose(prior.xyz[:, 0], wall_splat()["x"])
    seed = prior.as_seed()
    assert seed.rgb.dtype == np.uint8 and seed.count == 400
    (workdir.input_path("poses") / "images.bin").write_bytes(b"other")
    prior, why = blocks.load_prior(workdir.checkpoint_dir("train"), workdir.input_path("poses"))
    assert prior is None and "other poses" in why


def test_a_splat_too_big_to_be_a_prior_leaves_none(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_walk(workdir, cameras=4)
    saved = blocks.save_prior(
        workdir.checkpoint_dir("train"),
        wall_splat(),
        workdir.input_path("poses"),
        scratch=tmp_path,
        source="test",
        max_gaussians=100,
    )
    assert saved is None
    assert not (workdir.checkpoint_dir("train") / blocks.PRIOR_DIR).exists()


# --- a block's COLMAP dataset ---------------------------------------------------------


def test_a_blocks_dataset_is_its_frames_renamed_with_their_tracks_only(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    write_walk(source / "sparse" / "0", cameras=10)
    (source / "images").mkdir(parents=True)
    for index in range(10):
        (source / "images" / f"frame_{index:04d}.jpg").write_bytes(bytes([index]))
    order = [("frame_0008.jpg", True), ("frame_0002.jpg", False), ("frame_0003.jpg", False)]
    rename = blocks.build_block_dataset(source, tmp_path / "block", order)
    model = sfm.read_model(tmp_path / "block" / "sparse" / "0")
    assert [image.name for image in model.images] == sorted(rename.values())
    assert rename["frame_0008.jpg"] == "00000_frame_0008.jpg"
    assert (tmp_path / "block" / "images" / "00001_frame_0002.jpg").read_bytes() == bytes([2])
    points = sfm.read_points3d(tmp_path / "block" / "sparse" / "0" / "points3D.bin")
    assert set(points.track[:, 0].tolist()) <= {3, 4, 9}
    assert len(points) == 80  # every point kept; those no kept frame saw have no track
    struct.unpack("<Q", (tmp_path / "block" / "sparse" / "0" / "images.bin").read_bytes()[:8])


# --- the stage, end to end, with stand-ins --------------------------------------------


def block_params(**overrides: object) -> dict[str, object]:
    params = stand_in_params(
        strategy="mcmc",
        cap_max=200,
        blocks=2,
        block_min_images=5,
        block_views_script=str(VIEWS_STAND_IN),
        block_eval=False,
        live=False,
    )
    params.update(overrides)
    return params


def run_blocks(workdir: Workdir, params: dict[str, object], attempt: int = 1) -> dict[str, Any]:
    execute(
        train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()), attempts={"train": attempt}
    )
    document: dict[str, Any] = json.loads(
        (workdir.out_dir("train") / "train_metrics.json").read_text()
    )
    return document


def test_forced_two_blocks_train_one_after_another_and_merge(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_walk(workdir)
    seed_prior(workdir, tmp_path / "scratch")

    document = run_blocks(workdir, block_params())

    record = document["blocks"]
    assert record["count"] == 2 and record["grid"] == [2, 1]
    assert record["prior"] == {"source": "test", "gaussians": 400}
    assert record["cameraTest"]["method"] == "render"
    assert record["cameraTest"]["epsilon"] == block_maths.DEFAULT_EPSILON
    trained = read_ply(workdir.out_dir("train") / "trained.ply")
    kept = [block["gaussiansKept"] for block in record["blocks"]]
    assert trained["x"].shape[0] == sum(kept) == document["gaussiansInPly"]
    # Each block: the frames standing in it, plus every third frame (the scripted camera
    # test), minus the global held-out ones, which it is evaluated on instead.
    names = [f"frame_{i:04d}.jpg" for i in range(48)]
    part = block_maths.Partition.from_dict(record["partition"])
    model = sfm.read_model(workdir.input_path("poses"))
    inside = block_maths.cameras_inside(part, model.centres())
    for index, block in enumerate(record["blocks"]):
        chosen = [i for i in range(48) if inside[i, index] or i % 3 == 0]
        assert block["cameras"]["train"] == sum(1 for i in chosen if i % 8 != 0)
        assert block["cameras"]["val"] == sum(1 for i in chosen if i % 8 == 0)
        dataset = workdir.work_dir("train") / "blocks" / f"b{index}" / "dataset" / "images"
        listed = sorted(p.name for p in dataset.iterdir())
        every = block["cameras"]["testEvery"]
        held = {listed[i].split("_", 1)[1] for i in range(len(listed)) if i % every == 0}
        assert held <= {names[i] for i in range(0, 48, 8)}
        assert block["ring"] > 0 and block["capMax"] is not None
    log = workdir.log_path("train").read_text()
    assert log.count("--test_every") == 2
    assert "block 2 of 2" in log
    step = json.loads(workdir.step_path("train").read_text())
    assert step["metrics"]["blocks"] == 2
    assert step["metrics"]["blockCameraTest"] == "render"
    # The finished blocks' splats leave the checkpoint once merged; the plan stays.
    left = sorted(p.name for p in (workdir.checkpoint_dir("train") / blocks.STATE_DIR).iterdir())
    assert left == ["plan.json", "state.json"]
    # A block run leaves the prior it used, not its own merged splat, for the next one.
    prior, _ = blocks.load_prior(workdir.checkpoint_dir("train"), workdir.input_path("poses"))
    assert prior is not None and prior.meta["source"] == "test"


def test_a_block_run_resumes_after_the_blocks_it_finished(tmp_path: Path) -> None:
    """An attempt out of time after block 1 ends; the next attempt trains only block 2."""
    workdir = Workdir.create(tmp_path / "run")
    seed_walk(workdir)
    seed_prior(workdir, tmp_path / "scratch")
    hurried = block_params(block_attempt_budget_s=0, block_sync_wait_s=0)

    with pytest.raises(StageFailedError, match="1 of 2 blocks trained"):
        run_blocks(workdir, hurried, attempt=1)

    state = json.loads((workdir.checkpoint_dir("train") / "blocks" / "state.json").read_text())
    assert sorted(state["blocks"]) == ["0"]
    assert (workdir.checkpoint_dir("train") / "blocks" / "block_000" / "part_0000.ply").is_file()
    log_before = workdir.log_path("train").read_text()

    # The budget is pacing, not substance: changing it keeps the finished block.
    document = run_blocks(workdir, block_params(), attempt=2)

    log = workdir.log_path("train").read_text()[len(log_before) :]
    assert "block 1 of 2 finished in an earlier attempt; skipped" in log
    assert log.count("/blocks/b0/dataset") == 0 and log.count("/blocks/b1/dataset") >= 1
    assert "resumes the plan" in log
    assert [block["attempt"] for block in document["blocks"]["blocks"]] == [1, 2]
    assert document["attempts"] == 2


def test_finished_blocks_come_back_through_the_cloud_seam_and_are_not_retrained(
    tmp_path: Path,
) -> None:
    """The Modal path: one remote call per attempt, `checkpoint/` synced out on an interval
    and brought back by `CloudRunner` whatever the attempt's end, so the next attempt's
    container starts with the finished block in it."""
    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = SubprocessAdapter(transfer, tmp_path / "sandbox", rates={"l4": Rate(0.80, "test")})
    cloud = CloudRunner(
        Placement((adapter,)), transfer, poll_interval_s=0.02, checkpoint_every_s=0.05
    )
    workdir = Workdir.create(tmp_path / "run")
    seed_walk(workdir)
    seed_prior(workdir, tmp_path / "scratch")
    gpu = {"tier": "l4", "preemptible": True}
    hurried = block_params(block_attempt_budget_s=0, block_sync_wait_s=0.5)

    with pytest.raises(RemoteStageError):
        execute(
            train_recipe(hurried, gpu=gpu), workdir, RunnerSet.cloud(cloud), attempts={"train": 1}
        )
    assert (workdir.checkpoint_dir("train") / "blocks" / "block_000" / "part_0000.ply").is_file()

    execute(
        train_recipe(block_params(), gpu=gpu),
        workdir,
        RunnerSet.cloud(cloud),
        attempts={"train": 2},
    )

    document = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert [block["attempt"] for block in document["blocks"]["blocks"]] == [1, 2]
    assert "finished in an earlier attempt; skipped" in workdir.log_path("train").read_text()
    ledger = AttemptLedger.read(workdir.attempts_path("train"))
    assert [entry.state for entry in ledger.entries] == ["failed", "succeeded"]
    # The merged blocks' splats are gone from the checkpoint that came home.
    home = workdir.checkpoint_dir("train") / "blocks"
    assert sorted(p.name for p in home.iterdir()) == ["plan.json", "state.json"]


def test_a_fresh_run_never_reuses_an_earlier_runs_blocks(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_walk(workdir)
    seed_prior(workdir, tmp_path / "scratch")
    with pytest.raises(StageFailedError):
        run_blocks(workdir, block_params(block_attempt_budget_s=0, block_sync_wait_s=0))

    document = run_blocks(workdir, block_params(), attempt=1)

    assert [block["attempt"] for block in document["blocks"]["blocks"]] == [1, 1]
    assert (
        "earlier attempt; skipped"
        not in workdir.log_path("train").read_text().split("cameras.bin")[-1]
    )


def test_with_no_prior_a_coarse_pass_is_trained_first_and_kept(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_walk(workdir)

    document = run_blocks(workdir, block_params(coarse_cap=64))

    assert document["blocks"]["prior"]["source"] == "coarse"
    log = workdir.log_path("train").read_text()
    assert "training a coarse whole-scene pass first" in log
    assert "--steps_scaler 0.25" in log and "--strategy.cap-max 64" in log
    prior, _ = blocks.load_prior(workdir.checkpoint_dir("train"), workdir.input_path("poses"))
    assert prior is not None and prior.meta["source"] == "coarse"


def test_without_a_gpu_render_the_camera_test_falls_back_to_visibility(tmp_path: Path) -> None:
    """The real block_views.py needs torch and CUDA; with neither it fails, and
    VastGaussian's projected-area test assigns the cameras instead."""
    workdir = Workdir.create(tmp_path / "run")
    seed_walk(workdir)
    seed_prior(workdir, tmp_path / "scratch")

    document = run_blocks(workdir, block_params(block_views_script=str(blocks.BLOCK_VIEWS)))

    assert document["blocks"]["cameraTest"]["method"] == "visibility"
    assert "VastGaussian's visibility test" in workdir.log_path("train").read_text()


def test_each_block_is_stopped_by_the_convergence_rule_through_the_block_wrapper(
    tmp_path: Path,
) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_walk(workdir)
    seed_prior(workdir, tmp_path / "scratch")
    params = block_params(
        trainer=str(LOOP_STAND_IN),
        iterations=30_000,
        schedule_scale=0.05,
        converge=True,
        extra_args=["--plateau-at", "0.84"],
    )

    document = run_blocks(workdir, params)

    for block in document["blocks"]["blocks"]:
        assert block["stoppedEarly"] is True and block["stepsRun"] == 1_351
    assert document["convergence"]["stoppedEarly"] is True
    log = workdir.log_path("train").read_text()
    assert log.count("block_trainer.py --trainer") == 2
    assert "--ring" in log and "no rasterization global; no ring" in log


def test_the_merged_splat_is_evaluated_or_says_why_not(tmp_path: Path) -> None:
    """Without torch the PLY cannot become a checkpoint; the stage still finishes."""
    workdir = Workdir.create(tmp_path / "run")
    seed_walk(workdir)
    seed_prior(workdir, tmp_path / "scratch")

    document = run_blocks(workdir, block_params(block_eval=True))

    evaluation = document["blocks"]["evaluation"]
    assert evaluation["status"] in ("ok", "failed")
    if evaluation["status"] == "failed":
        assert document["psnr"] is None
        step = json.loads(workdir.step_path("train").read_text())
        assert step["metrics"]["metricsSource"] == "none"


def test_one_block_after_the_minimum_rule_trains_as_a_single_run(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_walk(workdir, cameras=20)
    seed_prior(workdir, tmp_path / "scratch")

    document = run_blocks(workdir, block_params(block_min_images=50))

    assert "blocks" not in document
    assert "one block is left after the minimum-cameras rule" in (
        workdir.log_path("train").read_text()
    )


def test_the_block_wrapper_hands_the_trainer_its_argv_unchanged() -> None:
    argv = ["py", "/t/simple_trainer.py", "mcmc", "--data_dir", "/d"]
    wrapped = blocks.trainer_argv(argv, script=Path("/p/block_trainer.py"), rule=None, ring=None)
    assert wrapped == [
        "py",
        "/p/block_trainer.py",
        "--trainer",
        "/t/simple_trainer.py",
        "--",
        "mcmc",
        "--data_dir",
        "/d",
    ]
    assert sys.executable
