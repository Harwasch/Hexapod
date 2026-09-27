"""A Refine that starts from the preview's splat (`init_seed.py`), with a stand-in trainer.

**Nothing here trains anything**: `gsplat_stand_in.py` plays the trainer, exactly as in
`test_train_gsplat.py`. What is tested is this project's half -- that a finished run
leaves a seed, that the next run of the stage finds it, checks it is in the same frame,
adds it to the initial points in the shape gsplat v1.5.3's parser reads, and shortens
the schedule -- and nothing about whether the result trains better. That is a GPU
measurement, and the README says what to compare on the first real one.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_train_gsplat import ring_points, seed_inputs, stand_in_params, train_recipe

import gaussians
import init_seed
import sfm
import support_mask
from adapters import LocalTransfer, SubprocessAdapter
from cloud import CloudRunner, Placement
from executor import execute
from runners import LocalRunner, RunnerSet
from workdir import Workdir

SFM_POINTS = 100


def preview_then(workdir: Workdir, **refine: object) -> dict[str, Any]:
    """Train once (the preview), then again in the same workdir (the Refine)."""
    seed_inputs(workdir)
    sfm.write_points3d(
        workdir.input_path("poses") / "points3D.bin",
        ring_points(SFM_POINTS, images=4, radius=3.0),
    )
    execute(
        train_recipe(stand_in_params(extra_args=["--gaussians", "400"])),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )
    execute(train_recipe(stand_in_params(**refine)), workdir, RunnerSet(cpu=LocalRunner()))
    document: dict[str, Any] = json.loads(
        (workdir.out_dir("train") / "train_metrics.json").read_text()
    )
    return document


def dataset_points(workdir: Workdir) -> sfm.Points3D:
    sparse = workdir.work_dir("train") / "dataset" / "sparse" / "0"
    return sfm.read_points3d(sparse / "points3D.bin")


def test_a_finished_run_leaves_a_seed_of_its_visible_gaussians(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    execute(train_recipe(stand_in_params()), workdir, RunnerSet(cpu=LocalRunner()))

    seed, why = init_seed.load(workdir.checkpoint_dir("train"), workdir.input_path("poses"))
    assert seed is not None, why
    trained = gaussians.read_splat(workdir.out_dir("train") / "trained.ply")
    assert seed.count == trained.count == 64
    np.testing.assert_allclose(seed.xyz, trained.xyz, atol=1e-6)
    expected = np.clip(np.round((gaussians.SH_C0 * trained.columns["f_dc_0"] + 0.5) * 255), 0, 255)
    np.testing.assert_array_equal(seed.rgb[:, 0], expected.astype(np.uint8))
    assert seed.meta["trainedWith"]["initFrom"] == "sfm"
    assert "left a 64-gaussian seed" in workdir.log_path("train").read_text()


def test_a_refine_starts_from_the_preview_on_half_the_schedule(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")

    document = preview_then(workdir, init_from="preview")

    assert document["settings"]["initFrom"] == "preview"
    assert document["settings"]["scheduleScale"] == init_seed.DEFAULT_SCHEDULE_SCALE
    assert document["requestedIterations"] == 150  # 300 x 0.5
    init = document["init"]
    assert init["from"] == "preview"
    assert init["seedGaussians"] == 400
    assert init["sfmPoints"] == SFM_POINTS
    points = dataset_points(workdir)
    assert len(points) == SFM_POINTS + init["seedPoints"] == init["initialPoints"]
    lengths = np.diff(points.track_offsets)
    # COLMAP's points keep their observations; the seed's have none, so the depth loss
    # (which reads tracks) sees exactly what it saw before.
    assert (lengths[:SFM_POINTS] == 1).all()
    assert (lengths[SFM_POINTS:] == 0).all()
    assert len(set(points.ids.tolist())) == len(points)
    # The poses artifact is untouched.
    assert len(sfm.read_points3d(workdir.input_path("poses") / "points3D.bin")) == SFM_POINTS
    log = workdir.log_path("train").read_text()
    assert "init_from=preview" in log and "--steps_scaler 0.5" in log


def test_the_seed_travels_to_the_gpu_box_and_back_through_the_cloud_seam(
    tmp_path: Path,
) -> None:
    """The production path: `train` on another machine. The preview's seed comes home in
    `checkpoint/` after the remote run, and goes out again with the Refine."""
    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = SubprocessAdapter(transfer, tmp_path / "sandbox")
    cloud = RunnerSet.cloud(
        CloudRunner(Placement((adapter,)), transfer, poll_interval_s=0.02, checkpoint_every_s=0.05)
    )
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)
    sfm.write_points3d(
        workdir.input_path("poses") / "points3D.bin",
        ring_points(SFM_POINTS, images=4, radius=3.0),
    )
    gpu = {"tier": "l4", "preemptible": True}

    execute(train_recipe(stand_in_params(), gpu=gpu), workdir, cloud)
    assert (workdir.checkpoint_dir("train") / init_seed.SEED_DIR / "seed.npz").is_file()
    execute(train_recipe(stand_in_params(init_from="preview"), gpu=gpu), workdir, cloud)

    document = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert document["init"]["seedGaussians"] == 64
    assert document["requestedIterations"] == 150


def test_the_seed_is_cropped_to_the_support_mask_like_the_sfm_points(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    mask = support_mask.build(rng.uniform([0.0, -4.0, -4.0], [4.0, 4.0, 4.0], size=(60_000, 3)))
    assert mask is not None
    workdir = Workdir.create(tmp_path / "run")

    document = preview_then(workdir, init_from="preview", support_mask=mask.to_dict())

    init = document["init"]
    assert 0 < init["insideRegion"] < 400
    assert init["outsideSampled"] == -(-(400 - init["insideRegion"]) // init_seed.OUTSIDE_EVERY)
    points = dataset_points(workdir)
    seeded = points.xyz[np.diff(points.track_offsets) == 0]
    inside = mask.contains(seeded)
    assert int(inside.sum()) == init["insideRegion"]
    assert int((~inside).sum()) == init["outsideSampled"]


def test_a_seed_in_other_poses_is_refused_and_the_schedule_is_left_alone(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)
    execute(train_recipe(stand_in_params()), workdir, RunnerSet(cpu=LocalRunner()))
    # The poses are recomputed between the preview and the Refine: a different frame.
    (workdir.input_path("poses") / "images.bin").write_bytes(b"other poses")

    execute(
        train_recipe(stand_in_params(init_from="preview", schedule_scale=1.0)),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )

    document = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert document["settings"]["initFrom"] == "sfm"
    assert document["init"] is None
    assert document["requestedIterations"] == 300
    assert "trained against other poses" in workdir.log_path("train").read_text()


def test_no_seed_trains_as_asked_and_says_why(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    execute(
        train_recipe(stand_in_params(init_from="preview")), workdir, RunnerSet(cpu=LocalRunner())
    )

    document = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert document["settings"]["initFrom"] == "sfm"
    assert "no earlier run of this stage left a seed" in workdir.log_path("train").read_text()


def test_an_unknown_init_is_refused(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    with pytest.raises(Exception, match="init_from must be sfm or preview"):
        execute(
            train_recipe(stand_in_params(init_from="random")),
            workdir,
            RunnerSet(cpu=LocalRunner()),
        )


def _seed(xyz: np.ndarray) -> init_seed.Seed:
    count = xyz.shape[0]
    return init_seed.Seed(
        xyz=xyz.astype(np.float32),
        rgb=np.full((count, 3), 200, dtype=np.uint8),
        alpha=np.ones(count, dtype=np.float32),
        meta={"trainedWith": {}},
    )


def test_duplicates_are_dropped_and_the_budget_is_kept(tmp_path: Path) -> None:
    """Three coincident points make the trainer's kNN scale log(0) = -inf."""
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    sfm.write_points3d(sparse / "points3D.bin", ring_points(10, images=2, radius=1.0))
    xyz = np.random.default_rng(0).normal(size=(1000, 3))
    xyz[500:600] = xyz[0]  # MCMC relocation copies positions exactly

    applied = init_seed.apply(sparse, _seed(xyz), None, budget=300)

    assert applied.duplicates == 100
    assert applied.seeded == 300
    points = sfm.read_points3d(sparse / "points3D.bin")
    assert len(points) == 310
    assert np.unique(points.xyz[10:], axis=0).shape[0] == 300


def test_the_budget_is_half_the_cap_unless_given() -> None:
    assert init_seed.budget_for(500_000, None) == 250_000
    assert init_seed.budget_for(500_000, 1234) == 1234
    assert init_seed.budget_for(None, None) == init_seed.DEFAULT_BUDGET


def test_gsplats_own_points_reader_accepts_an_empty_track(tmp_path: Path) -> None:
    """The record layout of pycolmap's `SceneManager._load_points3D_bin` at the commit
    gsplat v1.5.3's `examples/requirements.txt` pins (rmbrualla/pycolmap@cc7ea4b), read
    at that commit: `<Q 3d 3B d Q` per point, then `2 * track_len` uint32s -- reshaped to
    `(track_len, 2)`, which for zero is `(0, 2)` and contributes no `point_indices`."""
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    sfm.write_points3d(sparse / "points3D.bin", ring_points(5, images=2, radius=1.0))
    init_seed.apply(sparse, _seed(np.eye(3)), None, budget=10)

    record = struct.Struct("<Q 3d 3B d Q")
    point_indices: dict[int, list[int]] = {}
    with (sparse / "points3D.bin").open("rb") as handle:
        count = struct.unpack("L", handle.read(8))[0]
        for index in range(count):
            data = record.unpack(handle.read(record.size))
            track_len = data[8]
            raw = struct.unpack(f"{2 * track_len}I", handle.read(2 * track_len * 4))
            track = np.array(raw, dtype=np.uint32).reshape(track_len, 2)
            for image_id, _ in track:
                point_indices.setdefault(int(image_id), []).append(index)
        assert handle.read() == b""
    assert count == 8
    assert sorted(i for v in point_indices.values() for i in v) == [0, 1, 2, 3, 4]
