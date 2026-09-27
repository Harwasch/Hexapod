"""`live.py`: the lines that let a person watch cameras being solved and a splat train.

The cameras half is exercised against COLMAP-format models written here (and, in
`test_pose_colmap.py`, against the real mapper's snapshots); the splat half against the
gsplat stand-in, which writes the intermediate PLYs `--ply_steps` asks for. Nothing here
trains anything.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import live
import sfm
import training
import tree_frames
from adapters import LocalTransfer, SubprocessAdapter
from captures_bridge import unpack_spz
from cloud import CloudRunner, Placement
from conftest import make_recipe
from executor import execute
from runners import LocalRunner, RunnerSet
from workdir import Workdir

STAND_IN = Path(__file__).resolve().parent / "gsplat_stand_in.py"


# --- a COLMAP model on disk ----------------------------------------------------------


def write_model(directory: Path, poses: tuple[tree_frames.Pose, ...], points: int = 50) -> None:
    """cameras.bin, images.bin and points3D.bin in COLMAP's binary layout."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<IiQQ", 1, 2, 640, 480))
        handle.write(struct.pack("<4d", 500.0, 320.0, 240.0, 0.0))
    with (directory / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(poses)))
        for index, pose in enumerate(poses, start=1):
            handle.write(struct.pack("<I4d3dI", index, *pose.qvec, *pose.translation, 1))
            handle.write(pose.name.encode() + b"\x00")
            handle.write(struct.pack("<Q", 0))
    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(points, 3)) + np.array([0.0, 0.0, 3.0])
    count = xyz.shape[0]
    sfm.write_points3d(
        directory / "points3D.bin",
        sfm.Points3D(
            ids=np.arange(1, count + 1, dtype=np.uint64),
            xyz=xyz,
            rgb=np.tile(np.array([[200, 120, 40]], dtype=np.uint8), (count, 1)),
            error=np.zeros(count),
            track=np.tile(np.array([[1, 0]], dtype=np.uint32), (3 * count, 1)),
            track_offsets=np.arange(0, 3 * count + 1, 3, dtype=np.int64),
        ),
    )


def test_cameras_round_trip_through_the_line_to_within_quantisation(tmp_path: Path) -> None:
    poses = tree_frames.orbit(12)
    write_model(tmp_path / "model", poses)

    lines: list[str] = []
    payload = live.emit_model(lines.append, tmp_path / "model", frames=20)

    assert payload is not None and len(lines) == 1
    kind, parsed = live.parse_line(lines[0]) or ("", {})
    assert kind == "cameras"
    assert parsed["registered"] == 12 and parsed["frames"] == 20 and parsed["final"] is True
    assert parsed["cameraCount"] == 12 and parsed["pointCount"] == 50
    assert parsed["aspect"] == pytest.approx(640 / 480, abs=1e-3)
    centres, forward, up = live.decode_cameras(parsed)
    # In name order, which is capture order.
    truth = np.stack([pose.centre for pose in poses])
    step = parsed["scale"] / 32767
    assert np.abs(centres - truth).max() <= step
    assert np.abs(forward - np.stack([pose.rotation[2] for pose in poses])).max() < 0.01
    assert np.abs(up - np.stack([-pose.rotation[1] for pose in poses])).max() < 0.01
    # The orbit's cameras are level, so the scene's up is the world's +z.
    assert parsed["up"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-3)


def test_the_line_stays_small_however_big_the_model_is(tmp_path: Path) -> None:
    write_model(tmp_path / "model", tree_frames.orbit(1_000), points=5_000)

    lines: list[str] = []
    payload = live.emit_model(lines.append, tmp_path / "model")

    assert payload is not None
    assert payload["cameraCount"] == live.MAX_CAMERAS
    assert payload["pointCount"] == live.MAX_POINTS and payload["pointsTotal"] == 5_000
    assert len(lines[0]) < 30_000


def test_a_model_that_cannot_be_read_logs_why_and_raises_nothing(tmp_path: Path) -> None:
    lines: list[str] = []

    assert live.emit_model(lines.append, tmp_path / "missing") is None
    assert lines and lines[0].startswith("live: could not read")
    assert live.latest("\n".join(lines)) == {}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_mapper_watch_logs_the_newest_snapshot_at_most_every_few_seconds(
    tmp_path: Path,
) -> None:
    lines: list[str] = []
    clock = Clock()
    snapshots = tmp_path / "snapshots"
    watch = live.MapperWatch(snapshots, lines.append, frames=12, every_s=5.0, clock=clock)
    # Entered by hand so the thread does not race the test; `poll` is what it calls.
    snapshots.mkdir()
    assert watch.poll() is False  # nothing written yet

    # COLMAP's snapshot names are an overflowed millisecond timestamp: newest is not the
    # largest name, so the watcher goes by mtime.
    write_model(snapshots / "-5000", tree_frames.orbit(12)[:4])
    assert watch.poll() is True
    assert live.latest("\n".join(lines))["cameras"]["registered"] == 4

    write_model(snapshots / "-9000", tree_frames.orbit(12)[:8])
    stamp = (snapshots / "-5000" / "points3D.bin").stat().st_mtime + 1
    os.utime(snapshots / "-9000" / "points3D.bin", (stamp, stamp))
    clock.now = 2.0
    assert watch.poll() is False  # throttled
    clock.now = 6.0
    assert watch.poll() is True
    assert live.latest("\n".join(lines))["cameras"]["registered"] == 8
    # The one it moved past is gone; the disk does not fill with copies of the solve.
    assert sorted(p.name for p in snapshots.iterdir()) == ["-9000"]
    clock.now = 20.0
    assert watch.poll() is False  # nothing new


def test_a_snapshot_still_being_written_is_read_on_the_next_tick(tmp_path: Path) -> None:
    lines: list[str] = []
    snapshots = tmp_path / "snapshots"
    write_model(snapshots / "-1", tree_frames.orbit(6))
    points = snapshots / "-1" / "points3D.bin"
    whole = points.read_bytes()
    points.write_bytes(whole[: len(whole) // 2])
    watch = live.MapperWatch(snapshots, lines.append, every_s=0.0)

    assert watch.poll() is False and lines == []
    points.write_bytes(whole)
    assert watch.poll() is True


def test_the_watch_thread_runs_beside_the_tool_and_cleans_up(tmp_path: Path) -> None:
    lines: list[str] = []
    snapshots = tmp_path / "snapshots"
    with live.MapperWatch(snapshots, lines.append, every_s=0.0, poll_s=0.01) as watch:
        write_model(snapshots / "-1", tree_frames.orbit(6))
        for _ in range(500):
            if watch.emitted:
                break
            time.sleep(0.01)
    assert watch.emitted == 1
    assert not snapshots.exists()


def test_the_mapper_argv_asks_for_snapshots_only_when_told_to() -> None:
    plain = sfm.mapper_argv(Path("/db"), Path("/i"), Path("/o"))
    assert "--Mapper.snapshot_path" not in plain
    argv = sfm.mapper_argv(
        Path("/db"), Path("/i"), Path("/o"), snapshot_path=Path("/s"), snapshot_every=3
    )
    assert argv[argv.index("--Mapper.snapshot_path") + 1] == "/s"
    assert argv[argv.index("--Mapper.snapshot_images_freq") + 1] == "3"
    assert live.snapshot_every(40) == 1 and live.snapshot_every(300) == 5


# --- reading lines back ----------------------------------------------------------------


def splat_line(step: int, key: str = "runs/r/train/checkpoint/live/splat_000030_1.spz") -> str:
    payload = {"v": 1, "step": step, "total": 300, "count": 10, "of": 64, "key": key, "up": None}
    return live.SPLAT_TAG + json.dumps(payload)


def test_latest_takes_the_newest_valid_line_of_each_kind() -> None:
    log = "\n".join(
        [
            "$ colmap mapper",
            splat_line(30),
            "loss=0.1| :  10%|#| 30/300 [00:01<00:09, 99.00it/s]",
            splat_line(75),
            # Somebody else's key is never passed on to be signed.
            splat_line(150, key="captures/secret/original.mov"),
            "live-splat: {not json",
        ]
    )

    found = live.latest(log)

    assert set(found) == {"splat"}
    assert found["splat"]["step"] == 75


def test_a_line_the_log_service_split_in_two_is_joined_back(tmp_path: Path) -> None:
    write_model(tmp_path / "model", tree_frames.orbit(12))
    lines: list[str] = []
    live.emit_model(lines.append, tmp_path / "model")
    whole = lines[0]
    log = "\n".join(["before", whole[:5000], whole[5000:], "after"])

    found = live.latest(log)

    assert found["cameras"]["registered"] == 12


def test_a_cameras_line_whose_counts_disagree_with_its_data_is_refused(tmp_path: Path) -> None:
    write_model(tmp_path / "model", tree_frames.orbit(12))
    lines: list[str] = []
    payload = live.emit_model(lines.append, tmp_path / "model")
    assert payload is not None
    bad = {**payload, "cameraCount": 99}

    assert live.parse_line(live.CAMERAS_TAG + json.dumps(bad)) is None


# --- splats --------------------------------------------------------------------------


def write_splat_ply(path: Path, opacity: list[float], log_scale: list[float]) -> None:
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    names += ["scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    count = len(opacity)
    rows = np.zeros(count, dtype=np.dtype([(name, "<f4") for name in names]))
    rows["x"] = np.arange(count, dtype=np.float32)
    rows["opacity"] = opacity
    for axis in range(3):
        rows[f"scale_{axis}"] = log_scale
    rows["rot_0"] = 1.0
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n"
        + "".join(f"property float {name}\n" for name in names)
        + "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode() + rows.tobytes())


def test_a_snapshot_keeps_the_most_opaque_and_largest_gaussians(tmp_path: Path) -> None:
    # Index: 0 opaque+small, 1 faint+large, 2 opaque+large, 3 faint+small, 4 far away.
    write_splat_ply(
        tmp_path / "in.ply",
        opacity=[4.0, -4.0, 4.0, -4.0, 4.0],
        log_scale=[-4.0, -1.0, -1.0, -4.0, 0.0],
    )
    data = (tmp_path / "in.ply").read_bytes()
    # Put the last one out past what SPZ's fixed point can hold.
    marker = data.index(b"end_header\n") + len(b"end_header\n")
    far = bytearray(data)
    struct.pack_into("<f", far, marker + 4 * 14 * 4, 5_000.0)
    (tmp_path / "in.ply").write_bytes(bytes(far))

    kept, of, size = live.snapshot_splat(tmp_path / "in.ply", tmp_path / "out.spz", max_gaussians=2)

    assert (kept, of) == (2, 5)
    assert size == (tmp_path / "out.spz").stat().st_size
    back = unpack_spz((tmp_path / "out.spz").read_bytes())
    assert sorted(float(x) for x in back["x"]) == [1.0, 2.0]


def test_the_splat_watch_waits_for_a_file_to_stop_growing(tmp_path: Path) -> None:
    lines: list[str] = []
    ply_dir = tmp_path / "ply"
    write_splat_ply(ply_dir / "point_cloud_29.ply", [1.0] * 8, [-2.0] * 8)
    write_splat_ply(ply_dir / "point_cloud_299.ply", [1.0] * 8, [-2.0] * 8)
    watch = live.SplatWatch(
        ply_dir,
        tmp_path / "checkpoint" / "live",
        lines.append,
        indices={29, 74},
        total=300,
        key_prefix="runs/r/train/checkpoint/live",
        up=[0.0, -1.0, 0.0],
        now=lambda: 1234.0,
    )

    assert watch.poll() is False  # first sight: the size might still change
    assert watch.poll() is True
    found = live.latest("\n".join(lines))["splat"]
    assert found["step"] == 30 and found["total"] == 300 and found["count"] == 8
    assert found["key"] == "runs/r/train/checkpoint/live/splat_000030_1234.spz"
    assert found["up"] == [0.0, -1.0, 0.0]
    assert (tmp_path / "checkpoint" / "live" / "splat_000030_1234.spz").is_file()
    # The full intermediate is gone; the final one is never touched.
    assert sorted(p.name for p in ply_dir.iterdir()) == ["point_cloud_299.ply"]
    assert training.latest_ply(ply_dir) == ply_dir / "point_cloud_299.ply"


def test_live_ply_steps_are_fractions_of_the_unscaled_schedule() -> None:
    assert live.train_ply_steps(30_000) == [3_000, 7_500, 15_000, 22_500]
    assert live.ply_indices([3_000, 7_500], 0.5) == {1_499, 3_749}
    argv = training.gsplat_argv(
        "python3", Path("/t.py"), Path("/d"), Path("/r"), max_steps=30_000, live_steps=[3_000]
    )
    at = argv.index("--ply_steps")
    assert argv[at + 1 : at + 3] == ["30000", "3000"]
    assert argv[at + 3] == "--save_steps"


def seed_inputs(workdir: Workdir) -> None:
    frame_dir = workdir.input_path("frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        (frame_dir / f"frame_{index:04d}.jpg").write_bytes(b"\xff\xd8\xff not a real jpeg")
    poses = workdir.input_path("poses")
    poses.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (poses / name).write_bytes(b"colmap")
    (poses / "poses.json").write_text(
        json.dumps({"registered": 4, "upEstimate": {"up": [0.0, -1.0, 0.0]}}), encoding="utf-8"
    )


def train(workdir: Workdir, **params: object) -> None:
    stage = {
        "id": "train",
        "impl": "gsplat",
        "params": {
            "iterations": 300,
            "trainer": str(STAND_IN),
            "python": sys.executable,
            "extra_args": ["--ckpt-every", "100", "--gaussians", "64"],
            **params,
        },
    }
    execute(make_recipe([stage], inputs=["frames", "poses"]), workdir, RunnerSet(cpu=LocalRunner()))


def test_the_train_stage_snapshots_intermediate_splats_and_keeps_the_final_one(
    tmp_path: Path,
) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    train(workdir)

    log = workdir.log_path("train").read_text()
    assert "--ply_steps 300 30 75 150 225" in log
    found = live.latest(log)["splat"]
    # The stand-in is faster than the watcher's poll, so the stage packs the newest
    # intermediate when the trainer exits: the 75% one, never the final.
    assert found["step"] in {30, 75, 150, 225}
    assert found["total"] == 300 and found["of"] == 64 and found["up"] == [0.0, -1.0, 0.0]
    assert found["key"].startswith("runs/run/train/checkpoint/live/splat_")
    snapshot = workdir.checkpoint_dir("train") / "live" / found["key"].rsplit("/", 1)[1]
    assert unpack_spz(snapshot.read_bytes())["x"].shape[0] == 64
    # Only the final PLY is left for `latest_ply`, and it is what was published.
    plys = sorted(p.name for p in (workdir.work_dir("train") / "gsplat" / "ply").iterdir())
    assert plys == ["point_cloud_299.ply"]
    metrics = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert metrics["gaussiansInPly"] == 64


def test_live_false_trains_exactly_as_before(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    train(workdir, live=False)

    log = workdir.log_path("train").read_text()
    assert "--ply_steps 300 --save_steps" in log
    assert live.latest(log) == {}
    assert not any(workdir.checkpoint_dir("train").rglob("*.spz"))


def test_on_a_remote_box_the_snapshot_is_uploaded_under_the_key_its_line_names(
    tmp_path: Path,
) -> None:
    """Through `CloudRunner` and a real subprocess, as a GPU box runs it: the line reaches
    the local stage log through the log tail, and the key it names is an object the
    checkpoint syncer put in the bucket -- the object the API signs for the viewer."""
    bucket = LocalTransfer(tmp_path / "bucket")
    adapter = SubprocessAdapter(bucket, tmp_path / "sandbox")
    cloud = CloudRunner(Placement((adapter,)), bucket, poll_interval_s=0.05, checkpoint_every_s=0.2)
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)
    stage = {
        "id": "train",
        "impl": "gsplat",
        "gpu": {"tier": "l4"},
        "params": {
            "iterations": 300,
            "trainer": str(STAND_IN),
            "python": sys.executable,
            "extra_args": ["--ckpt-every", "100", "--gaussians", "64"],
        },
    }

    execute(make_recipe([stage], inputs=["frames", "poses"]), workdir, RunnerSet.cloud(cloud))

    found = live.latest(workdir.log_path("train").read_text())["splat"]
    assert bucket.exists(found["key"])
    assert (bucket.root / found["key"]).read_bytes()[:2] == b"\x1f\x8b"  # gzip: an SPZ
