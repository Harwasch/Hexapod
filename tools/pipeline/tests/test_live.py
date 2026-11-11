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
from typing import Any

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
    assert parsed["fovY"] == pytest.approx(np.degrees(2 * np.arctan(480 / 1000)), abs=1e-3)
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


def test_the_mapper_argv_asks_for_snapshots_only_when_told_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This checks argv, not an installed COLMAP or a running mapper.
    monkeypatch.setattr(sfm, "colmap_exe", lambda: "/stand-in/colmap")
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
    assert live.train_ply_steps(30_000) == list(range(1500, 30000, 1500))
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
            "schedule_scale": 1.0,
            "checkpoint_every": 0,
            "test_every": 8,
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
    write_model(workdir.input_path("poses"), tree_frames.orbit(4))

    train(workdir)

    log = workdir.log_path("train").read_text()
    assert "--ply_steps 300 " + " ".join(map(str, range(15, 300, 15))) in log
    seeds = live.latest(log)["seeds"]
    assert seeds["sources"] == {"sfm": 50, "dense": 0, "seed": 0}
    assert (workdir.checkpoint_dir("train") / "live" / "seeds.bin").stat().st_size == 500
    found = live.latest(log)["splat"]
    # Even a fast producer keeps every snapshot for timeline scrubbing.
    assert found["step"] == 300
    assert len(list((workdir.checkpoint_dir("train") / "live").glob("*.spz"))) == 20
    assert found["total"] == 300 and found["of"] == 64 and found["up"] == [0.0, -1.0, 0.0]
    assert found["key"].startswith("runs/run/train/checkpoint/live/splat_")
    snapshot = workdir.checkpoint_dir("train") / "live" / found["key"].rsplit("/", 1)[1]
    assert unpack_spz(snapshot.read_bytes())["x"].shape[0] == 64
    # Only the final PLY is left for `latest_ply`, and it is what was published.
    plys = sorted(p.name for p in (workdir.work_dir("train") / "gsplat" / "ply").iterdir())
    assert plys == ["point_cloud_299.ply"]
    metrics = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert metrics["gaussiansInPly"] == 64
    tiles = live.latest(log)["tiles"]
    assert tiles["step"] == 300 and tiles["final"] and tiles["gaussians"] > 0
    assert (workdir.checkpoint_dir("train") / "live" / "tiles" / "300" / "tileset.json").is_file()


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
            "schedule_scale": 1.0,
            "checkpoint_every": 0,
            "test_every": 8,
            "trainer": str(STAND_IN),
            "python": sys.executable,
            "extra_args": ["--ckpt-every", "100", "--gaussians", "64"],
        },
    }

    execute(make_recipe([stage], inputs=["frames", "poses"]), workdir, RunnerSet.cloud(cloud))

    found = live.latest(workdir.log_path("train").read_text())["splat"]
    tiles = live.latest(workdir.log_path("train").read_text())["tiles"]
    assert bucket.exists(tiles["key"])
    manifest = json.loads((bucket.root / tiles["key"]).read_text())
    assert (
        (bucket.root / tiles["key"]).parent.joinpath(manifest["root"]["content"]["uri"]).is_file()
    )
    assert bucket.exists(found["key"])
    assert (bucket.root / found["key"]).read_bytes()[:2] == b"\x1f\x8b"  # gzip: an SPZ


def test_records_only_parses_live_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    parser = Mock(wraps=live.parse_line)
    monkeypatch.setattr(live, "parse_line", parser)
    line = 'live-train: {"v":1,"k":"s","step":200}'
    assert live.records("ordinary output\n" * 1000 + line + "\nmore output") == [
        ("train", {"v": 1, "k": "s", "step": 200})
    ]
    assert parser.call_count == 1


def test_final_snapshot_is_packed_without_changing_the_final_ply(tmp_path: Path) -> None:
    ply_dir = tmp_path / "ply"
    final = ply_dir / "point_cloud_99.ply"
    write_splat_ply(final, [1.0] * 8, [-2.0] * 8)
    before = final.read_bytes()
    lines: list[str] = []
    with live.SplatWatch(
        ply_dir,
        tmp_path / "live",
        lines.append,
        indices={49},
        total=100,
        key_prefix="runs/r/train/checkpoint/live",
        poll_s=1000,
    ):
        pass
    assert final.read_bytes() == before
    assert live.latest("\n".join(lines))["splat"]["step"] == 100


def test_a_crashed_run_leaves_no_intermediate_ply_to_mistake_for_a_result(tmp_path: Path) -> None:
    ply_dir = tmp_path / "ply"
    # The trainer died after writing the step-50 intermediate and before any final PLY.
    write_splat_ply(ply_dir / "point_cloud_49.ply", [1.0] * 8, [-2.0] * 8)
    lines: list[str] = []
    with live.SplatWatch(
        ply_dir,
        tmp_path / "live",
        lines.append,
        indices={49},
        total=100,
        key_prefix="runs/r/train/checkpoint/live",
        poll_s=1000,
    ):
        pass
    assert not list(ply_dir.glob("*.ply"))
    assert live.latest("\n".join(lines))["splat"]["step"] == 50


def test_seeds_are_bounded_deterministic_and_source_tagged(tmp_path: Path) -> None:
    count = 250001
    xyz = np.arange(count * 3, dtype=np.float64).reshape(-1, 3) / 10000
    rgb = np.tile(np.array([10, 20, 30], dtype=np.uint8), (count, 1))
    source = np.arange(count, dtype=np.uint32) % 3
    lines: list[str] = []
    state = np.random.get_state()
    payload = live.write_seeds(
        xyz, rgb, tmp_path, lines.append, key_prefix="runs/r/train/checkpoint/live", source=source
    )
    assert payload is not None and payload["count"] == 200000 and payload["of"] == count
    assert payload["bytes"] == 2000000 and sum(payload["sources"].values()) == count
    first = (tmp_path / "seeds.bin").read_bytes()
    assert live.parse_line(lines[0]) == ("seeds", payload)
    assert (
        live.parse_line(
            lines[0].replace("runs/r/train/checkpoint/live/seeds.bin", "captures/seeds.bin")
        )
        is None
    )
    live.write_seeds(
        xyz, rgb, tmp_path, lines.append, key_prefix="runs/r/train/checkpoint/live", source=source
    )
    assert (tmp_path / "seeds.bin").read_bytes() == first
    assert np.array_equal(state[1], np.random.get_state()[1])
    packed = np.frombuffer(
        first, dtype=[("xyz", "<i2", (3,)), ("rgb", "u1", (3,)), ("source", "u1")]
    )
    decoded = packed["xyz"] * payload["scale"] / 32767 + payload["origin"]
    selected = np.linspace(0, count - 1, 200000, dtype=np.int64)
    assert np.abs(decoded - xyz[selected]).max() <= payload["scale"] / 32767
    assert np.array_equal(packed["rgb"], rgb[selected])
    assert np.array_equal(packed["source"], source[selected])


def test_seeds_failure_is_only_a_missing_observation(tmp_path: Path) -> None:
    def broken(_line: str) -> None:
        raise OSError("log unavailable")

    assert (
        live.write_seeds(
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            tmp_path,
            broken,
            key_prefix="runs/r/train/checkpoint/live",
        )
        is None
    )
    live.emit_seeds(
        tmp_path / "missing", tmp_path, broken, key_prefix="runs/r/train/checkpoint/live"
    )


def test_dataset_seed_sources_include_the_appended_preview(tmp_path: Path) -> None:
    model = tmp_path / "sparse"
    write_model(model, tree_frames.orbit(3), points=10)
    lines: list[str] = []
    live.emit_seeds(
        model, tmp_path / "live", lines.append, key_prefix="runs/r/train/checkpoint/live", seeded=3
    )
    assert live.latest("\n".join(lines))["seeds"]["sources"] == {"sfm": 7, "dense": 0, "seed": 3}


def test_final_pack_failure_never_fails_training_or_removes_the_ply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "ply" / "point_cloud_99.ply"
    write_splat_ply(final, [1.0], [-2.0])
    before = final.read_bytes()

    def broken(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("packer unavailable")

    monkeypatch.setattr(live, "snapshot_splat", broken)
    with live.SplatWatch(
        final.parent,
        tmp_path / "live",
        broken,
        indices={49},
        total=100,
        key_prefix="runs/r/train/checkpoint/live",
        poll_s=1000,
    ):
        pass
    assert final.read_bytes() == before


def test_thumbnail_keeps_fine_opaque_ground_over_huge_needles(tmp_path: Path) -> None:
    path = tmp_path / "needles.ply"
    write_splat_ply(path, [5.0] * 120, [float(np.log(0.02))] * 120)
    raw = bytearray(path.read_bytes())
    start = raw.index(b"end_header\n") + len(b"end_header\n")
    for i in range(120):
        # Interleave needles and ground in the same spatial cells.
        struct.pack_into("<3f", raw, start + i * 56, (i % 10) * 0.1, (i // 10) * 0.1, 2.0)
        if i % 6 == 0:
            struct.pack_into(
                "<3f", raw, start + i * 56 + 28, np.log(100), np.log(0.001), np.log(0.001)
            )
    path.write_bytes(raw)
    target = tmp_path / "thumb.spz"
    live.snapshot_splat(
        path, target, max_gaussians=40, camera_centres=np.array([[0.0, 0.0, 0.0]]), focal_px=500.0
    )
    first = target.read_bytes()
    live.snapshot_splat(
        path, target, max_gaussians=40, camera_centres=np.array([[0.0, 0.0, 0.0]]), focal_px=500.0
    )
    assert first == target.read_bytes()
    back = unpack_spz(first)
    assert len(back["x"]) == 40
    assert np.count_nonzero(back["scale_0"] < 0) >= 35
    assert np.ptp(back["x"]) > 0.7 and np.ptp(back["y"]) > 0.8


def test_camera_fov_uses_vertical_focal_and_stage_reads_training_centres(tmp_path: Path) -> None:
    from dataclasses import replace

    poses = tree_frames.orbit(4)
    write_model(tmp_path / "model", poses)
    model = sfm.read_model(tmp_path / "model")
    camera = replace(
        model.cameras[0],
        model="PINHOLE",
        params=(500.0, 800.0, 320.0, 240.0),
        param_names=("fx", "fy", "cx", "cy"),
    )
    payload = live.encode_model(
        replace(model, cameras=(camera,)), np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    )
    assert payload["fovY"] == pytest.approx(np.degrees(2 * np.arctan(480 / 1600)), abs=1e-3)
    sampler = live.snapshot_cameras(tmp_path / "model")
    assert sampler["focal_px"] == 500
    np.testing.assert_allclose(sampler["camera_centres"], model.centres())
    assert live.snapshot_cameras(tmp_path / "missing") == {}


def test_live_lod_rotates_and_preserves_the_final_ply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "checkpoint" / "live"
    ply = tmp_path / "work" / "ply"
    lines: list[str] = []
    counts: list[int] = []

    def record(line: str) -> None:
        lines.append(line)
        if line.startswith(live.TILES_TAG):
            counts.append(len(list((directory / "tiles").iterdir())))

    # Fast tiny fixtures deliberately have no training interval to amortize startup.
    monkeypatch.setattr(live.SplatWatch, "_tiles_budget", lambda *_: 10.0)
    with live.SplatWatch(
        ply,
        directory,
        record,
        indices={9, 19},
        total=30,
        key_prefix="runs/r/train/checkpoint/live",
        poll_s=1000,
    ) as watch:
        for index in (9, 19):
            write_splat_ply(ply / f"point_cloud_{index}.ply", [2.0] * 16, [-2.0] * 16)
            assert watch.poll(settled=True)
            assert sorted(p.name for p in (directory / "tiles").iterdir()) == [str(index + 1)]
        final = ply / "point_cloud_29.ply"
        write_splat_ply(final, [2.0] * 16, [-2.0] * 16)
        before = final.read_bytes()
    assert final.read_bytes() == before
    records = [v for k, v in live.records("\n".join(lines)) if k == "tiles"]
    assert [r["step"] for r in records] == [10, 20, 30]
    assert [r["final"] for r in records] == [False, False, True]
    assert max(counts) <= 2
    assert sorted(p.name for p in (directory / "tiles").iterdir()) == ["30"]
    assert not list((tmp_path / "work").glob("live-tiles-*"))
    last_record = records[-1]
    assert (
        last_record["gaussians"] == 16 and last_record["tiles"] == 1 and last_record["seconds"] > 0
    )
    assert last_record["bytes"] == sum(
        p.stat().st_size for p in (directory / "tiles" / "30").iterdir()
    )
    manifest = json.loads((directory / "tiles" / "30" / "tileset.json").read_text())
    # The neutral placement only changes the root. The console uses the solve frame.
    assert manifest["root"]["transform"][12:15] == [6378137.0, 0.0, 0.0]


def test_lod_timeout_and_failure_are_optional_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    ply = tmp_path / "ply"
    lines: list[str] = []
    watch = live.SplatWatch(
        ply,
        tmp_path / "live",
        lines.append,
        indices={9, 19},
        total=30,
        key_prefix="runs/r/train/checkpoint/live",
    )
    monkeypatch.setattr(watch, "_tiles_budget", lambda *_: 0.000001)
    for index in (9, 19):
        write_splat_ply(ply / f"point_cloud_{index}.ply", [2.0] * 16, [-2.0] * 16)
        assert watch.poll(settled=True)
    assert len([line for line in lines if "LOD tiles skipped" in line]) == 1
    assert "tiles" not in live.latest("\n".join(lines))
    assert len(list((tmp_path / "live").glob("*.spz"))) == 2
    assert not list(tmp_path.glob("live-tiles-*"))

    def broken(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "splat_tiles")

    monkeypatch.setattr(watch, "_convert_tiles", broken)
    monkeypatch.setattr(watch, "_tiles_budget", lambda *_: 10.0)
    final = ply / "point_cloud_29.ply"
    write_splat_ply(final, [2.0] * 16, [-2.0] * 16)
    before = final.read_bytes()
    watch._finish()
    assert final.read_bytes() == before
    assert live.latest("\n".join(lines))["splat"]["step"] == 30


def test_tiles_line_rejects_unbounded_or_unsafe_records() -> None:
    valid: dict[str, Any] = {
        "v": 1,
        "key": "runs/r/train/checkpoint/live/tiles/30/tileset.json",
        "step": 30,
        "final": True,
        "gaussians": 16,
        "tiles": 1,
        "bytes": 1024,
        "seconds": 0.5,
    }
    assert live.parse_line(live.TILES_TAG + json.dumps(valid)) == ("tiles", valid)
    retry = {**valid, "key": valid["key"].replace("/checkpoint/", "/checkpoint-a2/")}
    assert live.parse_line(live.TILES_TAG + json.dumps(retry)) == ("tiles", retry)
    for updates in (
        {"key": "captures/tileset.json"},
        {"step": 31},
        {"step": "["},
        {"seconds": float("nan")},
        {"final": 1},
        {"bytes": -1},
        {"extra": "x" * 1024},
        {"key": valid["key"].replace("/30/", "/../30/")},
    ):
        assert live.parse_line(live.TILES_TAG + json.dumps({**valid, **updates})) is None


def test_tiles_packer_failure_does_not_fail_the_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("splat_tiles failed")

    monkeypatch.setattr(live.SplatWatch, "_convert_tiles", broken)
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)
    train(workdir)
    log = workdir.log_path("train").read_text()
    assert live.latest(log)["splat"]["step"] == 300
    assert "tiles" not in live.latest(log)
    assert log.count("LOD tiles skipped") == 1
    assert (workdir.out_dir("train") / "trained.ply").is_file()


def test_real_lod_filter_failure_keeps_thumbnail_and_read_only_final(tmp_path: Path) -> None:
    final = tmp_path / "ply" / "point_cloud_99.ply"
    # The thumbnail accepts finite gaussians; the full packer's opacity filter rejects all.
    write_splat_ply(final, [-20.0] * 8, [-2.0] * 8)
    before = final.read_bytes()
    lines: list[str] = []
    with live.SplatWatch(
        final.parent,
        tmp_path / "live",
        lines.append,
        indices={49},
        total=100,
        key_prefix="runs/r/train/checkpoint/live",
        poll_s=1000,
    ):
        pass
    assert final.read_bytes() == before
    assert live.latest("\n".join(lines))["splat"]["step"] == 100
    assert "tiles" not in live.latest("\n".join(lines))
    assert sum("LOD tiles skipped" in line for line in lines) == 1
    assert not (tmp_path / "live" / "tiles").exists()
    assert not list(tmp_path.glob("live-tiles-*"))


def test_estimated_pack_time_skips_a_short_snapshot_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    watch = live.SplatWatch(
        tmp_path,
        tmp_path / "live",
        lambda _: None,
        indices={9, 19},
        total=30,
        key_prefix="runs/r/train/checkpoint/live",
    )
    watch._started = 0.0
    watch._tiles_seconds = 110.0
    pending = live._Pending(9, tmp_path / "point_cloud_9.ply", 1)
    assert watch._tiles_budget(pending) == 100.0
    convert = Mock()
    monkeypatch.setattr(watch, "_convert_tiles", convert)
    watch._pack_tiles(pending)
    convert.assert_not_called()


def test_a_bad_intermediate_does_not_disable_the_final_lod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ply = tmp_path / "ply"
    directory = tmp_path / "live"
    lines: list[str] = []
    monkeypatch.setattr(live.SplatWatch, "_tiles_budget", lambda *_: 10.0)
    with live.SplatWatch(
        ply,
        directory,
        lines.append,
        indices={9},
        total=20,
        key_prefix="runs/r/train/checkpoint/live",
        poll_s=1000,
    ) as watch:
        write_splat_ply(ply / "point_cloud_9.ply", [-20.0] * 8, [-2.0] * 8)
        assert watch.poll(settled=True)
        write_splat_ply(ply / "point_cloud_19.ply", [2.0] * 8, [-2.0] * 8)
    assert live.latest("\n".join(lines))["tiles"]["final"]
    assert sorted(p.name for p in (directory / "tiles").iterdir()) == ["20"]
    assert sum("LOD tiles skipped" in line for line in lines) == 1


def test_lod_only_at_quarters_and_final_but_every_thumbnail_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ply = tmp_path / "ply"
    watch = live.SplatWatch(
        ply,
        tmp_path / "live",
        lambda _: None,
        indices=live.ply_indices(live.train_ply_steps(100), 1),
        total=100,
        key_prefix="runs/r/train/checkpoint/live",
    )
    packaged: list[int] = []

    def convert(pending: live._Pending, staged: Path, budget: float) -> dict[str, int]:
        packaged.append(pending.index + 1)
        staged.mkdir()
        (staged / "tileset.json").write_text("{}")
        return {"gaussians": 16, "tiles": 1, "bytes": 2}

    monkeypatch.setattr(watch, "_convert_tiles", convert)
    monkeypatch.setattr(watch, "_tiles_budget", lambda *_: 10.0)
    for step in range(5, 100, 5):
        write_splat_ply(ply / f"point_cloud_{step - 1}.ply", [2.0] * 16, [-2.0] * 16)
        watch.poll(settled=True)
    write_splat_ply(ply / "point_cloud_99.ply", [2.0] * 16, [-2.0] * 16)
    watch._finish()
    assert packaged == [25, 50, 75, 100]
    assert len(list((tmp_path / "live").glob("*.spz"))) == 20
    assert [p.name for p in (tmp_path / "live/tiles").iterdir()] == ["100"]
