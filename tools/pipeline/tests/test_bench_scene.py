import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

import sfm
from experiments.bench.rehearsal import synthetic_scene
from experiments.bench.scene import prepare_pool, read_json, split_frames, verify, write_json


@pytest.mark.parametrize("count,expected", [(1, 1), (3, 1), (8, 1), (12, 2), (16, 2), (100, 13)])
def test_split_is_stable_disjoint_covering_and_hash_ranked(count: int, expected: int) -> None:
    names = [f"frame{i}.jpg" for i in range(count)]
    split = split_frames("spool", names)
    assert split == split_frames("spool", names)
    assert split == split_frames("spool", list(reversed(names)))
    assert len(split["test"]) == expected
    assert not set(split["test"]) & set(split["pool"])
    assert set(split["test"]) | set(split["pool"]) == set(names)
    ranked = sorted(names, key=lambda n: hashlib.sha256(f"spool:{n}".encode()).digest())
    assert set(split["test"]) == set(ranked[:expected])
    assert split["digest"] != split_frames("spool", [*names, "another.jpg"])["digest"]


def test_split_rejects_empty_or_duplicate_names() -> None:
    for names in ([], ["a", "a"]):
        with pytest.raises(ValueError):
            split_frames("spool", names)


def test_reduced_model_keeps_pool_poses_and_cameras_but_no_test_observations(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene"
    synthetic_scene(scene, "spool", "commit")
    _, split = verify(scene)
    original = sfm.read_model(scene / "poses")
    prepare_pool(scene, tmp_path / "inputs", split)
    poses = tmp_path / "inputs/poses"
    reduced = sfm.read_model(poses)
    assert reduced.cameras == original.cameras
    assert reduced.images == tuple(i for i in original.images if i.name in split["pool"])
    assert {p.name for p in (tmp_path / "inputs/frames").iterdir()} == set(split["pool"])
    points = sfm.read_points3d(poses / "points3D.bin")
    kept_ids = {i.id for i in reduced.images}
    assert set(points.track[:, 0]) == kept_ids
    assert len(points.track) == len(split["pool"]) * len(points)
    assert not (poses / "poses.json").exists()
    assert (poses / "cameras.bin").read_bytes() == (scene / "poses/cameras.bin").read_bytes()


def test_points_seen_only_by_test_images_are_removed(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    synthetic_scene(scene, "spool", "commit")
    # One point, visible only in frame_000; it must not initialise the trainer.
    poses = scene / "poses"
    points = sfm.read_points3d(poses / "points3D.bin").subset(np.array([True] + [False] * 7))
    points = sfm.Points3D(
        points.ids,
        points.xyz,
        points.rgb,
        points.error,
        np.array([[1, 0]], dtype=np.uint32),
        np.array([0, 1], dtype=np.int64),
    )
    sfm.write_points3d(poses / "points3D.bin", points)
    sfm.keep_images(poses, {"frame_001.png"})
    assert len(sfm.read_points3d(poses / "points3D.bin")) == 0
    assert struct.unpack("<Q", (poses / "images.bin").read_bytes()[:8]) == (1,)


@pytest.mark.parametrize("change", ["frame", "split", "manifest", "extra"])
def test_run_refuses_changed_frozen_input_or_split(tmp_path: Path, change: str) -> None:
    scene = tmp_path / "scene"
    manifest = synthetic_scene(scene, "spool", "commit")
    if change == "frame":
        (scene / "frames/frame_000.png").write_bytes(b"changed")
    elif change == "extra":
        (scene / "frames/extra.png").write_bytes(b"extra")
    else:
        path = scene / f"{change}.json"
        data = read_json(path)
        data["digest"] = "changed"
        write_json(path, data)
    with pytest.raises(ValueError, match="digest changed"):
        verify(scene, manifest["digest"])


def test_batch_digest_pins_refreeze(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    synthetic_scene(scene, "spool", "commit")
    with pytest.raises(ValueError, match="since this batch"):
        verify(scene, "old-manifest")
