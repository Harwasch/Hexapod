"""Training on part of a posed capture, and on frames re-sized from the posed photos.

Two things a pose run does not need to be repeated for: a subset of its images (one
capture session of several) and the same photos at another size. `sfm.keep_images` cuts
a binary model to a subset for a trainer's dataset; `sfm.poses_serve_frames` says whether
a model serves frames of another size, and at what scale gsplat's parser will rescale it.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import sfm


def write_tracked_model(directory: Path) -> dict[str, int]:
    """A COLMAP binary model with real 2D points and tracks: images 1-4 (`day-a-1.jpg`,
    `day-a-2.jpg`, `day-b-3.jpg`, `day-b-4.jpg`); point p < 8 seen by images p % 4 + 1
    and (p + 1) % 4 + 1, point 8 by image 3 alone, point 9 by nobody (an empty track)."""
    directory.mkdir(parents=True, exist_ok=True)
    names = {1: "day-a-1.jpg", 2: "day-a-2.jpg", 3: "day-b-3.jpg", 4: "day-b-4.jpg"}
    tracks: dict[int, list[tuple[int, int]]] = {}
    for point in range(9):
        seen = [point % 4 + 1, (point + 1) % 4 + 1] if point < 8 else [3]
        tracks[point + 1] = [(image, point) for image in seen]
    tracks[10] = []
    with (directory / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<IiQQ", 1, 1, 1600, 1066))  # PINHOLE
        handle.write(struct.pack("<4d", 1200.0, 1200.0, 800.0, 533.0))
    with (directory / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(names)))
        for image_id, name in names.items():
            handle.write(struct.pack("<IdddddddI", image_id, 1, 0, 0, 0, image_id, 0, 0, 1))
            handle.write(name.encode() + b"\0")
            observed = [
                (pid, idx) for pid, track in tracks.items() for img, idx in track if img == image_id
            ]
            handle.write(struct.pack("<Q", len(observed)))
            for pid, idx in observed:
                handle.write(struct.pack("<ddQ", float(idx), float(idx), pid))
    ids = np.asarray(sorted(tracks), dtype=np.uint64)
    rows = [np.asarray(tracks[int(i)], dtype=np.uint32).reshape(-1, 2) for i in ids]
    sfm.write_points3d(
        directory / "points3D.bin",
        sfm.Points3D(
            ids=ids,
            xyz=np.zeros((len(ids), 3)),
            rgb=np.zeros((len(ids), 3), dtype=np.uint8),
            error=np.zeros(len(ids)),
            track=np.concatenate(rows).astype(np.uint32),
            track_offsets=np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64),
        ),
    )
    return {name: image_id for image_id, name in names.items()}


def test_a_subset_of_images_cuts_the_tracks_and_the_points_only_others_saw(
    tmp_path: Path,
) -> None:
    ids = write_tracked_model(tmp_path / "model")
    counts = sfm.keep_images(tmp_path / "model", {"day-a-1.jpg", "day-a-2.jpg"})
    # Points 1-8 minus the two seen only by images 3 and 4 (p = 2) and by 3 alone (p = 8),
    # plus point 10, whose empty track says nothing about who saw it.
    assert counts == {"images": 2, "imagesBefore": 4, "points3D": 7, "points3DBefore": 10}
    model = sfm.read_model(tmp_path / "model")
    assert [image.name for image in model.images] == ["day-a-1.jpg", "day-a-2.jpg"]
    assert all(image.points2d > 0 for image in model.images)  # their 2D points kept
    points = sfm.read_points3d(tmp_path / "model" / "points3D.bin")
    assert set(points.track[:, 0].tolist()) <= {ids["day-a-1.jpg"], ids["day-a-2.jpg"]}
    assert 9 not in points.ids.tolist() and 3 not in points.ids.tolist()
    assert 10 in points.ids.tolist()


def test_poses_solved_at_one_size_serve_frames_at_another(tmp_path: Path) -> None:
    write_tracked_model(tmp_path / "model")
    model = sfm.read_model(tmp_path / "model")
    frames = ["day-a-1.jpg", "day-a-2.jpg", "day-b-3.jpg", "day-b-4.jpg", "unposed.jpg"]
    fit = sfm.poses_serve_frames(model, [[2400, 1599]], frames)
    assert fit["pixelScale"] == 1.5 and fit["registered"] == 4 and fit["unposedFrames"] == 1
    assert fit["posedSize"] == [1600, 1066] and fit["frameSize"] == [2400, 1599]
    assert sfm.poses_serve_frames(model, [[800, 533]], frames)["pixelScale"] == 0.5
    with pytest.raises(ValueError, match="same shape"):
        sfm.poses_serve_frames(model, [[2400, 1350]], frames)
    with pytest.raises(ValueError, match="sizes"):
        sfm.poses_serve_frames(model, [[2400, 1599], [1599, 2400]], frames)
    with pytest.raises(ValueError, match="not among these frames"):
        sfm.poses_serve_frames(model, [[2400, 1599]], frames[1:])
