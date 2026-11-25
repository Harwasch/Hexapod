"""Known-surface depth geometry and real parser/loss checks in cached trainer Python."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2  # type: ignore[import-not-found,unused-ignore]
import numpy as np
import pycolmap  # type: ignore[import-not-found,unused-ignore]
import torch
from PIL import Image

import arkit_depth
import blocks
import capture_bundle
import depth_trainer
import sfm
import training


def hashes(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def geometry() -> None:
    camera = sfm.Camera(1, "PINHOLE", 80, 60, (50, 50, 40, 30), ("fx", "fy", "cx", "cy"))
    # A wall at world z=-2. Test ARKit identity and an accepted-style final pose.
    cases: list[tuple[float, list[float]]] = [(0.0, [0, 0, 0]), (0.02, [0.03, -0.01, 0.02])]
    for angle, centre in cases:
        c, s = np.cos(angle), np.sin(angle)
        r_cw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = r_cw, centre
        rotation, translation = capture_bundle.world_to_camera(transform)
        pose = sfm.Image(
            1,
            "frame_00000000.png",
            tuple(capture_bundle.quaternion(rotation)),
            tuple(translation),
            1,
            0,
            0,
        )
        y, x = np.indices((6, 8))
        # Independent ARKit rays: +Y up, forward -Z; intersect the known wall.
        arkit_rays = np.stack([(x - 4) / 5, -(y - 3) / 5, -np.ones_like(x)], axis=-1)
        world_rays = arkit_rays @ r_cw.T
        z = (-2 - centre[2]) / world_rays[..., 2]
        expected = np.asarray(centre) + world_rays * z[..., None]
        confidence = np.full(z.shape, 2, dtype=np.uint8)
        confidence[0, :2] = [0, 1]
        z[0, 2:5] = [np.nan, np.inf, -1]
        xyz, xy = arkit_depth.backproject(z, confidence, camera, pose)
        np.testing.assert_allclose(
            xyz, expected[arkit_depth.valid_depth(z, confidence)], atol=1e-10
        )
        np.testing.assert_allclose(xyz[:, 2], -2, atol=1e-10)
        assert len(xy) == 43
        quantized, _ = arkit_depth.backproject(
            z.astype("<f2").astype(float), confidence, camera, pose
        )
        # Binary16 spacing at 2--4 m is 1/512 m; max rounding error is 1/1024 m.
        np.testing.assert_allclose(quantized[:, 2], -2, atol=1 / 1024)
    # Nonzero OPENCV distortion and an explicit crop offset: independent forward polynomial.
    raw = replace(camera, model="OPENCV", params=(*camera.params, 0.08, -0.02, 0.003, -0.002))
    target = sfm.Camera(1, "PINHOLE", 65, 45, (52, 51, 31, 21), camera.param_names)
    depth = np.arange(48).reshape(6, 8).astype(np.float32) + 1
    confidence = np.full((6, 8), 2, dtype=np.uint8)
    confidence[1:3, 2:4] = 1
    actual = arkit_depth.resample(depth, confidence, raw, target)
    y, x = np.indices(actual.shape)
    a, b = (x - 31) / 52, (y - 21) / 51
    r2 = a * a + b * b
    radial = 1 + 0.08 * r2 - 0.02 * r2 * r2
    u = 5 * (a * radial + 2 * 0.003 * a * b - 0.002 * (r2 + 2 * a * a)) + 4
    v = 5 * (b * radial + 0.003 * (r2 + 2 * b * b) - 2 * 0.002 * a * b) + 3
    u, v = np.rint(u).astype(int), np.rint(v).astype(int)
    expected = np.zeros(actual.shape)
    valid = (u >= 0) & (u < 8) & (v >= 0) & (v < 6)
    expected[valid] = np.where(confidence[v[valid], u[valid]] == 2, depth[v[valid], u[valid]], 0)
    np.testing.assert_array_equal(actual, expected)
    # The inverse OPENCV rays must project to their original native-depth coordinates.
    pose = sfm.Image(1, "a", (1, 0, 0, 0), (0, 0, 0), 1, 0, 0)
    xyz, xy = arkit_depth.backproject(depth, confidence, raw, pose)
    k, coefficients = arkit_depth.calibration(raw, 8, 6)
    projected, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, coefficients)
    np.testing.assert_allclose(projected.reshape(-1, 2), xy, atol=2e-5)


def loss() -> None:
    rendered = torch.full((1, 4, 6, 1), 3.0, requires_grad=True)
    target = torch.tensor([[[1.0, 0.0, float("nan")], [5.0, -1.0, 2.0]]])
    assert depth_trainer.depth_loss(rendered, target, 999, 1000, 0.01).item() == 0
    value = depth_trainer.depth_loss(rendered, target, 1000, 1000, 0.01)
    torch.testing.assert_close(value, torch.tensor(0.01 * 5 / 3))
    value.backward()
    assert torch.isfinite(rendered.grad).all()
    assert depth_trainer.depth_loss(rendered, torch.zeros_like(target), 1000, 1000, 0.01) == 0
    # Downsampling uses half-pixel centres, not corner alignment.
    grid = torch.arange(24.0).reshape(1, 4, 6, 1)
    sampled = torch.tensor([[[3.5, 5.5, 7.5], [15.5, 17.5, 19.5]]])
    assert depth_trainer.depth_loss(grid, sampled, 1000, 1000, 1) == 0


def integration(root: Path) -> None:
    frames, poses, payload = root / "frames", root / "poses", root / "payload"
    for directory in (frames, poses, payload):
        directory.mkdir()
    names = [f"frame_{i:08d}.png" for i in range(6)]
    (poses / "cameras.txt").write_text("1 OPENCV 80 60 50 50 40 30 .02 -.01 .001 -.002\n")
    (poses / "images.txt").write_text(
        "".join(
            f"{i + 1} 0 1 0 0 {-i * 0.01} 0 0 1 {name}\n40 30 1\n" for i, name in enumerate(names)
        )
    )
    (poses / "points3D.txt").write_text(
        "1 0 0 -2 10 20 30 0 " + " ".join(f"{i + 1} 0" for i in range(6)) + "\n"
    )
    pycolmap.Reconstruction(poses).write_binary(poses)
    (poses / "poses.json").write_text(
        json.dumps(
            {
                "source": "hex-capture-v1",
                "refinement": {
                    "status": "accepted",
                    "lensEnabled": True,
                    "corrections": {n: {} for n in names},
                },
            }
        )
    )
    records: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        Image.new("RGB", (80, 60), (i * 20, 90, 140)).save(frames / name)
        depth: dict[str, Any] = {
            "file": f"smooth_{i}.f16",
            "confidence": f"smooth_{i}.u8",
            "width": 8,
            "height": 6,
        }
        np.full((6, 8), 2, dtype="<f2").tofile(payload / depth["file"])
        np.full((6, 8), 2, dtype=np.uint8).tofile(payload / depth["confidence"])
        records.append({"index": i, "smoothedDepth": depth})
    arkit_depth.export(frames, payload, records)
    original = hashes(poses)
    params = {
        "test_every": 2,
        "arkit_depth_stride": 1,
        "arkit_depth_voxel_m": 0.1,
        "arkit_depth_max_points": 11,
        "arkit_depth_loss": True,
    }
    # Validation depth is deliberately unreadable. It must be excluded before I/O.
    for i in (0, 2, 4):
        (frames / arkit_depth.DIRECTORY / records[i]["smoothedDepth"]["file"]).unlink()
    dataset = training.build_dataset(frames, poses, root / "dataset", max_side=40)
    result = arkit_depth.prepare(frames, poses, dataset, params)
    assert result is not None and result["seedPoints"] == 11 and result["priorFrames"] == 3
    assert result["confidence2Samples"] == 144
    points = sfm.read_points3d(dataset / "sparse" / "0" / "points3D.bin")
    assert len(points) == 12 and len(points.track) == 6
    assert np.diff(points.track_offsets)[1:].sum() == 0
    assert len(np.unique(np.floor(points.xyz[1:] / 0.1), axis=0)) == 11
    assert set(points.rgb[1:, 0]) <= {20, 60, 100}
    np.testing.assert_array_equal(points.rgb[1:, 1:], np.tile([90, 140], (11, 1)))
    assert hashes(poses) == original
    token = training.checkpoint_token(params, dataset)
    training.build_dataset(frames, poses, dataset, max_side=40)
    arkit_depth.prepare(frames, poses, dataset, params)
    assert training.checkpoint_token(params, dataset) == token
    capped = training.build_dataset(frames, poses, root / "capped")
    limited = arkit_depth.prepare(frames, poses, capped, params, cap_max=4)
    assert limited is not None and limited["initialPoints"] == 4
    # Init-only resume must notice changed measurements even when their mtimes and
    # sizes are preserved (derived point counts need not change either).
    seed_only = {**params, "arkit_depth_loss": False}
    identity_data = training.build_dataset(frames, poses, root / "identity")
    arkit_depth.prepare(frames, poses, identity_data, seed_only)
    old_token = training.checkpoint_token(seed_only, identity_data)
    changed = frames / arkit_depth.DIRECTORY / "smooth_1.f16"
    old_bytes, stamp = changed.read_bytes(), changed.stat()
    np.full((6, 8), 2.5, dtype="<f2").tofile(changed)
    os.utime(changed, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    training.build_dataset(frames, poses, identity_data)
    arkit_depth.prepare(frames, poses, identity_data, seed_only)
    assert training.checkpoint_token(seed_only, identity_data) != old_token
    changed.write_bytes(old_bytes)
    os.utime(changed, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    # Load through the real pinned parser, with final rectified calibration and split.
    sys.path.insert(0, os.environ["ARKIT_TEST_GSPLAT_EXAMPLES"])
    from datasets.colmap import Dataset, Parser

    parser = Parser(str(dataset), factor=1, normalize=False, test_every=2)
    wrapped = depth_trainer.DepthDataset(
        Dataset(parser, split="train"), dataset / arkit_depth.PRIORS
    )
    assert len(wrapped) == 3
    for i in range(len(wrapped)):
        sample = wrapped[i]
        assert "arkit_depth" in sample and sample["arkit_depth"].max() == 2
    assert "arkit_depth" not in Dataset(parser, split="val")[0]
    runner = SimpleNamespace(
        cfg=SimpleNamespace(
            normalize_world_space=False, patch_size=None, batch_size=1, data_dir=str(dataset)
        ),
        trainset=Dataset(parser, split="train"),
    )
    depth_trainer.configure(runner)
    assert isinstance(runner.trainset, depth_trainer.DepthDataset)
    # Block renaming carries only the prepared training priors; validation still has none.
    blocks.build_block_dataset(dataset, root / "block", [(names[0], True), (names[1], False)])
    assert sorted(p.name for p in (root / "block" / arkit_depth.PRIORS).iterdir()) == [
        "00001_" + names[1] + ".npy"
    ]
    # External holdouts are physically excluded, including their manifest entries.
    scene = root / "scene"
    scene.mkdir()
    import shutil

    shutil.copytree(frames, scene / "frames")
    shutil.copytree(poses, scene / "poses")
    (root / "inputs" / "frames").mkdir(parents=True)
    arkit_depth.subset(scene / "frames", root / "inputs" / "frames", {names[1], names[3], names[5]})
    manifest = json.loads(
        (root / "inputs" / "frames" / arkit_depth.DIRECTORY / "manifest.json").read_text()
    )
    assert set(manifest) == {names[1], names[3], names[5]}
    # Both flags off reproduces PR1's private dataset exactly; no source mutation.
    off = training.build_dataset(frames, poses, root / "off")
    before = hashes(off)
    assert arkit_depth.prepare(frames, poses, off, {"arkit_depth_init": False}) is None
    assert hashes(off) == before
    # Ordinary OPENCV inputs never enable either path, even with explicit flags.
    ordinary = root / "ordinary"
    shutil.copytree(frames, ordinary)
    shutil.rmtree(ordinary / arkit_depth.DIRECTORY)
    clean = training.build_dataset(ordinary, poses, root / "nonbundle")
    before = hashes(clean)
    assert (
        arkit_depth.prepare(
            ordinary, poses, clean, {"arkit_depth_init": True, "arkit_depth_loss": True}
        )
        is None
    )
    assert hashes(clean) == before


if __name__ == "__main__":
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    geometry()
    loss()
    integration(Path(sys.argv[1]))
    sys.stdout.write(
        "depth geometry, confidence, voxel/cap, rectification, lazy loss, "
        "splits, resume, off-path parity: OK\n"
    )
