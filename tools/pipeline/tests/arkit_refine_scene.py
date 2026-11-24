"""CPU pycolmap integration fixture; invoked by test_arkit_refine under heavy."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import arkit_refine
import capture_bundle
import sfm


def scene(
    root: Path, pycolmap: Any, *, short_baseline: bool = False, seed: int = 120
) -> tuple[Path, Path, Path, list[dict[str, Any]], np.ndarray]:
    rng = np.random.default_rng(seed)
    frames, source = root / "frames", root / "fixed"
    frames.mkdir()
    source.mkdir()
    points = rng.uniform([-0.7, -0.6, 3], [0.7, 0.6, 5], (120, 3))
    cameras, images, records, centres = [], [], [], []
    observations = []
    count = 80 if short_baseline else 12
    for index in range(count):
        theta = index * 2 * np.pi / count
        radii = [0.25, 0.15, 0.03] if short_baseline else [0.7, 0.35, 0.1]
        centre = np.array(radii) * [np.cos(theta), np.sin(theta), np.sin(2 * theta)]
        centres.append(centre)
        if short_baseline:
            noisy = centre + rng.normal(0, 0.002, 3)
            # 2 mm differential error on a ~13 mm video baseline. All frames
            # remain noisy; no privileged, exact anchor pair like the lens fixture.
            if index < 2:
                noisy = centre + np.array([0, -0.001 if index == 0 else 0.001, 0])
        else:
            noisy = centre + (rng.normal(0, 0.006, 3) if index > 1 else 0)
        rotation = pycolmap.Rotation3d(np.array([0, 0.02 * np.sin(theta), 0])).matrix()
        translation = -rotation @ noisy
        transform = capture_bundle.camera_to_world(rotation, translation)
        k = np.array([[500 + 0.2 * np.sin(theta), 0, 320], [0, 500, 240], [0, 0, 1]])
        records.append(
            {
                "index": index,
                "transform": transform.ravel().tolist(),
                "intrinsics": k.ravel().tolist(),
                "imageWidth": 640,
                "imageHeight": 480,
            }
        )
        cameras.append(f"{index + 1} PINHOLE 640 480 {k[0, 0]} 500 320 240")
        xyz = points - centre
        xy = xyz[:, :2] / xyz[:, 2:] * 500 + [320, 240]
        observations.append(xy)
        name = f"frame_{index:08d}.png"
        Image.new("RGB", (640, 480), (index * 10, 100, 200)).save(frames / name)
        pose = [*capture_bundle.quaternion(rotation), *translation]
        images.append(f"{index + 1} " + " ".join(map(str, pose)) + f" {index + 1} {name}")
        images.append(" ".join(f"{x} {y} {j + 1}" for j, (x, y) in enumerate(xy)))
    (source / "cameras.txt").write_text("\n".join(cameras) + "\n")
    (source / "images.txt").write_text("\n".join(images) + "\n")
    (source / "points3D.txt").write_text(
        "\n".join(
            f"{j + 1} {x} {y} {z} 100 150 200 0 " + " ".join(f"{i + 1} {j}" for i in range(count))
            for j, (x, y, z) in enumerate(points)
        )
        + "\n"
    )
    original = pycolmap.Reconstruction(source)
    original.write_binary(source)
    database = root / "features.db"
    db = pycolmap.Database.open(database)
    for camera in original.cameras.values():
        db.write_camera(camera, use_camera_id=True)
    for i, xy in enumerate(observations):
        image = original.images[i + 1]
        db.write_image(
            pycolmap.Image(image_id=image.image_id, name=image.name, camera_id=image.camera_id),
            use_image_id=True,
        )
        db.write_keypoints(i + 1, xy.astype(np.float32))
    matches = np.column_stack([np.arange(len(points)), np.arange(len(points))]).astype(np.uint32)
    for a in range(1, count + 1):
        for b in range(a + 1, count + 1):
            geometry = pycolmap.TwoViewGeometry()
            geometry.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
            geometry.inlier_matches = matches
            db.write_two_view_geometry(a, b, geometry)
    db.close()
    return source, database, frames, records, np.array(centres)


def run(root: Path) -> None:
    import pycolmap  # type: ignore[import-not-found,unused-ignore]

    source, database, frames, records, truth = scene(root, pycolmap)
    work = root / "refine"
    work.mkdir()
    report = arkit_refine.refine(pycolmap, source, database, frames, work / "model", lens=True)
    (root / "report.json").write_text(json.dumps(report))
    assert report["status"] == "accepted", report.get("fallbackReason")
    model = sfm.read_model(work / "model")
    arkit_refine.verify(model, records, report)
    before = np.array([i.centre for i in sfm.read_model(source).images])
    after = np.array([i.centre for i in model.images])
    assert (
        np.mean(np.linalg.norm(after - truth, axis=1))
        < np.mean(np.linalg.norm(before - truth, axis=1)) * 0.7
    )
    before_rotation = np.mean(
        [arkit_refine.angle(i.rotation) for i in sfm.read_model(source).images]
    )
    after_rotation = np.mean([arkit_refine.angle(i.rotation) for i in model.images])
    assert after_rotation < before_rotation * 0.7
    assert report["reprojectionAfterPx"] < report["reprojectionBeforePx"] * 0.1
    assert len(model.cameras) == 1 and model.cameras[0].model == "OPENCV"
    assert abs(report["sim3"]["scale"] - 1) < 0.005
    # Alignment returns the global centroid and orientation to ARKit's gravity gauge.
    np.testing.assert_allclose(after.mean(0), before.mean(0), atol=1e-10)
    scale, rotation, _ = arkit_refine.align(after, before)
    assert abs(scale - 1) < 1e-10
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-10)
    summary: dict[str, Any] = {
        "source": "hex-capture-v1",
        "refinement": report,
        "images": [i.to_dict() for i in model.images],
    }
    assert arkit_refine.verified_summary(summary)
    (work / "model" / "poses.json").write_text(json.dumps(summary))
    summary["images"][2]["tvec"][0] += 0.02
    assert not arkit_refine.verified_summary(summary)
    pinhole_work = root / "pinhole"
    pinhole_work.mkdir()
    pinhole_report = arkit_refine.refine(
        pycolmap, source, database, frames, pinhole_work / "model", lens=False
    )
    assert pinhole_report["status"] == "accepted", pinhole_report.get("fallbackReason")
    assert sfm.read_model(pinhole_work / "model").cameras == sfm.read_model(source).cameras
    with_undistortion(root, frames, work / "model")


def with_undistortion(root: Path, frames: Path, poses: Path) -> None:
    import cv2  # type: ignore[import-not-found,unused-ignore]

    # Force a lens with a nonzero crop origin: verify the offset even when the BA
    # fixture happens to solve a nearly pinhole lens.
    import pycolmap  # type: ignore[import-not-found,unused-ignore]

    import distortion
    import training
    from quality import Cameras

    model = pycolmap.Reconstruction(poses)
    model.cameras[1].params = [500, 510, 310, 230, 0.2, 0.01, 0.02, -0.01]
    model.write_binary(poses)
    # Exercise a nonzero valid crop even on OpenCV builds where alpha=0 returns
    # origin (0,0). This is an inset of the real valid ROI, not a calibrated tolerance.
    optimal = cv2.getOptimalNewCameraMatrix

    def inset(*args: Any, **kwargs: Any) -> Any:
        matrix, (x, y, w, h) = optimal(*args, **kwargs)
        return matrix, (x + 7, y + 9, w - 14, h - 18)

    cv2.getOptimalNewCameraMatrix = inset
    raw = sfm.read_model(poses)
    # Survey clicks stay in the original distorted capture pixels.
    import check_targets

    target = np.array([0.1, 0.2, 4.0])
    clicks = []
    lens = model.cameras[1]
    lens_k = lens.calibration_matrix()
    for image in raw.images:
        vector, _ = cv2.Rodrigues(image.rotation)
        pixels, _ = cv2.projectPoints(
            target.reshape(1, 3), vector, np.array(image.tvec), lens_k, np.array(lens.params[4:])
        )
        u, v = pixels.reshape(2)
        clicks.append(check_targets.Observation("survey", image.name, float(u), float(v)))
    assert check_targets.triangulate(raw, clicks)["survey"].status == "unsupported-camera"
    surveyed = check_targets.triangulate(raw, clicks, allow_opencv=True)["survey"]
    assert surveyed.status == "ok", surveyed
    assert surveyed.position is not None
    np.testing.assert_allclose(surveyed.position, target, atol=1e-8)

    camera = raw.cameras[0]
    projected = distortion.projection_model(raw, poses)
    expected, k, new, roi = distortion.geometry(camera)
    assert roi[0] or roi[1], roi
    original = (frames / raw.images[0].name).read_bytes()
    Image.new("RGB", (640, 480)).save(frames / "untracked.png")
    dataset = training.build_dataset(frames, poses, root / "dataset")
    assert not (dataset / "images" / "untracked.png").exists()
    token = training.checkpoint_token({}, dataset)
    training.build_dataset(frames, poses, dataset)
    assert training.checkpoint_token({}, dataset) == token
    result = sfm.read_model(dataset / "sparse" / "0")
    assert result.cameras == projected.cameras
    rectified = pycolmap.Reconstruction(dataset / "sparse" / "0")
    first_id = min(rectified.reg_image_ids())
    xy = model.images[first_id].points2D[0].xy
    expected_xy = cv2.undistortPoints(
        np.array(xy).reshape(1, 1, 2), k, np.array(camera.params[4:]), P=new
    ).reshape(2)
    np.testing.assert_allclose(rectified.images[first_id].points2D[0].xy, expected_xy)
    import live

    assert live.snapshot_cameras(poses)["focal_px"] == expected.focal_px
    console = live.read_snapshot(poses)
    assert console["aspect"] == round(expected.width / expected.height, 4)
    assert (frames / raw.images[0].name).read_bytes() == original
    assert sfm.read_model(poses).cameras[0].model == "OPENCV"
    with Image.open(dataset / "images" / raw.images[0].name) as raster:
        assert raster.size == (expected.width, expected.height)
    cam = Cameras.from_model(projected)
    assert cam.cx[0] == new[0, 2]
    # cv2's original full-image K must be shifted by exactly the crop origin.
    full, _ = cv2.getOptimalNewCameraMatrix(k, np.array(camera.params[4:]), (640, 480), 0)
    np.testing.assert_allclose(new[:2, 2], full[:2, 2] - roi[:2])
    # Pinned gsplat parser consumes our PINHOLE, with no double undistortion.
    examples = Path(__import__("os").environ["ARKIT_TEST_GSPLAT_EXAMPLES"])
    sys.path.insert(0, str(examples))
    # Pipeline and trainer normally run in separate interpreters; both have exif.py.
    sys.modules.pop("exif", None)
    from datasets.colmap import Dataset, Parser

    parser = Parser(str(dataset), factor=1, normalize=False, test_every=8)
    item = Dataset(parser, split="val")[0]
    np.testing.assert_allclose(item["K"].numpy(), new, atol=1e-4)
    assert tuple(item["image"].shape[:2]) == (expected.height, expected.width)
    # Smaller training images get intrinsics in their own pixel coordinates.
    small = training.build_dataset(frames, poses, root / "small", max_side=320)
    parser = Parser(str(small), factor=1, normalize=False, test_every=8)
    small_camera = sfm.read_model(small / "sparse" / "0").cameras[0]
    assert parser.imsize_dict[1] == (small_camera.width, small_camera.height)
    # The benchmark strips held-out frames but must retain rectification provenance.
    frozen = root / "frozen"
    import shutil

    shutil.copytree(poses, frozen / "poses")
    shutil.copytree(frames, frozen / "frames")
    # Explicitly prove ordinary OPENCV datasets still copy every original byte.
    (poses / "poses.json").write_text(json.dumps({"source": "ordinary-video"}))
    ordinary = training.build_dataset(frames, poses, root / "ordinary")
    for path in poses.iterdir():
        if path.is_file():
            assert (ordinary / "sparse" / "0" / path.name).read_bytes() == path.read_bytes()
    for path in frames.iterdir():
        assert (ordinary / "images" / path.name).read_bytes() == path.read_bytes()


if __name__ == "__main__":
    run(Path(sys.argv[1]))
