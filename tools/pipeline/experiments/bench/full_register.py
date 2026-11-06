"""COLMAP 4.2 registration in the frozen model's frame, with all backbone poses fixed.

SIFT descriptors are re-extracted: old point2D indices cannot be reused against a new
feature database. Triangulate backbone matches at their fixed poses before registration.
"""

from __future__ import annotations

import shutil
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import sfm


def temporal_pairs(
    backbone: dict[str, float], native: dict[str, float], neighbours: int = 4
) -> list[tuple[str, str]]:
    if neighbours < 1:
        raise ValueError("neighbours must be positive")
    ordered = sorted(backbone, key=lambda n: (backbone[n], n))
    times = np.array([backbone[n] for n in ordered])
    pairs: set[tuple[str, str]] = set()
    for index, name in enumerate(ordered):
        for other in ordered[index + 1 : index + 1 + neighbours]:
            pairs.add((f"backbone/{name}", f"backbone/{other}"))
    for name, offset in native.items():
        index = int(np.searchsorted(times, offset))
        for other in ordered[max(0, index - neighbours) : index + neighbours]:
            pairs.add((f"backbone/{other}", f"native/{name}"))
    return sorted(pairs)


def options(pc: Any) -> Any:
    if pc.__version__ != "4.2.0":
        raise ValueError("full-data registration requires pycolmap==4.2.0")
    value = pc.IncrementalPipelineOptions()
    value.num_threads = 4
    value.ba_use_gpu = False
    value.fix_existing_frames = True
    value.multiple_models = False
    value.ba_refine_focal_length = False
    value.ba_refine_principal_point = False
    value.ba_refine_extra_params = False
    value.ba_refine_sensor_from_rig = False
    value.mapper.abs_pose_refine_focal_length = False
    value.mapper.abs_pose_refine_extra_params = False
    value.random_seed = 42
    return value


def restore_exact_backbone(poses: Path, original: sfm.Model) -> dict[str, float]:
    """Restore quaternion storage bits after verifying the fixed-pose invariant.

    Use the existing fixed-reference registration contract (test_pose_colmap4:
    test_new_frames_register_into_a_fixed_reference): 1e-9 scene units. Apply the
    same absolute bound to rotation coefficients (sub-nanoradian); do not fit it
    to this capture or machine. COLMAP BA repeatedly converts rigid transforms.
    Final storage is exactly the original, including quaternion sign and last bits.
    """
    before = {i.name: i for i in original.images}
    after = {i.name: i for i in sfm.read_model(poses).images}
    if before.keys() - after.keys():
        raise ValueError("registration lost backbone cameras")
    bound = 1e-9
    maximum = {
        "maxCentreDrift": 0.0,
        "maxRotationCoefficientDrift": 0.0,
        "maxTranslationDrift": 0.0,
    }
    for name, image in before.items():
        other = after[name]
        maximum["maxCentreDrift"] = max(
            maximum["maxCentreDrift"], float(np.linalg.norm(image.centre - other.centre))
        )
        maximum["maxRotationCoefficientDrift"] = max(
            maximum["maxRotationCoefficientDrift"],
            float(np.max(np.abs(image.rotation - other.rotation))),
        )
        maximum["maxTranslationDrift"] = max(
            maximum["maxTranslationDrift"], float(np.max(np.abs(np.array(image.tvec) - other.tvec)))
        )
        if max(maximum.values()) > bound:
            raise ValueError(f"fixed backbone camera moved: {name}")
    path = poses / "images.bin"
    with path.open("r+b") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            start = handle.tell()
            handle.read(64)
            name_bytes = bytearray()
            while (char := handle.read(1)) != b"\0":
                if not char:
                    raise ValueError("truncated image name")
                name_bytes.extend(char)
            name = name_bytes.decode()
            count2d = struct.unpack("<Q", handle.read(8))[0]
            end = handle.tell() + 24 * count2d
            if name in before:
                image = before[name]
                handle.seek(start + 4)
                handle.write(struct.pack("<7d", *image.qvec, *image.tvec))
            handle.seek(end)
    return maximum


def gauge_transform(
    before_centres: np.ndarray,
    before_rotation: np.ndarray,
    after_centres: np.ndarray,
    after_rotation: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Undo a similarity using one fixed orientation and a fixed camera baseline.

    COLMAP's centre-only Sim3 fit may be rank-deficient on a planar/linear path. An
    orientation supplies the missing axes. No camera or point is independently refit;
    the subsequent all-backbone check rejects any non-rigid movement.
    """
    lengths = np.linalg.norm(before_centres - before_centres[0], axis=1)
    far = int(np.argmax(lengths))
    other = float(np.linalg.norm(after_centres[far] - after_centres[0]))
    if lengths[far] <= 0 or other <= 0:
        raise ValueError("fixed camera baseline cannot define scale")
    scale = float(lengths[far] / other)
    rotation = before_rotation.T @ after_rotation
    translation = before_centres[0] - scale * rotation @ after_centres[0]
    return scale, rotation, translation


def restore_gauge(pc: Any, chosen: Any, original: sfm.Model) -> dict[str, Any]:
    after = {image.name.rsplit("/", 1)[-1]: image for image in chosen.images.values()}
    if {image.name for image in original.images} - after.keys():
        raise ValueError("registration lost backbone cameras")
    first = original.images[0]
    scale, rotation, translation = gauge_transform(
        np.stack([image.centre for image in original.images]),
        first.rotation,
        np.stack([after[image.name].projection_center() for image in original.images]),
        np.asarray(after[first.name].cam_from_world().rotation.matrix()),
    )
    chosen.transform(pc.Sim3d(scale, pc.Rotation3d(rotation), translation))
    return {
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "reason": "undo COLMAP internal normalization using fixed orientation/baseline",
    }


def register(
    base: Path, root: Path, work: Path, backbone: dict[str, float], native: dict[str, float]
) -> dict[str, Any]:
    import pycolmap as pc  # type: ignore[import-not-found]

    config = options(pc)
    started = time.monotonic()
    seconds: dict[str, float] = {}
    images = work / "images"
    for side, names in (("backbone", backbone), ("native", native)):
        (images / side).mkdir(parents=True)
        for name in names:
            shutil.copyfile(root / "frames" / name, images / side / name)
    reference = pc.Reconstruction(base / "poses")
    original = sfm.read_model(base / "poses")
    if len(original.cameras) != 1:
        raise ValueError("full-data currently requires one frozen phone camera")
    camera = original.cameras[0]
    reader = pc.ImageReaderOptions()
    reader.camera_model = camera.model
    extraction = pc.FeatureExtractionOptions()
    extraction.use_gpu = False
    extraction.num_threads = 4
    extraction.max_image_size = 1600
    extraction.sift.max_num_features = 4096
    extraction.sift.first_octave = 0
    database = work / "database.db"
    pc.extract_features(
        database,
        images,
        camera_mode=pc.CameraMode.PER_FOLDER,
        reader_options=reader,
        extraction_options=extraction,
        device=pc.Device.cpu,
    )
    seconds["extract"] = time.monotonic() - started
    db = pc.Database.open(database)
    seed = pc.Reconstruction()
    try:
        db.clear_pose_priors()
        by_id = {i.image_id: i for i in db.read_all_images()}
        cameras, rigs, frames = db.read_all_cameras(), db.read_all_rigs(), db.read_all_frames()
        for target in cameras:
            name = next(i.name for i in by_id.values() if i.camera_id == target.camera_id)
            with Image.open(images / name) as photo:
                width, height = photo.size
            sx, sy = width / camera.width, height / camera.height
            if abs(sx - sy) > 1e-12:
                raise ValueError("native source aspect differs from frozen camera")
            target.model = camera.model
            target.width, target.height = width, height
            target.params = np.array(
                [
                    v * sx if n in {"f", "fx", "fy", "cx", "cy"} else v
                    for n, v in zip(camera.param_names, camera.params, strict=True)
                ]
            )
            db.update_camera(target)
            seed.add_camera(target)
        config.constant_cameras = set(seed.cameras)
    finally:
        db.close()
    began = time.monotonic()
    pairs = temporal_pairs(backbone, native)
    pair_file = work / "pairs.txt"
    pair_file.write_text("".join(f"{a} {b}\n" for a, b in pairs))
    matching = pc.FeatureMatchingOptions()
    matching.num_threads = 4
    matching.use_gpu = False
    pairing = pc.ImportedPairingOptions()
    pairing.match_list_path = str(pair_file)
    pc.match_image_pairs(
        database, matching_options=matching, pairing_options=pairing, device=pc.Device.cpu
    )
    seconds["match"] = time.monotonic() - began
    db = pc.Database.open(database)
    try:
        pair_ids, geometries = db.read_two_view_geometries()
    finally:
        db.close()
    connected = set()
    for pair_id, geometry in zip(pair_ids, geometries, strict=True):
        if len(geometry.inlier_matches):
            connected.update(pc.pair_id_to_image_pair(pair_id))
    before = {i.name: i for i in reference.images.values()}
    for rig in rigs:
        seed.add_rig(rig)
    for frame in frames:
        image = by_id[min(d.id for d in frame.data_ids)]
        if not image.name.startswith("backbone/"):
            continue
        if image.image_id not in connected:
            raise ValueError(f"backbone image has no verified temporal matches: {image.name}")
        frame.rig_from_world = before[image.name.split("/", 1)[1]].cam_from_world()
        seed.add_frame(frame)
        seed.add_image(image)
        seed.register_frame(frame.frame_id)
    began = time.monotonic()
    (work / "tri").mkdir()
    pc.triangulate_points(seed, database, images, work / "tri", clear_points=True, options=config)
    seconds["triangulate"] = time.monotonic() - began
    began = time.monotonic()
    (work / "reg").mkdir()
    models = pc.incremental_mapping(database, images, work / "reg", config, input_path=work / "tri")
    if len(models) != 1:
        raise ValueError(f"expected one anchored reconstruction; got {len(models)}")
    chosen = next(iter(models.values()))
    seconds["registerAndBundleAdjust"] = time.monotonic() - began
    restored_gauge = restore_gauge(pc, chosen, original)
    for image in chosen.images.values():
        image.name = image.name.split("/", 1)[1]
    poses = root / "poses"
    poses.mkdir()
    chosen.write(poses)
    # Preserve original quaternion encoding exactly, and fail if BA moved any backbone.
    drift = restore_exact_backbone(poses, original)
    for path in poses.iterdir():
        if path.name not in {"cameras.bin", "images.bin", "points3D.bin"}:
            path.unlink()
    final = sfm.read_model(poses)
    registered_names = {i.name for i in final.images}
    failed = sorted(native.keys() - registered_names)
    return {
        "normalizationRestored": restored_gauge,
        "backboneDriftBeforeExactRestore": drift,
        "pycolmap": "4.2.0",
        "attempted": len(native),
        "registered": len(native) - len(failed),
        "failures": failed,
        "fixedBackbone": len(backbone),
        "pairs": len(pairs),
        "meanReprojectionErrorPx": final.mean_reprojection_error,
        "points": final.points3d,
        "wallSeconds": time.monotonic() - started,
        "phasesSeconds": seconds,
        "limits": {
            "featureMaxSide": 1600,
            "featuresPerImage": 4096,
            "threads": 4,
            "reason": "same SIFT settings as original pose stage; CPU budget",
        },
    }
