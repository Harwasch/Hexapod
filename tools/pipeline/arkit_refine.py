"""Guarded bundle-only BA, isolated in the production pycolmap 4.2 interpreter.

The fixed model is never edited. A candidate is published only after alignment and
validation. Bounds are policy, not accuracy claims: 15 cm, 5 degrees, 0.5% scale.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import capture_bundle as bundle
import sfm

F64 = npt.NDArray[np.float64]
MAX_CM = 15.0
MAX_DEG = 5.0
MAX_SCALE_DELTA = 0.005
POSITION_PRIOR_SIGMA_M = 0.03
# pycolmap 4.2's 95% chi-square threshold for a 3D whitened position residual.
POSITION_PRIOR_LOSS_SCALE = float(np.sqrt(7.815))


def empty_report(status: str = "fallback") -> dict[str, Any]:
    return {
        "status": status,
        "fallbackReason": None,
        "reprojectionBeforePx": None,
        "reprojectionAfterPx": None,
        "translationCm": None,
        "rotationDeg": None,
        "sim3": None,
        "lensParams": None,
        "gauge": None,
        "bounds": {
            "translationCm": MAX_CM,
            "rotationDeg": MAX_DEG,
            "sim3ScaleDelta": MAX_SCALE_DELTA,
        },
    }


def align(source: F64, target: F64) -> tuple[float, F64, F64]:
    """All-centre Umeyama fit into ARKit's metric/gravity frame; reject line gauges."""
    a, b = source - source.mean(0), target - target.mean(0)
    if len(a) < 3 or np.linalg.matrix_rank(a, tol=1e-8) < 2:
        raise ValueError("degenerate ARKit trajectory for metric/gravity alignment")
    u, values, vt = np.linalg.svd(b.T @ a)
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = (u * signs) @ vt
    scale = float(np.sum(values * signs) / np.sum(a * a))
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    return scale, rotation, translation


def stats(values: F64) -> dict[str, float]:
    return dict(
        zip(("median", "p95", "max"), map(float, np.percentile(values, [50, 95, 100])), strict=True)
    )


def angle(rotation: F64) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def guard(report: dict[str, Any]) -> str | None:
    """Accept only finite, improving, bounded corrections with a near-unit raw gauge."""
    try:
        before, after = report["reprojectionBeforePx"], report["reprojectionAfterPx"]
        scales = [report["sim3"]["scale"]] + [
            item["scale"] for item in (report.get("gauge") or {}).get("scaleChecks", [])
        ]
        cm, deg = report["translationCm"]["max"], report["rotationDeg"]["max"]
        if not np.isfinite([before, after, *scales, cm, deg]).all():
            return "non-finite refinement"
        if any(abs(scale - 1) >= MAX_SCALE_DELTA for scale in scales):
            return "Sim3 scale differs from ARKit by >=0.5%"
        if after >= before:
            return "mean reprojection error did not improve"
        if cm > MAX_CM or deg > MAX_DEG:
            return "pose correction exceeds 15 cm or 5 degrees"
    except (KeyError, TypeError, ValueError):
        return "incomplete refinement evidence"
    return None


def verify(
    model: sfm.Model,
    records: list[dict[str, Any]],
    report: dict[str, Any] | None,
    *,
    suffix: str = ".png",
) -> None:
    """Strict fixed poses unless an accepted correction accounts for each changed pose."""
    accepted = report is not None and report.get("status") == "accepted"
    if accepted and report is not None and guard(report):
        raise ValueError("invalid ARKit refinement evidence")
    by_name = model.by_name()
    for record in records:
        name = bundle.frame_name(record["index"], suffix)
        r, t = bundle.world_to_camera(bundle.matrix(record["transform"], 4))
        image = by_name[name]
        if accepted:
            assert report is not None
            correction = report["corrections"][name]
            # Record the raw BA pose and the applied global similarity, so verifying
            # is independent of simply echoing the final pose in poses.json.
            similarity = report["sim3"]
            rotation = np.asarray(similarity["rotation"])
            centre = similarity["scale"] * rotation @ np.asarray(
                correction["rawCentre"]
            ) + np.asarray(similarity["translation"])
            expected_r = np.asarray(correction["rawRotation"]) @ rotation.T
            if not np.allclose(image.rotation, expected_r, atol=1e-5, rtol=0) or not np.allclose(
                image.centre, centre, atol=1e-5, rtol=0
            ):
                raise ValueError("refined pose disagrees with recorded Sim3")
            if (
                np.linalg.norm(image.centre + r.T @ t) * 100 > MAX_CM
                or angle(image.rotation @ r.T) > MAX_DEG
            ):
                raise ValueError("refined pose exceeds ARKit correction bounds")
        elif not np.allclose(image.rotation, r, atol=1e-5, rtol=0) or not np.allclose(
            image.tvec, t, atol=1e-5, rtol=0
        ):
            raise ValueError("COLMAP changed a fixed ARKit pose")


def verified_summary(poses: dict[str, Any]) -> bool:
    """Downstream cannot trust a source tag after a free-pose solve."""
    report = poses.get("refinement")
    if report is not None and not isinstance(report, dict):
        return False
    if report is None or report.get("status") in {"disabled", "fallback"}:
        return True  # legacy/disabled/fallback producer performed the strict check
    if report.get("status") != "accepted" or guard(report):
        return False
    try:
        images = poses["images"]
        corrections = report["corrections"]
        similarity = report["sim3"]
        rotation = np.asarray(similarity["rotation"], dtype=float)
        translation = np.asarray(similarity["translation"], dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            return False
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            or np.linalg.det(rotation) < 0
        ):
            return False
        if not images or {image["name"] for image in images} != set(corrections):
            return False
        for image in images:
            item = corrections[image["name"]]
            r = sfm.quat_to_matrix(image["qvec"])
            centre = -r.T @ np.asarray(image["tvec"])
            expected = similarity["scale"] * rotation @ np.asarray(item["rawCentre"]) + translation
            if not np.allclose(centre, expected, atol=1e-5, rtol=0) or not np.allclose(
                r, np.asarray(item["rawRotation"]) @ rotation.T, atol=1e-5, rtol=0
            ):
                return False
            if (
                np.linalg.norm(centre - item["arkitCentre"]) * 100 > MAX_CM
                or angle(r @ np.asarray(item["arkitRotation"]).T) > MAX_DEG
            ):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def bundle_adjust(
    pycolmap: Any,
    model: Any,
    *,
    lens: bool,
    original: Any,
    scale_checks: list[dict[str, Any]],
    phase: str,
) -> None:
    """Anchor the metric gauge to all original ARKit centres, in both BA passes.

    TWO_CAMS_FROM_WORLD fixes scale using one translation component of a second
    camera. Millimetre noise on a centimetre video baseline can therefore give a
    good reconstruction an arbitrary, multi-percent raw scale error.
    """
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = lens
    options.refine_principal_point = lens
    options.refine_extra_params = lens
    options.refine_sensor_from_rig = False
    options.ceres.use_gpu = False
    options.ceres.loss_function_type = pycolmap.LossFunctionType.SOFT_L1
    options.ceres.loss_function_scale = 1.0
    options.ceres.solver_options.num_threads = 4
    options.ceres.solver_options.max_num_iterations = 150
    options.ceres.solver_options.function_tolerance = 1e-7
    config = pycolmap.BundleAdjustmentConfig()
    priors = []
    for image_id in sorted(model.reg_image_ids()):
        config.add_image(image_id)
        # The shared-lens model rebuilds sensor IDs. Use its data ID but always
        # the immutable input's centre, never the first BA pass's fitted position.
        priors.append(
            pycolmap.PosePrior(
                corr_data_id=model.images[image_id].data_id,
                position=original.images[image_id].projection_center(),
                position_covariance=np.eye(3) * POSITION_PRIOR_SIGMA_M**2,
                coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
            )
        )
    centres = np.array([p.position for p in priors])
    align(centres, centres)  # Reject an unobservable gauge before entering Ceres.

    def check_scale(stage: str) -> None:
        raw = np.array([model.images[i].projection_center() for i in sorted(model.reg_image_ids())])
        scale, _, _ = align(raw, centres)
        scale_checks.append({"stage": f"{phase}.{stage}", "scale": scale})
        if not np.isfinite(scale):
            raise ValueError("non-finite refinement")
        if abs(scale - 1) >= MAX_SCALE_DELTA:
            raise ValueError("Sim3 scale differs from ARKit by >=0.5%")

    # The factory aligns to priors before solving. Check its input too, so neither
    # that alignment nor retriangulation can conceal an out-of-bounds raw scale.
    check_scale("beforeAlignment")
    prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
    prior_options.ceres.prior_position_loss_function_type = pycolmap.LossFunctionType.SOFT_L1
    prior_options.ceres.prior_position_loss_scale = POSITION_PRIOR_LOSS_SCALE
    prior_options.alignment_ransac.max_error = MAX_CM / 100
    prior_options.alignment_ransac.random_seed = 0
    prior_options.alignment_ransac.num_threads = 1
    # Leave config's gauge UNSPECIFIED: the position priors constrain the Sim3.
    adjuster = pycolmap.create_pose_prior_bundle_adjuster(
        options, prior_options, config, priors, model
    )
    # In 4.2 failed prior alignment logs a warning and switches to TWO_CAMS.
    # A usable solve would not establish that the trajectory priors were used.
    if adjuster.config.fixed_gauge != pycolmap.BundleAdjustmentGauge.UNSPECIFIED:
        raise ValueError("position-prior alignment failed; refusing a fallback camera gauge")
    summary = adjuster.solve()
    if not summary.is_solution_usable():
        raise ValueError("bundle adjustment has no usable solution")
    check_scale("afterSolve")


def shared_camera(pycolmap: Any, original: Any, work: Path) -> Any:
    """Rebuild trivial rigs as well as image camera IDs; one lens, median K seed."""
    sizes = {(c.width, c.height) for c in original.cameras.values()}
    if len(sizes) != 1:
        raise ValueError("capture stream has varying image dimensions")
    params = np.median(
        [original.cameras[i.camera_id].params[:4] for i in original.images.values()], axis=0
    )
    width, height = next(iter(sizes))
    text = work / "shared"
    text.mkdir()
    original.write_text(text)
    (text / "cameras.txt").write_text(
        f"1 OPENCV {width} {height} " + " ".join(map(str, [*params, 0, 0, 0, 0])) + "\n"
    )
    lines = (text / "images.txt").read_text().splitlines()
    result = []
    pose_line = True
    for line in lines:
        if line.startswith("#"):
            continue
        if pose_line:
            fields = line.split()
            fields[8] = "1"
            line = " ".join(fields)
        result.append(line)
        pose_line = not pose_line
    (text / "images.txt").write_text("\n".join(result) + "\n")
    for filename in ("frames.txt", "rigs.txt"):
        (text / filename).unlink(missing_ok=True)
    return pycolmap.Reconstruction(text)


def refine(
    pycolmap: Any,
    source: Path,
    database: Path,
    frames: Path,
    output: Path,
    *,
    lens: bool,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original = pycolmap.Reconstruction(source)
    original.update_point_3d_errors()
    work = output.parent
    if report is None:
        report = empty_report()
    report.update(
        lensEnabled=lens,
        reprojectionBeforePx=original.compute_mean_reprojection_error(),
        loss="SOFT_L1",
        lossScalePx=1.0,
        gauge={
            "type": "arkit-position-priors",
            "positionSigmaM": POSITION_PRIOR_SIGMA_M,
            "loss": "SOFT_L1",
            "lossScaleSigma": POSITION_PRIOR_LOSS_SCALE,
            "alignmentMaxErrorM": MAX_CM / 100,
            "numPositions": original.num_reg_images(),
            "scaleChecks": [],
        },
    )
    model = shared_camera(pycolmap, original, work) if lens else pycolmap.Reconstruction(source)
    bundle_adjust(
        pycolmap,
        model,
        lens=lens,
        original=original,
        scale_checks=report["gauge"]["scaleChecks"],
        phase="initial",
    )
    # Upgrade only a copy of the CLI database; the fixed fallback remains reusable.
    db_path = work / "refine.db"
    shutil.copy2(database, db_path)
    if lens:
        # Rebuild the database's trivial rigs/frames too. Merely changing image
        # camera_id leaves COLMAP 4's sensor data IDs referring to old cameras.
        old_db = pycolmap.Database.open(db_path)
        shared_db = work / "shared.db"
        new_db = pycolmap.Database.open(shared_db)
        try:
            new_db.write_camera(model.cameras[1], use_camera_id=True)
            for image in old_db.read_all_images():
                new_db.write_image(
                    pycolmap.Image(image_id=image.image_id, name=image.name, camera_id=1),
                    use_image_id=True,
                )
                new_db.write_keypoints(image.image_id, old_db.read_keypoints(image.image_id))
            pair_ids, geometries = old_db.read_two_view_geometries()
            for pair_id, geometry in zip(pair_ids, geometries, strict=True):
                a, b = pycolmap.pair_id_to_image_pair(pair_id)
                new_db.write_two_view_geometry(a, b, geometry)
        finally:
            new_db.close()
            old_db.close()
        db_path = shared_db
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 4
    options.fix_existing_frames = True
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.ba_use_gpu = False
    model = pycolmap.triangulate_points(
        model,
        db_path,
        frames,
        work / "retriangulated",
        clear_points=False,
        options=options,
        refine_intrinsics=False,
    )
    bundle_adjust(
        pycolmap,
        model,
        lens=lens,
        original=original,
        scale_checks=report["gauge"]["scaleChecks"],
        phase="final",
    )
    ids = sorted(original.reg_image_ids())
    if model.num_points3D() == 0:
        raise ValueError("refinement lost every triangulated point")
    if sorted(model.reg_image_ids()) != ids:
        raise ValueError("refinement lost ARKit frames")
    source_centres = np.array([model.images[i].projection_center() for i in ids])
    target_centres = np.array([original.images[i].projection_center() for i in ids])
    scale, rotation, translation = align(source_centres, target_centres)
    report["sim3"] = {
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
    }
    corrections = {}
    for i in ids:
        raw, arkit = model.images[i], original.images[i]
        corrections[raw.name] = {
            "rawCentre": raw.projection_center().tolist(),
            "rawRotation": raw.cam_from_world().rotation.matrix().tolist(),
            "arkitCentre": arkit.projection_center().tolist(),
            "arkitRotation": arkit.cam_from_world().rotation.matrix().tolist(),
        }
    report["corrections"] = corrections
    model.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(rotation), translation))
    report["translationCm"] = stats(
        np.array(
            [
                np.linalg.norm(
                    model.images[i].projection_center() - original.images[i].projection_center()
                )
                * 100
                for i in ids
            ]
        )
    )
    report["rotationDeg"] = stats(
        np.array(
            [
                angle(
                    model.images[i].cam_from_world().rotation.matrix()
                    @ original.images[i].cam_from_world().rotation.matrix().T
                )
                for i in ids
            ]
        )
    )
    model.update_point_3d_errors()
    report["reprojectionAfterPx"] = model.compute_mean_reprojection_error()
    report["lensParams"] = {
        str(c.camera_id): {"model": c.model.name, "params": c.params.tolist()}
        for c in model.cameras.values()
    }
    for camera in model.cameras.values():
        if not np.isfinite(camera.params).all() or np.any(camera.params[:2] <= 0):
            raise ValueError("refinement produced invalid camera parameters")
        if lens:
            import distortion

            # Reject an unusable lens on CPU, before purchasing a training GPU.
            distortion.geometry(
                sfm.Camera(
                    camera.camera_id,
                    "OPENCV",
                    camera.width,
                    camera.height,
                    tuple(camera.params),
                    sfm.CAMERA_MODELS[4][1],
                )
            )
    report["observationsBefore"] = original.compute_num_observations()
    report["observationsAfter"] = model.compute_num_observations()
    report["fallbackReason"] = guard(report)
    if report["fallbackReason"] is None:
        report["status"] = "accepted"
        output.mkdir()
        model.write_binary(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("source", "database", "frames", "output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--no-lens", action="store_true")
    args = parser.parse_args()
    import pycolmap  # type: ignore[import-not-found,unused-ignore]

    if pycolmap.__version__ != "4.2.0":
        raise ValueError("ARKit refinement requires pycolmap 4.2.0")
    report = empty_report()
    try:
        report = refine(
            pycolmap,
            args.source,
            args.database,
            args.frames,
            args.output,
            lens=not args.no_lens,
            report=report,
        )
    except (ValueError, RuntimeError) as error:
        report.update(status="fallback", fallbackReason=str(error))
    # A rejected numerical failure still needs a portable JSON report.
    report = json.loads(json.dumps(report), parse_constant=lambda _: None)
    args.report.write_text(json.dumps(report, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
