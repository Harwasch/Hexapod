"""Short-baseline gauge regression using the installed CPU pycolmap 4.2 wheel."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from arkit_refine_scene import scene

import arkit_refine
import sfm


def legacy_bundle_adjust(pycolmap: Any, model: Any, *, lens: bool, **_: Any) -> None:
    """PR #147's original solve, retained only as the regression's negative control."""
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
    for image_id in model.reg_image_ids():
        config.add_image(image_id)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    summary = pycolmap.create_default_bundle_adjuster(options, config, model).solve()
    assert summary.is_solution_usable()


def run(root: Path, *, lens: bool, seed: int) -> None:
    import pycolmap  # type: ignore[import-not-found,unused-ignore]

    assert pycolmap.__version__ == "4.2.0"
    source, database, frames, records, truth = scene(root, pycolmap, short_baseline=True, seed=seed)
    initial = sfm.read_model(source)
    original_bytes = {p.name: p.read_bytes() for p in source.iterdir()}
    before = np.array([i.centre for i in initial.images])
    assert np.linalg.norm(before[1] - before[0]) < 0.02
    solve = arkit_refine.bundle_adjust
    factory = pycolmap.create_pose_prior_bundle_adjuster
    prior_passes = []

    def checked_factory(
        options: Any, prior_options: Any, config: Any, priors: Any, reconstruction: Any
    ) -> Any:
        # Exercise the real 4.2 API, including image/sensor IDs after the shared
        # camera rebuild. The second pass must still target the immutable input.
        assert config.fixed_gauge == pycolmap.BundleAdjustmentGauge.UNSPECIFIED
        assert len(priors) == len(initial.images)
        for prior in priors:
            image = reconstruction.images[prior.corr_data_id.id]
            assert prior.corr_data_id == image.data_id
            np.testing.assert_allclose(prior.position, initial.by_name()[image.name].centre)
            np.testing.assert_allclose(prior.position_covariance, np.eye(3) * 0.03**2)
        prior_passes.append(len(priors))
        return factory(options, prior_options, config, priors, reconstruction)

    reports = {}
    pycolmap.create_pose_prior_bundle_adjuster = checked_factory
    for name, adjust in (("old", legacy_bundle_adjust), ("new", solve)):
        work = root / name
        work.mkdir()
        arkit_refine.bundle_adjust = adjust
        try:
            report = arkit_refine.refine(
                pycolmap, source, database, frames, work / "model", lens=lens
            )
        finally:
            arkit_refine.bundle_adjust = solve
        reports[name] = report
        (root / f"{name}.json").write_text(json.dumps(report))
    old, new = reports["old"], reports["new"]
    pycolmap.create_pose_prior_bundle_adjuster = factory
    assert prior_passes == [len(initial.images)] * 2
    assert new["gauge"]["type"] == "arkit-position-priors"
    assert new["gauge"]["numPositions"] == len(initial.images)
    assert [item["stage"] for item in new["gauge"]["scaleChecks"]] == [
        "initial.beforeAlignment",
        "initial.afterSolve",
        "final.beforeAlignment",
        "final.afterSolve",
    ]
    assert abs(old["sim3"]["scale"] - 1) > 0.03, old["sim3"]
    assert old["fallbackReason"] == "Sim3 scale differs from ARKit by >=0.5%"
    assert old["reprojectionAfterPx"] < old["reprojectionBeforePx"] * 0.5
    assert not (root / "old" / "model").exists()
    assert abs(new["sim3"]["scale"] - 1) < 0.005, new["sim3"]
    assert new["status"] == "accepted", new["fallbackReason"]
    assert new["reprojectionAfterPx"] < new["reprojectionBeforePx"] * 0.5
    model = sfm.read_model(root / "new" / "model")
    arkit_refine.verify(model, records, new)
    after = np.array([i.centre for i in model.images])
    assert (
        np.mean(np.linalg.norm(after - truth, axis=1))
        < np.mean(np.linalg.norm(before - truth, axis=1)) * 0.7
    )
    assert (
        np.mean([arkit_refine.angle(i.rotation) for i in model.images])
        < np.mean([arkit_refine.angle(i.rotation) for i in initial.images]) * 0.7
    )
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original_bytes
    if not lens:
        rejected_cases(root, pycolmap, source, database, frames)


def rejected_cases(root: Path, pycolmap: Any, source: Path, database: Path, frames: Path) -> None:
    """Prior-factory alignment must never erase drift or silently restore TWO_CAMS."""
    factory = pycolmap.create_pose_prior_bundle_adjuster
    triangulate = pycolmap.triangulate_points
    drift = pycolmap.Sim3d(1.02, pycolmap.Rotation3d(), np.zeros(3))
    for failure in ("initial.afterSolve", "final.beforeAlignment", "priorFallback"):
        work = root / failure
        work.mkdir()
        report = arkit_refine.empty_report()

        def adjusted_factory(
            options: Any,
            prior_options: Any,
            config: Any,
            priors: Any,
            model: Any,
            failure: str = failure,
        ) -> Any:
            # Real 4.2 behavior: fewer than three priors logs an alignment failure
            # and silently selects TWO_CAMS_FROM_WORLD instead of throwing.
            if failure == "priorFallback":
                return factory(options, prior_options, config, priors[:2], model)
            adjuster = factory(options, prior_options, config, priors, model)

            def solve_with_drift() -> Any:
                summary = adjuster.solve()
                model.transform(drift)
                return summary

            if failure == "initial.afterSolve":
                return SimpleNamespace(config=adjuster.config, solve=solve_with_drift)
            return adjuster

        def triangulate_with_drift(*args: Any, **kwargs: Any) -> Any:
            model = triangulate(*args, **kwargs)
            model.transform(drift)
            return model

        pycolmap.create_pose_prior_bundle_adjuster = adjusted_factory
        if failure == "final.beforeAlignment":
            pycolmap.triangulate_points = triangulate_with_drift
        try:
            arkit_refine.refine(
                pycolmap, source, database, frames, work / "model", lens=False, report=report
            )
        except ValueError as error:
            if failure == "priorFallback":
                assert "position-prior alignment failed" in str(error), str(error)
            else:
                assert "Sim3 scale differs" in str(error), str(error)
                last = report["gauge"]["scaleChecks"][-1]
                assert last["stage"] == failure
                assert abs(last["scale"] - 1) > 0.005
        else:
            raise AssertionError(f"accepted {failure}")
        finally:
            pycolmap.create_pose_prior_bundle_adjuster = factory
            pycolmap.triangulate_points = triangulate
        assert not (work / "model").exists()


if __name__ == "__main__":
    run(Path(sys.argv[1]), lens=sys.argv[2] == "lens", seed=int(sys.argv[3]))
