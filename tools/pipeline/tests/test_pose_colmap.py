"""`pose` / `colmap`, scored against known poses rather than against "a file appeared".

The fixture is 40 frames of the committed synthetic tree, rendered here at test time from
a closed orbit whose poses are known exactly (`tree_frames.py`). COLMAP then runs for
real -- feature extraction, exhaustive matching, incremental mapping, CPU only -- and what
is asserted is how many frames registered and how far the recovered poses are from the
ones the frames were rendered from, after the similarity transform that any image-only
reconstruction is defined up to.

Measured on 4 cores while this was written, against A0 #7's numbers on its own fixture:

                          this fixture      A0 #7
    registered            40 / 40           40 / 40
    median rotation       0.059 deg         0.422 deg
    median translation    0.092% of extent  0.636% of extent
    focal error           +0.17%            -3.1%
    wall clock            ~55 s             48.3 s

The first three are *better* than A0's and the fourth has the opposite sign, and that is a
fact about the fixtures, not about COLMAP: this one is a noise-free pinhole render with a
perfectly planar, non-repeating textured ground and no rolling shutter, so it is an easier
scene than whatever A0 measured. The thresholds below are therefore set where a real
regression would trip them, not at the measured values -- a fixture this clean drifting to
A0's numbers would already be a problem worth seeing.

On CI: `colmap` is installed by the workflow (`apt-get install -y colmap`, measured at a
176-package, 184 MB closure that downloads in ~10 s and unpacks in ~3 s), so this runs
there. `test_colmap_is_installed_in_ci` is the one test here that never skips: without it
a workflow that dropped the install would leave every test in this file skipping and the
job green.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import sfm
import stages
import tree_frames
from conftest import FIXTURE_PLY, make_recipe
from executor import execute
from runners import LocalRunner, RunnerSet
from workdir import Workdir

#: A0 #7 ran 40 on a closed orbit; the same count, so the numbers are comparable.
FRAMES = 40

requires_colmap = pytest.mark.skipif(
    not sfm.colmap_available(),
    reason="colmap is not on this machine (CI installs it; see .github/workflows/ci.yml)",
)


def test_colmap_is_installed_in_ci() -> None:
    """The one test in this file that never skips.

    A skipped suite is indistinguishable from a passing one in a CI summary. The same
    arrangement `apps/api/tests/test_storage_minio.py` uses for MinIO, for the same
    reason: if the `apt-get install -y colmap` step disappears from the workflow, this
    fails instead of the pose stage quietly becoming untested everywhere.
    """
    if os.environ.get("CI", "").lower() == "true":
        assert sfm.colmap_available(), (
            "colmap is missing under CI: the install step is gone from "
            ".github/workflows/ci.yml, so every COLMAP test here would skip and `pose` "
            "would be unverified"
        )


# --- the fixture itself --------------------------------------------------------------


def test_an_improper_rotation_cannot_become_a_pose() -> None:
    """A0 #7's third finding, baked into the fixture builder so it cannot be re-entered.

    A0's own evaluator built a reflection and measured a constant 90.000 degree error
    that read exactly like broken SfM. `Pose` refuses to exist rather than letting a
    caller remember to check.
    """
    reflection = np.diag([1.0, 1.0, -1.0])
    assert np.linalg.det(reflection) == pytest.approx(-1.0)

    with pytest.raises(ValueError, match="det"):
        tree_frames.Pose(name="bad", rotation=reflection, translation=np.zeros(3))


#: How far a pose may move when it round-trips through the quaternion COLMAP stores.
#:
#: Not a precision claim -- a guard against A0 #7's trap, where an improper rotation in
#: the evaluator produced a constant 90.000 deg error that read exactly like broken SfM.
#: A flip lands at 90 or 180 degrees, so anything far below that catches it.
#:
#: It is 1e-4 rather than the 1e-9 this test shipped with because 1e-9 was calibrated on
#: one machine's luck. Measured: this development VM round-trips all 40 poses at *exactly*
#: 0.0 deg, while `ubuntu-latest` reached 1.2e-6 deg on frame_0006 -- the same matrix
#: element coming out as -5.55e-17 there and 0.0 here. Different CPU, different BLAS,
#: different rounding in the quaternion extraction's square root. A double-precision
#: round trip is good to about 1e-6 deg and no better, so 1e-9 was asserting that the
#: arithmetic is exact, which is not a property any machine owes us. This leaves two
#: orders of magnitude of headroom over the worst observed value and still sits six
#: orders below the flip it exists to catch.
QUATERNION_ROUND_TRIP_DEG = 1e-4


def test_every_pose_the_fixture_builds_is_a_proper_rotation() -> None:
    for pose in tree_frames.orbit(FRAMES):
        assert float(np.linalg.det(pose.rotation)) == pytest.approx(1.0, abs=1e-12)
        # Round-tripping through the quaternion COLMAP compares against must not flip it.
        error = sfm.rotation_angle_deg(sfm.quat_to_matrix(pose.qvec), pose.rotation)
        assert error < QUATERNION_ROUND_TRIP_DEG, f"{pose.name} moved {error:.3e} deg"


def test_the_rendered_orbit_is_byte_identical_between_runs(tmp_path: Path) -> None:
    """It has to be: a fixture regenerated per run that is not reproducible is a test
    whose failures cannot be reproduced either."""
    first = tmp_path / "a" / "frames"
    second = tmp_path / "b" / "frames"
    tree_frames.render_orbit(FIXTURE_PLY, first, count=3)
    tree_frames.render_orbit(FIXTURE_PLY, second, count=3)

    a = {p.name: p.read_bytes() for p in sorted(first.iterdir())}
    b = {p.name: p.read_bytes() for p in sorted(second.iterdir())}
    assert a == b and len(a) == 3


# --- argv, which needs no COLMAP -----------------------------------------------------


def test_the_default_matcher_is_the_one_that_closes_an_orbit() -> None:
    """A0 #7: `sequential_matcher` without loop detection registered 2 of 40."""
    assert "exhaustive" in sfm.MATCHERS
    argv = sfm.matcher_argv(Path("db.db")) if sfm.colmap_available() else None
    if argv is not None:
        assert argv[1] == "exhaustive_matcher"
        assert (
            "--SiftMatching.use_gpu" in argv
            and argv[argv.index("--SiftMatching.use_gpu") + 1] == "0"
        )


def test_an_unknown_matcher_is_refused_rather_than_passed_through() -> None:
    with pytest.raises(ValueError, match="unknown matcher"):
        sfm.matcher_argv(Path("db.db"), "nearest_friend")


@requires_colmap
def test_a_focal_prior_is_held_rather_than_refined() -> None:
    """The difference between recording a scale bias and having one."""
    free = sfm.mapper_argv(Path("d"), Path("i"), Path("o"), refine_focal_length=True)
    held = sfm.mapper_argv(Path("d"), Path("i"), Path("o"), refine_focal_length=False)

    assert free[free.index("--Mapper.ba_refine_focal_length") + 1] == "1"
    assert held[held.index("--Mapper.ba_refine_focal_length") + 1] == "0"
    with_prior = sfm.feature_extractor_argv(
        Path("d"), Path("i"), camera_model="SIMPLE_RADIAL", camera_params=(520.0, 320.0, 240.0, 0.0)
    )
    assert with_prior[with_prior.index("--ImageReader.camera_params") + 1] == "520,320,240,0"


def test_umeyama_never_returns_a_reflection() -> None:
    """The same trap from the other side: a reflection fits mirrored points better and
    is not a rotation, and the symptom downstream is a plausible constant error."""
    rng = np.random.default_rng(7)
    source = rng.normal(size=(12, 3))
    mirrored = source * np.array([1.0, 1.0, -1.0])

    fit = sfm.umeyama(source, mirrored)

    assert float(np.linalg.det(fit.rotation)) == pytest.approx(1.0)


def test_umeyama_recovers_a_similarity_it_was_given() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(20, 3))
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1.0]]
    )
    target = (2.5 * (rotation @ source.T)).T + np.array([1.0, -2.0, 3.0])

    fit = sfm.umeyama(source, target)

    assert fit.scale == pytest.approx(2.5)
    assert np.allclose(fit.apply(source), target)


# --- the measured integration test ---------------------------------------------------


@pytest.fixture(scope="module")
def reconstruction(tmp_path_factory: pytest.TempPathFactory) -> tuple[Workdir, tree_frames.Truth]:
    """Render the orbit, run the real `pose` stage over it, once for this module."""
    root = tmp_path_factory.mktemp("orbit")
    workdir = Workdir.create(root / "run")
    frames = workdir.input_path("frames")
    truth = tree_frames.render_orbit(FIXTURE_PLY, frames, count=FRAMES)
    recipe = make_recipe(
        [{"id": "pose", "impl": "colmap", "params": {"matcher": "exhaustive"}}], inputs=["frames"]
    )
    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    return workdir, truth


@requires_colmap
def test_colmap_registers_every_frame_of_a_closed_orbit(
    reconstruction: tuple[Workdir, tree_frames.Truth],
) -> None:
    workdir, truth = reconstruction

    poses = json.loads((workdir.artifact_path("pose", "poses") / "poses.json").read_text())

    assert poses["frames"] == FRAMES
    assert poses["registered"] == FRAMES, (
        f"{poses['registered']} of {FRAMES} registered. On a closed orbit that is the "
        f"failure A0 #7 saw from sequential_matcher; this ran {poses['matcher']}"
    )
    assert poses["points3D"] > 1000
    assert poses["meanTrackLength"] > 3.0
    assert len(truth.poses) == FRAMES


@requires_colmap
def test_the_recovered_poses_are_within_a_degree_and_one_percent_of_the_truth(
    reconstruction: tuple[Workdir, tree_frames.Truth],
) -> None:
    """The centrepiece. Measured 0.059 deg and 0.092% of extent; the bounds are where a
    real regression would show, not at the measured values."""
    workdir, truth = reconstruction
    model = sfm.read_model(workdir.artifact_path("pose", "poses"))
    known = truth.by_name()
    assert set(image.name for image in model.images) == set(known)

    # A reconstruction from images alone is defined up to a similarity, so solve for it
    # from the camera centres before comparing anything.
    estimated_centres = np.stack([image.centre for image in model.images])
    true_centres = np.stack([known[image.name].centre for image in model.images])
    fit = sfm.umeyama(estimated_centres, true_centres)
    translation_error = np.linalg.norm(fit.apply(estimated_centres) - true_centres, axis=1)
    rotation_error = np.array(
        [
            sfm.rotation_angle_deg(fit.rotation @ image.rotation.T, known[image.name].rotation.T)
            for image in model.images
        ]
    )

    median_rotation = float(np.median(rotation_error))
    median_translation_pct = 100.0 * float(np.median(translation_error)) / truth.extent_m
    worst_translation_pct = 100.0 * float(translation_error.max()) / truth.extent_m
    assert median_rotation < 1.0, f"median rotation error {median_rotation:.3f} deg"
    assert float(rotation_error.max()) < 3.0
    assert median_translation_pct < 1.0, f"median translation {median_translation_pct:.3f}%"
    assert worst_translation_pct < 3.0
    # Scale is not a nuisance parameter here: it is the reconstruction's whole size, and
    # a similarity fit that had to stretch by 10x would mean the geometry is wrong even
    # though every angle looked fine.
    assert 0.1 < fit.scale < 100.0


@requires_colmap
def test_the_camera_up_estimate_is_the_scenes_up(
    reconstruction: tuple[Workdir, tree_frames.Truth],
) -> None:
    """What `place` levels a GPS-less capture by, scored against the orbit's own up.

    The rendered orbit is in the fixture's east/north/up frame, so its up is +z. The
    estimate is in COLMAP's arbitrary frame, so it is carried into the truth's by the same
    similarity the pose score uses before it is compared. Held upright, a camera orbiting
    a tree averages its tilt away, so this should be within a degree or two; the bound is
    where a sign or axis error would show (those land at 90 or 180 degrees).
    """
    workdir, truth = reconstruction
    poses = json.loads((workdir.artifact_path("pose", "poses") / "poses.json").read_text())
    model = sfm.read_model(workdir.artifact_path("pose", "poses"))
    known = truth.by_name()
    fit = sfm.umeyama(
        np.stack([image.centre for image in model.images]),
        np.stack([known[image.name].centre for image in model.images]),
    )

    estimate = np.asarray(poses["upEstimate"]["up"])
    in_truth = fit.rotation @ estimate
    angle = float(np.degrees(np.arccos(np.clip(in_truth @ np.array([0.0, 0.0, 1.0]), -1, 1))))

    assert poses["upEstimate"]["method"] == "camera-up"
    assert poses["upEstimate"]["frames"] == FRAMES
    assert angle < 5.0, f"camera-up is {angle:.2f} deg from the scene's up"
    # And the rotation `place` builds from it takes it exactly onto +z.
    # (poses.json rounds to six places, so it is renormalised first.)
    unit = estimate / np.linalg.norm(estimate)
    assert sfm.rotation_onto_z(unit) @ unit == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)


def test_rotation_onto_z_is_proper_and_handles_both_poles() -> None:
    for up in ([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [0.3, -0.9, 0.1]):
        rotation = sfm.rotation_onto_z(up)
        unit = np.asarray(up) / np.linalg.norm(up)
        assert rotation @ unit == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-9)


@requires_colmap
def test_the_self_calibrated_focal_is_recorded_with_its_bias_rather_than_trusted(
    reconstruction: tuple[Workdir, tree_frames.Truth],
) -> None:
    """A0 #7 measured the free focal 3.1% low; this fixture measures it 0.17% high.

    The two disagree, so what the stage writes down is both measurements and the fact
    that the scale is unverified -- not a correction factor somebody would then apply.
    """
    workdir, truth = reconstruction

    poses = json.loads((workdir.artifact_path("pose", "poses") / "poses.json").read_text())

    assert poses["focal"]["priorPx"] is None
    assert poses["focal"]["refined"] is True
    assert "unverified" in poses["focal"]["biasNote"]
    assert poses["scale"]["metric"] is False
    error_pct = 100.0 * (poses["focal"]["px"] - truth.intrinsics.fx) / truth.intrinsics.fx
    assert abs(error_pct) < 5.0, f"focal off by {error_pct:.2f}%"


@requires_colmap
def test_the_poses_artifact_is_a_colmap_model_anything_downstream_can_read(
    reconstruction: tuple[Workdir, tree_frames.Truth],
) -> None:
    """gsplat, OpenSplat and nerfstudio all read this layout; so does `sfm.read_model`."""
    workdir, _ = reconstruction
    out = workdir.artifact_path("pose", "poses")

    names = sorted(p.name for p in out.iterdir())

    assert {"cameras.bin", "images.bin", "points3D.bin", "poses.json"} <= set(names)
    step = json.loads(workdir.step_path("pose").read_text())
    assert step["metrics"]["registered"] == FRAMES
    assert step["metrics"]["matcher"] == "exhaustive"
    assert step["metrics"]["focalPrior"] is False
    artifact = next(a for a in step["artifacts"] if a["name"] == "poses")
    assert artifact["kind"] == "dir" and artifact["bytes"] > 0


# --- partial registration -------------------------------------------------------------
#
# Added by the orchestrator after a one-in-ten standalone run of the fixture above came
# back with a subset of the orbit: COLMAP's mapper had split the reconstruction and
# `_largest_model` took the biggest component, which is the right behaviour and also the
# quiet one. The count was already in the metrics, the summary and the log; nothing said
# it was a disappointment. These cover the sentence that now does.


def test_a_complete_reconstruction_warns_about_nothing() -> None:
    assert stages.partial_registration_warning(40, 40) is None


def test_a_partial_reconstruction_says_how_many_frames_were_lost() -> None:
    warning = stages.partial_registration_warning(27, 40)
    assert warning is not None
    assert "13 of 40 frames did not register" in warning
    assert "67.5% registered" in warning


def test_no_frames_is_not_a_division_by_zero() -> None:
    """`_largest_model` returning None already raises for an empty model; this is the
    guard that stops the reporting path dividing by zero on the way there."""
    assert stages.partial_registration_warning(0, 0) is None


# --- retries: the smoke's 2/40 ---------------------------------------------------------


def _scripted(results: dict[tuple[int, int], int]):  # type: ignore[no-untyped-def]
    """An `attempt` that answers from a table, and records what it was asked."""
    asked: list[tuple[int, int]] = []

    def attempt(round_: int, seed: int) -> tuple[Path | None, int]:
        asked.append((round_, seed))
        registered = results.get((round_, seed), 0)
        return (Path(f"m{round_}-{seed}") if registered else None), registered

    return attempt, asked


def test_a_good_first_mapping_is_not_retried() -> None:
    attempt, asked = _scripted({(0, 0): 40})
    found, tries = stages._best_reconstruction(attempt, rounds=2, seeds=(0, 1, 2), enough=32)
    assert found == Path("m0-0")
    assert asked == [(0, 0)]
    assert tries == [{"match": 0, "seed": 0, "registered": 40}]


def test_a_mapping_that_closes_on_two_frames_is_retried_with_other_seeds() -> None:
    """The first real Modal smoke: 2 of 40 on the runner's matches, seed 0."""
    attempt, asked = _scripted({(0, 0): 2, (0, 1): 40})
    found, _ = stages._best_reconstruction(attempt, rounds=2, seeds=(0, 1, 2), enough=32)
    assert found == Path("m0-1")
    assert asked == [(0, 0), (0, 1)]


def test_when_no_seed_helps_the_frames_are_matched_again() -> None:
    attempt, asked = _scripted({(0, 0): 2, (0, 1): 3, (0, 2): 2, (1, 0): 39})
    found, _ = stages._best_reconstruction(attempt, rounds=2, seeds=(0, 1, 2), enough=32)
    assert found == Path("m1-0")
    assert asked[-1] == (1, 0)


def test_the_best_attempt_is_kept_when_none_is_enough() -> None:
    """A capture that genuinely does not close still gets its best model, not its last."""
    attempt, _ = _scripted({(0, 0): 2, (0, 1): 20, (0, 2): 5, (1, 0): 3, (1, 1): 4, (1, 2): 1})
    found, tries = stages._best_reconstruction(attempt, rounds=2, seeds=(0, 1, 2), enough=32)
    assert found == Path("m0-1")
    assert len(tries) == 6
