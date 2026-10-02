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
import shutil
from pathlib import Path

import numpy as np
import pytest

import live
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


@requires_colmap
def test_the_log_carries_cameras_as_they_were_solved_and_the_final_model(
    reconstruction: tuple[Workdir, tree_frames.Truth],
) -> None:
    """The live viewer's cameras come from the real mapper's own snapshots.

    COLMAP 3.9.1 writes one every `snapshot_images_freq` registered images; the watcher
    logs them as `live-cameras:` lines, and the chosen model is logged once more, final.
    """
    workdir, _ = reconstruction
    log = workdir.log_path("pose").read_text()
    lines = [line for line in log.splitlines() if line.startswith(live.CAMERAS_TAG)]

    assert "--Mapper.snapshot_path" in log
    assert lines, "no live-cameras line at all"
    parsed = [live.parse_line(line) for line in lines]
    assert all(entry is not None and entry[0] == "cameras" for entry in parsed)
    final = live.latest(log)["cameras"]
    assert final["final"] is True and final["registered"] == FRAMES
    assert final["cameraCount"] == FRAMES and final["pointCount"] > 100
    # Snapshots are working files: none is left behind.
    assert not (workdir.work_dir("pose") / "snapshots").exists() or not any(
        (workdir.work_dir("pose") / "snapshots").rglob("*.bin")
    )


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
    found, tries = stages._best_reconstruction(attempt, rounds=[(0, 1, 2), (0, 1, 2)], enough=32)
    assert found == Path("m0-0")
    assert asked == [(0, 0)]
    assert tries == [{"match": 0, "seed": 0, "registered": 40}]


def test_a_mapping_that_closes_on_two_frames_is_retried_with_other_seeds() -> None:
    """The first real Modal smoke: 2 of 40 on the runner's matches, seed 0."""
    attempt, asked = _scripted({(0, 0): 2, (0, 1): 40})
    found, _ = stages._best_reconstruction(attempt, rounds=[(0, 1, 2), (0, 1, 2)], enough=32)
    assert found == Path("m0-1")
    assert asked == [(0, 0), (0, 1)]


def test_when_no_seed_helps_the_frames_are_matched_again() -> None:
    attempt, asked = _scripted({(0, 0): 2, (0, 1): 3, (0, 2): 2, (1, 0): 39})
    found, _ = stages._best_reconstruction(attempt, rounds=[(0, 1, 2), (0, 1, 2)], enough=32)
    assert found == Path("m1-0")
    assert asked[-1] == (1, 0)


def test_the_best_attempt_is_kept_when_none_is_enough() -> None:
    """A capture that genuinely does not close still gets its best model, not its last."""
    attempt, _ = _scripted({(0, 0): 2, (0, 1): 20, (0, 2): 5, (1, 0): 3, (1, 1): 4, (1, 2): 1})
    found, tries = stages._best_reconstruction(attempt, rounds=[(0, 1, 2), (0, 1, 2)], enough=32)
    assert found == Path("m0-1")
    assert len(tries) == 6


def test_a_sequential_round_maps_once_and_the_fallback_gets_every_seed() -> None:
    """Rounds carry their own seeds: one mapping after sequential matching, then the
    exhaustive fill-in with all three."""
    attempt, asked = _scripted({(0, 0): 20, (1, 0): 2, (1, 1): 40})
    found, tries = stages._best_reconstruction(attempt, rounds=[(0,), (0, 1, 2)], enough=32)
    assert found == Path("m1-1")
    assert asked == [(0, 0), (1, 0), (1, 1)]
    assert [t["match"] for t in tries] == [0, 1, 1]


# --- which pairs get matched ------------------------------------------------------------


def test_a_video_is_matched_sequentially_with_loop_closure_then_exhaustively() -> None:
    plan = sfm.matching_plan("auto", source_format="video", vocab_tree=True, rematches=1)

    assert [(p.matcher, p.loop_detection, p.clear) for p in plan] == [
        ("sequential", True, False),
        # The fallback fills in: no clear, so only the pairs sequential skipped are matched.
        ("exhaustive", False, False),
        ("exhaustive", False, True),
    ]
    assert "fallback" in plan[1].why


def test_a_photo_set_or_an_unknown_source_is_matched_exhaustively() -> None:
    for source in ("images", None):
        plan = sfm.matching_plan("auto", source_format=source, vocab_tree=True, rematches=0)
        assert [p.matcher for p in plan] == ["exhaustive"]


def test_sequential_never_runs_without_a_vocabulary_tree() -> None:
    """A0 #7's 2/40 is sequential matching with no loop detection. Asked for by name or
    chosen for a video, with no tree it is exhaustive, and the plan says why."""
    for requested, source in (("auto", "video"), ("sequential", "images")):
        plan = sfm.matching_plan(requested, source_format=source, vocab_tree=False, rematches=0)
        assert [p.matcher for p in plan] == ["exhaustive"]
        assert "no vocabulary tree" in plan[0].why


def test_exhaustive_by_name_is_exhaustive_and_an_unknown_matcher_is_refused() -> None:
    plan = sfm.matching_plan("exhaustive", source_format="video", vocab_tree=True, rematches=2)
    assert [(p.matcher, p.clear) for p in plan] == [
        ("exhaustive", False),
        ("exhaustive", True),
        ("exhaustive", True),
    ]
    with pytest.raises(ValueError, match="unknown matcher"):
        sfm.matching_plan("nearest_friend", source_format=None, vocab_tree=True)


def test_the_vocabulary_tree_is_found_by_param_or_environment_and_only_if_it_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = tmp_path / "vocab_tree_flickr100K_words32K.bin"
    tree.write_bytes(b"tree")
    monkeypatch.delenv(sfm.VOCAB_TREE_ENV, raising=False)

    assert sfm.vocab_tree_path() is None
    assert sfm.vocab_tree_path(str(tree)) == tree.resolve()
    monkeypatch.setenv(sfm.VOCAB_TREE_ENV, str(tree))
    assert sfm.vocab_tree_path() == tree.resolve()
    monkeypatch.setenv(sfm.VOCAB_TREE_ENV, str(tmp_path / "missing.bin"))
    assert sfm.vocab_tree_path() is None


@requires_colmap
def test_the_threads_and_octave_flags_reach_colmap_only_when_given() -> None:
    plain = sfm.feature_extractor_argv(Path("d"), Path("i"))
    tuned = sfm.feature_extractor_argv(Path("d"), Path("i"), first_octave=0, num_threads=4)

    assert "--SiftExtraction.first_octave" not in plain
    assert tuned[tuned.index("--SiftExtraction.first_octave") + 1] == "0"
    assert tuned[tuned.index("--SiftExtraction.num_threads") + 1] == "4"
    matcher = sfm.matcher_argv(Path("d"), "exhaustive", num_threads=4)
    assert matcher[matcher.index("--SiftMatching.num_threads") + 1] == "4"
    mapper = sfm.mapper_argv(Path("d"), Path("i"), Path("o"), num_threads=4)
    assert mapper[mapper.index("--Mapper.num_threads") + 1] == "4"


# --- a video's frames, matched for real ------------------------------------------------
#
# These need COLMAP and the vocabulary tree. The Modal CPU image has both; CI and the
# worker have no tree, so there the plan is exhaustive (tested above) and these skip.

VOCAB_TREE = sfm.vocab_tree_path()
requires_vocab_tree = pytest.mark.skipif(
    not sfm.colmap_available() or VOCAB_TREE is None,
    reason=f"needs colmap and a vocabulary tree at ${sfm.VOCAB_TREE_ENV} (infra/modal/app.py)",
)
VIDEO_FRAMES = 30


def _pose_a_video(root: Path, frames: Path, **params: object) -> Workdir:
    workdir = Workdir.create(root)
    shutil.copytree(frames, workdir.input_path("frames"))
    workdir.input_path("source_meta.json").write_text(json.dumps({"format": "video"}))
    recipe = make_recipe(
        [{"id": "pose", "impl": "colmap", "params": {"matcher": "auto", **params}}],
        inputs=["frames", "source_meta.json"],
    )
    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    return workdir


@pytest.fixture(scope="module")
def video_frames(tmp_path_factory: pytest.TempPathFactory) -> Path:
    frames = tmp_path_factory.mktemp("video") / "frames"
    tree_frames.render_orbit(FIXTURE_PLY, frames, count=VIDEO_FRAMES)
    return frames


@requires_vocab_tree
def test_a_videos_frames_are_matched_sequentially_and_close_the_orbit(
    tmp_path: Path, video_frames: Path
) -> None:
    workdir = _pose_a_video(tmp_path / "run", video_frames)

    poses = json.loads((workdir.artifact_path("pose", "poses") / "poses.json").read_text())
    assert poses["registered"] == VIDEO_FRAMES
    assert poses["matcher"] == "sequential"
    matching = poses["matching"]
    assert matching["sourceFormat"] == "video"
    assert matching["fellBackToExhaustive"] is False
    assert [p["matcher"] for p in matching["passes"]] == ["sequential"]
    assert matching["passes"][0]["loopDetection"] is True
    assert matching["passes"][0]["pairs"] < VIDEO_FRAMES * (VIDEO_FRAMES - 1) // 2


@requires_vocab_tree
def test_a_short_sequential_result_falls_back_to_filling_in_every_pair(
    tmp_path: Path, video_frames: Path
) -> None:
    """Forced: a fraction no model can reach, so every planned pass runs. The fill-in
    matches only what sequential skipped -- the pair count ends at n(n-1)/2, not above."""
    workdir = _pose_a_video(
        tmp_path / "run", video_frames, min_registered_fraction=1.01, rematches=0
    )

    poses = json.loads((workdir.artifact_path("pose", "poses") / "poses.json").read_text())
    matching = poses["matching"]
    assert matching["fellBackToExhaustive"] is True
    passes = matching["passes"]
    assert [(p["matcher"], p["clear"]) for p in passes] == [
        ("sequential", False),
        ("exhaustive", False),
    ]
    assert passes[0]["pairs"] < passes[1]["pairs"] == VIDEO_FRAMES * (VIDEO_FRAMES - 1) // 2
    assert poses["registered"] == VIDEO_FRAMES
    assert [t["seed"] for t in poses["mapperAttempts"] if t["matcher"] == "sequential"] == [0]
    log = workdir.log_path("pose").read_text()
    assert "short of min_registered_fraction" in log
    assert "fallback" in log
