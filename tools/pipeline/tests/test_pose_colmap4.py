"""`pose` / `colmap` with `colmap: "4.2"`: COLMAP 4.2 through pycolmap (`colmap4.py`).

Scored the way `test_pose_colmap.py` scores 3.9.1 -- the rendered orbit, against its
known poses after a similarity fit -- plus what makes the variant fit to A/B: the same
`poses` artifact (COLMAP 3.9.1's three files, which 3.9.1's own tools read), the version
recorded, the same exhaustive fill-in fallback, and no quiet fall back to 3.9.1.

Needs pycolmap in this interpreter or `$COLMAP_GLOBAL_PYTHON`, as `test_pose_global.py`
does (CI installs the wheel). The sequential tests also need the faiss vocabulary tree at
`$COLMAP4_VOCAB_TREE`, which only the Modal CPU image carries, so they skip elsewhere.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import colmap4
import global_sfm
import live
import sfm
import tree_frames
from conftest import FIXTURE_PLY, make_recipe
from executor import execute
from runners import LocalRunner, RunnerSet
from workdir import Workdir

FRAMES = 40
#: photo-reconstruct's own pose settings.
RECIPE_POSE = {
    "matcher": "exhaustive",
    "max_features": 4096,
    "max_image_size": 1600,
    "first_octave": 0,
}

PYCOLMAP = colmap4.version(colmap4.python())
requires_pycolmap = pytest.mark.skipif(
    PYCOLMAP is None,
    reason=f"needs pycolmap=={global_sfm.PYCOLMAP_VERSION} (or ${global_sfm.PYTHON_ENV})",
)
VOCAB_TREE = colmap4.vocab_tree_path()
requires_faiss_tree = pytest.mark.skipif(
    PYCOLMAP is None or VOCAB_TREE is None,
    reason=f"needs pycolmap and a faiss vocabulary tree at ${colmap4.VOCAB_TREE_ENV}",
)


# --- argv, which needs nothing installed -------------------------------------------------


def test_the_extraction_argv_carries_every_setting_the_cli_one_does() -> None:
    argv = colmap4.extract_argv(
        "/py",
        Path("/w/db"),
        Path("/f"),
        camera_params=(520.0, 320.0, 240.0, 0.0),
        max_image_size=1600,
        max_features=4096,
        first_octave=0,
        num_threads=4,
    )

    assert argv[:3] == ["/py", str(Path(colmap4.__file__).resolve()), "extract"]
    value = {argv[i]: argv[i + 1] for i in range(3, len(argv) - 1) if argv[i].startswith("--")}
    assert value["--camera-params"] == "520,320,240,0"
    assert value["--max-image-size"] == "1600" and value["--max-features"] == "4096"
    assert value["--first-octave"] == "0" and value["--threads"] == "4"
    plain = colmap4.extract_argv("/py", Path("d"), Path("f"))
    assert "--first-octave" not in plain and "--threads" not in plain


def test_loop_detection_is_asked_for_by_handing_sequential_a_tree() -> None:
    looped = colmap4.match_argv(
        "/py", Path("d"), "sequential", overlap=5, vocab_tree=Path("/t.bin"), num_threads=2
    )
    assert looped[looped.index("--vocab-tree") + 1] == "/t.bin"
    assert looped[looped.index("--overlap") + 1] == "5"
    exhaustive = colmap4.match_argv("/py", Path("d"), "exhaustive", vocab_tree=Path("/t.bin"))
    assert "--vocab-tree" not in exhaustive
    with pytest.raises(ValueError, match="unknown matcher"):
        colmap4.match_argv("/py", Path("d"), "nearest_friend")


def test_the_mapper_argv_holds_a_focal_prior_and_passes_the_seed_and_snapshots() -> None:
    argv = colmap4.map_argv(
        "/py",
        Path("d"),
        Path("i"),
        Path("o"),
        refine_focal_length=False,
        random_seed=2,
        num_threads=4,
        snapshot_path=Path("/s"),
        snapshot_every=3,
    )
    assert "--hold-focal" in argv and "--hold-extra" not in argv
    assert argv[argv.index("--seed") + 1] == "2"
    assert argv[argv.index("--snapshot-every") + 1] == "3"
    assert "--snapshot-path" not in colmap4.map_argv("/py", Path("d"), Path("i"), Path("o"))


def test_the_faiss_tree_has_its_own_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3.9.1's FLANN tree and 4.2's faiss tree are different files; neither finds the other."""
    tree = tmp_path / "vocab_tree_faiss_flickr100K_words32K.bin"
    tree.write_bytes(b"tree")
    monkeypatch.delenv(colmap4.VOCAB_TREE_ENV, raising=False)
    monkeypatch.setenv(sfm.VOCAB_TREE_ENV, str(tree))

    assert colmap4.vocab_tree_path() is None
    monkeypatch.setenv(colmap4.VOCAB_TREE_ENV, str(tree))
    assert colmap4.vocab_tree_path() == tree.resolve()
    assert colmap4.vocab_tree_path(str(tmp_path / "missing.bin")) is None


def _refused(tmp_path: Path, **params: object) -> Workdir:
    workdir = Workdir.create(tmp_path / "run")
    frames = workdir.input_path("frames")
    frames.mkdir(parents=True)
    (frames / "a.jpg").write_bytes(b"x")
    recipe = make_recipe(
        [{"id": "pose", "impl": "colmap", "params": params}],
        inputs=["frames"],
    )
    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    return workdir


def test_an_unknown_colmap_version_is_refused(tmp_path: Path) -> None:
    with pytest.raises(Exception, match=r"colmap must be one of 3\.9, 4\.2"):
        _refused(tmp_path, colmap="3.10")


def test_asking_for_4_2_without_pycolmap_fails_rather_than_running_3_9(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(global_sfm.PYTHON_ENV, str(Path(sys.executable).parent / "no-python"))
    with pytest.raises(Exception, match=r"not falling back to 3\.9\.1"):
        _refused(tmp_path, colmap="4.2")


# --- the real thing ------------------------------------------------------------------------


@requires_pycolmap
def test_the_self_check_the_modal_image_runs_passes_here() -> None:
    """Every option colmap4.py sets exists on this pycolmap -- the image build's check."""
    tree = [] if VOCAB_TREE is None else ["--vocab-tree", str(VOCAB_TREE)]
    done = subprocess.run(
        [colmap4.python(), colmap4.__file__, "self-check", *tree],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "COLMAP 4.2" in done.stdout
    # And 3.9.1's FLANN tree, where there is one, is refused by it rather than accepted.
    flann = sfm.vocab_tree_path()
    if flann is not None:
        refused = subprocess.run(
            [colmap4.python(), colmap4.__file__, "self-check", "--vocab-tree", str(flann)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert refused.returncode != 0 and "faiss" in refused.stderr


@pytest.fixture(scope="module")
def orbit(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, tree_frames.Truth]:
    frames = tmp_path_factory.mktemp("orbit") / "frames"
    truth = tree_frames.render_orbit(FIXTURE_PLY, frames, count=FRAMES)
    return frames, truth


def _pose(
    root: Path, frames: Path, *, video: bool = False, **params: object
) -> tuple[Workdir, dict[str, object]]:
    workdir = Workdir.create(root)
    shutil.copytree(frames, workdir.input_path("frames"))
    inputs = ["frames"]
    # A video is posed with the stage's defaults, as test_pose_colmap.py's video tests
    # pose it on 3.9.1, so the two are the same run on different COLMAPs. (With the
    # recipe's `first_octave: 0` these 640x480 frames have ~800 features each and 4.2's
    # sequential pass registered 15 of 30 before the exhaustive fallback -- colmap4.py.)
    base: dict[str, object] = {"matcher": "auto"} if video else dict(RECIPE_POSE)
    if video:
        workdir.input_path("source_meta.json").write_text(json.dumps({"format": "video"}))
        inputs.append("source_meta.json")
    recipe = make_recipe(
        [{"id": "pose", "impl": "colmap", "params": {**base, "colmap": "4.2", **params}}],
        inputs=inputs,
    )
    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    poses = json.loads((workdir.artifact_path("pose", "poses") / "poses.json").read_text())
    return workdir, poses


def _errors(model: sfm.Model, truth: tree_frames.Truth) -> tuple[np.ndarray, np.ndarray]:
    known = truth.by_name()
    estimated = np.stack([image.centre for image in model.images])
    true = np.stack([known[image.name].centre for image in model.images])
    fit = sfm.umeyama(estimated, true)
    translation = 100.0 * np.linalg.norm(fit.apply(estimated) - true, axis=1) / truth.extent_m
    rotation = np.array(
        [
            sfm.rotation_angle_deg(fit.rotation @ image.rotation.T, known[image.name].rotation.T)
            for image in model.images
        ]
    )
    return rotation, translation


@pytest.fixture(scope="module")
def reconstruction(
    tmp_path_factory: pytest.TempPathFactory, orbit: tuple[Path, tree_frames.Truth]
) -> tuple[Workdir, dict[str, object]]:
    if PYCOLMAP is None:
        pytest.skip("needs pycolmap")
    frames, _truth = orbit
    return _pose(tmp_path_factory.mktemp("four") / "run", frames)


@requires_pycolmap
def test_4_2_registers_the_orbit_within_the_3_9_bounds(
    reconstruction: tuple[Workdir, dict[str, object]], orbit: tuple[Path, tree_frames.Truth]
) -> None:
    """The bounds `test_pose_colmap.py` holds 3.9.1 to; see colmap4.py for the numbers."""
    workdir, poses = reconstruction
    _frames, truth = orbit

    assert poses["version"] == PYCOLMAP == global_sfm.PYCOLMAP_VERSION
    assert poses["colmap"] == {"requested": "4.2", "via": "pycolmap"}
    assert poses["registered"] == FRAMES
    assert isinstance(poses["meanReprojectionErrorPx"], float)
    assert 0.0 < poses["meanReprojectionErrorPx"] < 2.0
    rotation, translation = _errors(sfm.read_model(workdir.artifact_path("pose", "poses")), truth)
    assert float(np.median(rotation)) < 1.0 and float(rotation.max()) < 3.0
    assert float(np.median(translation)) < 1.0 and float(translation.max()) < 3.0
    metrics = json.loads(workdir.step_path("pose").read_text())["metrics"]
    assert metrics["colmap"] == PYCOLMAP
    assert metrics["matchedPairs"] == FRAMES * (FRAMES - 1) // 2
    assert {"extractS", "matchS", "mapS", "meanTrackLength", "meanReprojectionErrorPx"} <= set(
        metrics
    )


@requires_pycolmap
def test_the_4_2_poses_artifact_is_3_9s_own_layout_and_3_9_reads_it(
    reconstruction: tuple[Workdir, dict[str, object]],
) -> None:
    """`georeference` runs 3.9.1's `model_aligner` on this, so 4.2's rig and frame files
    are left out and 3.9.1's `model_analyzer` must read what remains."""
    workdir, _poses = reconstruction
    out = workdir.artifact_path("pose", "poses")

    assert sorted(p.name for p in out.iterdir()) == [
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "poses.json",
    ]
    if sfm.colmap_available():
        done = subprocess.run(
            [sfm.colmap_exe(), "model_analyzer", "--path", str(out)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert f"Registered images: {FRAMES}" in done.stdout + done.stderr


@requires_pycolmap
def test_4_2s_mapper_snapshots_reach_the_live_viewer(
    reconstruction: tuple[Workdir, dict[str, object]],
) -> None:
    workdir, _poses = reconstruction
    log = workdir.log_path("pose").read_text()

    assert "--snapshot-every" in log
    assert [line for line in log.splitlines() if line.startswith(live.CAMERAS_TAG)]
    assert live.latest(log)["cameras"]["final"] is True


@requires_pycolmap
def test_4_2_and_the_global_mapper_run_on_the_same_database(
    tmp_path: Path, orbit: tuple[Path, tree_frames.Truth]
) -> None:
    frames, _truth = orbit
    _workdir, poses = _pose(tmp_path / "run", frames, mapper="global")

    mapper = poses["mapper"]
    assert isinstance(mapper, dict) and mapper["used"] == "global"
    assert poses["registered"] == FRAMES


VIDEO_FRAMES = 30


@pytest.fixture(scope="module")
def video_frames(tmp_path_factory: pytest.TempPathFactory) -> Path:
    frames = tmp_path_factory.mktemp("video") / "frames"
    tree_frames.render_orbit(FIXTURE_PLY, frames, count=VIDEO_FRAMES)
    return frames


@requires_faiss_tree
def test_4_2_matches_a_video_sequentially_with_loop_closure(
    tmp_path: Path, video_frames: Path
) -> None:
    _workdir, poses = _pose(tmp_path / "run", video_frames, video=True)

    assert poses["registered"] == VIDEO_FRAMES and poses["matcher"] == "sequential"
    matching = poses["matching"]
    assert isinstance(matching, dict)
    assert VOCAB_TREE is not None and matching["vocabTree"] == VOCAB_TREE.name
    assert matching["passes"][0]["loopDetection"] is True
    assert matching["passes"][0]["pairs"] < VIDEO_FRAMES * (VIDEO_FRAMES - 1) // 2
    # 3.9.1's pairs: every frame within `overlap` (10) of each other, and i + 16 too --
    # not 4.2's own quadratic-only window (colmap4._matching), which is 180 here.
    linear = sum(min(10, VIDEO_FRAMES - 1 - i) for i in range(VIDEO_FRAMES))
    assert matching["passes"][0]["pairs"] >= linear + (VIDEO_FRAMES - 16)


@requires_faiss_tree
def test_4_2s_exhaustive_fallback_fills_in_only_the_pairs_sequential_skipped(
    tmp_path: Path, video_frames: Path
) -> None:
    _workdir, poses = _pose(
        tmp_path / "run",
        video_frames,
        video=True,
        min_registered_fraction=1.01,
        rematches=0,
    )

    matching = poses["matching"]
    assert isinstance(matching, dict) and matching["fellBackToExhaustive"] is True
    passes = matching["passes"]
    assert [(p["matcher"], p["clear"]) for p in passes] == [
        ("sequential", False),
        ("exhaustive", False),
    ]
    assert passes[0]["pairs"] < passes[1]["pairs"] == VIDEO_FRAMES * (VIDEO_FRAMES - 1) // 2
