"""`pose` / `colmap` with `mapper: global`: GLOMAP through pycolmap (`global_sfm.py`).

Scored the way `test_pose_colmap.py` scores the incremental mapper -- against the known
poses of the rendered orbit, after a similarity fit -- plus the two ways the global
mapper hands over to the incremental one: it cannot run, or it registers too few.

The global mapper needs pycolmap, in this interpreter or in `$COLMAP_GLOBAL_PYTHON`. CI
installs the wheel (`.github/workflows/ci.yml`), and `test_pycolmap_is_installed_in_ci`
never skips there, for the reason `test_colmap_is_installed_in_ci` gives.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import global_sfm
import sfm
import tree_frames
from conftest import FIXTURE_PLY, make_recipe
from executor import execute
from runners import LocalRunner, RunnerSet
from workdir import Workdir

FRAMES = 40
#: photo-reconstruct's own pose settings, which are also what keep this file quick.
RECIPE_POSE = {
    "matcher": "exhaustive",
    "max_features": 4096,
    "max_image_size": 1600,
    "first_octave": 0,
}


def _has_pycolmap() -> bool:
    try:
        found = subprocess.run(
            [global_sfm.python(), "-c", "import pycolmap"], capture_output=True, check=False
        )
    except OSError:
        return False
    return found.returncode == 0


HAS_PYCOLMAP = _has_pycolmap()
requires_global = pytest.mark.skipif(
    not sfm.colmap_available() or not HAS_PYCOLMAP,
    reason=f"needs colmap and pycolmap (pip install pycolmap=={global_sfm.PYCOLMAP_VERSION}, "
    f"or ${global_sfm.PYTHON_ENV})",
)
requires_colmap = pytest.mark.skipif(not sfm.colmap_available(), reason="needs colmap")


def test_pycolmap_is_installed_in_ci() -> None:
    if os.environ.get("CI", "").lower() == "true":
        assert HAS_PYCOLMAP, "pycolmap is missing under CI, so mapper: global is untested"


def test_the_argv_runs_this_file_with_the_chosen_interpreter() -> None:
    argv = global_sfm.argv(
        "/py", Path("/w/db"), Path("/f"), Path("/w/out"), refine_focal_length=False, num_threads=4
    )

    assert argv[:2] == ["/py", str(Path(global_sfm.__file__).resolve())]
    assert argv[argv.index("--database") + 1] == "/w/db"
    assert "--hold-focal" in argv
    assert argv[-2:] == ["--threads", "4"]


def test_an_unknown_mapper_is_refused(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    frames = workdir.input_path("frames")
    frames.mkdir(parents=True)
    (frames / "a.jpg").write_bytes(b"x")
    recipe = make_recipe(
        [{"id": "pose", "impl": "colmap", "params": {"mapper": "hierarchical"}}],
        inputs=["frames"],
    )
    with pytest.raises(Exception, match="mapper must be incremental or global"):
        execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))


@pytest.fixture(scope="module")
def orbit(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, tree_frames.Truth]:
    frames = tmp_path_factory.mktemp("orbit") / "frames"
    truth = tree_frames.render_orbit(FIXTURE_PLY, frames, count=FRAMES)
    return frames, truth


def _pose(root: Path, frames: Path, **params: object) -> tuple[Workdir, dict[str, object]]:
    workdir = Workdir.create(root)
    shutil.copytree(frames, workdir.input_path("frames"))
    recipe = make_recipe(
        [{"id": "pose", "impl": "colmap", "params": {**RECIPE_POSE, **params}}],
        inputs=["frames"],
    )
    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    poses = json.loads((workdir.artifact_path("pose", "poses") / "poses.json").read_text())
    return workdir, poses


@requires_global
def test_the_global_mapper_recovers_the_orbit_within_the_incremental_bounds(
    tmp_path: Path, orbit: tuple[Path, tree_frames.Truth]
) -> None:
    """Measured 40/40, 0.104 deg and 0.172% of extent (incremental: 0.122 deg, 0.207%)."""
    frames, truth = orbit
    workdir, poses = _pose(tmp_path / "run", frames, mapper="global")

    mapper = poses["mapper"]
    assert isinstance(mapper, dict)
    assert mapper["used"] == "global" and mapper["fellBackToIncremental"] is False
    assert poses["registered"] == FRAMES
    out = workdir.artifact_path("pose", "poses")
    # COLMAP 3.9.1's own model files and nothing of COLMAP 4's rig/frame tables.
    assert sorted(p.name for p in out.iterdir()) == [
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "poses.json",
    ]
    model = sfm.read_model(out)
    known = truth.by_name()
    estimated = np.stack([image.centre for image in model.images])
    true = np.stack([known[image.name].centre for image in model.images])
    fit = sfm.umeyama(estimated, true)
    translation = np.linalg.norm(fit.apply(estimated) - true, axis=1)
    rotation = np.array(
        [
            sfm.rotation_angle_deg(fit.rotation @ image.rotation.T, known[image.name].rotation.T)
            for image in model.images
        ]
    )
    assert float(np.median(rotation)) < 1.0
    assert float(rotation.max()) < 3.0
    assert 100.0 * float(np.median(translation)) / truth.extent_m < 1.0
    assert 100.0 * float(translation.max()) / truth.extent_m < 3.0


@requires_global
def test_a_short_global_result_falls_back_to_the_incremental_mapper(
    tmp_path: Path, orbit: tuple[Path, tree_frames.Truth]
) -> None:
    frames, _truth = orbit
    workdir, poses = _pose(
        tmp_path / "run",
        frames,
        mapper="global",
        min_registered_fraction=1.01,
        rematches=0,
        mapper_seeds=1,
    )

    attempts = poses["mapperAttempts"]
    assert isinstance(attempts, list)
    assert [a["mapper"] for a in attempts] == ["global", "incremental"]
    assert "short of" in workdir.log_path("pose").read_text()


@requires_colmap
def test_a_global_mapper_that_cannot_run_hands_over_to_the_incremental_one(
    tmp_path: Path, orbit: tuple[Path, tree_frames.Truth], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(global_sfm.PYTHON_ENV, str(Path(sys.executable).parent / "no-python"))
    frames, _truth = orbit

    workdir, poses = _pose(tmp_path / "run", frames, mapper="global")

    mapper = poses["mapper"]
    assert isinstance(mapper, dict)
    assert mapper["used"] == "incremental" and mapper["fellBackToIncremental"] is True
    assert poses["registered"] == FRAMES
    assert "the global mapper did not run" in workdir.log_path("pose").read_text()
