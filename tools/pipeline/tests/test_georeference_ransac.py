"""`georeference` / `exif_gps` with `aligner: ransac`: the seeded numpy Sim3 (`sim3.py`).

The same rendered orbit and exact EXIF GPS as `test_georeference_exif.py`, posed by COLMAP
4.2 through pycolmap (so it runs wherever pycolmap is, with no COLMAP binary). The A/B
against `model_aligner` needs COLMAP 3.9.1 and runs in CI only.

Bounds, and where they come from (none is measured off a run here):

* the residual against exact GPS is COLMAP's own pose error plus the fixture's 3 mm GPS
  quantisation, so it is held to the bound `test_pose_colmap4.py` already holds 4.2's
  poses to -- median centre error under 1% of the scene's extent -- in metres;
* the scale is held to umeyama-against-the-truth within 1%, the bound
  `test_georeference_exif.py` holds `model_aligner` to.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import gps_frames
import numpy as np
import pytest

import colmap4
import global_sfm
import sfm
import tree_frames
from conftest import FIXTURE_PLY, make_recipe
from executor import execute
from runners import LocalRunner, RunnerSet
from workdir import Workdir

FRAMES = 40
ORIGIN = (28.0389, -82.6966, 10.8)
RECIPE_POSE = {
    "matcher": "exhaustive",
    "max_features": 4096,
    "max_image_size": 1600,
    "first_octave": 0,
    "colmap": "4.2",
}
#: test_pose_colmap4's bound on 4.2's median centre error, as a fraction of the extent.
POSE_BOUND_OF_EXTENT = 0.01

requires_pycolmap = pytest.mark.skipif(
    colmap4.version(colmap4.python()) is None,
    reason=f"needs pycolmap=={global_sfm.PYCOLMAP_VERSION} (or ${global_sfm.PYTHON_ENV})",
)
requires_colmap = pytest.mark.skipif(
    not sfm.colmap_available(),
    reason="colmap 3.9.1 is not on this machine (CI installs it)",
)


@pytest.fixture(scope="module")
def orbit(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, tree_frames.Truth]:
    """Untagged frames, and the pose model of the exactly tagged ones (GPS plays no part
    in posing with priors off, so one model serves every tagging below)."""
    if colmap4.version(colmap4.python()) is None:
        pytest.skip("needs pycolmap")
    root = tmp_path_factory.mktemp("ransac-orbit")
    plain = root / "plain" / "frames"
    truth = tree_frames.render_orbit(FIXTURE_PLY, plain, count=FRAMES)
    workdir = Workdir.create(root / "pose")
    frames = workdir.input_path("frames")
    shutil.copytree(plain, frames)
    _tag(frames, truth)
    execute(
        make_recipe([{"id": "pose", "impl": "colmap", "params": RECIPE_POSE}], inputs=["frames"]),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )
    return plain, workdir.artifact_path("pose", "poses"), truth


def _tag(
    frames: Path, truth: tree_frames.Truth, *, moved: dict[str, np.ndarray] | None = None
) -> None:
    centres = {p.name: p.centre + (moved or {}).get(p.name, np.zeros(3)) for p in truth.poses}
    gps_frames.tag_orbit(frames, centres, ORIGIN)


def _georeference(
    root: Path,
    plain: Path,
    poses: Path,
    truth: tree_frames.Truth,
    *,
    params: dict[str, object],
    moved: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], bytes]:
    workdir = Workdir.create(root)
    frames = workdir.input_path("frames")
    shutil.copytree(plain, frames)
    _tag(frames, truth, moved=moved)
    shutil.copytree(poses, workdir.input_path("poses"))
    execute(
        make_recipe(
            [{"id": "georeference", "impl": "exif_gps", "params": params}],
            inputs=["frames", "poses"],
        ),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )
    raw = workdir.artifact_path("georeference", "georef.json").read_bytes()
    document = json.loads(raw)
    assert isinstance(document, dict)
    return document, raw


@requires_pycolmap
def test_ransac_recovers_the_exif_frame(
    orbit: tuple[Path, Path, tree_frames.Truth], tmp_path: Path
) -> None:
    plain, poses, truth = orbit
    georef, _ = _georeference(tmp_path / "g", plain, poses, truth, params={"aligner": "ransac"})

    alignment = georef["alignment"]
    assert alignment["tool"] == "numpy ransac (sim3.py)"
    assert alignment["version"] is None
    assert alignment["seed"] == 0 and alignment["iterations"] == 2000
    assert alignment["inliers"] == FRAMES
    bound = POSE_BOUND_OF_EXTENT * truth.extent_m
    assert alignment["residualM"]["median"] < bound, (alignment["residualM"], bound)
    model = sfm.read_model(poses)
    known = truth.by_name()
    fit = sfm.umeyama(
        np.stack([image.centre for image in model.images]),
        np.stack([known[image.name].centre for image in model.images]),
    )
    assert alignment["scale"] == pytest.approx(fit.scale, rel=0.01)
    assert georef["frame"]["source"] == "exif-gps-similarity"
    assert georef["frame"]["recentre"] is False


@requires_pycolmap
def test_ransac_georef_is_byte_identical_across_two_runs(
    orbit: tuple[Path, Path, tree_frames.Truth], tmp_path: Path
) -> None:
    plain, poses, truth = orbit
    _, first = _georeference(tmp_path / "a", plain, poses, truth, params={"aligner": "ransac"})
    _, second = _georeference(tmp_path / "b", plain, poses, truth, params={"aligner": "ransac"})

    assert first == second


@requires_pycolmap
def test_one_wild_fix_is_dropped(
    orbit: tuple[Path, Path, tree_frames.Truth], tmp_path: Path
) -> None:
    plain, poses, truth = orbit
    wild = truth.poses[7].name
    georef, _ = _georeference(
        tmp_path / "w",
        plain,
        poses,
        truth,
        params={"aligner": "ransac"},
        moved={wild: np.array([50.0, 0.0, 0.0])},
    )

    alignment = georef["alignment"]
    assert alignment["inliers"] == FRAMES - 1
    # 50 m off against a 5 m threshold: the fit that ignores it leaves it ~50 m out.
    assert alignment["residualM"]["perImage"][wild] > alignment["maxErrorM"]


@requires_pycolmap
def test_fixes_with_no_consensus_are_located_not_aligned(
    orbit: tuple[Path, Path, tree_frames.Truth], tmp_path: Path
) -> None:
    """R2 review (#51, medium): fixes that agree on no similarity (every one moved by
    hundreds of metres, independently) gave the sampled fit as the capture's frame. The
    georeference now falls back to the fixes' median, located and not aligned."""
    plain, poses, truth = orbit
    rng = np.random.default_rng(1)
    scattered = {p.name: rng.normal(scale=200.0, size=3) for p in truth.poses}
    georef, _ = _georeference(
        tmp_path / "n", plain, poses, truth, params={"aligner": "ransac"}, moved=scattered
    )

    assert georef["alignment"] is None
    assert georef["frame"]["recentre"] is True
    assert georef["georefMethod"] == "exif-gps"
    assert "no consensus" in georef["note"]


def test_an_unknown_aligner_is_refused(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "r")
    frames = workdir.input_path("frames")
    frames.mkdir(parents=True)
    (frames / "frame_0000.jpg").write_bytes(b"x")
    with pytest.raises(Exception, match="aligner must be model_aligner or ransac, not 'icp'"):
        execute(
            make_recipe(
                [{"id": "georeference", "impl": "exif_gps", "params": {"aligner": "icp"}}],
                inputs=["frames"],
            ),
            workdir,
            RunnerSet(cpu=LocalRunner()),
        )


@requires_pycolmap
@requires_colmap
def test_ransac_and_model_aligner_agree_on_the_exif_orbit(
    orbit: tuple[Path, Path, tree_frames.Truth], tmp_path: Path
) -> None:
    """CI only (COLMAP 3.9.1). Two robust fits of one similarity to exact GPS: they agree
    to the bound each is held to against the truth, not to each other's digits."""
    plain, poses, truth = orbit
    ours, _ = _georeference(tmp_path / "o", plain, poses, truth, params={"aligner": "ransac"})
    theirs, _ = _georeference(tmp_path / "t", plain, poses, truth, params={})

    a, b = ours["alignment"], theirs["alignment"]
    assert a["scale"] == pytest.approx(b["scale"], rel=1e-3)
    gap = np.linalg.norm(np.asarray(a["translationM"]) - np.asarray(b["translationM"]))
    assert gap < 0.05
    angle = sfm.rotation_angle_deg(np.asarray(a["rotation"]), np.asarray(b["rotation"]))
    assert angle < 0.1
