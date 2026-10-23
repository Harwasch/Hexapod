"""`exif_gps` when pose priors already put the model in the fixes' frame.

The NWM drone capture (2026-10-23) is why this exists: two flights from take-offs whose
GPS altitudes were ~9 m apart, bundle-adjusted with each frame's fix as a prior, came out
of `pose` level -- every constant-altitude flight within ~1 deg -- and `georeference` then
fitted a free similarity to the same fixes and tilted the whole site 10.5 deg, because a
tilt kept more of the biased fixes within 5 m than level did. With the priors held in
bundle adjustment about the same origin, the model is already in that frame
(`priors.py`): the similarity is the identity, and the free fit is recorded as a check.

No pycolmap here: the pose model is written as COLMAP writes one (`synthetic_scene`), and
the frames are tagged with GPS (`gps_frames`). The geometry is the drone's: two lawnmower
flights side by side, 30 m up, cameras pointing down.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import gps_frames
import numpy as np
import PIL.Image
import pytest

import exif
import priors
from conftest import make_recipe
from executor import execute
from runners import LocalRunner, RunnerSet
from synthetic_scene import write_model
from workdir import Workdir

ORIGIN = (48.20609, 16.231378, 326.8)
#: Half the take-off gap, metres: the west flight's fixes read high, the east's low.
BIAS_M = 4.5
#: The bound on the placed model's tilt (DONE WHEN for the fix): a level model stays level.
MAX_TILT_DEG = 1.5


def _flights(seed: int = 0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Camera centres in true east/north/up, and the GPS fixes the drone wrote for them."""
    rng = np.random.default_rng(seed)
    centres, fixes = [], []
    for west, bias in ((True, BIAS_M), (False, -BIAS_M)):
        for row in range(5):
            for col in range(6):
                east = (-70.0 if west else 10.0) + 12.0 * col
                north = -50.0 + 25.0 * row
                centre = np.array([east, north, 30.0])
                centres.append(centre)
                noise = np.array([*rng.normal(0.0, 1.5, 2), rng.normal(0.0, 0.5)])
                fixes.append(centre + np.array([0.0, 0.0, bias]) + noise)
    return centres, fixes


def _georeference(root: Path, *, priors_block: str | None, seed: int = 0) -> dict[str, Any]:
    """Run `exif_gps` over the two flights. `priors_block`: None (no priors in poses.json),
    `off`, `held` or `held-no-vertical`."""
    centres, fixes = _flights(seed)
    down = np.diag([1.0, -1.0, -1.0])  # world_from_camera: looking straight down
    workdir = Workdir.create(root)
    frames = workdir.input_path("frames")
    frames.mkdir(parents=True)
    names = {}
    for index, fix in enumerate(fixes):
        name = f"frame_{index:04d}.jpg"
        PIL.Image.new("RGB", (16, 12)).save(frames / name)
        names[name] = fix
    gps_frames.tag_orbit(frames, names, ORIGIN)
    poses = workdir.input_path("poses")
    write_model(poses, [(down.T, centre) for centre in centres])
    document: dict[str, Any] = {"tool": "colmap", "scale": {"metric": priors_block is not None}}
    if priors_block is not None:
        # The origin bundle adjustment held the fixes about: their median, as `pose` takes it.
        lat, lon, height = exif.median_fix(exif.read_fixes(frames))
        document["priors"] = {
            "used": priors_block != "off",
            "coordinateSystem": priors.COORDINATE_SYSTEM,
            "vertical": "unconstrained" if priors_block == "held-no-vertical" else "constrained",
            "origin": {"lat": lat, "lon": lon, "height": height, "source": "median-fix"},
        }
    (poses / "poses.json").write_text(json.dumps(document))
    execute(
        make_recipe(
            [
                {
                    "id": "georeference",
                    "impl": "exif_gps",
                    "params": {"aligner": "ransac", "max_error_m": 5.0, "seed": 0},
                }
            ],
            inputs=["frames", "poses"],
        ),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )
    georef = json.loads(workdir.artifact_path("georeference", "georef.json").read_text())
    assert isinstance(georef, dict)
    return georef


def _tilt_deg(rotation: list[list[float]]) -> float:
    """How far the placed frame's up is from the model's up."""
    up = np.asarray(rotation, dtype=np.float64) @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.acos(min(1.0, float(up[2]))))


def test_with_pose_priors_a_level_model_stays_level_through_biased_fixes(
    tmp_path: Path,
) -> None:
    georef = _georeference(tmp_path / "held", priors_block="held")
    assert _tilt_deg(georef["frame"]["rotation"]) < MAX_TILT_DEG
    assert georef["frame"]["scale"] == pytest.approx(1.0)


def test_with_pose_priors_the_free_fit_is_recorded_as_a_check(tmp_path: Path) -> None:
    georef = _georeference(tmp_path / "held", priors_block="held")
    assert georef["alignment"]["heldByPosePriors"]["origin"]["source"] == "median-fix"
    check = georef["alignment"]["freeFitCheck"]
    # The fit the stage would have applied: it does tilt, which is why it is not applied.
    assert check["rotationDeg"] > MAX_TILT_DEG
    assert check["tiltDeg"] > MAX_TILT_DEG
    assert check["inliers"] >= 3
    assert {"scale", "rotation", "translationM", "rotationDeg", "tiltDeg", "inliers"} <= set(check)


@pytest.mark.parametrize("priors_block", [None, "off", "held-no-vertical"])
def test_without_held_priors_the_free_similarity_is_applied_as_before(
    tmp_path: Path, priors_block: str | None
) -> None:
    """The phone path (Pumpkin, Spool: no per-frame fixes held in bundle adjustment) keeps
    today's free fit, so ground scans do not change: the similarity applied is exactly the
    free fit the held run records as its check, tilt and all."""
    free = _georeference(tmp_path / "free", priors_block=priors_block)
    held = _georeference(tmp_path / "held", priors_block="held")
    check = held["alignment"]["freeFitCheck"]
    assert free["frame"]["rotation"] == check["rotation"]
    assert free["frame"]["scale"] == check["scale"]
    assert free["frame"]["translationM"] == check["translationM"]
    assert _tilt_deg(free["frame"]["rotation"]) > MAX_TILT_DEG
    assert "heldByPosePriors" not in free["alignment"]
