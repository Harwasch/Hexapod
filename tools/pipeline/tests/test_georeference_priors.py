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

import alignment
import exif
import priors
import stages
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


def _flights(seed: int = 0, *, line: bool = False) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Camera centres in true east/north/up, and the GPS fixes the drone wrote for them.
    `line`: one straight strip, fixes exact -- a model whose centres fix no rotation."""
    rng = np.random.default_rng(seed)
    centres, fixes = [], []
    if line:
        for step in range(30):
            centre = np.array([0.0, -60.0 + 4.0 * step, 30.0])
            centres.append(centre)
            fixes.append(centre.copy())
        return centres, fixes
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


def _georeference(
    root: Path,
    *,
    priors_block: str | None,
    sigma: tuple[float, float] = (5.0, 10.0),
    sigma_source: str = "default",
    shift_east_m: float = 0.0,
    params: dict[str, object] | None = None,
    seed: int = 0,
    line: bool = False,
    dropped: int = 0,
    dropped_sigma: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Run `exif_gps` over the two flights. `priors_block`: None (no priors in poses.json),
    `off`, `held` or `held-no-vertical`; the model is written in the frame bundle adjustment
    held it in (about the fixes' median), moved `shift_east_m` to make one that was not. The
    last `dropped` frames carry fixes and priors (at `dropped_sigma`) but are not in the
    model, as frames pose could not register."""
    centres, fixes = _flights(seed, line=line)
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
    # The origin the priors were held about: the fixes' median, as `pose` takes it, and
    # each true camera centre expressed about it, as bundle adjustment would leave it.
    held_at = exif.median_fix(exif.read_fixes(frames))
    true_fixes = tuple(
        exif.Fix(
            f"c{index}",
            *gps_frames.enu_to_geodetic(
                float(centre[0]), float(centre[1]), float(centre[2]), *ORIGIN
            ),
        )
        for index, centre in enumerate(centres)
    )
    about = exif.enu_offsets(true_fixes, held_at)
    shift = np.array([shift_east_m, 0.0, 0.0])
    model = [np.asarray(about[f"c{i}"]) + shift for i in range(len(centres))]
    poses = workdir.input_path("poses")
    write_model(poses, [(down.T, centre) for centre in model[: len(model) - dropped]])
    document: dict[str, Any] = {"tool": "colmap", "scale": {"metric": priors_block is not None}}
    per_frame = {
        f"frame_{index:04d}.jpg": list(
            dropped_sigma if dropped_sigma and index >= len(fixes) - dropped else sigma
        )
        for index in range(len(fixes))
    }
    if priors_block is not None:
        lat, lon, height = held_at
        document["priors"] = {
            "used": priors_block != "off",
            "coordinateSystem": priors.COORDINATE_SYSTEM,
            "vertical": "unconstrained" if priors_block == "held-no-vertical" else "constrained",
            "origin": {"lat": lat, "lon": lon, "height": height, "source": "median-fix"},
            "sigmaM": {
                "horizontalMedian": float(np.median([p[0] for p in per_frame.values()])),
                "verticalMedian": float(np.median([p[1] for p in per_frame.values()])),
                "source": sigma_source,
            },
            "sigmaByFrame": per_frame,
        }
    (poses / "poses.json").write_text(json.dumps(document))
    execute(
        make_recipe(
            [
                {
                    "id": "georeference",
                    "impl": "exif_gps",
                    "params": {
                        "aligner": "ransac",
                        "max_error_m": 5.0,
                        "seed": 0,
                        **(params or {}),
                    },
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
    # Held where its fixes say, give or take the take-off bias and the noise: not merely
    # level because the identity is level by definition.
    held = georef["alignment"]["heldByPosePriors"]
    assert held["held"] is True
    assert held["residualMedianM"] < BIAS_M + 1.5
    assert georef["alignment"]["residualM"]["median"] < BIAS_M + 1.5
    assert georef["alignment"]["inliers"] > len(_flights()[0]) // 2


def test_with_pose_priors_the_free_fit_is_recorded_as_a_check(tmp_path: Path) -> None:
    georef = _georeference(tmp_path / "held", priors_block="held")
    assert georef["alignment"]["heldByPosePriors"]["origin"]["source"] == "median-fix"
    check = georef["alignment"]["freeFitCheck"]
    assert check["status"] == "fitted"
    # The fit the stage would have applied: it does tilt, which is why it is not applied.
    assert check["tiltDeg"] > MAX_TILT_DEG
    assert check["inliers"] >= 3
    assert {"scale", "rotation", "translationM", "offsetM", "rotationDeg", "inliers"} <= set(check)


@pytest.mark.parametrize(
    ("sigma", "source", "held"),
    [
        ((5.0, 10.0), "default", True),  # the recipe's bound exactly: consumer GNSS
        ((4.7, 9.4), "exif", True),  # an iPhone still outdoors (GPSHPositioningError 4.7 m)
        ((5.01, 10.0), "default", False),
        ((5.0, 10.01), "default", False),
        ((65.0, 130.0), "exif", False),  # an iPhone still indoors: Wi-Fi only
    ],
    ids=["at-bound", "phone-still-outdoors", "h-over", "v-over", "phone-still-indoors"],
)
def test_the_priors_hold_the_model_only_within_the_sigma_bound(
    tmp_path: Path, sigma: tuple[float, float], source: str, held: bool
) -> None:
    georef = _georeference(tmp_path / "s", priors_block="held", sigma=sigma, sigma_source=source)
    decision = georef["alignment"]["heldByPosePriors"]
    assert decision["held"] is held, decision
    tilt = _tilt_deg(georef["frame"]["rotation"])
    assert (tilt < MAX_TILT_DEG) if held else (tilt > MAX_TILT_DEG)
    if not held:
        assert "looser than 5 m / 10 m" in decision["why"]
        assert "freeFitCheck" not in georef["alignment"]


def test_a_model_far_from_its_fixes_is_not_held_whatever_poses_json_says(
    tmp_path: Path,
) -> None:
    georef = _georeference(tmp_path / "far", priors_block="held", shift_east_m=40.0)
    decision = georef["alignment"]["heldByPosePriors"]
    assert decision["held"] is False
    assert decision["residualMedianM"] > 15.0
    assert "was not held to these fixes" in decision["why"]
    assert "freeFitCheck" not in georef["alignment"]


def test_a_held_model_stays_held_when_the_free_fit_finds_no_consensus(tmp_path: Path) -> None:
    """With the production aligner (ransac) a free fit can find no consensus; the check
    failing must not unplace a model the priors held."""
    georef = _georeference(tmp_path / "nc", priors_block="held", params={"max_error_m": 0.001})
    assert georef["alignment"]["heldByPosePriors"]["held"] is True
    assert georef["alignment"]["freeFitCheck"]["status"] == "no-consensus"
    assert "no consensus" in georef["alignment"]["freeFitCheck"]["reason"]
    assert georef["frame"]["source"] == "exif-gps-similarity"
    assert _tilt_deg(georef["frame"]["rotation"]) < MAX_TILT_DEG


def test_without_held_priors_the_same_no_consensus_still_locates_it(tmp_path: Path) -> None:
    georef = _georeference(tmp_path / "nc", priors_block=None, params={"max_error_m": 0.001})
    assert georef["frame"]["source"] != "exif-gps-similarity"


def test_a_non_finite_origin_is_never_held(tmp_path: Path) -> None:
    poses = tmp_path / "poses"
    poses.mkdir()
    (poses / "poses.json").write_text(
        json.dumps(
            {
                "priors": {
                    "used": True,
                    "vertical": "constrained",
                    "origin": {"lat": float("nan"), "lon": 16.2, "height": 300.0},
                    "sigmaM": {"horizontalMedian": 5.0, "verticalMedian": 10.0},
                }
            }
        )
    )
    decision = stages._held_by_pose_priors(
        poses, (float("nan"), 16.2, 300.0), {}, {}, max_sigma_h_m=5.0, max_sigma_v_m=10.0
    )
    assert decision is not None and decision["held"] is False


@pytest.mark.parametrize("priors_block", [None, "off", "held-no-vertical"])
def test_without_held_priors_the_free_similarity_is_applied_as_before(
    tmp_path: Path, priors_block: str | None
) -> None:
    """The phone path (Pumpkin, Spool: videos, whose frames carry no fixes to hold) keeps
    today's free fit, so ground scans do not change: the similarity applied is exactly the
    free fit the held run records as its check, tilt and all."""
    free = _georeference(tmp_path / "free", priors_block=priors_block)
    held = _georeference(tmp_path / "held", priors_block="held")
    check = held["alignment"]["freeFitCheck"]
    assert free["frame"]["rotation"] == check["rotation"]
    assert free["frame"]["scale"] == check["scale"]
    assert free["frame"]["translationM"] == check["translationM"]
    assert _tilt_deg(free["frame"]["rotation"]) > MAX_TILT_DEG
    if priors_block in (None, "off"):
        assert "heldByPosePriors" not in free["alignment"]
    else:
        assert free["alignment"]["heldByPosePriors"]["held"] is False


def test_a_held_model_stays_held_when_its_fixes_fix_no_rotation(tmp_path: Path) -> None:
    """One straight strip: every sample is collinear and the free fit is degenerate. That is
    the check failing, not the model: it stays held."""
    georef = _georeference(tmp_path / "line", priors_block="held", line=True)
    assert georef["alignment"]["heldByPosePriors"]["held"] is True
    check = georef["alignment"]["freeFitCheck"]
    assert check["status"] == "degenerate"
    assert "collinear" in check["reason"]
    assert _tilt_deg(georef["frame"]["rotation"]) == 0.0


def test_the_sigmas_are_the_kept_frames_not_the_planned_ones(tmp_path: Path) -> None:
    """Tight priors on 40 frames pose dropped cannot vouch for loose ones on the 20 it
    kept: the median over every planned frame is 0.05 m, over the kept ones 8 m."""
    georef = _georeference(
        tmp_path / "dropped",
        priors_block="held",
        sigma=(8.0, 16.0),
        dropped=40,
        dropped_sigma=(0.05, 0.1),
    )
    decision = georef["alignment"]["heldByPosePriors"]
    assert decision["held"] is False
    assert decision["sigmaM"]["horizontalMedian"] == 8.0
    assert decision["sigmaM"]["basis"] == "the 20 kept frames"


def test_default_sigma_frames_rest_on_the_residual_guard(tmp_path: Path) -> None:
    """Frames with no accuracy metadata get the consumer default, 5 m / 10 m -- the bound
    exactly, so the sigma gate passes them (NWM's frames are such). For them the residual
    guard is the protection: cameras 40 m from their fixes are not held."""
    georef = _georeference(
        tmp_path / "default", priors_block="held", sigma=(5.0, 10.0), shift_east_m=40.0
    )
    decision = georef["alignment"]["heldByPosePriors"]
    assert decision["sigmaM"]["horizontalMedian"] == priors.DEFAULT_STD_M
    assert decision["held"] is False
    assert "was not held to these fixes" in decision["why"]


def test_a_held_model_few_fixes_agree_with_is_not_called_aligned(tmp_path: Path) -> None:
    """Held, 9 m from its fixes (inside the 15 m guard) and so with none of them within
    max_error_m's 5 m: `align` must not report it as placed by its GNSS."""
    georef = _georeference(tmp_path / "few", priors_block="held", shift_east_m=9.0)
    assert georef["alignment"]["heldByPosePriors"]["held"] is True
    assert georef["alignment"]["inliers"] < 3
    poses = {"priors": {"used": True, "sigmaM": {"horizontalMedian": 5.0}}}
    candidate = alignment.gnss_candidate(georef, poses)
    assert candidate.aligned is False
    assert any("fixes agree with the similarity" in reason for reason in candidate.reasons)
    # D4: an epoch that is not aligned is `unaligned`, the tier below `coarse`, with why.
    verdict, reasons = alignment.decide(
        aligned=candidate.aligned,
        tier=alignment.TIER_OF["drone"],
        checks={},
        platform="drone",
        method=candidate.method,
    )
    assert verdict == "unaligned"
    assert reasons


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_a_zero_or_negative_sigma_is_no_sigma(tmp_path: Path, bad: float) -> None:
    georef = _georeference(tmp_path / "bad", priors_block="held", sigma=(bad, 10.0))
    decision = georef["alignment"]["heldByPosePriors"]
    assert decision["held"] is False
    assert decision["sigmaM"]["horizontalMedian"] is None
