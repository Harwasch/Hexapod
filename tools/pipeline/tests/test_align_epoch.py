"""`align_epoch` (`alignment.py`): the verdict an epoch may claim, and how it is decided.

The decision functions are pure. The stage runs here without pycolmap: poses from
`synthetic_scene.write_model`, georeferences written by hand, and -- where a visual
alignment must have happened -- `colmap4 register`'s output fabricated in its exact shape,
so the rules that choose a frame are tested apart from SIFT. The real registration is
`test_registration_epochs.py`'s.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import alignment
import check_targets
import exif
import global_sfm
import sfm
import synthetic_scene
from conftest import make_recipe
from errors import QualityGateError, StageFailedError
from executor import execute
from runners import LocalRunner, RunnerSet
from workdir import Workdir

ORIGIN = (28.0389, -82.6966, 10.8)
COUNT = 10


# --- the decision, pure -----------------------------------------------------------------


def test_rtk_with_passing_targets_is_survey() -> None:
    checks = {"checks": 3, "withinTolerance": True}
    assert alignment.decide(aligned=True, tier="survey", checks=checks, platform="rtk-drone") == (
        "survey",
        [],
    )


def test_rtk_without_targets_is_coarse_and_says_why() -> None:
    verdict, reasons = alignment.decide(
        aligned=True, tier="survey", checks={"checks": 0}, platform="rtk-drone"
    )
    assert verdict == "coarse"
    assert reasons == ["accuracy unverified: 0 check target(s) evaluated, fewer than 3"]
    failing = {
        "checks": 8,
        "withinTolerance": False,
        "rmseM": {"horizontal": 0.4, "vertical": 0.1},
        "tolerance": {"horizontalM": 0.12, "verticalM": 0.11},
    }
    verdict, reasons = alignment.decide(
        aligned=True, tier="survey", checks=failing, platform="rtk-drone"
    )
    assert verdict == "coarse" and "out of tolerance" in reasons[0]


def test_a_phone_epoch_is_never_survey_even_with_perfect_numbers() -> None:
    assert [p for p, tier in alignment.TIER_OF.items() if tier == "survey"] == ["rtk-drone"]
    perfect = {"checks": 8, "withinTolerance": True}
    for platform in ("phone-video", "phone-photo", "drone", "unknown"):
        verdict, reasons = alignment.decide(
            aligned=True, tier=alignment.TIER_OF[platform], checks=perfect, platform=platform
        )
        assert verdict == "coarse", platform
        assert "only an RTK drone can be surveyed" in reasons[0]
    assert alignment.decide(aligned=False, tier="survey", checks=perfect, platform="rtk-drone")[
        0
    ] == ("unaligned")


def _ring_model(root: Path) -> tuple[Path, sfm.Model]:
    poses = synthetic_scene.write_model(root, synthetic_scene.ring(COUNT, radius=6.0, height=2.0))
    _write_points(poses)
    return poses, sfm.read_model(poses)


def _write_points(poses: Path) -> None:
    """A handful of sparse points round the ring's target, with empty tracks: enough for
    the pixel footprint (camera to the points' centroid over the focal)."""
    points = synthetic_scene.TARGET + np.random.default_rng(4).normal(scale=0.3, size=(12, 3))
    with (poses / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for index, xyz in enumerate(points):
            handle.write(struct.pack("<Q3d3Bd", index + 1, *xyz, 128, 128, 128, 0.5))
            handle.write(struct.pack("<Q", 0))


def _registered(model: sfm.Model, similarity: sfm.Similarity, names: list[str]) -> dict[str, Any]:
    by = model.by_name()
    return {
        "newFramesUsed": COUNT,
        "registeredNew": {
            name: {
                "centre": similarity.apply(by[name].centre.reshape(1, 3))[0].tolist(),
                "rotation": (by[name].rotation @ similarity.rotation.T).tolist(),
            }
            for name in names
        },
    }


TRUE = sfm.Similarity(2.0, sfm.quat_to_matrix([0.9, 0.1, -0.2, 0.3]), np.array([5.0, -3.0, 1.0]))
IDENTITY = sfm.Similarity(1.0, np.eye(3), np.zeros(3))


def _reference(root: Path, verdict: str = "survey", uncertainty_m: float = 0.03) -> Any:
    return alignment.Reference(
        epoch_id="e1",
        job_id="j1",
        observed_at=None,
        platform="rtk-drone",
        verdict=verdict,
        uncertainty_m=uncertainty_m,
        frames=root,
        poses=root,
        placement=root,
        georef=root,
    )


def test_too_few_inliers_or_registered_frames_is_unaligned(tmp_path: Path) -> None:
    poses, model = _ring_model(tmp_path / "m")
    names = [image.name for image in model.images]
    common = {
        "model_dir": poses,
        "model": model,
        "to_target": IDENTITY,
        "reference": _reference(tmp_path),
        "min_registered_fraction": 0.5,
        "seed": 0,
    }

    all_of_them = alignment.visual_candidate(_registered(model, TRUE, names), **common)
    assert all_of_them.aligned and all_of_them.inliers == COUNT
    assert all_of_them.similarity is not None
    assert all_of_them.similarity.scale == pytest.approx(2.0)

    four = alignment.visual_candidate(_registered(model, TRUE, names[:4]), **common)
    assert not four.aligned
    assert any("under min_registered_fraction" in r for r in four.reasons)
    assert any("fewer than 6" in r for r in four.reasons)

    # All ten registered, but six of them each somewhere else: no majority agrees.
    wild = _registered(model, TRUE, names)
    rng = np.random.default_rng(1)
    for name in names[:6]:
        wild["registeredNew"][name]["centre"] = (
            np.asarray(wild["registeredNew"][name]["centre"]) + rng.uniform(20, 40, 3)
        ).tolist()
    split = alignment.visual_candidate(wild, **common)
    assert not split.aligned and split.inliers < alignment.MIN_SIM3_INLIERS


def test_confidence_is_zero_when_unaligned_and_falls_with_rms() -> None:
    def candidate(residual: float) -> alignment.Candidate:
        names = tuple(f"f{i}" for i in range(20))
        return alignment.Candidate(
            method="visual",
            aligned=True,
            similarity=IDENTITY,
            correspondences=20,
            inliers=20,
            residuals=dict.fromkeys(names, residual),
            inlier_names=names,
            registered_fraction=1.0,
            tol_align=0.06,
        )

    assert alignment.confidence(candidate(0.01), verdict="unaligned", checks={}) == 0.0
    good = alignment.confidence(candidate(0.01), verdict="coarse", checks={})
    worse = alignment.confidence(candidate(0.04), verdict="coarse", checks={})
    assert 0.0 < worse < good <= 1.0
    assert good == pytest.approx(1 - 0.01 / 0.06, abs=1e-3)
    assert alignment.confidence(candidate(0.07), verdict="coarse", checks={}) == 0.0


def test_labels() -> None:
    assert (
        alignment.label(
            platform="phone-video",
            verdict="coarse",
            method="visual",
            uncertainty_m=0.0654,
            manual=False,
        )
        == "Phone video, aligned visually, ±0.065 m"
    )
    assert (
        alignment.label(
            platform="rtk-drone",
            verdict="survey",
            method="gnss-prior-ba",
            uncertainty_m=0.0243,
            manual=False,
        )
        == "RTK drone, surveyed, check targets ±0.024 m"
    )
    assert (
        alignment.label(
            platform="drone", verdict="unaligned", method="none", uncertainty_m=10.0, manual=True
        )
        == "Drone, not aligned: placed by hand, ±10 m"
    )
    assert alignment.label(
        platform="unknown", verdict="unaligned", method="none", uncertainty_m=10.0, manual=False
    ) == ("Capture, not aligned: placed by its GPS point, ±10 m")


def test_uncertainty_follows_the_evidence() -> None:
    targets = tuple(
        check_targets.Target(f"T{i}", "check", 0, 0, 0, "msl-unverified", 0.005, 0.01)
        for i in range(3)
    )
    checks = {
        "evaluated": 3,
        "rmseM": {"total": 0.002},
        "perTarget": [{"label": t.label, "errorM": [0, 0, 0]} for t in targets],
    }
    # Floored at the targets' own declared sigma: they cannot measure better than that.
    floor = math.sqrt(2 * 0.005**2 + 0.01**2)
    common: dict[str, Any] = {
        "targets": targets,
        "reference_m": 0.03,
        "rms_m": 0.01,
        "georef_m": 10.0,
    }
    assert alignment.uncertainty(verdict="survey", method="visual", checks=checks, **common) == (
        pytest.approx(floor)
    )
    assert alignment.uncertainty(verdict="coarse", method="visual", checks={}, **common) == (
        pytest.approx(math.hypot(0.03, 0.03))
    )
    assert alignment.uncertainty(verdict="unaligned", method="none", checks={}, **common) == 10.0


def test_reference_frame_composes_placement_and_reframe(tmp_path: Path) -> None:
    placement = {
        "scale": 1.5,
        "rotation": sfm.quat_to_matrix([0.8, 0.0, 0.0, 0.6]).tolist(),
        "translation": [1.0, 2.0, 0.5],
    }
    (tmp_path / "placement.json").write_text(json.dumps(placement))
    source = gps_origin = ORIGIN
    (tmp_path / "georef.json").write_text(
        json.dumps({"lat": source[0], "lon": source[1], "height": source[2]})
    )
    reference = alignment.Reference(
        epoch_id=None,
        job_id=None,
        observed_at=None,
        platform=None,
        verdict=None,
        uncertainty_m=0.03,
        frames=tmp_path,
        poses=tmp_path,
        placement=tmp_path / "placement.json",
        georef=tmp_path / "georef.json",
    )
    target = (gps_origin[0] + 0.004, gps_origin[1] - 0.003, gps_origin[2] + 7.0)
    points = np.random.default_rng(2).uniform(-20, 20, size=(15, 3))

    moved = alignment.reference_frame(reference, target).apply(points)

    rotation = np.asarray(placement["rotation"])
    in_source = 1.5 * points @ rotation.T + np.asarray(placement["translation"])
    expected = exif.reframe_enu(in_source, source, target)
    assert np.abs(moved - expected).max() < 1e-6


# --- the stage, no pycolmap -------------------------------------------------------------


def _georef_located() -> dict[str, Any]:
    return {
        "lat": ORIGIN[0],
        "lon": ORIGIN[1],
        "height": ORIGIN[2],
        "georefMethod": "exif-gps",
        "uncertaintyM": 10.0,
        "alignment": None,
        "frame": {
            "source": "camera-up",
            "scale": 1.0,
            "rotation": np.eye(3).tolist(),
            "translationM": None,
            "recentre": True,
            "headingDeg": 0.0,
            "headingSource": "none",
            "up": None,
        },
    }


def _georef_aligned(similarity: sfm.Similarity, names: list[str]) -> dict[str, Any]:
    return {
        "lat": ORIGIN[0],
        "lon": ORIGIN[1],
        "height": ORIGIN[2],
        "georefMethod": "exif-gps",
        "uncertaintyM": 5.0,
        "alignment": {
            "images": len(names),
            "inliers": len(names),
            "maxErrorM": 5.0,
            "residualM": {"perImage": dict.fromkeys(names, 0.01)},
        },
        "frame": {
            "source": "exif-gps-similarity",
            "scale": similarity.scale,
            "rotation": similarity.rotation.tolist(),
            "translationM": similarity.translation.tolist(),
            "recentre": False,
            "headingDeg": None,
            "headingSource": "exif-gps",
            "up": None,
        },
    }


def _run(
    root: Path,
    georef: dict[str, Any],
    *,
    params: dict[str, object] | None = None,
    reference: dict[str, Any] | None = None,
    poses_json: dict[str, Any] | None = None,
) -> tuple[Workdir, dict[str, Any]]:
    workdir = Workdir.create(root)
    poses = workdir.input_path("poses")
    synthetic_scene.write_model(poses, synthetic_scene.ring(COUNT, radius=6.0, height=2.0))
    _write_points(poses)
    (poses / "poses.json").write_text(json.dumps(poses_json or {}))
    frames = workdir.input_path("frames")
    frames.mkdir(parents=True)
    for i in range(COUNT):
        (frames / f"frame_{i:04d}.jpg").write_bytes(b"x")
    workdir.input_path("georef.json").write_text(json.dumps(georef))
    context = workdir.input_path("site_context")
    context.mkdir(parents=True)
    document = alignment.empty_site_context("site-1")
    if reference is not None:
        document["reference"] = reference
        for name in ("frames", "poses"):
            (context / "reference" / name).mkdir(parents=True, exist_ok=True)
        for name in ("placement.json", "georef.json"):
            (context / "reference" / name).write_text(
                json.dumps(
                    {
                        "scale": 1.0,
                        "rotation": np.eye(3).tolist(),
                        "translation": [0, 0, 0],
                        "lat": ORIGIN[0],
                        "lon": ORIGIN[1],
                        "height": ORIGIN[2],
                    }
                )
            )
    (context / alignment.SITE_CONTEXT_FILE).write_text(json.dumps(document))
    execute(
        make_recipe(
            [{"id": "align", "impl": "align_epoch", "params": params or {}}],
            inputs=["frames", "poses", "georef.json", "site_context"],
        ),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )
    out = json.loads(workdir.artifact_path("align", "epoch_alignment.json").read_text())
    assert isinstance(out, dict)
    return workdir, out


REFERENCE = {
    "epochId": "e1",
    "jobId": "j1",
    "observedAt": "2026-10-01T10:00:00Z",
    "platform": "rtk-drone",
    "verdict": "survey",
    "uncertaintyM": 0.024,
    "frames": "reference/frames",
    "poses": "reference/poses",
    "placement": "reference/placement.json",
    "georef": "reference/georef.json",
}


def test_no_reference_and_a_located_georef_is_unaligned_and_targets_are_not_evaluated(
    tmp_path: Path,
) -> None:
    _, document = _run(tmp_path / "w", _georef_located(), params={"platform": "phone-video"})

    assert document["method"] == "none" and document["verdict"] == "unaligned"
    assert document["confidence"] == 0.0 and document["uncertaintyM"] == 10.0
    assert document["frameSource"] == "georef"
    assert document["frame"] == _georef_located()["frame"]
    assert document["checkTargets"]["status"] == "not-evaluated"
    assert document["checkTargets"]["reason"] == alignment.NOT_EVALUATED
    assert document["label"] == "Phone video, not aligned: placed by its GPS point, ±10 m"
    assert all(gate["enabled"] is False for gate in document["gates"])


def test_no_reference_and_a_gnss_similarity_is_coarse(tmp_path: Path) -> None:
    names = [f"frame_{i:04d}.jpg" for i in range(COUNT)]
    _, document = _run(tmp_path / "w", _georef_aligned(TRUE, names))

    assert document["method"] == "gnss-similarity" and document["verdict"] == "coarse"
    assert document["platform"] == "unknown" and document["platformSource"] == "default"
    assert document["frameSource"] == "georef" and document["inliers"] == COUNT
    assert document["label"] == "Capture, aligned to its GNSS, ±5 m"
    assert document["checkTargets"]["status"] == "none"
    assert 0.0 < document["confidence"] <= 1.0


def test_fail_unaligned_raises_a_quality_gate_error(tmp_path: Path) -> None:
    with pytest.raises(QualityGateError, match="quality gate unaligned"):
        _run(tmp_path / "w", _georef_located(), params={"fail_unaligned": True})


def test_a_reference_without_pycolmap_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(global_sfm.PYTHON_ENV, str(tmp_path / "no-such-python"))
    with pytest.raises(StageFailedError, match="not falling back to GNSS"):
        _run(tmp_path / "w", _georef_located(), reference=REFERENCE)


def _fake_register(registered: dict[str, Any]) -> Any:
    def fake(*_: object, **__: object) -> dict[str, Any]:
        return registered

    return fake


def test_a_visual_alignment_to_a_survey_reference_is_the_frame_place_uses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reference here is an identity placement at the same origin, so the registered
    centres are already this epoch's east/north/up, and the fit must be `TRUE`."""
    _, model = _ring_model(tmp_path / "m")
    names = [image.name for image in model.images]
    monkeypatch.setattr(alignment, "_register", _fake_register(_registered(model, TRUE, names)))
    _, document = _run(
        tmp_path / "w", _georef_located(), reference=REFERENCE, params={"platform": "phone-video"}
    )

    assert document["method"] == "visual" and document["verdict"] == "coarse"
    assert document["frameSource"] == "epoch-alignment"
    assert document["frame"]["source"] == "visual-epoch" and document["frame"]["recentre"] is False
    assert document["frame"]["scale"] == pytest.approx(2.0)
    assert document["referenceEpochId"] == "e1"
    assert document["label"].startswith("Phone video, aligned visually, ±")
    assert document["gnssCheck"]["kind"] == "located-point"
    assert document["rotationResidualDeg"]["max"] < 1e-6


def test_an_rtk_epoch_keeps_its_own_frame_over_a_coarser_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never trade an RTK epoch's centimetres for a phone reference's metres: it keeps its
    GNSS-prior frame and the visual fit is recorded as a check (the review's fix 11)."""
    _, model = _ring_model(tmp_path / "m")
    names = [image.name for image in model.images]
    monkeypatch.setattr(alignment, "_register", _fake_register(_registered(model, TRUE, names)))
    priors = {
        "priors": {"used": True, "sigmaM": {"horizontalMedian": 0.02, "verticalMedian": 0.02}}
    }
    coarse = {**REFERENCE, "platform": "phone-video", "verdict": "coarse", "uncertaintyM": 0.5}
    _, document = _run(
        tmp_path / "w",
        _georef_aligned(IDENTITY, names),
        reference=coarse,
        params={"platform": "rtk-drone"},
        poses_json=priors,
    )

    assert document["method"] == "gnss-prior-ba" and document["frameSource"] == "georef"
    assert document["visualCheck"]["inliers"] == COUNT
    assert any("kept its own RTK frame" in r for r in document["reasons"])
    # And with a survey reference the same epoch does align visually.
    _, surveyed = _run(
        tmp_path / "v",
        _georef_aligned(IDENTITY, names),
        reference=REFERENCE,
        params={"platform": "rtk-drone"},
        poses_json=priors,
    )
    assert surveyed["method"] == "visual" and surveyed["frameSource"] == "epoch-alignment"


def test_an_unaccepted_visual_alignment_falls_back_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, model = _ring_model(tmp_path / "m")
    names = [image.name for image in model.images]
    monkeypatch.setattr(alignment, "_register", _fake_register(_registered(model, TRUE, names[:2])))
    _, document = _run(tmp_path / "w", _georef_located(), reference=REFERENCE)

    assert document["method"] == "none" and document["verdict"] == "unaligned"
    assert document["frameSource"] == "georef" and document["confidence"] == 0.0
    assert document["registration"]["registered"] == 2
    assert any("fell back" in r for r in document["reasons"])
