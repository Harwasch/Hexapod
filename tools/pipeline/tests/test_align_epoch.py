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

import gps_frames
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
#: What `alignment.rtk_evidence` says of frames that carry an RTK fixed solution.
RTK = "10/10 frames carry an RTK fixed solution (RtkFlag 50)"


# --- the decision, pure -----------------------------------------------------------------


def test_rtk_with_passing_targets_is_survey() -> None:
    checks = {"checks": 3, "verticalChecks": 3, "withinTolerance": True}
    assert alignment.decide(
        aligned=True,
        tier="survey",
        checks=checks,
        platform="rtk-drone",
        rtk_evidence=RTK,
        uncertainty_m=0.03,
    ) == (
        "survey",
        [],
    )


def test_rtk_without_targets_is_coarse_and_says_why() -> None:
    verdict, reasons = alignment.decide(
        aligned=True, tier="survey", checks={"checks": 0}, platform="rtk-drone", rtk_evidence=RTK
    )
    assert verdict == "coarse"
    assert reasons == ["accuracy unverified: 0 check target(s) evaluated, fewer than 3"]
    failing = {
        "checks": 8,
        "verticalChecks": 8,
        "withinTolerance": False,
        "rmseM": {"horizontal": 0.4, "vertical": 0.1},
        "tolerance": {"horizontalM": 0.12, "verticalM": 0.11},
    }
    verdict, reasons = alignment.decide(
        aligned=True, tier="survey", checks=failing, platform="rtk-drone", rtk_evidence=RTK
    )
    assert verdict == "coarse" and "out of tolerance" in reasons[0]


def test_survey_needs_three_check_targets_scored_vertically() -> None:
    """R2 review (#51, high): `decide` never asked whether any height was checked, so
    targets all surveyed on another height datum (vertical unscored) could still give
    `survey`. A survey verdict needs three check targets scored vertically too."""
    for vertical in (0, 2):
        checks = {"checks": 3, "verticalChecks": vertical, "withinTolerance": True}
        verdict, reasons = alignment.decide(
            aligned=True, tier="survey", checks=checks, platform="rtk-drone", rtk_evidence=RTK
        )
        assert verdict == "coarse", vertical
        assert "accuracy unverified vertically" in reasons[0]
        assert f"{vertical} check target(s)" in reasons[0]


def test_survey_has_an_absolute_ceiling_half_an_l1_carrier_cycle() -> None:
    """R2 review (#51, high): the tolerance grows with the anchor's sigma, so a 10 m
    reference made a 42 m tolerance and a 5 m error was still 'surveyed'. Survey now also
    needs the claim under `SURVEY_CEILING_M`, half the GPS L1 carrier wavelength (ADR
    0015: a claim that wide cannot tell an RTK fixed solution from one a cycle wrong)."""
    assert abs(alignment.SURVEY_CEILING_M - 299_792_458 / 1_575.42e6 / 2) < 1e-12
    assert 0.095 < alignment.SURVEY_CEILING_M < 0.0952
    perfect = {"checks": 8, "verticalChecks": 8, "withinTolerance": True}
    common: dict[str, Any] = {
        "aligned": True,
        "tier": "survey",
        "checks": perfect,
        "platform": "rtk-drone",
        "rtk_evidence": RTK,
    }
    assert alignment.decide(**common, uncertainty_m=0.05)[0] == "survey"
    for claim in (0.0952, 10.0, None):
        verdict, reasons = alignment.decide(**common, uncertainty_m=claim)
        assert verdict == "coarse", claim
        assert "survey ceiling" in reasons[-1], claim


def test_an_epoch_aligned_to_a_reference_that_is_not_survey_is_not_survey() -> None:
    perfect = {"checks": 8, "verticalChecks": 8, "withinTolerance": True}
    for reference in ("coarse", "unaligned", None):
        verdict, reasons = alignment.decide(
            aligned=True, tier="survey", checks=perfect, platform="rtk-drone",
            rtk_evidence=RTK, uncertainty_m=0.03, method="visual",
            reference_verdict=reference,
        )  # fmt: skip
        assert verdict == "coarse", reference
        assert "reference epoch that is not surveyed" in reasons[0], reference
    assert alignment.decide(
        aligned=True, tier="survey", checks=perfect, platform="rtk-drone", rtk_evidence=RTK,
        uncertainty_m=0.03, method="visual", reference_verdict="survey",
    )[0] == "survey"  # fmt: skip


def test_the_rtk_drone_platform_alone_is_never_survey() -> None:
    """Coordinator (R2 review of #51): a platform string -- a param, or normalize's
    override -- must not upgrade a capture to survey; only evidence in the data can."""
    perfect = {"checks": 8, "verticalChecks": 8, "withinTolerance": True}
    verdict, reasons = alignment.decide(
        aligned=True, tier="survey", checks=perfect, platform="rtk-drone", rtk_evidence=None
    )
    assert verdict == "coarse"
    assert "no RTK fixed solution" in reasons[0]


def test_rtk_evidence_is_a_fixed_solution_on_at_least_half_the_frames() -> None:
    def meta(flags: list[int | None]) -> dict[str, Any]:
        return {
            "frames": [
                {"name": f"f{i}.jpg", "dji": None if f is None else {"rtk": {"flag": f}}}
                for i, f in enumerate(flags)
            ]
        }

    assert alignment.rtk_evidence(None) is None
    assert alignment.rtk_evidence({"platform": {"value": "rtk-drone", "source": "param"}}) is None
    assert alignment.rtk_evidence(meta([50, 50, 16, None])) == (
        "2/4 frames carry an RTK fixed solution (RtkFlag 50)"
    )
    assert alignment.rtk_evidence(meta([50, 16, 16, None])) is None  # float is not fixed
    assert alignment.rtk_evidence(meta([])) is None


def test_a_phone_epoch_is_never_survey_even_with_perfect_numbers() -> None:
    assert [p for p, tier in alignment.TIER_OF.items() if tier == "survey"] == ["rtk-drone"]
    perfect = {"checks": 8, "verticalChecks": 8, "withinTolerance": True}
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
    common: dict[str, Any] = {"targets": targets, "reference_m": 0.03, "rms_m": 0.01}
    visual_floor = math.hypot(0.03, 3 * 0.01)
    # Visually aligned, no targets: the reference and three times the fit's RMS.
    assert alignment.uncertainty(
        verdict="coarse", method="visual", checks={}, georef_m=10.0, **common
    ) == pytest.approx(visual_floor)
    # GNSS with targets: their RMSE, floored at their own sigma and the anchor's.
    target_floor = math.sqrt(2 * 0.005**2 + 0.01**2)
    assert alignment.uncertainty(
        verdict="survey", method="gnss-prior-ba", checks=checks, georef_m=5.0, **common
    ) == pytest.approx(target_floor)
    assert alignment.uncertainty(
        verdict="survey", method="gnss-prior-ba", checks=checks, georef_m=5.0, anchor_m=0.03,
        **common,
    ) == pytest.approx(0.03)  # fmt: skip
    # Unaligned: the georeference's own figure, whatever else was measured.
    assert (
        alignment.uncertainty(
            verdict="unaligned", method="gnss-similarity", checks=checks, georef_m=5.0, **common
        )
        == 5.0
    )


def test_check_targets_never_lower_a_visual_epochs_claim() -> None:
    """Review finding 2: with targets, a visual epoch's claim dropped its own fit's error
    and kept only the reference's, so adding targets lowered it -- one run claimed
    ±0.0247 m with the cameras 0.028 m from the truth. Targets may only raise it."""
    targets = tuple(
        check_targets.Target(f"T{i}", "check", 0, 0, 0, "msl-unverified", 0.005, 0.01)
        for i in range(8)
    )
    checks = {
        "evaluated": 8,
        "rmseM": {"total": 0.012},
        "perTarget": [{"label": t.label, "errorM": [0, 0, 0]} for t in targets],
    }
    without = alignment.uncertainty(
        verdict="coarse", method="visual", checks={}, targets=targets, reference_m=0.024,
        rms_m=0.012, georef_m=5.0, anchor_m=0.024,
    )  # fmt: skip
    with_targets = alignment.uncertainty(
        verdict="survey", method="visual", checks=checks, targets=targets, reference_m=0.024,
        rms_m=0.012, georef_m=5.0, anchor_m=0.024,
    )  # fmt: skip
    assert with_targets >= without == pytest.approx(math.hypot(0.024, 3 * 0.012))


def test_an_unconstrained_vertical_anchor_is_not_scored_and_cannot_be_survey() -> None:
    """Review finding 3: priors whose vertical was unconstrained (a datum mismatch: sigma
    1 km) made a 3 km vertical tolerance, and a target 30 m out vertically passed as
    "RTK drone, surveyed". Vertical is now not scored and survey not allowed."""
    names = [f"frame_{i:04d}.jpg" for i in range(COUNT)]
    poses = {
        "priors": {
            "used": True,
            "vertical": "unconstrained: fix altitude is msl-unverified, the frame is "
            "ellipsoid-wgs84",
            "sigmaM": {"horizontalMedian": 0.02, "verticalMedian": 1000.0},
        }
    }
    candidate = alignment.gnss_candidate(_georef_aligned(IDENTITY, names), poses)
    assert candidate.vertical_free is not None and "unconstrained" in candidate.vertical_free

    ring = synthetic_scene.ring(8, radius=6.0, height=2.0)
    model = sfm.Model(
        cameras=(sfm.Camera(1, "PINHOLE", 1600, 1200, (1300.0, 1300.0, 800.0, 600.0),
                            ("fx", "fy", "cx", "cy")),),
        images=tuple(
            sfm.Image(i + 1, f"frame_{i:04d}.jpg", synthetic_scene._quat(r),
                      tuple(float(v) for v in -r @ c), 1, 0, 0)  # type: ignore[arg-type]
            for i, (r, c) in enumerate(ring)
        ),
        points3d=0,
        mean_track_length=0.0,
    )  # fmt: skip
    survey = gps_frames.enu_to_geodetic(1.0, 1.0, 0.0, *ORIGIN)
    targets = [
        check_targets.Target(f"T{i}", "check", *survey, "ellipsoid-wgs84", 0.005, 0.01)
        for i in range(3)
    ]
    found = {
        t.label: check_targets.Triangulated(
            t.label, (1.0, 1.0, 30.0), 8, tuple(i.name for i in model.images), 40.0, 0.5, "ok"
        )
        for t in targets
    }
    block = check_targets.evaluate(
        found, targets, to_frame=IDENTITY, origin=ORIGIN, frame_datum="ellipsoid-wgs84",
        model=model, anchor_h=candidate.anchor_h, anchor_v=candidate.anchor_v,
        vertical_anchor_free=candidate.vertical_free,
    )  # fmt: skip

    assert block["verticalEvaluated"] is False
    assert block["tolerance"]["verticalM"] is None  # type: ignore[index]
    assert all(entry["verticalM"] is None for entry in block["perTarget"])  # type: ignore[attr-defined]
    verdict, reasons = alignment.decide(
        aligned=True, tier="survey", checks=block, platform="rtk-drone", rtk_evidence=RTK
    )
    assert verdict == "coarse"
    assert "accuracy unverified vertically" in reasons[0]
    claim = alignment.uncertainty(
        verdict=verdict, method="gnss-prior-ba", checks=block, targets=tuple(targets),
        reference_m=None, rms_m=0.01, georef_m=5.0, anchor_m=0.02, vertical_free=True,
    )  # fmt: skip
    assert claim >= 5.0


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
    frames_meta: dict[str, Any] | None = None,
) -> tuple[Workdir, dict[str, Any]]:
    workdir = Workdir.create(root)
    inputs = ["frames", "poses", "georef.json", "site_context"]
    if frames_meta is not None:
        workdir.input_path("frames_meta.json").write_text(json.dumps(frames_meta))
        inputs.append("frames_meta.json")
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
            inputs=inputs,
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
    registered = {**_registered(model, TRUE, names), "referenceFramesUnmatched": ["f.jpg"]}
    monkeypatch.setattr(alignment, "_register", _fake_register(registered))
    _, document = _run(
        tmp_path / "w", _georef_located(), reference=REFERENCE, params={"platform": "phone-video"}
    )
    # What the registration left out of the fixed model is said, not hidden.
    assert document["registration"]["referenceFramesUnmatched"] == ["f.jpg"]

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


# --- place and catalog read the alignment ------------------------------------------------------


def _gaussians(points: np.ndarray) -> dict[str, np.ndarray]:
    count = points.shape[0]
    columns = {axis: points[:, i] for i, axis in enumerate("xyz")}
    columns.update({f"f_dc_{i}": np.zeros(count) for i in range(3)})
    columns.update({"opacity": np.full(count, 2.0)})
    columns.update({f"scale_{i}": np.full(count, -4.0) for i in range(3)})
    columns.update({"rot_0": np.ones(count), "rot_1": np.zeros(count)})
    columns.update({"rot_2": np.zeros(count), "rot_3": np.zeros(count)})
    return {name: np.ascontiguousarray(v, dtype=np.float32) for name, v in columns.items()}


POINTS = np.random.default_rng(5).uniform(-2.0, 2.0, size=(50, 3))
FRAME_A = sfm.Similarity(1.5, sfm.quat_to_matrix([0.9, 0.0, 0.0, 0.4359]), np.array([1.0, 2, 3]))
FRAME_B = sfm.Similarity(0.7, sfm.quat_to_matrix([0.8, 0.6, 0.0, 0.0]), np.array([-4.0, 0, 1]))


def _frame(similarity: sfm.Similarity, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "scale": similarity.scale,
        "rotation": similarity.rotation.tolist(),
        "translationM": similarity.translation.tolist(),
        "recentre": False,
        "headingDeg": None,
        "headingSource": "x",
        "up": None,
    }


def _place(root: Path, georef: dict[str, Any], aligned: dict[str, Any] | None) -> np.ndarray:
    import gaussians

    workdir = Workdir.create(root)
    trained = workdir.input_path("trained.ply")
    trained.parent.mkdir(parents=True, exist_ok=True)
    gaussians.write_ply(trained, _gaussians(POINTS))
    workdir.input_path("georef.json").write_text(json.dumps(georef))
    inputs = ["trained.ply", "georef.json"]
    if aligned is not None:
        workdir.input_path("epoch_alignment.json").write_text(json.dumps(aligned))
        inputs.append("epoch_alignment.json")
    execute(
        make_recipe([{"id": "place", "impl": "place_splat"}], inputs=inputs),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )
    placed = gaussians.read_splat(workdir.artifact_path("place", "canonical.ply"))
    return np.asarray(placed.xyz, dtype=np.float64)


def _georef_with(frame: dict[str, Any], offset: list[float] | None = None) -> dict[str, Any]:
    georef: dict[str, Any] = {
        "lat": ORIGIN[0],
        "lon": ORIGIN[1],
        "height": ORIGIN[2],
        "georefMethod": "exif-gps",
        "scaleSource": "exif-gps",
        "uncertaintyM": 5.0,
        "frame": frame,
    }
    if offset is not None:
        georef["siteFrame"] = {
            "siteId": "s",
            "lat": ORIGIN[0],
            "lon": ORIGIN[1],
            "height": ORIGIN[2],
            "offsetM": offset,
            "offsetSource": "video",
            "located": None,
        }
    return georef


def test_place_prefers_the_epoch_alignment_frame(tmp_path: Path) -> None:
    georef = _georef_with(_frame(FRAME_A, "exif-gps-similarity"))
    aligned = {
        "frameSource": "epoch-alignment",
        "method": "visual",
        "verdict": "coarse",
        "frame": _frame(FRAME_B, "visual-epoch"),
    }

    with_alignment = _place(tmp_path / "b", georef, aligned)
    without = _place(tmp_path / "a", georef, None)

    # float32 positions in the PLY: equal to its precision.
    assert np.abs(with_alignment - FRAME_B.apply(POINTS)).max() < 1e-5
    assert np.abs(without - FRAME_A.apply(POINTS)).max() < 1e-5


def test_place_ignores_an_alignment_that_kept_the_georef_frame(tmp_path: Path) -> None:
    georef = _georef_with(_frame(FRAME_A, "exif-gps-similarity"))
    kept = {"frameSource": "georef", "method": "gnss-prior-ba", "frame": _frame(FRAME_B, "x")}

    assert np.abs(_place(tmp_path / "k", georef, kept) - FRAME_A.apply(POINTS)).max() < 1e-5


def test_place_never_adds_the_site_offset_to_an_epoch_aligned_frame(tmp_path: Path) -> None:
    """A located phone epoch in a site frame (the review's fix 2): its georef carries the
    GPS point's offset from the frame origin, which `place` adds to a recentred footprint.
    The epoch alignment already put the capture where it is; adding the offset as well
    would move it by the GPS error a second time."""
    offset = [8.0, 5.0, 0.0]
    located = {**_frame(sfm.Similarity(1.0, np.eye(3), np.zeros(3)), "camera-up"), "recentre": True}
    aligned = {
        "frameSource": "epoch-alignment",
        "method": "visual",
        "verdict": "coarse",
        "frame": _frame(FRAME_B, "visual-epoch"),
    }

    placed = _place(tmp_path / "o", _georef_with(located, offset), aligned)
    assert np.abs(placed - FRAME_B.apply(POINTS)).max() < 1e-5
    # What it would have been had the offset been applied to the same frame.
    doubled = _place(tmp_path / "d", _georef_with(_frame(FRAME_B, "x"), offset), None)
    assert np.abs(doubled - (FRAME_B.apply(POINTS) + np.asarray(offset))).max() < 1e-5


def test_catalog_carries_the_alignment_without_per_image_residuals(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "c")
    splat = workdir.input_path("splat")
    splat.mkdir(parents=True)
    (splat / "tileset.json").write_text("{}")
    workdir.input_path("georef.json").write_text(json.dumps(_georef_with(_frame(FRAME_A, "x"))))
    document = {
        "version": 1,
        "verdict": "coarse",
        "label": "Phone video, aligned visually, ±0.065 m",
        "uncertaintyM": 0.065,
        "residualM": {
            "rms": 0.01,
            "median": 0.01,
            "p90": 0.02,
            "max": 0.03,
            "perImage": {"frame_0000.jpg": 0.01},
        },
    }
    workdir.input_path("epoch_alignment.json").write_text(json.dumps(document))
    execute(
        make_recipe(
            [{"id": "register", "impl": "catalog"}],
            inputs=["splat", "georef.json", "epoch_alignment.json"],
        ),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )
    registration = json.loads(workdir.artifact_path("register", "registration.json").read_text())

    carried = registration["alignment"]
    assert carried["label"] == document["label"] and carried["verdict"] == "coarse"
    assert carried["uncertaintyM"] == 0.065
    assert carried["residualM"] == {"rms": 0.01, "median": 0.01, "p90": 0.02, "max": 0.03}


def test_orientations_that_disagree_with_the_fit_are_not_an_alignment(tmp_path: Path) -> None:
    """Centres alone leave a rotation about their own line free; the orientations must
    agree too. Every registered camera turned 5 deg (far past 3*sqrt(6) pixels' angle at
    the ring's 1300 px focal, ~0.32 deg) while its centre stays put: not aligned."""
    poses, model = _ring_model(tmp_path / "m")
    names = [image.name for image in model.images]
    registered = _registered(model, TRUE, names)
    turn = sfm.quat_to_matrix([math.cos(math.radians(2.5)), math.sin(math.radians(2.5)), 0, 0])
    for name in names:
        rotation = np.asarray(registered["registeredNew"][name]["rotation"])
        registered["registeredNew"][name]["rotation"] = (turn @ rotation).tolist()
    candidate = alignment.visual_candidate(
        registered, model_dir=poses, model=model, to_target=IDENTITY,
        reference=_reference(tmp_path), min_registered_fraction=0.5, seed=0,
    )  # fmt: skip

    assert not candidate.aligned
    assert any("orientations disagree" in reason for reason in candidate.reasons)


def test_a_gnss_epoch_that_fails_visual_alignment_is_unaligned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding 5 (D4): an epoch that fails alignment to its site's reference is
    unaligned and labelled so, even with a GNSS similarity of its own -- not "coarse,
    aligned to its GNSS". It keeps that placement and the georeference's uncertainty."""
    _, model = _ring_model(tmp_path / "m")
    names = [image.name for image in model.images]
    monkeypatch.setattr(alignment, "_register", _fake_register(_registered(model, TRUE, names[:2])))
    _, document = _run(
        tmp_path / "w", _georef_aligned(IDENTITY, names), reference=REFERENCE,
        params={"platform": "drone"},
    )  # fmt: skip

    assert document["verdict"] == "unaligned" and document["method"] == "gnss-similarity"
    assert document["frameSource"] == "georef" and document["confidence"] == 0.0
    assert document["label"] == "Drone, not aligned: placed by its GPS, ±5 m"
    assert document["uncertaintyM"] == 5.0
    # No reference at all is not a failure: the same epoch is then coarse.
    _, alone = _run(tmp_path / "a", _georef_aligned(IDENTITY, names), params={"platform": "drone"})
    assert alone["verdict"] == "coarse" and alone["label"] == "Drone, aligned to its GNSS, ±5 m"


def test_a_registration_that_crashes_is_an_unaligned_epoch_not_a_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding 1's backstop: whatever makes `colmap4 register` die, the epoch is
    recorded unaligned with the reason, and the run goes on."""
    import subprocess

    def crash(*_: object, **__: object) -> dict[str, Any]:
        raise subprocess.CalledProcessError(1, ["colmap4.py", "register"])

    names = [f"frame_{i:04d}.jpg" for i in range(COUNT)]
    monkeypatch.setattr(alignment, "_register", crash)
    _, document = _run(
        tmp_path / "w", _georef_aligned(IDENTITY, names), reference=REFERENCE,
        params={"platform": "drone"},
    )  # fmt: skip

    assert document["verdict"] == "unaligned" and document["frameSource"] == "georef"
    assert any("registration into the reference failed" in r for r in document["reasons"])
    assert document["label"].startswith("Drone, not aligned: placed by its GPS")


def _perfect_checks(*_: object, **__: object) -> dict[str, object]:
    """Eight check targets scored dead on, horizontally and vertically: the targets are
    not what is under test here."""
    return {
        "status": "evaluated",
        "evaluated": 8,
        "checks": 8,
        "verticalChecks": 8,
        "verticalEvaluated": True,
        "verticalAnchorFree": None,
        "perTarget": [],
        "rmseM": {"horizontal": 0.001, "vertical": 0.001, "total": 0.0014},
        "tolerance": {"horizontalM": 0.06, "verticalM": 0.05},
        "withinTolerance": True,
    }


def test_a_platform_param_never_makes_survey_without_rtk_evidence_in_the_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coordinator (R2 review of #51): every path by which the platform string alone gave
    `survey` -- the stage's `platform` param, normalize's own `platform` override in
    `frames_meta`, an iPhone's metadata with the param saying RTK -- is closed. Only an
    RTK fixed solution in the frames' own metadata (plus the targets) is survey."""
    monkeypatch.setattr(alignment.check_targets, "evaluate", _perfect_checks)
    names = [f"frame_{i:04d}.jpg" for i in range(COUNT)]
    priors = {
        "priors": {"used": True, "sigmaM": {"horizontalMedian": 0.01, "verticalMedian": 0.01}}
    }
    phone = {
        "platform": {"value": "phone-video", "source": "phone-quicktime", "evidence": "Apple"},
        "frames": [{"name": n, "original": "clip.mov", "dji": None} for n in names],
    }
    claimed = {
        "platform": {"value": "rtk-drone", "source": "param", "evidence": "asked for"},
        "frames": [{"name": n, "dji": None} for n in names],
    }
    fixed = {
        "platform": {"value": "rtk-drone", "source": "dji-rtk-fixed", "evidence": "10/10"},
        "frames": [{"name": n, "dji": {"rtk": {"flag": 50}}} for n in names],
    }
    cases: list[tuple[str, dict[str, object], dict[str, Any] | None, str]] = [
        ("param, no metadata", {"platform": "rtk-drone"}, None, "coarse"),
        ("param over an iPhone", {"platform": "rtk-drone"}, phone, "coarse"),
        ("normalize's override", {}, claimed, "coarse"),
        ("RTK fixed frames", {}, fixed, "survey"),
    ]
    for index, (case, params, meta, expected) in enumerate(cases):
        _, document = _run(
            tmp_path / f"w{index}",
            _georef_aligned(IDENTITY, names),
            params=params,
            poses_json=priors,
            frames_meta=meta,
        )
        assert document["verdict"] == expected, case
        assert document["platform"] == "rtk-drone", case
        if expected == "coarse":
            assert any("no RTK fixed solution" in r for r in document["reasons"]), case
            assert document["rtkEvidence"] is None, case
        else:
            assert document["rtkEvidence"].startswith("10/10 frames"), case


def test_an_rtk_epoch_without_priors_is_not_surveyed_through_a_coarse_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2 review (#51, high): keeping its own RTK frame needed `gnss-prior-ba`, so an RTK
    epoch with priors off (COLMAP 3.9, `pose_priors: off`) adopted a ±10 m phone
    reference, whose sigma widened the check targets' tolerance to ~42 m: 'surveyed'.
    Now a survey-tier epoch keeps whichever frame is anchored tighter, and neither path
    reaches survey: not over the ceiling, not through a reference that is not survey."""
    monkeypatch.setattr(alignment.check_targets, "evaluate", _perfect_checks)
    _, model = _ring_model(tmp_path / "m")
    names = [image.name for image in model.images]
    monkeypatch.setattr(alignment, "_register", _fake_register(_registered(model, TRUE, names)))
    fixed = {"frames": [{"name": n, "dji": {"rtk": {"flag": 50}}} for n in names]}
    for index, (uncertainty_m, method) in enumerate(((10.0, "gnss-similarity"), (0.5, "visual"))):
        coarse = {**REFERENCE, "platform": "phone-video", "verdict": "coarse",
                  "uncertaintyM": uncertainty_m}  # fmt: skip
        _, document = _run(
            tmp_path / f"w{index}",
            _georef_aligned(IDENTITY, names),
            reference=coarse,
            params={"platform": "rtk-drone"},
            frames_meta=fixed,
        )
        assert document["rtkEvidence"] is not None
        assert document["method"] == method, uncertainty_m
        assert document["verdict"] == "coarse", uncertainty_m
        assert "surveyed" not in document["label"], uncertainty_m


def test_a_gnss_similarity_with_fewer_than_three_inliers_is_not_aligned() -> None:
    """R2 review (#51, medium): `gnss_candidate` marked any recorded similarity aligned,
    whatever its inlier count (`model_aligner`'s path records it after the fact)."""
    names = [f"frame_{i:04d}.jpg" for i in range(COUNT)]
    georef = _georef_aligned(IDENTITY, names)
    georef["alignment"]["inliers"] = 2
    candidate = alignment.gnss_candidate(georef, {})
    assert candidate.aligned is False
    assert "2 of 10 fixes agree" in candidate.reasons[0]
    georef["alignment"]["inliers"] = 3
    assert alignment.gnss_candidate(georef, {}).aligned is True
