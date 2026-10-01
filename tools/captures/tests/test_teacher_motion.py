"""Test L1 (docs/LIVING_ENGINE.md): Teacher A recovers the motion it was shown.

The committed synthetic tree, posed by its own allometric sidecar -- every limb a damped
oscillator at the frequency the sidecar gives it -- is rendered from three cameras around
it, four clips of each (`OscillatorClips`, the CPU stand-in for a video model). The teacher
labels each limb's pixels in its own render of frame 0, tracks them, regresses out what
carries them and fits each limb's resonance. It never sees the sidecar's frequencies.

What it must recover: the trunk within 5 %; at least ten limbs fitted; 80 % of those within
10 %, the median within 5 %; and a sidecar that is the prior with exactly the fitted limbs'
frequencies changed, tagged `fitted-generated`. At 12 frames a second and 480 x 360 this
takes about a minute and a half on a CI runner's CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import teacher_motion as tm
from splat_render import Camera, Splats, load_ply, render

SOURCE = Path(__file__).resolve().parents[3] / "data" / "tiles" / "synthetic-tree" / "source"


@pytest.fixture(scope="module")
def tree() -> tuple[Splats, dict, dict, tm.Rig]:
    rig_doc = json.loads((SOURCE / "rig.json").read_text())
    prior = json.loads((SOURCE / "motion.json").read_text())
    splats = load_ply(SOURCE / "splat.ply")
    return splats, rig_doc, prior, tm.Rig(rig_doc, prior)


@pytest.fixture(scope="module")
def lesson(tree: tuple[Splats, dict, dict, tm.Rig]) -> tm.Lesson:
    splats, rig_doc, prior, rig = tree
    cameras = tm.cameras_around(rig, count=3, width=480, height=360)
    source = tm.OscillatorClips(splats, rig, seconds=10.0, fps=12.0)
    return tm.teach(splats, rig_doc, prior, source, cameras=cameras, seeds=(1, 2, 3, 4))


def test_the_trunk_is_recovered(lesson: tm.Lesson, tree: tuple) -> None:
    rig = tree[3]
    trunk = next(f for f in lesson.fits if f.limb == rig.tree)
    assert trunk.frequency_hz is not None
    assert abs(trunk.frequency_hz - trunk.prior_hz) / trunk.prior_hz < 0.05


def test_the_limbs_it_fits_are_the_limbs_it_was_shown(lesson: tm.Lesson) -> None:
    fitted = [f for f in lesson.fits if f.frequency_hz is not None]
    errors = np.array([abs(f.frequency_hz - f.prior_hz) / f.prior_hz for f in fitted])  # type: ignore[operator]
    assert len(fitted) >= 10, lesson.report
    assert np.mean(errors < 0.10) >= 0.8, [f.to_json() for f in fitted]
    assert np.median(errors) < 0.05


def test_the_sidecar_changes_only_what_was_fitted(lesson: tm.Lesson, tree: tuple) -> None:
    _, _, prior, rig = tree
    sidecar = lesson.sidecar
    assert sidecar["motionEvidence"] == "fitted-generated"
    assert sidecar["provenance"]["fittedGenerated"]["status"] == "fitted"
    fitted = {f.limb: f.frequency_hz for f in lesson.fits if f.frequency_hz is not None}
    for i, (before, after) in enumerate(
        zip(prior["nodes"]["frequencyHz"], sidecar["nodes"]["frequencyHz"], strict=True)
    ):
        limb = int(rig.branch[i])
        if i > 0 and limb in fitted:
            assert after == round(fitted[limb], 4)
        else:
            assert after == before
    assert sidecar["nodes"]["damping"] == prior["nodes"]["damping"]
    for key in ("format", "version", "rigChecksum", "nodeCount", "wind", "seasons"):
        assert sidecar[key] == prior[key]


def test_a_limb_hidden_from_the_cameras_keeps_its_prior(lesson: tm.Lesson) -> None:
    unfitted = [f for f in lesson.fits if f.frequency_hz is None]
    assert unfitted  # a crown always hides some limbs
    assert {f.status for f in unfitted} <= {
        "untracked",
        "weak",
        "inherited",
        "carrier-untracked",
        "few-tracks",
    }


def test_the_renderer_labels_what_it_draws(tree: tuple) -> None:
    splats, _, _, rig = tree
    camera = tm.cameras_around(rig, count=1, width=160, height=120)[0]
    labels = np.arange(len(splats)) % 3
    frame = render(splats, camera, labels=labels)
    drawn = frame.alpha > 0.5
    assert drawn.mean() > 0.05
    assert np.all(frame.label[drawn] >= 0) and np.all(np.isfinite(frame.depth[drawn]))
    assert np.all((frame.purity[drawn] > 0) & (frame.purity[drawn] <= 1 + 1e-9))
    again = render(splats, camera, labels=labels)
    assert np.array_equal(frame.rgb, again.rgb)  # the same scan renders the same


def test_a_camera_round_trips_and_projects_its_target() -> None:
    camera = Camera.look_at([10.0, 0.0, 2.0], [0.0, 0.0, 2.0], width=200, height=100)
    uv, z = camera.project(np.array([[0.0, 0.0, 2.0]]))
    assert np.allclose(uv[0], [100.0, 50.0]) and np.isclose(z[0], 10.0)
    assert Camera.from_json(camera.to_json()).to_json() == camera.to_json()
