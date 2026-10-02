"""The materials teacher (step C2): it recovers the stiffness, damping and drag it was shown.

The yard's snag (instance 9) is swayed by the browser's own wind model (`skin_wind.py`, the
port held to `skinWind.ts`) with a known material -- not the prior -- and rendered from a camera
across the wind. The teacher tracks it in the clip, lifts the tracks onto the scan and fits
the model's predicted spectrum, driven by *other* realisations of the same wind, to what it
saw. It starts once from the property prior and once from a prior four times too stiff, a
quarter of the drag and over four times the damping: both must land on the truth.

The pieces are checked on their own first: the transfer function against the integrator,
the channels against the per-point spectra they stand for, the screen Jacobian against a
finite difference, and `materials.json` round trips.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest

import skin_wind as sw
import teacher_materials as tmat

YARD = Path(__file__).resolve().parents[3] / "data" / "tiles" / "synthetic-yard"
WIND = sw.SkinWind(sw.speed_from_strength(0.5), 60.0)


@cache
def scene() -> tmat.Scene:
    return tmat.load_scene(YARD / "splat", YARD / "skin")


def _prior(instance: int) -> sw.SkinMaterial:
    return tmat.prior_of(
        json.loads((YARD / "instances" / "instances.json").read_text()),
        json.loads((YARD / "skin" / "materials.json").read_text()),
        instance,
    )


def test_the_scene_carries_the_skins_weights():
    s = scene()
    assert {k: int((s.owner == k).sum()) for k in s.skins} == {1: 8980, 9: 1281, 10: 1238, 12: 682}
    at = s.members(9)
    w = s.weights[at, : s.skins[9].handles - 1]
    assert np.abs(w).max() == pytest.approx(1.0, abs=0.01)
    # A constant handle moves the whole object and nothing else.
    q = np.zeros((s.skins[9].handles, 2))
    q[0] = (0.5, -0.25)
    moved = s.moved(9, q).positions - s.splats.positions
    assert np.allclose(moved[at], [0.5, -0.25, 0])
    assert np.abs(np.delete(moved, at, axis=0)).max() == 0


def test_the_spectral_model_is_the_integrator():
    s = scene()

    class Seen:
        fps, frames = 15.0, 150

        def channels(self) -> np.ndarray:
            return np.eye(2 * s.skins[9].handles)

    spectra = tmat.ModelSpectra(s.skins[9], Seen(), WIND, seeds=(5,))  # type: ignore[arg-type]
    material = sw.SkinMaterial(4.2, 0.08, 0.03)
    model = sw.skin_wind_model(s.skins[9], material)
    assert model is not None
    h = sw.SKIN_WIND_STEP_S
    grid = (1000.0 + np.arange(spectra.steps) + 0.5) * h
    force = sw.modal_forces(
        model,
        sw.handle_forces(model, sw.SkinWindField(5, modes=tmat.ENSEMBLE_FIELD_MODES), WIND, grid),
    )
    states = sw.simulate(model.omega, model.damping, force * material.drag)
    k = spectra.warm + np.round(spectra.times / h).astype(int)
    expected = sw.bounded_handles(model, states[k])
    got = spectra.handles(material.stiffness, material.damping, material.drag, 0)
    assert np.abs(got - expected).max() < 1e-9 * np.abs(expected).max()


def test_channels_add_up_to_the_mean_point_spectrum():
    rng = np.random.default_rng(3)
    m, points, frames = 5, 40, 512
    gains = rng.normal(size=(points, 2, 2 * m))
    camera = tmat.Camera.look_at([0, -10, 1], [0, 0, 1])
    seen = tmat.Observation(9, 15.0, camera, np.zeros((points, 2)), np.zeros((points, 3)),
                            gains, np.zeros((frames, points, 2)))  # fmt: skip
    q = np.cumsum(rng.normal(size=(frames, 2 * m)), axis=0)
    per_point = np.einsum("pak,tk->tpa", gains, q)
    _, direct = tmat._welch(per_point, 15.0, 128)
    _, channels = tmat._welch(q @ seen.channels(), 15.0, 128)
    assert np.allclose(direct.mean(1).sum(-1), channels.sum(1), rtol=1e-9)


def test_the_screen_jacobian_is_the_projection_derivative():
    s = scene()
    camera = tmat.camera_for(s, 9, 60.0)
    rest = tmat.render(s.splats, camera, labels=s.owner, index=s.index)
    rows, cols = np.nonzero(rest.label == 9)
    pick = np.linspace(0, rows.size - 1, 20).astype(int)
    uv = np.c_[cols[pick], rows[pick]].astype(np.float32)
    seen = tmat.lift(s, 9, camera, 15.0, uv, np.zeros((4, 20, 2)), rest.label, rest.depth)
    assert seen.points.shape[0] >= 15
    # The constant handle's columns are the pinhole's derivative along east and north.
    eps = 1e-4
    for p, x in enumerate(seen.points):
        for axis in range(2):
            d = np.zeros(3)
            d[axis] = eps
            numeric = (camera.project((x + d)[None])[0] - camera.project((x - d)[None])[0]) / (
                2 * eps
            )
            assert np.allclose(seen.gains[p, :, axis], numeric[0], rtol=1e-4, atol=1e-6)
    # Lifted points sit on the snag.
    members = s.splats.positions[s.members(9)]
    from scipy.spatial import cKDTree

    assert np.median(cKDTree(members).query(seen.points)[0]) < 0.1


def test_materials_are_written_with_their_evidence(tmp_path: Path):
    path = tmp_path / "materials.json"
    path.write_text(json.dumps({"format": "hexapod.materials", "version": 1, "materials": [
        {"instance": 9, "wind": True, "evidence": "prior"},
        {"instance": 12, "stiffness": 9.0},
    ]}))  # fmt: skip
    prior = _prior(9)
    fit = tmat.FitResult(9, sw.SkinMaterial(4.81234, 0.0712, 0.031234), prior, "fitted", 1.0,
                         0.1, 40.0, 50, 900, 15.0)  # fmt: skip
    weak = tmat.FitResult(10, prior, prior, "weak", 1.0, 0.1, 1.0, 50, 900, 15.0)
    doc = tmat.write_materials(path, [fit, weak], tmat.EVIDENCE["generated"], "clip.avi")
    records = {r["instance"]: r for r in doc["materials"]}
    assert sorted(records) == [9, 12]  # the weak fit is not written; 12 is kept
    assert records[9]["evidence"] == "fitted-generated"
    assert (records[9]["stiffness"], records[9]["damping"], records[9]["drag"]) == (
        4.8123,
        0.0712,
        0.03123,
    )
    assert records[12] == {"instance": 12, "stiffness": 9.0}
    again = json.loads(path.read_text())
    assert again == doc
    # What the browser would read: the record over the prior.
    merged = sw.merge_material(prior, records[9])
    assert merged.stiffness == pytest.approx(4.8123) and merged.evidence == "fitted-generated"
    assert tmat.EVIDENCE["real"] == "fitted-real"


@pytest.fixture(scope="module")
def snag() -> tuple[sw.SkinMaterial, tmat.Observation]:
    """The snag swaying for 90 s at 15 fps (320 x 240) with a known material, in a wind of
    320 modes (continuous, as real wind is; the browser draws 64, see ENSEMBLE_FIELD_MODES)."""
    prior = _prior(9)
    truth = sw.SkinMaterial(prior.stiffness * 0.8, 0.07, prior.drag * 1.3)
    s = scene()
    camera = tmat.camera_for(s, 9, WIND.bearing_deg)
    clip = tmat.render_clip(
        s, 9, truth, WIND, seed=7, field_modes=320, seconds=90, fps=15, camera=camera
    )
    return truth, tmat.observe(s, 9, clip.frames, clip.fps, camera)


@pytest.mark.parametrize("start", ["prior", "wrong"])
def test_it_recovers_the_material_it_was_shown(snag, start: str):
    truth, seen = snag
    assert seen.motion.shape[1] >= 40
    prior = _prior(9)
    if start == "wrong":
        prior = sw.SkinMaterial(prior.stiffness * 4, 0.3, prior.drag / 4)
    fit = tmat.fit_material(scene(), seen, WIND, prior)
    assert fit.status == "fitted"
    m = fit.material
    assert abs(m.stiffness / truth.stiffness - 1) < 0.05, fit.to_json()
    assert abs(m.damping / truth.damping - 1) < 0.35, fit.to_json()
    assert abs(m.drag / truth.drag - 1) < 0.2, fit.to_json()
