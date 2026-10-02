"""The Python port of the skin wind model (`skin_wind.py`) against the TypeScript one.

`data/tiles/synthetic-yard/skin/wind_parity.json` is written by
`packages/world/src/skinWind.test.ts` (which checks it too): the frozen field, the anchored
modes of the yard's tree and a shrub under two materials, and their handles over four seconds
from rest. The port must give the same numbers; the fast path (`simulate`) the same handles
as the step-by-step oscillator.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import skin_wind as sw

YARD = Path(__file__).resolve().parents[3] / "data" / "tiles" / "synthetic-yard" / "skin"
PARITY = json.loads((YARD / "wind_parity.json").read_text(encoding="utf-8"))
SKINS = {s["instance"]: s for s in json.loads((YARD / "skin.json").read_text())["skins"]}
PLANT = sw.material_prior({"vegetation": 1, "elastic": 1, "rigid": 0}, "in-place")


def _material(run: dict) -> sw.SkinMaterial:
    return sw.SkinMaterial(run["stiffness"], run["damping"], run["drag"])


def test_hash_and_field_match_the_typescript():
    field = sw.frozen_turbulence(PARITY["field"]["seed"], PARITY["field"]["modes"])
    assert np.allclose(
        field.wavevectors.reshape(-1)[:12], PARITY["field"]["wavevectors"], rtol=1e-12
    )
    assert np.allclose(field.amplitudes[:4], PARITY["field"]["amplitudes"], rtol=1e-12)
    assert np.allclose(field.phases.reshape(-1)[:8], PARITY["field"]["phases"], atol=1e-12)
    # hash32 on ints and on arrays agree.
    values = np.arange(0, 4000, 37)
    assert [int(v) for v in sw.hash32(values, 99)] == [int(sw.hash32(int(v), 99)) for v in values]


def test_priors_match_the_typescript():
    assert PLANT.stiffness == pytest.approx(3.5)
    assert PLANT.damping == pytest.approx(0.1)
    assert PLANT.drag == pytest.approx(0.025)
    post = sw.material_prior({"vegetation": 0, "elastic": 0, "rigid": 1}, "in-place")
    assert post.stiffness == pytest.approx(14.0)
    assert not sw.material_prior({"vegetation": 0.9}, "movable").wind
    merged = sw.merge_material(PLANT, {"stiffness": 2, "damping": 7, "drag": math.nan})
    assert (merged.stiffness, merged.damping, merged.drag) == (2, 0.95, PLANT.drag)
    assert sw.speed_from_strength(0.1) == pytest.approx(PARITY["wind"]["speedMps"], rel=1e-15)


@pytest.mark.parametrize("index", range(len(PARITY["runs"])))
def test_modes_and_handles_match_the_typescript(index: int):
    run = PARITY["runs"][index]
    source = sw.SkinDynamics.from_skin(SKINS[run["instance"]])
    model = sw.skin_wind_model(source, _material(run))
    assert model is not None
    assert model.modes == run["modes"]
    assert np.allclose(model.omega, run["omega"], rtol=1e-9)
    assert model.anchor_residual == pytest.approx(run["anchorResidual"], rel=1e-6, abs=1e-12)
    wind = sw.SkinWind(PARITY["wind"]["speedMps"], PARITY["wind"]["bearingDeg"])
    field = sw.SkinWindField(PARITY["wind"]["seed"])
    oscillator = sw.SkinWindOscillator(model)
    t0, fps = PARITY["wind"]["t0"], PARITY["wind"]["fps"]
    peak = max(max(abs(v) for v in h) for h in run["handles"].values())
    assert peak > 1e-4  # it moved
    for k in range(121):
        t = t0 + k / fps
        oscillator.advance(field, wind, t)
        if str(k) in run["handles"]:
            got = oscillator.handles(t)
            assert np.allclose(got, run["handles"][str(k)], rtol=0, atol=1e-9 * peak), k


def test_the_fast_path_is_the_oscillator():
    source = sw.SkinDynamics.from_skin(SKINS[1])
    material = sw.SkinMaterial(2.4, 0.07, 0.04)
    model = sw.skin_wind_model(source, material)
    assert model is not None
    wind = sw.SkinWind(6.3, 60)
    field = sw.SkinWindField(11)
    h = sw.SKIN_WIND_STEP_S
    start = 30000  # a grid step: t = 500 s
    steps = 240
    mid = (start + np.arange(steps) + 0.5) * h
    force = sw.modal_forces(model, sw.handle_forces(model, field, wind, mid)) * material.drag
    states = sw.simulate(model.omega, model.damping, force)
    fast = sw.bounded_handles(model, states)
    oscillator = sw.SkinWindOscillator(model)
    for k in range(steps + 1):
        t = (start + k) * h
        oscillator.advance(field, wind, t)
        z = oscillator.handles(t)
        assert np.allclose(fast[k, :, 0], z[3::12], atol=1e-12)
        assert np.allclose(fast[k, :, 1], z[7::12], atol=1e-12)
    # Ω ∝ c, and a material swap keeps the shapes.
    other = model.with_material(sw.SkinMaterial(4.8, 0.2, 0.01), material.stiffness)
    direct = sw.skin_wind_model(source, sw.SkinMaterial(4.8, 0.2, 0.01))
    assert direct is not None
    assert np.allclose(other.omega, direct.omega, rtol=1e-9)
    assert other.drag == pytest.approx(direct.drag)
