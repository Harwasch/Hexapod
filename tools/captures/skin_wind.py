"""Wind on scene-object skins, in Python: a port of `packages/world/src/skinWind.ts`.

The video teacher (`teacher_materials.py`, step C2 of docs/LIVING_PLAN.md) fits the materials
the browser's wind model reads (`materials.json`, SCENE_OBJECTS.md §4), so it must simulate the
same model: the anchored modal model of a skin (`skin_wind_model`), the frozen EN 1991-1-4
turbulence field it reads (`frozen_turbulence`, `SkinWindField`), the exact 60 Hz integrator and
the bounded handles (`SkinWindOscillator`). They are transcribed line by line from the
TypeScript (and from `turbulence.ts`, `noise.ts`, `spectral.ts`), with the eigenproblems in
NumPy; `tests/test_skin_wind.py` and `skinWind.test.ts` both check them against
`data/tiles/synthetic-yard/skin/wind_parity.json`, numbers the TypeScript wrote, so the two
cannot drift.

Beyond the port, `handle_forces` and `modal_forces` evaluate the load on a whole time grid at
once (the load does not depend on the state), and `simulate` steps every mode of many
materials together: what a fit needs, many times faster than a step-by-step oscillator, and
equal to it (tested).

    ω_j = c·√λ_j / scale       c: stiffness (m/s), λ_j: the skin's eigenvalues
    s̈ + 2ζΩṡ + Ω²s = Φᵀ F     every anchored mode, ζ: damping
    F_j = (D / scale)(M_0j a_0 + M_jj (a_j − a_0)),  a = |v| v      D: drag
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "ANCHOR_TOLERANCE",
    "HANDLE_REACH",
    "SKIN_WIND_STEP_S",
    "SkinDynamics",
    "SkinMaterial",
    "SkinWind",
    "SkinWindField",
    "SkinWindModel",
    "SkinWindOscillator",
    "bounded_handles",
    "frozen_turbulence",
    "handle_forces",
    "hash32",
    "material_prior",
    "merge_material",
    "modal_forces",
    "simulate",
    "skin_wind_model",
    "speed_from_strength",
    "turbulence_intensity",
    "turbulence_length_scale",
]

# --------------------------------------------------------------------------- constants (TS)

SKIN_WAVE_SPEED_MPS = 3.5
SKIN_RIGID_STIFFENING = 4.0
SKIN_BASE_DAMPING = 0.05
SKIN_FOLIAGE_DAMPING = 0.05
SKIN_DRAG = 0.025
SKIN_MAX_DAMPING = 0.95
ANCHOR_TOLERANCE = 0.05
HANDLE_REACH = 0.25
SKIN_WIND_STEP_S = 1 / 60
SKIN_WIND_MAX_CATCHUP_S = 0.5
SKIN_WIND_FIELD_MODES = 64
FIELD_CUTOFF_HZ = 25.0
MIN_MODE_RATIO = 0.25
WIND_FULL_SCALE_MPS = 20.0

# turbulence.ts
TERRAIN_ROUGHNESS_M = 0.3
TERRAIN_MIN_HEIGHT_M = 5.0
LENGTH_SCALE_REF_M = 300.0
LENGTH_SCALE_REF_HEIGHT_M = 200.0
LATERAL_TURBULENCE_RATIO = 0.75
MEAN_WIND_PERIOD_S = 600.0
BACKGROUND_SOFT_CLIP = 3.0
KAIMAL_A = 10.2
K1_MIN = 0.002
K1_MAX = 1000.0
UINT32 = 0x1_0000_0000
_M32 = 0xFFFFFFFF

# ----------------------------------------------------------------------------- materials


@dataclass(frozen=True)
class SkinMaterial:
    """One instance's material (`SkinMaterial` in skinWind.ts; a `materials.json` record)."""

    stiffness: float  # c, m/s
    damping: float  # ζ
    drag: float  # D, dimensionless
    wind: bool = True
    evidence: str = "prior"


def _score(properties: Mapping[str, float], name: str) -> float:
    v = properties.get(name)
    if isinstance(v, (int, float)) and math.isfinite(v):
        return min(1.0, max(0.0, float(v)))
    return 0.0


def material_prior(properties: Mapping[str, float] | None, behaviour: str | None) -> SkinMaterial:
    """`materialPrior`: from property scores, never class names."""
    p = properties or {}
    vegetation = _score(p, "vegetation")
    if p:
        softness = min(
            1.0, max(0.0, max(vegetation, _score(p, "elastic")) - 0.5 * _score(p, "rigid"))
        )
    else:
        softness = 0.5
    return SkinMaterial(
        stiffness=SKIN_WAVE_SPEED_MPS * SKIN_RIGID_STIFFENING ** (1 - softness),
        damping=SKIN_BASE_DAMPING + SKIN_FOLIAGE_DAMPING * vegetation,
        drag=SKIN_DRAG * (0.25 + 0.75 * vegetation),
        wind=behaviour == "in-place",
        evidence="prior",
    )


def merge_material(base: SkinMaterial, override: Mapping[str, object] | None) -> SkinMaterial:
    """`mergeMaterial`: a record's valid fields over `base`."""
    if not override:
        return base

    def number(name: str) -> float | None:
        v = override.get(name)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            return float(v)
        return None

    stiffness = number("stiffness")
    damping = number("damping")
    drag = number("drag")
    wind = override.get("wind")
    evidence = override.get("evidence")
    return SkinMaterial(
        stiffness=stiffness if stiffness is not None and stiffness > 0 else base.stiffness,
        damping=min(SKIN_MAX_DAMPING, max(0.0, damping)) if damping is not None else base.damping,
        drag=drag if drag is not None and drag >= 0 else base.drag,
        wind=wind if isinstance(wind, bool) else base.wind,
        evidence=evidence if isinstance(evidence, str) else base.evidence,
    )


# ------------------------------------------------------------------------------ the model


@dataclass(frozen=True)
class SkinDynamics:
    """What a skin carries for its dynamics (`SkinDynamicsSource`): one `skin.json` record."""

    handles: int
    origin: np.ndarray  # (3,)
    scale: float
    eigenvalues: np.ndarray  # (m − 1,)
    centres: np.ndarray  # (m − 1, 3), rest frame, from origin
    radii: np.ndarray  # (m − 1,)
    mass: np.ndarray  # (m, m)
    anchor_gram: np.ndarray  # (m, m)
    instance: int = 0

    @staticmethod
    def from_skin(record: Mapping) -> SkinDynamics:
        """A `skin.json` skin record (its `dynamics` block required)."""
        m = int(record["handles"])
        dynamics = record["dynamics"]
        return SkinDynamics(
            handles=m,
            origin=np.asarray(record["origin"], np.float64),
            scale=float(record["scale"]),
            eigenvalues=np.asarray(record["eigenvalues"], np.float64),
            centres=np.asarray([s["centre"] for s in record["support"]], np.float64),
            radii=np.asarray([s["radius"] for s in record["support"]], np.float64),
            mass=from_upper(dynamics["mass"], m),
            anchor_gram=from_upper(dynamics["anchor"]["gram"], m),
            instance=int(record.get("instance", 0)),
        )


def from_upper(upper: Sequence[float], m: int) -> np.ndarray:
    """A symmetric `m × m` from its upper triangle, row by row (`fromUpper`)."""
    if len(upper) != m * (m + 1) // 2:
        raise ValueError(f"{len(upper)} numbers is not the upper triangle of {m} x {m}")
    out = np.zeros((m, m))
    rows, cols = np.triu_indices(m)
    out[rows, cols] = upper
    out[cols, rows] = upper
    return out


@dataclass(frozen=True)
class SkinWindModel:
    """`SkinWindModel`: the anchored modes of one skin under one material."""

    handles: int
    modes: int
    shapes: np.ndarray  # Φ, (m, r)
    omega: np.ndarray  # Ω, rad/s, ascending (r,)
    damping: float
    drag: float  # D / scale
    load: np.ndarray  # M_0j (m,)
    self_mass: np.ndarray  # M_jj (m,)
    points: np.ndarray  # (m, 3) tileset local ENU
    intensity: np.ndarray  # (m,)
    limits: np.ndarray  # (m,)
    anchor_residual: float
    scale: float

    def with_material(self, material: SkinMaterial, base_stiffness: float) -> SkinWindModel:
        """The same skin under another material: Φ does not depend on it, `Ω ∝ c`."""
        return replace(
            self,
            omega=self.omega * (material.stiffness / base_stiffness),
            damping=min(SKIN_MAX_DAMPING, max(0.0, material.damping)),
            drag=max(0.0, material.drag) / self.scale,
        )


def turbulence_intensity(height_m: float) -> float:
    z = max(height_m if math.isfinite(height_m) else 0.0, TERRAIN_MIN_HEIGHT_M)
    return 1.0 / math.log(z / TERRAIN_ROUGHNESS_M)


def turbulence_length_scale(height_m: float = TERRAIN_MIN_HEIGHT_M) -> float:
    z = max(height_m if math.isfinite(height_m) else 0.0, TERRAIN_MIN_HEIGHT_M)
    alpha = 0.67 + 0.05 * math.log(TERRAIN_ROUGHNESS_M)
    return LENGTH_SCALE_REF_M * (z / LENGTH_SCALE_REF_HEIGHT_M) ** alpha


def skin_wind_model(source: SkinDynamics, material: SkinMaterial) -> SkinWindModel | None:
    """`skinWindModel`: None without a direction that keeps the base still."""
    m = source.handles
    if m < 2 or len(source.eigenvalues) < m - 1 or not source.scale > 0:
        return None
    mass, gram = source.mass, source.anchor_gram
    ridged = mass + np.eye(m) * (1e-9 * float(np.trace(mass)))
    try:
        lower = np.linalg.cholesky(ridged)
    except np.linalg.LinAlgError:
        return None
    x = np.linalg.solve(lower, gram)
    c = np.linalg.solve(lower, x.T)
    c = (c + c.T) / 2
    mu, y = np.linalg.eigh(c)
    kept = 0
    residual = 0.0
    while kept < m and mu[kept] <= ANCHOR_TOLERANCE * ANCHOR_TOLERANCE:
        residual = math.sqrt(max(0.0, float(mu[kept])))
        kept += 1
    if kept == 0:
        return None
    b = np.linalg.solve(lower.T, y[:, :kept])  # M-orthonormal
    k2 = (material.stiffness / source.scale) ** 2
    stiffness = np.zeros(m)
    stiffness[1:] = k2 * np.maximum(0.0, source.eigenvalues[: m - 1]) * np.diag(mass)[1:]
    kr = b.T @ (stiffness[:, None] * b)
    kr = (kr + kr.T) / 2
    omega2, u = np.linalg.eigh(kr)
    first = math.sqrt(k2 * max(0.0, float(source.eigenvalues[0])))
    modes = [i for i in range(kept) if math.sqrt(max(0.0, omega2[i])) >= MIN_MODE_RATIO * first]
    shapes = b @ u[:, modes]
    omega = np.sqrt(np.maximum(0.0, omega2[modes]))
    centres = np.vstack([[0.0, 0.0, source.scale], source.centres[: m - 1]])
    points = source.origin[None, :] + centres
    intensity = np.array([turbulence_intensity(max(0.0, float(z))) for z in centres[:, 2]])
    limits = HANDLE_REACH * np.r_[source.scale, np.maximum(source.radii[: m - 1], 1e-3)]
    return SkinWindModel(
        handles=m,
        modes=len(modes),
        shapes=shapes,
        omega=omega,
        damping=min(SKIN_MAX_DAMPING, max(0.0, material.damping)),
        drag=max(0.0, material.drag) / source.scale,
        load=mass[0].copy(),
        self_mass=np.diag(mass).copy(),
        points=points,
        intensity=intensity,
        limits=limits,
        anchor_residual=residual,
        scale=source.scale,
    )


# ------------------------------------------------------------------------- the wind field


def hash32(value: int | np.ndarray, seed: int) -> int | np.ndarray:
    """`hash32` of noise.ts: the same 32-bit avalanche, on Python ints or uint64 arrays."""
    if isinstance(value, np.ndarray):
        h = ((int(seed) * 0x9E3779B1) & _M32) ^ (value.astype(np.uint64) & np.uint64(_M32))
        h = h.astype(np.uint64)
        h = ((h ^ (h >> np.uint64(16))) * np.uint64(0x85EBCA6B)) & np.uint64(_M32)
        h = ((h ^ (h >> np.uint64(13))) * np.uint64(0xC2B2AE35)) & np.uint64(_M32)
        return (h ^ (h >> np.uint64(16))) & np.uint64(_M32)
    seed32 = math.trunc(seed) & _M32
    h = ((seed32 * 0x9E3779B1) & _M32) ^ (int(value) & _M32)
    h = ((h ^ (h >> 16)) * 0x85EBCA6B) & _M32
    h = ((h ^ (h >> 13)) * 0xC2B2AE35) & _M32
    return (h ^ (h >> 16)) & _M32


def unit_uniform(counter: int, seed: int) -> float:
    return (int(hash32(counter, seed)) + 0.5) / UINT32


@dataclass(frozen=True)
class FrozenTurbulence:
    wavevectors: np.ndarray  # (n, 3) wind frame, cycles per L
    amplitudes: np.ndarray  # (n,)
    phases: np.ndarray  # (n, 4): cos, sin of the along phase, then of the across phase

    @property
    def count(self) -> int:
        return int(self.amplitudes.size)


def frozen_turbulence(seed: int, count: int = 320) -> FrozenTurbulence:
    """`frozenTurbulence` of turbulence.ts."""
    salt = int(hash32(0x7B1E, seed))
    log_min = math.log(K1_MIN)
    log_span = math.log(K1_MAX) - log_min

    def density(k1: float) -> float:
        return math.sqrt(k1 * math.pow(1 + KAIMAL_A * k1, -5 / 3))

    steps = 4096
    cumulative = [0.0] * (steps + 1)
    previous = density(K1_MIN)
    for i in range(1, steps + 1):
        nxt = density(math.exp(log_min + (i / steps) * log_span))
        cumulative[i] = cumulative[i - 1] + 0.5 * (previous + nxt)
        previous = nxt
    end = cumulative[steps]

    def quantile(u: float) -> float:
        target = u * end
        lo, hi = 0, steps
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if cumulative[mid] < target:
                lo = mid
            else:
                hi = mid
        a, b = cumulative[lo], cumulative[hi]
        f = (target - a) / (b - a) if b > a else 0.0
        return math.exp(log_min + ((lo + f) / steps) * log_span)

    wavevectors = np.zeros((count, 3))
    amplitudes = np.zeros(count)
    phases = np.zeros((count, 4))
    total = 0.0
    for j in range(count):
        k1 = quantile((j + unit_uniform(j * 8, salt)) / count)
        share = density(k1)
        amplitudes[j] = share
        total += share
        u = unit_uniform(j * 8 + 1, salt)
        magnitude = ((1 + KAIMAL_A * k1) * math.pow(1 - u, -3 / 5) - 1) / KAIMAL_A
        across = math.sqrt(max(0.0, magnitude * magnitude - k1 * k1))
        azimuth = 2 * math.pi * unit_uniform(j * 8 + 2, salt)
        wavevectors[j] = (k1, across * math.cos(azimuth), across * math.sin(azimuth))
        pu = 2 * math.pi * unit_uniform(j * 8 + 3, salt)
        pv = 2 * math.pi * unit_uniform(j * 8 + 4, salt)
        phases[j] = (math.cos(pu), math.sin(pu), math.cos(pv), math.sin(pv))
    amplitudes = np.sqrt(2 * amplitudes / total)
    return FrozenTurbulence(wavevectors, amplitudes, phases)


def speed_from_strength(strength: float) -> float:
    """`speedFromStrength` of living.ts: the Living Survey's wind slider to m/s."""
    s = min(1.0, max(0.0, strength)) if math.isfinite(strength) else 0.0
    return 0.0 if s == 0 else WIND_FULL_SCALE_MPS * math.sqrt(s)


@dataclass(frozen=True)
class SkinWind:
    speed_mps: float
    bearing_deg: float
    turbulence: float = 1.0


class SkinWindField:
    """`SkinWindField`: one frozen field for a scene, deterministic from its seed."""

    def __init__(
        self,
        seed: int,
        length_scale_m: float | None = None,
        modes: int = SKIN_WIND_FIELD_MODES,
    ) -> None:
        self.field = frozen_turbulence(seed, modes)
        self.length_scale_m = (
            turbulence_length_scale(TERRAIN_MIN_HEIGHT_M)
            if length_scale_m is None
            else length_scale_m
        )

    def phases(self, points: np.ndarray, bearing_deg: float) -> np.ndarray:
        """`turbulencePhases` at each of `points` (n, 3): (n, modes, 4)."""
        b = math.radians(bearing_deg)
        de, dn = math.sin(b), math.cos(b)
        p = np.atleast_2d(np.asarray(points, np.float64))
        along = p[:, 0] * de + p[:, 1] * dn
        across = p[:, 0] * dn - p[:, 1] * de
        k = self.field.wavevectors
        dot = along[:, None] * k[None, :, 0] + across[:, None] * k[None, :, 1]
        dot = dot + p[:, 2:3] * k[None, :, 2]
        phase = 2 * math.pi * dot / self.length_scale_m
        c, s = np.cos(phase), np.sin(phase)
        cu, su, cv, sv = (self.field.phases[:, i][None] for i in range(4))
        return np.stack([c * cu - s * su, s * cu + c * su, c * cv - s * sv, s * cv + c * sv], -1)

    def frames(self, times: np.ndarray, speed_mps: float) -> tuple[np.ndarray, np.ndarray]:
        """`frame(t, U)` at every time: `P`, `Q` (len(times), modes) (`turbulenceClock`
        folded through the 25 Hz low-pass, as `foldClock`)."""
        t = np.asarray(times, np.float64)
        k1 = self.field.wavevectors[:, 0]
        f = (speed_mps / self.length_scale_m) * k1
        high = 1 / MEAN_WIND_PERIOD_S
        weights = self.field.amplitudes * f / np.sqrt(f * f + high * high)
        r = f / FIELD_CUTOFF_HZ
        re = 1 - r * r
        two_zeta = 2 * math.sqrt(0.5)
        gain = weights / (re * re + two_zeta * two_zeta * r * r)
        a, b = re * gain, two_zeta * r * gain
        angle = -2 * math.pi * f[None, :] * t[:, None]
        c, s = np.cos(angle), np.sin(angle)
        return a * c - b * s, a * s + b * c


def sample_wind(phases: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """`sampleSkinWind` for points (n, modes, 4) at times (T, modes): (T, n, 2), clipped."""
    along = p @ phases[:, :, 0].T - q @ phases[:, :, 1].T
    across = p @ phases[:, :, 2].T - q @ phases[:, :, 3].T
    out = np.stack([along, across], -1)
    return BACKGROUND_SOFT_CLIP * np.tanh(out / BACKGROUND_SOFT_CLIP)


def handle_forces(
    model: SkinWindModel, field: SkinWindField, wind: SkinWind, times: np.ndarray
) -> np.ndarray:
    """Each handle's load per unit drag number at each time, (T, m, 2) east/north:
    `(M_0j a_0 + M_jj (a_j − a_0)) / scale` (handle 0: `a_0 / scale`) -- the `#forces` of the
    TS oscillator before the modal projection and `D`."""
    times = np.asarray(times, np.float64)
    m = model.handles
    turbulence = max(0.0, wind.turbulence)
    if turbulence > 0:
        p, q = field.frames(times, wind.speed_mps)
        sample = sample_wind(field.phases(model.points, wind.bearing_deg), p, q)
    else:
        sample = np.zeros((times.size, m, 2))
    b = math.radians(wind.bearing_deg)
    de, dn = math.sin(b), math.cos(b)
    u = wind.speed_mps
    intensity = turbulence * model.intensity[None, :]
    along = u * (1 + intensity * sample[..., 0])
    across = u * LATERAL_TURBULENCE_RATIO * intensity * sample[..., 1]
    ve = along * de + across * dn
    vn = along * dn - across * de
    speed = np.hypot(ve, vn)
    a = np.stack([speed * ve, speed * vn], -1)  # (T, m, 2)
    a0 = a[:, :1]
    out = model.load[None, :, None] * a0 + model.self_mass[None, :, None] * (a - a0)
    out[:, 0] = a[:, 0]
    return out / model.scale


def modal_forces(model: SkinWindModel, forces: np.ndarray) -> np.ndarray:
    """`Φᵀ F` per unit drag number, (T, r, 2), from `handle_forces`."""
    return np.einsum("jr,tja->tra", model.shapes, forces)


# ------------------------------------------------------------------------- the oscillator


def transition(omega: np.ndarray, zeta: np.ndarray | float, h: float) -> np.ndarray:
    """Exact one-step transition of `s̈ + 2ζΩṡ + Ω²s = f` (f held) over `h`: (..., 4)."""
    omega = np.asarray(omega, np.float64)
    zeta = np.asarray(zeta, np.float64)
    a = zeta * omega
    wd = omega * np.sqrt(np.maximum(1e-12, 1 - zeta * zeta))
    e = np.exp(-a * h)
    c, s = np.cos(wd * h), np.sin(wd * h)
    return np.stack([e * (c + (a / wd) * s), e * s / wd, -e * omega * omega * s / wd, e * (c - (a / wd) * s)], -1)  # fmt: skip


def simulate(omega: np.ndarray, zeta: np.ndarray | float, force: np.ndarray) -> np.ndarray:
    """Modal displacements on the grid from rest, (T + 1, ..., r, 2): state 0 at rest, state
    `k + 1` after step `k` under `force[k]` (T, ..., r, 2) -- the force held over the step,
    sampled at its middle, as the TS oscillator. `omega` (..., r), `zeta` broadcasts."""
    coef = transition(
        omega, np.asarray(zeta)[..., None] if np.ndim(zeta) else zeta, SKIN_WIND_STEP_S
    )
    a11, a12, a21, a22 = (coef[..., i][..., None] for i in range(4))
    w2 = (np.asarray(omega, np.float64) ** 2)[..., None]
    steps = force.shape[0]
    x = np.zeros(force.shape[1:])
    v = np.zeros(force.shape[1:])
    out = np.empty((steps + 1,) + force.shape[1:])
    out[0] = x
    for k in range(steps):
        eq = force[k] / w2
        dx = x - eq
        x, v = eq + a11 * dx + a12 * v, a21 * dx + a22 * v
        out[k + 1] = x
    return out


def bounded_handles(model: SkinWindModel, modal: np.ndarray) -> np.ndarray:
    """Modal displacements (..., r, 2) to handle translations (..., m, 2), bounded as
    `handles()` does: every `q` scaled by `tanh(ρ)/ρ`, `ρ = max_j |q_j| / limit_j`."""
    q = np.einsum("jr,...ra->...ja", model.shapes, modal)
    ratio = (np.hypot(q[..., 0], q[..., 1]) / model.limits).max(-1)
    factor = np.where(ratio > 1e-12, np.tanh(ratio) / np.maximum(ratio, 1e-300), 1.0)
    return q * factor[..., None, None]


class SkinWindOscillator:
    """`SkinWindOscillator`, step for step (for parity; `simulate` is the fast path)."""

    def __init__(self, model: SkinWindModel) -> None:
        self.model = model
        r = model.modes
        self.x = np.zeros((r, 2))
        self.v = np.zeros((r, 2))
        self.previous = np.zeros((r, 2))
        self.coefficients = transition(model.omega, model.damping, SKIN_WIND_STEP_S)
        self.step: int | None = None
        self._phases: np.ndarray | None = None
        self._bearing = math.nan
        self._field: SkinWindField | None = None

    def reset(self) -> None:
        self.step = None
        self.x[:] = 0
        self.v[:] = 0
        self.previous[:] = 0

    def _forces(self, field: SkinWindField, wind: SkinWind, t: float) -> np.ndarray:
        f = handle_forces(self.model, field, wind, np.array([t]))[0] * self.model.scale
        return np.einsum("jr,ja->ra", self.model.shapes * self.model.drag, f)

    def advance(self, field: SkinWindField, wind: SkinWind, t: float) -> None:
        if not wind.speed_mps > 0 or not math.isfinite(t) or self.model.modes == 0:
            self.reset()
            return
        h = SKIN_WIND_STEP_S
        target = math.floor(t / h + 1e-9)
        w2 = (self.model.omega**2)[:, None]
        if self.step is None:
            self.step = target
            self.x[:] = 0
            self.v[:] = 0
            self.previous[:] = 0
        elif t < (self.step - 1) * h or t - self.step * h > SKIN_WIND_MAX_CATCHUP_S:
            self.step = target
            self.x = self._forces(field, wind, target * h) / w2
            self.v[:] = 0
            self.previous = self.x.copy()
        a11, a12, a21, a22 = (self.coefficients[:, i][:, None] for i in range(4))
        while self.step * h < t - 1e-9:
            force = self._forces(field, wind, (self.step + 0.5) * h)
            self.previous = self.x.copy()
            eq = force / w2
            dx = self.x - eq
            self.x, self.v = eq + a11 * dx + a12 * self.v, a21 * dx + a22 * self.v
            self.step += 1

    def handles(self, t: float) -> np.ndarray:
        """`12·m` numbers, `Z_j = [0 | q_j]` row-major (zeros at rest)."""
        m = self.model.handles
        out = np.zeros(m * 12)
        if self.step is None or self.model.modes == 0:
            return out
        h = SKIN_WIND_STEP_S
        alpha = min(1.0, max(0.0, (t - (self.step - 1) * h) / h))
        modal = self.previous + alpha * (self.x - self.previous)
        q = bounded_handles(self.model, modal)
        out[3::12] = q[:, 0]
        out[7::12] = q[:, 1]
        return out
