/**
 * The wind a tree stands in: turbulence whose spectrum depends on the mean speed, and the
 * along-wind buffeting response of a limb to it, split into background and resonance the way
 * the Eurocode splits a structure's response (EN 1991-1-4:2005+A1:2010, §6.3 and Annex B).
 *
 * Why this exists. A limb's motion used to be read from a texture whose spectrum was the
 * limb's oscillator response to *flat* forcing, placed at the limb's own frequency: the same
 * narrow resonance at every wind speed, only louder. Real response is
 * `|H(f)|² · S_wind(f; U)`, and the wind spectrum's energy sits at `f ≈ 0.15·U/L` with `L` the
 * turbulent length scale — tens of metres near the ground — so at 2 m/s almost all of it is
 * far below a limb's 1–3 Hz and the limb simply follows the gusts, slowly and in step with its
 * neighbours; the resonance only shows as the wind rises. Here:
 *
 * - **Turbulence intensity** `I_v(z) = 1 / ln(z/z0)` (EN eq. 4.7, `k_I = c0 = 1`).
 * - **Length scale** `L(z) = 300·(z/200)^α`, `α = 0.67 + 0.05·ln z0`, and `L(z_min)` below
 *   `z_min` (EN eq. B.1).
 * - **Spectrum** `S_L(f_L) = n·S_v/σ² = 6.8·f_L / (1 + 10.2·f_L)^(5/3)`, `f_L = n·L/U` (EN
 *   eq. B.2). It integrates to exactly 1 over `f_L`.
 * - **Background** `B² = 1 / (1 + 0.9·((b + h)/L)^0.63)` (EN eq. B.3): the lack of full
 *   correlation of the gusts over a limb `b` wide and `h` tall.
 * - **Resonance** `R² = π²/(2δ) · S_L(f_n) · R_h(η_h) · R_b(η_b)`, `R = 1/η − (1 − e^(−2η))/(2η²)`,
 *   `η = 4.6·(h or b)·f_L/L` (EN eqs. B.6–B.8), `δ` the logarithmic decrement `2π(ζ_s + ζ_a)`.
 * - **Aerodynamic damping** `δ_a = c_f·ρ·b·v_m / (2·n·m_e)` (EN eq. F.18). With the static
 *   deflection `x_s = (½ρ c_f b v²)/(m_e (2πn)²)` it is `ζ_a = δ_a/2π = 2π·n·x_s / U`: known from
 *   the limb's own static bend, without its mass.
 * - **Fluctuating response**: `2·I_v·√(B² + R²)` of the mean (EN eq. 6.3; the factor 2 is the
 *   linearised drag, `(U + u)² ≈ U² + 2Uu`).
 *
 * The background is not a number but a signal: a **frozen turbulence field** (Taylor's
 * hypothesis, as Habel et al. 2009 §7.2 advect their leaf field) — a sum of random Fourier
 * modes with isotropic wavevectors whose one-dimensional spectrum along any line is exactly
 * `S_L` — carried downwind at `U`. A point sees mode `j` at `f_j = U·(k_j·ŵ)`, so the corner of
 * the spectrum moves with `U` by itself; neighbouring limbs read the same field at their own
 * positions, so a gust reaches the downwind limb `Δx/U` later and limbs a small fraction of `L`
 * apart move nearly together at low frequency and independently at high. Each limb reads the
 * field through its static compliance, low-passed at its own frequency; the resonance, a narrow
 * band that the flat-forced motion texture already has, is added on top with variance `R²`.
 *
 * Everything here is a pure function of its arguments.
 */

import { hash32 } from "./noise";
import { unitUniform } from "./spectral";

// ---------------------------------------------------------------------------------------------
// EN 1991-1-4 turbulence, Annex B
// ---------------------------------------------------------------------------------------------

/** EN 1991-1-4 Table 4.1, terrain category III: villages, suburban terrain, permanent forest. */
export const TERRAIN_ROUGHNESS_M = 0.3;
/** EN 1991-1-4 Table 4.1, the minimum height `z_min` for terrain category III, metres. */
export const TERRAIN_MIN_HEIGHT_M = 5;
/** EN 1991-1-4 eq. B.1: reference length scale `L_t`, metres, at reference height `z_t`. */
export const LENGTH_SCALE_REF_M = 300;
export const LENGTH_SCALE_REF_HEIGHT_M = 200;
/**
 * RMS of the across-wind to the along-wind turbulence, `σ_w/σ_u` in the horizontal plane.
 * Estimate: the previous sway ratios' own proportion (0.6 : 0.8); the IEC 61400-1 Kaimal model
 * is quoted as using 0.8, which could not be read at the source.
 */
export const LATERAL_TURBULENCE_RATIO = 0.75;
/**
 * The mean wind is a 10-minute mean (EN 1991-1-4 B.2 (3): averaging time `T = 600 s`), so
 * fluctuations slower than that are changes of the mean, not turbulence. High-pass corner, Hz.
 */
export const MEAN_WIND_PERIOD_S = 600;

/** EN eq. 4.7: turbulence intensity at height `z`, `1/ln(z/z0)`, held below `z_min`. */
export function turbulenceIntensity(
  heightM: number,
  roughnessM = TERRAIN_ROUGHNESS_M,
  minHeightM = TERRAIN_MIN_HEIGHT_M,
): number {
  const z = Math.max(Number.isFinite(heightM) ? heightM : 0, minHeightM);
  return 1 / Math.log(z / roughnessM);
}

/** EN eq. B.1: turbulent length scale at height `z`, metres, held below `z_min`. */
export function turbulenceLengthScaleM(
  heightM: number,
  roughnessM = TERRAIN_ROUGHNESS_M,
  minHeightM = TERRAIN_MIN_HEIGHT_M,
): number {
  const z = Math.max(Number.isFinite(heightM) ? heightM : 0, minHeightM);
  const alpha = 0.67 + 0.05 * Math.log(roughnessM);
  return LENGTH_SCALE_REF_M * Math.pow(z / LENGTH_SCALE_REF_HEIGHT_M, alpha);
}

/** EN eq. B.2: `n·S_v(n)/σ²` at the non-dimensional frequency `f_L = n·L/U`. */
export function kaimalSpectrum(fL: number): number {
  if (!(fL > 0)) return 0;
  return (6.8 * fL) / Math.pow(1 + 10.2 * fL, 5 / 3);
}

/** EN eqs. B.7–B.8: aerodynamic admittance `R(η) = 1/η − (1 − e^(−2η))/(2η²)`, 1 at 0. */
export function aerodynamicAdmittance(eta: number): number {
  if (!(eta > 1e-6)) return 1;
  return 1 / eta - (1 - Math.exp(-2 * eta)) / (2 * eta * eta);
}

/** EN eq. B.3: background factor `B²` for a body `b` wide and `h` tall in turbulence of scale `L`. */
export function backgroundFactor(sizeM: number, lengthScaleM: number): number {
  const ratio = Math.max(0, sizeM) / lengthScaleM;
  return 1 / (1 + 0.9 * Math.pow(ratio, 0.63));
}

/** What a limb's buffeting response is made of at one mean speed. */
export interface BuffetingFactors {
  /** `B²`, EN eq. B.3. */
  readonly background2: number;
  /** `R²`, EN eq. B.6, with the aerodynamic damping of F.18 in its decrement. */
  readonly resonance2: number;
  /** Aerodynamic damping ratio at this speed. */
  readonly aerodynamicDamping: number;
}

/**
 * Background and resonance of one oscillator at mean speed `U` (EN 1991-1-4 Annex B).
 *
 * `widthM`, `heightM` are the oscillator's extent across the wind and vertically; `staticTipM`
 * its static tip deflection at this speed, from which the aerodynamic damping follows (F.18).
 */
export function buffetingFactors(
  frequencyHz: number,
  structuralDamping: number,
  widthM: number,
  heightM: number,
  staticTipM: number,
  speedMps: number,
  lengthScaleM: number,
): BuffetingFactors {
  const background2 = backgroundFactor(widthM + heightM, lengthScaleM);
  if (!(speedMps > 0) || !(frequencyHz > 0))
    return { background2, resonance2: 0, aerodynamicDamping: 0 };
  const fL = (frequencyHz * lengthScaleM) / speedMps;
  const aerodynamicDamping = (2 * Math.PI * frequencyHz * Math.max(0, staticTipM)) / speedMps;
  const delta = 2 * Math.PI * (structuralDamping + aerodynamicDamping);
  const rh = aerodynamicAdmittance((4.6 * heightM * fL) / lengthScaleM);
  const rb = aerodynamicAdmittance((4.6 * widthM * fL) / lengthScaleM);
  const resonance2 = ((Math.PI * Math.PI) / (2 * delta)) * kaimalSpectrum(fL) * rh * rb;
  return { background2, resonance2, aerodynamicDamping };
}

// ---------------------------------------------------------------------------------------------
// Frozen turbulence: random Fourier modes
// ---------------------------------------------------------------------------------------------

/** Modes per field. Enough that no autocorrelation from 10 s to an hour stands out. */
export const FROZEN_TURBULENCE_MODES = 320;

/** `a` in EN eq. B.2's `(1 + a·f_L)`. */
const KAIMAL_A = 10.2;

/**
 * The along-wind wavenumbers the modes span, cycles per length scale. A point sees mode `j` at
 * `f = k1·U/L`: from the 10-minute mean's `1/600 Hz` at 20 m/s (`k1 ≈ 0.003` for `L` = 35 m) to
 * 10 Hz at 0.5 m/s (`k1 ≈ 700`). The spectrum outside holds 1.6 % of the variance.
 */
const K1_MIN = 0.002;
const K1_MAX = 1000;

/**
 * A frozen turbulence field of unit variance, in the **wind frame** (along, across, up): each mode
 * a wavevector in cycles per length scale, an amplitude, and two phases, one for the along-wind
 * velocity component and one for the across-wind. Deterministic from the seed; independent of
 * `L` and `U`, which only scale it.
 *
 * What a fixed point sees is set by the along-wind wavenumbers alone (`f = k1·U/L`). Drawn
 * from the energy, they would leave the octaves a limb sways in (0.1–1 Hz) with a handful of
 * lines; drawn evenly in `log k1`, the energetic octaves would hold few enough modes that the
 * sum partly repeats within an hour. So they are stratified with density `√(k1·F(k1))` per
 * unit `log k1` — `k1·F(k1)` being the variance per unit `log k1` of EN eq. B.2's one-sided
 * `F(k1) ∝ (1 + a·k1)^(−5/3)` — and each mode carries its stratum's variance, `∝ √(k1·F)`. The
 * across and vertical components come from the isotropic field the spectrum is a line through:
 * for a scalar field of random-direction modes with magnitudes `κ ~ p(κ)`, `F(k1) = ∫ p(κ)/κ dκ`
 * over `κ > k1`, so given `k1` the magnitude has density `∝ p(κ)/κ = −F'(κ) ∝ (1 + aκ)^(−8/3)`,
 * which inverts in closed form. They are what decorrelates two limbs side by side.
 */
export interface FrozenTurbulence {
  /** Wavevectors in the wind frame, cycles per `L`: `[along0, across0, up0, along1, ...]`. */
  readonly wavevectors: Float64Array;
  /** Per mode `√(2·variance share)`: the amplitude of its cosine. `Σ share = 1`. */
  readonly amplitudes: Float64Array;
  /** `cos`, `sin` of each mode's along-wind phase, then of its across-wind phase: 4 per mode. */
  readonly phases: Float64Array;
  readonly count: number;
}

const FIELD_CACHE = new Map<string, FrozenTurbulence>();

export function frozenTurbulence(seed: number, count = FROZEN_TURBULENCE_MODES): FrozenTurbulence {
  const key = `${count}:${seed}`;
  const cached = FIELD_CACHE.get(key);
  if (cached !== undefined) return cached;
  const salt = hash32(0x7b1e, seed);
  const wavevectors = new Float64Array(count * 3);
  const amplitudes = new Float64Array(count);
  const phases = new Float64Array(count * 4);
  // Stratify log k1 with density √(k1·F(k1)): a table of its cumulative integral, inverted.
  const logMin = Math.log(K1_MIN);
  const logSpan = Math.log(K1_MAX) - logMin;
  const density = (k1: number): number => Math.sqrt(k1 * Math.pow(1 + KAIMAL_A * k1, -5 / 3));
  const steps = 4096;
  const cumulative = new Float64Array(steps + 1);
  let previous = density(K1_MIN);
  for (let i = 1; i <= steps; i += 1) {
    const next = density(Math.exp(logMin + (i / steps) * logSpan));
    cumulative[i] = (cumulative[i - 1] ?? 0) + 0.5 * (previous + next);
    previous = next;
  }
  const end = cumulative[steps] ?? 1;
  const quantile = (u: number): number => {
    const target = u * end;
    let lo = 0;
    let hi = steps;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if ((cumulative[mid] ?? 0) < target) lo = mid;
      else hi = mid;
    }
    const a = cumulative[lo] ?? 0;
    const b = cumulative[hi] ?? a;
    const f = b > a ? (target - a) / (b - a) : 0;
    return Math.exp(logMin + ((lo + f) / steps) * logSpan);
  };
  let total = 0;
  for (let j = 0; j < count; j += 1) {
    const k1 = quantile((j + unitUniform(j * 8, salt)) / count);
    // Its stratum's variance: k1·F(k1) over the sampling density √(k1·F(k1)).
    const share = density(k1);
    amplitudes[j] = share;
    total += share;
    // The magnitude given k1: (1 + aκ)^(−5/3) = (1 + a k1)^(−5/3)·(1 − u).
    const u = unitUniform(j * 8 + 1, salt);
    const magnitude = ((1 + KAIMAL_A * k1) * Math.pow(1 - u, -3 / 5) - 1) / KAIMAL_A;
    const across = Math.sqrt(Math.max(0, magnitude * magnitude - k1 * k1));
    const azimuth = 2 * Math.PI * unitUniform(j * 8 + 2, salt);
    wavevectors[j * 3] = k1;
    wavevectors[j * 3 + 1] = across * Math.cos(azimuth);
    wavevectors[j * 3 + 2] = across * Math.sin(azimuth);
    const pu = 2 * Math.PI * unitUniform(j * 8 + 3, salt);
    const pv = 2 * Math.PI * unitUniform(j * 8 + 4, salt);
    phases[j * 4] = Math.cos(pu);
    phases[j * 4 + 1] = Math.sin(pu);
    phases[j * 4 + 2] = Math.cos(pv);
    phases[j * 4 + 3] = Math.sin(pv);
  }
  for (let j = 0; j < count; j += 1) amplitudes[j] = Math.sqrt((2 * (amplitudes[j] ?? 0)) / total);
  const field = { wavevectors, amplitudes, phases, count };
  FIELD_CACHE.set(key, field);
  return field;
}

/**
 * The field's phases at a point, for a wind blowing towards `downwind` (horizontal unit
 * vector): per mode, `cos` and `sin` of `2π·k·x/L + φ_u` and of `2π·k·x/L + φ_v`, `x` the point
 * in the wind frame. 4 numbers a mode; fixed while the point and the bearing are.
 */
export function turbulencePhases(
  field: FrozenTurbulence,
  position: readonly [number, number, number],
  downwind: readonly [number, number, number],
  lengthScaleM: number,
): Float64Array {
  const out = new Float64Array(field.count * 4);
  const k = field.wavevectors;
  const p = field.phases;
  // Wind frame: along ŵ, across it horizontally, up.
  const along = position[0] * downwind[0] + position[1] * downwind[1];
  const across = position[0] * downwind[1] - position[1] * downwind[0];
  const up = position[2];
  for (let j = 0; j < field.count; j += 1) {
    const dot = (k[j * 3] ?? 0) * along + (k[j * 3 + 1] ?? 0) * across + (k[j * 3 + 2] ?? 0) * up;
    const phase = (2 * Math.PI * dot) / lengthScaleM;
    const c = Math.cos(phase);
    const s = Math.sin(phase);
    const cu = p[j * 4] ?? 1;
    const su = p[j * 4 + 1] ?? 0;
    const cv = p[j * 4 + 2] ?? 1;
    const sv = p[j * 4 + 3] ?? 0;
    out[j * 4] = c * cu - s * su;
    out[j * 4 + 1] = s * cu + c * su;
    out[j * 4 + 2] = c * cv - s * sv;
    out[j * 4 + 3] = s * cv + c * sv;
  }
  return out;
}

/**
 * The field advected to time `t`: what every point needs, computed once per frame. Per mode, the
 * frequency a fixed point sees it at, `f_j = k1_j·U/L` (Hz), its amplitude times the
 * 10-minute-mean high-pass, and `cos`, `sin` of `−2π·f_j·t`.
 */
export interface TurbulenceClock {
  readonly frequencies: Float64Array;
  readonly weights: Float64Array;
  readonly rotor: Float64Array;
}

export function turbulenceClock(
  field: FrozenTurbulence,
  lengthScaleM: number,
  speedMps: number,
  t: number,
): TurbulenceClock {
  const n = field.count;
  const frequencies = new Float64Array(n);
  const weights = new Float64Array(n);
  const rotor = new Float64Array(n * 2);
  const k = field.wavevectors;
  const scale = speedMps / lengthScaleM;
  const highPass = 1 / MEAN_WIND_PERIOD_S;
  for (let j = 0; j < n; j += 1) {
    const f = scale * (k[j * 3] ?? 0);
    frequencies[j] = f;
    weights[j] = ((field.amplitudes[j] ?? 0) * f) / Math.sqrt(f * f + highPass * highPass);
    const angle = -2 * Math.PI * f * t;
    rotor[j * 2] = Math.cos(angle);
    rotor[j * 2 + 1] = Math.sin(angle);
  }
  return { frequencies, weights, rotor };
}

/** Background samples are softly limited to this many standard deviations of the raw field. */
export const BACKGROUND_SOFT_CLIP = 3;

/** Damping of the background low-pass: the flattest second-order response, no peak. */
const BACKGROUND_DAMPING = Math.SQRT1_2;

/**
 * The quasi-static response of an oscillator at `frequencyHz` to the frozen field advected past
 * a point: along- and across-wind components into `out`, in units of the field's RMS (so at
 * most {@link BACKGROUND_SOFT_CLIP} in magnitude). `phases` is {@link turbulencePhases} for the
 * point, `clock` the frame's {@link turbulenceClock}.
 *
 * Each mode passes through a second-order low-pass at the oscillator's frequency with `ζ = 1/√2`
 * — its static compliance up to its frequency, falling as `f^−4` above like the oscillator's own
 * response, with the oscillator's phase lag, and no peak (the peak is the resonant channel's) —
 * and the clock's high-pass at the 10-minute mean's period.
 */
export function backgroundResponse(
  phases: Float64Array,
  clock: TurbulenceClock,
  frequencyHz: number,
  out: Float64Array,
): void {
  const { frequencies, weights, rotor } = clock;
  const twoZeta = 2 * BACKGROUND_DAMPING;
  const inverse = 1 / frequencyHz;
  let along = 0;
  let across = 0;
  for (let j = 0; j < frequencies.length; j += 1) {
    // H(−ω) for a mode e^{i(k·x − ωt)}: (1 − r² + i·2ζr) / D.
    const r = (frequencies[j] ?? 0) * inverse;
    const re = 1 - r * r;
    const gain = (weights[j] ?? 0) / (re * re + twoZeta * twoZeta * r * r);
    const a = re * gain;
    const b = twoZeta * r * gain;
    // e^{i(2πk·x + φ − 2πft)}, then Re[(a + ib)·that] = a·cos − b·sin.
    const c = rotor[j * 2] ?? 1;
    const s = rotor[j * 2 + 1] ?? 0;
    const cu = phases[j * 4] ?? 1;
    const su = phases[j * 4 + 1] ?? 0;
    const cv = phases[j * 4 + 2] ?? 1;
    const sv = phases[j * 4 + 3] ?? 0;
    along += a * (cu * c - su * s) - b * (su * c + cu * s);
    across += a * (cv * c - sv * s) - b * (sv * c + cv * s);
  }
  out[0] = BACKGROUND_SOFT_CLIP * Math.tanh(along / BACKGROUND_SOFT_CLIP);
  out[1] = BACKGROUND_SOFT_CLIP * Math.tanh(across / BACKGROUND_SOFT_CLIP);
}
