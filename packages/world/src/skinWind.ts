/**
 * Wind on scene-object skins (step C1 of docs/LIVING_PLAN.md; the skins are SCENE_OBJECTS.md
 * §4): a reduced-order linear model per skinned instance whose output is the handles
 * `Z_j` the viewer's skinning path draws (`x' = x + Σ_j w_j Z_j [x − origin; 1]`).
 *
 * ### Coordinates
 *
 * Each handle `j` (the constant one included) carries a horizontal **translation** `q_j`:
 * `Z_j = [0 | q_j]`, so a splat moves by `u(x) = Σ_j w_j(x) q_j` (`w_0 ≡ 1`). Translations are
 * what wind does to a plant at small strain; the handles' linear parts stay zero, so the
 * covariances are drawn as measured. Vertical motion is not driven (the wind has no updraft).
 *
 * ### Mass, stiffness, load (from `skin.json`)
 *
 * - **Mass** `M_ij = mean over the object's splats of w_i w_j` (`dynamics.mass`): the kinetic
 *   energy of `u` is `½ q̇ᵀ M q̇` per unit mass.
 * - **Stiffness** `K = (c / scale)² · diag(0, λ_1 M_11, …, λ_{m−1} M_{m−1,m−1})`. The handles
 *   are the eigenmodes of `H c = λ M c` on the object mapped into a unit box with `E = 1`, so a
 *   real material of wave speed `c = √(E/ρ)` gives `ω_j² = c² λ_j / scale²` (the box's size is
 *   `2·scale` up to the fit's 2% pad, absorbed into `c`). The modes are `M`-orthogonal, so `K` is
 *   diagonal; the constant handle has none. `c` is the **stiffness** of the material record
 *   (`SkinMaterial.stiffness`, m/s).
 * - **Load**: a uniform drag `f` drives handle `j` by `M_0j f` (its mean weight). A gust that
 *   varies across the object drives handle `j` by what it feels at its own support centre
 *   beyond the mean: `F_j = (D / scale)(M_0j a_0 + M_jj (a_j − a_0))`, `a_j = |v_j| v_j` the
 *   wind velocity's square at handle `j`'s support centre (`a_0` at the object's mid-height),
 *   `D` the **drag** number (dimensionless; drag per unit mass falls as `1/size`). That is the first-order expansion `f(x) ≈ f_0 + Σ_j w_j (f_j − f_0)`
 *   with `M` taken diagonal for the learned handles, as the eigenmodes make it.
 *
 * ### The anchor
 *
 * The eigenmodes are free: on their own they move an object's base as well as its crown. What
 * holds it is data: `dynamics.anchor.gram`, `G_ij = mean over its anchor splats of w_i w_j`,
 * the anchor splats being the lowest tenth of its height (`skin_scene.anchor_mask`). The
 * generalized eigenproblem `G y = μ M y` orders handle-space directions by how much of their
 * motion reaches the anchor (`√μ`: rms displacement there over rms displacement of the whole
 * object); the directions with `√μ ≤ ANCHOR_TOLERANCE` span the motions the object may make.
 * Projecting `M` and `K` onto them (Galerkin) and diagonalising gives the anchored modes:
 * shapes `Φ` (`m × r`, `M`-orthonormal), frequencies `Ω_i`. This is a least-squares projection
 * in handle space, computed once per skin; the constant handle is what cancels a learned
 * mode's motion at the base.
 *
 * ### Integration
 *
 * Each anchored mode is a linear oscillator `s̈ + 2ζΩ ṡ + Ω² s = Φᵀ F(t)`, advanced on a fixed
 * grid of `SKIN_WIND_STEP_S` by its exact discrete-time solution with the force held over the
 * step and sampled at its middle: unconditionally stable, and the state at a grid time does not
 * depend on the frame rate at all (frames in between interpolate). The force is the Living
 * Survey's wind: the mean speed `U = speedFromStrength(strength)` (`living.ts`), turbulence
 * from the same frozen EN 1991-1-4 field (`turbulence.ts`) sampled at each handle's support
 * centre, intensity `I(z)` at its height, `v = U((1 + I a) ŵ + 0.75 I b ĉ)`.
 *
 * ### Bounded
 *
 * The skin work found the shape folds once a handle moves beyond its support radius, so the
 * output is softly limited: `ρ = max_j |q_j| / (HANDLE_REACH · r_j)` and every `q` scaled by
 * `tanh(ρ)/ρ`, one factor for the whole object so the anchor still holds. No handle moves
 * more than `HANDLE_REACH` of its support radius (handle 0: of the object's `scale`).
 *
 * Pure: everything is a function of the skin, the material, the seed and the scene time.
 */

import { speedFromStrength } from "./living";
import type { MotionEvidence } from "./motionParams";
import {
  cholesky,
  backSubstituteTransposed,
  forwardSubstitute,
  symmetricEigen,
} from "./symmetricEigen";
import {
  BACKGROUND_SOFT_CLIP,
  frozenTurbulence,
  LATERAL_TURBULENCE_RATIO,
  TERRAIN_MIN_HEIGHT_M,
  turbulenceClock,
  turbulenceIntensity,
  turbulenceLengthScaleM,
  turbulencePhases,
  type FrozenTurbulence,
  type TurbulenceClock,
} from "./turbulence";
import { DEG_TO_RAD, type Vec3 } from "./vec";
import type { WindSettings } from "./wind";

// ---------------------------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------------------------

/**
 * Wave speed `c` of a vegetation-like material, m/s: `ω_j = c·√λ_j / scale`. Calibrated so the
 * yard's 9.7 m tree (`scale` 4.84, `λ_1` 12.2) sways at 0.4 Hz and its 1.9 m shrub at 3.7 Hz,
 * within the 0.2–0.6 Hz of measured conifers and broadleaves that height and the 1–4 Hz of
 * shrubs. A prior, not a measurement: what the video teacher (C2) overwrites.
 */
export const SKIN_WAVE_SPEED_MPS = 3.5;
/** How much stiffer than vegetation a fully rigid, inelastic object is (frequency ratio). */
export const SKIN_RIGID_STIFFENING = 4;
/** Damping ratio of an object with no foliage; foliage adds `SKIN_FOLIAGE_DAMPING`. */
export const SKIN_BASE_DAMPING = 0.05;
export const SKIN_FOLIAGE_DAMPING = 0.05;
/**
 * Drag number of foliage, dimensionless: a handle's acceleration is `drag · |v|v / scale`. Drag
 * goes with area and mass with volume, so per unit mass it falls as `1/size`; with `ω ∝ 1/size`
 * that makes the sway a fixed share of an object's size, other things equal. Calibrated so
 * the yard's 9.7 m tree sways about 0.1 m rms over its splats at the default strength (0.1,
 * 6.3 m/s); a fully bare object takes a quarter of it.
 */
export const SKIN_DRAG = 0.025;
/** Largest damping ratio a record may carry: the exact integrator here is the underdamped one. */
export const SKIN_MAX_DAMPING = 0.95;

/** How much evidence stands behind a material: the motion ladder, below it the prior. */
export type MaterialEvidence = MotionEvidence | "prior";

/**
 * One instance's material: what the video teacher (C2) fits and `materials.json` stores
 * (SCENE_OBJECTS.md §4).
 */
export interface SkinMaterial {
  /** Wave speed `c`, m/s: `ω_j = c·√λ_j / scale`. */
  readonly stiffness: number;
  /** Damping ratio `ζ` of every mode, `0 … SKIN_MAX_DAMPING`. */
  readonly damping: number;
  /** Drag number `D`, dimensionless: acceleration `D · |v|v / scale`. */
  readonly drag: number;
  /** Whether the wind drives it at all. */
  readonly wind: boolean;
  readonly evidence: MaterialEvidence;
}

function score(properties: Readonly<Record<string, number>>, name: string): number {
  const v = properties[name];
  return typeof v === "number" && Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0;
}

/**
 * The material an instance gets before anything is fitted, from its property scores
 * (`instances.json`), never its class:
 *
 * - `wind`: behaviour `in-place` (a movable object -- a vehicle, a person -- is not swayed,
 *   and a static one has no skin).
 * - softness `σ = clamp(max(vegetation, elastic) − rigid/2, 0, 1)` (0.5 without properties);
 *   `stiffness = SKIN_WAVE_SPEED_MPS · SKIN_RIGID_STIFFENING^(1 − σ)`.
 * - `damping = SKIN_BASE_DAMPING + SKIN_FOLIAGE_DAMPING · vegetation`.
 * - `drag = SKIN_DRAG · (0.25 + 0.75 · vegetation)`: foliage catches the wind.
 */
export function materialPrior(
  properties: Readonly<Record<string, number>> | undefined,
  behaviour: string | undefined,
): SkinMaterial {
  const p = properties ?? {};
  const known = Object.keys(p).length > 0;
  const vegetation = score(p, "vegetation");
  const softness = known
    ? Math.min(1, Math.max(0, Math.max(vegetation, score(p, "elastic")) - 0.5 * score(p, "rigid")))
    : 0.5;
  return {
    stiffness: SKIN_WAVE_SPEED_MPS * Math.pow(SKIN_RIGID_STIFFENING, 1 - softness),
    damping: SKIN_BASE_DAMPING + SKIN_FOLIAGE_DAMPING * vegetation,
    drag: SKIN_DRAG * (0.25 + 0.75 * vegetation),
    wind: behaviour === "in-place",
    evidence: "prior",
  };
}

/** `override`'s valid fields over `base`: a fitted record may set any subset. */
export function mergeMaterial(
  base: SkinMaterial,
  override: Partial<SkinMaterial> | undefined,
): SkinMaterial {
  if (!override) return base;
  const positive = (v: unknown, fallback: number): number =>
    typeof v === "number" && Number.isFinite(v) && v > 0 ? v : fallback;
  const damping =
    typeof override.damping === "number" && Number.isFinite(override.damping)
      ? Math.min(SKIN_MAX_DAMPING, Math.max(0, override.damping))
      : base.damping;
  return {
    stiffness: positive(override.stiffness, base.stiffness),
    damping,
    drag:
      typeof override.drag === "number" && Number.isFinite(override.drag) && override.drag >= 0
        ? override.drag
        : base.drag,
    wind: typeof override.wind === "boolean" ? override.wind : base.wind,
    evidence: override.evidence ?? base.evidence,
  };
}

// ---------------------------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------------------------

/** Anchored directions keep the base's rms motion within this share of the object's. */
export const ANCHOR_TOLERANCE = 0.05;
/** No handle moves more than this share of its support radius. */
export const HANDLE_REACH = 0.25;
/** The fixed integration step, seconds. */
export const SKIN_WIND_STEP_S = 1 / 60;
/** A longer gap in scene time (or any step back) restarts from the equilibrium, seconds. */
export const SKIN_WIND_MAX_CATCHUP_S = 0.5;
/** Modes of the frozen turbulence field the skins read (the rigs read 320). */
export const SKIN_WIND_FIELD_MODES = 64;
/** The field passes everything below this to the oscillators, Hz. */
const FIELD_CUTOFF_HZ = 25;
/** Anchored modes slower than this share of the first handle's frequency are dropped. */
const MIN_MODE_RATIO = 0.25;
const TWELVE = 12;
const EMPTY = new Float64Array(0);

/** What a skin carries for its dynamics (`skin.json`, SCENE_OBJECTS.md §4). */
export interface SkinDynamicsSource {
  /** `m`, the constant handle included. */
  readonly handles: number;
  /** Rest-frame origin, tileset local ENU metres: the object's base. */
  readonly origin: Vec3;
  /** Half its largest extent, metres. */
  readonly scale: number;
  /** Per learned handle `1..m−1`. */
  readonly eigenvalues: readonly number[];
  /** Per learned handle: where it acts (rest frame, from `origin`) and how far around. */
  readonly support: readonly { readonly centre: Vec3; readonly radius: number }[];
  /** `M`, upper triangle row by row (`m(m+1)/2`). */
  readonly mass: readonly number[];
  /** `G` over the anchor splats, the same layout. */
  readonly anchorGram: readonly number[];
}

export interface SkinWindModel {
  /** `m`. */
  readonly handles: number;
  /** `r`, anchored modes. */
  readonly modes: number;
  /** `Φ`, row-major `m × r`. */
  readonly shapes: Float64Array;
  /** `Ω_i`, rad/s, ascending. */
  readonly omega: Float64Array;
  readonly damping: number;
  /** `D / scale`, 1/m: acceleration per (m/s)² of wind. */
  readonly drag: number;
  /** `M_0j`: each handle's mean weight. */
  readonly load: Float64Array;
  /** `M_jj`. */
  readonly selfMass: Float64Array;
  /** Where each handle reads the wind, tileset local ENU: `m × 3`. */
  readonly points: Float64Array;
  /** Turbulence intensity at each point's height above the base. */
  readonly intensity: Float64Array;
  /** Per handle, the most it may move, metres. */
  readonly limits: Float64Array;
  /** The largest `√μ` kept: the base's rms motion over the object's, at worst. */
  readonly anchorResidual: number;
}

/** A symmetric `m × m` from its upper triangle, or undefined when the length is wrong. */
export function fromUpper(upper: readonly number[], m: number): Float64Array | undefined {
  if (upper.length !== (m * (m + 1)) / 2) return undefined;
  const out = new Float64Array(m * m);
  let k = 0;
  for (let i = 0; i < m; i += 1) {
    for (let j = i; j < m; j += 1) {
      const v = upper[k] ?? 0;
      out[i * m + j] = v;
      out[j * m + i] = v;
      k += 1;
    }
  }
  return out;
}

/**
 * The anchored modal model of one skin under one material, or undefined when the skin carries
 * no dynamics (an old `skin.json`: re-run `skin_scene.py`), is malformed, or has no direction
 * that keeps its base still.
 */
export function skinWindModel(
  source: SkinDynamicsSource,
  material: SkinMaterial,
): SkinWindModel | undefined {
  const m = source.handles;
  if (!(m >= 2) || source.eigenvalues.length < m - 1 || source.support.length < m - 1)
    return undefined;
  const mass = fromUpper(source.mass, m);
  const gram = fromUpper(source.anchorGram, m);
  if (!mass || !gram || !(source.scale > 0)) return undefined;
  // M = L·Lᵀ (a ridge for the rounding of five-digit files), then C = L⁻¹ G L⁻ᵀ.
  let trace = 0;
  for (let i = 0; i < m; i += 1) trace += mass[i * m + i] ?? 0;
  const ridged = Float64Array.from(mass);
  for (let i = 0; i < m; i += 1) ridged[i * m + i] = (ridged[i * m + i] ?? 0) + 1e-9 * trace;
  const l = cholesky(ridged, m);
  if (!l) return undefined;
  const x = forwardSubstitute(l, m, gram, m);
  const xt = new Float64Array(m * m);
  for (let i = 0; i < m; i += 1) for (let j = 0; j < m; j += 1) xt[i * m + j] = x[j * m + i] ?? 0;
  const c = forwardSubstitute(l, m, xt, m);
  const { values: mu, vectors: y } = symmetricEigen(c, m);
  let kept = 0;
  let residual = 0;
  while (kept < m && (mu[kept] ?? Infinity) <= ANCHOR_TOLERANCE * ANCHOR_TOLERANCE) {
    residual = Math.sqrt(Math.max(0, mu[kept] ?? 0));
    kept += 1;
  }
  if (kept === 0) return undefined;
  // B = L⁻ᵀ Y (first `kept` columns): M-orthonormal, so Bᵀ M B = I.
  const yk = new Float64Array(m * kept);
  for (let i = 0; i < m; i += 1)
    for (let k = 0; k < kept; k += 1) yk[i * kept + k] = y[i * m + k] ?? 0;
  const b = backSubstituteTransposed(l, m, yk, kept);
  // K, then K_r = Bᵀ K B.
  const k2 = (material.stiffness / source.scale) ** 2;
  const stiffness = new Float64Array(m);
  for (let j = 1; j < m; j += 1)
    stiffness[j] = k2 * Math.max(0, source.eigenvalues[j - 1] ?? 0) * (mass[j * m + j] ?? 0);
  const kr = new Float64Array(kept * kept);
  for (let p = 0; p < kept; p += 1) {
    for (let q = p; q < kept; q += 1) {
      let sum = 0;
      for (let j = 0; j < m; j += 1)
        sum += (b[j * kept + p] ?? 0) * (stiffness[j] ?? 0) * (b[j * kept + q] ?? 0);
      kr[p * kept + q] = sum;
      kr[q * kept + p] = sum;
    }
  }
  const { values: omega2, vectors: u } = symmetricEigen(kr, kept);
  const first = Math.sqrt(k2 * Math.max(0, source.eigenvalues[0] ?? 0));
  const modes: number[] = [];
  for (let i = 0; i < kept; i += 1) {
    if (Math.sqrt(Math.max(0, omega2[i] ?? 0)) >= MIN_MODE_RATIO * first) modes.push(i);
  }
  const r = modes.length;
  const shapes = new Float64Array(m * r);
  const omega = new Float64Array(r);
  modes.forEach((mode, col) => {
    omega[col] = Math.sqrt(omega2[mode] ?? 0);
    for (let j = 0; j < m; j += 1) {
      let sum = 0;
      for (let p = 0; p < kept; p += 1) sum += (b[j * kept + p] ?? 0) * (u[p * kept + mode] ?? 0);
      shapes[j * r + col] = sum;
    }
  });
  const load = new Float64Array(m);
  const selfMass = new Float64Array(m);
  const points = new Float64Array(m * 3);
  const intensity = new Float64Array(m);
  const limits = new Float64Array(m);
  const [ox, oy, oz] = source.origin;
  for (let j = 0; j < m; j += 1) {
    load[j] = mass[j] ?? 0;
    selfMass[j] = mass[j * m + j] ?? 0;
    const support = j === 0 ? undefined : source.support[j - 1];
    const centre: Vec3 = support?.centre ?? [0, 0, source.scale];
    points[j * 3] = ox + centre[0];
    points[j * 3 + 1] = oy + centre[1];
    points[j * 3 + 2] = oz + centre[2];
    intensity[j] = turbulenceIntensity(Math.max(0, centre[2]));
    limits[j] = HANDLE_REACH * (support ? Math.max(support.radius, 1e-3) : source.scale);
  }
  return {
    handles: m,
    modes: r,
    shapes,
    omega,
    damping: Math.min(SKIN_MAX_DAMPING, Math.max(0, material.damping)),
    drag: Math.max(0, material.drag) / source.scale,
    load,
    selfMass,
    points,
    intensity,
    limits,
    anchorResidual: residual,
  };
}

// ---------------------------------------------------------------------------------------------
// The wind field
// ---------------------------------------------------------------------------------------------

/** The wind skins read: the scene's settings as a speed and a bearing. */
export interface SkinWind {
  /** Mean speed, m/s; 0 is calm. */
  readonly speedMps: number;
  /** Downwind bearing, degrees clockwise from north. */
  readonly bearingDeg: number;
  /** Scales the turbulence intensity: 1 (the default) is EN 1991-1-4's, 0 a steady wind. */
  readonly turbulence?: number;
}

/** The scene's `WindSettings` as the speed the Living Survey reads them at (`speedFromStrength`). */
export function skinWindFromSettings(settings: WindSettings): SkinWind {
  return {
    speedMps: speedFromStrength(settings.strength),
    bearingDeg: Number.isFinite(settings.bearingDeg) ? settings.bearingDeg : 0,
  };
}

/**
 * One frozen turbulence field shared by every skin in a scene, so a gust crosses from object to
 * object at the mean speed. Deterministic from its seed.
 */
export class SkinWindField {
  readonly field: FrozenTurbulence;
  readonly lengthScaleM: number;
  readonly #frames = new Map<string, SkinWindFrame>();

  constructor(
    seed: number,
    lengthScaleM = turbulenceLengthScaleM(TERRAIN_MIN_HEIGHT_M),
    modes = SKIN_WIND_FIELD_MODES,
  ) {
    this.field = frozenTurbulence(seed, modes);
    this.lengthScaleM = lengthScaleM;
  }

  /** The field's phases at `point` (tileset local ENU) for a wind towards `bearingDeg`. */
  phases(point: Vec3, bearingDeg: number): Float64Array {
    const b = bearingDeg * DEG_TO_RAD;
    return turbulencePhases(this.field, point, [Math.sin(b), Math.cos(b), 0], this.lengthScaleM);
  }

  /**
   * The field advected to `t` at `speedMps`, folded for sampling: per mode `P = a·c − b·s` and
   * `Q = a·s + b·c`, `(a, b)` the mode's weight through the low-pass `backgroundResponse`
   * applies (at `FIELD_CUTOFF_HZ`) and `(c, s)` its rotor. Cached for a few instants: every
   * oscillator steps the same grid.
   */
  frame(t: number, speedMps: number): SkinWindFrame {
    const key = `${String(t)}|${String(speedMps)}`;
    let frame = this.#frames.get(key);
    if (frame === undefined) {
      if (this.#frames.size >= 16) this.#frames.clear();
      frame = foldClock(turbulenceClock(this.field, this.lengthScaleM, speedMps, t));
      this.#frames.set(key, frame);
    }
    return frame;
  }
}

/** A {@link SkinWindField} at one instant: two numbers a mode. */
export interface SkinWindFrame {
  readonly p: Float64Array;
  readonly q: Float64Array;
}

/** `backgroundResponse`'s per-mode arithmetic, done once for every point. */
function foldClock(clock: TurbulenceClock): SkinWindFrame {
  const { frequencies, weights, rotor } = clock;
  const n = frequencies.length;
  const p = new Float64Array(n);
  const q = new Float64Array(n);
  const twoZeta = 2 * Math.SQRT1_2;
  for (let j = 0; j < n; j += 1) {
    const r = (frequencies[j] ?? 0) / FIELD_CUTOFF_HZ;
    const re = 1 - r * r;
    const gain = (weights[j] ?? 0) / (re * re + twoZeta * twoZeta * r * r);
    const a = re * gain;
    const b = twoZeta * r * gain;
    const c = rotor[j * 2] ?? 1;
    const s = rotor[j * 2 + 1] ?? 0;
    p[j] = a * c - b * s;
    q[j] = a * s + b * c;
  }
  return { p, q };
}

/**
 * The field at a point (its `phases`) in units of its rms, along and across the wind, softly
 * clipped at `BACKGROUND_SOFT_CLIP` as `backgroundResponse` does (and equal to it).
 */
export function sampleSkinWind(
  phases: Float64Array,
  frame: SkinWindFrame,
  out: Float64Array,
): Float64Array {
  const { p, q } = frame;
  let along = 0;
  let across = 0;
  for (let j = 0; j < p.length; j += 1) {
    const pj = p[j] ?? 0;
    const qj = q[j] ?? 0;
    along += (phases[j * 4] ?? 1) * pj - (phases[j * 4 + 1] ?? 0) * qj;
    across += (phases[j * 4 + 2] ?? 1) * pj - (phases[j * 4 + 3] ?? 0) * qj;
  }
  out[0] = BACKGROUND_SOFT_CLIP * Math.tanh(along / BACKGROUND_SOFT_CLIP);
  out[1] = BACKGROUND_SOFT_CLIP * Math.tanh(across / BACKGROUND_SOFT_CLIP);
  return out;
}

// ---------------------------------------------------------------------------------------------
// The oscillator
// ---------------------------------------------------------------------------------------------

/** Exact one-step transition of `s̈ + 2ζΩṡ + Ω²s = f` (f constant) over `h`. */
function transition(omega: number, zeta: number, h: number): [number, number, number, number] {
  const a = zeta * omega;
  const wd = omega * Math.sqrt(Math.max(1e-12, 1 - zeta * zeta));
  const e = Math.exp(-a * h);
  const c = Math.cos(wd * h);
  const s = Math.sin(wd * h);
  return [
    e * (c + (a / wd) * s),
    (e * s) / wd,
    (-e * omega * omega * s) / wd,
    e * (c - (a / wd) * s),
  ];
}

/**
 * One instance's state: per anchored mode, the east and north modal displacements and
 * velocities, on the fixed grid `k · SKIN_WIND_STEP_S`.
 */
export class SkinWindOscillator {
  readonly model: SkinWindModel;
  /** Per mode: east then north. */
  readonly #x: Float64Array;
  readonly #v: Float64Array;
  readonly #previous: Float64Array;
  readonly #coefficients: Float64Array;
  #step: number | undefined;
  #phases: Float64Array[] = [];
  #bearing = Number.NaN;
  #field: SkinWindField | undefined;
  readonly #force: Float64Array;
  readonly #sample = new Float64Array(2);

  constructor(model: SkinWindModel) {
    this.model = model;
    const r = model.modes;
    this.#x = new Float64Array(2 * r);
    this.#v = new Float64Array(2 * r);
    this.#previous = new Float64Array(2 * r);
    this.#force = new Float64Array(2 * r);
    this.#coefficients = new Float64Array(4 * r);
    for (let i = 0; i < r; i += 1)
      this.#coefficients.set(
        transition(model.omega[i] ?? 0, model.damping, SKIN_WIND_STEP_S),
        4 * i,
      );
  }

  /** The scene time the state stands at, or undefined at rest. */
  get time(): number | undefined {
    return this.#step === undefined ? undefined : this.#step * SKIN_WIND_STEP_S;
  }

  /** Back to rest: the next `advance` starts from still. */
  reset(): void {
    this.#step = undefined;
    this.#x.fill(0);
    this.#v.fill(0);
    this.#previous.fill(0);
  }

  /**
   * Advances the state to scene time `t` under `wind`. From rest when it was at rest; from the
   * equilibrium under the wind at `t` after a jump in scene time (or a step back). Calm resets.
   */
  advance(field: SkinWindField, wind: SkinWind, t: number): void {
    if (!(wind.speedMps > 0) || !Number.isFinite(t) || this.model.modes === 0) {
      this.reset();
      return;
    }
    if (wind.bearingDeg !== this.#bearing || field !== this.#field) {
      this.#bearing = wind.bearingDeg;
      this.#field = field;
      const p = this.model.points;
      this.#phases = Array.from({ length: this.model.handles }, (_, j) =>
        field.phases([p[j * 3] ?? 0, p[j * 3 + 1] ?? 0, p[j * 3 + 2] ?? 0], wind.bearingDeg),
      );
    }
    const h = SKIN_WIND_STEP_S;
    const target = Math.floor(t / h + 1e-9);
    if (this.#step === undefined) {
      this.#step = target;
      this.#x.fill(0);
      this.#v.fill(0);
      this.#previous.fill(0);
    } else if (t < (this.#step - 1) * h || t - this.#step * h > SKIN_WIND_MAX_CATCHUP_S) {
      this.#step = target;
      this.#forces(field, wind, target * h);
      const r = this.model.modes;
      for (let i = 0; i < r; i += 1) {
        const w2 = (this.model.omega[i] ?? 1) ** 2;
        this.#x[2 * i] = (this.#force[2 * i] ?? 0) / w2;
        this.#x[2 * i + 1] = (this.#force[2 * i + 1] ?? 0) / w2;
      }
      this.#v.fill(0);
      this.#previous.set(this.#x);
    }
    while (this.#step * h < t - 1e-9) {
      this.#forces(field, wind, (this.#step + 0.5) * h);
      this.#previous.set(this.#x);
      const r = this.model.modes;
      for (let i = 0; i < r; i += 1) {
        const w2 = (this.model.omega[i] ?? 1) ** 2;
        const a11 = this.#coefficients[4 * i] ?? 0;
        const a12 = this.#coefficients[4 * i + 1] ?? 0;
        const a21 = this.#coefficients[4 * i + 2] ?? 0;
        const a22 = this.#coefficients[4 * i + 3] ?? 0;
        for (let axis = 0; axis < 2; axis += 1) {
          const k = 2 * i + axis;
          const equilibrium = (this.#force[k] ?? 0) / w2;
          const dx = (this.#x[k] ?? 0) - equilibrium;
          const v = this.#v[k] ?? 0;
          this.#x[k] = equilibrium + a11 * dx + a12 * v;
          this.#v[k] = a21 * dx + a22 * v;
        }
      }
      this.#step += 1;
    }
  }

  /** The modal forces `Φᵀ F` at scene time `t`, into `#force`. */
  #forces(field: SkinWindField, wind: SkinWind, t: number): void {
    const model = this.model;
    const { handles: m, modes: r, shapes } = model;
    const turbulence = Math.max(0, wind.turbulence ?? 1);
    const frame = turbulence > 0 ? field.frame(t, wind.speedMps) : undefined;
    const b = wind.bearingDeg * DEG_TO_RAD;
    const de = Math.sin(b);
    const dn = Math.cos(b);
    const u = wind.speedMps;
    this.#force.fill(0);
    this.#sample.fill(0);
    let a0e = 0;
    let a0n = 0;
    for (let j = 0; j < m; j += 1) {
      if (frame) sampleSkinWind(this.#phases[j] ?? EMPTY, frame, this.#sample);
      const intensity = turbulence * (model.intensity[j] ?? 0);
      const along = u * (1 + intensity * (this.#sample[0] ?? 0));
      const across = u * LATERAL_TURBULENCE_RATIO * intensity * (this.#sample[1] ?? 0);
      // Downwind (de, dn); across, 90° clockwise from it: (dn, −de).
      const ve = along * de + across * dn;
      const vn = along * dn - across * de;
      const speed = Math.hypot(ve, vn);
      let ae = speed * ve;
      let an = speed * vn;
      if (j === 0) {
        a0e = ae;
        a0n = an;
      } else {
        const load = model.load[j] ?? 0;
        const self = model.selfMass[j] ?? 0;
        ae = load * a0e + self * (ae - a0e);
        an = load * a0n + self * (an - a0n);
      }
      for (let i = 0; i < r; i += 1) {
        const phi = (shapes[j * r + i] ?? 0) * model.drag;
        this.#force[2 * i] = (this.#force[2 * i] ?? 0) + phi * ae;
        this.#force[2 * i + 1] = (this.#force[2 * i + 1] ?? 0) + phi * an;
      }
    }
  }

  /**
   * The handles at scene time `t` (interpolated between the last two grid states), bounded:
   * `12·m` numbers, `Z_j = [0 | q_j]` row-major, into `out`. All zeros at rest.
   */
  handles(
    t: number,
    out: Float64Array = new Float64Array(this.model.handles * TWELVE),
  ): Float64Array {
    out.fill(0);
    const { handles: m, modes: r, shapes, limits } = this.model;
    if (this.#step === undefined || r === 0) return out;
    const h = SKIN_WIND_STEP_S;
    const alpha = Math.min(1, Math.max(0, (t - (this.#step - 1) * h) / h));
    let ratio = 0;
    for (let j = 0; j < m; j += 1) {
      let qe = 0;
      let qn = 0;
      for (let i = 0; i < r; i += 1) {
        const phi = shapes[j * r + i] ?? 0;
        const pe = this.#previous[2 * i] ?? 0;
        const pn = this.#previous[2 * i + 1] ?? 0;
        qe += phi * (pe + alpha * ((this.#x[2 * i] ?? 0) - pe));
        qn += phi * (pn + alpha * ((this.#x[2 * i + 1] ?? 0) - pn));
      }
      out[j * TWELVE + 3] = qe;
      out[j * TWELVE + 7] = qn;
      ratio = Math.max(ratio, Math.hypot(qe, qn) / (limits[j] ?? 1));
    }
    const factor = ratio > 1e-12 ? Math.tanh(ratio) / ratio : 1;
    if (factor !== 1) for (let k = 0; k < out.length; k += 1) out[k] = (out[k] ?? 0) * factor;
    return out;
  }
}

/** Each handle's translation `|q_j|` in `handles` (12 numbers a handle), metres. */
export function handleReach(handles: ArrayLike<number>, count: number): Float64Array {
  const out = new Float64Array(count);
  for (let j = 0; j < count; j += 1) {
    out[j] = Math.hypot(
      handles[j * TWELVE + 3] ?? 0,
      handles[j * TWELVE + 7] ?? 0,
      handles[j * TWELVE + 11] ?? 0,
    );
  }
  return out;
}
