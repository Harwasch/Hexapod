/**
 * Poke and drag on scene-object skins (docs/SCENE_OBJECTS.md §9, "The poke driver"): press on a
 * skinned object, drag, and a spring at the grabbed splat pulls its handles; let go and it
 * rings down at its own frequencies. The output is the same `Z_j = [0 | q_j]` handle
 * translations the wind writes (`skinWind.ts`), added on top of them by the skin part.
 *
 * ### The modes
 *
 * - **Rooted** (behaviour `in-place`, or anything not movable): the wind's anchored modes
 *   (`skinWindModel`): the handle-space directions that leave the base still, `M`-orthonormal
 *   shapes `Φ` (`m × r`) and frequencies `Ω_i = c·√λ / scale` of the material.
 * - **Movable**: every handle takes part. The constant handle (the whole object) gets a soft
 *   spring home, `POKE_RETURN_HZ` (there is no ground or gravity yet: the rigid-body layer is
 *   P4), and the learned handles their elastic stiffness; `M q̈ + K q` diagonalised whole.
 *
 * ### The spring
 *
 * A grab at a splat of weights `w` (`w_0 = 1`) pulls its displacement `u = Σ_j w_j q_j` towards
 * the cursor's offset `d` (on the plane through the grabbed point facing the camera) with an
 * acceleration `κ (d − u)` per unit of the object's mass: `κ = (2π·POKE_SPRING_HZ)²`. In modal
 * coordinates `s` (per axis, east, north and up alike), `u = bᵀ s`, `b = Φᵀ w`, and
 *
 *     s̈ + 2ζΩ ṡ + Ω² s = κ b (d − bᵀ s)
 *
 * so a soft object (its `Ω` well below the spring's) follows the cursor and a stiff one barely
 * gives: you can pull a branch, not a pumpkin -- though a movable pumpkin slides whole. `d` is
 * capped at `POKE_MAX_PULL` of the object's size.
 *
 * ### Integration
 *
 * Newmark's average-acceleration rule (the trapezoidal rule: unconditionally stable, no
 * numerical damping, a frequency error under 0.4% at `Ω h < 0.2`) on a fixed grid of
 * `POKE_STEP_S`; the spring is a rank-one term, so each step is a diagonal solve plus
 * Sherman-Morrison. Released, the modes are uncoupled again and ring at `Ω_i √(1 − ζ²)`.
 * Once everything is within `POKE_REST_SHARE` of the object's size and nearly still, the
 * state is dropped and the handles are `null`: the measured frame, exactly.
 *
 * ### Bounded
 *
 * As the wind: `ρ = max_j |q_j| / limit_j`, every `q` scaled by `tanh(ρ)/ρ`, one factor for
 * the object; the limits are `POKE_REACH` of each handle's support radius (the shape folds
 * beyond it) and, for a movable object's constant handle, its size.
 *
 * Pure and deterministic: a function of the skin, the material and the times and targets fed.
 */

import {
  HANDLE_REACH,
  skinWindModel,
  fromUpper,
  type SkinDynamicsSource,
  type SkinMaterial,
} from "./skinWind";
import {
  backSubstituteTransposed,
  cholesky,
  forwardSubstitute,
  symmetricEigen,
} from "./symmetricEigen";

/** The grab spring's own frequency, Hz: an object much softer than this follows the cursor. */
export const POKE_SPRING_HZ = 3;
/** A released movable object springs home at this frequency, Hz, damped by `POKE_RETURN_DAMPING`. */
export const POKE_RETURN_HZ = 0.8;
export const POKE_RETURN_DAMPING = 0.6;
/** The fixed integration step, seconds. */
export const POKE_STEP_S = 1 / 120;
/** A longer gap than this (a hidden tab) is not caught up: the state jumps to now. */
export const POKE_MAX_CATCHUP_S = 0.25;
/** How far the cursor may pull, as a share of the object's half-size (`scale`). */
export const POKE_MAX_PULL = 0.6;
/** No handle moves more than this share of its support radius. */
export const POKE_REACH = 0.35;
/** At rest once every handle is within this share of the object's size and nearly still. */
export const POKE_REST_SHARE = 1e-3;
const TWELVE = 12;
const AXES = 3;

/** One skin's modes for poking. */
export interface SkinPokeModel {
  readonly handles: number;
  readonly modes: number;
  /** `Φ`, row-major `m × r`, `M`-orthonormal. */
  readonly shapes: Float64Array;
  /** `Ω_i`, rad/s. */
  readonly omega: Float64Array;
  /** `ζ_i`. */
  readonly damping: Float64Array;
  /** Per handle, the most it may move, metres. */
  readonly limits: Float64Array;
  /** The object's half-size, metres. */
  readonly scale: number;
  /** Whether the whole object moves (its constant handle takes part). */
  readonly movable: boolean;
}

/**
 * The poke model of one skin under one material, or undefined when nothing of it can move: a
 * rooted rigid object (one handle), or a skin without dynamics.
 */
export function skinPokeModel(
  source: SkinDynamicsSource,
  material: SkinMaterial,
  movable: boolean,
): SkinPokeModel | undefined {
  const m = source.handles;
  if (!(m >= 1) || !(source.scale > 0)) return undefined;
  const limits = new Float64Array(m);
  for (let j = 0; j < m; j += 1) {
    const support = j === 0 ? undefined : source.support[j - 1];
    limits[j] =
      j === 0
        ? movable
          ? source.scale
          : HANDLE_REACH * source.scale
        : POKE_REACH * Math.max(support?.radius ?? 0, 1e-3);
  }
  const zeta = Math.min(0.95, Math.max(0, material.damping));
  if (!movable) {
    const wind = skinWindModel(source, material);
    if (!wind || wind.modes === 0) return undefined;
    return {
      handles: m,
      modes: wind.modes,
      shapes: wind.shapes,
      omega: wind.omega,
      damping: new Float64Array(wind.modes).fill(zeta),
      limits,
      scale: source.scale,
      movable,
    };
  }
  // Movable: M q̈ + K q, K = diag(k_0, k2 λ_j M_jj), k_0 the spring home on the whole object.
  const mass = fromUpper(source.mass, m);
  if (!mass || source.eigenvalues.length < m - 1) return undefined;
  let trace = 0;
  for (let i = 0; i < m; i += 1) trace += mass[i * m + i] ?? 0;
  const ridged = Float64Array.from(mass);
  for (let i = 0; i < m; i += 1) ridged[i * m + i] = (ridged[i * m + i] ?? 0) + 1e-9 * trace;
  const l = cholesky(ridged, m);
  if (!l) return undefined;
  const k2 = (material.stiffness / source.scale) ** 2;
  const home = (2 * Math.PI * POKE_RETURN_HZ) ** 2;
  const stiffness = new Float64Array(m);
  stiffness[0] = home * (mass[0] ?? 1);
  for (let j = 1; j < m; j += 1)
    stiffness[j] = k2 * Math.max(0, source.eigenvalues[j - 1] ?? 0) * (mass[j * m + j] ?? 0);
  // C = L⁻¹ K L⁻ᵀ, eigenvectors Y, shapes B = L⁻ᵀ Y.
  const kd = new Float64Array(m * m);
  for (let j = 0; j < m; j += 1) kd[j * m + j] = stiffness[j] ?? 0;
  const x = forwardSubstitute(l, m, kd, m);
  const xt = new Float64Array(m * m);
  for (let i = 0; i < m; i += 1) for (let j = 0; j < m; j += 1) xt[i * m + j] = x[j * m + i] ?? 0;
  const c = forwardSubstitute(l, m, xt, m);
  for (let i = 0; i < m; i += 1)
    for (let j = i + 1; j < m; j += 1) {
      const v = ((c[i * m + j] ?? 0) + (c[j * m + i] ?? 0)) / 2;
      c[i * m + j] = v;
      c[j * m + i] = v;
    }
  const { values, vectors } = symmetricEigen(c, m);
  const shapes = backSubstituteTransposed(l, m, vectors, m);
  const omega = new Float64Array(m);
  const damping = new Float64Array(m);
  for (let i = 0; i < m; i += 1) {
    omega[i] = Math.sqrt(Math.max(0, values[i] ?? 0));
    // The mode that is mostly the whole object takes the return's damping.
    let rigid = 0;
    let total = 0;
    for (let j = 0; j < m; j += 1) {
      const v = (shapes[j * m + i] ?? 0) ** 2;
      total += v;
      if (j === 0) rigid = v;
    }
    damping[i] = total > 0 && rigid / total > 0.5 ? POKE_RETURN_DAMPING : zeta;
  }
  return { handles: m, modes: m, shapes, omega, damping, limits, scale: source.scale, movable };
}

/** What a grab holds: the grabbed splat's weights `w` (`w_0 = 1` first, `m` numbers). */
export interface PokeGrab {
  readonly weights: ArrayLike<number>;
}

/**
 * One poked skin: its modal state, advanced on a fixed grid from the times it is fed.
 * `grab` / `pull` / `release` while the pointer is down; `advance(t)` each frame; `handles()`
 * the bounded handles to add to the skin's.
 */
export class SkinPoke {
  readonly model: SkinPokeModel;
  /** Per mode and axis: position and velocity (r × 3). */
  readonly #s: Float64Array;
  readonly #v: Float64Array;
  /** `b = Φᵀ w` of the grab, or undefined while released. */
  #b: Float64Array | undefined;
  /** The cursor's pull `d` (metres, scan axes). */
  readonly #pull = new Float64Array(AXES);
  readonly #kappa = (2 * Math.PI * POKE_SPRING_HZ) ** 2;
  /** Grid step index the state is at (time `step · h`), undefined before the first advance. */
  #step: number | undefined;
  #resting = true;

  constructor(model: SkinPokeModel) {
    this.model = model;
    const n = model.modes * AXES;
    this.#s = new Float64Array(n);
    this.#v = new Float64Array(n);
  }

  /** Whether it is held or still moving (false: its handles are `null`). */
  get active(): boolean {
    return this.#b !== undefined || !this.#resting;
  }

  get grabbed(): boolean {
    return this.#b !== undefined;
  }

  /** Takes hold at a splat of weights `grab.weights` (`w_0 = 1` first). */
  grab(grab: PokeGrab): void {
    const { modes: r, handles: m, shapes } = this.model;
    const b = new Float64Array(r);
    for (let i = 0; i < r; i += 1) {
      let sum = 0;
      for (let j = 0; j < m; j += 1) sum += (shapes[j * r + i] ?? 0) * (grab.weights[j] ?? 0);
      b[i] = sum;
    }
    this.#b = b;
    this.#pull.fill(0);
    this.#resting = false;
  }

  /** The cursor's pull from the grabbed splat's rest position, metres (scan axes); capped. */
  pull(d: readonly [number, number, number]): void {
    const cap = POKE_MAX_PULL * this.model.scale;
    const length = Math.hypot(d[0], d[1], d[2]);
    const f = length > cap ? cap / length : 1;
    this.#pull[0] = d[0] * f;
    this.#pull[1] = d[1] * f;
    this.#pull[2] = d[2] * f;
  }

  /** Lets go: it rings down from where it is. */
  release(): void {
    this.#b = undefined;
    this.#pull.fill(0);
  }

  /** Everything at rest, now. */
  reset(): void {
    this.#s.fill(0);
    this.#v.fill(0);
    this.#b = undefined;
    this.#pull.fill(0);
    this.#resting = true;
  }

  /** Advances the state to time `t` (seconds, any clock that only moves forward). */
  advance(t: number): void {
    const h = POKE_STEP_S;
    const target = Math.floor(t / h);
    if (this.#step === undefined || target < this.#step) {
      this.#step = target;
      return;
    }
    if (target - this.#step > POKE_MAX_CATCHUP_S / h)
      this.#step = target - Math.round(POKE_MAX_CATCHUP_S / h);
    while (this.#step < target) {
      this.#stepOnce(h);
      this.#step += 1;
    }
    if (this.#b === undefined && this.#still()) {
      this.reset();
    }
  }

  /**
   * The handles now, bounded: `12·m` numbers, `Z_j = [0 | q_j]` row-major, into `out`; null at
   * rest. (The state is at most one `POKE_STEP_S` behind the time last advanced to.)
   */
  handles(out?: Float64Array): Float64Array | null {
    if (!this.active || this.#step === undefined) return null;
    const { handles: m, modes: r, shapes, limits } = this.model;
    const result = out ?? new Float64Array(m * TWELVE);
    result.fill(0);
    let ratio = 0;
    for (let j = 0; j < m; j += 1) {
      let length = 0;
      for (let a = 0; a < AXES; a += 1) {
        let q = 0;
        for (let i = 0; i < r; i += 1) q += (shapes[j * r + i] ?? 0) * (this.#s[i * AXES + a] ?? 0);
        result[j * TWELVE + 3 + 4 * a] = q;
        length += q * q;
      }
      ratio = Math.max(ratio, Math.sqrt(length) / (limits[j] ?? 1));
    }
    const factor = ratio > 1e-12 ? Math.tanh(ratio) / ratio : 1;
    if (factor !== 1)
      for (let k = 0; k < result.length; k += 1) result[k] = (result[k] ?? 0) * factor;
    return result;
  }

  /** The grabbed splat's displacement now (unbounded), metres: what the spring pulls. */
  grabbedDisplacement(): [number, number, number] {
    const b = this.#b;
    const out: [number, number, number] = [0, 0, 0];
    if (!b) return out;
    for (let a = 0; a < AXES; a += 1) {
      let u = 0;
      for (let i = 0; i < b.length; i += 1) u += (b[i] ?? 0) * (this.#s[i * AXES + a] ?? 0);
      out[a] = u;
    }
    return out;
  }

  /** The modal coordinates (r × 3, row-major), for tests. */
  get state(): Float64Array {
    return this.#s.slice();
  }

  /**
   * One Newmark (average acceleration) step of `s̈ + C ṡ + (Ω² + κ b bᵀ) s = κ b d`, per axis:
   * `K_eff = Ω² + (2/h) C + 4/h² + κ b bᵀ`, a diagonal plus rank one.
   */
  #stepOnce(h: number): void {
    const { modes: r, omega, damping } = this.model;
    const b = this.#b;
    const kappa = b ? this.#kappa : 0;
    const diag = new Float64Array(r);
    const c = new Float64Array(r);
    for (let i = 0; i < r; i += 1) {
      const w = omega[i] ?? 0;
      c[i] = 2 * (damping[i] ?? 0) * w;
      diag[i] = w * w + (2 / h) * (c[i] ?? 0) + 4 / (h * h);
    }
    // bᵀ D⁻¹ b, once per step.
    let bdb = 0;
    if (b) for (let i = 0; i < r; i += 1) bdb += (b[i] ?? 0) ** 2 / (diag[i] ?? 1);
    const rhs = new Float64Array(r);
    for (let a = 0; a < AXES; a += 1) {
      // u = bᵀ s now, for the consistent acceleration.
      let u = 0;
      if (b) for (let i = 0; i < r; i += 1) u += (b[i] ?? 0) * (this.#s[i * AXES + a] ?? 0);
      const d = this.#pull[a] ?? 0;
      for (let i = 0; i < r; i += 1) {
        const k = i * AXES + a;
        const s = this.#s[k] ?? 0;
        const v = this.#v[k] ?? 0;
        const w = omega[i] ?? 0;
        const force = b ? kappa * (b[i] ?? 0) * (d - u) : 0;
        const acc = force - (c[i] ?? 0) * v - w * w * s; // consistent with the forces now
        // f_{n+1} (held constant over the step) + M(4/h² s + 4/h v + a) + C(2/h s + v)
        const f1 = b ? kappa * (b[i] ?? 0) * d : 0;
        rhs[i] = f1 + (4 / (h * h)) * s + (4 / h) * v + acc + (c[i] ?? 0) * ((2 / h) * s + v);
      }
      // Solve (D + κ b bᵀ) x = rhs by Sherman-Morrison.
      let bdr = 0;
      if (b) for (let i = 0; i < r; i += 1) bdr += ((b[i] ?? 0) * (rhs[i] ?? 0)) / (diag[i] ?? 1);
      const scale = b ? (kappa * bdr) / (1 + kappa * bdb) : 0;
      for (let i = 0; i < r; i += 1) {
        const k = i * AXES + a;
        const x = ((rhs[i] ?? 0) - (b ? scale * (b[i] ?? 0) : 0)) / (diag[i] ?? 1);
        const s = this.#s[k] ?? 0;
        const v = this.#v[k] ?? 0;
        this.#v[k] = (2 / h) * (x - s) - v;
        this.#s[k] = x;
      }
    }
  }

  #still(): boolean {
    const { modes: r, omega, scale } = this.model;
    const tolerance = POKE_REST_SHARE * scale;
    for (let i = 0; i < r; i += 1) {
      const w = Math.max(omega[i] ?? 0, 1e-3);
      for (let a = 0; a < AXES; a += 1) {
        const k = i * AXES + a;
        // Amplitude of the oscillation: position and velocity over Ω together.
        if (Math.hypot(this.#s[k] ?? 0, (this.#v[k] ?? 0) / w) > tolerance) return false;
      }
    }
    return true;
  }
}
