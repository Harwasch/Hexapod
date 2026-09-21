/**
 * Band-limited turbulent forcing, and the damped oscillator each rig node presents to it.
 *
 * The model this replaced was quasi-static: it computed where a tree would come to *rest* under
 * a steady wind. That is a correct load model and it is not motion — nothing in it had a natural
 * frequency, so nothing could ring. Measured on the synthetic tree at the default wind, the
 * highest tip moved 0.15 mm per frame, which on a 6 m tree filling 800 px is 0.02 px. Below the
 * threshold of being motion at all.
 *
 * What is here instead: the wind is expressed as a sum of sinusoids at fixed frequencies with
 * fixed phases — band-limited turbulence — and each node is given the **analytic steady-state
 * response of a damped second-order oscillator** to that forcing. A node amplifies what lands
 * near its own natural frequency and ignores the rest, so a thin twig picks up the fast energy
 * and the trunk picks up the slow, without anything being told which is which.
 *
 * Crucially this is still a **closed form in `t`**: no integration, no previous frame, no state.
 * Determinism, `snapshot()` and the byte-immutability guarantees all depend on that, and a
 * numerically integrated oscillator would have cost every one of them.
 */

import { hash32, hashString } from "./noise";
import { type MotionRig } from "./rig";
import { clamp, distance } from "./vec";

const UINT32 = 0x1_0000_0000;

/** A unit-interval value from a 32-bit hash. */
function unitHash(value: number, seed: number): number {
  return hash32(value, seed) / UINT32;
}

// ---------------------------------------------------------------------------------------------
// The forcing: band-limited turbulence as a fixed sum of sinusoids.
// ---------------------------------------------------------------------------------------------

/** Sinusoids summed to make the turbulent forcing. Seven is enough to sound aperiodic. */
export const TURBULENCE_MODE_COUNT = 7;

/**
 * The forcing band, hertz.
 *
 * The bottom is 0.4 Hz because below that the response is a lean rather than a sway, and that
 * part of the load is already carried by the steady term in `deform` — the slow gust envelope in
 * `wind.ts` modulates it. The top is 6 Hz because above it there is little energy a member as
 * large as a branch can feel, and because nothing in a tree-sized rig has a natural frequency
 * that high without being a twig that simply follows its parent.
 */
const FORCING_MIN_HZ = 0.4;
const FORCING_MAX_HZ = 6;

/**
 * Amplitude falloff exponent: component `k`'s amplitude goes as `f_k^(-SPECTRAL_EXPONENT)`.
 *
 * A Kolmogorov inertial subrange would give `1/2` in power and so `5/6` in amplitude, and the
 * shape here is that family but **flatter**, at `1/2`, and the reason is not physics: over this
 * narrow a band the steeper law put so much of the energy at the bottom that the tip's dominant
 * response came out near 0.5 Hz, half of where a 6 m tree actually rings. Flattening it moves
 * the response into the measured band. Treat the exponent as a tuning constant with a physical
 * pedigree, not as a spectrum anybody measured.
 */
const SPECTRAL_EXPONENT = 0.5;

/** Fraction by which a component's frequency is nudged off the geometric ladder. */
const FORCING_JITTER = 0.16;

/** One sinusoid of the turbulent forcing. Fixed for all time and every rig. */
export interface TurbulenceMode {
  /** Frequency, hertz. */
  readonly hz: number;
  /** Angular frequency, radians per second. */
  readonly omega: number;
  /** Share of the forcing. The amplitudes sum to exactly 1, which is what makes bounds exact. */
  readonly amplitude: number;
  /** Fixed phase offset, radians. Deterministic, from a hash — never random. */
  readonly phase: number;
}

/**
 * The forcing spectrum: frequencies on a jittered geometric ladder so no two are rationally
 * related and the sum never audibly repeats, amplitudes normalised to sum to 1.
 */
export const TURBULENCE_MODES: readonly TurbulenceMode[] = buildTurbulenceModes();

function buildTurbulenceModes(): readonly TurbulenceMode[] {
  const seed = 0x5eed_7b01;
  const ratio = Math.pow(FORCING_MAX_HZ / FORCING_MIN_HZ, 1 / (TURBULENCE_MODE_COUNT - 1));
  const raw: { hz: number; weight: number; phase: number }[] = [];
  let total = 0;
  for (let k = 0; k < TURBULENCE_MODE_COUNT; k += 1) {
    const jitter = 1 + FORCING_JITTER * (unitHash(k, seed) * 2 - 1);
    const hz = FORCING_MIN_HZ * Math.pow(ratio, k) * jitter;
    const weight = Math.pow(hz, -SPECTRAL_EXPONENT);
    total += weight;
    raw.push({ hz, weight, phase: unitHash(k, seed + 977) * 2 * Math.PI });
  }
  return raw.map((mode) => ({
    hz: mode.hz,
    omega: 2 * Math.PI * mode.hz,
    amplitude: mode.weight / total,
    phase: mode.phase,
  }));
}

// ---------------------------------------------------------------------------------------------
// The oscillator: one per node, derived from rig geometry.
// ---------------------------------------------------------------------------------------------

/**
 * Hertz per unit of `radius / length²`.
 *
 * A uniform cantilever's fundamental goes as `(r / L²)·√(E/ρ)` up to a mode constant, so the
 * *shape* of this law is beam theory. The constant is **not**: taken literally, green wood would
 * put a 6 m trunk near 6 Hz, where real trees of that size are measured at 0.5–1.5 Hz. Taper,
 * crown mass, root compliance and aerodynamic added mass all soften a real tree far below a
 * prismatic beam. So the scale is fitted to the observed band, and this comment is the
 * derivation: **the exponents are physics, the constant is calibration.** Nothing here is
 * traceable to a biomechanical measurement of any particular species.
 */
const FREQ_SCALE_HZ = 220;

/** Natural frequency is clamped into this band, so a degenerate rig cannot produce nonsense. */
const MIN_NODE_HZ = 0.25;
const MAX_NODE_HZ = 12;

/** Shortest cantilever length used, metres. Keeps `radius / length²` finite for a stub node. */
const MIN_LENGTH_M = 0.15;

/** Damping ratio: thin members shed energy to the air far faster than a trunk does. */
const ZETA_MIN = 0.05;
const ZETA_SPAN = 0.1;
const ZETA_RADIUS_M = 0.15;

/**
 * Half-width of the per-node bending plane, radians from downwind.
 *
 * This is the constant that stops the crown moving as one flat sheet. Every node used to share a
 * single bending axis — `up × downwind`, computed once for the whole rig — so the tree differed
 * from a rigid plate only in amplitude. It applies to the *oscillating* part alone: steady drag
 * is downwind for every node, because that is what steady drag is.
 */
const AZIMUTH_SPREAD_RAD = 0.7;

/** Out-of-plane tilt of the bending axis, as a fraction of the in-plane axis. Adds torsion. */
const TWIST_SPREAD = 0.35;

/** Longest convection delay across the crown, seconds: an eddy does not arrive everywhere at once. */
const GUST_LAG_S = 0.9;

const AZIMUTH_SEED = 0x5eed_a21f;
const TWIST_SEED = 0x5eed_7715;
const LAG_SEED = 0x5eed_1a60;

/**
 * Everything the deformation needs to know about one node's own dynamics. A pure function of the
 * rig's geometry and node ids — no time, no wind, no state.
 */
export interface NodeMode {
  /** Natural angular frequency, radians per second. */
  readonly omega: number;
  /** Damping ratio, dimensionless. Trees are lightly damped; that is why they ring. */
  readonly zeta: number;
  /** Cantilever length this node's frequency was derived from, metres. Diagnostic. */
  readonly lengthM: number;
  /** Per-forcing-mode amplitude gain `A_k · H(ω_k)`. */
  readonly gains: readonly number[];
  /** Per-forcing-mode total phase: forcing phase, transfer lag and convection delay. */
  readonly phases: readonly number[];
  /** `Σ_k A_k·H(ω_k)`: the exact worst case of the oscillating term. */
  readonly response: number;
  /** `Σ_k A_k·H(ω_k)·ω_k`: the exact worst case of its time derivative. */
  readonly responseRate: number;
  /** Rotation of this node's oscillating bending plane from downwind, radians, about up. */
  readonly azimuthRad: number;
  /** Tilt of the bending axis out of horizontal, dimensionless. */
  readonly twist: number;
}

/**
 * Steady-state amplitude gain of a damped oscillator driven at `omega`.
 *
 * `H(ω) = 1 / √((1 − (ω/ω₀)²)² + (2ζ·ω/ω₀)²)`. One at low frequency (the node follows the wind
 * quasi-statically), `1/(2ζ)` at resonance, and falling as `(ω₀/ω)²` above it — which is exactly
 * the frequency selection the old model had no way to express.
 */
export function responseGain(omega: number, naturalOmega: number, zeta: number): number {
  if (!(naturalOmega > 0) || !Number.isFinite(omega)) return 0;
  const r = omega / naturalOmega;
  const real = 1 - r * r;
  const imag = 2 * zeta * r;
  const denominator = Math.sqrt(real * real + imag * imag);
  return denominator > 0 ? 1 / denominator : 0;
}

/** Phase lag of that response, radians, in `(−π, 0]`. Zero below resonance, `−π/2` at it. */
export function responseLag(omega: number, naturalOmega: number, zeta: number): number {
  if (!(naturalOmega > 0) || !Number.isFinite(omega)) return 0;
  const r = omega / naturalOmega;
  return -Math.atan2(2 * zeta * r, 1 - r * r);
}

/**
 * Path length from each node to the furthest tip below it, metres.
 *
 * This is the cantilever a node actually carries: a trunk joint carries the whole tree above it,
 * a twig carries only itself. One reverse pass suffices because `nodes[i].parent < i` is a rig
 * invariant, so every child has been visited before its parent is.
 */
export function tipReaches(rig: MotionRig): number[] {
  const reach = new Array<number>(rig.nodes.length).fill(0);
  for (let i = rig.nodes.length - 1; i >= 1; i -= 1) {
    const node = rig.nodes[i];
    if (node === undefined || node.parent < 0) continue;
    const parent = rig.nodes[node.parent];
    if (parent === undefined) continue;
    const candidate = distance(node.position, parent.position) + (reach[i] ?? 0);
    if (candidate > (reach[node.parent] ?? 0)) reach[node.parent] = candidate;
  }
  return reach;
}

/** The natural frequency a node's own geometry implies, hertz. */
export function nodeNaturalHz(radiusM: number, lengthM: number): number {
  const length = Math.max(lengthM, MIN_LENGTH_M);
  const radius = Number.isFinite(radiusM) && radiusM > 0 ? radiusM : 0.01;
  return clamp((FREQ_SCALE_HZ * radius) / (length * length), MIN_NODE_HZ, MAX_NODE_HZ);
}

/** Damping ratio implied by a node's thickness, in `[ZETA_MIN, ZETA_MIN + ZETA_SPAN]`. */
export function nodeZeta(radiusM: number): number {
  const radius = Number.isFinite(radiusM) && radiusM > 0 ? radiusM : 0;
  return ZETA_MIN + ZETA_SPAN * Math.exp(-radius / ZETA_RADIUS_M);
}

/**
 * The oscillator every node presents to the forcing.
 *
 * Derived from **geometry**, never from the band label: `band` sets an angular limit and nothing
 * else, because a rig that comes out of `skeleton.py` will not have the synthetic fixture's tidy
 * structure and its bands are inferred rather than known. Radius and the distance to the tips it
 * carries are measurable on any rig; "this is a branch" is an opinion.
 */
export function nodeModes(rig: MotionRig): NodeMode[] {
  const reach = tipReaches(rig);
  return rig.nodes.map((node, index) => {
    const parent = node.parent >= 0 ? rig.nodes[node.parent] : undefined;
    const segment = parent === undefined ? 0 : distance(node.position, parent.position);
    const lengthM = Math.max(reach[index] ?? 0, segment, MIN_LENGTH_M);
    const omega = 2 * Math.PI * nodeNaturalHz(node.radius, lengthM);
    const zeta = nodeZeta(node.radius);
    const idHash = hashString(node.id);
    const lag = unitHash(idHash, LAG_SEED) * GUST_LAG_S;
    const gains: number[] = [];
    const phases: number[] = [];
    let response = 0;
    let responseRate = 0;
    for (const mode of TURBULENCE_MODES) {
      const gain = mode.amplitude * responseGain(mode.omega, omega, zeta);
      gains.push(gain);
      phases.push(mode.phase + responseLag(mode.omega, omega, zeta) + mode.omega * lag);
      response += gain;
      responseRate += gain * mode.omega;
    }
    return {
      omega,
      zeta,
      lengthM,
      gains,
      phases,
      response,
      responseRate,
      azimuthRad: AZIMUTH_SPREAD_RAD * (unitHash(idHash, AZIMUTH_SEED) * 2 - 1),
      twist: TWIST_SPREAD * (unitHash(idHash, TWIST_SEED) * 2 - 1),
    };
  });
}

/**
 * The oscillating part of a node's load at time `t`, dimensionless and in `[−response, response]`.
 *
 * `Σ_k A_k·H(ω_k)·sin(ω_k·t + φ_k + ψ(ω_k) + ω_k·τ)`. Every term is a closed form of `t`, so the
 * whole sum is, and the value at `t` owes nothing to the value at `t − dt`.
 */
export function turbulentLoad(mode: NodeMode, t: number): number {
  const time = Number.isFinite(t) ? t : 0;
  let sum = 0;
  for (let k = 0; k < TURBULENCE_MODES.length; k += 1) {
    const forcing = TURBULENCE_MODES[k];
    if (forcing === undefined) continue;
    sum += (mode.gains[k] ?? 0) * Math.sin(forcing.omega * time + (mode.phases[k] ?? 0));
  }
  return sum;
}

/** A node's natural frequency in hertz, for anything that wants to print it. */
export function nodeNaturalFrequencyHz(mode: NodeMode): number {
  return mode.omega / (2 * Math.PI);
}
