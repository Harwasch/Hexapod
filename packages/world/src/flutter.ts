/**
 * Per-splat flutter: the shimmer a canopy has and a rigid block of splats does not.
 *
 * Everything else in this package moves *nodes*. A node's transform is applied to every splat
 * assigned to it, so roughly sixty splats of one leaf cluster travel as a welded block — and a
 * block of foliage that translates rigidly is the single most recognisable way for a tree in
 * wind to look wrong. Real leaves move relative to the twig they hang off, at frequencies well
 * above anything a limb can reach, and that relative motion is most of what the eye uses to
 * decide a tree is alive.
 *
 * The rig cannot express it: a leaf is not a node and never will be, at any node count that
 * assignment can afford. So this is a second, additive term, evaluated per splat on top of the
 * node transform, and it is deliberately the cheapest thing that could carry the cue:
 *
 * ```text
 * displaced = transform(canonical) + flutter(splatIndex, node, t)
 * ```
 *
 * ### What it must not break
 *
 * - **`strength === 0` is exactly identity.** `windMagnitude` is exactly zero at calm, every
 *   node amplitude is then exactly zero, `still` is true, and `deformPositions` takes a branch
 *   that never touches the coordinate — not even to add `0`, which would turn a `-0` into a
 *   `+0` and break the byte-identity the restore path and `snapshot()` depend on.
 * - **Determinism.** Seeded by splat index through `hash32`, with no `Math.random`, no
 *   `Date.now`, and no state carried between frames. The same `(index, node, field)` is the
 *   same three numbers on every machine.
 * - **The bound stays a bound.** A splat's offset is two orthogonal sinusoids whose amplitudes
 *   are `PRIMARY_SHARE` and `1 - PRIMARY_SHARE` of the node's amplitude, so the offset's
 *   magnitude is `A·√(p² + (1−p)²) ≤ A` exactly. `maxFlutterAmplitude` adds that `A` to
 *   `maxDisplacement`, which is what `sortStaleness` divides.
 *
 * ### Why it is not `Math.sin` per splat
 *
 * At 150k splats and 60 Hz, two `Math.sin` per splat is 18 M transcendentals a second, which is
 * milliseconds of frame time for a term worth micrometres of accuracy. Instead every splat of a
 * node shares that node's angular frequency — which is true of real foliage on one twig, and is
 * the reason a cluster ripples rather than boils — so `sin(ω·t)` and `cos(ω·t)` are computed
 * once per node per frame, and the per-splat phase becomes an angle-addition against a
 * 1024-entry table of `(cos φ, sin φ)` pairs. The per-splat cost is a hash, three table reads
 * and a dozen multiply-adds, with no trigonometry at all.
 */

import { hash32 } from "./noise";
import { nodeModes, nodeNaturalFrequencyHz } from "./modes";
import { type MotionRig } from "./rig";
import { clamp, softLimit, VEC3_ZERO, type Vec3 } from "./vec";
import { gustDelaySeconds, maxWindMagnitude, windAt, type WindSettings } from "./wind";

/**
 * Largest flutter amplitude, metres, at a node of zero radius and saturating wind.
 *
 * Millimetres, not centimetres: at the default wind a leaf cluster's splats swing about 5 mm,
 * which at 4–15 Hz is a per-frame step several times the *node's* own, and costs a quarter of a
 * reference splat radius of draw-order staleness. That asymmetry — cheap in amplitude, rich in
 * speed — is the whole reason this term is worth having, and it is the same trade S8 made when
 * it moved the node model's energy from 0.3 Hz to 1.1 Hz without raising its reach.
 */
export const FLUTTER_SCALE_M = 0.09;

/**
 * Thickness at which flutter has fallen by `1/e`: `amplitude ∝ exp(−radius / this)`.
 *
 * Geometry, not the band label, for the same reason `modes.ts` derives natural frequency from
 * geometry: a rig out of `skeleton.py` has bands that were *inferred*, and a mislabelled trunk
 * that shimmered would be a far worse artifact than one that did not. Radius is measured on any
 * rig; "this is a leaf" is an opinion.
 */
const FLUTTER_RADIUS_M = 0.015;

/**
 * Amplitude shapes below this fraction are taken to be exactly zero.
 *
 * Without it a trunk gets five micrometres of flutter — physically negligible and operationally
 * not, because a non-zero amplitude makes `markMovingNodes` call the node moving and re-upload
 * every bole splat every frame for a motion nobody can see. On this fixture the cut falls at a
 * 4.5 cm radius, which puts the whole trunk at exactly zero and leaves the crown untouched.
 */
const FLUTTER_SHAPE_MIN = 0.05;

/** Flutter band, hertz. A node's own natural frequency is clamped into it. */
const FLUTTER_MIN_HZ = 4;
const FLUTTER_MAX_HZ = 15;

/** The second sinusoid's frequency, as a multiple of the first. Irrational, so it never repeats. */
const SECONDARY_RATIO = Math.SQRT2;

/** Share of the amplitude carried by the first sinusoid; the rest goes to the second. */
const PRIMARY_SHARE = 0.7;

/**
 * Wind magnitude at which flutter saturates.
 *
 * Leaf flutter does not grow without bound with wind speed — leaves reconfigure and streamline,
 * which is measured, though not at any amplitude this constant could claim to reproduce. The
 * saturation here is `softLimit`, the same smooth one the bend angles use, and its practical
 * job is to keep a gale's shimmer at a couple of centimetres rather than ten.
 */
const FLUTTER_WIND_SATURATION = 0.4;

const FLUTTER_SEED = 0x5eed_f1c7;

/** Entries in the phase table. A power of two, so the index is a mask rather than a modulo. */
const PHASE_STEPS = 1024;
const PHASE_MASK = PHASE_STEPS - 1;

/** Orthonormal axis pairs. Also a power of two. */
const AXIS_COUNT = 256;
const AXIS_MASK = AXIS_COUNT - 1;

/** `[cos φ, sin φ]` at `φ = 2πk / PHASE_STEPS`. Built once; never mutated. */
const PHASE_TABLE: Float64Array = (() => {
  const table = new Float64Array(PHASE_STEPS * 2);
  for (let k = 0; k < PHASE_STEPS; k += 1) {
    const angle = (2 * Math.PI * k) / PHASE_STEPS;
    table[k * 2] = Math.cos(angle);
    table[k * 2 + 1] = Math.sin(angle);
  }
  return table;
})();

/**
 * Pairs of orthonormal axes, six numbers each: the plane a splat's flutter sweeps.
 *
 * The first axis is a Fibonacci-sphere direction, which spreads `AXIS_COUNT` of them about as
 * evenly over the sphere as anything closed-form does; the second is its cross product with
 * whichever cardinal axis it is least aligned with, so the pair is always well conditioned.
 */
const AXIS_TABLE: Float64Array = (() => {
  const table = new Float64Array(AXIS_COUNT * 6);
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let k = 0; k < AXIS_COUNT; k += 1) {
    const z = 1 - (2 * k + 1) / AXIS_COUNT;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const theta = golden * k;
    const ax = r * Math.cos(theta);
    const ay = r * Math.sin(theta);
    const az = z;
    // Cross with the cardinal axis this direction is least aligned with.
    const seed: [number, number, number] = Math.abs(az) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    let bx = ay * seed[2] - az * seed[1];
    let by = az * seed[0] - ax * seed[2];
    let bz = ax * seed[1] - ay * seed[0];
    const length = Math.hypot(bx, by, bz) || 1;
    bx /= length;
    by /= length;
    bz /= length;
    table[k * 6] = ax;
    table[k * 6 + 1] = ay;
    table[k * 6 + 2] = az;
    table[k * 6 + 3] = bx;
    table[k * 6 + 4] = by;
    table[k * 6 + 5] = bz;
  }
  return table;
})();

/**
 * One frame of the flutter field: what every node contributes, evaluated at one instant.
 *
 * A plain value with no methods and no hidden state, computed fresh each frame from
 * `(rig, t, settings)` alone. Two copies made from the same arguments are equal number for
 * number, which is what keeps the whole deformation reproducible.
 */
export interface FlutterField {
  /** Per-node amplitude, metres. Index-aligned with `rig.nodes`. Exactly 0 means no flutter. */
  readonly amplitudeM: Float64Array;
  /** Per node: `sin(ω·t)`, `cos(ω·t)`, `sin(ω₂·t)`, `cos(ω₂·t)`. Four entries per node. */
  readonly phase: Float64Array;
  /** True when every amplitude is exactly zero, so a caller can skip the term entirely. */
  readonly still: boolean;
}

/** The flutter field is entirely absent: every amplitude zero, nothing to add. */
export const FLUTTER_STILL: FlutterField = {
  amplitudeM: new Float64Array(0),
  phase: new Float64Array(0),
  still: true,
};

/** Flutter frequency a node's geometry implies, hertz: its own resonance, held inside the band. */
export function nodeFlutterHz(naturalHz: number): number {
  if (!Number.isFinite(naturalHz)) return FLUTTER_MIN_HZ;
  return clamp(naturalHz, FLUTTER_MIN_HZ, FLUTTER_MAX_HZ);
}

/**
 * How much a node of this thickness flutters, as a fraction of {@link FLUTTER_SCALE_M}.
 *
 * Exactly zero above the thickness cut, so a trunk is still rather than nearly still.
 */
export function flutterShape(radiusM: number): number {
  if (!Number.isFinite(radiusM) || radiusM < 0) return 0;
  const shape = Math.exp(-radiusM / FLUTTER_RADIUS_M);
  return shape < FLUTTER_SHAPE_MIN ? 0 : shape;
}

/** The largest amplitude any node of this rig reaches at this wind, metres. Exact, not sampled. */
export function maxFlutterAmplitude(rig: MotionRig, settings: WindSettings): number {
  const windTerm = softLimit(maxWindMagnitude(settings.strength), FLUTTER_WIND_SATURATION);
  let worst = 0;
  for (const node of rig.nodes) {
    const amplitude = FLUTTER_SCALE_M * flutterShape(node.radius) * windTerm;
    if (amplitude > worst) worst = amplitude;
  }
  return worst;
}

/** The largest flutter *speed* any splat reaches, metres per second. `A·ω`, summed over both terms. */
export function maxFlutterSpeed(rig: MotionRig, settings: WindSettings): number {
  const windTerm = softLimit(maxWindMagnitude(settings.strength), FLUTTER_WIND_SATURATION);
  const modes = nodeModes(rig);
  let worst = 0;
  rig.nodes.forEach((node, index) => {
    const mode = modes[index];
    if (mode === undefined) return;
    const amplitude = FLUTTER_SCALE_M * flutterShape(node.radius) * windTerm;
    if (amplitude === 0) return;
    const omega = 2 * Math.PI * nodeFlutterHz(nodeNaturalFrequencyHz(mode));
    // Each term's speed is its own amplitude times its own frequency; they can peak together.
    const speed =
      amplitude * PRIMARY_SHARE * omega + amplitude * (1 - PRIMARY_SHARE) * omega * SECONDARY_RATIO;
    if (speed > worst) worst = speed;
  });
  return worst;
}

/**
 * The flutter field at time `t`. Pure: no state, no clock, no randomness.
 *
 * At `settings.strength === 0` the gust is exactly `[0, 0, 0]`, so every amplitude is exactly
 * `0` and `still` is true — the branch `deformPositions` needs in order to leave the canonical
 * bytes alone.
 */
export function flutterField(rig: MotionRig, t: number, settings: WindSettings): FlutterField {
  const time = Number.isFinite(t) ? t : 0;
  const count = rig.nodes.length;
  const amplitudeM = new Float64Array(count);
  const phase = new Float64Array(count * 4);
  // Nothing can flutter at calm, whatever the positions: `wind` returns exactly `[0, 0, 0]`.
  if (maxWindMagnitude(settings.strength) === 0) return { amplitudeM, phase, still: true };

  const modes = nodeModes(rig);
  let still = true;
  for (let n = 0; n < count; n += 1) {
    const node = rig.nodes[n];
    const mode = modes[n];
    if (node === undefined || mode === undefined) continue;
    const shape = flutterShape(node.radius);
    if (shape === 0) continue;
    // The gust that reached *this* node, so the shimmer ebbs and swells as a gust crosses the
    // crown rather than everywhere at once.
    const delay = gustDelaySeconds(settings.bearingDeg, node.position);
    const gust = windAt(settings.strength, settings.bearingDeg, time, node.position);
    const windTerm = softLimit(Math.hypot(gust[0], gust[1], gust[2]), FLUTTER_WIND_SATURATION);
    const amplitude = FLUTTER_SCALE_M * shape * windTerm;
    if (amplitude === 0) continue;
    amplitudeM[n] = amplitude;
    still = false;
    const omega = 2 * Math.PI * nodeFlutterHz(nodeNaturalFrequencyHz(mode));
    const primary = omega * (time - delay);
    const secondary = omega * SECONDARY_RATIO * (time - delay);
    phase[n * 4] = Math.sin(primary);
    phase[n * 4 + 1] = Math.cos(primary);
    phase[n * 4 + 2] = Math.sin(secondary);
    phase[n * 4 + 3] = Math.cos(secondary);
  }
  return { amplitudeM, phase, still };
}

/**
 * One splat's flutter offset, metres, in the rig's local frame.
 *
 * `sin(ω·t + φ) = sin(ω·t)·cos φ + cos(ω·t)·sin φ`, with the node's `sin/cos(ω·t)` already in
 * the field and the splat's `cos/sin φ` read out of a table by a hash of its index. No
 * trigonometry runs here.
 *
 * Returns exactly `[0, 0, 0]` — the shared frozen tuple, so it costs nothing — when the node
 * does not flutter.
 */
export function splatFlutter(splatIndex: number, node: number, field: FlutterField): Vec3 {
  if ((field.amplitudeM[node] ?? 0) === 0) return VEC3_ZERO;
  const out = new Float64Array(3);
  splatFlutterInto(splatIndex, node, flutterCoefficients(field), out);
  return [out[0] ?? 0, out[1] ?? 0, out[2] ?? 0];
}

/**
 * Per-node coefficients for {@link splatFlutterInto}: four numbers per node.
 *
 * `A·p·sin(ω·t)`, `A·p·cos(ω·t)` and the same pair for the second sinusoid. Everything in a
 * splat's offset that does not depend on the splat, folded once per node per frame so that the
 * per-splat loop is a hash, eight table reads and a dozen multiply-adds. All zero for a node
 * that does not flutter, which the caller is expected to test for before calling.
 */
function flutterCoefficients(field: FlutterField): Float64Array {
  const nodes = field.amplitudeM.length;
  const coefficients = new Float64Array(nodes * 4);
  for (let n = 0; n < nodes; n += 1) {
    const amplitude = field.amplitudeM[n] ?? 0;
    if (amplitude === 0) continue;
    const primary = amplitude * PRIMARY_SHARE;
    const secondary = amplitude * (1 - PRIMARY_SHARE);
    coefficients[n * 4] = primary * (field.phase[n * 4] ?? 0);
    coefficients[n * 4 + 1] = primary * (field.phase[n * 4 + 1] ?? 0);
    coefficients[n * 4 + 2] = secondary * (field.phase[n * 4 + 2] ?? 0);
    coefficients[n * 4 + 3] = secondary * (field.phase[n * 4 + 3] ?? 0);
  }
  return coefficients;
}

/**
 * Adds every splat's flutter offset to `target`, in place, over `count` splats.
 *
 * The whole per-splat pass, kept in this module so the tables stay private and the loop stays
 * one small, inlinable body. Measured at 150,000 splats: this form costs about 1 ms a frame,
 * where the same arithmetic fused into `deformPositions`' transform loop cost 6 ms and calling
 * out to a helper per splat cost 5 ms. The transform and the flutter are independent, so there
 * is nothing to be gained by fusing them and, empirically, a great deal to be lost.
 *
 * Does nothing at all for a still field, which is what keeps calm bit-exact: no coordinate is
 * read, no zero is added.
 */
export function applyFlutter(
  target: Float32Array,
  assignment: Uint16Array,
  field: FlutterField,
  count: number,
): void {
  if (field.still) return;
  const coefficients = flutterCoefficients(field);
  const amplitudes = field.amplitudeM;
  const limit = Math.min(count, assignment.length, Math.floor(target.length / 3));
  for (let i = 0; i < limit; i += 1) {
    const node = assignment[i] ?? 0;
    if ((amplitudes[node] ?? 0) === 0) continue;
    // `hash32` written out: it lives in another module, and 150,000 calls a frame that the
    // JIT declines to inline is most of this loop's cost. Identical arithmetic, and
    // `flutter.test.ts` pins the two together through `splatFlutter`.
    const h0 = (Math.imul(FLUTTER_SEED | 0, 0x9e3779b1) ^ (i | 0)) >>> 0;
    const h1 = Math.imul(h0 ^ (h0 >>> 16), 0x85ebca6b) >>> 0;
    const h2 = Math.imul(h1 ^ (h1 >>> 13), 0xc2b2ae35) >>> 0;
    const h = (h2 ^ (h2 >>> 16)) >>> 0;
    const p1 = (h & PHASE_MASK) * 2;
    const p2 = ((h >>> 10) & PHASE_MASK) * 2;
    const axis = ((h >>> 20) & AXIS_MASK) * 6;
    const c = node * 4;
    const wave1 =
      (coefficients[c] ?? 0) * (PHASE_TABLE[p1] ?? 1) +
      (coefficients[c + 1] ?? 0) * (PHASE_TABLE[p1 + 1] ?? 0);
    const wave2 =
      (coefficients[c + 2] ?? 0) * (PHASE_TABLE[p2] ?? 1) +
      (coefficients[c + 3] ?? 0) * (PHASE_TABLE[p2 + 1] ?? 0);
    const base = i * 3;
    target[base] =
      (target[base] ?? 0) + wave1 * (AXIS_TABLE[axis] ?? 0) + wave2 * (AXIS_TABLE[axis + 3] ?? 0);
    target[base + 1] =
      (target[base + 1] ?? 0) +
      wave1 * (AXIS_TABLE[axis + 1] ?? 0) +
      wave2 * (AXIS_TABLE[axis + 4] ?? 0);
    target[base + 2] =
      (target[base + 2] ?? 0) +
      wave1 * (AXIS_TABLE[axis + 2] ?? 1) +
      wave2 * (AXIS_TABLE[axis + 5] ?? 0);
  }
}

/**
 * The offset for one splat, written into `out[0..2]` from precomputed node coefficients.
 *
 * The readable statement of what {@link applyFlutter} does per splat, and what
 * {@link splatFlutter} is built on, so the batch loop above has something to be compared
 * against rather than being the only definition of the arithmetic.
 */
function splatFlutterInto(
  splatIndex: number,
  node: number,
  coefficients: Float64Array,
  out: Float64Array,
): void {
  const h = hash32(splatIndex, FLUTTER_SEED);
  const p1 = (h & PHASE_MASK) * 2;
  const p2 = ((h >>> 10) & PHASE_MASK) * 2;
  const axis = ((h >>> 20) & AXIS_MASK) * 6;
  const c = node * 4;
  // sin(ω·t + φ) = sin(ω·t)·cos φ + cos(ω·t)·sin φ, twice, already scaled by the amplitude.
  const wave1 =
    (coefficients[c] ?? 0) * (PHASE_TABLE[p1] ?? 1) +
    (coefficients[c + 1] ?? 0) * (PHASE_TABLE[p1 + 1] ?? 0);
  const wave2 =
    (coefficients[c + 2] ?? 0) * (PHASE_TABLE[p2] ?? 1) +
    (coefficients[c + 3] ?? 0) * (PHASE_TABLE[p2 + 1] ?? 0);
  out[0] = wave1 * (AXIS_TABLE[axis] ?? 0) + wave2 * (AXIS_TABLE[axis + 3] ?? 0);
  out[1] = wave1 * (AXIS_TABLE[axis + 1] ?? 0) + wave2 * (AXIS_TABLE[axis + 4] ?? 0);
  out[2] = wave1 * (AXIS_TABLE[axis + 2] ?? 1) + wave2 * (AXIS_TABLE[axis + 5] ?? 0);
}
