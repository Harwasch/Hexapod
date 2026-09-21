/**
 * Bounded, deterministic deformation of a skeleton rig under wind.
 *
 * Every transform is computed from the rig's rest pose, never from the previous frame, so error
 * cannot accumulate and `strength = 0` returns the canonical pose exactly rather than nearly.
 *
 * A node's local bend is the sum of two rotations:
 *
 * 1. a **steady lean**, exactly downwind, proportional to the wind magnitude — real wind does
 *    load a tree, and a tree in a breeze does sit off its rest pose; and
 * 2. an **oscillation** about the node's own bending plane, being that node's damped-oscillator
 *    response to band-limited turbulent forcing (see `modes.ts`).
 *
 * The split matters. The lean is where the old model put *all* of the motion, which is why the
 * tree bent once and then appeared static: at the default wind its tip moved 0.15 mm per frame.
 * The lean is now a minority share of each node's angular budget ({@link MEAN_SHARE}) and the
 * oscillation carries what the eye sees. Peak displacement is deliberately close to what it was
 * — sort staleness is driven by amplitude, not by speed — and the change is in *frequency*.
 */

import { splatFlutter, type FlutterField } from "./flutter";
import { nodeModes, turbulentLoad, type NodeMode } from "./modes";
import { nodeAngleLimit, type MotionRig, type SkeletonNode } from "./rig";
import {
  add,
  cross,
  normalize,
  quatFromAxisAngle,
  quatMultiply,
  quatRotate,
  QUAT_IDENTITY,
  softLimit,
  subtract,
  magnitude,
  VEC3_UP,
  VEC3_ZERO,
  type Quat,
  type Vec3,
} from "./vec";
import { maxWindMagnitude, maxWindSlope, wind as sampleWind, type WindSettings } from "./wind";

/**
 * A node's rest-to-displaced transform: `displaced = rotation ⊗ p + translation`, with `p` a
 * canonical position in the rig's local ENU frame.
 */
export interface NodeTransform {
  readonly rotation: Quat;
  readonly translation: Vec3;
}

/** The rest transform. Bit-exact: `rotation` is `[0, 0, 0, 1]`, `translation` is `[0, 0, 0]`. */
export const IDENTITY_TRANSFORM: NodeTransform = {
  rotation: QUAT_IDENTITY,
  translation: VEC3_ZERO,
};

/** Radians of local bend per unit wind magnitude, at stiffness 1, before the limiter. */
export const BEND_GAIN = 0.31;

/**
 * Share of the load carried by the steady downwind lean, against {@link TURBULENT_LOAD}.
 *
 * Deliberately the smaller of the two. A steady lean is the part of wind loading a person reads
 * as "bent", not as "moving"; it is kept because a tree in a breeze genuinely does sit off its
 * rest pose, and dropped to a minority because the eye has to be given something to watch.
 *
 * It fell from 0.4 to 0.3 when the fixture gained a real crown. The lean is proportional to the
 * wind magnitude, whose envelope turns over in five to fifteen seconds, so every unit of it
 * lands below 0.4 Hz; with 0.4 the tip's slow band came back up to 50 % of its alternating
 * power — half the motion was drift again. At 0.3 it is 38 %.
 */
export const MEAN_LOAD = 0.3;

/** Share carried by the turbulent, oscillating load. The two shares sum to 1 by construction. */
export const TURBULENT_LOAD = 1 - MEAN_LOAD;

/**
 * Fraction of a node's angular limit the steady lean may spend, the rest going to the
 * oscillation.
 *
 * Splitting the budget rather than limiting the sum is what keeps the bound closed-form: each
 * term is saturated against its own share, so their magnitudes sum to strictly less than the
 * node's limit and the composed rotation's angle — subadditive over a product of rotations —
 * stays inside it, whatever the two bending planes are.
 */
export const MEAN_SHARE = 0.35;

/** The two halves of a node's angular limit: what the lean may spend, and what the sway may. */
function angleBudget(node: SkeletonNode): { mean: number; oscillation: number } {
  const limit = nodeAngleLimit(node);
  return { mean: limit * MEAN_SHARE, oscillation: limit * (1 - MEAN_SHARE) };
}

/**
 * Unlimited steady lean, radians: the quasi-static part of the load.
 *
 * The bend is a **curvature times a length**, not a bend: `mode.segmentM` is how much limb this
 * joint stands for. That is what makes the model invariant to how finely a rig samples a limb —
 * see `NodeMode.segmentM` — and it is the difference between a stiffness that describes the
 * tree and one that describes the extractor's node spacing.
 */
function rawLean(node: SkeletonNode, mode: NodeMode, windMagnitude: number): number {
  return (BEND_GAIN * MEAN_LOAD * windMagnitude * mode.segmentM) / node.stiffness;
}

/** Unlimited oscillation, radians, at the given dimensionless turbulent load. */
function rawSway(node: SkeletonNode, mode: NodeMode, windMagnitude: number, load: number): number {
  return (BEND_GAIN * TURBULENT_LOAD * windMagnitude * load * mode.segmentM) / node.stiffness;
}

/**
 * The largest local bend angle a node can reach at this strength, radians. Exact, not sampled.
 *
 * Both halves are closed-form: the wind magnitude is bounded by `maxWindMagnitude`, and the
 * oscillation by `Σ_k A_k·H(ω_k)` — the sum of its per-forcing-mode gains, reached when every
 * sinusoid peaks together. `softLimit` is monotone, so saturating each half against its own
 * share of the limit and adding gives a bound on the composed rotation's angle.
 */
export function maxNodeAngle(node: SkeletonNode, mode: NodeMode, settings: WindSettings): number {
  const windMagnitude = maxWindMagnitude(settings.strength);
  const budget = angleBudget(node);
  return (
    softLimit(rawLean(node, mode, windMagnitude), budget.mean) +
    softLimit(rawSway(node, mode, windMagnitude, mode.response), budget.oscillation)
  );
}

/**
 * Bound on `|dθ/dt|` for a node, radians per second. `softLimit` is a contraction (`|f'| ≤ 1`),
 * so a bound on the unlimited angles carries through to the limited ones, and the composed
 * angle's rate is bounded by the sum of the two.
 *
 * The lean's rate is the wind's own slope. The oscillation is a product of the wind magnitude
 * and a sum of sinusoids, so the product rule gives two terms: the envelope changing under a
 * fixed waveform, and the waveform changing under a fixed envelope. The second is the one that
 * matters and it is `Σ_k A_k·H(ω_k)·ω_k` — the frequency content the old model did not have.
 */
export function maxNodeAngleRate(
  node: SkeletonNode,
  mode: NodeMode,
  settings: WindSettings,
): number {
  const slope = maxWindSlope(settings.strength);
  const magnitudeBound = maxWindMagnitude(settings.strength);
  const leanRate = (BEND_GAIN * MEAN_LOAD * slope * mode.segmentM) / node.stiffness;
  const swayRate =
    (BEND_GAIN *
      TURBULENT_LOAD *
      mode.segmentM *
      (slope * mode.response + magnitudeBound * mode.responseRate)) /
    node.stiffness;
  return leanRate + swayRate;
}

/** Unit vector pointing downwind, from a bearing in degrees clockwise from north. */
function downwindOf(gust: Vec3, fallback: Vec3): Vec3 {
  return normalize(gust, fallback);
}

/**
 * The axis a node oscillates about: `up × downwind`, swung by the node's own azimuth and tilted
 * out of horizontal by its own twist.
 *
 * Built in the wind's frame rather than the world's, so rotating the bearing rotates the whole
 * field rigidly and the tree has no preferred compass direction of its own.
 */
function swayAxis(downwind: Vec3, mode: NodeMode): Vec3 {
  const c = Math.cos(mode.azimuthRad);
  const s = Math.sin(mode.azimuthRad);
  // Rotate the downwind direction about up by the node's azimuth.
  const bend: Vec3 = [
    downwind[0] * c - downwind[1] * s,
    downwind[0] * s + downwind[1] * c,
    downwind[2],
  ];
  const inPlane = cross(VEC3_UP, bend);
  return normalize(
    [inPlane[0], inPlane[1], inPlane[2] + mode.twist * magnitude(inPlane)],
    [1, 0, 0],
  );
}

/**
 * Per-node transforms at time `t`, parent-composed down the tree.
 *
 * Node `i` rotates about its own rest position by a steady downwind lean plus its own
 * oscillation, each saturated smoothly against its share of the node's angular limit; that
 * rotation is then composed with its parent's world transform. The root is the anchor: its
 * transform is exactly identity always, so the trunk base never moves. A tree that slides is the
 * classic bug, and this rules it out structurally rather than by tuning.
 *
 * Deterministic: the same `(rig, t, settings)` returns bit-identical numbers, and the returned
 * array is fresh each call — nothing is cached between frames, and nothing is integrated.
 */
export function deform(rig: MotionRig, t: number, settings: WindSettings): NodeTransform[] {
  const gust = sampleWind(settings.strength, settings.bearingDeg, t);
  const windMagnitude = magnitude(gust);
  // Rotating about `up × downwind` tips a vertical node towards the wind. An axis swap here
  // sways the tree sideways into the ground, which is why the direction test exists.
  const downwind = downwindOf(gust, [0, 1, 0]);
  const leanAxis = normalize(cross(VEC3_UP, downwind), [1, 0, 0]);
  const time = Number.isFinite(t) ? t : 0;
  const modes = nodeModes(rig);
  const transforms: NodeTransform[] = [];

  for (let i = 0; i < rig.nodes.length; i += 1) {
    const node = rig.nodes[i];
    const mode = modes[i];
    if (node === undefined || mode === undefined || node.parent < 0) {
      transforms.push(IDENTITY_TRANSFORM);
      continue;
    }
    const parent = transforms[node.parent] ?? IDENTITY_TRANSFORM;
    const budget = angleBudget(node);
    const lean = softLimit(rawLean(node, mode, windMagnitude), budget.mean);
    const sway = softLimit(
      rawSway(node, mode, windMagnitude, turbulentLoad(mode, time)),
      budget.oscillation,
    );
    // Two rotations, not one: the lean is downwind for every node, the sway is in the node's own
    // plane. Sharing a single axis is what made the whole crown move as one flat sheet.
    const local = quatMultiply(
      quatFromAxisAngle(leanAxis, lean),
      quatFromAxisAngle(swayAxis(downwind, mode), sway),
    );
    const pivot = node.position;
    // Rotate about the node's own rest position, then carry the parent's transform.
    const pivotOffset = subtract(pivot, quatRotate(local, pivot));
    transforms.push({
      rotation: quatMultiply(parent.rotation, local),
      translation: add(quatRotate(parent.rotation, pivotOffset), parent.translation),
    });
  }
  return transforms;
}

/** Applies a node transform to a canonical position. */
export function applyTransform(transform: NodeTransform, position: Vec3): Vec3 {
  return add(quatRotate(transform.rotation, position), transform.translation);
}

/** How far a node's own rest position moves under a set of transforms, metres. */
export function nodeDisplacement(
  rig: MotionRig,
  transforms: readonly NodeTransform[],
  index: number,
): Vec3 {
  const node = rig.nodes[index];
  const transform = transforms[index];
  if (node === undefined || transform === undefined) return VEC3_ZERO;
  return subtract(applyTransform(transform, node.position), node.position);
}

/**
 * Displaces canonical splat positions into `out`, leaving `positions` untouched.
 *
 * `positions` is the immutable canonical copy: writing to it would make deformation permanent
 * and compound across snapshot rebuilds, silently destroying the measured geometry. That is why
 * aliasing `out` onto `positions` throws rather than being tolerated.
 */
export function deformPositions(
  positions: Float32Array,
  assignment: Uint16Array,
  transforms: readonly NodeTransform[],
  out?: Float32Array,
  flutter?: FlutterField,
): Float32Array {
  if (out === positions)
    throw new Error("deformPositions: canonical positions must not be the output");
  const count = Math.min(Math.floor(positions.length / 3), assignment.length);
  const target = out ?? new Float32Array(positions.length);
  // Hoisted, and checked for `still` rather than for presence: at calm there is no flutter at
  // all, and the inner loop must not so much as add a zero — `-0 + 0` is `+0`, and the restore
  // path's byte-identity guarantee is over bytes.
  const fluttering = flutter !== undefined && !flutter.still;
  for (let i = 0; i < count; i += 1) {
    const base = i * 3;
    const node = assignment[i] ?? 0;
    const transform = transforms[node] ?? IDENTITY_TRANSFORM;
    const moved = applyTransform(transform, [
      positions[base] ?? 0,
      positions[base + 1] ?? 0,
      positions[base + 2] ?? 0,
    ]);
    if (fluttering) {
      const offset = splatFlutter(i, node, flutter);
      // `splatFlutter` returns the shared zero tuple for a node that does not flutter, and
      // `VEC3_ZERO[0]` is `+0`, so this is still a no-op on the bytes for a trunk splat.
      if (offset !== VEC3_ZERO) {
        target[base] = moved[0] + offset[0];
        target[base + 1] = moved[1] + offset[1];
        target[base + 2] = moved[2] + offset[2];
        continue;
      }
    }
    target[base] = moved[0];
    target[base + 1] = moved[1];
    target[base + 2] = moved[2];
  }
  return target;
}
