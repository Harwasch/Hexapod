/**
 * Bounded, deterministic deformation of a skeleton rig under wind.
 *
 * Every transform is computed from the rig's rest pose, never from the previous frame, so error
 * cannot accumulate and `strength = 0` returns the canonical pose exactly rather than nearly.
 */

import { fbm1d, FBM_MAX_SLOPE, hashString } from "./noise";
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
export const BEND_GAIN = 0.3;
/** Fractional amplitude wobble that gives each node its own life. Stays below 1, so never negative. */
export const FLUTTER = 0.35;
/** Noise coordinates per second for the flutter term. Faster than the gust: leaves quiver. */
const FLUTTER_RATE = 0.9;
const FLUTTER_SEED = 0x5eed_f17;
/** Spreads per-node flutter phases across the noise lattice. */
const PHASE_SPREAD = 997;

/** Deterministic noise phase for a node, from its id. Two nodes with the same id move together. */
export function nodePhase(node: SkeletonNode): number {
  return (hashString(node.id) / 0x1_0000_0000) * PHASE_SPREAD;
}

/**
 * The largest local bend angle a node can reach at this strength, radians. Exact, not sampled:
 * both the gust and the flutter noise are bounded, so the worst case is closed-form.
 */
export function maxNodeAngle(node: SkeletonNode, settings: WindSettings): number {
  const raw = (BEND_GAIN * maxWindMagnitude(settings.strength) * (1 + FLUTTER)) / node.stiffness;
  return softLimit(raw, nodeAngleLimit(node));
}

/**
 * Bound on `|dθ/dt|` for a node, radians per second. `softLimit` is a contraction (`|f'| ≤ 1`),
 * so the bound on the unlimited angle carries through.
 */
export function maxNodeAngleRate(node: SkeletonNode, settings: WindSettings): number {
  const gustTerm = maxWindSlope(settings.strength) * (1 + FLUTTER);
  const flutterTerm = maxWindMagnitude(settings.strength) * FLUTTER * FBM_MAX_SLOPE * FLUTTER_RATE;
  return (BEND_GAIN * (gustTerm + flutterTerm)) / node.stiffness;
}

/**
 * Per-node transforms at time `t`, parent-composed down the tree.
 *
 * Node `i` rotates about its own rest position by an angle proportional to the wind magnitude
 * and inversely proportional to its stiffness, smoothly saturated at its angular limit; that
 * rotation is then composed with its parent's world transform. The root is the anchor: its
 * transform is exactly identity always, so the trunk base never moves. A tree that slides is the
 * classic bug, and this rules it out structurally rather than by tuning.
 *
 * Deterministic: the same `(rig, t, settings)` returns bit-identical numbers, and the returned
 * array is fresh each call — nothing is cached between frames.
 */
export function deform(rig: MotionRig, t: number, settings: WindSettings): NodeTransform[] {
  const gust = sampleWind(settings.strength, settings.bearingDeg, t);
  const windMagnitude = magnitude(gust);
  // Rotating about `up × downwind` tips a vertical node towards the wind. An axis swap here
  // sways the tree sideways into the ground, which is why the direction test exists.
  const axis = normalize(cross(VEC3_UP, gust), [1, 0, 0]);
  const time = Number.isFinite(t) ? t : 0;
  const transforms: NodeTransform[] = [];

  for (const node of rig.nodes) {
    if (node.parent < 0) {
      transforms.push(IDENTITY_TRANSFORM);
      continue;
    }
    const parent = transforms[node.parent] ?? IDENTITY_TRANSFORM;
    const flutter = 1 + FLUTTER * fbm1d(time * FLUTTER_RATE + nodePhase(node), FLUTTER_SEED);
    const raw = (BEND_GAIN * windMagnitude * flutter) / node.stiffness;
    const local = quatFromAxisAngle(axis, softLimit(raw, nodeAngleLimit(node)));
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
): Float32Array {
  if (out === positions)
    throw new Error("deformPositions: canonical positions must not be the output");
  const count = Math.min(Math.floor(positions.length / 3), assignment.length);
  const target = out ?? new Float32Array(positions.length);
  for (let i = 0; i < count; i += 1) {
    const base = i * 3;
    const transform = transforms[assignment[i] ?? 0] ?? IDENTITY_TRANSFORM;
    const moved = applyTransform(transform, [
      positions[base] ?? 0,
      positions[base + 1] ?? 0,
      positions[base + 2] ?? 0,
    ]);
    target[base] = moved[0];
    target[base + 1] = moved[1];
    target[base + 2] = moved[2];
  }
  return target;
}
