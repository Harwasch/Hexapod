/**
 * Numbers that make the model's artifacts arguable instead of impressionistic.
 *
 * The sort-staleness metric in particular exists because the splat sorter reads canonical
 * positions we never touch, so displaced splats carry stale draw-order keys. That is an
 * artifact to measure, not a correctness bug — and measuring it in CI beats debating it.
 */

import { maxNodeAngle, maxNodeAngleRate } from "./deform";
import { ancestorsOf, type MotionRig } from "./rig";
import { distance } from "./vec";
import { type WindSettings } from "./wind";

/**
 * Worst-case displacement of each node's rest position, metres — a proven upper bound, not a
 * sampled maximum.
 *
 * Each ancestor rotation about pivot `a` by `θ` moves a point at distance `r` by
 * `2r·sin(θ/2) ≤ rθ`, and composing isometries lets those errors add, so the sum over the
 * ancestor chain bounds the total. Monotone in strength by construction, because every `θ` is.
 */
export function maxNodeDisplacements(rig: MotionRig, settings: WindSettings): number[] {
  return rig.nodes.map((node, index) => {
    let bound = 0;
    for (const ancestor of ancestorsOf(rig, index)) {
      const joint = rig.nodes[ancestor];
      // The root is the anchor and never rotates, so it contributes nothing.
      if (joint === undefined || joint.parent < 0) continue;
      bound += maxNodeAngle(joint, settings) * distance(node.position, joint.position);
    }
    return bound;
  });
}

/** The largest displacement any node can reach at this wind, metres. */
export function maxDisplacement(rig: MotionRig, settings: WindSettings): number {
  let worst = 0;
  for (const bound of maxNodeDisplacements(rig, settings)) {
    if (bound > worst) worst = bound;
  }
  return worst;
}

/** Worst-case speed of each node, metres per second. Same construction as the displacement bound. */
export function maxNodeSpeeds(rig: MotionRig, settings: WindSettings): number[] {
  return rig.nodes.map((node, index) => {
    let bound = 0;
    for (const ancestor of ancestorsOf(rig, index)) {
      const joint = rig.nodes[ancestor];
      if (joint === undefined || joint.parent < 0) continue;
      bound += maxNodeAngleRate(joint, settings) * distance(node.position, joint.position);
    }
    return bound;
  });
}

/** The largest speed any node can reach at this wind, metres per second. */
export function maxDeformSpeed(rig: MotionRig, settings: WindSettings): number {
  let worst = 0;
  for (const bound of maxNodeSpeeds(rig, settings)) {
    if (bound > worst) worst = bound;
  }
  return worst;
}

/**
 * Median splat extent, metres: per splat the largest of its scale components, then the median.
 * The largest component is the right summary because it is what decides when two splats visibly
 * swap depth order.
 */
export function medianGaussianScale(scales: Float32Array, componentsPerSplat = 3): number {
  const stride = Math.max(1, Math.floor(componentsPerSplat));
  const count = Math.floor(scales.length / stride);
  if (count === 0) return 0;
  const extents = new Float64Array(count);
  for (let i = 0; i < count; i += 1) {
    let largest = 0;
    for (let c = 0; c < stride; c += 1) {
      const value = Math.abs(scales[i * stride + c] ?? 0);
      if (value > largest) largest = value;
    }
    extents[i] = largest;
  }
  extents.sort();
  const mid = count >> 1;
  if (count % 2 === 1) return extents[mid] ?? 0;
  return ((extents[mid - 1] ?? 0) + (extents[mid] ?? 0)) / 2;
}

/**
 * How far the draw order can have gone stale, in splat radii: `maxDisplacement /
 * medianGaussianScale`.
 *
 * Below 1 a splat has not moved its own width and the sort key it was given is still broadly
 * right. Well above 1 the painter's order no longer matches the geometry and blending artifacts
 * become likely. Returns `NaN` for a non-positive scale rather than pretending to a number.
 */
export function sortStaleness(
  rig: MotionRig,
  settings: WindSettings,
  medianGaussianScaleM: number,
): number {
  if (!Number.isFinite(medianGaussianScaleM) || medianGaussianScaleM <= 0) return Number.NaN;
  return maxDisplacement(rig, settings) / medianGaussianScaleM;
}

/**
 * The staleness above which draw order is expected to read as wrong. A hypothesis, not a
 * measurement: headless GL here is SwiftShader, so only a human eye on real hardware can settle
 * where the threshold actually sits.
 */
export const SORT_STALENESS_NOTICEABLE = 1;
