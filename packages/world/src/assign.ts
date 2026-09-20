/**
 * Assigning splats to skeleton nodes: nearest node wins.
 *
 * This is the join that replaces an index list. It is geometric, so it survives re-tiling, and
 * it is order-independent, so a shuffled input gives the same mapping.
 */

import { type MotionRig } from "./rig";

/** Returned for a splat whose position is not finite. The root never moves, so it is inert. */
export const UNASSIGNED_NODE = 0;

/**
 * Nearest-skeleton-node assignment, one `Uint16` per splat.
 *
 * `positions` is a flat `[x, y, z, x, y, z, …]` array in the rig's local ENU frame. Ties go to
 * the lowest node index, which is what makes the result independent of the order the splats
 * arrive in — an important property, because splat order is a tiler detail we do not control.
 *
 * Cost is O(N·K). With K in the 20–60 range the inner loop is a handful of FLOPs and ~150k
 * splats costs a few million operations — a one-off at attach time, well under a frame. If K
 * ever grows past a few hundred, or assignment has to run per frame, this wants a uniform grid
 * or KD-tree over the nodes; it does not need one today.
 */
export function assignSplatsToNodes(positions: Float32Array, rig: MotionRig): Uint16Array {
  const count = Math.floor(positions.length / 3);
  const assignment = new Uint16Array(count);
  const nodeCount = rig.nodes.length;
  if (nodeCount === 0) return assignment;

  // Flatten node positions once: the inner loop is hot and should not chase object properties.
  const nodeX = new Float64Array(nodeCount);
  const nodeY = new Float64Array(nodeCount);
  const nodeZ = new Float64Array(nodeCount);
  for (let n = 0; n < nodeCount; n += 1) {
    const node = rig.nodes[n];
    if (node === undefined) continue;
    nodeX[n] = node.position[0];
    nodeY[n] = node.position[1];
    nodeZ[n] = node.position[2];
  }

  for (let i = 0; i < count; i += 1) {
    const base = i * 3;
    const px = positions[base] ?? Number.NaN;
    const py = positions[base + 1] ?? Number.NaN;
    const pz = positions[base + 2] ?? Number.NaN;
    if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) {
      assignment[i] = UNASSIGNED_NODE;
      continue;
    }
    let best = 0;
    let bestDistanceSq = Number.POSITIVE_INFINITY;
    for (let n = 0; n < nodeCount; n += 1) {
      const dx = px - (nodeX[n] ?? 0);
      const dy = py - (nodeY[n] ?? 0);
      const dz = pz - (nodeZ[n] ?? 0);
      const d2 = dx * dx + dy * dy + dz * dz;
      // Strictly less than: ties keep the earlier node, so the result never depends on order.
      if (d2 < bestDistanceSq) {
        bestDistanceSq = d2;
        best = n;
      }
    }
    assignment[i] = best;
  }
  return assignment;
}

/** Number of splats assigned to each node. Useful for spotting a node that owns nothing. */
export function assignmentHistogram(assignment: Uint16Array, nodeCount: number): Uint32Array {
  const counts = new Uint32Array(nodeCount);
  for (const node of assignment) {
    if (node < nodeCount) counts[node] = (counts[node] ?? 0) + 1;
  }
  return counts;
}
