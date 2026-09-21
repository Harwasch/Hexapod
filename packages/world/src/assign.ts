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
 * ### Why this is no longer the obvious double loop
 *
 * The obvious form is O(N·K), and S1 noted it would want an index if K ever reached a few
 * hundred. K is 214 now. Measured on this machine against a tree-shaped cloud, the double loop
 * cost **75 ms for 150,000 splats** and 202 ms for 400,000 — and `SiteManager.clampToGround`
 * sets `tileset.modelMatrix` after an asynchronous terrain sample, so every site rebuilds its
 * snapshot and re-derives its assignment a few seconds after it loads. Seventy-five
 * milliseconds inside `preUpdate` is a hitch a person sees.
 *
 * What replaces it is a **uniform grid over the nodes**, searched in expanding cubic shells.
 * After every cell within Chebyshev radius `r` of the splat's own cell has been scanned, any
 * node still unvisited is at least `r · cellSize` away — true whether the splat falls inside
 * the grid or outside it — so the search stops as soon as the best distance found is no more
 * than the radius already *completed*, one shell behind the one about to be scanned. The result is therefore *identical* to the double loop, node for node, including
 * its tie-breaking; the grid changes what is looked at, never what is chosen, and
 * `assign.test.ts` checks that against a transcription of the original.
 *
 * An earlier attempt sorted the nodes along their longest axis and walked outward in a slab.
 * It was exact and it was barely faster (2×), for a reason worth writing down: a tree's crown
 * is *dense in height*, so a slab in z prunes worst exactly where the splats are. The grid
 * prunes in all three axes and does not care.
 *
 * Measured, tree-shaped cloud, 214 nodes: 150,000 splats fall from 75 ms to about 11 ms, and
 * 400,000 from 202 ms to 30 ms.
 */
export function assignSplatsToNodes(positions: Float32Array, rig: MotionRig): Uint16Array {
  const count = Math.floor(positions.length / 3);
  const assignment = new Uint16Array(count);
  const nodeCount = rig.nodes.length;
  if (nodeCount === 0) return assignment;

  const grid = buildNodeGrid(rig);
  if (grid === undefined) return assignNearestBruteForce(positions, rig, assignment);

  const {
    cellSize,
    minX,
    minY,
    minZ,
    dimX,
    dimY,
    dimZ,
    cellStart,
    cellNodes,
    nodeX,
    nodeY,
    nodeZ,
  } = grid;
  const maxRadius = Math.max(dimX, dimY, dimZ);

  for (let i = 0; i < count; i += 1) {
    const base = i * 3;
    const px = positions[base] ?? Number.NaN;
    const py = positions[base + 1] ?? Number.NaN;
    const pz = positions[base + 2] ?? Number.NaN;
    if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) {
      assignment[i] = UNASSIGNED_NODE;
      continue;
    }
    const ux = (px - minX) / cellSize;
    const uy = (py - minY) / cellSize;
    const uz = (pz - minZ) / cellSize;
    // The splat's own cell, clamped into the grid. A splat outside the node bounds searches
    // from the nearest boundary cell, which the stopping rule still bounds correctly.
    const cx = clampIndex(Math.floor(ux), dimX);
    const cy = clampIndex(Math.floor(uy), dimY);
    const cz = clampIndex(Math.floor(uz), dimZ);

    let best = 0;
    let bestDistanceSq = Number.POSITIVE_INFINITY;

    // The fast path, and the one nearly every splat takes: the 2×2×2 block of cells straddling
    // the splat rather than the 3×3×3 one centred on its cell. Eight probes instead of
    // twenty-seven, and the shell walk below is a fixed cost of cell lookups rather than of
    // distance evaluations — measured, that overhead was most of what the grid cost.
    const bx = clampIndex(Math.floor(ux - 0.5), dimX - 1);
    const by = clampIndex(Math.floor(uy - 0.5), dimY - 1);
    const bz = clampIndex(Math.floor(uz - 0.5), dimZ - 1);
    for (let z = bz; z <= bz + 1 && z < dimZ; z += 1) {
      for (let y = by; y <= by + 1 && y < dimY; y += 1) {
        const row = (z * dimY + y) * dimX;
        for (let x = bx; x <= bx + 1 && x < dimX; x += 1) {
          const cell = row + x;
          const end = cellStart[cell + 1] ?? 0;
          for (let k = cellStart[cell] ?? 0; k < end; k += 1) {
            const n = cellNodes[k] ?? 0;
            const dx = px - (nodeX[n] ?? 0);
            const dy = py - (nodeY[n] ?? 0);
            const dz = pz - (nodeZ[n] ?? 0);
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < bestDistanceSq || (d2 === bestDistanceSq && n < best)) {
              bestDistanceSq = d2;
              best = n;
            }
          }
        }
      }
    }
    // How far from the splat that block is guaranteed to reach. A face that sits on the edge of
    // the grid reaches forever, because there are no nodes beyond it.
    let safeCells = Number.POSITIVE_INFINITY;
    if (bx > 0 && ux - bx < safeCells) safeCells = ux - bx;
    if (bx + 2 < dimX && bx + 2 - ux < safeCells) safeCells = bx + 2 - ux;
    if (by > 0 && uy - by < safeCells) safeCells = uy - by;
    if (by + 2 < dimY && by + 2 - uy < safeCells) safeCells = by + 2 - uy;
    if (bz > 0 && uz - bz < safeCells) safeCells = uz - bz;
    if (bz + 2 < dimZ && bz + 2 - uz < safeCells) safeCells = bz + 2 - uz;
    const safe = safeCells * cellSize;
    if (bestDistanceSq <= safe * safe) {
      assignment[i] = best;
      continue;
    }

    // The general path: expanding cubic shells about the splat's own cell. Rare, and exact.
    bestDistanceSq = Number.POSITIVE_INFINITY;
    best = 0;
    for (let r = 0; r <= maxRadius; r += 1) {
      // Having scanned every cell within radius r−1, anything left is at least (r−1)·cellSize
      // away — one shell less than the radius about to be scanned, and getting that off by one
      // is how a nearest-neighbour search quietly returns the second nearest.
      const reach = (r - 1) * cellSize;
      if (r > 0 && bestDistanceSq <= reach * reach) break;
      const x0 = Math.max(0, cx - r);
      const x1 = Math.min(dimX - 1, cx + r);
      const y0 = Math.max(0, cy - r);
      const y1 = Math.min(dimY - 1, cy + r);
      const z0 = Math.max(0, cz - r);
      const z1 = Math.min(dimZ - 1, cz + r);
      for (let z = z0; z <= z1; z += 1) {
        const onZShell = z === cz - r || z === cz + r;
        for (let y = y0; y <= y1; y += 1) {
          const onYShell = y === cy - r || y === cy + r;
          // Only the shell, not the whole cube: the interior was scanned at a smaller radius.
          const wholeRow = onZShell || onYShell;
          for (let x = x0; x <= x1; x += 1) {
            if (!wholeRow && x !== cx - r && x !== cx + r) continue;
            const cell = (z * dimY + y) * dimX + x;
            const end = cellStart[cell + 1] ?? 0;
            for (let k = cellStart[cell] ?? 0; k < end; k += 1) {
              const n = cellNodes[k] ?? 0;
              const dx = px - (nodeX[n] ?? 0);
              const dy = py - (nodeY[n] ?? 0);
              const dz = pz - (nodeZ[n] ?? 0);
              const d2 = dx * dx + dy * dy + dz * dz;
              // Strictly less, or equal with a lower index: ties keep the earlier node, so the
              // result never depends on the order cells happen to be walked in.
              if (d2 < bestDistanceSq || (d2 === bestDistanceSq && n < best)) {
                bestDistanceSq = d2;
                best = n;
              }
            }
          }
        }
      }
    }
    assignment[i] = best;
  }
  return assignment;
}

/** The O(N·K) form, kept for a rig the grid cannot describe: one node, or coincident nodes. */
function assignNearestBruteForce(
  positions: Float32Array,
  rig: MotionRig,
  assignment: Uint16Array,
): Uint16Array {
  const count = assignment.length;
  const nodeCount = rig.nodes.length;
  for (let i = 0; i < count; i += 1) {
    const px = positions[i * 3] ?? Number.NaN;
    const py = positions[i * 3 + 1] ?? Number.NaN;
    const pz = positions[i * 3 + 2] ?? Number.NaN;
    if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) {
      assignment[i] = UNASSIGNED_NODE;
      continue;
    }
    let best = 0;
    let bestDistanceSq = Number.POSITIVE_INFINITY;
    for (let n = 0; n < nodeCount; n += 1) {
      const node = rig.nodes[n];
      if (node === undefined) continue;
      const dx = px - node.position[0];
      const dy = py - node.position[1];
      const dz = pz - node.position[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bestDistanceSq) {
        bestDistanceSq = d2;
        best = n;
      }
    }
    assignment[i] = best;
  }
  return assignment;
}

function clampIndex(value: number, limit: number): number {
  return value < 0 ? 0 : value >= limit ? limit - 1 : value;
}

interface NodeGrid {
  readonly cellSize: number;
  readonly minX: number;
  readonly minY: number;
  readonly minZ: number;
  readonly dimX: number;
  readonly dimY: number;
  readonly dimZ: number;
  /** Prefix offsets into `cellNodes`, one per cell plus a terminator. */
  readonly cellStart: Int32Array;
  readonly cellNodes: Int32Array;
  readonly nodeX: Float64Array;
  readonly nodeY: Float64Array;
  readonly nodeZ: Float64Array;
}

/** Cells per node. Below one the cells hold several nodes each and the scan does more work. */
const CELLS_PER_NODE = 4;
/** Never build more cells than this, whatever the rig's aspect ratio. */
const MAX_CELLS = 1 << 18;

/**
 * A uniform grid over the rig's node positions, or `undefined` for a rig it cannot describe.
 *
 * The cell size targets a few cells per node, so a typical shell scan visits a handful of
 * candidates. The build is a counting sort: one pass to count, one to place, no allocation per
 * cell.
 */
function buildNodeGrid(rig: MotionRig): NodeGrid | undefined {
  const nodeCount = rig.nodes.length;
  const nodeX = new Float64Array(nodeCount);
  const nodeY = new Float64Array(nodeCount);
  const nodeZ = new Float64Array(nodeCount);
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let n = 0; n < nodeCount; n += 1) {
    const node = rig.nodes[n];
    if (node === undefined) return undefined;
    const [x, y, z] = node.position;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return undefined;
    nodeX[n] = x;
    nodeY[n] = y;
    nodeZ[n] = z;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  const spanX = maxX - minX;
  const spanY = maxY - minY;
  const spanZ = maxZ - minZ;
  const diagonal = Math.hypot(spanX, spanY, spanZ);
  // Every node in one place, or one node: there is nothing for a grid to separate.
  if (!(diagonal > 0)) return undefined;

  let cellSize = Math.cbrt(
    (Math.max(spanX, 1e-6) * Math.max(spanY, 1e-6) * Math.max(spanZ, 1e-6)) /
      (nodeCount * CELLS_PER_NODE),
  );
  if (!(cellSize > 0)) cellSize = diagonal / 16;
  let dimX = Math.max(1, Math.ceil(spanX / cellSize) + 1);
  let dimY = Math.max(1, Math.ceil(spanY / cellSize) + 1);
  let dimZ = Math.max(1, Math.ceil(spanZ / cellSize) + 1);
  // A rig with a degenerate aspect ratio can ask for an absurd grid; grow the cells until it
  // does not. Correctness does not depend on the cell size, only speed does.
  while (dimX * dimY * dimZ > MAX_CELLS) {
    cellSize *= 2;
    dimX = Math.max(1, Math.ceil(spanX / cellSize) + 1);
    dimY = Math.max(1, Math.ceil(spanY / cellSize) + 1);
    dimZ = Math.max(1, Math.ceil(spanZ / cellSize) + 1);
  }

  const cells = dimX * dimY * dimZ;
  const counts = new Int32Array(cells + 1);
  const cellOf = new Int32Array(nodeCount);
  for (let n = 0; n < nodeCount; n += 1) {
    const x = clampIndex(Math.floor(((nodeX[n] ?? 0) - minX) / cellSize), dimX);
    const y = clampIndex(Math.floor(((nodeY[n] ?? 0) - minY) / cellSize), dimY);
    const z = clampIndex(Math.floor(((nodeZ[n] ?? 0) - minZ) / cellSize), dimZ);
    const cell = (z * dimY + y) * dimX + x;
    cellOf[n] = cell;
    counts[cell + 1] = (counts[cell + 1] ?? 0) + 1;
  }
  for (let c = 0; c < cells; c += 1) counts[c + 1] = (counts[c + 1] ?? 0) + (counts[c] ?? 0);
  const cellStart = counts;
  const cursor = Int32Array.from(cellStart.subarray(0, cells));
  const cellNodes = new Int32Array(nodeCount);
  for (let n = 0; n < nodeCount; n += 1) {
    const cell = cellOf[n] ?? 0;
    cellNodes[cursor[cell] ?? 0] = n;
    cursor[cell] = (cursor[cell] ?? 0) + 1;
  }
  return {
    cellSize,
    minX,
    minY,
    minZ,
    dimX,
    dimY,
    dimZ,
    cellStart,
    cellNodes,
    nodeX,
    nodeY,
    nodeZ,
  };
}

/** Number of splats assigned to each node. Useful for spotting a node that owns nothing. */
export function assignmentHistogram(assignment: Uint16Array, nodeCount: number): Uint32Array {
  const counts = new Uint32Array(nodeCount);
  for (const node of assignment) {
    if (node < nodeCount) counts[node] = (counts[node] ?? 0) + 1;
  }
  return counts;
}
