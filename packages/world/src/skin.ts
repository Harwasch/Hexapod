/**
 * Smooth skinning: every splat follows a blend of up to four rig nodes, not the one nearest.
 *
 * Nearest-node binding (`assignSplatsToNodes`) moves each splat rigidly with one joint. Joint
 * `i`'s transform rotates about `i`'s rest position and carries its parent's, so a splat just
 * either side of the Voronoi boundary between two joints is moved by two different rigid
 * motions: the displacement field jumps there by `θ·r` (the joint's bend times the distance
 * from the pivot), and a limb bends as a chain of rigid sticks with a visible seam at every
 * boundary. Blending the nodes' transforms by weights that vary *continuously* with position
 * removes both: the displacement field becomes a continuous function of the splat's rest
 * position, and a limb bends as a curve.
 *
 * ### Weights
 *
 * The `K = 4` nearest nodes, weighted by the **modified Shepard** (Franke & Nielson 1980)
 * inverse-distance weight with its radius at the fifth-nearest node:
 *
 * ```text
 * w_k = ((R − d_k) / (R · d_k))²,   R = d_5,   normalised to sum 1
 * ```
 *
 * `w_k` falls to zero as `d_k → R`, so a node enters or leaves a splat's four exactly when its
 * weight is zero: the weights, and the field, are continuous everywhere, including where the
 * set of four changes — which plain `1/d²` over the nearest four is not. At a node (`d = 0`)
 * that node's weight is 1: the field interpolates the joints' own motion. A rig of five nodes
 * or fewer has no fifth node, and `R = ∞` gives plain Shepard `1/d²` over all of them.
 *
 * The four are the four *nearest*, whatever limb they sit on. A rule that kept only the
 * nearest node's own limb would be discontinuous wherever two limbs' territories meet — which
 * is exactly where the seams are. Along a limb its own joints are the nearest, so a splat on a
 * limb blends that limb's joints; where two twigs' foliage touches it blends both, which is
 * what continuity requires.
 *
 * ### Linear blending
 *
 * The blended displacement is `Σ_k w_k·(T_k(x) − x)` — linear blend skinning (Magnenat-
 * Thalmann et al. 1988). Its known failure is the "candy-wrapper" collapse under large twist,
 * which dual-quaternion skinning (Kavan et al. 2008) exists to avoid. Neither applies here: no
 * joint of this model twists about its own limb (every rotation axis is across it), and a
 * joint's bend is a few hundredths of a radian, where LBS shortens a blended lever arm by
 * `1 − cos(Δθ/2) ≈ Δθ²/8` — 0.3 mm at 1 m for a 0.05 rad difference. LBS is also linear in
 * the per-node data, which is what lets the vertex shader apply it as four matrix rows.
 *
 * ### Weights as bytes
 *
 * Weights are quantised to 10-bit integers summing to exactly 1023, so the CPU and GPU paths
 * blend the same numbers, a splat bound to one node has weight exactly 1, and three of the four
 * fit one 32-bit texel channel (the first is the remainder). A quantisation step moves a splat
 * by 1/1023 of the difference between two neighbouring joints' motions — 0.1 mm at a gale — where
 * 8 bits left 0.4 mm steps. The displacement form
 * `x + Σ w·(T(x) − x)` keeps calm exact: a node at rest contributes exactly zero.
 *
 * Order-independent and deterministic: a splat's four depend on its own position alone, ties
 * going to the lower node index, and slot 0 is always the node `assignSplatsToNodes` returns.
 */

import { UNASSIGNED_NODE } from "./assign";
import { type MotionRig } from "./rig";

/** Nodes a splat blends. Four fit one `RGBA32UI` texel with the weights and flutter hash. */
export const SKIN_INFLUENCES = 4;

/** Weights are integers summing to this. */
export const SKIN_WEIGHT_TOTAL = 1023;

/** Each splat's nodes and weights, `SKIN_INFLUENCES` per splat, nearest first. */
export interface SplatSkin {
  /** Node indices. Slot 0 is the nearest node; a slot with weight 0 is unused. */
  readonly nodes: Uint16Array;
  /** Integer weights, summing to {@link SKIN_WEIGHT_TOTAL} per splat. */
  readonly weights: Uint16Array;
}

/** Splats in a skin. */
export function skinCount(skin: SplatSkin): number {
  return Math.floor(skin.nodes.length / SKIN_INFLUENCES);
}

/** A rigid skin: each splat entirely on its one assigned node. The nearest-node binding. */
export function rigidSkin(assignment: Uint16Array): SplatSkin {
  const nodes = new Uint16Array(assignment.length * SKIN_INFLUENCES);
  const weights = new Uint16Array(assignment.length * SKIN_INFLUENCES);
  for (let i = 0; i < assignment.length; i += 1) {
    const node = assignment[i] ?? 0;
    for (let k = 0; k < SKIN_INFLUENCES; k += 1) nodes[i * SKIN_INFLUENCES + k] = node;
    weights[i * SKIN_INFLUENCES] = SKIN_WEIGHT_TOTAL;
  }
  return { nodes, weights };
}

/** Slot 0 of every splat: its nearest node, as `assignSplatsToNodes` gives it. */
export function skinPrimary(skin: SplatSkin): Uint16Array {
  const count = skinCount(skin);
  const out = new Uint16Array(count);
  for (let i = 0; i < count; i += 1) out[i] = skin.nodes[i * SKIN_INFLUENCES] ?? 0;
  return out;
}

/** Whether any node splat `i` has weight on is flagged in `nodeMoves`. */
export function skinMoves(skin: SplatSkin, i: number, nodeMoves: Uint8Array): boolean {
  const base = i * SKIN_INFLUENCES;
  for (let k = 0; k < SKIN_INFLUENCES; k += 1) {
    if ((skin.weights[base + k] ?? 0) === 0) continue;
    if ((nodeMoves[skin.nodes[base + k] ?? 0] ?? 0) !== 0) return true;
  }
  return false;
}

/** Concatenates skins in order, into fresh arrays. */
export function concatSkins(parts: readonly SplatSkin[], starts: readonly number[], count: number) {
  const nodes = new Uint16Array(count * SKIN_INFLUENCES);
  const weights = new Uint16Array(count * SKIN_INFLUENCES);
  parts.forEach((part, j) => {
    const at = (starts[j] ?? 0) * SKIN_INFLUENCES;
    nodes.set(part.nodes, at);
    weights.set(part.weights, at);
  });
  return { nodes, weights } satisfies SplatSkin;
}

/** Per rig, each grid cell's candidate list. Weak: a rig let go takes its lists with it. */
const CANDIDATE_LISTS = new WeakMap<MotionRig, Map<number, Int32Array>>();

/** Candidates kept while scanning: the four and the fifth, which sets the radius. */
const KEPT = SKIN_INFLUENCES + 1;
/** Cell coordinates beyond this fall back to scanning every node. */
const CELL_LIMIT = 1 << 16;
/** Cells per node spacing: smaller cells, shorter candidate lists, more lists to build. */
const CELLS_PER_SPACING = 4;

/**
 * The modified Shepard weights for one splat, quantised, from its `KEPT` nearest distances
 * (squared, ascending; `Infinity` for a missing one). Writes `SKIN_INFLUENCES` bytes at `at`.
 * Exported for tests.
 */
export function quantiseSkinWeights(
  distancesSq: ArrayLike<number>,
  weights: Uint16Array,
  at: number,
): void {
  const d0 = distancesSq[0] ?? Number.POSITIVE_INFINITY;
  weights.fill(0, at, at + SKIN_INFLUENCES);
  const far = distancesSq[SKIN_INFLUENCES] ?? Number.POSITIVE_INFINITY;
  if (!(d0 > 0) || !Number.isFinite(d0)) {
    // On a node, or no node at all: entirely the nearest.
    weights[at] = SKIN_WEIGHT_TOTAL;
    return;
  }
  const radius = Math.sqrt(far);
  const raw = new Float64Array(SKIN_INFLUENCES);
  let sum = 0;
  for (let k = 0; k < SKIN_INFLUENCES; k += 1) {
    const d2 = distancesSq[k] ?? Number.POSITIVE_INFINITY;
    if (!Number.isFinite(d2)) continue;
    const d = Math.sqrt(d2);
    // ((R − d)/(R·d))² = (1/d − 1/R)²; with R = ∞, plain 1/d².
    const inverse = 1 / d - (Number.isFinite(radius) ? 1 / radius : 0);
    const w = inverse > 0 ? inverse * inverse : 0;
    raw[k] = w;
    sum += w;
  }
  if (!(sum > 0)) {
    weights[at] = SKIN_WEIGHT_TOTAL;
    return;
  }
  let rest = SKIN_WEIGHT_TOTAL;
  for (let k = 1; k < SKIN_INFLUENCES; k += 1) {
    const q = Math.floor(((raw[k] ?? 0) / sum) * SKIN_WEIGHT_TOTAL + 0.5);
    weights[at + k] = q;
    rest -= q;
  }
  // The nearest carries the largest weight, so the remainder never goes negative.
  weights[at] = rest;
}

/**
 * Binds every splat to its four nearest nodes with modified Shepard weights.
 *
 * `positions` is flat `[x, y, z, …]` in the rig's frame. A non-finite splat is bound entirely
 * to the root, as `assignSplatsToNodes` does.
 *
 * ### The search
 *
 * A uniform grid over the splats, cells half a node spacing wide, each cell's **candidate
 * list** built on first use: every node within `r₅ + 2h` of the cell's centre, `r₅` the
 * fifth-nearest node's distance from the centre and `h` the cell's half-diagonal. For any
 * point `p` in the cell, its fifth-nearest distance is at most `r₅ + h` (the centre's five are
 * all within that of `p`), so any node among `p`'s five is within `r₅ + 2h` of the centre: the
 * list is a superset, and the result is exactly the brute-force answer. `skin.test.ts` checks
 * that against the brute-force form. Measured costs are in docs/LIVING_SURVEY.md.
 */
export function skinSplatsToNodes(positions: Float32Array, rig: MotionRig): SplatSkin {
  const count = Math.floor(positions.length / 3);
  const nodes = new Uint16Array(count * SKIN_INFLUENCES);
  const weights = new Uint16Array(count * SKIN_INFLUENCES);
  const nodeCount = rig.nodes.length;
  if (nodeCount === 0) return { nodes, weights };
  const nodeX = new Float64Array(nodeCount);
  const nodeY = new Float64Array(nodeCount);
  const nodeZ = new Float64Array(nodeCount);
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  let finiteNodes = true;
  rig.nodes.forEach((node, n) => {
    const [x, y, z] = node.position;
    nodeX[n] = x;
    nodeY[n] = y;
    nodeZ[n] = z;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) finiteNodes = false;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  });
  const volume =
    Math.max(maxX - minX, 1e-3) * Math.max(maxY - minY, 1e-3) * Math.max(maxZ - minZ, 1e-3);
  const spacing = Math.cbrt(volume / nodeCount);
  const cell = finiteNodes && spacing > 0 ? spacing / CELLS_PER_SPACING : 0;
  const halfDiagonal = (cell * Math.sqrt(3)) / 2;

  const allNodes = new Int32Array(nodeCount);
  for (let n = 0; n < nodeCount; n += 1) allNodes[n] = n;
  // Candidate lists depend on the rig alone, so every tile bound to one rig shares them.
  let lists = CANDIDATE_LISTS.get(rig);
  if (lists === undefined) {
    lists = new Map<number, Int32Array>();
    CANDIDATE_LISTS.set(rig, lists);
  }
  const cache = lists;
  const best = new Float64Array(KEPT);
  const bestNode = new Int32Array(KEPT);
  const scratch = new Float64Array(KEPT);

  const candidatesFor = (cx: number, cy: number, cz: number): Int32Array => {
    if (nodeCount <= KEPT) return allNodes;
    const key =
      ((cx + CELL_LIMIT) * 2 * CELL_LIMIT + (cy + CELL_LIMIT)) * 2 * CELL_LIMIT + (cz + CELL_LIMIT);
    const cached = cache.get(key);
    if (cached !== undefined) return cached;
    const ox = (cx + 0.5) * cell;
    const oy = (cy + 0.5) * cell;
    const oz = (cz + 0.5) * cell;
    scratch.fill(Number.POSITIVE_INFINITY);
    for (let n = 0; n < nodeCount; n += 1) {
      const dx = ox - (nodeX[n] ?? 0);
      const dy = oy - (nodeY[n] ?? 0);
      const dz = oz - (nodeZ[n] ?? 0);
      insertDistance(scratch, undefined, dx * dx + dy * dy + dz * dz, n);
    }
    // Slack for rounding: a superset is harmless, a missed node is not.
    const reach = Math.sqrt(scratch[KEPT - 1] ?? 0) + 2 * halfDiagonal + 1e-9 * (1 + cell);
    const reachSq = reach * reach;
    const picked: number[] = [];
    for (let n = 0; n < nodeCount; n += 1) {
      const dx = ox - (nodeX[n] ?? 0);
      const dy = oy - (nodeY[n] ?? 0);
      const dz = oz - (nodeZ[n] ?? 0);
      if (dx * dx + dy * dy + dz * dz <= reachSq) picked.push(n);
    }
    const list = Int32Array.from(picked);
    cache.set(key, list);
    return list;
  };

  let lastX = Number.NaN;
  let lastY = Number.NaN;
  let lastZ = Number.NaN;
  let lastList: Int32Array = allNodes;
  for (let i = 0; i < count; i += 1) {
    const px = positions[i * 3] ?? Number.NaN;
    const py = positions[i * 3 + 1] ?? Number.NaN;
    const pz = positions[i * 3 + 2] ?? Number.NaN;
    const at = i * SKIN_INFLUENCES;
    if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) {
      nodes.fill(UNASSIGNED_NODE, at, at + SKIN_INFLUENCES);
      weights[at] = SKIN_WEIGHT_TOTAL;
      continue;
    }
    let candidates: Int32Array = allNodes;
    if (cell > 0) {
      const cx = Math.floor(px / cell);
      const cy = Math.floor(py / cell);
      const cz = Math.floor(pz / cell);
      if (
        Math.abs(cx) < CELL_LIMIT - 1 &&
        Math.abs(cy) < CELL_LIMIT - 1 &&
        Math.abs(cz) < CELL_LIMIT - 1
      ) {
        // Consecutive splats are usually neighbours: skip the map when the cell repeats.
        if (cx !== lastX || cy !== lastY || cz !== lastZ) {
          lastList = candidatesFor(cx, cy, cz);
          lastX = cx;
          lastY = cy;
          lastZ = cz;
        }
        candidates = lastList;
      }
    }
    best.fill(Number.POSITIVE_INFINITY);
    bestNode.fill(0);
    let worst = Number.POSITIVE_INFINITY;
    // eslint-disable-next-line @typescript-eslint/prefer-for-of -- indexed: see checksumPositions
    for (let j = 0; j < candidates.length; j += 1) {
      const n = candidates[j] ?? 0;
      // The same arithmetic as `assignSplatsToNodes`, so slot 0 is its answer bit for bit.
      const dx = px - (nodeX[n] ?? 0);
      const dy = py - (nodeY[n] ?? 0);
      const dz = pz - (nodeZ[n] ?? 0);
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > worst) continue;
      insertDistance(best, bestNode, d2, n);
      worst = best[KEPT - 1] ?? Number.POSITIVE_INFINITY;
    }
    for (let k = 0; k < SKIN_INFLUENCES; k += 1) {
      nodes[at + k] = Number.isFinite(best[k] ?? Number.NaN)
        ? (bestNode[k] ?? 0)
        : (bestNode[0] ?? 0);
    }
    quantiseSkinWeights(best, weights, at);
  }
  return { nodes, weights };
}

/**
 * Inserts `(d2, node)` into an ascending list of `KEPT`, ties to the lower node index. Nodes are
 * offered in ascending index order within one call's scan, but candidate lists are not
 * guaranteed sorted, so the tie is decided explicitly.
 */
function insertDistance(
  distances: Float64Array,
  indices: Int32Array | undefined,
  d2: number,
  node: number,
): void {
  const last = KEPT - 1;
  const worst = distances[last] ?? Number.POSITIVE_INFINITY;
  if (d2 > worst || (d2 === worst && indices !== undefined && node >= (indices[last] ?? 0))) return;
  if (d2 === worst && indices === undefined) return;
  let k = last;
  while (k > 0) {
    const prev = distances[k - 1] ?? Number.POSITIVE_INFINITY;
    const prevNode = indices?.[k - 1] ?? 0;
    if (prev < d2 || (prev === d2 && (indices === undefined || prevNode < node))) break;
    distances[k] = prev;
    if (indices !== undefined) indices[k] = prevNode;
    k -= 1;
  }
  distances[k] = d2;
  if (indices !== undefined) indices[k] = node;
}

/** Splats `start .. start + count` of a skin, as views (no copy). */
export function skinSlice(skin: SplatSkin, start: number, count: number): SplatSkin {
  const from = start * SKIN_INFLUENCES;
  const to = (start + count) * SKIN_INFLUENCES;
  return { nodes: skin.nodes.subarray(from, to), weights: skin.weights.subarray(from, to) };
}

/**
 * What a splat follows: one node (a nearest-node `assignment`) or a skin. The per-frame
 * bookkeeping that asks "does this splat move?" takes either.
 */
export type SplatBinding = Uint16Array | SplatSkin;

/** Splats a binding covers. */
export function bindingCount(binding: SplatBinding): number {
  return binding instanceof Uint16Array ? binding.length : skinCount(binding);
}

/** Whether splat `i` follows any node flagged in `nodeMoves`. */
export function splatMoves(binding: SplatBinding, i: number, nodeMoves: Uint8Array): boolean {
  if (binding instanceof Uint16Array) return (nodeMoves[binding[i] ?? 0] ?? 0) !== 0;
  return skinMoves(binding, i, nodeMoves);
}

/**
 * {@link splatMoves} for every splat at once, into `out` (1 where the splat moves): the form a
 * per-frame, per-splat loop should use.
 */
export function markMovingSplats(
  binding: SplatBinding,
  nodeMoves: Uint8Array,
  out: Uint8Array,
): Uint8Array {
  const count = Math.min(bindingCount(binding), out.length);
  if (binding instanceof Uint16Array) {
    for (let i = 0; i < count; i += 1) out[i] = (nodeMoves[binding[i] ?? 0] ?? 0) !== 0 ? 1 : 0;
    return out;
  }
  const { nodes, weights } = binding;
  const influences = SKIN_INFLUENCES;
  for (let i = 0; i < count; i += 1) {
    const at = i * influences;
    let moves = 0;
    for (let k = 0; k < influences; k += 1) {
      if ((weights[at + k] ?? 0) !== 0 && (nodeMoves[nodes[at + k] ?? 0] ?? 0) !== 0) {
        moves = 1;
        break;
      }
    }
    out[i] = moves;
  }
  return out;
}
