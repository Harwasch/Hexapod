/**
 * Splats back to front by distance from the eye: the draw order for alpha blending that does
 * not change when the view turns (cesium/splatSort.worker.ts runs it off the main thread).
 *
 * A counting sort on a 16-bit key of log distance: relative precision about 0.03% whether
 * the splats are a millimetre or a kilometre away, so near splats are ordered as finely as far
 * ones in proportion, in two linear passes.
 */

const BINS = 1 << 16;
let counts = new Uint32Array(BINS);
let keys = new Uint16Array(0);
let logs = new Float32Array(0);

export interface SortedSplats {
  /** Indexes, farthest from the eye first. */
  order: Uint32Array;
  /** Distance from the eye to the nearest splat sorted (Infinity when none). */
  nearest: number;
}

/**
 * Splats that move rigidly as groups (`cesium/splatRigid.ts`): `groups[i]` is splat `i`'s group
 * (0: none) and `eyes[3g..3g+2]` the eye carried back by group `g`'s motion. A rigid motion
 * keeps distances, so a moved splat's distance from the eye is its rest position's distance
 * from that eye: the order follows the motion without the moved positions.
 */
export interface SortGroups {
  readonly groups: Uint8Array | Uint16Array;
  readonly eyes: Float64Array;
}

/**
 * The back-to-front order of `count` splats (xyz in `positions`). A splat is left out when
 * `live` says 0 for it (a hidden or freed slot: splatSort.worker.ts) or when its position is
 * not finite (a slot never written), so the order can be shorter than `count`.
 */
export function sortBackToFront(
  positions: Float32Array,
  count: number,
  eye: readonly [number, number, number],
  live?: Uint8Array,
  moving?: SortGroups,
): SortedSplats {
  if (keys.length < count) keys = new Uint16Array(count);
  if (logs.length < count) logs = new Float32Array(count);
  if (counts.length !== BINS) counts = new Uint32Array(BINS);
  const [ex, ey, ez] = eye;
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  let kept = 0;
  for (let i = 0; i < count; i++) {
    if (live !== undefined && live[i] !== 1) {
      logs[i] = Number.NaN;
      continue;
    }
    let gx = ex;
    let gy = ey;
    let gz = ez;
    const group = moving?.groups[i] ?? 0;
    if (group > 0 && moving !== undefined && group * 3 + 2 < moving.eyes.length) {
      gx = moving.eyes[group * 3] ?? ex;
      gy = moving.eyes[group * 3 + 1] ?? ey;
      gz = moving.eyes[group * 3 + 2] ?? ez;
    }
    const dx = (positions[i * 3] ?? 0) - gx;
    const dy = (positions[i * 3 + 1] ?? 0) - gy;
    const dz = (positions[i * 3 + 2] ?? 0) - gz;
    // The log of the squared distance orders as the distance does, without a square root.
    logs[i] = Math.log(dx * dx + dy * dy + dz * dz + 1e-12);
    // The range from the stored (float32) values, so every key falls inside it.
    const value = logs[i] ?? Number.NaN;
    if (!Number.isFinite(value)) continue;
    kept++;
    if (value < lo) lo = value;
    if (value > hi) hi = value;
  }
  const scale = hi > lo ? (BINS - 1) / (hi - lo) : 0;
  counts.fill(0);
  for (let i = 0; i < count; i++) {
    const value = logs[i] ?? Number.NaN;
    if (!Number.isFinite(value)) continue;
    // The farthest gets key 0, so it is drawn first.
    // Clamped: at the far end (x - lo) * scale can round past BINS - 1.
    const bin = Math.max(0, Math.min(BINS - 1, Math.floor((value - lo) * scale)));
    const key = BINS - 1 - bin;
    keys[i] = key;
    counts[key] = (counts[key] ?? 0) + 1;
  }
  let running = 0;
  for (let k = 0; k < BINS; k++) {
    const c = counts[k] ?? 0;
    counts[k] = running;
    running += c;
  }
  const order = new Uint32Array(kept);
  for (let i = 0; i < count; i++) {
    if (!Number.isFinite(logs[i] ?? Number.NaN)) continue;
    const key = keys[i] ?? 0;
    const at = counts[key] ?? 0;
    order[at] = i;
    counts[key] = at + 1;
  }
  // lo is the log of the nearest squared distance.
  return { order, nearest: kept > 0 ? Math.sqrt(Math.exp(lo)) : Number.POSITIVE_INFINITY };
}

/** Indexes of `count` splats, farthest from `eye` first (see sortBackToFront). */
export function backToFront(
  positions: Float32Array,
  count: number,
  eye: readonly [number, number, number],
): Uint32Array {
  return sortBackToFront(positions, count, eye).order;
}
