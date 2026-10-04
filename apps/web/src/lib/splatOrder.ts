/**
 * Splats back to front by distance from the eye: the draw order for alpha blending that does
 * not change when the view turns (cesium/splatSort.worker.ts runs it off the main thread).
 *
 * A counting sort on a 17-bit key of each splat's squared distance, in two linear passes. The
 * key used to be the log of the squared distance, mapped over each sort's own range: relative
 * precision the same whether the splats are a millimetre or a kilometre away -- but a
 * `Math.log` per splat was most of a sort (3M splats: about 100 ms). The bit pattern of a
 * non-negative float32 is already almost its log2: the exponent counts octaves and the
 * mantissa runs linearly across each, and the pattern orders as the value does. So the key is
 * the float32 bits of d², re-based to the sort's own lowest and highest pattern and scaled
 * onto the bins, with no log at all.
 *
 * Within an octave the mantissa is linear where the log is not, so a bin is up to 1.44 times
 * as wide (in log terms) at the bottom of an octave as the log key's was, and half that at the
 * top. 2^17 bins instead of 2^16 more than pays for it: the worst bin is now 0.72 of the log
 * key's (worst distance error on a 150 m scan seen from inside: 4.9e-5 against 7.0e-5,
 * `splatOrder.test.ts`). Taking the top 16 bits of the pattern instead, without re-basing,
 * would have spent most of the key on exponents no scan uses: about 0.4%, 4 cm at 10 m, which
 * is a visible shimmer between overlapping splats.
 */

const BINS = 1 << 17;
const TOP = BINS - 1;
/** The pattern a left-out splat gets: above every finite non-negative float32's. */
const SKIP = 0xffffffff;
let counts = new Uint32Array(BINS);
let squared = new Float32Array(0);
/** `squared`'s bits: the keys. */
let patterns = new Uint32Array(squared.buffer);
const scratch = new Float32Array(1);
const scratchBits = new Uint32Array(scratch.buffer);

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
  if (squared.length < count) {
    squared = new Float32Array(count);
    patterns = new Uint32Array(squared.buffer);
  }
  if (counts.length !== BINS) counts = new Uint32Array(BINS);
  const [ex, ey, ez] = eye;
  let lo = SKIP;
  let hi = 0;
  let kept = 0;
  const groups = moving?.groups;
  const eyes = moving?.eyes;
  for (let i = 0; i < count; i++) {
    if (live !== undefined && live[i] !== 1) {
      patterns[i] = SKIP;
      continue;
    }
    let dx = (positions[i * 3] ?? 0) - ex;
    let dy = (positions[i * 3 + 1] ?? 0) - ey;
    let dz = (positions[i * 3 + 2] ?? 0) - ez;
    if (groups !== undefined && eyes !== undefined) {
      const group = groups[i] ?? 0;
      if (group > 0 && group * 3 + 2 < eyes.length) {
        dx = (positions[i * 3] ?? 0) - (eyes[group * 3] ?? ex);
        dy = (positions[i * 3 + 1] ?? 0) - (eyes[group * 3 + 1] ?? ey);
        dz = (positions[i * 3 + 2] ?? 0) - (eyes[group * 3 + 2] ?? ez);
      }
    }
    // The squared distance orders as the distance does, without a square root.
    const d2 = dx * dx + dy * dy + dz * dz;
    // NaN (a slot never written) and Infinity are left out.
    if (!(d2 < Number.POSITIVE_INFINITY)) {
      patterns[i] = SKIP;
      continue;
    }
    squared[i] = d2;
    // The range from the stored (float32) values' patterns, so every key falls inside it.
    const pattern = patterns[i] ?? SKIP;
    kept++;
    if (pattern < lo) lo = pattern;
    if (pattern > hi) hi = pattern;
  }
  const scale = hi > lo ? TOP / (hi - lo) : 0;
  counts.fill(0);
  // The farthest gets key 0, so it is drawn first. Clamped: at the far end (x - lo) * scale
  // can round past TOP. The key is worked out again in the last pass rather than kept: a
  // subtraction and a multiply cost less than writing and reading back a key per splat.
  for (let i = 0; i < count; i++) {
    const pattern = patterns[i] ?? SKIP;
    if (pattern === SKIP) continue;
    const key = TOP - Math.min(TOP, Math.floor((pattern - lo) * scale));
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
    const pattern = patterns[i] ?? SKIP;
    if (pattern === SKIP) continue;
    const key = TOP - Math.min(TOP, Math.floor((pattern - lo) * scale));
    const at = counts[key] ?? 0;
    order[at] = i;
    counts[key] = at + 1;
  }
  // lo is the pattern of the nearest squared distance.
  scratchBits[0] = lo;
  return { order, nearest: kept > 0 ? Math.sqrt(scratch[0] ?? 0) : Number.POSITIVE_INFINITY };
}

/** Indexes of `count` splats, farthest from `eye` first (see sortBackToFront). */
export function backToFront(
  positions: Float32Array,
  count: number,
  eye: readonly [number, number, number],
): Uint32Array {
  return sortBackToFront(positions, count, eye).order;
}
