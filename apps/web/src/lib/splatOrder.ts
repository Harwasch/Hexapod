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

/** Indexes of `count` splats (xyz in `positions`), farthest from `eye` first. */
export function backToFront(
  positions: Float32Array,
  count: number,
  eye: readonly [number, number, number],
): Uint32Array {
  if (keys.length < count) keys = new Uint16Array(count);
  if (logs.length < count) logs = new Float32Array(count);
  if (counts.length !== BINS) counts = new Uint32Array(BINS);
  const [ex, ey, ez] = eye;
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < count; i++) {
    const dx = (positions[i * 3] ?? 0) - ex;
    const dy = (positions[i * 3 + 1] ?? 0) - ey;
    const dz = (positions[i * 3 + 2] ?? 0) - ez;
    // The log of the squared distance orders as the distance does, without a square root.
    logs[i] = Math.log(dx * dx + dy * dy + dz * dz + 1e-12);
    // The range from the stored (float32) values, so every key falls inside it.
    const value = logs[i] ?? 0;
    if (value < lo) lo = value;
    if (value > hi) hi = value;
  }
  const scale = hi > lo ? (BINS - 1) / (hi - lo) : 0;
  counts.fill(0);
  for (let i = 0; i < count; i++) {
    // The farthest gets key 0, so it is drawn first.
    // Clamped: at the far end (x - lo) * scale can round past BINS - 1.
    const bin = Math.max(0, Math.min(BINS - 1, Math.floor(((logs[i] ?? lo) - lo) * scale)));
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
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const key = keys[i] ?? 0;
    const at = counts[key] ?? 0;
    order[at] = i;
    counts[key] = at + 1;
  }
  return order;
}
