/**
 * Seeded one-dimensional gradient noise. Deterministic by construction: no `Math.random`, no
 * `Date.now`, no `performance.now`. The same `(x, seed)` always yields bit-identical output,
 * which is what makes the whole motion model reproducible in CI.
 */

const UINT32 = 0x1_0000_0000;

/**
 * A 32-bit integer hash (a Wang/Murmur-style avalanche). Pure, branch-free and stable across
 * engines because every step is a 32-bit integer operation.
 */
export function hash32(value: number, seed: number): number {
  let h = (Math.imul(Math.trunc(seed) | 0, 0x9e3779b1) ^ (Math.trunc(value) | 0)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h;
}

/** Hashes a string to a 32-bit unsigned integer (FNV-1a). Used to seed per-node phases. */
export function hashString(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    h = Math.imul(h ^ text.charCodeAt(i), 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** Pseudo-random gradient in `[-1, 1)` for integer lattice point `i`. */
function gradient(i: number, seed: number): number {
  return (hash32(i, seed) / UINT32) * 2 - 1;
}

/** Quintic fade, `6t⁵ − 15t⁴ + 10t³`. Its first and second derivatives vanish at 0 and 1. */
function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/**
 * Perlin-style gradient noise in `[-1, 1]`, C² continuous everywhere. The bound is tight: the
 * extreme `0.5` occurs at a lattice midpoint with opposing unit gradients, so the `* 2` below
 * normalises without clipping — no clamp, and therefore no derivative discontinuity.
 */
export function noise1d(x: number, seed = 0): number {
  if (!Number.isFinite(x)) return 0;
  const i0 = Math.floor(x);
  const f = x - i0;
  const g0 = gradient(i0, seed);
  const g1 = gradient(i0 + 1, seed);
  const n0 = g0 * f;
  const n1 = g1 * (f - 1);
  const u = fade(f);
  return (n0 + u * (n1 - n0)) * 2;
}

/** Number of octaves summed by {@link fbm1d}. */
export const FBM_OCTAVES = 3;

const FBM_GAIN = 0.5;
const FBM_LACUNARITY = 2.17;
const FBM_NORM = (() => {
  let sum = 0;
  let amplitude = 1;
  for (let o = 0; o < FBM_OCTAVES; o += 1) {
    sum += amplitude;
    amplitude *= FBM_GAIN;
  }
  return sum;
})();

/**
 * Fractional Brownian motion over {@link noise1d}: `FBM_OCTAVES` octaves, amplitude-normalised
 * so the result stays in `[-1, 1]`. Lacunarity is irrational-ish (2.17) so the octaves do not
 * share a period and the gust never visibly repeats.
 */
export function fbm1d(x: number, seed = 0): number {
  if (!Number.isFinite(x)) return 0;
  let sum = 0;
  let amplitude = 1;
  let frequency = 1;
  for (let o = 0; o < FBM_OCTAVES; o += 1) {
    sum += amplitude * noise1d(x * frequency, seed + o * 1013);
    amplitude *= FBM_GAIN;
    frequency *= FBM_LACUNARITY;
  }
  return sum / FBM_NORM;
}

/**
 * Lipschitz bound on `d/dx noise1d`. Differentiating the lerp gives
 * `2·(g0 + u'·(n1 − n0) + u·(g1 − g0))`; with `|g| ≤ 1`, `max|u'| = 1.875`, `|n1 − n0| ≤ 1` and
 * `|g1 − g0| ≤ 2` that is at most `2 · 4.875 = 9.75`. Rounded up to 10.
 *
 * Conservative on purpose: it is used to bound the deformation velocity, and a bound that is
 * loose is merely unhelpful, whereas a bound that is wrong is a lie.
 */
export const NOISE1D_MAX_SLOPE = 10;

/** Lipschitz bound on `d/dx fbm1d`: each octave contributes its amplitude times its frequency. */
export const FBM_MAX_SLOPE = (() => {
  let sum = 0;
  let amplitude = 1;
  let frequency = 1;
  for (let o = 0; o < FBM_OCTAVES; o += 1) {
    sum += NOISE1D_MAX_SLOPE * amplitude * frequency;
    amplitude *= FBM_GAIN;
    frequency *= FBM_LACUNARITY;
  }
  return sum / FBM_NORM;
})();
