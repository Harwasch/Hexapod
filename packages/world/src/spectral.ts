/**
 * Spectral motion textures: the stateless, aperiodic noise the Living Mode motion model samples.
 *
 * The method is Habel, Kusternig & Wimmer, "Physically Guided Animation of Trees", Eurographics
 * 2009, §5.3–5.4, with one correction:
 *
 * 1. Shape a random Gaussian field in the frequency domain by a target spectrum and inverse-FFT
 *    it into a 2D, doubly periodic **motion texture** (deterministic from a seed).
 * 2. Sample it along a straight line `p(t) = p0 + v·t` with wrap-around. A line on a torus whose
 *    direction ratio `v_x / v_y` is irrational never closes on itself, so the 1D signal is
 *    **aperiodic** although the texture is finite; Habel rejects directions "very close to a
 *    rational value with a small numerator and denominator", and so does
 *    {@link trajectoryDirection}, which also keeps the line from passing close to its own start
 *    within the hour the believability tests check.
 * 3. The speed `|v|` sets where the texture's spectrum lands in time: a texture whose resonance
 *    sits at `ρ0` cycles per texel, sampled at `|v|` texels per second, rings at `ρ0·|v|` Hz. So
 *    one texture serves every branch of a damping class, each at its own frequency (Habel:
 *    "we vary f_h individually by varying the length of the motion vector").
 *
 * **The correction.** Habel states that each trajectory "creates a 1D signal with a spectrum of
 * V_h(f)" when the 2D spectrum is the radially symmetric `V_h(√(x²+y²))`. It does not: the 1D
 * power spectrum of an isotropic 2D field sampled along a line is the **Abel projection** of
 * the 2D one, `S1(k) = ∫ S2(√(k² + q²)) dq`, which smears every ring downward in frequency — a
 * resonance at ζ = 0.2 lands about 20 % low. {@link designRadialSpectrum} instead solves for
 * the 2D radial spectrum whose projection *is* the target, by inverse Abel transform, clamped
 * at zero where the inverse goes negative. The believability tests measure the result.
 *
 * Nothing here reads a clock or a random source: every texture is a pure function of its
 * parameters and seed. Textures are memoised by those parameters, which is invisible to callers.
 */

import { hash32 } from "./noise";

const UINT32 = 0x1_0000_0000;

/** A 2D, doubly periodic, unit-RMS field. Read-only by contract once built. */
export interface MotionTexture {
  /** Side length in texels. A power of two. */
  readonly size: number;
  /** Row-major values, `size × size`. Unit RMS, zero mean. */
  readonly data: Float32Array;
  /** Largest `|value|` in `data`: an exact bound on anything bilinear sampling can return. */
  readonly maxAbs: number;
}

// ---------------------------------------------------------------------------------------------
// FFT
// ---------------------------------------------------------------------------------------------

/** In-place iterative radix-2 complex FFT. `inverse` uses `e^{+i}` and does not scale. */
export function fft(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  if (n !== im.length || (n & (n - 1)) !== 0) throw new Error("fft: length must be a power of 2");
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i] ?? 0;
      re[i] = re[j] ?? 0;
      re[j] = tr;
      const ti = im[i] ?? 0;
      im[i] = im[j] ?? 0;
      im[j] = ti;
    }
  }
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (sign * 2 * Math.PI) / len;
    const wr = Math.cos(angle);
    const wi = Math.sin(angle);
    const half = len >> 1;
    for (let start = 0; start < n; start += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < half; k += 1) {
        const a = start + k;
        const b = a + half;
        const br = re[b] ?? 0;
        const bi = im[b] ?? 0;
        const xr = br * cr - bi * ci;
        const xi = br * ci + bi * cr;
        const ar = re[a] ?? 0;
        const ai = im[a] ?? 0;
        re[a] = ar + xr;
        im[a] = ai + xi;
        re[b] = ar - xr;
        im[b] = ai - xi;
        const next = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = next;
      }
    }
  }
}

/** 2D inverse FFT of an `n × n` row-major complex array, in place. */
function ifft2(re: Float64Array, im: Float64Array, n: number): void {
  const rowRe = new Float64Array(n);
  const rowIm = new Float64Array(n);
  for (let y = 0; y < n; y += 1) {
    const base = y * n;
    for (let x = 0; x < n; x += 1) {
      rowRe[x] = re[base + x] ?? 0;
      rowIm[x] = im[base + x] ?? 0;
    }
    fft(rowRe, rowIm, true);
    for (let x = 0; x < n; x += 1) {
      re[base + x] = rowRe[x] ?? 0;
      im[base + x] = rowIm[x] ?? 0;
    }
  }
  for (let x = 0; x < n; x += 1) {
    for (let y = 0; y < n; y += 1) {
      rowRe[y] = re[y * n + x] ?? 0;
      rowIm[y] = im[y * n + x] ?? 0;
    }
    fft(rowRe, rowIm, true);
    for (let y = 0; y < n; y += 1) {
      re[y * n + x] = rowRe[y] ?? 0;
      im[y * n + x] = rowIm[y] ?? 0;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Deterministic randomness
// ---------------------------------------------------------------------------------------------

/** A uniform value in `(0, 1)` from a counter and a seed. Never exactly 0, so `log` is safe. */
export function unitUniform(counter: number, seed: number): number {
  return (hash32(counter, seed) + 0.5) / UINT32;
}

/** Two independent standard normals from two hashed uniforms (Box–Muller), into `out`. */
function gaussianPair(counter: number, seed: number, out: Float64Array): void {
  const radius = Math.sqrt(-2 * Math.log(unitUniform(counter * 2, seed)));
  const angle = 2 * Math.PI * unitUniform(counter * 2 + 1, seed);
  out[0] = radius * Math.cos(angle);
  out[1] = radius * Math.sin(angle);
}

// ---------------------------------------------------------------------------------------------
// Spectra
// ---------------------------------------------------------------------------------------------

/**
 * Amplitude response of a damped harmonic oscillator at frequency ratio `r = f / f_n`:
 * `|H| = 1 / √((1 − r²)² + (2ζr)²)`. Its square is the displacement power a white force puts
 * in, which is the stationary response Habel's eq. (15) describes.
 */
export function oscillatorGain(r: number, zeta: number): number {
  const real = 1 - r * r;
  const imag = 2 * zeta * r;
  return 1 / Math.sqrt(real * real + imag * imag);
}

/** `d|H|²/dr`, analytic. */
function oscillatorPowerSlope(r: number, zeta: number): number {
  const real = 1 - r * r;
  const d = real * real + 4 * zeta * zeta * r * r;
  return (4 * r * real - 8 * zeta * zeta * r) / (d * d);
}

/** Raised-cosine roll-off from 1 at `lo` to 0 at `hi`, and its derivative. */
function taper(k: number, lo: number, hi: number): { value: number; slope: number } {
  if (k <= lo) return { value: 1, slope: 0 };
  if (k >= hi) return { value: 0, slope: 0 };
  const phase = (Math.PI * (k - lo)) / (hi - lo);
  return {
    value: 0.5 * (1 + Math.cos(phase)),
    slope: (-0.5 * Math.PI * Math.sin(phase)) / (hi - lo),
  };
}

/** Frequencies above this multiple of the resonance are rolled off to zero... */
const ROLL_OFF_START = 2;
/** ...by this multiple. Keeps every component below the texture's Nyquist and away from it. */
const ROLL_OFF_END = 2.8;

/**
 * The 1D power spectrum a branch's motion should have, in texture frequency `k` (cycles per
 * texel), and its derivative. `|H(k/ρ0)|²` — the oscillator's response to a force that is flat
 * over the oscillator's own band — rolled off above `2·ρ0`.
 *
 * Flat forcing is the narrow-band reading of the wind spectrum: `motion.ts` multiplies each
 * branch by the Simiu–Scanlan spectrum's value *at that branch's own frequency*, which is exact
 * as ζ → 0 and is what lets one texture serve every branch of a damping class.
 */
function branchSpectrum(k: number, rho0: number, zeta: number): { value: number; slope: number } {
  const r = k / rho0;
  const gain = oscillatorGain(r, zeta);
  const roll = taper(k, ROLL_OFF_START * rho0, ROLL_OFF_END * rho0);
  const power = gain * gain;
  return {
    value: power * roll.value,
    slope: (oscillatorPowerSlope(r, zeta) / rho0) * roll.value + power * roll.slope,
  };
}

/**
 * The radial 2D spectrum whose projection along any line is `target`: the inverse Abel transform
 * `S2(ρ) = −(1/π) ∫_ρ^∞ S1'(k) / √(k² − ρ²) dk`, evaluated with `k = ρ·cosh u` so the integrand
 * has no singularity, and clamped at zero where it goes negative (a projection of a
 * non-negative field cannot rise faster than the inverse allows; what is lost is a sliver of
 * energy far below the resonance).
 *
 * Returns `samples + 1` values of `S2` on `ρ = i · kMax / samples`.
 */
export function designRadialSpectrum(
  slope: (k: number) => number,
  kMax: number,
  samples = 2048,
  steps = 1024,
): Float64Array {
  const out = new Float64Array(samples + 1);
  for (let i = 0; i <= samples; i += 1) {
    const rho = (i / samples) * kMax;
    if (rho >= kMax) break;
    const safeRho = Math.max(rho, kMax / (samples * 64));
    const uMax = Math.acosh(kMax / safeRho);
    const du = uMax / steps;
    let sum = 0;
    for (let s = 0; s <= steps; s += 1) {
      const weight = s === 0 || s === steps ? 0.5 : 1;
      sum += weight * slope(safeRho * Math.cosh(s * du));
    }
    out[i] = Math.max(0, (-sum * du) / Math.PI);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Texture synthesis
// ---------------------------------------------------------------------------------------------

/** Texture values are softly limited to this many standard deviations. */
const SOFT_CLIP = 3;

/**
 * Inverse-FFTs a random field shaped by a radial amplitude, normalises it to unit RMS.
 * `radialPower(ρ)` is the 2D power density at `ρ` cycles per texel.
 */
function synthesise(
  size: number,
  seed: number,
  radialPower: (rho: number) => number,
): MotionTexture {
  const n = size;
  const re = new Float64Array(n * n);
  const im = new Float64Array(n * n);
  const pair = new Float64Array(2);
  for (let y = 0; y < n; y += 1) {
    const ky = y < n / 2 ? y : y - n;
    for (let x = 0; x < n; x += 1) {
      const kx = x < n / 2 ? x : x - n;
      const rho = Math.hypot(kx, ky) / n;
      const index = y * n + x;
      if (rho === 0) continue;
      const amplitude = Math.sqrt(Math.max(0, radialPower(rho)));
      if (amplitude === 0) continue;
      gaussianPair(index, seed, pair);
      re[index] = amplitude * (pair[0] ?? 0);
      im[index] = amplitude * (pair[1] ?? 0);
    }
  }
  ifft2(re, im, n);
  const rms = (values: Float64Array): number => {
    let mean = 0;
    for (const value of values) mean += value;
    mean /= values.length;
    let sumSquares = 0;
    for (let i = 0; i < values.length; i += 1) {
      const v = (values[i] ?? 0) - mean;
      values[i] = v;
      sumSquares += v * v;
    }
    return Math.sqrt(sumSquares / values.length);
  };
  // Unit RMS, then a soft clip at ±SOFT_CLIP standard deviations: a Gaussian field of a million
  // texels reaches 5σ somewhere, and every displacement *bound* is proportional to the largest
  // texel. The clip costs well under 1 % of the variance and takes the bound down by ~40 %.
  const first = rms(re);
  for (let i = 0; i < n * n; i += 1) {
    const v = first > 0 ? (re[i] ?? 0) / first : 0;
    re[i] = SOFT_CLIP * Math.tanh(v / SOFT_CLIP);
  }
  const second = rms(re);
  const scale = second > 0 ? 1 / second : 0;
  const data = new Float32Array(n * n);
  let maxAbs = 0;
  for (let i = 0; i < n * n; i += 1) {
    const v = Math.fround((re[i] ?? 0) * scale);
    data[i] = v;
    const a = Math.abs(v);
    if (a > maxAbs) maxAbs = a;
  }
  return { size: n, data, maxAbs };
}

/** Side of a branch motion texture, texels. 4 MB as Float32. */
export const BRANCH_TEXTURE_SIZE = 1024;

/**
 * Texels per cycle at a branch texture's resonance.
 *
 * Sets two things against each other. Bilinear filtering wants many texels per cycle; the
 * aperiodicity budget wants few, because an hour of a `f`-hertz branch walks `3600·f·λ` texels
 * and the line must not pass close to where it started. With a 1024² texture and λ = 6 the
 * nearest return over an hour stays beyond ~⅓ of a wavelength for branches below about 9 Hz,
 * which is where the autocorrelation stays under 0.2 (see the believability tests).
 */
export const BRANCH_TEXELS_PER_CYCLE = 4;

const TEXTURE_CACHE = new Map<string, MotionTexture>();

/**
 * The shared motion texture for one damping ratio: displacement of a damped oscillator whose
 * resonance sits at `1 / BRANCH_TEXELS_PER_CYCLE` cycles per texel, driven by locally flat
 * forcing, with the 2D spectrum solved so that **every straight-line sample has that 1D
 * spectrum** (see the module comment).
 */
export function branchMotionTexture(
  zeta: number,
  seed: number,
  size = BRANCH_TEXTURE_SIZE,
): MotionTexture {
  const key = `branch:${size}:${zeta}:${seed}`;
  const cached = TEXTURE_CACHE.get(key);
  if (cached !== undefined) return cached;
  const rho0 = 1 / BRANCH_TEXELS_PER_CYCLE;
  const kMax = ROLL_OFF_END * rho0;
  const samples = 2048;
  const table = designRadialSpectrum((k) => branchSpectrum(k, rho0, zeta).slope, kMax, samples);
  const texture = synthesise(size, hash32(Math.round(zeta * 1e6), seed), (rho) => {
    const position = (rho / kMax) * samples;
    if (position >= samples) return 0;
    const i = Math.floor(position);
    const f = position - i;
    return (table[i] ?? 0) * (1 - f) + (table[i + 1] ?? 0) * f;
  });
  TEXTURE_CACHE.set(key, texture);
  return texture;
}

/** Side of the leaf-flutter texture, texels. One texel is one leaf size. */
export const FLUTTER_TEXTURE_SIZE = 1024;

/**
 * Shortest wavelength in the flutter field, in leaf sizes. Habel §7.2: "the minimum wavelength
 * represented in the leaf motion textures is 4 times the maximum leaf size", so the vertices of
 * one leaf never move inconsistently.
 */
export const FLUTTER_MIN_WAVELENGTH_LEAVES = 4;
/** Longest wavelength in the flutter field, in leaf sizes: the size of a coherent patch. */
export const FLUTTER_MAX_WAVELENGTH_LEAVES = 10;

/**
 * The leaf-flutter texture: a spatial field whose wavelengths lie between 4 and 10 leaf sizes,
 * with a Kolmogorov-like fall-off in between (`ρ^{-8/3}` in 2D). One texel is one leaf size, so
 * the same texture serves any tree. Unit RMS.
 */
export function flutterTexture(seed: number, size = FLUTTER_TEXTURE_SIZE): MotionTexture {
  const key = `flutter:${size}:${seed}`;
  const cached = TEXTURE_CACHE.get(key);
  if (cached !== undefined) return cached;
  const lo = 1 / FLUTTER_MAX_WAVELENGTH_LEAVES;
  const hi = 1 / FLUTTER_MIN_WAVELENGTH_LEAVES;
  const edge = 0.15;
  const texture = synthesise(size, hash32(0xf1a7, seed), (rho) => {
    if (rho <= lo * (1 - edge) || rho >= hi) return 0;
    const rise = 1 - taper(rho, lo * (1 - edge), lo * (1 + edge)).value;
    const fall = taper(rho, hi * (1 - edge), hi).value;
    return rise * fall * Math.pow(rho, -8 / 3);
  });
  TEXTURE_CACHE.set(key, texture);
  return texture;
}

/**
 * Bilinear, wrapping sample of a texture at texel coordinates `(x, y)`.
 *
 * Texel centres sit on integers. Coordinates may be any finite size; they are reduced modulo
 * the texture first, in float64, so a trajectory a year long keeps sub-texel precision.
 */
export function sampleTexture(texture: MotionTexture, x: number, y: number): number {
  const n = texture.size;
  const fx = x - Math.floor(x / n) * n;
  const fy = y - Math.floor(y / n) * n;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;
  const mask = n - 1;
  const xa = x0 & mask;
  const xb = (x0 + 1) & mask;
  const ya = (y0 & mask) * n;
  const yb = ((y0 + 1) & mask) * n;
  const d = texture.data;
  const a = d[ya + xa] ?? 0;
  const b = d[ya + xb] ?? 0;
  const c = d[yb + xa] ?? 0;
  const e = d[yb + xb] ?? 0;
  return (a + (b - a) * tx) * (1 - ty) + (c + (e - c) * tx) * ty;
}

// ---------------------------------------------------------------------------------------------
// Trajectories
// ---------------------------------------------------------------------------------------------

/** Denominators up to this are checked for a near-rational direction ratio. */
const RATIONAL_TEST_DENOMINATOR = 64;
/** A ratio `r` passes when `q·|q·r − p| ≥ this` for every `q` up to the bound. */
const RATIONAL_TEST_MARGIN = 0.05;

/**
 * Whether `ratio` is far from every rational `p/q` with `q ≤ 64`, in the Diophantine sense
 * `q·|q·r − p| ≥ 0.05`. The golden ratio scores 0.447, the best any number can.
 */
export function isBadlyApproximable(ratio: number): boolean {
  if (!Number.isFinite(ratio)) return false;
  const r = Math.abs(ratio);
  for (let q = 1; q <= RATIONAL_TEST_DENOMINATOR; q += 1) {
    const qr = q * r;
    if (q * Math.abs(qr - Math.round(qr)) < RATIONAL_TEST_MARGIN) return false;
  }
  return true;
}

/**
 * How close a wrapped straight line comes to revisiting its own start, texels: the smallest
 * distance from `direction·s` to any lattice point `(a·size, b·size) ≠ 0`, over path lengths `s`
 * in `[sMin, sMax]`.
 *
 * This is the quantity aperiodicity actually depends on. A trajectory that passes within a
 * fraction of a wavelength of where it started replays what it did then, and the signal's
 * autocorrelation at that lag is the texture's own correlation at that offset. "Irrational
 * slope" guarantees the distance is never exactly zero; it does not keep it large over an
 * hour, which is what the believability test asks for.
 */
export function trajectoryClearance(
  direction: readonly [number, number],
  size: number,
  sMin: number,
  sMax: number,
): number {
  const [dx, dy] = direction;
  let best = Number.POSITIVE_INFINITY;
  // Walk the lattice columns the line crosses (or rows, for a steep line) and test the two
  // lattice points either side of it in each.
  const steep = Math.abs(dy) > Math.abs(dx);
  const major = steep ? dy : dx;
  const minor = steep ? dx : dy;
  const count = Math.ceil((Math.abs(major) * sMax) / size) + 1;
  for (let a = 0; a <= count; a += 1) {
    const along = a * size * Math.sign(major || 1);
    const s = along / major;
    const across = s * minor;
    const b0 = Math.floor(across / size);
    for (let b = b0 - 1; b <= b0 + 2; b += 1) {
      const px = steep ? b * size : along;
      const py = steep ? along : b * size;
      if (px === 0 && py === 0) continue;
      const projection = px * dx + py * dy;
      if (projection < sMin || projection > sMax) continue;
      const distance = Math.abs(px * dy - py * dx);
      if (distance < best) best = distance;
    }
  }
  return best;
}

/** How many candidate directions {@link trajectoryDirection} considers. */
const DIRECTION_CANDIDATES = 96;

/**
 * A unit direction for a trajectory that must not revisit itself for `horizonTexels` of travel,
 * drawn deterministically from `(key, seed)`.
 *
 * Habel §5.4 draws a random direction and rejects it when its component ratio is close to a
 * rational with a small denominator, so the line never closes. This keeps that test and adds
 * the one the hour-long autocorrelation needs: among a fixed sequence of hashed candidates it
 * takes the first whose {@link trajectoryClearance} over the horizon is at least
 * `wantedClearance`, or failing that the candidate with the largest. Deterministic: the same
 * arguments always give the same direction.
 */
export function trajectoryDirection(
  key: number,
  seed: number,
  size: number,
  horizonTexels: number,
  wantedClearance: number,
): readonly [number, number] {
  let best: readonly [number, number] | undefined;
  let bestClearance = -1;
  for (let attempt = 0; attempt < DIRECTION_CANDIDATES; attempt += 1) {
    const angle = 2 * Math.PI * unitUniform(key * 131 + attempt, seed);
    const x = Math.cos(angle);
    const y = Math.sin(angle);
    if (Math.abs(y) < 1e-3 || Math.abs(x) < 1e-3) continue;
    // Both ratios: a steep line returns near its start after a single wrap, which only the
    // reciprocal's `p = 0` case catches.
    if (!isBadlyApproximable(x / y) || !isBadlyApproximable(y / x)) continue;
    const clearance = trajectoryClearance([x, y], size, 0, horizonTexels);
    if (clearance >= wantedClearance) return [x, y];
    if (clearance > bestClearance) {
      bestClearance = clearance;
      best = [x, y];
    }
  }
  if (best !== undefined) return best;
  const golden = (1 + Math.sqrt(5)) / 2;
  const length = Math.hypot(1, golden);
  return [1 / length, golden / length];
}
