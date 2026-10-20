/**
 * The arithmetic of the sunlight relight (cesium/splatSunlight.ts), restated in TypeScript so
 * that it can be tested headlessly and so that the scene spec can say what the shader should
 * draw. Pure: no Cesium, no WebGL. The GLSL is a transcription of these functions, line for
 * line; a change to one is a change to the other.
 *
 * A Gaussian splat has no normal. A flat one -- a disc, the shape a splat fitted to a surface
 * takes -- has one in all but name: the axis of its smallest variance. So a splat is relit
 * only when it is clearly flat, its smallest eigenvalue under `FLAT_RATIO` of the middle one;
 * anything rounder (a blob of foliage, a needle with no preferred side) counts as facing the
 * sun: drawn as photographed by day, at the ambient floor at night.
 *
 * The eigenvalues are the closed-form solution of a symmetric 3x3 (the trigonometric method of
 * Smith, "Eigenvalues of a symmetric 3 x 3 matrix", Comm. ACM 4(4), 1961); the eigenvector of
 * the smallest is the largest cross product of two rows of `A - λ3·I`, which has rank two
 * whenever λ3 is simple -- and under the flatness test it is, since λ3 < λ2 / 4.
 *
 * What it cannot know: the capture's own shading. Shadows the photographs had are in the
 * colours; the relight multiplies them, it never removes them (the badge says so).
 */

/** A symmetric 3x3 as its six independent entries: xx, xy, xz, yy, yz, zz. */
export type Sym3 = readonly [number, number, number, number, number, number];
export type Vec3 = readonly [number, number, number];

/** Flat enough to have a normal: smallest variance under this share of the middle one. */
export const FLAT_RATIO = 0.25;
/** What a surface turned fully away from the sun keeps of its photographed colour. */
export const SUN_AMBIENT = 0.35;
/**
 * The sun's elevation over which the relight fades in (degrees): from the horizon, where the
 * direct term is 0, to this, where it is whole -- so sunset dims a scan instead of switching
 * it. A rendering choice, not a measurement.
 */
export const SUN_UP_RAMP_DEG = 5;
/** Below this spread of the eigenvalues (relative to the largest variance) a splat is round. */
export const ROUND_SPREAD = 1e-5;

const TWO_THIRDS_PI = (2 * Math.PI) / 3;

/** λ1 ≥ λ2 ≥ λ3 of a symmetric 3x3, or null for a round (isotropic) or empty one. */
export function symmetricEigenvalues(a: Sym3): Vec3 | null {
  const [xx, xy, xz, yy, yz, zz] = a;
  // Scaled by the largest diagonal entry, which bounds every entry of a PSD matrix: the
  // thresholds below are then relative, whatever the splat's size.
  const scale = Math.max(xx, yy, zz);
  if (!(scale > 0)) return null;
  const b00 = xx / scale;
  const b01 = xy / scale;
  const b02 = xz / scale;
  const b11 = yy / scale;
  const b12 = yz / scale;
  const b22 = zz / scale;
  const q = (b00 + b11 + b22) / 3;
  const p1 = b01 * b01 + b02 * b02 + b12 * b12;
  const d0 = b00 - q;
  const d1 = b11 - q;
  const d2 = b22 - q;
  const p = Math.sqrt((d0 * d0 + d1 * d1 + d2 * d2 + 2 * p1) / 6);
  if (p < ROUND_SPREAD) return null;
  // det((A - qI) / p) / 2: the cosine of three times the angle.
  const c00 = d0 / p;
  const c11 = d1 / p;
  const c22 = d2 / p;
  const c01 = b01 / p;
  const c02 = b02 / p;
  const c12 = b12 / p;
  const det =
    c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02) + c02 * (c01 * c12 - c11 * c02);
  const r = Math.min(1, Math.max(-1, det / 2));
  const phi = Math.acos(r) / 3;
  const l1 = q + 2 * p * Math.cos(phi);
  const l3 = q + 2 * p * Math.cos(phi + TWO_THIRDS_PI);
  const l2 = 3 * q - l1 - l3;
  return [l1 * scale, l2 * scale, l3 * scale];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * The unit normal of a flat splat (its smallest axis, either sign), or null when it is not
 * flat enough to have one: round, a needle, empty, or not a covariance at all.
 */
export function splatNormal(a: Sym3): Vec3 | null {
  const values = symmetricEigenvalues(a);
  if (!values) return null;
  const [, l2, l3] = values;
  if (!(l3 < FLAT_RATIO * l2)) return null;
  const [xx, xy, xz, yy, yz, zz] = a;
  const r0: Vec3 = [xx - l3, xy, xz];
  const r1: Vec3 = [xy, yy - l3, yz];
  const r2: Vec3 = [xz, yz, zz - l3];
  let best = cross(r0, r1);
  let length2 = dot(best, best);
  for (const c of [cross(r0, r2), cross(r1, r2)]) {
    const l = dot(c, c);
    if (l > length2) {
      best = c;
      length2 = l;
    }
  }
  if (!(length2 > 0)) return null;
  const inv = 1 / Math.sqrt(length2);
  return [best[0] * inv, best[1] * inv, best[2] * inv];
}

/** `Σ = R · diag(s²) · Rᵀ` for a splat of axis scales `s` turned by the quaternion `xyzw`. */
export function covarianceOf(scale: Vec3, quat: readonly [number, number, number, number]): Sym3 {
  const n = Math.hypot(quat[0], quat[1], quat[2], quat[3]) || 1;
  const [x, y, z, w] = [quat[0] / n, quat[1] / n, quat[2] / n, quat[3] / n];
  const r = [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
  const s2 = [scale[0] ** 2, scale[1] ** 2, scale[2] ** 2];
  const at = (i: number, j: number): number => {
    let sum = 0;
    for (let k = 0; k < 3; k += 1) sum += (r[i]?.[k] ?? 0) * (r[j]?.[k] ?? 0) * (s2[k] ?? 0);
    return sum;
  };
  return [at(0, 0), at(0, 1), at(0, 2), at(1, 1), at(1, 2), at(2, 2)];
}

/**
 * The nearest IEEE half to `x`, as the engine stores a covariance entry (`packHalf2x16`), so
 * a test can ask what the shader reads rather than what was meant.
 */
export function toHalf(x: number): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const sign = Math.sign(x);
  const a = Math.abs(x);
  if (a >= 65520) return sign * Infinity;
  // Subnormals, below 2^-14, are spaced 2^-24 apart; normals have 10 fraction bits.
  const exponent = Math.max(Math.floor(Math.log2(a)), -14);
  const step = 2 ** (exponent - 10);
  // Round to nearest, ties to even, as the conversion does.
  const units = a / step;
  let rounded = Math.round(units);
  if (rounded - units === 0.5 && rounded % 2 !== 0) rounded -= 1;
  return sign * rounded * step;
}

/** A covariance as the shader reads it: every entry rounded to a half. */
export function halfCovariance(a: Sym3): Sym3 {
  return [toHalf(a[0]), toHalf(a[1]), toHalf(a[2]), toHalf(a[3]), toHalf(a[4]), toHalf(a[5])];
}

/**
 * How much of the direct term the relight gives at a sun elevation whose sine is `sine`: 0 at
 * and below the horizon, 1 from `SUN_UP_RAMP_DEG` up.
 */
export function sunUpFactor(sine: number): number {
  const top = Math.sin((SUN_UP_RAMP_DEG * Math.PI) / 180);
  return Math.min(1, Math.max(0, sine / top));
}

/** What the shader multiplies one splat's colour by, from: (eye coordinates throughout) */
export interface SunlightInput {
  /** The splat's covariance in the draw command's model frame. */
  covariance: Sym3;
  /** The model frame's rotation into eye coordinates, column-major (`mat3(czm_modelView)`). */
  modelView3: readonly number[];
  /** The splat's centre in eye coordinates (the eye at the origin). */
  positionEC: Vec3;
  /** Unit direction to the sun, eye coordinates (`czm_sunDirectionEC`). */
  sunEC: Vec3;
  /** `sunUpFactor` of the sun's elevation at the site. */
  sunUp: number;
  /** 0 draws the photograph, 1 the whole relight. */
  strength: number;
}

/**
 * The colour multiplier `splatSunlightColor` applies:
 * `mix(1, ambient + (1 - ambient) · lambert · sunUp, strength)`, where a splat with no normal
 * counts as facing the sun (lambert 1).
 */
export function sunlightFactor(input: SunlightInput): number {
  const n = splatNormal(input.covariance);
  let lambert = 1;
  if (n) {
    const m = input.modelView3;
    const ec: Vec3 = [
      (m[0] ?? 0) * n[0] + (m[3] ?? 0) * n[1] + (m[6] ?? 0) * n[2],
      (m[1] ?? 0) * n[0] + (m[4] ?? 0) * n[1] + (m[7] ?? 0) * n[2],
      (m[2] ?? 0) * n[0] + (m[5] ?? 0) * n[1] + (m[8] ?? 0) * n[2],
    ];
    const length = Math.hypot(ec[0], ec[1], ec[2]) || 1;
    let nEC: Vec3 = [ec[0] / length, ec[1] / length, ec[2] / length];
    // Toward the eye: a splat is seen from the side it is drawn on.
    if (dot(nEC, input.positionEC) > 0) nEC = [-nEC[0], -nEC[1], -nEC[2]];
    lambert = Math.max(dot(nEC, input.sunEC), 0);
  }
  const light = SUN_AMBIENT + (1 - SUN_AMBIENT) * lambert * input.sunUp;
  return 1 + (light - 1) * input.strength;
}
