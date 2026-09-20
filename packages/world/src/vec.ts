/** Small fixed-size vector and quaternion maths. Tuples, so every index is statically known. */

/** A point or direction in the rig's local ENU frame: `[east, north, up]` metres. */
export type Vec3 = readonly [number, number, number];

/** A unit quaternion, `[x, y, z, w]`. `QUAT_IDENTITY` is the exact rest value. */
export type Quat = readonly [number, number, number, number];

export const VEC3_ZERO: Vec3 = [0, 0, 0];
export const VEC3_UP: Vec3 = [0, 0, 1];
export const QUAT_IDENTITY: Quat = [0, 0, 0, 1];

export const DEG_TO_RAD = Math.PI / 180;

export function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function subtract(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale(a: Vec3, k: number): Vec3 {
  return [a[0] * k, a[1] * k, a[2] * k];
}

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function magnitude(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

export function distance(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/** Returns the unit vector, or `fallback` when `a` is degenerate or non-finite. */
export function normalize(a: Vec3, fallback: Vec3 = VEC3_UP): Vec3 {
  const len = magnitude(a);
  if (!(len > 0) || !Number.isFinite(len)) return fallback;
  return [a[0] / len, a[1] / len, a[2] / len];
}

/**
 * A rotation of `angleRad` about `axis` (assumed unit length).
 *
 * `angleRad === 0` returns bit-exact `QUAT_IDENTITY`: `Math.sin(0)` is `+0` and
 * `Math.cos(0)` is `1` in IEEE-754, so the rest pose is recoverable exactly rather
 * than approximately. The zero-wind identity guarantee rests on this.
 */
export function quatFromAxisAngle(axis: Vec3, angleRad: number): Quat {
  // Returned by value, not computed: `axis[k] * Math.sin(0)` is `-0` for a negative axis
  // component, and `-0` is not `0` under `Object.is`, so the rest pose would compare unequal.
  if (angleRad === 0) return QUAT_IDENTITY;
  const half = angleRad / 2;
  const s = Math.sin(half);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(half)];
}

/** Hamilton product `a ⊗ b`: apply `b` first, then `a`. Identity inputs give exact identity. */
export function quatMultiply(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

/**
 * Rotates `v` by `q`. With `q === QUAT_IDENTITY` the vector part is exactly zero, so both
 * cross products are exactly zero and the result is bit-identical to `v`.
 */
export function quatRotate(q: Quat, v: Vec3): Vec3 {
  const u: Vec3 = [q[0], q[1], q[2]];
  const w = q[3];
  const uv = cross(u, v);
  const t: Vec3 = [uv[0] + w * v[0], uv[1] + w * v[1], uv[2] + w * v[2]];
  const r = cross(u, t);
  return [v[0] + 2 * r[0], v[1] + 2 * r[1], v[2] + 2 * r[2]];
}

/** The inverse of a unit quaternion. */
export function quatConjugate(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
}

/** The rotation angle of a unit quaternion, in radians, in `[0, π]`. */
export function quatAngle(q: Quat): number {
  const w = Math.min(1, Math.max(-1, Math.abs(q[3])));
  return 2 * Math.acos(w);
}

export function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

/**
 * Smooth saturation: maps `value` into `(-limit, limit)` monotonically and never reaches the
 * bound. Unlike a hard clamp it is differentiable everywhere, which keeps the deformation
 * velocity continuous where a gust would otherwise push a branch onto its stop.
 * `softLimit(0, limit)` is exactly `0`.
 */
export function softLimit(value: number, limit: number): number {
  if (!(limit > 0)) return 0;
  return limit * Math.tanh(value / limit);
}
