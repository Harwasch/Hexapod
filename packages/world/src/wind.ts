/**
 * The gust model: a deterministic, dimensionless wind field.
 *
 * Wind strength here is **not** a speed. See {@link WindStrength}.
 */

import { fbm1d, FBM_MAX_SLOPE } from "./noise";
import { clamp, DEG_TO_RAD, type Vec3 } from "./vec";

/**
 * A **dimensionless** wind strength in `0..1`, where `0` is dead calm and `1` is the strongest
 * gust this model produces.
 *
 * It is deliberately not metres per second. Calling it m/s would assert that the resulting sway
 * amplitude had been validated against a real tree at that speed; it has not, and no
 * biomechanical source backs the stiffness values. Anything shown to a user must be labelled a
 * simulated, arbitrary scale — never a measurement.
 */
export type WindStrength = number;

/** The wind parameters a scene holds: how hard it blows and which way. */
export interface WindSettings {
  /** Dimensionless `0..1`. See {@link WindStrength}. */
  readonly strength: WindStrength;
  /**
   * Compass bearing in degrees clockwise from north that the wind blows **towards** (downwind).
   * This is the opposite of the meteorological convention, which names the direction wind comes
   * from. Stated explicitly because an inverted bearing is invisible in a screenshot.
   */
  readonly bearingDeg: number;
}

/** Calm: the canonical, measured pose. The default everywhere. */
export const WIND_CALM: WindSettings = { strength: 0, bearingDeg: 0 };

/** Steady component of the gust, as a fraction of full strength. */
const WIND_BASE = 0.7;
/** Gusting component, modulated by noise in `[-1, 1]`, so along-wind is always in `[0.4, 1]`. */
const WIND_GUST = 0.3;
/** Cross-wind wander, zero-mean, as a fraction of full strength. */
const WIND_CROSS = 0.15;

/** Noise coordinates advanced per second. Slow, so gusts read as gusts and not as jitter. */
const GUST_RATE = 0.22;
const CROSS_RATE = 0.13;

const GUST_SEED = 0x5eed_9e1;
const CROSS_SEED = 0x5eed_c05;
/** Keeps the cross-wind lattice from aligning with the gust lattice. */
const CROSS_OFFSET = 37.19;

/**
 * The largest magnitude {@link wind} can return for a given strength. Exact, not empirical:
 * both noise terms are bounded by 1, so the extremes are `WIND_BASE + WIND_GUST` along-wind and
 * `WIND_CROSS` across it.
 */
export function maxWindMagnitude(strength: WindStrength): number {
  return clamp(strength, 0, 1) * Math.hypot(WIND_BASE + WIND_GUST, WIND_CROSS);
}

/**
 * Bound on `|d(wind)/dt|` at a given strength, used to prove the deformation velocity is
 * bounded. Both terms are a noise slope times the rate they are sampled at, times the strength
 * that scales them.
 */
export function maxWindSlope(strength: WindStrength): number {
  const s = Number.isFinite(strength) ? clamp(strength, 0, 1) : 0;
  return s * FBM_MAX_SLOPE * (WIND_GUST * GUST_RATE + WIND_CROSS * CROSS_RATE);
}

/** {@link maxWindSlope} at full strength. */
export const WIND_MAX_SLOPE = FBM_MAX_SLOPE * (WIND_GUST * GUST_RATE + WIND_CROSS * CROSS_RATE);

/** `-0` compares unequal to `0` under `Object.is`; rest-state assertions must be exact. */
function unsignZero(value: number): number {
  return value === 0 ? 0 : value;
}

/**
 * The gust vector at time `t`, in the rig's local ENU frame (metres-scale directions, but the
 * magnitude is dimensionless — see {@link WindStrength}). Vertical component is always zero:
 * this model has no updraft.
 *
 * Deterministic: the same `(strength, bearingDeg, t)` always returns bit-identical numbers.
 * `strength === 0` returns exactly `[0, 0, 0]`.
 */
export function wind(strength: WindStrength, bearingDeg: number, t: number): Vec3 {
  const s = Number.isFinite(strength) ? clamp(strength, 0, 1) : 0;
  const time = Number.isFinite(t) ? t : 0;
  const along = s * (WIND_BASE + WIND_GUST * fbm1d(time * GUST_RATE, GUST_SEED));
  const across = s * WIND_CROSS * fbm1d(time * CROSS_RATE + CROSS_OFFSET, CROSS_SEED);
  const bearing = (Number.isFinite(bearingDeg) ? bearingDeg : 0) * DEG_TO_RAD;
  // Downwind unit vector: bearing is clockwise from north, so east is sin and north is cos.
  const downEast = Math.sin(bearing);
  const downNorth = Math.cos(bearing);
  // Cross-wind unit vector, 90 degrees clockwise from downwind.
  const crossEast = downNorth;
  const crossNorth = -downEast;
  return [
    unsignZero(along * downEast + across * crossEast),
    unsignZero(along * downNorth + across * crossNorth),
    0,
  ];
}
