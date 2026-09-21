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

/**
 * Strength the wind control lands on when someone turns wind on — **not** the value the app
 * starts at, which is always 0.
 *
 * Chosen from the sort-staleness bound, not from a screenshot. The splat sorter reads canonical
 * positions the deformer never touches, so a displaced splat carries a draw-order key that is
 * stale by `maxDisplacement / medianGaussianScale` splat radii (`sortStaleness` in `metrics`).
 * Against the ~2 cm median gaussian of a real drone capture and the 6 m synthetic tree's rig:
 *
 * | strength | worst-case displacement | staleness |
 * | --- | --- | --- |
 * | 0.02 | 3.7 cm | 1.8 radii |
 * | 0.1 | 18.4 cm | 9.2 radii |
 * | 0.5 | 90.1 cm | 45.1 radii |
 * | 1 | 1.69 m | 84.6 radii |
 *
 * `SORT_STALENESS_NOTICEABLE` is 1, and it is a *hypothesis*. 0.1 is inside the largest
 * strength whose staleness bound stays within one decade of that hypothesis (0.108), which is
 * the honest width of our ignorance, rounded to a number a person can read.
 *
 * **The constant moved from 0.12 to 0.1 and the budget it buys did not.** The rig it is derived
 * from changed underneath it: 33 nodes became 214, every limb went from 5–15× too thick to
 * physically proportioned, and a tree with slender branches genuinely sways more per unit of
 * wind. Holding the derivation fixed and letting the constant move is what keeps the link
 * between this number and the artifact it is guarding against; holding the constant fixed would
 * have spent nearly twice the draw-order budget without anyone deciding to.
 *
 * Three things keep the bound pessimistic, and none is an argument for going higher. It is a
 * proven maximum over the outermost leaf's whole ancestor chain with the gust at its extreme and
 * every forcing sinusoid peaking at once — and that chain is now 13 joints deep rather than 4,
 * so the bound sits about **3.6×** the displacement actually seen over a minute (5.1 cm), where
 * the 33-node rig's sat at 2.5×. And the fixture's own gaussians are coarser than the yardstick
 * (5.0 cm median measured over `data/tiles/synthetic-tree`, down from 10.7 cm), where the same
 * strength is 3.7 radii rather than 9.2.
 */
export const DEFAULT_WIND_STRENGTH: WindStrength = 0.1;

/**
 * Steady component of the gust, as a fraction of full strength.
 *
 * Seventy per cent of this field is a constant, which is a great deal — and it used to be the
 * first reason the tree looked static, because the whole deformation was proportional to this
 * magnitude and nothing else. It is no longer, and the number can stay. `deform` now spends this
 * field on the **steady lean** alone, a minority share of each node's angular budget; what the
 * eye watches is the resonant response in `modes.ts`, which has its own frequencies. A field that
 * is mostly steady is the right description of mean wind. It was the wrong thing to hang all of
 * the motion on.
 */
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
