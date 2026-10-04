/**
 * Steering a camera flight that is already under way, without a jolt.
 *
 * A fly-to now leaves on the click, towards the best pose known at that moment (the catalog
 * summary's centre), while the site's details and model load (SiteManager.flyTo). When the
 * details arrive with the authored bookmark, or the model settles on the ground, the flight
 * is pointed at the better pose. Cesium has no "change destination": a new `camera.flyTo`
 * replaces the old flight and starts from where the camera is -- and with the usual
 * quadratic in-out easing it starts from *rest*. Mid-flight that is a visible brake and
 * re-launch, the one thing a maps app never does.
 *
 * So the replacement flight is given an easing whose starting slope matches the speed the
 * camera already has: a cubic with s(0) = 0, s'(0) = k, s(1) = 1, s'(1) = 0, where k is the
 * current speed in the new flight's units (its length over its duration). It still arrives
 * at rest, and the camera's speed is continuous through the hand-over.
 *
 * Lengths are chords between camera positions, not the arcs Cesium actually flies; both the
 * old and the new flight are measured the same way and their paths are near-identical in
 * shape, so the error largely cancels, and an estimate is all a speed match needs. Pure
 * numbers, no Cesium, so it is tested directly.
 */

export type Easing = (t: number) => number;

/** Cesium's `EasingFunction.QUADRATIC_IN_OUT` (tween.js Quadratic.InOut): every flight's curve. */
export const quadraticInOut: Easing = (t) => (t < 0.5 ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t));

/** The largest starting slope for which the continuing cubic still never overshoots. */
const MAX_START_SLOPE = 3;

/** Slope of an easing at `t`: the share of the flight covered per unit of its time. */
export function easingSlope(easing: Easing, t: number): number {
  const h = 1e-4;
  const a = Math.max(0, t - h);
  const b = Math.min(1, t + h);
  return b > a ? (easing(b) - easing(a)) / (b - a) : 0;
}

/**
 * The easing that leaves at slope `k` and arrives at rest: the cubic Hermite through (0, 0)
 * with slope k and (1, 1) with slope 0. Monotonic for k in [0, 3], which is where k is held.
 */
export function continuingEasing(k: number): Easing {
  const slope = Math.min(MAX_START_SLOPE, Math.max(0, Number.isFinite(k) ? k : 0));
  return (t) => {
    const u = Math.min(1, Math.max(0, t));
    return slope * u + (3 - 2 * slope) * u * u + (slope - 2) * u * u * u;
  };
}

/** A flight as SiteManager started it: enough to know how fast the camera is going now. */
export interface FlightProgress {
  /** Seconds since the flight left. */
  elapsedS: number;
  durationS: number;
  easing: Easing;
  /** Chord from where it left to where it was going (m). */
  lengthM: number;
}

export interface Retarget {
  durationS: number;
  easing: Easing;
}

/**
 * The duration and easing of a flight from the camera's current position to a new
 * destination `lengthM` away, continuing at the current speed. Null when the flight is
 * already over (the caller decides whether a fresh flight is still wanted).
 *
 * The replacement keeps the old flight's remaining time, so the arrival is not postponed by
 * the details turning up, but never takes less than `minimumS`: a destination that moved
 * while the camera was nearly there still gets a readable approach.
 */
export function retarget(
  flight: FlightProgress,
  lengthM: number,
  minimumS: number,
): Retarget | null {
  if (!(flight.durationS > 0)) return null;
  const tau = flight.elapsedS / flight.durationS;
  if (!(tau < 1)) return null;
  const progress = Math.max(0, tau);
  const speed = (flight.lengthM * easingSlope(flight.easing, progress)) / flight.durationS;
  const durationS = Math.max((1 - progress) * flight.durationS, minimumS);
  const k = lengthM > 0 ? (speed * durationS) / lengthM : 0;
  return { durationS, easing: continuingEasing(k) };
}

/** A camera pose to arrive at, in degrees and metres. */
export interface ArrivalPose {
  longitude: number;
  latitude: number;
  height: number;
  heading: number;
  pitch: number;
}

/**
 * Whether two arrival poses are close enough that steering from one to the other is not
 * worth doing: within `toleranceM` of each other (the caller scales it with the distance
 * still to fly) and within 2° of heading and pitch.
 */
export function samePose(
  a: ArrivalPose,
  b: ArrivalPose,
  separationM: number,
  toleranceM: number,
): boolean {
  const turn = Math.abs(((a.heading - b.heading + 540) % 360) - 180);
  return separationM <= toleranceM && turn <= 2 && Math.abs(a.pitch - b.pitch) <= 2;
}
