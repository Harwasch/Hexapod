/**
 * One continuous camera flight whose destination may move while it flies.
 *
 * A fly-to leaves on the click, for the best pose known at that moment, and better poses
 * arrive on the way: the site's record, the terrain under it, the model's own bounds once it
 * rests on the ground (SiteManager.flyTo). Cesium has no "change destination", so each of
 * those used to be a new `camera.flyTo` leg from wherever the camera was: a new height curve, a
 * new pitch curve and, for a correction that came too late to re-point for, a second flight
 * from rest after landing. Seen from the seat that is a lurch on every correction, and a
 * landing followed by a take-off. Here the flight is one curve from start to end, and the end
 * itself moves smoothly:
 *
 * - **The path** is van Wijk and Nuij's smooth zoom-and-pan ("Smooth and efficient zooming and
 *   panning", 2003; the curve behind Mapbox's and d3's fly-to): the camera's ground track runs
 *   along the geodesic while its altitude rises as far as the distance asks and comes back
 *   down, at a constant perceived speed. Its pitch looks straight down while it is high above
 *   both ends and tilts to the arrival pitch on the way down, so the long middle of a flight
 *   shows the ground below rather than a horizon full of tiles still loading.
 * - **The clock** eases that path in and out (a half cosine), so it leaves and arrives at rest.
 * - **The destination** follows the latest pose through a critically damped spring: a
 *   correction moves where the flight is going over a fraction of a second, with no jump in
 *   the camera's position or speed however late it comes, and one that comes after the clock
 *   has run out glides the camera on from where it is.
 * - **The ground**: the path runs above the ground interpolated between its ends, and an end
 *   aimed at a guessed ground is re-aimed at the real one as the terrain under it loads. The
 *   ground the globe has loaded under the path itself is not a floor: on the way down it is a
 *   coarse tile, hundreds of metres off in the mountains, and the zoom-out arc clears any
 *   ridge between two ends by a margin of the distance between them.
 *
 * Pure numbers apart from the geodesic, so the path, the spring and the driver are tested
 * directly, frame by frame (glide.test.ts).
 */
import { Cartographic, EllipsoidGeodesic, Math as CesiumMath } from "cesium";

import type { ArrivalPose } from "./flightRetarget";
import { plausibleGround as plausible } from "./placement";

export { plausible };

/** Van Wijk and Nuij's ρ, Mapbox's default: how far a flight zooms out to cross a distance. */
const RHO = 1.42;
/** Ground width seen per metre of altitude at Cesium's 60° field of view: 2 tan 30°. */
const WIDTH_PER_ALTITUDE = 2 * Math.tan(Math.PI / 6);
/** Altitudes below this (m) are taken as this, so the path's logarithms stay finite. */
const MIN_ALTITUDE_M = 0.5;
/**
 * An end's pitch is held while the camera is within this factor of that end's altitude, and
 * gives way to looking straight down beyond it: a flight from orbit looks down until it is
 * about this many times its arrival height, then tilts in.
 */
const TILT_RANGE = 25;
/** Seconds a flight takes per unit of path length, and its shortest and longest. */
const SECONDS_PER_PATH = 0.4;
const BASE_S = 1;
const MIN_S = 1.2;
const MAX_S = 5.5;
/**
 * How quickly the destination follows a correction (rad/s, critically damped): within 2 % of
 * it in about 4/ω, 0.8 s, and never overshooting.
 */
const DESTINATION_OMEGA = 5;
/** A destination this close to its target, and this slow, has arrived (m; m/s; degrees). */
const SETTLED_M = 0.005;
const SETTLED_SPEED = 0.01;
const SETTLED_DEG = 0.01;
/** A re-aimed end is kept at least this far above the ground it is re-aimed at (m). */
const REAIM_CLEARANCE_M = 2;
/**
 * An end aimed at a guessed ground is re-aimed once the terrain under it differs by more than
 * this (m): terrain tiles refine in steps of metres, and each step need not be chased.
 */
const REAIM_THRESHOLD_M = 1;

/** The ground an arrival pose was framed above, and whether it was measured or guessed. */
export interface PoseGround {
  /** Ellipsoid height of the ground (m). */
  readonly height: number;
  /**
   * Measured: the catalog's height, the terrain sampled at full detail, the model resting on
   * its ground. Guessed: whatever the globe had loaded there, or the ellipsoid. A guessed end
   * is re-aimed at the terrain under it as that loads, keeping its height above the ground.
   */
  readonly measured: boolean;
}

/** An arrival pose with what it was framed above, when the caller knows. */
export interface GlidePose extends ArrivalPose {
  readonly ground?: PoseGround;
}

/** Van Wijk and Nuij's path between two altitudes `distanceM` apart on the ground. */
export interface ZoomPath {
  /** Its length, in the path's own units (screen widths, roughly). */
  readonly length: number;
  /** The share of the ground distance covered and the altitude (m) at path share `s`. */
  at(s: number): { along: number; altitude: number };
  /** The highest altitude on the path (m) and the path share where it is reached. */
  readonly peak: { altitude: number; at: number };
}

/**
 * Van Wijk and Nuij's optimal zoom-and-pan from altitude `altitude0` to `altitude1` across
 * `distanceM` of ground. Widths are the ground the camera sees, proportional to altitude.
 * `r0` and `r1` are written as `-asinh(b)`: the textbook `log(sqrt(b² + 1) - b)` loses every
 * digit to cancellation for the large `b` of a long, low flight.
 */
export function zoomPath(altitude0: number, altitude1: number, distanceM: number): ZoomPath {
  const a0 = Math.max(altitude0, MIN_ALTITUDE_M);
  const a1 = Math.max(altitude1, MIN_ALTITUDE_M);
  const w0 = a0 * WIDTH_PER_ALTITUDE;
  const w1 = a1 * WIDTH_PER_ALTITUDE;
  const d = Math.max(0, distanceM);
  const rho2 = RHO * RHO;
  const rho4 = rho2 * rho2;
  if (d < 1e-6 * Math.max(w0, w1) || d < 1e-3) {
    const ratio = Math.log(w1 / w0);
    return {
      length: Math.abs(ratio) / RHO,
      at: (s) => ({ along: clamp01(s), altitude: a0 * Math.exp(ratio * clamp01(s)) }),
      peak: a0 >= a1 ? { altitude: a0, at: 0 } : { altitude: a1, at: 1 },
    };
  }
  const b0 = (w1 * w1 - w0 * w0 + rho4 * d * d) / (2 * w0 * rho2 * d);
  const b1 = (w1 * w1 - w0 * w0 - rho4 * d * d) / (2 * w1 * rho2 * d);
  const r0 = -Math.asinh(b0);
  const r1 = -Math.asinh(b1);
  const length = (r1 - r0) / RHO;
  const coshR0 = Math.cosh(r0);
  const sinhR0 = Math.sinh(r0);
  const raw = (s: number) => {
    const x = RHO * s * length + r0;
    return {
      along: (w0 / (rho2 * d)) * (coshR0 * Math.tanh(x) - sinhR0),
      width: (w0 * coshR0) / Math.cosh(x),
    };
  };
  // The closed form ends within rounding of (1, w1); the ends are made exact, and the error
  // spread over the path so nothing jumps at the last frame.
  const end = raw(1);
  const alongScale = end.along > 0 ? 1 / end.along : 1;
  const widthCorrection = Math.log(w1 / end.width);
  const peakAt = r0 < 0 && r1 > 0 ? clamp01(-r0 / (r1 - r0)) : a0 >= a1 ? 0 : 1;
  const at = (s: number) => {
    const u = clamp01(s);
    const point = raw(u);
    return {
      along: clamp01(point.along * alongScale),
      altitude: (point.width * Math.exp(widthCorrection * u)) / WIDTH_PER_ALTITUDE,
    };
  };
  return { length, at, peak: { altitude: at(peakAt).altitude, at: peakAt } };
}

/** How long a flight along `path` takes (s): longer for a longer path, within bounds. */
export function glideDuration(path: ZoomPath): number {
  return Math.min(MAX_S, Math.max(MIN_S, BASE_S + SECONDS_PER_PATH * path.length));
}

/** The flight's clock: a half cosine, leaving and arriving at rest. */
export function easeInOut(t: number): number {
  const u = clamp01(t);
  return (1 - Math.cos(Math.PI * u)) / 2;
}

/** One end of a flight: a pose and the ground under it. */
export interface GlideEnd {
  longitude: number;
  latitude: number;
  height: number;
  heading: number;
  pitch: number;
  /** Ellipsoid height of the ground under it (m). */
  ground: number;
}

/** The geometry between two ends that the pose along the way is read from. */
export interface GlideGeometry {
  readonly path: ZoomPath;
  readonly geodesic: EllipsoidGeodesic | null;
  readonly distanceM: number;
}

/** The path and the ground track from `start` to `end`. */
export function glideGeometry(start: GlideEnd, end: GlideEnd): GlideGeometry {
  const from = Cartographic.fromDegrees(start.longitude, start.latitude);
  const to = Cartographic.fromDegrees(end.longitude, end.latitude);
  let geodesic: EllipsoidGeodesic | null = null;
  let distanceM = 0;
  // A geodesic between points this close (or the same point) is degenerate; a turn or a zoom
  // on the spot needs none.
  if (
    Math.abs(from.longitude - to.longitude) > 1e-12 ||
    Math.abs(from.latitude - to.latitude) > 1e-12
  ) {
    try {
      geodesic = new EllipsoidGeodesic(from, to);
      distanceM = geodesic.surfaceDistance;
      if (!Number.isFinite(distanceM)) {
        geodesic = null;
        distanceM = 0;
      }
    } catch {
      geodesic = null;
    }
  }
  const path = zoomPath(start.height - start.ground, end.height - end.ground, distanceM);
  return { path, geodesic, distanceM };
}

const scratchCarto = new Cartographic();

/**
 * The camera's pose at path share `s` of the flight from `start` to `end`: the ground track
 * along the geodesic, the altitude along the zoom path above the ground interpolated between
 * the ends, the heading turned in step, and the pitch held near each end and looking straight
 * down while high above both.
 */
export function glidePose(
  start: GlideEnd,
  end: GlideEnd,
  s: number,
  geometry: GlideGeometry = glideGeometry(start, end),
): ArrivalPose {
  const u = clamp01(s);
  if (u <= 0) return poseOf(start);
  if (u >= 1) return poseOf(end);
  const { along, altitude } = geometry.path.at(u);
  let longitude: number;
  let latitude: number;
  if (geometry.geodesic) {
    const point = geometry.geodesic.interpolateUsingFraction(along, scratchCarto);
    longitude = CesiumMath.toDegrees(point.longitude);
    latitude = CesiumMath.toDegrees(point.latitude);
  } else {
    longitude = start.longitude + wrapDegrees(end.longitude - start.longitude) * along;
    latitude = start.latitude + (end.latitude - start.latitude) * along;
  }
  const ground = start.ground + (end.ground - start.ground) * along;
  const height = ground + altitude;
  const heading = start.heading + wrapDegrees(end.heading - start.heading) * u;
  // Each end's pitch is held near its own altitude (within TILT_RANGE of it, in log terms) and
  // weighted by how near that end the flight is; what neither claims looks straight down.
  const log = Math.log(Math.max(altitude, MIN_ALTITUDE_M));
  const near = (endAltitude: number) =>
    smoothstep(
      1 - Math.abs(log - Math.log(Math.max(endAltitude, MIN_ALTITUDE_M))) / Math.log(TILT_RANGE),
    );
  const startWeight = near(start.height - start.ground) * (1 - u);
  const endWeight = near(end.height - end.ground) * u;
  const pitch = -90 + startWeight * (start.pitch + 90) + endWeight * (end.pitch + 90);
  return { longitude, latitude, height, heading: normalizeHeading(heading), pitch };
}

function poseOf(end: GlideEnd): ArrivalPose {
  return {
    longitude: end.longitude,
    latitude: end.latitude,
    height: end.height,
    heading: normalizeHeading(end.heading),
    pitch: end.pitch,
  };
}

/** What the glide needs from the scene, injected so it runs without one in tests. */
export interface GlideScene {
  /** The ground the globe has loaded under a point (m), or undefined when it has none worth
   *  believing (no tile there yet, a placeholder kilometres under the sea). */
  ground(longitude: number, latitude: number): number | undefined;
}

/** A destination as the spring carries it: the components that move, unwrapped. */
interface Destination {
  longitude: number;
  latitude: number;
  height: number;
  heading: number;
  pitch: number;
  ground: number;
}

const KEYS = ["longitude", "latitude", "height", "heading", "pitch", "ground"] as const;

/**
 * One flight, stepped frame by frame: `step(now)` is the camera's pose for that moment. The
 * destination is carried by a critically damped spring towards the latest `retarget`, the
 * clock by `easeInOut` over the duration the first path asked for, and a guessed end follows
 * the terrain the globe loads under it (`reaim`).
 */
export class Glide {
  private readonly start: GlideEnd;
  private readonly startedAt: number;
  readonly durationS: number;
  /** The pose aimed at, as last handed in (guessed grounds re-aimed as terrain loads). */
  private target: Destination;
  private targetGround: PoseGround | undefined;
  /** Where the destination is now, and how fast it is moving, per component. */
  private readonly destination: Destination;
  private readonly velocity: Destination;
  private lastStep: number;
  private finished = false;

  /**
   * `startGround` is the ground last known under the camera, for when the globe has nothing
   * worth believing there now (its tiles outside a site's outline are not loaded in the
   * photorealistic world); without either, the end's ground is the best guess. Taken as 0, a
   * short hop between two scans in the mountains believed itself two kilometres up and arced
   * hundreds of metres to get there.
   */
  constructor(
    start: { longitude: number; latitude: number; height: number; heading: number; pitch: number },
    end: GlidePose,
    private readonly scene: GlideScene,
    now: number,
    durationS?: number,
    startGround?: number,
  ) {
    this.target = this.destinationOf(end);
    this.targetGround = end.ground;
    const ground =
      plausible(scene.ground(start.longitude, start.latitude)) ??
      plausible(startGround) ??
      this.target.ground;
    this.start = { ...start, ground: Math.min(ground, start.height - MIN_ALTITUDE_M) };
    this.destination = { ...this.target };
    this.velocity = { longitude: 0, latitude: 0, height: 0, heading: 0, pitch: 0, ground: 0 };
    this.startedAt = now;
    this.lastStep = now;
    this.durationS =
      durationS ?? glideDuration(glideGeometry(this.start, this.endOf(this.destination)).path);
  }

  /** Whether the flight has arrived (its last `step` returned the destination itself). */
  get done(): boolean {
    return this.finished;
  }

  /** The pose the flight is now going to: the latest `retarget`, or the first. */
  get aim(): ArrivalPose {
    return poseOf(this.endOf(this.target));
  }

  /** Points the flight at a better pose; the destination moves to it smoothly from now. */
  retarget(end: GlidePose): void {
    this.target = this.destinationOf(end, this.destination);
    this.targetGround = end.ground;
    this.finished = false;
  }

  /** The camera's pose at `now` (ms, the clock the glide was started with). */
  step(now: number): ArrivalPose {
    const dt = Math.max(0, (now - this.lastStep) / 1000);
    this.lastStep = now;
    this.reaim();
    for (const key of KEYS) {
      const next = criticallyDamped(
        this.destination[key],
        this.velocity[key],
        this.target[key],
        DESTINATION_OMEGA,
        dt,
      );
      this.destination[key] = next.value;
      this.velocity[key] = next.velocity;
    }
    const t = this.durationS > 0 ? (now - this.startedAt) / 1000 / this.durationS : 1;
    const end = this.endOf(this.destination);
    if (t >= 1 && this.settled()) {
      this.finished = true;
      return poseOf(this.endOf(this.target));
    }
    const geometry = glideGeometry(this.start, end);
    const pose = glidePose(this.start, end, easeInOut(t), geometry);
    return pose;
  }

  /**
   * A guessed end is re-aimed at the terrain under it once the globe has some worth believing
   * there, at the same height above it: an aim at the ellipsoid two kilometres under a
   * mountain scan comes up to it as the mountain's tiles load during the descent. A measured
   * end is left alone: what the globe has loaded on the way down is a coarse tile, which in
   * the mountains is hundreds of metres off the terrain sampled at full detail (Spool's said
   * 2,968 m over ground at 2,001 m, and an end re-aimed at it landed a kilometre up).
   */
  private reaim(): void {
    if (this.targetGround?.measured !== false) return;
    const terrain = plausible(
      this.scene.ground(normalizeLongitude(this.target.longitude), this.target.latitude),
    );
    if (terrain === undefined || Math.abs(terrain - this.target.ground) < REAIM_THRESHOLD_M) return;
    const above = Math.max(this.target.height - this.target.ground, REAIM_CLEARANCE_M);
    this.target = { ...this.target, height: terrain + above, ground: terrain };
  }

  private settled(): boolean {
    for (const key of KEYS) {
      const tolerance =
        key === "longitude" || key === "latitude"
          ? SETTLED_M / 111_000
          : key === "heading" || key === "pitch"
            ? SETTLED_DEG
            : SETTLED_M;
      if (Math.abs(this.destination[key] - this.target[key]) > tolerance) return false;
      const speedTolerance =
        key === "longitude" || key === "latitude" ? SETTLED_SPEED / 111_000 : SETTLED_SPEED;
      if (Math.abs(this.velocity[key]) > speedTolerance) return false;
    }
    return true;
  }

  /** A pose as the spring carries it: longitude and heading unwrapped next to `near`. */
  private destinationOf(pose: GlidePose, near?: Destination): Destination {
    const ground =
      pose.ground?.height ??
      plausible(this.scene.ground(pose.longitude, pose.latitude)) ??
      Math.min(0, pose.height - MIN_ALTITUDE_M);
    return {
      longitude: near
        ? near.longitude + wrapDegrees(pose.longitude - near.longitude)
        : pose.longitude,
      latitude: pose.latitude,
      height: pose.height,
      heading: near ? near.heading + wrapDegrees(pose.heading - near.heading) : pose.heading,
      pitch: pose.pitch,
      ground: Math.min(ground, pose.height - MIN_ALTITUDE_M),
    };
  }

  private endOf(destination: Destination): GlideEnd {
    return {
      longitude: normalizeLongitude(destination.longitude),
      latitude: destination.latitude,
      height: destination.height,
      heading: destination.heading,
      pitch: destination.pitch,
      ground: Math.min(destination.ground, destination.height - MIN_ALTITUDE_M),
    };
  }
}

/**
 * One step of a critically damped spring from `value` (moving at `velocity`) towards
 * `target`, solved exactly for `dt` seconds: frame-rate independent, and never overshooting
 * from rest.
 */
export function criticallyDamped(
  value: number,
  velocity: number,
  target: number,
  omega: number,
  dt: number,
): { value: number; velocity: number } {
  if (!(dt > 0)) return { value, velocity };
  const offset = value - target;
  const c = velocity + omega * offset;
  const decay = Math.exp(-omega * dt);
  return {
    value: target + (offset + c * dt) * decay,
    velocity: (velocity - omega * c * dt) * decay,
  };
}

function clamp01(x: number): number {
  return x <= 0 ? 0 : x >= 1 ? 1 : x;
}

function smoothstep(x: number): number {
  const u = clamp01(x);
  return u * u * (3 - 2 * u);
}

/** The shortest signed turn from 0 to `degrees`, in (-180, 180]. */
function wrapDegrees(degrees: number): number {
  const wrapped = ((((degrees + 180) % 360) + 360) % 360) - 180;
  return wrapped === -180 ? 180 : wrapped;
}

function normalizeHeading(degrees: number): number {
  return ((degrees % 360) + 360) % 360;
}

function normalizeLongitude(degrees: number): number {
  return wrapDegrees(degrees);
}
