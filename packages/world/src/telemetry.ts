/**
 * Telemetry poses for live mode (step C3, docs/SCENE_OBJECTS.md §4 "Telemetry on instances"):
 * the frames a pose may arrive in, the rigid motion it means for a scanned object, and the
 * playout track that turns irregular, late, missing readings into a pose at every frame.
 *
 * Framework-free: every clock is a number handed in (milliseconds), so a frame is a function
 * of the readings and the time asked for.
 *
 * **Frames.** A reading's pose is the pose of the object's body frame:
 *
 * - `scan`: the tileset's local ENU frame (the root transform's frame, metres; what
 *   `instances.json` and `skin.json` are written in), orientation body → scan;
 * - `ecef`: WGS84 earth-centred, earth-fixed metres, orientation body → ECEF;
 * - `geodetic`: `[longitude°, latitude°, ellipsoid height m]`, orientation body → the local
 *   east-north-up frame at that point (what a GNSS receiver and a compass give).
 *
 * Everything is brought into the scan frame with the root's computed transform
 * (`scanToEcef`, column-major 4×4), whose linear part is taken as a rotation.
 *
 * **Rigid motion.** A binding names the body frame's pose at capture (`rest`). A reading
 * `P` then moves the object by `M = P · rest⁻¹`: `x' = R x + t` in the scan frame, the
 * identity at rest.
 */

import {
  DEG_TO_RAD,
  QUAT_IDENTITY,
  quatConjugate,
  quatFromAxisAngle,
  quatMultiply,
  quatRotate,
  type Quat,
  type Vec3,
} from "./vec";

export type PoseFrame = "scan" | "ecef" | "geodetic";

export const POSE_FRAMES: readonly PoseFrame[] = ["scan", "ecef", "geodetic"];

/** A body frame's pose: where its origin is, and its rotation body → frame. */
export interface Pose {
  readonly position: Vec3;
  readonly orientation: Quat;
}

/** One reading: a pose in `frame` at source time `t` (milliseconds, the source's clock). */
export interface PoseSample extends Pose {
  readonly t: number;
  readonly frame: PoseFrame;
}

/** A rigid motion of the scan frame: `x' = R x + t`. */
export interface RigidMotion {
  readonly rotation: Quat;
  readonly translation: Vec3;
}

export const RIGID_IDENTITY: RigidMotion = { rotation: QUAT_IDENTITY, translation: [0, 0, 0] };

/** A column-major 4×4 matrix (a CesiumJS `Matrix4` as an array). */
export type Matrix4Like = ArrayLike<number>;

// ---- Quaternions -------------------------------------------------------------------------

export function quatNormalize(q: Quat): Quat {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!(n > 0) || !Number.isFinite(n)) return QUAT_IDENTITY;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/**
 * Spherical interpolation from `a` (`s = 0`) to `b` (`s = 1`) the short way round; `s` outside
 * `[0, 1]` extrapolates along the same great circle.
 */
export function quatSlerp(a: Quat, b: Quat, s: number): Quat {
  let [bx, by, bz, bw] = b;
  let cos = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
  if (cos < 0) {
    bx = -bx;
    by = -by;
    bz = -bz;
    bw = -bw;
    cos = -cos;
  }
  if (cos > 1 - 1e-9) {
    return quatNormalize([
      a[0] + (bx - a[0]) * s,
      a[1] + (by - a[1]) * s,
      a[2] + (bz - a[2]) * s,
      a[3] + (bw - a[3]) * s,
    ]);
  }
  const angle = Math.acos(Math.min(1, cos));
  const sin = Math.sin(angle);
  const ka = Math.sin((1 - s) * angle) / sin;
  const kb = Math.sin(s * angle) / sin;
  return quatNormalize([
    a[0] * ka + bx * kb,
    a[1] * ka + by * kb,
    a[2] * ka + bz * kb,
    a[3] * ka + bw * kb,
  ]);
}

/** The unit quaternion of a rotation matrix given as rows `m[r * 3 + c]`. */
export function quatFromRotationRows(m: ArrayLike<number>): Quat {
  const at = (r: number, c: number): number => m[r * 3 + c] ?? 0;
  const trace = at(0, 0) + at(1, 1) + at(2, 2);
  let q: Quat;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    q = [(at(2, 1) - at(1, 2)) / s, (at(0, 2) - at(2, 0)) / s, (at(1, 0) - at(0, 1)) / s, s / 4];
  } else if (at(0, 0) > at(1, 1) && at(0, 0) > at(2, 2)) {
    const s = Math.sqrt(1 + at(0, 0) - at(1, 1) - at(2, 2)) * 2;
    q = [s / 4, (at(0, 1) + at(1, 0)) / s, (at(0, 2) + at(2, 0)) / s, (at(2, 1) - at(1, 2)) / s];
  } else if (at(1, 1) > at(2, 2)) {
    const s = Math.sqrt(1 + at(1, 1) - at(0, 0) - at(2, 2)) * 2;
    q = [(at(0, 1) + at(1, 0)) / s, s / 4, (at(1, 2) + at(2, 1)) / s, (at(0, 2) - at(2, 0)) / s];
  } else {
    const s = Math.sqrt(1 + at(2, 2) - at(0, 0) - at(1, 1)) * 2;
    q = [(at(0, 2) + at(2, 0)) / s, (at(1, 2) + at(2, 1)) / s, s / 4, (at(1, 0) - at(0, 1)) / s];
  }
  return quatNormalize(q);
}

/**
 * Body → local level frame (x east, y north, z up) from a heading (degrees clockwise from
 * north), pitch (nose up) and roll (right side down), for a body whose x axis points forward,
 * y left and z up.
 */
export function quatFromHeadingPitchRoll(headingDeg: number, pitchDeg = 0, rollDeg = 0): Quat {
  const yaw = quatFromAxisAngle([0, 0, 1], (90 - headingDeg) * DEG_TO_RAD);
  const pitch = quatFromAxisAngle([0, 1, 0], -pitchDeg * DEG_TO_RAD);
  const roll = quatFromAxisAngle([1, 0, 0], rollDeg * DEG_TO_RAD);
  return quatMultiply(yaw, quatMultiply(pitch, roll));
}

/** A turn of `yawRad` about the vertical (counter-clockwise from above). */
export function quatFromYaw(yawRad: number): Quat {
  return quatFromAxisAngle([0, 0, 1], yawRad);
}

// ---- WGS84 -------------------------------------------------------------------------------

const WGS84_A = 6378137;
const WGS84_F = 1 / 298.257223563;
const WGS84_E2 = WGS84_F * (2 - WGS84_F);
const WGS84_B = WGS84_A * (1 - WGS84_F);
const WGS84_EP2 = (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B);

/** `[lon°, lat°, h]` (ellipsoid height, metres) to ECEF metres. */
export function geodeticToEcef(geodetic: Vec3): Vec3 {
  const lon = geodetic[0] * DEG_TO_RAD;
  const lat = geodetic[1] * DEG_TO_RAD;
  const h = geodetic[2];
  const sinLat = Math.sin(lat);
  const n = WGS84_A / Math.sqrt(1 - WGS84_E2 * sinLat * sinLat);
  return [
    (n + h) * Math.cos(lat) * Math.cos(lon),
    (n + h) * Math.cos(lat) * Math.sin(lon),
    (n * (1 - WGS84_E2) + h) * sinLat,
  ];
}

/** ECEF metres to `[lon°, lat°, h]`: Bowring's start, then two Newton steps (sub-millimetre). */
export function ecefToGeodetic(ecef: Vec3): Vec3 {
  const [x, y, z] = ecef;
  const p = Math.hypot(x, y);
  const lon = Math.atan2(y, x);
  if (p < 1e-9) {
    const lat = z >= 0 ? Math.PI / 2 : -Math.PI / 2;
    return [lon / DEG_TO_RAD, lat / DEG_TO_RAD, Math.abs(z) - WGS84_B];
  }
  const theta = Math.atan2(z * WGS84_A, p * WGS84_B);
  let lat = Math.atan2(
    z + WGS84_EP2 * WGS84_B * Math.sin(theta) ** 3,
    p - WGS84_E2 * WGS84_A * Math.cos(theta) ** 3,
  );
  let h = 0;
  for (let k = 0; k < 3; k += 1) {
    const sinLat = Math.sin(lat);
    const n = WGS84_A / Math.sqrt(1 - WGS84_E2 * sinLat * sinLat);
    h = p / Math.cos(lat) - n;
    lat = Math.atan2(z, p * (1 - (WGS84_E2 * n) / (n + h)));
  }
  return [lon / DEG_TO_RAD, lat / DEG_TO_RAD, h];
}

/** The rotation local ENU → ECEF at `[lon°, lat°]`: columns east, north, up. */
export function enuToEcefRotation(lonDeg: number, latDeg: number): Quat {
  const lon = lonDeg * DEG_TO_RAD;
  const lat = latDeg * DEG_TO_RAD;
  const sl = Math.sin(lon);
  const cl = Math.cos(lon);
  const sp = Math.sin(lat);
  const cp = Math.cos(lat);
  // Rows of [e n u].
  return quatFromRotationRows([-sl, -sp * cl, cp * cl, cl, -sp * sl, cp * sl, 0, cp, sp]);
}

// ---- The scan frame ----------------------------------------------------------------------

/** The scan frame's place on the earth: the root's computed transform, split and inverted. */
export interface ScanFrame {
  /** Scan → ECEF, column-major. */
  readonly toEcef: readonly number[];
  /** ECEF → scan, column-major. */
  readonly fromEcef: readonly number[];
  /** The rotation part of scan → ECEF (orthonormalised). */
  readonly rotation: Quat;
}

/** A scan frame from `scanToEcef`, or undefined when it is singular. */
export function scanFrame(scanToEcef: Matrix4Like): ScanFrame | undefined {
  const m = Array.from({ length: 16 }, (_, i) => scanToEcef[i] ?? 0);
  const inverse = invertAffine4(m);
  if (inverse === undefined) return undefined;
  // Columns 0..2 of the linear part, Gram-Schmidt so a scaled or sheared root still gives a
  // rotation.
  const c0 = unit([m[0] ?? 0, m[1] ?? 0, m[2] ?? 0]);
  const c1raw: Vec3 = [m[4] ?? 0, m[5] ?? 0, m[6] ?? 0];
  const d = c0[0] * c1raw[0] + c0[1] * c1raw[1] + c0[2] * c1raw[2];
  const c1 = unit([c1raw[0] - d * c0[0], c1raw[1] - d * c0[1], c1raw[2] - d * c0[2]]);
  const c2: Vec3 = [
    c0[1] * c1[2] - c0[2] * c1[1],
    c0[2] * c1[0] - c0[0] * c1[2],
    c0[0] * c1[1] - c0[1] * c1[0],
  ];
  const rotation = quatFromRotationRows([
    c0[0],
    c1[0],
    c2[0],
    c0[1],
    c1[1],
    c2[1],
    c0[2],
    c1[2],
    c2[2],
  ]);
  return { toEcef: m, fromEcef: inverse, rotation };
}

function unit(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]);
  return n > 0 ? [v[0] / n, v[1] / n, v[2] / n] : [0, 0, 0];
}

function invertAffine4(m: readonly number[]): number[] | undefined {
  const a = (r: number, c: number): number => m[c * 4 + r] ?? 0;
  const det =
    a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) -
    a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
    a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
  if (!(Math.abs(det) > 1e-300) || !Number.isFinite(det)) return undefined;
  const inv = [
    (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) / det,
    (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) / det,
    (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) / det,
    (a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) / det,
    (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) / det,
    (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) / det,
    (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)) / det,
    (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) / det,
    (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) / det,
  ];
  const r = (i: number, j: number): number => inv[i * 3 + j] ?? 0;
  const t = [a(0, 3), a(1, 3), a(2, 3)];
  const out = new Array<number>(16).fill(0);
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) out[j * 4 + i] = r(i, j);
    out[12 + i] = -(r(i, 0) * (t[0] ?? 0) + r(i, 1) * (t[1] ?? 0) + r(i, 2) * (t[2] ?? 0));
  }
  out[15] = 1;
  return out;
}

function applyPoint(m: readonly number[], p: Vec3): Vec3 {
  return [
    (m[0] ?? 0) * p[0] + (m[4] ?? 0) * p[1] + (m[8] ?? 0) * p[2] + (m[12] ?? 0),
    (m[1] ?? 0) * p[0] + (m[5] ?? 0) * p[1] + (m[9] ?? 0) * p[2] + (m[13] ?? 0),
    (m[2] ?? 0) * p[0] + (m[6] ?? 0) * p[1] + (m[10] ?? 0) * p[2] + (m[14] ?? 0),
  ];
}

/** A pose in `frame` brought into the scan frame; undefined when the frame needs a placement. */
export function poseToScan(
  pose: Pose,
  frame: PoseFrame,
  scan: ScanFrame | undefined,
): Pose | undefined {
  if (frame === "scan") return { position: pose.position, orientation: pose.orientation };
  if (!scan) return undefined;
  let position = pose.position;
  let orientation = pose.orientation;
  if (frame === "geodetic") {
    orientation = quatMultiply(enuToEcefRotation(position[0], position[1]), orientation);
    position = geodeticToEcef(position);
  }
  return {
    position: applyPoint(scan.fromEcef, position),
    orientation: quatNormalize(quatMultiply(quatConjugate(scan.rotation), orientation)),
  };
}

/** A scan-frame pose expressed in `frame` (what a synthetic source in that frame emits). */
export function poseFromScan(
  pose: Pose,
  frame: PoseFrame,
  scan: ScanFrame | undefined,
): Pose | undefined {
  if (frame === "scan") return { position: pose.position, orientation: pose.orientation };
  if (!scan) return undefined;
  const position = applyPoint(scan.toEcef, pose.position);
  const orientation = quatNormalize(quatMultiply(scan.rotation, pose.orientation));
  if (frame === "ecef") return { position, orientation };
  const geodetic = ecefToGeodetic(position);
  return {
    position: geodetic,
    orientation: quatNormalize(
      quatMultiply(quatConjugate(enuToEcefRotation(geodetic[0], geodetic[1])), orientation),
    ),
  };
}

// ---- Rigid motion ------------------------------------------------------------------------

/** `M = pose · rest⁻¹`: what moves the object scanned at `rest` to `pose` (scan frame). */
export function relativeMotion(rest: Pose, pose: Pose): RigidMotion {
  const rotation = quatNormalize(quatMultiply(pose.orientation, quatConjugate(rest.orientation)));
  const moved = quatRotate(rotation, rest.position);
  return {
    rotation,
    translation: [
      pose.position[0] - moved[0],
      pose.position[1] - moved[1],
      pose.position[2] - moved[2],
    ],
  };
}

/** Applies `motion` to a point. */
export function applyMotion(motion: RigidMotion, point: Vec3): Vec3 {
  const r = quatRotate(motion.rotation, point);
  return [r[0] + motion.translation[0], r[1] + motion.translation[1], r[2] + motion.translation[2]];
}

/**
 * The translation of the same motion written about `origin`, `x' = x + (R − I)(x − o) + t_o`
 * (a skin's constant handle): `t_o = t + (R − I) o`.
 */
export function translationAbout(motion: RigidMotion, origin: Vec3): Vec3 {
  const ro = quatRotate(motion.rotation, origin);
  return [
    motion.translation[0] + ro[0] - origin[0],
    motion.translation[1] + ro[1] - origin[1],
    motion.translation[2] + ro[2] - origin[2],
  ];
}

/** Whether `motion` moves nothing (exactly). */
export function isIdentityMotion(motion: RigidMotion): boolean {
  const [x, y, z] = motion.rotation;
  const [tx, ty, tz] = motion.translation;
  return x === 0 && y === 0 && z === 0 && tx === 0 && ty === 0 && tz === 0;
}

/** Linear between poses (position lerp, orientation slerp); `s` past 1 extrapolates. */
export function interpolatePose(a: Pose, b: Pose, s: number): Pose {
  return {
    position: [
      a.position[0] + (b.position[0] - a.position[0]) * s,
      a.position[1] + (b.position[1] - a.position[1]) * s,
      a.position[2] + (b.position[2] - a.position[2]) * s,
    ],
    orientation: quatSlerp(a.orientation, b.orientation, s),
  };
}

// ---- Playout track -----------------------------------------------------------------------

/** What happens once a track's last reading is older than `staleMs`. */
export type StalePolicy = "freeze" | "rest";

export interface TrackOptions {
  /**
   * Playout delay, ms: the pose shown is the one `latencyMs` behind the newest the clock
   * estimate allows, so readings arriving up to that late are still interpolated, not
   * extrapolated.
   */
  readonly latencyMs: number;
  /** How far past the last reading to dead-reckon (constant velocity), ms. */
  readonly extrapolateMs: number;
  /** A reading older than this (at the playout time) makes the track stale, ms. */
  readonly staleMs: number;
  /** Stale: hold the last pose (`freeze`) or fade back to the rest pose (`rest`). */
  readonly stale: StalePolicy;
  /** How long the fade to rest takes, ms. */
  readonly fadeMs: number;
}

export const DEFAULT_TRACK_OPTIONS: TrackOptions = {
  latencyMs: 250,
  extrapolateMs: 1000,
  staleMs: 3000,
  stale: "rest",
  fadeMs: 2000,
};

/** Readings the clock-offset estimate looks back over. */
const OFFSET_WINDOW = 32;
/** Readings kept behind the playout time (at least two, for extrapolation). */
const KEEP_BEHIND = 4;
/** Two readings further apart than this are not used for a velocity, ms. */
const MAX_VELOCITY_GAP_MS = 5000;

/**
 * - `none`: no reading yet (rest);
 * - `live`: between readings (or before the first: held at it);
 * - `extrapolated`: past the last reading, dead-reckoned;
 * - `held`: past the extrapolation limit, held there;
 * - `stale`: past `staleMs`: frozen, or fading to rest;
 * - `rest`: faded back to rest.
 */
export type TrackState = "none" | "live" | "extrapolated" | "held" | "stale" | "rest";

export interface TrackReading {
  readonly state: TrackState;
  /** The pose to show (scan frame), or null at rest. */
  readonly pose: Pose | null;
  /** Share of `pose` over the rest pose: 1 except while fading to rest. */
  readonly weight: number;
  /** Playout time minus the last reading's time, ms (null with no reading). */
  readonly ageMs: number | null;
  /** The source time shown, ms. */
  readonly playoutMs: number;
}

interface Stamped extends Pose {
  readonly t: number;
}

/**
 * Turns readings into a pose at any local time: a playout buffer with a clock-offset estimate,
 * interpolation, bounded dead reckoning and a staleness rule.
 *
 * The source's clock and the local one need not agree: the offset is the smallest
 * `arrival − t` over the last readings (the fastest transport seen), so the playout time is
 * `now − offset − latencyMs` in source time.
 */
export class PoseTrack {
  readonly options: TrackOptions;
  #samples: Stamped[] = [];
  #offsets: number[] = [];
  #offset: number | undefined;

  constructor(options: Partial<TrackOptions> = {}) {
    this.options = { ...DEFAULT_TRACK_OPTIONS, ...options };
  }

  /** Readings held. */
  get size(): number {
    return this.#samples.length;
  }

  /** The source-minus-local clock offset in use, ms. */
  get offsetMs(): number | undefined {
    return this.#offset;
  }

  /** Adds a reading (scan frame) that arrived at local time `arrivalMs`. */
  push(t: number, pose: Pose, arrivalMs: number): void {
    if (!Number.isFinite(t) || !Number.isFinite(arrivalMs)) return;
    const sample: Stamped = { t, position: pose.position, orientation: pose.orientation };
    const samples = this.#samples;
    let at = samples.length;
    while (at > 0 && (samples[at - 1]?.t ?? -Infinity) > t) at -= 1;
    if (at > 0 && samples[at - 1]?.t === t) samples[at - 1] = sample;
    else samples.splice(at, 0, sample);
    this.#offsets.push(arrivalMs - t);
    if (this.#offsets.length > OFFSET_WINDOW) this.#offsets.shift();
    this.#offset = Math.min(...this.#offsets);
  }

  /** Forgets everything. */
  clear(): void {
    this.#samples = [];
    this.#offsets = [];
    this.#offset = undefined;
  }

  /** The pose at local time `nowMs`. */
  read(nowMs: number): TrackReading {
    const samples = this.#samples;
    const offset = this.#offset;
    const last = samples[samples.length - 1];
    const first = samples[0];
    if (last === undefined || first === undefined || offset === undefined) {
      return { state: "none", pose: null, weight: 0, ageMs: null, playoutMs: nowMs };
    }
    const o = this.options;
    const playout = nowMs - offset - o.latencyMs;
    this.#trim(playout);
    if (playout <= first.t) {
      return { state: "live", pose: first, weight: 1, ageMs: playout - last.t, playoutMs: playout };
    }
    if (playout <= last.t) {
      let i = samples.length - 1;
      while (i > 0 && (samples[i - 1]?.t ?? 0) > playout) i -= 1;
      const a = samples[i - 1] ?? first;
      const b = samples[i] ?? last;
      const s = b.t > a.t ? (playout - a.t) / (b.t - a.t) : 1;
      return {
        state: "live",
        pose: interpolatePose(a, b, s),
        weight: 1,
        ageMs: playout - last.t,
        playoutMs: playout,
      };
    }
    const age = playout - last.t;
    const ahead = Math.min(age, o.extrapolateMs);
    const before = samples[samples.length - 2];
    let pose: Pose = last;
    if (before !== undefined && ahead > 0) {
      const gap = last.t - before.t;
      if (gap > 0 && gap <= MAX_VELOCITY_GAP_MS)
        pose = interpolatePose(before, last, 1 + ahead / gap);
    }
    if (age <= o.extrapolateMs) {
      return { state: "extrapolated", pose, weight: 1, ageMs: age, playoutMs: playout };
    }
    if (age <= o.staleMs) return { state: "held", pose, weight: 1, ageMs: age, playoutMs: playout };
    if (o.stale === "freeze") {
      return { state: "stale", pose, weight: 1, ageMs: age, playoutMs: playout };
    }
    const weight = o.fadeMs > 0 ? Math.max(0, 1 - (age - o.staleMs) / o.fadeMs) : 0;
    if (weight <= 0)
      return { state: "rest", pose: null, weight: 0, ageMs: age, playoutMs: playout };
    return { state: "stale", pose, weight, ageMs: age, playoutMs: playout };
  }

  /** Drops readings well behind the playout time, keeping a few for interpolation. */
  #trim(playout: number): void {
    const samples = this.#samples;
    let behind = 0;
    while (behind < samples.length && (samples[behind]?.t ?? 0) <= playout) behind += 1;
    const drop = behind - KEEP_BEHIND;
    if (drop > 0) samples.splice(0, drop);
  }
}

/**
 * The pose a reading means once faded toward rest: `weight` 1 is `pose`, 0 is `rest`.
 */
export function fadeTowardRest(rest: Pose, pose: Pose, weight: number): Pose {
  return weight >= 1 ? pose : interpolatePose(rest, pose, Math.max(0, weight));
}

// ---- Synthetic telemetry ------------------------------------------------------------------

/** A deterministic path in the scan frame; the body's x axis points along it. */
export type SyntheticPath =
  | {
      readonly kind: "circle";
      readonly centre: Vec3;
      readonly radius: number;
      /** Seconds per lap. */
      readonly periodS: number;
      /** Where on the circle at t = 0, degrees counter-clockwise from east. */
      readonly phaseDeg?: number;
      readonly clockwise?: boolean;
    }
  | {
      readonly kind: "polyline";
      readonly points: readonly Vec3[];
      readonly speedMps: number;
      /** Back to the first point and round again (the default), or stop at the last. */
      readonly closed?: boolean;
    };

/** The body pose on `path` at `tS` seconds (scan frame). */
export function pathPose(path: SyntheticPath, tS: number): Pose {
  if (path.kind === "circle") {
    const sign = path.clockwise === true ? -1 : 1;
    const turn = path.periodS > 0 ? (sign * 2 * Math.PI * tS) / path.periodS : 0;
    const theta = (path.phaseDeg ?? 0) * DEG_TO_RAD + turn;
    return {
      position: [
        path.centre[0] + path.radius * Math.cos(theta),
        path.centre[1] + path.radius * Math.sin(theta),
        path.centre[2],
      ],
      orientation: quatFromYaw(theta + (sign * Math.PI) / 2),
    };
  }
  const points = path.points;
  const closed = path.closed !== false;
  const legs: { from: Vec3; to: Vec3; length: number }[] = [];
  const n = points.length;
  for (let i = 0; i < (closed ? n : n - 1); i += 1) {
    const from = points[i];
    const to = points[(i + 1) % n];
    if (!from || !to) continue;
    const length = Math.hypot(to[0] - from[0], to[1] - from[1], to[2] - from[2]);
    if (length > 0) legs.push({ from, to, length });
  }
  const total = legs.reduce((sum, leg) => sum + leg.length, 0);
  const start = points[0] ?? [0, 0, 0];
  if (legs.length === 0 || !(total > 0)) return { position: start, orientation: QUAT_IDENTITY };
  let s = Math.max(0, path.speedMps * tS);
  s = closed ? s % total : Math.min(s, total);
  for (const leg of legs) {
    if (s <= leg.length || leg === legs[legs.length - 1]) {
      const k = Math.min(1, s / leg.length);
      const dx = leg.to[0] - leg.from[0];
      const dy = leg.to[1] - leg.from[1];
      return {
        position: [
          leg.from[0] + dx * k,
          leg.from[1] + dy * k,
          leg.from[2] + (leg.to[2] - leg.from[2]) * k,
        ],
        orientation: quatFromYaw(Math.atan2(dy, dx)),
      };
    }
    s -= leg.length;
  }
  return { position: start, orientation: QUAT_IDENTITY };
}

/** How a synthetic source samples and delivers its path. */
export interface SyntheticTelemetry {
  readonly path: SyntheticPath;
  /** Readings a second. */
  readonly rateHz: number;
  /** Transport delay, ms (every reading arrives this much after its time). */
  readonly delayMs?: number;
  /** Extra delay per reading, uniform in `[0, jitterMs)`, from a hash of its index. */
  readonly jitterMs?: number;
  /** Source-time windows `[from, to)` in which nothing is sent, ms. */
  readonly dropouts?: readonly (readonly [number, number])[];
  /** Source time of reading 0, ms (the path's t = 0). */
  readonly startMs?: number;
}

/** A hash of `k` in `[0, 1)`: the same jitter for the same reading, every run. */
function unitHash(k: number): number {
  let h = Math.imul(k | 0, 0x9e3779b1) ^ 0x85ebca6b;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d);
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b);
  return ((h ^ (h >>> 16)) >>> 0) / 2 ** 32;
}

/** Reading `k` of a synthetic source: its time, when it arrives, its pose; null in a dropout. */
export function syntheticReading(
  config: SyntheticTelemetry,
  k: number,
): { t: number; arrival: number; pose: Pose } | null {
  const period = 1000 / config.rateHz;
  const t = (config.startMs ?? 0) + k * period;
  for (const [from, to] of config.dropouts ?? []) if (t >= from && t < to) return null;
  const arrival = t + (config.delayMs ?? 0) + (config.jitterMs ?? 0) * unitHash(k);
  return { t, arrival, pose: pathPose(config.path, (t - (config.startMs ?? 0)) / 1000) };
}
