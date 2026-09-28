/**
 * Living Mode, phase 1: stateless modal wind animation of a rigged tree from its skeleton alone.
 *
 * Every rig joint belongs to one **branch** (see `branchStructure`), and every branch is one damped
 * harmonic oscillator — the trunk's is the whole-tree pendulum mode. What a branch does at time
 * `t` is read out of a precomputed **spectral motion texture** along a straight, irrationally
 * sloped trajectory (`spectral.ts`, after Habel, Kusternig & Wimmer, EG 2009), so the signal has
 * the oscillator's stationary response spectrum, never repeats, and needs no previous frame.
 * Parent motion is carried to children as `deform` does — each joint's local rotation, then its
 * parent's transform — with one difference: joint `i` is a hinge at its **parent's** rest
 * position, so its rotation bends the segment `parent → i` and everything beyond it. (`deform`
 * pivots each joint about itself, which leaves every limb's first segment rigid and a
 * single-joint limb unable to bend at all.)
 *
 * ```text
 * U_i(t)   = U · (1 + gust(t − x_i/U))                          gust convected at the mean speed
 * q_i(t)   = (U_i(t) / U_ref)²                                  drag: deflection ∝ U²
 * a_b      = (1 + f_b/U)^(−5/6)                                 Simiu–Scanlan shape at f_b, √
 * r_i(t)   = g_i · q_i(t) · [ d_i × ŵ  +  α·a_b·s1_b(t)·e1_i  +  β·a_b·s2_b(t)·e2_i ]
 * s_b(t)   = T_ζb( p0_b + v_b·t ),  |v_b| = f_b · λ             texture, trajectory, speed
 * ```
 *
 * `r_i` is the joint's rotation vector: a mean lean about `d_i × ŵ` (the limb's own direction —
 * its chord, attachment to far end, shared by all its joints so it bends in one plane — crossed
 * with downwind, so a limb pointing downwind is not bent by it — Habel eqs. 18–19) plus
 * turbulent sway about the two axes across the limb. `g_i` is the joint's share of its branch's
 * bend; `α`, `β` are the along/across turbulence ratios.
 *
 * **Pure.** The output is a function of `(t, wind, sidecar seed, rig)` and nothing else. The
 * textures and trajectories are memoised, which is not observable. At `U = 0` every rotation is
 * the identity *by value*, so calm restores the canonical bytes exactly.
 */

import { IDENTITY_TRANSFORM, type NodeTransform } from "./deform";
import { FLUTTER_STILL } from "./flutter";
import { advectedFlutterUnitBound, flutterLookups, type AdvectedFlutterField } from "./leafFlutter";
import {
  branchStructure,
  parseMotionSidecar,
  validateMotionSidecar,
  type GustSettings,
  type MotionSidecar,
} from "./motionParams";
import { hash32 } from "./noise";
import { nodeAngleLimit, type MotionRig } from "./rig";
import {
  BRANCH_TEXELS_PER_CYCLE,
  BRANCH_TEXTURE_SIZE,
  branchMotionTexture,
  FLUTTER_MAX_WAVELENGTH_LEAVES,
  FLUTTER_MIN_WAVELENGTH_LEAVES,
  FLUTTER_TEXTURE_SIZE,
  flutterTexture,
  sampleTexture,
  trajectoryDirection,
  unitUniform,
  type MotionTexture,
} from "./spectral";
import {
  add,
  quatMultiply,
  quatRotate,
  softLimit,
  subtract,
  DEG_TO_RAD,
  type Quat,
  type Vec3,
} from "./vec";
import { type WindSettings } from "./wind";

export type Season = "summer" | "winter";

/** The wind a Living Mode frame is evaluated under. Physical units, unlike `WindSettings`. */
export interface LivingWind {
  /** Mean wind speed, m/s. `0` is calm and restores the measured pose exactly. */
  readonly speedMps: number;
  /** Downwind bearing, degrees clockwise from north. */
  readonly bearingDeg: number;
  readonly gust: GustSettings;
  readonly season: Season;
}

/**
 * Mean speed at the top of the scene's dimensionless wind control, m/s (Beaufort 8).
 *
 * The control's `strength` is read as **dynamic pressure**: `U = 20·√strength`, so deflection —
 * which goes as `U²` — is linear in the control. The default strength of 0.1 is then 6.3 m/s,
 * a moderate breeze. The amplitude at that speed is still not validated against any real tree;
 * only the scaling law is.
 */
export const WIND_FULL_SCALE_MPS = 20;

export function speedFromStrength(strength: number): number {
  const s = Number.isFinite(strength) ? Math.min(1, Math.max(0, strength)) : 0;
  return s === 0 ? 0 : WIND_FULL_SCALE_MPS * Math.sqrt(s);
}

/** The scene's `WindSettings`, read through a sidecar's gust defaults. */
export function livingWindFromSettings(
  settings: WindSettings,
  sidecar: MotionSidecar,
  season: Season = "summer",
): LivingWind {
  return {
    speedMps: speedFromStrength(settings.strength),
    bearingDeg: Number.isFinite(settings.bearingDeg) ? settings.bearingDeg : 0,
    gust: sidecar.wind.gust,
    season,
  };
}

/** No gusts: a steady mean speed. For tests that want the turbulence alone. */
export const NO_GUSTS: GustSettings = {
  strength: 0,
  variance: 0,
  frequencyPerMin: 1,
  durationS: 1,
};

// ---------------------------------------------------------------------------------------------
// Gusts
// ---------------------------------------------------------------------------------------------

const GUST_SEED = 0x6057;

/**
 * The gust envelope at time `t`: the fractional increase of the mean speed, `≥ 0`.
 *
 * SpeedTree-style: gusts arrive `frequencyPerMin` times a minute, last about `durationS`, and
 * peak at about `strength`, each varying by `±variance`. Time is cut into slots one gust-period
 * long; each slot holds exactly one gust whose start, length and peak come from a hash of the
 * slot number, and which ends inside its slot, so the value needs only the current slot. A
 * `sin²` envelope keeps the speed smooth.
 */
export function gustEnvelope(gust: GustSettings, t: number, seed: number): number {
  if (!(gust.strength > 0) || !(gust.frequencyPerMin > 0) || !Number.isFinite(t)) return 0;
  const period = 60 / gust.frequencyPerMin;
  const slot = Math.floor(t / period);
  const variance = Math.min(1, Math.max(0, gust.variance));
  const key = hash32(slot, seed ^ GUST_SEED);
  const u1 = unitUniform(key, 1);
  const u2 = unitUniform(key, 2);
  const u3 = unitUniform(key, 3);
  const duration = Math.min(period, Math.max(0.2, gust.durationS * (1 + variance * (2 * u1 - 1))));
  const start = slot * period + u2 * (period - duration);
  const local = t - start;
  if (local <= 0 || local >= duration) return 0;
  const peak = gust.strength * (1 + variance * (2 * u3 - 1));
  const s = Math.sin((Math.PI * local) / duration);
  return peak * s * s;
}

/** The largest value {@link gustEnvelope} can return. */
export function maxGust(gust: GustSettings): number {
  if (!(gust.strength > 0)) return 0;
  return gust.strength * (1 + Math.min(1, Math.max(0, gust.variance)));
}

// ---------------------------------------------------------------------------------------------
// The runtime model
// ---------------------------------------------------------------------------------------------

/** One oscillator's trajectory through its texture. */
interface Trajectory {
  readonly x0: number;
  readonly y0: number;
  /** Texels per second. */
  readonly vx: number;
  readonly vy: number;
}

interface BranchRuntime {
  readonly base: number;
  readonly frequencyHz: number;
  readonly damping: number;
  readonly texture: MotionTexture;
  readonly along: Trajectory;
  readonly across: Trajectory;
}

interface SeasonRuntime {
  readonly branches: ReadonlyMap<number, BranchRuntime>;
  readonly flutterLookups: Float64Array;
}

/** A rig, its sidecar, and everything derived from them once. Immutable after construction. */
export interface LivingMotion {
  readonly rig: MotionRig;
  readonly sidecar: MotionSidecar;
  /** Per node: unit direction of the limb it lies on — the limb's chord, attachment to far end. */
  readonly limbDirection: readonly Vec3[];
  /** @internal memoised per season. */
  readonly seasons: Map<Season, SeasonRuntime>;
}

/**
 * Binds a sidecar to its rig. Throws when they do not belong together — a sidecar for another
 * capture would move the wrong joints with confidence.
 */
export function createLivingMotion(rig: MotionRig, sidecar: MotionSidecar): LivingMotion {
  const issues = validateMotionSidecar(sidecar, rig);
  if (issues.length > 0)
    throw new Error(`motion sidecar does not fit this rig:\n  ${issues.join("\n  ")}`);
  const structure = branchStructure(rig);
  // Topological order: a limb's far end is its highest-indexed joint.
  const far = new Map<number, number>();
  rig.nodes.forEach((_, i) => {
    if (i > 0) far.set(structure.limb[i] ?? i, i);
  });
  const limbDirection = rig.nodes.map((node, i): Vec3 => {
    if (i === 0) return [0, 0, 1];
    const base = structure.limb[i] ?? i;
    const from = rig.nodes[rig.nodes[base]?.parent ?? 0]?.position;
    const to = rig.nodes[far.get(base) ?? i]?.position ?? node.position;
    if (from === undefined) return [0, 0, 1];
    const d = subtract(to, from);
    const length = Math.hypot(d[0], d[1], d[2]);
    return length > 0 ? [d[0] / length, d[1] / length, d[2] / length] : [0, 0, 1];
  });
  return { rig, sidecar, limbDirection, seasons: new Map() };
}

/** Parses a sidecar and binds it to its rig in one step. */
export function loadLivingMotion(rig: MotionRig, sidecarText: string): LivingMotion {
  return createLivingMotion(rig, parseMotionSidecar(sidecarText, rig));
}

/** Seconds of motion the aperiodicity budget is spent on: an hour, as the tests check. */
const APERIODIC_HORIZON_S = 3600;
/** Clearance wanted between a trajectory and its own start over that horizon, wavelengths. */
const WANTED_CLEARANCE_WAVELENGTHS = 1.5;

function trajectory(key: number, seed: number, frequencyHz: number): Trajectory {
  const speed = frequencyHz * BRANCH_TEXELS_PER_CYCLE;
  const [dx, dy] = trajectoryDirection(
    key,
    seed,
    BRANCH_TEXTURE_SIZE,
    speed * APERIODIC_HORIZON_S,
    WANTED_CLEARANCE_WAVELENGTHS * BRANCH_TEXELS_PER_CYCLE,
  );
  return {
    x0: unitUniform(key * 2, seed ^ 0x51a7) * BRANCH_TEXTURE_SIZE,
    y0: unitUniform(key * 2 + 1, seed ^ 0x51a7) * BRANCH_TEXTURE_SIZE,
    vx: dx * speed,
    vy: dy * speed,
  };
}

function seasonRuntime(motion: LivingMotion, season: Season): SeasonRuntime {
  const cached = motion.seasons.get(season);
  if (cached !== undefined) return cached;
  const { sidecar } = motion;
  const winter = sidecar.seasons.winter;
  const branches = new Map<number, BranchRuntime>();
  const seed = sidecar.seed;
  sidecar.nodes.branch.forEach((base, i) => {
    if (i === 0 || branches.has(base)) return;
    const isTree = sidecar.nodes.mode[i] === 0;
    const summerHz = sidecar.nodes.frequencyHz[i] ?? 1;
    const summerZeta = sidecar.nodes.damping[i] ?? 0.1;
    const frequencyHz =
      season === "winter" && !isTree ? summerHz * winter.branchFrequencyScale : summerHz;
    // Damping is quantised by the generator, and rounded here, so branches share textures.
    const damping =
      Math.round((season === "winter" ? summerZeta * winter.dampingScale : summerZeta) * 1e4) / 1e4;
    branches.set(base, {
      base,
      frequencyHz,
      damping,
      texture: branchMotionTexture(damping, seed),
      along: trajectory(base * 2, seed, frequencyHz),
      across: trajectory(base * 2 + 1, seed, frequencyHz),
    });
  });
  const reference = sidecar.referenceSpeedMps * sidecar.wind.canopyAdvection;
  const runtime: SeasonRuntime = {
    branches,
    flutterLookups: flutterLookups(
      seed,
      FLUTTER_TEXTURE_SIZE,
      (reference * APERIODIC_HORIZON_S) / sidecar.leafSizeM,
      0.5 * (FLUTTER_MIN_WAVELENGTH_LEAVES + FLUTTER_MAX_WAVELENGTH_LEAVES),
    ),
  };
  motion.seasons.set(season, runtime);
  return runtime;
}

/** Builds every texture a season needs now, rather than on the first frame. */
export function prepareLivingMotion(motion: LivingMotion, season: Season = "summer"): void {
  seasonRuntime(motion, season);
  flutterTexture(motion.sidecar.seed);
}

/** The value a trajectory reads at time `t`. Unit RMS. */
function sampleAt(texture: MotionTexture, path: Trajectory, t: number): number {
  return sampleTexture(texture, path.x0 + path.vx * t, path.y0 + path.vy * t);
}

/** Square root of the Simiu–Scanlan spectrum's shape at `f`: `(1 + f/U)^(−5/6)`. */
export function spectralShape(frequencyHz: number, speedMps: number): number {
  if (!(speedMps > 0)) return 0;
  return Math.pow(1 + frequencyHz / speedMps, -5 / 6);
}

function downwindOf(bearingDeg: number): Vec3 {
  const bearing = (Number.isFinite(bearingDeg) ? bearingDeg : 0) * DEG_TO_RAD;
  return [Math.sin(bearing), Math.cos(bearing), 0];
}

/** Two unit axes across a limb: `e1` along `d × ŵ` (the lean axis), `e2 = d × e1`. */
function crossAxes(d: Vec3, w: Vec3): [Vec3, Vec3] {
  let e1: Vec3 = [d[1] * w[2] - d[2] * w[1], d[2] * w[0] - d[0] * w[2], d[0] * w[1] - d[1] * w[0]];
  let length = Math.hypot(e1[0], e1[1], e1[2]);
  if (!(length > 1e-9)) {
    // The limb points along the wind: any axis across it will do.
    const helper: Vec3 = Math.abs(d[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    e1 = [
      d[1] * helper[2] - d[2] * helper[1],
      d[2] * helper[0] - d[0] * helper[2],
      d[0] * helper[1] - d[1] * helper[0],
    ];
    length = Math.hypot(e1[0], e1[1], e1[2]);
  }
  e1 = [e1[0] / length, e1[1] / length, e1[2] / length];
  const e2: Vec3 = [
    d[1] * e1[2] - d[2] * e1[1],
    d[2] * e1[0] - d[0] * e1[2],
    d[0] * e1[1] - d[1] * e1[0],
  ];
  return [e1, e2];
}

/** A rotation by the vector `r` (axis × angle), with the angle smoothly capped at `limit`. */
function rotationFromVector(r: Vec3, limit: number): Quat {
  const angle = Math.hypot(r[0], r[1], r[2]);
  if (angle === 0) return IDENTITY_TRANSFORM.rotation;
  const capped = softLimit(angle, limit);
  const s = Math.sin(capped / 2) / angle;
  return [r[0] * s, r[1] * s, r[2] * s, Math.cos(capped / 2)];
}

/** Per-node load factor `q_i(t) = ((U/U_ref)(1 + gust(t − x_i/U)))²`, into `out`. */
function loadFactors(motion: LivingMotion, t: number, wind: LivingWind, out: Float64Array): void {
  const { rig, sidecar } = motion;
  const w = downwindOf(wind.bearingDeg);
  const ratio = wind.speedMps / sidecar.referenceSpeedMps;
  for (let i = 0; i < rig.nodes.length; i += 1) {
    const p = rig.nodes[i]?.position ?? [0, 0, 0];
    // The gust is convected at the mean speed: it reaches the downwind side of the crown later.
    const delay = (p[0] * w[0] + p[1] * w[1]) / wind.speedMps;
    const u = ratio * (1 + gustEnvelope(wind.gust, t - delay, sidecar.seed));
    out[i] = u * u;
  }
}

/**
 * Per-node rest-to-displaced transforms at time `t`, parent-composed, as `deform` returns them.
 * `wind.speedMps === 0` returns the identity for every node, by value.
 */
export function livingTransforms(
  motion: LivingMotion,
  t: number,
  wind: LivingWind,
): NodeTransform[] {
  const { rig, sidecar } = motion;
  const count = rig.nodes.length;
  const transforms: NodeTransform[] = [];
  if (!(wind.speedMps > 0) || !Number.isFinite(wind.speedMps)) {
    for (let i = 0; i < count; i += 1) transforms.push(IDENTITY_TRANSFORM);
    return transforms;
  }
  const time = Number.isFinite(t) ? t : 0;
  const runtime = seasonRuntime(motion, wind.season);
  const w = downwindOf(wind.bearingDeg);
  const q = new Float64Array(count);
  loadFactors(motion, time, wind, q);
  const alongRatio = sidecar.wind.turbulence.along;
  const acrossRatio = sidecar.wind.turbulence.across;
  // One read per branch per frame, shared by every joint of the branch.
  const sway = new Map<number, [number, number]>();
  for (const branch of runtime.branches.values()) {
    const shape = spectralShape(branch.frequencyHz, wind.speedMps);
    sway.set(branch.base, [
      alongRatio * shape * sampleAt(branch.texture, branch.along, time),
      acrossRatio * shape * sampleAt(branch.texture, branch.across, time),
    ]);
  }
  for (let i = 0; i < count; i += 1) {
    const node = rig.nodes[i];
    const gain = sidecar.nodes.gainRad[i] ?? 0;
    if (node === undefined || node.parent < 0 || gain === 0) {
      transforms.push(
        node === undefined || node.parent < 0
          ? IDENTITY_TRANSFORM
          : (transforms[node.parent] ?? IDENTITY_TRANSFORM),
      );
      continue;
    }
    const d = motion.limbDirection[i] ?? [0, 0, 1];
    const [e1, e2] = crossAxes(d, w);
    const [s1, s2] = sway.get(sidecar.nodes.branch[i] ?? i) ?? [0, 0];
    const k = gain * (q[i] ?? 0);
    // Mean lean: d × ŵ, whose length is the sine between limb and wind.
    const lean: Vec3 = [
      d[1] * w[2] - d[2] * w[1],
      d[2] * w[0] - d[0] * w[2],
      d[0] * w[1] - d[1] * w[0],
    ];
    const r: Vec3 = [
      k * (lean[0] + s1 * e1[0] + s2 * e2[0]),
      k * (lean[1] + s1 * e1[1] + s2 * e2[1]),
      k * (lean[2] + s1 * e1[2] + s2 * e2[2]),
    ];
    const local = rotationFromVector(r, nodeAngleLimit(node));
    const parent = transforms[node.parent] ?? IDENTITY_TRANSFORM;
    // A hinge at the joint this segment hangs from.
    const pivot = rig.nodes[node.parent]?.position ?? node.position;
    const pivotOffset = subtract(pivot, quatRotate(local, pivot));
    transforms.push({
      rotation: quatMultiply(parent.rotation, local),
      translation: add(quatRotate(parent.rotation, pivotOffset), parent.translation),
    });
  }
  return transforms;
}

/**
 * Saturation of leaf flutter, as a multiple of its reference amplitude. Leaves reconfigure and
 * streamline in strong wind (Vogel 1989), so their drag grows more slowly than `U²`; the shape of
 * this cap is not measured.
 */
export const FLUTTER_SATURATION = 2;

/** The advected leaf-flutter field for time `t`. Still (and exactly inert) at calm and in winter. */
export function livingFlutter(
  motion: LivingMotion,
  t: number,
  wind: LivingWind,
): AdvectedFlutterField {
  const { rig, sidecar } = motion;
  const count = rig.nodes.length;
  const texture = flutterTexture(sidecar.seed);
  const runtime = seasonRuntime(motion, wind.season);
  const w = downwindOf(wind.bearingDeg);
  const amplitudeM = new Float64Array(count);
  const seasonScale = wind.season === "winter" ? sidecar.seasons.winter.flutterScale : 1;
  const base = {
    kind: "advected" as const,
    texture,
    downwind: [w[0], w[1]] as const,
    texelsPerMeter: 1 / sidecar.leafSizeM,
    lookups: runtime.flutterLookups,
    phase: FLUTTER_STILL.phase,
  };
  if (!(wind.speedMps > 0) || !Number.isFinite(wind.speedMps) || seasonScale === 0) {
    return { ...base, amplitudeM, advectionTexels: 0, still: true };
  }
  const time = Number.isFinite(t) ? t : 0;
  const q = new Float64Array(count);
  loadFactors(motion, time, wind, q);
  let still = true;
  for (let i = 0; i < count; i += 1) {
    const reference = (sidecar.nodes.flutterM[i] ?? 0) * seasonScale;
    if (reference === 0) continue;
    const amplitude = softLimit(reference * (q[i] ?? 0), FLUTTER_SATURATION * reference);
    if (amplitude === 0) continue;
    amplitudeM[i] = amplitude;
    still = false;
  }
  const advection = wind.speedMps * sidecar.wind.canopyAdvection * time;
  return { ...base, amplitudeM, advectionTexels: advection / sidecar.leafSizeM, still };
}

/** Both halves of a frame. */
export function livingFrame(
  motion: LivingMotion,
  t: number,
  wind: LivingWind,
): { transforms: NodeTransform[]; flutter: AdvectedFlutterField } {
  return { transforms: livingTransforms(motion, t, wind), flutter: livingFlutter(motion, t, wind) };
}

// ---------------------------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------------------------

/**
 * The largest local rotation each joint can reach under this wind, radians. Exact bound: the
 * lean is at most 1, each texture read at most its `maxAbs`, the gust at most `maxGust`.
 */
export function livingMaxNodeAngles(motion: LivingMotion, wind: LivingWind): number[] {
  const { rig, sidecar } = motion;
  if (!(wind.speedMps > 0)) return rig.nodes.map(() => 0);
  const runtime = seasonRuntime(motion, wind.season);
  const u = (wind.speedMps / sidecar.referenceSpeedMps) * (1 + maxGust(wind.gust));
  const qMax = u * u;
  return rig.nodes.map((node, i) => {
    const gain = sidecar.nodes.gainRad[i] ?? 0;
    if (node.parent < 0 || gain === 0) return 0;
    const branch = runtime.branches.get(sidecar.nodes.branch[i] ?? i);
    const shape = branch === undefined ? 0 : spectralShape(branch.frequencyHz, wind.speedMps);
    const peak = branch?.texture.maxAbs ?? 0;
    const turbulence = sidecar.wind.turbulence;
    const raw = gain * qMax * (1 + shape * peak * (turbulence.along + turbulence.across));
    return softLimit(raw, nodeAngleLimit(node));
  });
}

/**
 * Worst-case displacement of any splat, metres: the chain sum of joint rotations times lever
 * arms (as `maxNodeDisplacements`), plus the flutter bound. A proven bound, not a sample.
 */
export function livingMaxDisplacement(motion: LivingMotion, wind: LivingWind): number {
  const { rig, sidecar } = motion;
  if (!(wind.speedMps > 0)) return 0;
  const angles = livingMaxNodeAngles(motion, wind);
  let worst = 0;
  rig.nodes.forEach((node, index) => {
    let bound = 0;
    let cursor = index;
    while (cursor > 0) {
      const joint = rig.nodes[cursor];
      if (joint === undefined) break;
      // Each joint's rotation pivots at its parent's rest position.
      const pivot = rig.nodes[joint.parent]?.position ?? joint.position;
      const dx = node.position[0] - pivot[0];
      const dy = node.position[1] - pivot[1];
      const dz = node.position[2] - pivot[2];
      bound += (angles[cursor] ?? 0) * Math.hypot(dx, dy, dz);
      cursor = joint.parent;
    }
    if (bound > worst) worst = bound;
  });
  const seasonScale = wind.season === "winter" ? sidecar.seasons.winter.flutterScale : 1;
  let flutter = 0;
  for (const value of sidecar.nodes.flutterM) flutter = Math.max(flutter, value);
  flutter *=
    FLUTTER_SATURATION * seasonScale * advectedFlutterUnitBound(flutterTexture(sidecar.seed));
  return worst + flutter;
}
